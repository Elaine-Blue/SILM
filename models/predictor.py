import sys
sys.path.append("/root/Demo")

# from dataset import ArgoverseV2Dataset
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from .hivt import LaplaceNLLLoss
from .hivt import SoftTargetCrossEntropyLoss
from .hivt import ADE
from .hivt import FDE
from .hivt import MR
from .hivt import WSADE
from .hivt import WSFDE
from .hivt import GlobalInteractor
from .hivt import LocalEncoder
from .hivt import MLPDecoder
from .hivt import PoseEncoder
import copy
from .lt3d import LSTMModel

EXTEND_SIZE = 5

class Predictor(pl.LightningModule):
    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 model_type: str = 'CYCLIST',
                 keypoints_threshold: float = 0.7,
                 scale_factor: list = [1.0, 1.0, 1.0],
                 **kwargs) -> None:
        super(Predictor, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.model_type = model_type
        self.keypoints_threshold = keypoints_threshold
        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        if model_type == 'PEDESTRIAN':
            self.pose_encoder = PoseEncoder()
        elif model_type == 'CAR':
            pass
        elif model_type == 'CYCLIST':
            pass
        
        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
        self.WSADE = WSADE(scale_factor=scale_factor)
        self.WSFDE = WSFDE(scale_factor=scale_factor)
        self.use_new_metrics = True

    def forward(self, data):
        if self.keypoints_threshold > 0:
            _, h, _ = data['keypoints'].shape
            keypoints = data['keypoints'].reshape(-1, 3)
            keypoints = torch.where(keypoints[:, [2]] >= self.keypoints_threshold, 
                                    keypoints[:, :2],
                                    torch.zeros_like(keypoints[:, :2]))
            keypoints = keypoints.reshape(data.num_nodes, h, -1)

        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None
        
        if self.model_type == 'PEDESTRIAN':
            pose_embed = self.pose_encoder(keypoints, data.valid_flags)
        elif self.model_type == 'CAR':
            pose_embed = None

        local_embed = self.local_encoder(data=data, pose_embed=pose_embed)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        fde_agents = torch.norm(y_hat[:, :, -1, : 2] - data.y[:, -1, :].repeat((6, 1, 1)), p=2, dim=-1)
        best_mode_agents = fde_agents.argmin(dim=0)
        y_hat_best_agents = y_hat[best_mode_agents, range(data.num_nodes), :, :2].squeeze(0)
        self.minADE.update(y_hat_best_agents, data.y, data.padding_mask[:, 20:])
        self.minFDE.update(y_hat_best_agents, data.y, data.padding_mask[:, 20:])
        self.minMR.update(y_hat_best_agents, data.y, data.padding_mask[:, 20:])

        if self.use_new_metrics:
            y_hat_best_agents_car, data_y_car, padding_mask_car = self.get_part_pred_results(y_hat, data, cls_id=0)
            y_hat_best_agents_ped, data_y_ped, padding_mask_ped = self.get_part_pred_results(y_hat, data, cls_id=1)
            y_hat_best_agents_cyc, data_y_cyc, padding_mask_cyc = self.get_part_pred_results(y_hat, data, cls_id=2)
            y_hat_best_agents_all = [y_hat_best_agents_car, y_hat_best_agents_ped, y_hat_best_agents_cyc]
            data_y_all = [data_y_car, data_y_ped, data_y_cyc]
            data_padding_mask_all = [padding_mask_car, padding_mask_ped, padding_mask_cyc]

            self.WSADE.update(y_hat_best_agents_all, data_y_all, data_padding_mask_all)
            self.WSFDE.update(y_hat_best_agents_all, data_y_all, data_padding_mask_all)

        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_nodes)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_nodes)
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_nodes)
        self.log('val_wsADE', self.WSADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_nodes)
        self.log('val_wsFDE', self.WSFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_nodes)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--model-type', type=str, default='PEDESTRIAN')
        parser.add_argument('--keypoints_threshold', type=float, default=0.7)
        parser.add_argument('--scale_factor', type=list, default=[0.20, 0.58, 0.22])
        return parent_parser

    def get_part_pred_results(self, y_hat, data, cls_id=0):
        if (data['gt_labels']==cls_id).any() == False:
            return [None, None, None]
        padding_mask_part = data.padding_mask[data['gt_labels']==cls_id]
        y_hat_part = y_hat[:, data['gt_labels']==cls_id, :, :]
        data_y_part = data.y[data['gt_labels']==cls_id, :, :]
        fde_agents_part = torch.norm(y_hat_part[:, :, -1, : 2] - data.y[data['gt_labels']==cls_id, -1, :].repeat((6, 1, 1)), p=2, dim=-1)
        best_mode_agents_part = fde_agents_part.argmin(dim=0)
        y_hat_best_agents_part = y_hat[best_mode_agents_part, range(padding_mask_part.size(0)), :, :2].squeeze(0)
        return y_hat_best_agents_part, data_y_part, padding_mask_part[:, 20:]