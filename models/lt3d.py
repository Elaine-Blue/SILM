import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

FEATURE_SCALING = np.array([20, 20, 5, 5, 1])
AV2_CLASS_NAMES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')



class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        prediction_len,
        k,
        num_layers=3,
        embedding_dim=32,
        dropout_p=0.1,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        self.k = k
        self.class_embeddings = nn.Embedding(
            len(AV2_CLASS_NAMES), embedding_dim=embedding_dim
        )
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        nn.init.kaiming_uniform_(self.input_proj.weight)
        self.regression_proj = nn.Linear(embedding_dim, 2 * prediction_len * k)
        self.classification_proj = nn.Linear(embedding_dim, k)
        self.lstm_layers = nn.ModuleList(
            [nn.LSTM(embedding_dim, embedding_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, class_cond=None):
        # breakpoint()
        B, L, D = input.shape
        embedding = self.input_proj(input.reshape(B * L, -1)).reshape(B, L, -1)
        # add class embedding
        # embedding += self.class_embeddings(class_cond.reshape(-1, 1))
        x = F.relu(embedding)
        for lstm_layer in self.lstm_layers:
            x_out, state = lstm_layer(x)
            x = x + self.dropout(x_out)
        reg = self.regression_proj(x.reshape(B * L, -1)).reshape(
            B, L, self.k, self.prediction_len, -1
        )
        cls = self.classification_proj(x.reshape(B * L, -1)).reshape(
            B, L, self.k
        )
        return reg, cls

    def calculate_loss(self, reg_prediction, cls_prediction, target, mask):
        """K best loss"""
        B, L, K = cls_prediction.shape
        # mask loss for invalid timesteps
        reg_loss = nn.MSELoss(reduction="none")(
            reg_prediction, target.unsqueeze(2).repeat(1, 1, self.k, 1, 1)
        ).mean(axis=-1)
        reg_loss = reg_loss.masked_fill(~mask.unsqueeze(dim=2), 0).mean(axis=-1)
        cls_label = torch.argmin(reg_loss, dim=-1)
        cls_loss = nn.CrossEntropyLoss(reduction="none")(
            cls_prediction.reshape(B * L, K), cls_label.reshape(B * L)
        ).reshape(B, L)
        cls_loss = cls_loss.masked_fill(~mask.any(axis=-1), 0)
        reg_loss = reg_loss.min(axis=-1).values
        # mask cls loss
        loss = (reg_loss.sum() + cls_loss.sum()) / mask.any(axis=-1).sum()
        return loss
    


# if __name__ == "__main__":
#     argparser = ArgumentParser()
#     argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
#     argparser.add_argument("--learning_rate", default=1e-3, type=float)
#     argparser.add_argument("--device", default="cuda")
#     config = argparser.parse_args()
#     config.prediction_length = 30
#     config.K = 5
#     config.epochs = 20
#     config.num_layers = 4

    
#     dataloader = DataLoader(
#         train_dataset,
#         batch_size=64,
#         shuffle=True,
#     )
#     model = LSTMModel(
#         train_dataset.input_dim,
#         train_dataset.prediction_len,
#         k=config.K,
#         num_layers=config.num_layers,
#     )

#     # train
#     model = model.to(config.device).train()
#     optim = torch.optim.Adam(model.parameters(), config.learning_rate)

#     for epoch in range(config.epochs):
#         epoch_loss = []
#         for input, class_cond, target, mask in dataloader:
#             input, class_cond, target, mask = map(
#                 lambda x: x.to(config.device), (input, class_cond, target, mask)
#             )
#             reg_prediction, cls_prediction = model(input, class_cond)

#             optim.zero_grad()
#             loss = model.calculate_loss(reg_prediction, cls_prediction, target, mask)
#             loss.backward()
#             optim.step()

#             epoch_loss.append(loss.detach().cpu().item())

#         print(f"Epoch: {epoch}, Loss: {np.mean(epoch_loss)}")

#     # save model
#     model = model.cpu()
#     model_path = f"models/{config.dataset}/lstm.pt"
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     torch.save(model, model_path)
