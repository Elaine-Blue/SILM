# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
# from models.movenet import SinglePoseDetector, opts

EXTEND_SIZE = 5

class PoseEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 34,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 num_layer: int = 3,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(PoseEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        
        # opt = opts().init()
        # self.extractor = SinglePoseDetector(opt)
        # self.extractor.eval()
        
    def forward_old(self,
                    bboxes_2d_batch: list, # N * T * 4
                    image_names_batch: list,
                    valid_flags_batch: list,
                    use_pose_batch: list,
                    num_nodes: int) -> torch.Tensor:
        device = self.mlp[0].weight.device
        t = bboxes_2d_batch[0].shape[1]
        pose_feature_batch = torch.zeros((num_nodes, t, self.input_dim), device=device)
        keypoints_batch = torch.zeros((num_nodes, t, self.input_dim), device=device)
        agent_idx = 0

        for bboxes_2d, image_files, valid_flags, use_pose in zip(bboxes_2d_batch, image_names_batch, valid_flags_batch, use_pose_batch):
            assert bboxes_2d.shape[0] == len(image_files) == valid_flags.shape[0] == use_pose.shape[0]
            num_agents_single = bboxes_2d.shape[0]
            for i in range(num_agents_single):
                if use_pose[i]:    
                    bboxes_2d_i = bboxes_2d[i]
                    image_files_i = image_files[i]
                    valid_flags_i = valid_flags[i]
                    for j in range(t):  
                        if valid_flags_i[j]:
                            corner, image_path = bboxes_2d_i[j], image_files_i[j]
                            img = cv2.imread(image_path)
                            assert img is not None, print("Couldn't find",  image_path)
                            img_cropped = img[corner[1]-EXTEND_SIZE:corner[3]+EXTEND_SIZE, corner[0]-EXTEND_SIZE:corner[2]+EXTEND_SIZE, :]
                            if img_cropped.shape[0] > 50 and img_cropped.shape[1] > 50:
                                ret = self.extractor.run(img_cropped)
                                keypoints_batch[agent_idx + i, j, :] = torch.from_numpy(ret['results'][:, :2]).reshape(1, -1).to(device) # 17 * 2
            agent_idx += num_agents_single 
        return keypoints_batch

    def forward(self, keypoints: torch.Tensor, valid_flags: torch.Tensor):
        valid_flags = valid_flags.unsqueeze(dim=-1).repeat(1, 1, 64)
        pose_embed = self.mlp(keypoints) * valid_flags
        pose_embed = self.encoder(pose_embed) 
        return torch.sum(pose_embed, dim=1) # B * embed_dim