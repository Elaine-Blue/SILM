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
from typing import Any, Callable, Optional, List

import torch
from torchmetrics import Metric

def get_last_valid_index(masks):
    end_index = torch.zeros(masks.size(0))
    for i in range(masks.size(0)):
        mask = masks[i, :]
        for j in list(range(mask.size(0)-1, -1, -1)):
            if mask[j] == False:
                end_index[i] = j
                break
    return end_index.to(torch.long).to(mask.device)
 
class WSFDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 scale_factor: list = [0.20, 0.58, 0.22]) -> None:
        super(WSFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.scale_factor = scale_factor
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: List[torch.Tensor],
               target: List[torch.Tensor],
               padding_mask: List[torch.Tensor]) -> None:
        for i, (pred_, target_, padding_mask_) in enumerate(zip(pred, target, padding_mask)):
            if pred_ == None:
                continue
            end_index = get_last_valid_index(padding_mask_) # 找到end point index
            fde_total = torch.norm(pred_[torch.arange(pred_.size(0)), end_index, :] - target_[torch.arange(pred_.size(0)), end_index, :], p=2, dim=-1).sum()
            self.sum += fde_total * self.scale_factor[i]
            self.count += pred_.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
