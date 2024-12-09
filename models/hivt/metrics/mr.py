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
from typing import Any, Callable, Optional

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

class MR(Metric):

    def __init__(self,
                 miss_threshold: float = 2.0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.miss_threshold = miss_threshold

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               padding_mask: torch.Tensor) -> None:
        end_index = get_last_valid_index(padding_mask)
        self.sum += torch.mul((torch.norm(pred[torch.arange(pred.size(0)), end_index, :] - target[torch.arange(pred.size(0)), end_index, :], p=2, dim=-1) > self.miss_threshold), end_index != 0).sum()
        self.count += (end_index != 0).size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
