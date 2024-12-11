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


class WSADE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 scale_factor: list = [0.20, 0.58, 0.22]) -> None:
        super(WSADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
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
            de_per_path = torch.mul(torch.norm(pred_ - target_, p=2, dim=-1), ~padding_mask_).sum(dim=-1)
            ade_per_path = torch.where((~padding_mask_).sum(dim=-1) > 0, de_per_path / (~padding_mask_).sum(dim=-1), torch.zeros(pred_.size(0)).to(pred_.device))
            self.sum += ade_per_path.sum() * self.scale_factor[i]
            self.count += pred_.size(0)

    def compute(self) -> torch.Tensor:
        breakpoint()
        return self.sum / self.count
