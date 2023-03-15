from typing import Tuple, Union

import torch
from torch import Tensor

from src.models.components.glow.base_flow_model import BaseFlowModel


class Split(BaseFlowModel):
    def forward(self, x, z=None, reverse=False, **kwargs) -> \
            Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        log_det: Tensor = torch.Tensor([0]).to(x.device)
        if not reverse:
            return *torch.chunk(x, 2, 1), log_det
        else:
            return torch.cat([x, z], 1), log_det
