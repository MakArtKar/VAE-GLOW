from typing import Tuple

import torch
from torch import Tensor

from src.models.components.base_flow_model import BaseFlowModel


class SqueezeBlock(BaseFlowModel):
    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        b, c, h, w = x.size()
        if not reverse:
            x = x.view(b, c, h // 2, 2, w // 2, 2)
            x = x.view(0, 1, 3, 5, 2, 4)
            x = x.view(b, c * 4, h // 2, w // 2)
        else:
            x = x.view(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.view(b, c // 4, h * 2, w * 2)
        return x, 0
