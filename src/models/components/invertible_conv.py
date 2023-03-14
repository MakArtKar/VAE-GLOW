from typing import Union, Tuple

import torch
from torch import Tensor, nn

from src.models.components.base_flow_model import BaseFlowModel


class InvertibleConv(BaseFlowModel):
    def __init__(self, in_channels: int):
        super().__init__()
        w = torch.qr(torch.randn(in_channels, in_channels))[0]
        self.w = nn.Parameter(w)

    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        w = self.w if not reverse else self.w.inverse()
        out = nn.functional.conv2d(x, w.unsqueeze(2).unsqueeze(2))
        log_det = torch.slogdet(self.w)[-1] * x.size(2) * x.size(3)
        if reverse:
            log_det = -log_det
        return out, log_det
