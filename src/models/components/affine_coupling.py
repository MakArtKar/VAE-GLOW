from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.act_norm import ActNorm
from src.models.components.base_flow_model import BaseFlowModel


class AffineCoupling(BaseFlowModel):
    def __init__(self, in_channels: int, hid_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels // 2, hid_channels, 3, padding='same'),
            ActNorm(hid_channels, return_log_det=False),
            nn.ReLU(),
            nn.Conv2d(hid_channels, hid_channels, 1),
            ActNorm(hid_channels, return_log_det=False),
            nn.ReLU(),
            nn.Conv2d(hid_channels, in_channels, kernel_size=3, padding='same')
        )
        self.model[-1].weight.data.zero_()
        self.model[-1].bias.data.zero_()

    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        xa, xb = torch.chunk(x, 2, 1)
        log_sigma, mu = torch.chunk(self.model(xb), 2, 1)
        sigma = torch.sigmoid(log_sigma + 2) # for preventing gradient explode; sigmoid(2) ~ identity

        log_det = sigma.log().view(x.size(0), -1).sum(1)

        if not reverse:
            ya = sigma * xa + mu
        else:
            ya = (xa - mu) / sigma
            log_det = -log_det

        y = torch.cat([ya, xb], dim=1)
        return y, log_det
