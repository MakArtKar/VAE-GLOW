from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.glow.base_flow_model import BaseFlowModel


class ActNorm(BaseFlowModel):
    EPSILON = 1e-6

    def __init__(self, channels: int, return_log_det: bool = True):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.is_initialized = False
        self.return_log_det = return_log_det

        self.channels = channels

    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        if not self.is_initialized and self.training:
            self.mu.data = x.transpose(0, 1).reshape(x.size(1), -1).mean(1).view_as(self.mu)
            self.sigma.data = (x.transpose(0, 1).reshape(x.size(1), -1).std(1) + self.EPSILON).view_as(self.sigma)
            self.is_initialized = True

        if not reverse:
            out = (x - self.mu) / self.sigma
            log_det = -self.sigma.abs().log().sum() * x.size(2) * x.size(3)
        else:
            out = x * self.sigma + self.mu
            log_det = self.sigma.abs().log().sum() * x.size(2) * x.size(3)

        if self.return_log_det:
            return out, log_det
        else:
            return out
