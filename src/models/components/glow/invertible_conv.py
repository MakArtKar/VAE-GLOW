from typing import Tuple

import torch
from torch import Tensor, nn

from src.models.components.glow.base_flow_model import BaseFlowModel


class InvertibleConv(BaseFlowModel):
    def __init__(self, in_channels: int):
        super().__init__()
        # w = torch.empty(in_channels, in_channels)
        # nn.init.orthogonal_(w)
        # if torch.linalg.det(w) < 0:
        #     w = -w

        # w = torch.linalg.qr(torch.randn(in_channels, in_channels))[0]
        # self.w = nn.Parameter(w)

        weight = torch.linalg.qr(torch.randn(in_channels, in_channels))[0]
        permutation, lower, upper = torch.lu_unpack(*torch.linalg.lu_factor(weight))
        self.permutation, self.lower, self.upper = nn.Parameter(permutation), nn.Parameter(lower), nn.Parameter(upper)
        s = self.upper.diag()
        self.register_buffer('s_sign', s.sign().detach())
        self.register_buffer('lower_mask', torch.tril(torch.ones_like(self.lower), -1))
        self.log_s = nn.Parameter(s.abs().log())

    # def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
    #     w = self.w if not reverse else self.w.inverse()
    #     out = nn.functional.conv2d(x, w.unsqueeze(2).unsqueeze(2))
    #     log_det = torch.slogdet(self.w)[-1] * x.size(2) * x.size(3)
    #     if reverse:
    #         log_det = -log_det
    #     return out, log_det

    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        log_det = self.log_s.sum() * x.size(2) * x.size(3)
        lower = self.lower * self.lower_mask + torch.eye(x.size(1), device=x.device)
        upper = self.upper * self.lower_mask.T + (self.s_sign * self.log_s.exp()).diag()
        if reverse:
            log_det = -log_det
            lower = lower.inverse()
            upper = upper.inverse()
            weight = upper @ lower @ self.permutation.inverse()
        else:
            weight = self.permutation @ lower @ upper
        return nn.functional.conv2d(x, weight.unsqueeze(2).unsqueeze(2)), log_det
