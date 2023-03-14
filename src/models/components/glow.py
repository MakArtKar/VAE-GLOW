from typing import Tuple, List

import torch
from torch import nn, Tensor

from src.models.components.base_flow_model import FlowSequential, BaseFlowModel
from src.models.components.flow_block import FlowBlock
from src.models.components.squeeze import Squeeze


class Glow(BaseFlowModel):
    def __init__(self, n_flows: int, depth: int, in_channels: int, hid_channels: int, image_size: int):
        super().__init__()
        self.glow_blocks = nn.ModuleList([
            FlowBlock(depth, in_channels * 2 ** k, hid_channels)
            for k in range(n_flows + 1)
        ])
        self.glow_blocks[-1].flow_models = self.glow_blocks[-1].flow_models[:-1]
        self.image_size = image_size
        self.out_shape = (in_channels * 2 ** (n_flows + 2), image_size // 2 ** n_flows, image_size // 2 ** n_flows)

    def forward(self, x, reverse=False, z_list=None, **kwargs) -> Tuple[Tensor, List[Tensor], Tensor]:
        if not reverse:
            z_list = []
        glow_blocks = self.glow_blocks if not reverse else self.glow_blocks[::-1]
        log_det = 0
        for i, m in enumerate(glow_blocks):
            if not reverse:
                if i + 1 != len(glow_blocks):
                    x, z, model_log_det = m(x, reverse=reverse)
                else:
                    x, model_log_det = m(x, reverse=reverse)
                    z = x
                z_list.append(z)
            else:
                if z_list is None:
                    z = self.sample_z(x.shape, x.device)
                else:
                    z = z_list[-i - 1]
                x, model_log_det = m(x, reverse=reverse, z=z)
            log_det = log_det + model_log_det
        return x, z_list, log_det

    @staticmethod
    def sample_z(shape, device):
        return torch.randn(shape).to(device)

    def sample(self, batch_size: int, device):
        z = self.sample_z((batch_size, *self.out_shape), device)
        return self.forward(z, reverse=True)[0]
