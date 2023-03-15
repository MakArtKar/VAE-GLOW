from typing import Tuple

from torch import Tensor

from src.models.components.glow.base_flow_model import BaseFlowModel


class Squeeze(BaseFlowModel):
    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        b, c, h, w = x.size()
        if not reverse:
            x = x.reshape(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(b, c * 4, h // 2, w // 2)
        else:
            x = x.reshape(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape(b, c // 4, h * 2, w * 2)
        return x, 0
