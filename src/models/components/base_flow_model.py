from abc import abstractmethod
from typing import Tuple, Sequence

import torch.nn as nn
from torch import Tensor


class BaseFlowModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, reverse=False) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class FlowSequential(BaseFlowModel):
    def __init__(self, *flow_models):
        super().__init__()
        self.flow_models = flow_models

    def forward(self, x, reverse=False) -> Tuple[Tensor, Tensor]:
        log_det: Tensor = 0
        blocks = self.flow_models if not reverse else self.flow_models[::-1]
        for model in blocks:
            x, model_log_det = model(x, log_det, reverse=reverse)
            log_det += model_log_det
        return x, log_det
