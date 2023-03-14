from abc import abstractmethod
from typing import Tuple, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class BaseFlowModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, ...]:
        raise NotImplementedError()


class FlowSequential(BaseFlowModel):
    def __init__(self, *flow_models):
        super().__init__()
        self.flow_models = flow_models

    def forward(self, x, reverse=False, **kwargs) -> Tuple[Tensor, Tensor]:
        log_det: Tensor = torch.tensor([0]).to(x.device)
        blocks = self.flow_models if not reverse else self.flow_models[::-1]
        for model in blocks:
            x, model_log_det = model(x, reverse=reverse, **kwargs)
            log_det += model_log_det
        return x, log_det
