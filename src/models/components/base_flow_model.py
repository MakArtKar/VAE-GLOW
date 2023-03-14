from abc import abstractmethod
from typing import Union, Tuple

import torch.nn as nn
from torch import Tensor


class BaseFlowModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, reverse=False, return_log_det=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError()
