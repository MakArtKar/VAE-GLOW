from abc import abstractmethod
from typing import Union, Dict

from torch import Tensor


class BaseEvaluator:
    def reset(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def add_images(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError()
