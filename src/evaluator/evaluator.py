from typing import List, Dict

from torch import Tensor

from src.evaluator.base_evaluator import BaseEvaluator


class Evaluator(BaseEvaluator):
    def __init__(self, evaluators: List[BaseEvaluator]):
        super().__init__()
        self.evaluators = evaluators

    def add_images(self, *args, **kwargs) -> None:
        for evaluator in self.evaluators:
            evaluator.add_images(*args, **kwargs)

    def calculate(self, *args, **kwargs) -> Dict[str, float]:
        result = {}
        for evaluator in self.evaluators:
            result.update(evaluator.calculate(*args, **kwargs))
        return result
