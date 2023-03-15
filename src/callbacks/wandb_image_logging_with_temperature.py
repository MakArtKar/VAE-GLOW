from typing import List

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from src.callbacks.wandb_image_logging import WandbImageLogging


class WandbImageLoggingWithTemperature(WandbImageLogging):
    def __init__(self, temperatures: List[float]):
        super().__init__()
        self.temperatures = temperatures

    def sample_images(self, trainer: pl.Trainer, pl_module) -> None:
        for temperature in self.temperatures:
            samples = pl_module.net.sample(16, pl_module.device, temperature=temperature)
            images = make_grid(samples, nrow=4)
            images = torch.clip((images + 1) / 2, 0, 1)
            trainer.logger.log_image(key=f'images/generated_images_temp={temperature:0.2}', images=[images])

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        self.sample_images(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.sample_images(trainer, pl_module)
