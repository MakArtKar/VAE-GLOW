import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from src.models.vae_module import VAELitModule


class WandbVAEImageLogging(pl.Callback):
    def __init__(self, epoch_frequency: int = 1):
        super().__init__()
        self.epoch_frequency = epoch_frequency

    def on_validation_epoch_start(self, trainer: VAELitModule, pl_module: pl.LightningModule) -> None:
        samples = trainer.net.sample(16)
        images = make_grid(samples)
        trainer.logger.log_image(key='images/generated_images', images=images)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 0:
            x, recon_x = outputs['x'], outputs['recon_x']
            samples = torch.cat([x, recon_x], dim=0)
            images = make_grid(samples, nrow=2)
            trainer.logger.log_image(key='images/reconstruction', images=images)
