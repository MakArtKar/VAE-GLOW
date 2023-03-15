import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid


def sample_images(trainer: pl.Trainer, pl_module) -> None:
    samples = pl_module.net.sample(16, pl_module.device)
    images = make_grid(samples, nrow=4)
    images = torch.clip((images + 1) / 2, 0, 1)
    trainer.logger.log_image(key='images/generated_images', images=[images])


def reconstruction(
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
) -> None:
    if batch_idx == 0:
        x, recon_x = outputs['x'][:8], outputs['recon_x'][:8]
        samples = torch.cat([x, recon_x], dim=0)
        images = make_grid(samples, nrow=x.size(0))
        images = torch.clip((images + 1) / 2, 0, 1)
        trainer.logger.log_image(key='images/reconstruction', images=[images])


class WandbImageLogging(pl.Callback):
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        sample_images(trainer, pl_module)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        reconstruction(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sample_images(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        reconstruction(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
