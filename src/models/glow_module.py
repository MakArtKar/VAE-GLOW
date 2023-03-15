import math
from typing import List

import pytorch_lightning as pl
import torch
from torch import Tensor

from src.evaluator.evaluator import Evaluator
from src.models.components.glow.glow import Glow


# Seminar 6
class GlowLitModule(pl.LightningModule):
    def __init__(
            self,
            net: Glow,
            evaluator: Evaluator,
            optimizer: torch.optim.Optimizer,
            scheduler=None,
            fid_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'evaluator'])
        self.net = net
        self.evaluator = evaluator

        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, x: torch.Tensor):
        z, z_list, log_det = self.net(x)
        z = torch.nan_to_num(z)
        z_list = [torch.nan_to_num(_z) for _z in z_list]
        log_det = torch.nan_to_num(log_det)
        return z, z_list, log_det

    def get_log_likelyhood(self, z_list: List[Tensor], log_det: Tensor) -> Tensor:
        return (
            sum(self.normal.log_prob(z).mean((1, 2, 3)) / math.log(2) for z in z_list) + log_det
        ).mean()

    def training_step(self, batch, batch_idx: int):
        x = batch['image']
        _, z_list, log_det = self(x)
        loss = -self.get_log_likelyhood(z_list, log_det)
        self.log('train/loss', loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.evaluator.reset()

    def validation_step(self, batch, batch_idx: int, mode='val'):
        x = batch['image']
        z_out, z_list, log_det = self(x)
        loss = -self.get_log_likelyhood(z_list, log_det)
        recon_x, _, _ = self.net(z_out, reverse=True, z_list=z_list)
        if batch_idx % self.hparams.fid_frequency == 0:
            self.evaluator.add_images(real_images=x, fake_images=recon_x)
        self.log(f'{mode}/loss', loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        result = {'loss': loss}
        if batch_idx == 0:
            result.update({
                'x': x,
                'recon_x': recon_x,
            })
        return result

    def validation_epoch_end(self, outputs, mode='val'):
        result = self.evaluator.calculate()
        for name, metric in result.items():
            self.log(f'{mode}/{name}', metric, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx: int):
        self.validation_step(batch, batch_idx, mode='test')

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, mode='test')

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/fid',
                    'mode': 'min',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
