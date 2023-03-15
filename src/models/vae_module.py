from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.evaluator.evaluator import Evaluator
from src.models.components.vae import VAE


# Seminar 6
class VAELitModule(pl.LightningModule):
    def __init__(
            self,
            net: VAE,
            evaluator: Evaluator,
            optimizer: torch.optim.Optimizer,
            scheduler=None,
            fid_frequency: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.evaluator = evaluator

    @staticmethod
    def mse_loss(x, recon_x):
        batch_size = x.size(0)
        return nn.functional.mse_loss(x.view(batch_size, -1), recon_x.view(batch_size, -1), reduction='sum')

    @staticmethod
    def kld_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx: int):
        x = batch['image']
        recon_x, mu, logvar = self(x)
        mse_loss = self.mse_loss(x, recon_x)
        kld_loss = self.kld_loss(mu, logvar)
        loss = mse_loss + kld_loss
        self.log('train/mse_loss', mse_loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log('train/kld_loss', kld_loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log('train/loss', loss, on_step=True, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.evaluator.reset()

    def validation_step(self, batch, batch_idx: int, mode='val'):
        x = batch['image']
        recon_x, mu, logvar = self(x)
        mse_loss = self.mse_loss(x, recon_x)
        kld_loss = self.kld_loss(mu, logvar)
        loss = mse_loss + kld_loss
        if batch_idx % self.hparams.fid_frequency == 0:
            fake_images = self.net.sample(x.size(0), self.device)
            self.evaluator.add_images(real_images=x, fake_images=fake_images)

        self.log(f'{mode}/mse_loss', mse_loss, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log(f'{mode}/kld_loss', kld_loss, prog_bar=True, sync_dist=True, batch_size=x.size(0))
        self.log(f'{mode}/loss', loss, prog_bar=True, sync_dist=True, batch_size=x.size(0))

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
