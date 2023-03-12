from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid

from src.evaluator.evaluator import Evaluator
from src.models.components.vae import VAE


# Seminar 6
class VAELitModule(pl.LightningModule):
    def __init__(
            self,
            net: VAE,
            evaluator: Evaluator,
            optimizer: torch.optim.optimizer,
            scheduler: Optional[torch.optim.lr_scheduler] = None,
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
        self.log('train/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/kld_loss', kld_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x = batch['image']
        recon_x, _, _ = self(x)
        self.evaluator.add_images(real_images=x, fake_images=recon_x)

    def validation_epoch_end(self, outputs):
        fid = self.evaluator.calculate()
        self.log('val/fid', fid, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/fid',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
