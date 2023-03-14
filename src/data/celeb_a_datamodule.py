import os
from typing import Optional

import gdown
import pytorch_lightning as pl
import requests
import zipfile
from albumentations import ImageOnlyTransform
from torch.utils.data import DataLoader, Subset

from src.data.components.celeb_a_dataset import WrappedCelebADataset


class CelebADataModule(pl.LightningDataModule):
    LIST_ATTR_URL = """https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/""" \
                   """spring-2022/data/celeba/list_attr_celeba.txt"""

    def __init__(
            self,
            data_dir: str,
            batch_size: int = 8,
            num_workers: int = 8,
            pin_memory: bool = True,
            train_transform: Optional[ImageOnlyTransform] = None,
            val_transform: Optional[ImageOnlyTransform] = None,
    ):
        super().__init__()

        self.data_train = self.data_val = self.data_test = None

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.hparams.data_dir, 'list_attr_celeba.txt')):
            open(os.path.join(self.hparams.data_dir, 'list_attr_celeba.txt'), 'wb')\
                .write(requests.get(self.LIST_ATTR_URL).content)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = WrappedCelebADataset(self.hparams.data_dir, transform=self.hparams.train_transform)
            self.data_val = self.data_test = WrappedCelebADataset(self.hparams.data_dir, transform=self.hparams.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )
