import os

import numpy as np
from PIL import Image

from src.data.components.celeba import CelebADataset


class WrappedCelebADataset(CelebADataset):
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        # Apply transformations to the image
        if self.transform:
            img = self.transform(image=img)['image']
        img = img / 255
        return {
            'image': img,
        }
