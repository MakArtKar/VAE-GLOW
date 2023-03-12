# Seminar 6
import os
from typing import Optional

import albumentations
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, root_path: str, transform: Optional[albumentations.ImageOnlyTransform] = None):
        super().__init__()
        self.transform = transform
        self.root_path = root_path
        self.img_paths = os.listdir(root_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(self.root_path, self.img_paths[idx]))
        if self.transform:
            img = self.transform(image=img)['image']
        return img
