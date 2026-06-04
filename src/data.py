import torch
import lightning as L
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


RGB_TO_CLASS = {
    (255, 255, 255): 0,  # impervious surface
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # clutter
}


class PotsdamDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 indices: list,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        n = self.indices[idx]
        img = Image.open(f'{self.data_dir}/Images/Image_{n}.tif')
        mask = Image.open(f'{self.data_dir}/Labels/Label_{n}.tif')

        img = ToTensor()(img)
        
        mask_np = np.array(mask)
        result = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        for rgb, class_idx in RGB_TO_CLASS.items():
            matches = np.all(mask_np == rgb, axis=-1)
            result[matches] = class_idx

        return img, torch.from_numpy(result)

class AerialSegmentationDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            val_split: float = 0.2,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: str):
        all_indices = list(range(2400))
        n_val = int(len(all_indices) * self.val_split)
        self.train_dataset = PotsdamDataset(self.data_dir, indices=all_indices[n_val:])
        self.val_dataset = PotsdamDataset(self.data_dir, indices=all_indices[:n_val])
 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
