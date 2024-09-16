import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os

img_size = (224, 224)

class CustomDataset(Dataset):
    def __init__(self, data, img_transform, mask_transform):
        self.data = data
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['img_path']
        mask_path = self.data.iloc[idx]['mask_path']

        img = Image.open(img_path).convert('RGB')


        img = self.img_transform(img)
        
        mask = Image.open(mask_path).convert('L')
        
        mask = self.mask_transform(mask)

        mask = mask.squeeze(0)

        return img, mask


def load_data(img_path, mask_path):
    img_path = Path(img_path)
    img_path_list = list(img_path.glob('*.png'))
    img_path_list = sorted(img_path_list)

    mask_path = Path(mask_path)
    mask_path_list = list(mask_path.glob('*.png'))
    mask_path_list = sorted(mask_path_list)

    data = pd.DataFrame({'img_path': img_path_list, 'mask_path': mask_path_list})

    data_train, data_rest = train_test_split(data, test_size=0.3)
    data_val, data_test = train_test_split(data_rest, test_size=0.5)

    return data_train, data_val, data_test


def create_dataset(data):

    img_transform = v2.Compose([
        v2.Resize(img_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    mask_transform = v2.Compose([
        v2.Resize(img_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.long)
    ])

    dataset = CustomDataset(data, img_transform, mask_transform)

    return dataset

def create_dataloader(batch_size):

    num_workers = os.cpu_count()
    train_data, val_data, test_data = load_data('data/aug_images', 'data/aug_masks')
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    test_dataset = create_dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
