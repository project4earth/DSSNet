import os
import torch
import random
import rasterio
import numpy as np
from utils import config
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

def minmax_norm(image, minmax):
    num_channels = image.shape[0]

    if minmax == 'sentinel1':
        lower_val = [
            -55.5199,
            -59.9488,
        ]
        upper_val = [
            26.5266,
            11.7958,
        ]
    elif minmax == 'sentinel2':
        lower_val = [
            0.0,
            0.0,
            0.0,
            0.0,
        ] 
        upper_val = [
            1.3224,
            1.4688,
            1.6036,
            1.6421,
        ]
    else:
        print("Invalid mode!")

    normalized_image = np.zeros_like(image, dtype=np.float32)

    for c in range(num_channels):
        channel_data = image[c]
        normalized_channel = (channel_data - lower_val[c]) / (upper_val[c] - lower_val[c])
        normalized_channel = np.clip(normalized_channel, 0, 1)
        normalized_image[c] = normalized_channel

    return normalized_image

def add_speckle_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + image * noise
    return noisy_image

def apply_gaussian_blur(image, sigma=1.0):
    blurred_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        blurred_image[c] = gaussian_filter(image[c], sigma=sigma)
    return blurred_image

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        self.sentinel1_dir = os.path.join(data_dir, 'sentinel1')
        self.sentinel2_dir = os.path.join(data_dir, 'sentinel2')
        self.label_dir = os.path.join(data_dir, 'labels')

        self.sentinel1_files = sorted(os.listdir(self.sentinel1_dir))
        self.sentinel2_files = sorted(os.listdir(self.sentinel2_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        self.sentinel1_ids = [f.split('_')[-1].split('.')[0] for f in self.sentinel1_files]
        self.sentinel2_ids = [f.split('_')[-1].split('.')[0] for f in self.sentinel2_files]
        self.label_ids = [f.split('_')[-1].split('.')[0] for f in self.label_files]

        self.valid_ids = set(self.label_ids)
        self.sentinel1_files = [f for f, id_ in zip(self.sentinel1_files, self.sentinel1_ids) if id_ in self.valid_ids]
        self.sentinel2_files = [f for f, id_ in zip(self.sentinel2_files, self.sentinel2_ids) if id_ in self.valid_ids]
        self.sentinel1_ids = [id_ for id_ in self.sentinel1_ids if id_ in self.valid_ids]
        self.sentinel2_ids = [id_ for id_ in self.sentinel2_ids if id_ in self.valid_ids]

    def normalize_label(self, label_data):
        label_data[label_data == 255] = 1
        label_data[label_data != 1] = 0
        return label_data.astype(np.uint8)
       
    def compute_indices1(self, image):
        vv = image[0]  
        vh = image[1]  
        eps = 1e-6
        rvi = (4 * vh) / (vh + vv + eps) 
        rndvi = (vh - vv) / (vh + vv + eps)

        return rvi*0.00001, rndvi*0.00001
    
    def compute_indices2(self, image):
        blue = image[0]  
        green = image[1]  
        red = image[2]  
        nir = image[3]  
        eps = 1e-6

        ndvi = ((nir - red) / (nir + red + eps))
        gndvi = (nir - green) / (nir + green + eps)
        savi = ((nir - red) / (nir + red + 0.5 + eps)) * (1 + 0.5)

        return ndvi, gndvi, savi
       
    def __len__(self):
        return len(self.sentinel1_files)

    def __getitem__(self, idx):
        sentinel1_file = self.sentinel1_files[idx]
        sentinel2_file = self.sentinel2_files[idx]
        image_id = self.sentinel1_ids[idx]

        sentinel1_path = os.path.join(self.sentinel1_dir, sentinel1_file)
        sentinel2_path = os.path.join(self.sentinel2_dir, sentinel2_file)
        label_path = os.path.join(self.label_dir, f'ind_{image_id}.tif')

        if self.transforms:
            add_blur = True if random.random() < 0.3 else False
            add_noise = True if random.random() < 0.3 else False

        with rasterio.open(sentinel1_path) as img:
            sentinel1_data = img.read()                  
            sentinel1_data = np.nan_to_num(sentinel1_data, nan=0.0, posinf=0.0, neginf=0.0)

            if self.transforms:
                if add_blur:
                    sentinel1_data = apply_gaussian_blur(sentinel1_data)
                if add_noise:
                    sentinel1_data = add_speckle_noise(sentinel1_data)

            sentinel1_data = minmax_norm(sentinel1_data, minmax='sentinel1')

        with rasterio.open(sentinel2_path) as img:
            sentinel2_data = img.read()                  
            sentinel2_data = np.nan_to_num(sentinel2_data, nan=0.0, posinf=0.0, neginf=0.0)
            sentinel2_data = sentinel2_data * 0.0001

            if self.transforms:
                if add_blur:
                    sentinel2_data = apply_gaussian_blur(sentinel2_data)
                if add_noise:
                    sentinel2_data = add_speckle_noise(sentinel2_data)

            sentinel2_data = minmax_norm(sentinel2_data, minmax='sentinel2')

        with rasterio.open(label_path) as lbl:
            label = lbl.read(1)
            label = self.normalize_label(label)
            label = torch.tensor(label, dtype=torch.long)
            label = torch.nn.functional.one_hot(label, num_classes=2).permute(2, 0, 1).float()

        return sentinel1_data, sentinel2_data, label
    
def get_dataloaders(data_root, mode, batch_size=config.BATCH_SIZE):
    if mode=='training':
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')

        train_dataset = SatelliteDataset(train_dir, transforms=True)
        print(f"Dataset pelatihan berhasil dimuat dengan {len(train_dataset)} sampel.")
        val_dataset = SatelliteDataset(val_dir, transforms=False)
        print(f"Dataset validasi berhasil dimuat dengan {len(val_dataset)} sampel.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

        return train_loader, val_loader
    elif mode=='testing':
        test_dir = os.path.join(data_root, 'test')

        test_dataset = SatelliteDataset(test_dir, transforms=False)
        print(f"Dataset pengujian berhasil dimuat dengan {len(test_dataset)} sampel.")

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

        return test_loader