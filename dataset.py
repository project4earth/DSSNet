import os
import torch
import random
import rasterio
import numpy as np
from utils import config
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

def minmax_norm(image):
    num_channels = image.shape[0]

    lower_val = [
            -55.5199,
            -59.9488,
            -2.8646,
            -1.4323,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.2014,
            -2.1762,
            -0.5324,
        ] 
    upper_val = [
            26.5266,
            11.7958,
            3.2421,
            1.6210,
            1.3224,
            1.4688,
            1.6036,
            1.6421,
            3.0082,
            1.3467,
            0.8394,
        ]

    normalized_image = np.zeros_like(image, dtype=np.float32)

    for c in range(num_channels):
        channel_data = image[c]
        normalized_channel = (channel_data - lower_val[c]) / (upper_val[c] - lower_val[c])
        normalized_channel = np.clip(normalized_channel, 0, 1)
        normalized_image[c] = normalized_channel

    return normalized_image

def add_speckle_noise(image, prob=0.3, mean=0, std=0.1):
    if random.random() < prob:
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + image * noise
        return noisy_image
    else:
        return image

def apply_gaussian_blur(image, prob=0.3, sigma=1.0):
    if random.random() < prob:
        blurred_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            blurred_image[c] = gaussian_filter(image[c], sigma=sigma)
        return blurred_image
    else:
        return image

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # Path ke folder citra Sentinel-1, Sentinel-2, dan label
        self.sentinel1_dir = os.path.join(data_dir, 'sentinel1')
        self.sentinel2_dir = os.path.join(data_dir, 'sentinel2')
        self.label_dir = os.path.join(data_dir, 'labels')

        # Daftar file citra dan label
        self.sentinel1_files = sorted(os.listdir(self.sentinel1_dir))
        self.sentinel2_files = sorted(os.listdir(self.sentinel2_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        # Ekstrak ID unik dari nama file citra dan label
        self.sentinel1_ids = [f.split('_')[-1].split('.')[0] for f in self.sentinel1_files]
        self.sentinel2_ids = [f.split('_')[-1].split('.')[0] for f in self.sentinel2_files]
        self.label_ids = [f.split('_')[-1].split('.')[0] for f in self.label_files]

        # Pastikan semua ID di citra ada di label
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
        # Ambil nama file dan ID
        sentinel1_file = self.sentinel1_files[idx]
        sentinel2_file = self.sentinel2_files[idx]
        image_id = self.sentinel1_ids[idx]

        # Path ke citra Sentinel-1, Sentinel-2, dan label
        sentinel1_path = os.path.join(self.sentinel1_dir, sentinel1_file)
        sentinel2_path = os.path.join(self.sentinel2_dir, sentinel2_file)
        label_path = os.path.join(self.label_dir, f'ind_{image_id}.tif')

        # Membaca citra Sentinel-1 dengan rasterio
        with rasterio.open(sentinel1_path) as img:
            sentinel1_data = img.read()                  
            sentinel1_data = np.nan_to_num(sentinel1_data, nan=0.0, posinf=0.0, neginf=0.0)

            rvi, rndvi = self.compute_indices1(sentinel1_data)
            indices1 = np.stack([rvi, rndvi], axis=0)

            sentinel1_data = np.concatenate([sentinel1_data, indices1], axis=0)

        # Membaca citra Sentinel-2 dengan rasterio
        with rasterio.open(sentinel2_path) as img:
            sentinel2_data = img.read()                  
            sentinel2_data = np.nan_to_num(sentinel2_data, nan=0.0, posinf=0.0, neginf=0.0)
            sentinel2_data = sentinel2_data * 0.0001

            ndvi, gndvi, savi = self.compute_indices2(sentinel2_data)
            indices2 = np.stack([ndvi, gndvi, savi], axis=0)

            sentinel2_data = np.concatenate([sentinel2_data, indices2], axis=0)

        sentinel_data = np.concatenate([sentinel1_data, sentinel2_data], axis=0)

        if self.transforms == True:
            sentinel_data = apply_gaussian_blur(sentinel_data)
            sentinel_data = add_speckle_noise(sentinel_data)

        sentinel_data = minmax_norm(sentinel_data)
        sentinel_data = torch.tensor(sentinel_data, dtype=torch.float32)

        # Membaca label dengan rasterio
        with rasterio.open(label_path) as lbl:
            label = lbl.read(1)  # Membaca channel pertama
            label = self.normalize_label(label)
            label = torch.tensor(label, dtype=torch.long)
            label = torch.nn.functional.one_hot(label, num_classes=2).permute(2, 0, 1).float()

        return sentinel_data, label
    
def get_dataloaders(data_root, mode, batch_size=config.BATCH_SIZE):
    if mode=='training':
         # Path untuk train, eval, dan test
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')

        # Membuat dataset untuk train, eval, dan test
        train_dataset = SatelliteDataset(train_dir, transforms=True)
        print(f"Dataset pelatihan berhasil dimuat dengan {len(train_dataset)} sampel.")
        val_dataset = SatelliteDataset(val_dir, transforms=False)
        print(f"Dataset validasi berhasil dimuat dengan {len(val_dataset)} sampel.")

        # Membuat DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

        return train_loader, val_loader
    elif mode=='testing':
        # Path untuk train, eval, dan test
        test_dir = os.path.join(data_root, 'test')

        # Membuat dataset untuk train, eval, dan test
        test_dataset = SatelliteDataset(test_dir, transforms=False)
        print(f"Dataset pengujian berhasil dimuat dengan {len(test_dataset)} sampel.")

        # Membuat DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

        return test_loader