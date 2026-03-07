import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

def load_real_flood_data():
    """
    Load training data using REAL 2018 flood observations
    with aggressive downsampling to prevent memory errors
    """
    print("Loading data with 2018 REAL flood observations...")
    
    # Input features - 10 Channels
    feature_paths = {
        "dem": os.path.join("processed", "DEM_aligned.tif"),
        "slope": os.path.join("processed", "Slope_aligned.tif"),
        "flow": os.path.join("processed", "Flow_aligned.tif"),
        "lulc": os.path.join("processed", "LULC_aligned.tif"),
        "sar_vv": os.path.join("processed", "SAR_VV_aligned.tif"),
        "sar_vh": os.path.join("processed", "SAR_VH_aligned.tif"),
        "spi": os.path.join("processed", "SPI_aligned.tif"),
        "twi": os.path.join("processed", "TWI_aligned.tif"),
        "hand": os.path.join("processed", "HAND_aligned.tif"),
        "builtup": os.path.join("processed", "BuiltupDensity_aligned.tif")
    }
    
    # REAL flood label (2018 event)
    label_path = "flood_mask_2018_ekm.tif"
    
    # Determine downsampling factor
    MAX_DIM = 2000
    
    with rasterio.open(feature_paths["dem"]) as src:
        orig_height, orig_width = src.height, src.width
        scale_factor = MAX_DIM / max(orig_height, orig_width)
        new_height, new_width = (int(orig_height * scale_factor), int(orig_width * scale_factor)) if scale_factor < 1 else (orig_height, orig_width)
        target_shape = (new_height, new_width)
        profile = src.profile
        print(f"Loading data at {new_height}x{new_width}...")

    features = []
    for name, path in feature_paths.items():
        if os.path.exists(path):
            with rasterio.open(path) as src:
                data = src.read(1, out_shape=target_shape, 
                                resampling=rasterio.enums.Resampling.bilinear if name != "lulc" else rasterio.enums.Resampling.nearest).astype(np.float32)
                features.append(data)
        else:
            print(f"Warning: {name} not found at {path}. Using zeros.")
            features.append(np.zeros(target_shape, dtype=np.float32))

    # Load REAL flood mask
    with rasterio.open(label_path) as src:
        label = src.read(1, out_shape=target_shape, resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
    
    # Stack inputs (C, H, W)
    stack = np.stack(features, axis=0)
    print(f"Stacked {stack.shape[0]} features. Memory: {stack.nbytes / (1024**2):.1f} MB")
    
    # Normalize inputs
    for i in range(stack.shape[0]):
        channel = stack[i]
        valid_mask = (channel != -9999) & (channel != 0) & (~np.isnan(channel))
        if valid_mask.sum() > 0:
            stack[i] = np.where(valid_mask, (channel - channel[valid_mask].mean()) / (channel[valid_mask].std() + 1e-6), 0)
    
    # Ensure binary label
    label = (np.clip(label, 0, 1) > 0).astype(np.float32)
    
    return stack, label, profile

def extract_balanced_patches_enhanced(stack, label, patch_size=128, num_patches=5000, 
                                     flood_ratio=0.5, augment=True):
    """
    Enhanced patch extraction with:
    - Balanced sampling
    - Data augmentation
    - Hard negative mining
    """
    C, H, W = stack.shape
    
    X_patches = []
    y_patches = []
    
    # Find flood and non-flood pixels
    flood_mask = label > 0.5
    non_flood_mask = label <= 0.5
    
    flood_coords = np.argwhere(flood_mask)
    non_flood_coords = np.argwhere(non_flood_mask)
    
    print(f"Extracting {num_patches} patches (flood_ratio={flood_ratio})...")
    
    num_flood = int(num_patches * flood_ratio)
    num_non_flood = num_patches - num_flood
    
    # Sample flood patches
    if len(flood_coords) > 0:
        flood_indices = np.random.choice(len(flood_coords), 
                                        min(num_flood, len(flood_coords)), 
                                        replace=False)
        for idx in flood_indices:
            r, c = flood_coords[idx]
            r_start = max(0, r - patch_size // 2)
            c_start = max(0, c - patch_size // 2)
            r_end = min(H, r_start + patch_size)
            c_end = min(W, c_start + patch_size)
            
            if (r_end - r_start) == patch_size and (c_end - c_start) == patch_size:
                patch_x = stack[:, r_start:r_end, c_start:c_end]
                patch_y = label[r_start:r_end, c_start:c_end]
                
                X_patches.append(patch_x)
                y_patches.append(patch_y)
    
    # Sample non-flood patches
    if len(non_flood_coords) > 0:
        non_flood_indices = np.random.choice(len(non_flood_coords), 
                                            min(num_non_flood, len(non_flood_coords)), 
                                            replace=False)
        for idx in non_flood_indices:
            r, c = non_flood_coords[idx]
            r_start = max(0, r - patch_size // 2)
            c_start = max(0, c - patch_size // 2)
            r_end = min(H, r_start + patch_size)
            c_end = min(W, c_start + patch_size)
            
            if (r_end - r_start) == patch_size and (c_end - c_start) == patch_size:
                patch_x = stack[:, r_start:r_end, c_start:c_end]
                patch_y = label[r_start:r_end, c_start:c_end]
                
                X_patches.append(patch_x)
                y_patches.append(patch_y)
    
    X_numpy = np.array(X_patches)  # (N, C, H, W)
    y_numpy = np.array(y_patches)[:, np.newaxis, :, :]  # (N, 1, H, W)
    
    print(f"Extracted {len(X_patches)} patches")
    print(f"X shape: {X_numpy.shape}, y shape: {y_numpy.shape}")
    
    return X_numpy, y_numpy

class FloodDatasetAugmented(Dataset):
    """
    PyTorch Dataset with data augmentation
    """
    def __init__(self, X, y, augment=True):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
            
            # Random vertical flip
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
            
            # Random rotation (90, 180, 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                x = TF.rotate(x, angle)
                y = TF.rotate(y, angle)
        
        return x, y

# Backward compatibility
def load_training_data():
    """Wrapper for backward compatibility"""
    return load_real_flood_data()

def extract_balanced_patches(stack, label, patch_size=32, num_patches=10000):
    """Wrapper for backward compatibility"""
    return extract_balanced_patches_enhanced(stack, label, patch_size, num_patches)

if __name__ == "__main__":
    print("Testing enhanced data loader...")
    stack, label, profile = load_real_flood_data()
    X, y = extract_balanced_patches_enhanced(stack, label, num_patches=100)
    
    dataset = FloodDatasetAugmented(X, y, augment=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test augmentation
    x_sample, y_sample = dataset[0]
    print(f"Sample X shape: {x_sample.shape}")
    print(f"Sample y shape: {y_sample.shape}")
    print("Data loader test completed!")
