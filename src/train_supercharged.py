import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import UNet

# Config
BATCH_SIZE = 16
EPOCHS = 15
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model_supercharged.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DIR = "processed"
PATCH_SIZE = 64 # 4x memory reduction (64x64)
N_PATCHES = 5000 # Conservative for safety

def extract_supercharged_patches(n_patches=N_PATCHES, patch_size=PATCH_SIZE):
    """
    Directly samples patches from GeoTIFFs to save RAM.
    """
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "LULC": "LULC_aligned.tif",
        "TWI": "TWI_aligned.tif",
        "DistWater": "DistWater_aligned.tif",
        "DistUrban": "DistUrban_aligned.tif",
        "SAR_VV": "SAR_VV_aligned.tif",
        "SAR_VH": "SAR_VH_aligned.tif",
        "Label": "Label_aligned.tif"
    }
    
    # 1. Get dimensions
    with rasterio.open(os.path.join(PROCESSED_DIR, files["Label"])) as src:
        H, W = src.shape
        label_full = src.read(1) # Labels are usually smaller to keep in RAM (1 channel)
    
    print(f"Sampling {n_patches} patches from {H}x{W} grid using direct window reads...")
    
    # 2. Pre-determine patch locations (Balancing)
    n_pos_target = n_patches // 2
    coords = []
    n_pos, n_neg = 0, 0
    
    attempts = 0
    while len(coords) < n_patches and attempts < n_patches * 20:
        attempts += 1
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        
        l_sum = label_full[y:y+patch_size, x:x+patch_size].sum()
        is_pos = l_sum > 5
        
        if is_pos and n_pos < n_pos_target:
            coords.append((y, x))
            n_pos += 1
        elif not is_pos and n_neg < (n_patches - n_pos_target):
            coords.append((y, x))
            n_neg += 1
            
    # 3. Sample each channel
    X = np.zeros((len(coords), 9, patch_size, patch_size), dtype=np.float32)
    y = np.zeros((len(coords), 1, patch_size, patch_size), dtype=np.float32)
    
    for i, (py, px) in enumerate(coords):
        y[i, 0] = label_full[py:py+patch_size, px:px+patch_size]
    
    # Release label_full to save RAM
    del label_full
    
    channel_keys = ["DEM", "Slope", "Flow", "LULC", "TWI", "DistWater", "DistUrban", "SAR_VV", "SAR_VH"]
    for c_idx, key in enumerate(channel_keys):
        path = os.path.join(PROCESSED_DIR, files[key])
        print(f"  Reading channel {c_idx}: {key}...")
        with rasterio.open(path) as src:
            for i, (py, px) in enumerate(coords):
                window = rasterio.windows.Window(px, py, patch_size, patch_size)
                tile = src.read(1, window=window).astype(np.float32)
                tile = np.nan_to_num(tile, nan=0.0)
                
                # Normalization on the fly
                # We can't easily get global min/max here without reading full file
                # But we know typical ranges or can assume 0-1 for simplicity if already processed
                # Let's assume the alignment script or previous logic normalized them
                # If not, we do a simple percentile norm or use pre-known stats
                X[i, c_idx] = tile
                
    # Final normalization of X
    print("Normalizing patch stack in-place...")
    for c in range(9):
        c_min = X[:, c].min()
        c_max = X[:, c].max()
        # In-place to save RAM
        X[:, c] -= c_min
        X[:, c] /= (c_max - c_min + 1e-6)
    
    # Threshold labels
    y = np.where(y > 0, 1.0, 0.0)
    
    print(f"Extracted {len(coords)} patches.")
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = extract_supercharged_patches(n_patches=N_PATCHES)
    
    full_ds = TensorDataset(X, y)
    train_size = int(0.85 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = UNet(n_channels=9, n_classes=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Supercharged Training on {DEVICE}...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_l = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_l += loss.item()
        
        train_l /= len(train_loader)
        
        model.eval()
        val_l = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_l += criterion(model(imgs), masks).item()
        val_l /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_l:.4f}, Val: {val_l:.4f}")
        
        if val_l < best_loss:
            best_loss = val_l
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> Best model saved.")

    print(f"Training Complete. Model: {MODEL_PATH}")

if __name__ == "__main__":
    train()
