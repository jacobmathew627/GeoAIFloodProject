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

# MEMORY-OPTIMIZED CONFIG
BATCH_SIZE = 16
EPOCHS = 15
MODEL_DIR = "models"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_DIM = 1500  # Aggressive downsampling

def load_real_flood_data_simple():
    """Load 2018 REAL flood data with aggressive memory optimization"""
    print("Loading REAL 2018 flood observations...")
    
    # Paths
    dem_path = os.path.join("processed", "DEM_aligned.tif")
    slope_path = os.path.join("processed", "Slope_aligned.tif")
    flow_path = os.path.join("processed", "Flow_aligned.tif")
    lulc_path = os.path.join("processed", "LULC_aligned.tif")
    label_path = "flood_mask_2018_ekm.tif"  # REAL FLOOD DATA
    
    # Determine target size
    with rasterio.open(dem_path) as src:
        orig_h, orig_w = src.height, src.width
        scale = MAX_DIM / max(orig_h, orig_w)
        new_h = int(orig_h * scale) if scale < 1 else orig_h
        new_w = int(orig_w * scale) if scale < 1 else orig_w
        print(f"Downsampling: {orig_h}x{orig_w} -> {new_h}x{new_w}")
        
        # Load and downsample
        dem = src.read(1, out_shape=(new_h, new_w), 
                      resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    
    with rasterio.open(slope_path) as src:
        slope = src.read(1, out_shape=(new_h, new_w),
                        resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    
    with rasterio.open(flow_path) as src:
        flow = src.read(1, out_shape=(new_h, new_w),
                       resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1, out_shape=(new_h, new_w),
                       resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
    
    # Load REAL flood mask
    with rasterio.open(label_path) as src:
        label = src.read(1, out_shape=(new_h, new_w),
                        resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
    
    # Normalize
    dem = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem) + 1e-6)
    slope = (slope - np.nanmin(slope)) / (np.nanmax(slope) - np.nanmin(slope) + 1e-6)
    flow = (flow - np.nanmin(flow)) / (np.nanmax(flow) - np.nanmin(flow) + 1e-6)
    lulc = lulc / 255.0
    
    # Replace NaN with 0
    dem = np.nan_to_num(dem, 0)
    slope = np.nan_to_num(slope, 0)
    flow = np.nan_to_num(flow, 0)
    lulc = np.nan_to_num(lulc, 0)
    
    # Binary label
    label = (label > 0).astype(np.float32)
    
    print(f"Flood pixels: {label.sum():,.0f} ({label.sum()/label.size*100:.2f}%)")
    
    # Stack (no UWI - just physical features!)
    stack = np.stack([dem, slope, flow, lulc], axis=0)  # 4 channels
    
    return stack, label

def extract_patches_simple(stack, label, patch_size=32, max_patches=8000):
    """Simple patch extraction with balancing"""
    C, H, W = stack.shape
    
    patches_x = []
    patches_y = []
    
    # Find flood and non-flood locations
    flood_locs = np.argwhere(label > 0)
    non_flood_locs = np.argwhere(label == 0)
    
    print(f"Extracting up to {max_patches} patches...")
    
    # Sample flood patches
    n_flood = min(max_patches // 2, len(flood_locs))
    flood_sample = flood_locs[np.random.choice(len(flood_locs), n_flood, replace=False)]
    
    for r, c in flood_sample:
        r_start = max(0, r - patch_size//2)
        c_start = max(0, c - patch_size//2)
        r_end = min(H, r_start + patch_size)
        c_end = min(W, c_start + patch_size)
        
        if (r_end - r_start) == patch_size and (c_end - c_start) == patch_size:
            patch_x = stack[:, r_start:r_end, c_start:c_end]
            patch_y = label[r_start:r_end, c_start:c_end]
            patches_x.append(patch_x)
            patches_y.append(patch_y)
    
    # Sample non-flood patches
    n_non_flood = min(max_patches - len(patches_x), len(non_flood_locs))
    non_flood_sample = non_flood_locs[np.random.choice(len(non_flood_locs), n_non_flood, replace=False)]
    
    for r, c in non_flood_sample:
        r_start = max(0, r - patch_size//2)
        c_start = max(0, c - patch_size//2)
        r_end = min(H, r_start + patch_size)
        c_end = min(W, c_start + patch_size)
        
        if (r_end - r_start) == patch_size and (c_end - c_start) == patch_size:
            patch_x = stack[:, r_start:r_end, c_start:c_end]
            patch_y = label[r_start:r_end, c_start:c_end]
            patches_x.append(patch_x)
            patches_y.append(patch_y)
    
    X = np.array(patches_x)  # (N, C, H, W)
    y = np.array(patches_y)[:, np.newaxis, :, :]  # (N, 1, H, W)
    
    print(f"Extracted {len(X)} patches")
    return X, y

def train_real_flood():
    """Train on REAL 2018 flood data"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("="*70)
    print("TRAINING ON REAL 2018 FLOOD DATA (Not Physics!)")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Load REAL flood data
    stack, label = load_real_flood_data_simple()
    X, y = extract_patches_simple(stack, label, patch_size=32, max_patches=8000)
    
    print(f"\nDataset: X={X.shape}, y={y.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model (4 channels now - no UWI!)
    model = UNet(n_channels=4, n_classes=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("="*70)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = train_loss / len(train_ds)
        history['train_loss'].append(epoch_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_loss / len(val_ds)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "flood_model_real2018.pth"))
            print(f"  -> Best model saved!")
    
    print("\n" + "="*70)
    print(f"Training complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: models/flood_model_real2018.pth")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training on Real 2018 Flood Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_real2018.png', dpi=150)
    print("Training plot saved: training_real2018.png")
    
    return model, history

if __name__ == "__main__":
    model, history = train_real_flood()
    print("\n[SUCCESS] Model trained on REAL flood observations!")
    print("\nNext: Update inference.py to use this model")
