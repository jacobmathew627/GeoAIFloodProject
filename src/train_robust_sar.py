import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import UNet

# Config
BATCH_SIZE = 16
EPOCHS = 15
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model_robust_sar.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DIR = "processed"
PATCH_SIZE = 128 # Larger patches for better context with SAR
STRIDE = 64

def load_robust_dataset():
    """
    Loads 6 channels: DEM, Slope, Flow, LULC, SAR_VV, SAR_VH
    Label: Real 2018 Flood Mask
    """
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "LULC": "LULC_aligned.tif",
        "SAR_VV": "SAR_VV_aligned.tif",
        "SAR_VH": "SAR_VH_aligned.tif",
        "Label": "Label_aligned.tif"
    }

    data = {}
    print("Loading rasters...")
    for key, fname in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0)
            data[key] = arr

    # Normalization
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-6)
    
    dem = norm(data["DEM"])
    slope = norm(data["Slope"])
    flow = norm(data["Flow"])
    lulc = data["LULC"] / 255.0
    vv = norm(data["SAR_VV"])
    vh = norm(data["SAR_VH"])
    
    label = np.where(data["Label"] > 0, 1.0, 0.0).astype(np.float32)

    # Stack: (6, H, W)
    stack = np.stack([dem, slope, flow, lulc, vv, vh], axis=0)
    return stack, label

def extract_patches_robust(stack, label, patch_size=PATCH_SIZE, n_patches=8000):
    print(f"Sampling {n_patches} random patches...")
    C, H, W = stack.shape
    
    patches = []
    labels = []
    
    # Target balancing
    target_pos = n_patches // 2
    n_pos = 0
    n_neg = 0
    
    attempts = 0
    while (n_pos + n_neg) < n_patches and attempts < n_patches * 5:
        attempts += 1
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        
        l_patch = label[y:y+patch_size, x:x+patch_size]
        s_patch = stack[:, y:y+patch_size, x:x+patch_size]
        
        # Check if empty (nodata)
        if s_patch[0].sum() == 0: continue
        
        is_pos = l_patch.sum() > 10 # Some flooded pixels
        
        if is_pos and n_pos < target_pos:
            patches.append(s_patch)
            labels.append(l_patch[np.newaxis, ...])
            n_pos += 1
        elif not is_pos and n_neg < (n_patches - target_pos):
            patches.append(s_patch)
            labels.append(l_patch[np.newaxis, ...])
            n_neg += 1
            
    print(f"Extracted {n_pos} positive and {n_neg} negative patches.")
    X = np.array(patches)
    y = np.array(labels)
    
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    stack, label = load_robust_dataset()
    X, y = extract_patches_robust(stack, label)
    
    print(f"Final training set: {X.shape}, {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), test_size=0.15, random_state=42)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = UNet(n_channels=6, n_classes=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Starting Robust Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> Saved Robust Model")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Robust SAR-CNN Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_robust_sar.png')
    print(f"Training complete. Model: {MODEL_PATH}")

if __name__ == "__main__":
    train()
