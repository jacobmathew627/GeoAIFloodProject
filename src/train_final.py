import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from model_final import GeoAIUNet
import json

# Config
BATCH_SIZE = 16
EPOCHS = 20
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "geoai_flood_final.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DIR = "processed"
PATCH_SIZE = 64
N_PATCHES = 5000

# We use 6 static spatial input channels + 1 dynamic rainfall channel = 7 channels
# Channel order: [DEM, Slope, Flow, DistDrainage, DrainageDens, BuiltupDens, Rainfall]
def extract_final_patches(n_patches=N_PATCHES, patch_size=PATCH_SIZE):
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "DistDrainage": "DistDrainage_aligned.tif",
        "DrainageDens": "DrainageDensity_aligned.tif",
        "BuiltupDens": "BuiltupDensity_aligned.tif",
        "Label": "Label_aligned.tif" # Used Sentinel-1 SAR 2018 flood mask
    }
    
    with rasterio.open(os.path.join(PROCESSED_DIR, "Label_aligned.tif")) as src:
        H, W = src.shape
        label_full = src.read(1)
    
    print(f"Sampling {n_patches} patches from {H}x{W} raster...")
    
    # Balanced sampling
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
            
    # 7 channels
    X = np.zeros((len(coords), 7, patch_size, patch_size), dtype=np.float32)
    y = np.zeros((len(coords), 1, patch_size, patch_size), dtype=np.float32)
    
    for i, (py, px) in enumerate(coords):
        y[i, 0] = label_full[py:py+patch_size, px:px+patch_size]
    
    del label_full
    
    channel_keys = ["DEM", "Slope", "Flow", "DistDrainage", "DrainageDens", "BuiltupDens"]
    
    for c_idx, key in enumerate(channel_keys):
        path = os.path.join(PROCESSED_DIR, files[key])
        with rasterio.open(path) as src:
            for i, (py, px) in enumerate(coords):
                window = rasterio.windows.Window(px, py, patch_size, patch_size)
                tile = src.read(1, window=window).astype(np.float32)
                tile = np.nan_to_num(tile, nan=0.0)
                X[i, c_idx] = tile
                
    print("Normalizing spatial features...")
    for c in range(6):
        c_min = X[:, c].min()
        c_max = X[:, c].max()
        if c_max > c_min:
            X[:, c] = (X[:, c] - c_min) / (c_max - c_min)
        else:
            X[:, c] = 0.0

    # Channel 6 is Rainfall. 
    # During the 2018 Kerala Flood, average extreme rainfall in Ernakulam was ~200mm to 300mm/day.
    # We simulate training samples where flood label=1 pairs with high rainfall (e.g., 200mm),
    # and label=0 pairs with a random lower rainfall OR high rainfall but high elevation/no flood.
    # To keep it simple, we inject uniform scalar patches 
    print("Injecting rainfall scenarios as 7th channel multiplier...")
    for i in range(len(coords)):
        is_flood = y[i].max() > 0
        if is_flood:
            # For 2018 event, assign high rainfall uniformly
            rain_mm = np.random.uniform(150.0, 300.0)
        else:
            # For non-flood, mostly lower rainfall or high rain on good terrain
            rain_mm = np.random.uniform(0.0, 150.0)
            if np.random.rand() < 0.2: # 20% high rain but safe terrain
                rain_mm = np.random.uniform(150.0, 300.0)
                
        # Normalize rainfall against 300mm max slider value
        rain_norm = rain_mm / 300.0
        X[i, 6] = np.full((patch_size, patch_size), rain_norm, dtype=np.float32)
    
    y = np.where(y > 0, 1.0, 0.0)
    print(f"Dataset Built: X shape {X.shape}, y shape {y.shape}")
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def train_and_eval():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = extract_final_patches()
    
    full_ds = TensorDataset(X, y)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = GeoAIUNet(n_channels=7, n_classes=1).to(DEVICE)
    # Using DiceLoss + BCE combination is often better, but stick to BCE for now to prevent over complication unless requested
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_iou = 0.0
    
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
        
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                val_l += criterion(out, masks).item()
                
                preds = (out > 0.5).float()
                
                y_true_list.append(masks.cpu().numpy().flatten())
                y_pred_list.append(preds.cpu().numpy().flatten())
                
        val_l /= len(val_loader)
        
        y_true_arr = np.concatenate(y_true_list)
        y_pred_arr = np.concatenate(y_pred_list)
        
        p = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        r = recall_score(y_true_arr, y_pred_arr, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        iou = jaccard_score(y_true_arr, y_pred_arr, zero_division=0)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_l:.4f} | Val Loss: {val_l:.4f} | Val IoU: {iou:.4f} | F1: {f1:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), MODEL_PATH)
            
            # Save eval metrics
            metrics = {
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f1),
                "iou": float(iou)
            }
            with open(os.path.join(MODEL_DIR, "final_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_and_eval()
