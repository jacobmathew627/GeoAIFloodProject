import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import UNet

# Config
MODEL_PATH = os.path.join("models", "flood_model_real2018.pth")
OUTPUT_DIR = "outputs"
PROCESSED_DIR = "processed"
TILE_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_normalization_stats():
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "LULC": "LULC_aligned.tif",
        "TWI": "TWI_aligned.tif",
        "DistWater": "DistWater_aligned.tif",
        "DistUrban": "DistUrban_aligned.tif",
        "SAR_VV": "SAR_VV_aligned.tif",
        "SAR_VH": "SAR_VH_aligned.tif"
    }
    stats = {}
    print("Calculating normalization statistics...")
    for key, fname in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            stats[key] = (0, 1)
            continue
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0)
            stats[key] = (np.min(arr), np.max(arr))
            stats[key] = {'min': np.min(arr), 'max': np.max(arr)}
    return stats

def normalize(arr, mn, mx):
    if mx - mn == 0: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn + 1e-6)

def predict_flood_susceptibility(rainfall_mm=100, mode="standard"):
    """
    Runs multi-channel inference.
    mode: "standard" (4ch), "robust" (6ch), "supercharged" (9ch)
    """
    model_name_map = {
        "standard": "flood_model_real2018.pth",
        "robust": "flood_model_robust_sar.pth",
        "supercharged": "flood_model_supercharged.pth"
    }
    path_to_model = os.path.join("models", model_name_map[mode])
    n_channels = {"standard": 4, "robust": 6, "supercharged": 9}[mode]
    
    # Load Model
    model = UNet(n_channels=n_channels, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))
    model.eval()
    stats = get_normalization_stats()

    # Pre-allocate stack to save RAM
    dem_src = rasterio.open(os.path.join(PROCESSED_DIR, "DEM_aligned.tif"))
    H, W = dem_src.shape
    stack = np.zeros((n_channels, H, W), dtype=np.float32)
    
    # helper for in-place norm
    def load_and_norm_inplace(idx, path, stat_key):
        print(f"  Loading channel {idx}: {stat_key}...")
        with rasterio.open(path) as src:
            stack[idx] = src.read(1).astype(np.float32)
            s = stats[stat_key]
            stack[idx] -= s['min']
            stack[idx] /= (s['max'] - s['min'] + 1e-6)
            np.clip(stack[idx], 0, 1, out=stack[idx])

    # 1. Topography
    load_and_norm_inplace(0, os.path.join(PROCESSED_DIR, "DEM_aligned.tif"), 'DEM')
    load_and_norm_inplace(1, os.path.join(PROCESSED_DIR, "Slope_aligned.tif"), 'Slope')
    load_and_norm_inplace(2, os.path.join(PROCESSED_DIR, "Flow_aligned.tif"), 'Flow')
    
    with rasterio.open(os.path.join(PROCESSED_DIR, "LULC_aligned.tif")) as src:
        stack[3] = src.read(1).astype(np.float32) / 255.0

    # 2. Advanced Layers
    if mode == "supercharged":
        load_and_norm_inplace(4, os.path.join(PROCESSED_DIR, "TWI_aligned.tif"), 'TWI')
        load_and_norm_inplace(5, os.path.join(PROCESSED_DIR, "DistWater_aligned.tif"), 'DistWater')
        load_and_norm_inplace(6, os.path.join(PROCESSED_DIR, "DistUrban_aligned.tif"), 'DistUrban')
        load_and_norm_inplace(7, os.path.join(PROCESSED_DIR, "SAR_VV_aligned.tif"), 'SAR_VV')
        load_and_norm_inplace(8, os.path.join(PROCESSED_DIR, "SAR_VH_aligned.tif"), 'SAR_VH')
    elif mode == "robust":
        load_and_norm_inplace(4, os.path.join(PROCESSED_DIR, "SAR_VV_aligned.tif"), 'SAR_VV')
        load_and_norm_inplace(5, os.path.join(PROCESSED_DIR, "SAR_VH_aligned.tif"), 'SAR_VH')

    print(f"Running {mode} inference on {H}x{W} stack...")
    output = np.zeros((H, W), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(0, H, TILE_SIZE - 64):
            for j in range(0, W, TILE_SIZE - 64):
                h_end, w_end = min(i + TILE_SIZE, H), min(j + TILE_SIZE, W)
                tile = stack[:, i:h_end, j:w_end]
                curr_h, curr_w = h_end - i, w_end - j
                
                if curr_h < TILE_SIZE or curr_w < TILE_SIZE:
                    pad_tile = np.zeros((n_channels, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    pad_tile[:, :curr_h, :curr_w] = tile
                    tile_t = torch.from_numpy(pad_tile).unsqueeze(0).to(DEVICE)
                else:
                    tile_t = torch.from_numpy(tile).unsqueeze(0).to(DEVICE)
                
                prob = model(tile_t).squeeze().cpu().numpy()
                output[i:h_end, j:w_end] = np.maximum(output[i:h_end, j:w_end], prob[:curr_h, :curr_w])

    # Save Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = {"standard": "", "robust": "_robust", "supercharged": "_supercharged"}[mode]
    output_path = os.path.join(OUTPUT_DIR, f"flood_prob_{rainfall_mm}mm{suffix}.tif")
    
    with rasterio.open(
        output_path, 'w', driver='GTiff', height=H, width=W, count=1,
        dtype=output.dtype, crs=dem_src.crs, transform=dem_src.transform, nodata=-9999
    ) as dst:
        dst.write(output, 1)
        
    print(f"Saved: {output_path}")
    del output, stack
    dem_src.close()
    import gc; gc.collect()
    return output_path

if __name__ == "__main__":
    for mode in ["standard", "robust", "supercharged"]:
        m_file = {"standard": "flood_model_real2018.pth", "robust": "flood_model_robust_sar.pth", "supercharged": "flood_model_supercharged.pth"}[mode]
        if os.path.exists(os.path.join("models", m_file)):
            print(f"\nProcessing mode: {mode}")
            for r in [100, 150, 200]:
                predict_flood_susceptibility(r, mode=mode)
        else:
            print(f"Skipping {mode}: Model not found.")
