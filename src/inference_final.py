import os
import sys
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_final import GeoAIUNet

PROCESSED_DIR = "processed"
MODEL_PATH = "models/geoai_flood_final.pth"
OUTPUT_DIR = "outputs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TILE_SIZE = 64
STEP = 32
RAINFALL_SCENARIOS = [100, 150, 200]  # mm

CHANNEL_FILES = [
    ("DEM",          "DEM_aligned.tif"),
    ("Slope",        "Slope_aligned.tif"),
    ("Flow",         "Flow_aligned.tif"),
    ("DistDrainage", "DistDrainage_aligned.tif"),
    ("DrainageDens", "DrainageDensity_aligned.tif"),
    ("BuiltupDens",  "BuiltupDensity_aligned.tif"),
]

def load_and_normalize(path, nodata=-9999):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else nodata
    data[data == nd] = np.nan
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)
    else:
        data = np.zeros_like(data)
    data = np.nan_to_num(data, nan=0.0)
    return data

def run_inference(rainfall_mm: float = 150.0):
    """
    Performs tile-based inference over the full Ernakulam raster.
    rainfall_mm: user-provided scalar controlling risk intensity.
    Returns a float32 numpy array of flood probability (H, W).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rain_norm = np.clip(rainfall_mm / 300.0, 0.0, 1.0)

    print(f"Loading model from {MODEL_PATH}...")
    model = GeoAIUNet(n_channels=7).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print("Loading and normalising spatial layers...")
    channels = []
    ref_profile = None
    for name, fname in CHANNEL_FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        arr = load_and_normalize(path)
        channels.append(arr)
        if ref_profile is None:
            with rasterio.open(path) as src:
                ref_profile = src.profile.copy()

    H, W = channels[0].shape
    stack = np.stack(channels, axis=0)  # (6, H, W)

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    rain_patch = np.full((TILE_SIZE, TILE_SIZE), rain_norm, dtype=np.float32)

    print(f"Running tile-based inference (rainfall={rainfall_mm}mm)...")
    total_tiles = 0
    with torch.no_grad():
        for r in range(0, H - TILE_SIZE + 1, STEP):
            for c in range(0, W - TILE_SIZE + 1, STEP):
                spatial = stack[:, r:r+TILE_SIZE, c:c+TILE_SIZE]  # (6, T, T)
                tile = np.concatenate([spatial, rain_patch[np.newaxis]], axis=0)  # (7, T, T)
                tensor = torch.from_numpy(tile).unsqueeze(0).to(DEVICE)
                out = model(tensor).squeeze().cpu().numpy()
                prob_map[r:r+TILE_SIZE, c:c+TILE_SIZE] += out
                count_map[r:r+TILE_SIZE, c:c+TILE_SIZE] += 1.0
                total_tiles += 1

    count_map[count_map == 0] = 1
    prob_map /= count_map
    print(f"Inference complete. Processed {total_tiles} tiles.")

    out_path = os.path.join(OUTPUT_DIR, f"flood_prob_final_{int(rainfall_mm)}mm.tif")
    ref_profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
    with rasterio.open(out_path, 'w', **ref_profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)
    print(f"Saved: {out_path}")
    return prob_map, ref_profile

if __name__ == "__main__":
    for r in RAINFALL_SCENARIOS:
        run_inference(rainfall_mm=r)
    print("All scenarios complete.")
