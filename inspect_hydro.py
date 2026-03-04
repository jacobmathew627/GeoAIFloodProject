import rasterio
import numpy as np
import os

PROCESSED_DIR = "processed"
layers = ["DEM_aligned.tif", "Slope_aligned.tif", "Flow_aligned.tif", "LULC_aligned.tif", "TWI_aligned.tif", "DistWater_aligned.tif"]

print(f"{'Layer':<20} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
print("-" * 65)

for layer in layers:
    path = os.path.join(PROCESSED_DIR, layer)
    if not os.path.exists(path):
        print(f"{layer:<20} | MISSING")
        continue
        
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            mask = data != nodata
            valid_data = data[mask]
        else:
            valid_data = data.flatten()
            
        if len(valid_data) == 0:
            print(f"{layer:<20} | ALL NODATA")
            continue
            
        print(f"{layer:<20} | {np.min(valid_data):10.2f} | {np.max(valid_data):10.2f} | {np.mean(valid_data):10.2f} | {np.std(valid_data):10.2f}")

# Check water pixels in LULC
path = os.path.join(PROCESSED_DIR, "LULC_aligned.tif")
if os.path.exists(path):
    with rasterio.open(path) as src:
        lulc = src.read(1)
        water_count = np.sum(np.isin(lulc, [80, 90, 95]))
        total_pixels = lulc.size
        print(f"\nLULC Water Pixels (80, 90, 95): {water_count} / {total_pixels} ({water_count/total_pixels:.4%})")
