"""
DEPRECATED -- DO NOT RUN. Kept only as a record of how the
`flood_prob_*mm_supercharged.tif` files in outputs/ were produced.

This script does not use the trained model at all. Its own comments describe
it as a "Mock-Simulation" that uses weighted Multi-Criteria Decision Analysis
to "simulate exactly what the trained Attention U-Net would predict". The
rasters it wrote were then surfaced in the dashboard as AI model output, and
the rainfall response was three hard-coded multipliers (0.8 / 1.6 / 2.6)
applied directly to a probability, which is why the shipped 100 mm map had a
higher mean probability than the 150 mm map.

It is also broken independently of that: TEMPLATE_PATH points at
`processed/LULC_aligned.tif`, a directory that is gitignored and absent.

Replaced by:
    python src/susceptibility.py --train --predict   # learned susceptibility
    python src/hazard.py                             # SCS-CN rainfall response
"""

import os
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GEOAI_NEW_DIR, OUTPUT_DIR, PROCESSED_DIR  # noqa: E402

DATA_DIR = str(GEOAI_NEW_DIR)
OUT_DIR = str(OUTPUT_DIR)
TEMPLATE_PATH = str(PROCESSED_DIR / "LULC_aligned.tif")

raise SystemExit(
    __doc__.strip() + "\n\nRefusing to run: this would overwrite outputs/ with non-model maps."
)

# 1. Load Alignment Template
with rasterio.open(TEMPLATE_PATH) as src:
    master_transform = src.transform
    master_crs = src.crs
    MASTER_SHAPE = src.shape
    print(f"Master Shape: {MASTER_SHAPE} | CRS: {master_crs}")


def load_and_warp(filename, is_categorical=False):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        # Fallback to older if new isn't found
        old_path = os.path.join(r"C:\Users\Asus\Documents\GeoAI_Flood_Project\GeoAI_Data", filename)
        if os.path.exists(old_path):
            path = old_path
        else:
            print(f"[WARN] Cannot find {filename}")
            return None

    with rasterio.open(path) as src:
        source_array = src.read(1)
        dest_array = np.zeros(MASTER_SHAPE, dtype=np.float32)
        resamp = Resampling.nearest if is_categorical else Resampling.bilinear

        reproject(
            source=source_array,
            destination=dest_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            resampling=resamp,
        )
        return dest_array


# 2. Extract Key Advanced Layers
print("\nExtracting & Warping Advanced TIFs...")
flow = load_and_warp("Ernakulam_Flow_Accumulation.tif")
slope = load_and_warp("Ernakulam_Slope.tif")
dist_urban = load_and_warp("Distance_to_Builtup_Final.tif")
ndwi = load_and_warp("NDWI_Aligned.tif")

# Crucial addition: Use LULC as the strict geographic boundary mask
lulc = load_and_warp("Ernakulam_LULC_2018.tif", is_categorical=True)

if any(v is None for v in [flow, slope, dist_urban, ndwi, lulc]):
    print("CRITICAL: Missing files!")
    exit(1)

# 3. Intelligent Normalization & Feature Engineering
print("Engineering Physics Constraints...")

# Normalizing Flow Accumulation (Logarithmic scale preferred)
flow_valid = flow > -9000
flow_safe = np.where(flow_valid, flow, 0)
flow_log = np.log1p(np.clip(flow_safe, 0, None))
flow_norm = flow_log / (np.percentile(flow_log[flow_log > 0], 99) + 1e-6)
flow_norm = np.clip(flow_norm, 0, 1)

# Inverse Slope (Flatter = Higher Flood Risk)
slope_valid = slope > -9000
slope_safe = np.where(slope_valid, slope, 90)  # default to 90 deg (no flood)
slope_norm = slope_safe / 90.0
slope_inv = np.clip(1.0 - slope_norm, 0, 1)

# Inverse Distance to Urban (Closer to Urban = higher impermeable runoff)
# Let's say max impact radius is 2000 meters.
urban_valid = dist_urban > -9000
urban_safe = np.where(urban_valid, dist_urban, 5000)
urban_risk = np.clip(1.0 - (urban_safe / 2000.0), 0, 1)

# NDWI (Higher = already water or highly saturated)
# Range usually -1 to 1. Scale 0-1
ndwi_valid = ndwi > -9000
ndwi_safe = np.where(ndwi_valid, ndwi, -1)
ndwi_norm = np.clip((ndwi_safe + 1.0) / 2.0, 0, 1)

# Identify valid landmask (Strictly clipped to LULC boundaries to prevent ocean spillage)
land_mask = (lulc > 0) & (lulc < 100)

# 4. Generate Supercharged Probability Maps
# We simulate the AI's complex weighting:
# Flood Prob = Base Rainfall factor * (35% Flow + 25% Slope + 20% NDWI + 20% Urban Runoff)
weights = {"flow": 0.35, "slope": 0.25, "ndwi": 0.20, "urban": 0.20}
base_risk_matrix = (
    flow_norm * weights["flow"]
    + slope_inv * weights["slope"]
    + ndwi_norm * weights["ndwi"]
    + urban_risk * weights["urban"]
)

scenarios = [100, 150, 200]
print("\nGenerating Intelligent Base Maps...")

for rain in scenarios:
    # Scale risk logarithmically based on rainfall (calibration curve)
    # Calibrated multipliers to ensure 200mm dynamically triggers Critical Socio-Economic warnings
    if rain == 100:
        rain_factor = 0.8
    elif rain == 150:
        rain_factor = 1.6
    elif rain == 200:
        rain_factor = 2.6

    # Mathematical simulation of AI probability
    prob_map = base_risk_matrix * rain_factor
    prob_map = np.clip(prob_map, 0.0, 1.0)

    # Re-apply nodata mask
    prob_map[~land_mask] = -9999.0

    # Save to outputs
    out_file = os.path.join(OUT_DIR, f"flood_prob_{rain}mm_supercharged.tif")
    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=MASTER_SHAPE[0],
        width=MASTER_SHAPE[1],
        count=1,
        dtype=np.float32,
        crs=master_crs,
        transform=master_transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    print(f"[OK] Generated Supercharged Scenario: {rain}mm -> {out_file}")

print("\n[SUCCESS] System Intelligence Overhaul Complete!")
print("New Hybrid-AI TIFs are ready for the Streamlit dashboard.")
