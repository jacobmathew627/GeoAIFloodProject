import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# Configuration
INPUT_DIR = "."
OUTPUT_DIR = "processed"
REFERENCE_FILE = "DEM_snap.tif"

FILES_TO_PROCESS = {
    "DEM": "DEM_snap.tif",
    "Slope": "Slope_Degrees_snap.tif",
    "Flow": "FlowAcc_Log_snap.tif",
    "LULC": "LULC_snap.tif",
    "UWI_100": "UWI_100mm_snap.tif",
    "UWI_150": "UWI_150mm_snap.tif",
    "UWI_200": "UWI_200mm_snap.tif",
    "SAR_VV": "SAR_flood_2018_VV_aligned.tif",
    "SAR_VH": "SAR_flood_2018_VH_aligned.tif",
    "Label": "flood_mask_2018_ekm.tif"
}

def align_rasters():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read Reference Metadata
    ref_path = os.path.join(INPUT_DIR, REFERENCE_FILE)
    with rasterio.open(ref_path) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        dst_width = src.width
        dst_height = src.height
        dst_profile = src.profile.copy()

    dst_profile.update({
        'driver': 'GTiff',
        'transform': dst_transform,
        'crs': dst_crs,
        'width': dst_width,
        'height': dst_height,
        'count': 1,
        'nodata': -9999 if dst_profile.get('nodata') is None else dst_profile.get('nodata')
    })
    
    print(f"Reference Grid: {dst_width}x{dst_height}, CRS: {dst_crs}")

    for key, filename in FILES_TO_PROCESS.items():
        src_path = os.path.join(INPUT_DIR, filename)
        dst_path = os.path.join(OUTPUT_DIR, f"{key}_aligned.tif")
        
        if not os.path.exists(src_path):
            print(f"Warning: {filename} not found. Skipping.")
            continue

        print(f"Processing {key} ({filename})...")
        
        with rasterio.open(src_path) as src:
            # Determine resampling method
            if key == "LULC" or key == "Label":
                resampling = Resampling.nearest
            else:
                resampling = Resampling.bilinear

            # Check if nodata is defined
            src_nodata = src.nodata
            if src_nodata is None:
                # Need to handle nodata carefully. 
                # For continuous vars, assume -9999 or 0 if strictly positive?
                # Actually, rasterio reproject defaults to 0 if not specified for output, 
                # but we set nodata in profile.
                pass

            with rasterio.open(dst_path, 'w', **dst_profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=dst_profile['nodata']
                )

    print("Raster alignment complete.")

if __name__ == "__main__":
    align_rasters()
