import os
import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

# Input and output paths
PROCESSED_DIR = "processed"
FLOW_ACC_PATH = os.path.join(PROCESSED_DIR, "Flow_aligned.tif")
LULC_PATH = os.path.join(PROCESSED_DIR, "LULC_aligned.tif")

OUT_DRAINAGE = os.path.join(PROCESSED_DIR, "Drainage_aligned.tif")
OUT_DIST_DRAINAGE = os.path.join(PROCESSED_DIR, "DistDrainage_aligned.tif")
OUT_DRAINAGE_DENSITY = os.path.join(PROCESSED_DIR, "DrainageDensity_aligned.tif")
OUT_BUILTUP_DENSITY = os.path.join(PROCESSED_DIR, "BuiltupDensity_aligned.tif")

def compute_proxies():
    print("Loading Flow Accumulation and LULC...")
    
    with rasterio.open(FLOW_ACC_PATH) as src_flow:
        flow_acc = src_flow.read(1)
        profile = src_flow.profile.copy()
        nodata_flow = src_flow.nodata if src_flow.nodata is not None else -9999
        resolution = src_flow.res[0] # assuming square pixels, ~30m
        
    with rasterio.open(LULC_PATH) as src_lulc:
        lulc = src_lulc.read(1)
        nodata_lulc = src_lulc.nodata if src_lulc.nodata is not None else 0

    print("Step 1/4: Extracting drainage networks using thresholding...")
    # Flow logic is log-scaled in the dataset (FlowAcc_Log_snap). Wait, checking `inspect_hydro.py`:
    # Usually log(1+flow), values ~ 0 to 15. A threshold of 5-7 is reasonable for logging, or check 90th percentile
    flow_valid = flow_acc[flow_acc != nodata_flow]
    threshold = np.percentile(flow_valid, 95)
    print(f"  Using threshold = {threshold:.2f} (95th percentile) for drainage extraction")
    
    drainage = np.where(flow_acc >= threshold, 1.0, 0.0)
    drainage[flow_acc == nodata_flow] = nodata_flow
    
    # Save Drainage raster
    profile.update(dtype=rasterio.float32, nodata=nodata_flow)
    with rasterio.open(OUT_DRAINAGE, 'w', **profile) as dst:
        dst.write(drainage.astype(np.float32), 1)

    print("Step 2/4: Computing Distance to Drainage...")
    # EDT needs background=1 for areas to measure distance TO 0
    # True mask: drainage == 1 -> distance = 0
    # Other areas -> distance > 0
    # To ignore nodata during edt, we usually set nodata pixels to distance 0 temporarily, 
    # but that pulls the edt toward nodata edges. Better to compute over everything and mask later.
    is_not_drainage = (drainage != 1.0)
    # EDT computes distance to nearest zero. So 1 means background, 0 means object.
    distances = distance_transform_edt(is_not_drainage)
    distances_meters = distances * resolution
    distances_meters[flow_acc == nodata_flow] = nodata_flow
    
    with rasterio.open(OUT_DIST_DRAINAGE, 'w', **profile) as dst:
        dst.write(distances_meters.astype(np.float32), 1)

    print("Step 3/4: Computing Drainage Density (Window = 33x33)...")
    # Uniform filter computes moving average. * window_size^2 = sum
    # Drainage density = average drainage pixels in a window
    # uniform_filter handles boundaries using reflection.
    valid_mask = (flow_acc != nodata_flow)
    # create masked-aware average:
    drainage_0 = np.where(valid_mask, drainage, 0.0)
    dens_drainage = uniform_filter(drainage_0, size=33)
    dens_drainage[~valid_mask] = nodata_flow
    
    with rasterio.open(OUT_DRAINAGE_DENSITY, 'w', **profile) as dst:
        dst.write(dens_drainage.astype(np.float32), 1)

    print("Step 4/4: Computing Built-up Density (Urban Drainage Stress Proxy)...")
    # Built-up class in ESA WorldCover is typically 50.
    valid_mask_lulc = (lulc != nodata_lulc)
    builtup = np.where(lulc == 50, 1.0, 0.0)
    dens_builtup = uniform_filter(builtup, size=33)
    dens_builtup[~valid_mask_lulc] = nodata_flow
    
    with rasterio.open(OUT_BUILTUP_DENSITY, 'w', **profile) as dst:
        dst.write(dens_builtup.astype(np.float32), 1)

    print("Drainage proxies generated and saved to /processed.")

if __name__ == "__main__":
    compute_proxies()
