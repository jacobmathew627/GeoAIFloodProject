import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, uniform_filter, zoom
import matplotlib.pyplot as plt
import gc

# Config
PROCESSED_DIR = "processed"
OUTPUT_DIR = "processed"
CELL_SIZE = 30.0

def lee_filter(img, size=5):
    """
    SNAP-style Lee Filter for SAR Speckle Reduction.
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-6)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def calculate_geomorphic_factors():
    print("Executing 'Real GeoAI' High-Fidelity Processing...")
    
    # Files to load
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "LULC": "LULC_aligned.tif",
        "SAR_VV": "SAR_VV_aligned.tif",
        "SAR_VH": "SAR_VH_aligned.tif"
    }
    
    data = {}
    profile = None
    nodata = -9999 # Default nodata value
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, uniform_filter
import gc

# Config
PROCESSED_DIR = "processed"
OUTPUT_DIR = "processed"
CELL_SIZE = 30.0
BLOCK_SIZE = 1024 # 1024x1024 blocks for memory safety

def lee_filter_block(img, size=5):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-6)
    return img_mean + img_weights * (img - img_mean)

def calculate_geomorphic_factors():
    print("Executing 'Real GeoAI' High-Fidelity Processing (Windowed Mode)...")
    
    dem_path = os.path.join(PROCESSED_DIR, "DEM_aligned.tif")
    slope_path = os.path.join(PROCESSED_DIR, "Slope_aligned.tif")
    flow_path = os.path.join(PROCESSED_DIR, "Flow_aligned.tif")
    lulc_path = os.path.join(PROCESSED_DIR, "LULC_aligned.tif")
    
    if not all(os.path.exists(p) for p in [dem_path, slope_path, flow_path, lulc_path]):
        print("Error: Required input rasters missing.")
        return

    with rasterio.open(dem_path) as dem_src:
        profile = dem_src.profile.copy()
        nodata = dem_src.nodata
        width = dem_src.width
        height = dem_src.height
        
        # 1. TWI, SPI, STI, TPI Generation (Windowed)
        twi_path = os.path.join(OUTPUT_DIR, "TWI_aligned.tif")
        spi_path = os.path.join(OUTPUT_DIR, "SPI_aligned.tif")
        sti_path = os.path.join(OUTPUT_DIR, "STI_aligned.tif")
        tpi_path = os.path.join(OUTPUT_DIR, "TPI_aligned.tif")
        profile.update(dtype=rasterio.float32, nodata=nodata)
        
        print(f"  -> Generating Advanced Geomorphology (TWI, SPI, STI, TPI) ({width}x{height})...")
        with rasterio.open(slope_path) as slope_src, \
             rasterio.open(flow_path) as flow_src, \
             rasterio.open(twi_path, 'w', **profile) as twi_dst, \
             rasterio.open(spi_path, 'w', **profile) as spi_dst, \
             rasterio.open(sti_path, 'w', **profile) as sti_dst, \
             rasterio.open(tpi_path, 'w', **profile) as tpi_dst:
            
            for y in range(0, height, BLOCK_SIZE):
                for x in range(0, width, BLOCK_SIZE):
                    win = Window(x, y, min(BLOCK_SIZE, width - x), min(BLOCK_SIZE, height - y))
                    
                    dem = dem_src.read(1, window=win).astype(np.float32)
                    slope = slope_src.read(1, window=win).astype(np.float32)
                    flow = flow_src.read(1, window=win).astype(np.float32)
                    
                    mask = (dem != nodata) & (slope != nodata) & (flow != nodata)
                    
                    twi_block = np.full(dem.shape, nodata, dtype=np.float32)
                    spi_block = np.full(dem.shape, nodata, dtype=np.float32)
                    sti_block = np.full(dem.shape, nodata, dtype=np.float32)
                    tpi_block = np.full(dem.shape, nodata, dtype=np.float32)
                    
                    if np.any(mask):
                        s_rad = np.deg2rad(np.maximum(slope[mask], 0.1))
                        sca_val = (np.exp(flow[mask]) * CELL_SIZE) if np.mean(flow[mask]) < 15 else (flow[mask] * CELL_SIZE)
                        
                        twi_block[mask] = np.log(sca_val / (np.tan(s_rad) + 1e-6))
                        np.clip(twi_block[mask], 0, 20, out=twi_block[mask])
                        
                        spi_val = sca_val * np.tan(s_rad)
                        spi_block[mask] = np.log1p(spi_val) 
                        
                        sti_val = ((sca_val / 22.13)**0.6) * ((np.sin(s_rad) / 0.0896)**1.3)
                        sti_block[mask] = np.log1p(sti_val)
                        
                        # TPI: Difference from mean of neighborhood
                        tpi_block = dem - uniform_filter(dem, size=11)
                        tpi_block[~mask] = nodata
                    
                    twi_dst.write(twi_block, 1, window=win)
                    spi_dst.write(spi_block, 1, window=win)
                    sti_dst.write(sti_block, 1, window=win)
                    tpi_dst.write(tpi_block, 1, window=win)
        
        # 2. Augmented Water & Distance (Coastal-Aware)
        print("  -> Augmented Water & Urban Distance (Coastal+Built-up)...")
        with rasterio.open(lulc_path) as lulc_src, \
             rasterio.open(flow_path) as flow_src:
            lulc = lulc_src.read(1)
            dem = dem_src.read(1)
            flow = flow_src.read(1)
            
            # Distance to Water (Coastal)
            lulc_water = np.isin(lulc, [80, 90, 95])
            sea_mask = (dem == nodata) | (dem < 0.5)
            water_mask = lulc_water | sea_mask
            print("  -> Distance transform (Water)...")
            dist_water = distance_transform_edt(~water_mask).astype(np.float32)
            dist_water *= CELL_SIZE
            
            # Distance to Urban (LULC 50)
            print("  -> Distance transform (Urban)...")
            urban_mask = (lulc == 50)
            dist_urban = distance_transform_edt(~urban_mask).astype(np.float32)
            dist_urban *= CELL_SIZE
            
            # 2.5 HAND Approximation
            print("  -> Computing HAND Approximation (Spatially optimized)...")
            flow_val = np.exp(flow) if np.mean(flow) < 15 else flow
            drainage_mask = flow_val > (1000 / CELL_SIZE)
            del flow, flow_val; gc.collect()
            
            ds_drainage = zoom(drainage_mask, 0.25, order=0)
            if np.any(ds_drainage):
                _, indices = distance_transform_edt(~ds_drainage, return_indices=True)
                ds_dem = zoom(dem, 0.25, order=0)
                ds_hand = ds_dem - ds_dem[indices[0], indices[1]]
                hand_up = zoom(ds_hand, 4.0, order=1)
                # Robust align: Pre-allocate and slice
                hand = np.full(dem.shape, nodata, dtype=np.float32)
                h_fit, w_fit = min(dem.shape[0], hand_up.shape[0]), min(dem.shape[1], hand_up.shape[1])
                hand[:h_fit, :w_fit] = hand_up[:h_fit, :w_fit]
                del ds_dem, ds_drainage, indices, ds_hand, hand_up; gc.collect()
            else:
                hand = np.full(dem.shape, nodata, dtype=np.float32)

            # Apply Master Mask and Save
            print("  -> Masking and saving distancing layers...")
            with rasterio.open(os.path.join(OUTPUT_DIR, "DistWater_aligned.tif"), 'w', **profile) as dst:
                dist_water[dem == nodata] = nodata
                dst.write(dist_water, 1)
            
            with rasterio.open(os.path.join(OUTPUT_DIR, "DistUrban_aligned.tif"), 'w', **profile) as dst:
                dist_urban[dem == nodata] = nodata
                dst.write(dist_urban, 1)
                
            with rasterio.open(os.path.join(OUTPUT_DIR, "HAND_aligned.tif"), 'w', **profile) as dst:
                # NoData already set in hand initialization, but to be sure:
                hand[dem == nodata] = nodata
                dst.write(hand, 1)
            
            del lulc, dem, water_mask, urban_mask, dist_water, dist_urban, hand; gc.collect()

    # 3. SAR Speckle Filtering (Windowed with Padding)
    print("  -> Applying Lee Filter to SAR (Windowed)...")
    for key in ["SAR_VV", "SAR_VH"]:
        path = os.path.join(PROCESSED_DIR, f"{key}_aligned.tif")
        if not os.path.exists(path): continue
        
        with rasterio.open(path) as sar_src:
            p = sar_src.profile.copy()
            raw = sar_src.read(1).astype(np.float32)
            filtered = lee_filter_block(raw)
            with rasterio.open(os.path.join(OUTPUT_DIR, f"{key}_aligned.tif"), 'w', **p) as sar_dst:
                sar_dst.write(filtered, 1)
            del raw, filtered; gc.collect()

    print(f"Real GeoAI Processing Complete. Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    calculate_geomorphic_factors()
