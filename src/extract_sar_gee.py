import ee
import geemap
import os
import rasterio
import numpy as np

# CONFIGURATION
# Using FAO/GAUL for precise Ernakulam district boundary
DISTRICT_NAME = 'Ernakulam'
OUTPUT_DIR = "processed"

def extract_sar_sigma0(start_date, end_date, suffix):
    """
    Extracts Sentinel-1 Sigma0 Backscatter (VV + VH) from GEE
    """
    print(f"\nProcessing SAR data for {start_date} to {end_date}...")

    # Get District Boundary
    roi = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq('ADM2_NAME', DISTRICT_NAME)) \
        .geometry()

    # Sentinel-1 GRD collection
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Median composite
    composite = collection.median().clip(roi)
    
    # Export VV and VH separately to stay under the 50MB direct download limit
    for band in ['VV', 'VH']:
        band_image = composite.select(band)
        out_path = os.path.join(OUTPUT_DIR, f"SAR_{suffix}_{band}_aligned.tif")
        
        print(f"  Exporting {band} to {out_path}...")
        try:
            geemap.ee_export_image(
                band_image,
                filename=out_path,
                scale=30,
                region=roi,
                file_per_band=False
            )
        except Exception as e:
            print(f"  Error exporting {band}: {e}")
    
    return True

def authenticate_and_run():
    """
    Guides the user through GEE authentication and extracts 2018 data
    """
    try:
        print("Initializing Google Earth Engine...")
        # Initializing with the user's project ID
        ee.Initialize(project='empyrean-backup-387418')
    except Exception as e:
        print(f"Initialization error: {e}")
        print("\nNote: If you have authenticated but still see this error, you might need to specify a Project ID.")
        print("Example: ee.Initialize(project='your-project-id')")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Flood Peak (August 2018)
    extract_sar_sigma0('2018-08-10', '2018-08-25', 'flood_2018')
    
    # Baseline (Dry Season - March 2018)
    extract_sar_sigma0('2018-03-01', '2018-03-31', 'baseline_2018')

if __name__ == "__main__":
    authenticate_and_run()
