import ee
import os
import rasterio
import numpy as np

# NOTE: this script takes a *median composite* over a date range, which is the
# wrong reduction for flood mapping -- a median over 10-25 Aug 2018 averages
# flooded and unflooded scenes together and washes the flood out. Use
# src/acquire_flood_event.py instead, which does pre/post change detection on
# individual acquisitions. This module is kept only because it documents how
# the earliest SAR extraction was done.
#
# `geemap` is imported lazily below rather than at module scope. It is not a
# project dependency, and importing it here made the whole module unimportable
# with a bare ModuleNotFoundError -- which looked like a broken script rather
# than a missing optional package.

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
            import geemap  # optional dependency; see the note at the top

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
    # No default project ID. The one previously hardcoded here
    # (empyrean-backup-387418) no longer exists, and a dead project fails with
    # a permission error that reads like an auth problem, sending you round the
    # authentication loop instead of at the real cause.
    project = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("EE_PROJECT")
    if not project:
        print(
            "No Earth Engine project set. Export EARTHENGINE_PROJECT (or "
            "EE_PROJECT) first, e.g.\n    set EARTHENGINE_PROJECT=my-ee-project"
        )
        return
    try:
        print(f"Initializing Google Earth Engine with project {project}...")
        ee.Initialize(project=project)
    except Exception as e:
        print(f"Initialization error with project {project!r}: {e}")
        print("\nCheck the project exists, has the Earth Engine API enabled, and")
        print("that the authenticated account can access it.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Flood Peak (August 2018)
    extract_sar_sigma0('2018-08-10', '2018-08-25', 'flood_2018')
    
    # Baseline (Dry Season - March 2018)
    extract_sar_sigma0('2018-03-01', '2018-03-31', 'baseline_2018')

if __name__ == "__main__":
    authenticate_and_run()
