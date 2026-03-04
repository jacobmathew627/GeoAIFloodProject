import os
import asf_search as asf
import rasterio
from datetime import datetime

# CONFIGURATION
# Coordinates for Ernakulam (approx bounding box)
BBOX = [76.1, 9.8, 76.6, 10.3] # [min_lon, min_lat, max_lon, max_lat]
SAR_DIR = "sentinel_data"

def search_sentinel_1(start_date, end_date):
    """
    Search for Sentinel-1 IW GRD products over Ernakulam
    """
    print(f"Searching for Sentinel-1 data from {start_date} to {end_date}...")
    
    results = asf.geo_search(
        platform=asf.PLATFORM.SENTINEL1,
        intersectsWith=f'POLYGON(({BBOX[0]} {BBOX[1]}, {BBOX[2]} {BBOX[1]}, {BBOX[2]} {BBOX[3]}, {BBOX[0]} {BBOX[3]}, {BBOX[0]} {BBOX[1]}))',
        start=start_date,
        end=end_date,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        beamMode=asf.BEAMMODE.IW
    )
    
    print(f"Found {len(results)} scenes.")
    for i, scene in enumerate(results):
        print(f"[{i}] {scene.properties['fileID']} | Date: {scene.properties['startTime']}")
    
    return results

def download_scenes(results, username, password):
    """
    Download scenes using NASA Earthdata credentials
    """
    if not os.path.exists(SAR_DIR):
        os.makedirs(SAR_DIR)
        
    print(f"Downloading {len(results)} scenes to {SAR_DIR}...")
    session = asf.ASFSession().auth_with_creds(username, password)
    results.download(path=SAR_DIR, session=session)
    print("Download complete.")

if __name__ == "__main__":
    # Example search for 2018 Flood Period
    # results = search_sentinel_1("2018-08-10", "2018-08-25")
    
    # NOTE: To download, you need an Earthdata Login account.
    # https://urs.earthdata.nasa.gov/
    
    print("Sentinel-1 Acquisition Utility")
    print("-" * 30)
    print("1. Search for scenes")
    print("2. Download scenes (Requires Earthdata Login)")
    print("\nPlease choose an action or configure parameters in the script.")
