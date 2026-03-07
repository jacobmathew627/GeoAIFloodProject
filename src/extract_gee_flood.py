import ee
import os
import rasterio
from rasterio.transform import from_origin

def initialize_gee():
    try:
        ee.Initialize(project='empyrean-backup-387418')
        print("Google Earth Engine initialized successfully with project: empyrean-backup-387418")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        print("Please run 'python src/auth_gee.py' first.")
        return False
    return True

def extract_gee_data(bbox, start_date, end_date, output_dir="processed"):
    """
    Automated extraction of Sentinel-1 and Sentinel-2 data
    bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    roi = ee.Geometry.Rectangle(bbox)
    
    print(f"Extracting Sentinel-1 SAR for {start_date} to {end_date}...")
    # Sentinel-1 GRD IW
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    
    # Simple median composite (Speckle filtering via aggregation)
    s1_img = s1_col.median().clip(roi)
    
    print("Extracting Sentinel-2 Optical Baseline (Pre-Flood, Dry Season)...")
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2018-01-01', '2018-05-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
    s2_img = s2_col.median().clip(roi)
    
    # Calculate Indices
    ndwi = s2_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Export VV/VH and NDVI/NDWI separately
    import geemap
    
    layer_map = {
        "SAR_VV_aligned.tif": s1_img.select('VV'),
        "SAR_VH_aligned.tif": s1_img.select('VH'),
        "NDWI_aligned.tif": ndwi,
        "NDVI_aligned.tif": ndvi
    }
    
    for filename, image in layer_map.items():
        out_path = os.path.join(output_dir, filename)
        print(f"  Downloading {filename} to {out_path}...")
        try:
            geemap.ee_export_image(
                image,
                filename=out_path,
                scale=30,
                region=roi,
                file_per_band=False
            )
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
    
    return True

if __name__ == "__main__":
    ERNAKULAM_BBOX = [76.16, 9.85, 76.45, 10.15]
    if initialize_gee():
        extract_gee_data(ERNAKULAM_BBOX, "2018-08-01", "2018-08-31")
