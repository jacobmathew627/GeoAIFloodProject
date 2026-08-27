import ee
import os

# No default project ID. The one previously hardcoded here
# (empyrean-backup-387418) no longer exists, so every call failed with an
# opaque permission error that read like an auth problem rather than a dead
# project. Set EARTHENGINE_PROJECT (or EE_PROJECT) to a Cloud project that has
# the Earth Engine API enabled and that your account can actually access.
PROJECT_ENV_VARS = ("EARTHENGINE_PROJECT", "EE_PROJECT")


def gee_project():
    """Earth Engine Cloud project from the environment, or None."""
    for var in PROJECT_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


def initialize_gee():
    project = gee_project()
    if not project:
        print(
            "No Earth Engine project set. Export one of "
            f"{' or '.join(PROJECT_ENV_VARS)} first, e.g.\n"
            "    set EARTHENGINE_PROJECT=my-ee-project"
        )
        return False
    try:
        ee.Initialize(project=project)
        print(f"Google Earth Engine initialized with project: {project}")
    except Exception as e:
        print(f"Error initializing GEE with project {project!r}: {e}")
        print("Check the project exists, has the Earth Engine API enabled, and")
        print("that your account can access it. Then run 'python src/auth_gee.py'.")
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
    s1_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    # Simple median composite (Speckle filtering via aggregation)
    s1_img = s1_col.median().clip(roi)

    print("Extracting Sentinel-2 Optical Baseline (Pre-Flood, Dry Season)...")
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2018-01-01", "2018-05-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    s2_img = s2_col.median().clip(roi)

    # Calculate Indices
    ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndvi = s2_img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Export VV/VH and NDVI/NDWI separately
    import geemap

    layer_map = {
        "SAR_VV_aligned.tif": s1_img.select("VV"),
        "SAR_VH_aligned.tif": s1_img.select("VH"),
        "NDWI_aligned.tif": ndwi,
        "NDVI_aligned.tif": ndvi,
    }

    for filename, image in layer_map.items():
        out_path = os.path.join(output_dir, filename)
        print(f"  Downloading {filename} to {out_path}...")
        try:
            geemap.ee_export_image(
                image, filename=out_path, scale=30, region=roi, file_per_band=False
            )
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")

    return True


if __name__ == "__main__":
    ERNAKULAM_BBOX = [76.16, 9.85, 76.45, 10.15]
    if initialize_gee():
        extract_gee_data(ERNAKULAM_BBOX, "2018-08-01", "2018-08-31")
