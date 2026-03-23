import os
import rasterio
import numpy as np

folder_path = r"C:\Users\Asus\Documents\GeoAI_Flood_Project\GeoAI_Data"
GT_MASTER = os.path.join(folder_path, "Ground_Truth_Fixed.tif")
print(f"============================================================")
print(f"       GEOAI FOLDER VALIDATION REPORT")
print(f"============================================================")

if not os.path.exists(folder_path):
    print(f"CRITICAL ERROR: Folder {folder_path} doesn't exist.")
    exit(1)
if not os.path.exists(GT_MASTER):
    print(f"CRITICAL ERROR: Master file {GT_MASTER} missing.")
    exit(1)

with rasterio.open(GT_MASTER) as src:
    master_meta = src.meta
    master_shape = src.shape
    master_crs = src.crs

print(f"MASTER GROUND TRUTH: {GT_MASTER}")
print(f"  -> Shape: {master_shape[0]} x {master_shape[1]}")
print(f"  -> CRS: {master_crs}")

tifs = [f for f in os.listdir(folder_path) if f.endswith(".tif")]

print(f"\nScanning {len(tifs)} TIF Files...\n")

for tif in tifs:
    filepath = os.path.join(folder_path, tif)
    try:
        with rasterio.open(filepath) as src:
            shape = src.shape
            crs = src.crs
            nodata = src.nodata
            dtype = src.dtypes[0]
            
            # Sub-sample array to prevent out of memory for large info
            arr = src.read(1, window=rasterio.windows.Window(0, 0, min(1000, shape[1]), min(1000, shape[0])))
            arr = arr.astype(np.float32)
            
            arr_valid = arr[arr != nodata] if nodata is not None else arr
            
            try:
                vmin = np.nanmin(arr_valid) if arr_valid.size > 0 else "NaN"
                vmax = np.nanmax(arr_valid) if arr_valid.size > 0 else "NaN"
            except:
                vmin, vmax = "Err", "Err"
                
            nan_count = np.isnan(arr).sum()

            
            status = "[PASS]" if shape == master_shape else "[FAIL] (MISMATCH)"
            crs_status = "[PASS]" if str(crs) == str(master_crs) else "[WARN] (CRS MISMATCH)"
            
            print(f"- {tif}:")
            print(f"    Shape: {shape} {status}")
            print(f"    CRS:   {crs} {crs_status}")
            print(f"    Type:  {dtype} (NoData: {nodata})")
            print(f"    Min/Max: {vmin} / {vmax}")
            if nan_count > 0:
                print(f"    WARNING: Found {nan_count} literal NaN values (in first 1000px chunk)")
            
            if "Ground_Truth" in tif:
                uniques = np.unique(arr_valid)
                print(f"    [GROUND TRUTH CHECK] Unique values: {uniques}")
                if len(uniques) == 2 and 0 in uniques and 1 in uniques:
                    flood_ratio = (arr_valid == 1).sum() / arr_valid.size
                    print(f"    [GROUND TRUTH CHECK] Flood ratio: {flood_ratio * 100:.2f}%")
                else:
                    print(f"    [GROUND TRUTH CHECK] CRITICAL: Not purely binary (0/1)!")
    except Exception as e:
        print(f"  -> {tif} ERROR: {str(e)}")
print("\nValidation complete.")
