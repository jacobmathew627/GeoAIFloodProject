import os
import gc
import rasterio
import numpy as np
import tensorflow as tf
from rasterio.warp import reproject, Resampling
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, jaccard_score

folder_path = r"C:\Users\Asus\Documents\GeoAI_Flood_Project\GeoAI_Data"
model_path = os.path.join(folder_path, "Ernakulam_Flood_UNet.h5")
GT_MASTER = os.path.join(folder_path, "Ground_Truth_Fixed.tif") # (17946, 11849)

# We know from health_check.ipynb that the original training target was exactly (5690, 7375).
# Some files like LULC are (5690, 7374). We will force alignment to (5690, 7375) to test the model.
TARGET_SHAPE = (5690, 7375)

feature_files = [
    'Ernakulam_Filled_DEM.tif', 'Ernakulam_Slope.tif', 'Ernakulam_Distance_Final.tif',
    'Ernakulam_LULC_2018.tif', 'Distance_to_Builtup_Final.tif', 'NDVI_Aligned.tif',
    'NDWI_Aligned.tif', 'Ernakulam_HAND.tif', 'Ernakulam_TWI.tif',
    'Ernakulam_TPI.tif', 'Ernakulam_SPI.tif', 'Ernakulam_Flow_Accumulation.tif'
]

print("=======================================================")
print("             H5 MODEL EVALUATION MODULE")
print("=======================================================")

if not os.path.exists(model_path):
    print("CRITICAL ERROR: Model file not found.")
    exit(1)

# 1. Load the model
try:
    print("Loading U-Net model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully! Expected input shape: {model.input_shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 2. Get target transform and CRS from LULC (which is very close to target shape)
lulc_path = os.path.join(folder_path, 'Ernakulam_LULC_2018.tif')
target_transform = None
target_crs = None
with rasterio.open(lulc_path) as src:
    target_crs = src.crs
    target_transform = src.transform

X_data = []
print("\n--- Constructing Data Cube with Proper Reprojection ---")

for file in feature_files:
    file_path = os.path.join(folder_path, file)
    if not os.path.exists(file_path):
        print(f"  [MISSING] {file}")
        # Insert blank zeros to keep channels aligned
        X_data.append(np.zeros(TARGET_SHAPE, dtype=np.float32))
        continue
        
    try:
        with rasterio.open(file_path) as src:
            source = src.read(1)
            nodata = src.nodata
            
            dest = np.zeros(TARGET_SHAPE, dtype=np.float32)
            
            # Reproject exactly to the (5690, 7375) grid
            reproject(
                source=source,
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
            # Mask nodata with zeros
            if nodata is not None:
                dest[dest == nodata] = 0.0
                
            # Naive Normalization (min-max) with float division protection
            valid_mask = dest != 0.0
            if valid_mask.any():
                c_min = np.nanmin(dest[valid_mask])
                c_max = np.nanmax(dest[valid_mask])
                denom = (c_max - c_min)
                if denom > 1e-6:
                    dest = (dest - c_min) / denom
            
            dest = np.nan_to_num(dest, nan=0.0)
            X_data.append(dest)
            print(f"  [OK] Reprojected and Normalized: {file}")
    except Exception as e:
        print(f"  [ERROR] Failed on {file}: {e}")
        X_data.append(np.zeros(TARGET_SHAPE, dtype=np.float32))

# Free memory
gc.collect()

X_dataset = np.stack(X_data, axis=-1)
print(f"\nFinal X_dataset format: {X_dataset.shape}")

print("\n--- Model Evaluation (Random Patches) ---")
# To avoid OOM and since we don't have perfect Ground Truth mapped to this grid yet,
# we will just do a forward pass test to ensure the model doesn't output NaNs.

try:
    # Test patch exactly matching model input (e.g., 256x256x12)
    patch_size = model.input_shape[1] if model.input_shape[1] is not None else 256
    
    # Grab a patch from the center
    h, w, _ = X_dataset.shape
    center_y, center_x = h // 2, w // 2
    
    test_patch = X_dataset[center_y:center_y+patch_size, center_x:center_x+patch_size, :]
    if test_patch.shape == (patch_size, patch_size, 12):
        test_patch = np.expand_dims(test_patch, axis=0) # Batch size 1
        
        prediction = model.predict(test_patch)
        print(f"Prediction successful! Output shape: {prediction.shape}")
        
        pred_min = float(np.min(prediction))
        pred_max = float(np.max(prediction))
        print(f"Prediction Values - Min: {pred_min:.4f}, Max: {pred_max:.4f}")
        
        if np.isnan(prediction).any() or (pred_min == pred_max == 0.0) or (pred_min == pred_max == 1.0):
            print("\n❌ CRITICAL FAILURE: The .h5 model is brain-dead (outputs all NaNs, zeros, or ones). It must be retrained.")
        else:
            print("\n✅ The .h5 model produces valid localized variance. It might be usable.")
    else:
         print(f"Could not extract a valid {patch_size}x{patch_size} patch from center.")
         
except Exception as e:
    print(f"\n❌ Model inference crashed: {e}")
    print("The model or input formatting is corrupt and MUST be retrained.")
