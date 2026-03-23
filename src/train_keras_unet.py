import os
import gc
import rasterio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from rasterio.warp import reproject, Resampling

# ==========================================
# CONFIGURATION
# ==========================================
FOLDER_PATH = r"C:\Users\Asus\Documents\GeoAI_Flood_Project\GeoAI_New"
MASTER_SHAPE = (5690, 7375)  # The known clean shape
PATCH_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
MODEL_SAVE_PATH = os.path.join(r"C:\Users\Asus\Documents\GeoAI_Flood_Project\models", "Ernakulam_Flood_UNet_Ultra.h5")

FEATURE_FILES = [
    'Ernakulam_Clipped_DEM.tif',        # ch 1  - elevation
    'Ernakulam_Slope.tif',              # ch 2  - slope
    'Ernakulam_River_Distance.tif',     # ch 3  - river proximity
    'Ernakulam_LULC_2018.tif',          # ch 4  - land use
    'Distance_to_Builtup_Final.tif',    # ch 5  - urban proximity
    'NDVI_Aligned.tif',                 # ch 6  - vegetation index
    'NDWI_Aligned.tif',                 # ch 7  - water index
    'Ernakulam_HAND.tif',               # ch 8  - height above drainage
    'Ernakulam_TWI.tif',                # ch 9  - topographic wetness
    'Ernakulam_TPI.tif',                # ch 10 - topographic position
    'Ernakulam_SPI.tif',                # ch 11 - stream power
    'Ernakulam_Flow_Accumulation.tif',  # ch 12 - flow accumulation
    'Urban_Mask.tif',                   # ch 13 - binary urban footprint (NEW)
]
N_CHANNELS = len(FEATURE_FILES)  # 13

# Pre-flight: verify all files exist before loading GBs of data
print("--- Pre-flight File Check ---")
for f in FEATURE_FILES:
    p = os.path.join(FOLDER_PATH, f)
    if not os.path.exists(p):
        raise FileNotFoundError(f"MISSING: {p}")
    print(f"  [OK] {f}")
gt_path = os.path.join(FOLDER_PATH, "Ground_Truth_Fixed.tif")
if not os.path.exists(gt_path):
    raise FileNotFoundError(f"MISSING Ground Truth: {gt_path}")
print(f"  [OK] Ground_Truth_Fixed.tif")
print(f"All {N_CHANNELS} feature channels verified.")


# We will use the LULC file as the master projection template
TEMPLATE_FILE = os.path.join(FOLDER_PATH, 'Ernakulam_LULC_2018.tif')

# ==========================================
# 1. LOAD AND WARP DATA CUBE
# ==========================================
print("--- Scanning and Warping Data Cube ---")
with rasterio.open(TEMPLATE_FILE) as src:
    target_crs = src.crs
    target_transform = src.transform

data_cube = []
for file in FEATURE_FILES:
    path = os.path.join(FOLDER_PATH, file)
    print(f"Loading {file}...")
    with rasterio.open(path) as src:
        source_array = src.read(1)
        dest_array = np.zeros(MASTER_SHAPE, dtype=np.float32)
        
        # Proper geographic alignment
        reproject(
            source=source_array,
            destination=dest_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        data_cube.append(dest_array)

X_dataset = np.stack(data_cube, axis=-1)
print(f"X_dataset shape: {X_dataset.shape}")

# Load Ground Truth
gt_path = os.path.join(FOLDER_PATH, "Ground_Truth_Fixed.tif")
print(f"Loading Ground Truth...")
with rasterio.open(gt_path) as src:
    source_array = src.read(1)
    Y_dataset = np.zeros(MASTER_SHAPE, dtype=np.float32)
    reproject(
        source=source_array,
        destination=Y_dataset,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )
    Y_dataset = np.expand_dims(Y_dataset, axis=-1)

print(f"Y_dataset shape: {Y_dataset.shape}")

# ==========================================
# 2. SLICE INTO PATCHES
# ==========================================
print("--- Slicing Patches ---")
X_patches = []
Y_patches = []
h, w, c = X_dataset.shape

for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
    for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
        X_patch = X_dataset[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
        Y_patch = Y_dataset[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
        
        # Only keep patches with actual land data (skip empty borders)
        if np.max(X_patch[:, :, 0]) > -9000:
            X_patches.append(X_patch)
            Y_patches.append(Y_patch)

X_train = np.array(X_patches)
Y_train = np.array(Y_patches)
print(f"Extracted {len(X_train)} valid patches of shape {X_train.shape}.")

# ==========================================
# 3. NAN-SAFE NORMALIZATION (Crucial Fix)
# ==========================================
print("--- Normalizing Data (NaN-Safe) ---")
X_train_norm = np.zeros_like(X_train, dtype=np.float32)
for i in range(N_CHANNELS):
    channel_data = X_train[:, :, :, i]
    # Filter out nodata values before finding min/max
    valid_mask = channel_data > -9000
    valid_data = channel_data[valid_mask]
    
    if valid_data.size > 0:
        c_min = np.percentile(valid_data, 1)
        c_max = np.percentile(valid_data, 99)
        denom = c_max - c_min if (c_max - c_min) > 1e-6 else 1e-6
        X_train_norm[:,:,:,i] = np.clip((channel_data - c_min) / denom, 0, 1)
    else:
        print(f"  WARNING: Channel {i} ({FEATURE_FILES[i]}) has no valid data!")

# Scrub any remaining NaNs to zeros
X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=1.0, neginf=0.0)

# Make Y strictly binary
Y_train = (Y_train > 0.5).astype(np.float32)

from sklearn.model_selection import train_test_split
X_t, X_v, Y_t, Y_v = train_test_split(X_train_norm, Y_train, test_size=0.2, random_state=42)

# ==========================================
# 4. SUPERCHARGED ATTENTION U-NET
# ==========================================
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def attention_gate(g, x, filters):
    Wg = layers.Conv2D(filters, 1, padding='same')(g)
    Wx = layers.Conv2D(filters, 1, padding='same')(x)
    out = layers.Activation('relu')(layers.add([Wg, Wx]))
    out = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(out)
    return layers.multiply([x, out])

inputs = layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS))  # 13 channels

# Encoder
c1 = conv_block(inputs, 32)
p1 = layers.MaxPooling2D((2,2))(c1); p1 = layers.Dropout(0.1)(p1)

c2 = conv_block(p1, 64)
p2 = layers.MaxPooling2D((2,2))(c2); p2 = layers.Dropout(0.1)(p2)

c3 = conv_block(p2, 128)
p3 = layers.MaxPooling2D((2,2))(c3); p3 = layers.Dropout(0.2)(p3)

c4 = conv_block(p3, 256)
p4 = layers.MaxPooling2D((2,2))(c4); p4 = layers.Dropout(0.2)(p4)

# Bridge
c5 = conv_block(p4, 512)

# Decoder with Attention
u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
a6 = attention_gate(u6, c4, 256)
u6 = layers.concatenate([u6, a6])
c6 = conv_block(u6, 256)

u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
a7 = attention_gate(u7, c3, 128)
u7 = layers.concatenate([u7, a7])
c7 = conv_block(u7, 128)

u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
a8 = attention_gate(u8, c2, 64)
u8 = layers.concatenate([u8, a8])
c8 = conv_block(u8, 64)

u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
a9 = attention_gate(u9, c1, 32)
u9 = layers.concatenate([u9, a9])
c9 = conv_block(u9, 32)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

print("--- Training Attention U-Net ---")
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
]

history = model.fit(
    X_t, Y_t,
    validation_data=(X_v, Y_v),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print(f"Training Complete! Valid Model Saved to {MODEL_SAVE_PATH}")
