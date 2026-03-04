import rasterio
import numpy as np

def load_raster(path):
    """Loads a raster and returns the data (H, W) and profile."""
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata
    return data, profile, nodata

def normalize(data, min_val=None, max_val=None):
    """Min-Max normalization to [0, 1]."""
    if min_val is None:
        min_val = np.nanmin(data)
    if max_val is None:
        max_val = np.nanmax(data)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.float32)
        
    return (data - min_val) / (max_val - min_val)

def extract_patches(stack, label, patch_size=32, stride=32, valid_mask=None):
    """
    Extracts patches from a stack of features and labels.
    
    Args:
        stack: (H, W, C) array of features.
        label: (H, W) array of labels.
        patch_size: int, size of the patch (square).
        stride: int, stride for sliding window.
        valid_mask: (H, W) boolean mask of valid pixels.
        
    Returns:
        patches: (N, patch_size, patch_size, C)
        labels: (N, patch_size, patch_size, 1)
    """
    H, W, C = stack.shape
    patches = []
    labels = []
    
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            if valid_mask is not None:
                # Check if patch is fully valid
                patch_mask = valid_mask[r:r+patch_size, c:c+patch_size]
                if not np.all(patch_mask):
                    continue
            
            patch = stack[r:r+patch_size, c:c+patch_size, :]
            lbl = label[r:r+patch_size, c:c+patch_size]
            
            patches.append(patch)
            labels.append(lbl)
            
    return np.array(patches), np.expand_dims(np.array(labels), axis=-1)
