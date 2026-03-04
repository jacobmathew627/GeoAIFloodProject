import os
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Configuration
PROCESSED_DIR = "processed"
PATCH_SIZE = 32
STRIDE = 16  # Overlap for training data augmentation

def load_training_data():
    """
    Loads aligned rasters, extracts patches, and returns training/validation sets.
    Features: DEM, Slope, Flow, LULC, UWI_200
    Label: Flood_Mask
    """
    files = {
        "DEM": "DEM_aligned.tif",
        "Slope": "Slope_aligned.tif",
        "Flow": "Flow_aligned.tif",
        "LULC": "LULC_aligned.tif",
        "UWI": "UWI_200_aligned.tif",
        "Label": "Label_aligned.tif"
    }

    data = {}
    
    # Load all files
    for key, fname in files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{fname} not found in {PROCESSED_DIR}")
            
        with rasterio.open(path) as src:
            arr = src.read(1)
            arr = np.nan_to_num(arr, nan=0.0) 
            data[key] = arr

    # Normalize Features
    dem = data["DEM"]
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
    
    slope = data["Slope"]
    slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
    
    flow = data["Flow"]
    flow = (flow - flow.min()) / (flow.max() - flow.min() + 1e-6)
    
    lulc = data["LULC"]
    lulc = lulc / 255.0 
    
    uwi = data["UWI"]
    uwi = (uwi - uwi.min()) / (uwi.max() - uwi.min() + 1e-6)

    label = data["Label"]
    label = np.where(label > 0, 1.0, 0.0).astype(np.float32)
    
    # Stack Features (C, H, W) for PyTorch Unfold
    # We maintain standard image format (H, W, C) for return, but use (1, C, H, W) for processing
    stack = np.stack([dem, slope, flow, lulc, uwi], axis=0) # (5, H, W)
    
    return stack, label

def extract_balanced_patches(stack, label, patch_size=PATCH_SIZE, stride=STRIDE):
    # stack: (C, H, W)
    # label: (H, W)
    
    print("Preparing tensors for patch extraction...")
    C, H, W = stack.shape
    
    # Convert to tensor
    stack_t = torch.from_numpy(stack).float().unsqueeze(0) # (1, C, H, W)
    label_t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # Unfold
    # Output: (1, C*k*k, L)
    print("Unfolding features...")
    patches_t = F.unfold(stack_t, kernel_size=patch_size, stride=stride)
    # patches_t: (1, 5*32*32, L)
    
    print("Unfolding labels...")
    labels_t = F.unfold(label_t, kernel_size=patch_size, stride=stride)
    # labels_t: (1, 1*32*32, L)

    # Permute to (L, C, k, k)
    # First (L, C*k*k)
    patches_t = patches_t.squeeze(0).permute(1, 0) # (L, 5*32*32)
    labels_t = labels_t.squeeze(0).permute(1, 0)   # (L, 1*32*32)
    
    # Reshape
    L = patches_t.size(0)
    patches_t = patches_t.view(L, C, patch_size, patch_size)
    labels_t = labels_t.view(L, 1, patch_size, patch_size)
    
    print(f"Total potential patches: {L}")
    
    # Filter
    # Need to check if patch is valid (not all zeros)
    # Efficient check: Sum of absolute values?
    # Or just check center?
    # Let's check if sum > 0 for UWI channel (index 4) or sum of all channels?
    # Or variance?
    # Let's use the same logic: check if ANY pixel in patch is non-zero?
    # Actually, "all zeros" check was to avoid empty padded areas.
    
    # Sum of features
    feature_sums = patches_t.sum(dim=(1, 2, 3)) # (L, )
    valid_indices = torch.nonzero(feature_sums > 0.001).squeeze()
    
    print(f"Valid patches (non-empty): {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        raise ValueError("No valid patches found.")
        
    patches_t = patches_t[valid_indices]
    labels_t = labels_t[valid_indices]
    
    # Balancing
    # Check if patch has flood
    # Sum of label
    label_sums = labels_t.sum(dim=(1, 2, 3)) # (L, )
    
    pos_indices = torch.nonzero(label_sums > 0).squeeze()
    neg_indices = torch.nonzero(label_sums == 0).squeeze()
    
    n_pos = len(pos_indices) if pos_indices.numel() > 0 else 0
    n_neg = len(neg_indices) if neg_indices.numel() > 0 else 0
    
    print(f"Positive: {n_pos}, Negative: {n_neg}")
    
    if n_pos == 0:
        raise ValueError("No positive patches found.")

    # Downsample negative
    n_keep_neg = min(n_neg, n_pos * 2)
    if n_neg > 0:
        perm = torch.randperm(n_neg)
        keep_neg_indices = neg_indices[perm[:n_keep_neg]]
        
        # Combine
        if n_pos > 0: # Ensure valid tensor concatenation
            if pos_indices.dim() == 0: pos_indices = pos_indices.unsqueeze(0)
            if keep_neg_indices.dim() == 0: keep_neg_indices = keep_neg_indices.unsqueeze(0)
            
            final_indices = torch.cat([pos_indices, keep_neg_indices])
        else:
            final_indices = keep_neg_indices
    else:
        final_indices = pos_indices

    # Shuffle
    perm = torch.randperm(len(final_indices))
    final_indices = final_indices[perm]
    
    X = patches_t[final_indices].numpy()
    y = labels_t[final_indices].numpy()
    
    # X is (N, C, H, W).
    # Model expects that (PyTorch convention).
    # Previous code expected (N, H, W, C) then transposed in train.py.
    # Let's adjust train.py to expect (N, C, H, W) directly from here.
    # But wait, `extract_balanced_patches` return signature change?
    # train.py did: `X_numpy = np.transpose(X_numpy, (0, 3, 1, 2))`
    # If I return (N, C, H, W), I should remove that transpose in train.py.
    
    return X, y

if __name__ == "__main__":
    s, l = load_training_data()
    X, y = extract_balanced_patches(s, l)
    print("Data shape:", X.shape, y.shape)
