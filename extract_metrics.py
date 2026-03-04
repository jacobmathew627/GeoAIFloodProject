import torch
import os

def extract_metrics():
    model_dir = "models"
    for model_name in os.listdir(model_dir):
        if not model_name.endswith(".pth"):
            continue
            
        model_path = os.path.join(model_dir, model_name)
        print("\n" + "=" * 50)
        print(f"MODEL: {model_name}")
        print("=" * 50)
        
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # Check if it's a full checkpoint or just state dict
            if isinstance(checkpoint, dict) and 'val_metrics' in checkpoint:
                print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Global IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
                
                val_metrics = checkpoint.get('val_metrics', {})
                for key, value in val_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key.capitalize()}: {value:.4f}")
                    elif isinstance(value, (int, str)):
                        print(f"  {key.capitalize()}: {value}")
            else:
                print("Note: This file appears to be a state_dict only (no metrics stored).")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            
    print("\n" + "=" * 50)

if __name__ == "__main__":
    extract_metrics()
