import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_loader_enhanced import load_real_flood_data, extract_balanced_patches_enhanced, FloodDatasetAugmented
from model_enhanced import AttentionUNet
from losses import CombinedLoss
from evaluation import FloodEvaluationMetrics

# QUICK CONFIG - Faster training for testing
BATCH_SIZE = 32  # Larger batches = faster
EPOCHS = 5       # Just 5 epochs for quick results
LEARNING_RATE = 0.001
MODEL_DIR = "models"
EVAL_DIR = "evaluation_quick"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_quick():
    """Quick training for fast results"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    print("=" * 70)
    print("QUICK TRAINING MODE (5 epochs for fast results)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\n[DATA] Loading data...")
    stack, label, profile = load_real_flood_data()
    X_numpy, y_numpy = extract_balanced_patches_enhanced(
        stack, label, 
        patch_size=32, 
        num_patches=5000,  # Fewer patches for speed
        flood_ratio=0.5
    )
    
    print(f"\n[OK] Dataset: {X_numpy.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_numpy, y_numpy, test_size=0.2, random_state=42
    )
    
    train_dataset = FloodDatasetAugmented(X_train, y_train, augment=True)
    val_dataset = FloodDatasetAugmented(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"[LOADER] {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Model
    print("\n[MODEL] Initializing Attention U-Net...")
    model = AttentionUNet(n_channels=4, n_classes=1).to(DEVICE)
    
    criterion = CombinedLoss(alpha=0.75, gamma=2.0, focal_weight=0.7, dice_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}
    best_val_iou = 0.0
    
    print("\n[TRAIN] Starting quick training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        epoch_train_loss = train_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        evaluator = FloodEvaluationMetrics(threshold=0.5)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                evaluator.update(outputs, targets)
        
        epoch_val_loss = val_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)
        
        val_metrics = evaluator.compute_all_metrics()
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"\n[EPOCH {epoch+1}/{EPOCHS}]")
        print(f"   Train Loss: {epoch_train_loss:.4f}")
        print(f"   Val Loss:   {epoch_val_loss:.4f}")
        print(f"   Val IoU:    {val_metrics['iou']:.4f}")
        print(f"   Val F1:     {val_metrics['f1']:.4f}")
        print(f"   Val Prec:   {val_metrics['precision']:.4f}")
        print(f"   Val Recall: {val_metrics['recall']:.4f}")
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            model_path = os.path.join(MODEL_DIR, "flood_model_quick.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': best_val_iou,
                'val_metrics': val_metrics
            }, model_path)
            print(f"   [SAVED] Best model! (IoU: {best_val_iou:.4f})")
        
        print("=" * 70)
    
    print(f"\n[COMPLETE] Quick training done!")
    print(f"   Best Val IoU: {best_val_iou:.4f}")
    
    # Final evaluation
    print("\n[EVAL] Generating evaluation report...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, "flood_model_quick.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    final_evaluator = FloodEvaluationMetrics(threshold=0.5)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            final_evaluator.update(outputs, targets)
    
    final_metrics = final_evaluator.generate_report(save_dir=EVAL_DIR)
    
    print(f"\n[SAVED] Results saved to: {EVAL_DIR}/")
    
    return model, history, final_metrics

if __name__ == "__main__":
    model, history, metrics = train_quick()
    print("\n[SUCCESS] Quick training completed!")
    print("\nNext steps:")
    print("1. Check evaluation/ folder for metrics and plots")
    print("2. Compare with full training when it completes")
    print("3. Use this model for immediate testing")
