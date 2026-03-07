import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_loader_enhanced import load_real_flood_data, extract_balanced_patches_enhanced, FloodDatasetAugmented
from model_enhanced import AttentionUNet, UNet
from losses import CombinedLoss, WeightedFocalLoss
from evaluation import FloodEvaluationMetrics

# Config
BATCH_SIZE = 16  # Reduced for larger model
EPOCHS = 30
LEARNING_RATE = 0.0001
MODEL_DIR = "models"
EVAL_DIR = "evaluation"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_enhanced():
    """
    Enhanced training with:
    - Attention U-Net architecture
    - Real 2018 flood observations
    - Weighted Focal + Dice Loss
    - Comprehensive evaluation metrics
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    print("=" * 70)
    print("ENHANCED FLOOD PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # Load data with REAL flood observations
    print("\n[DATA] Loading data...")
    stack, label, profile = load_real_flood_data()
    X_numpy, y_numpy = extract_balanced_patches_enhanced(
        stack, label, 
        patch_size=128, 
        num_patches=5000,  # Balanced for memory at 128x128
        flood_ratio=0.5
    )
    
    print(f"\n[OK] Dataset prepared:")
    print(f"   X shape: {X_numpy.shape}")
    print(f"   y shape: {y_numpy.shape}")
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_numpy, y_numpy, test_size=0.2, random_state=42
    )
    
    # Create datasets with augmentation
    train_dataset = FloodDatasetAugmented(X_train, y_train, augment=True)
    val_dataset = FloodDatasetAugmented(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n[LOADER] Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Model
    print("\n[MODEL] Initializing Attention U-Net with 10 Input Channels...")
    model = AttentionUNet(n_channels=10, n_classes=1).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.7, dice_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*10, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    best_val_iou = 0.0
    best_epoch = 0
    
    print("\n[TRAIN] Starting training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        epoch_train_loss = train_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        evaluator = FloodEvaluationMetrics(threshold=0.5)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
            # Update metrics
            evaluator.update(outputs, targets)
        
        epoch_val_loss = val_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)
        
        # Compute validation metrics
        val_metrics = evaluator.compute_all_metrics()
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print epoch summary
        print(f"\n[EPOCH] Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"   Train Loss: {epoch_train_loss:.4f}")
        print(f"   Val Loss:   {epoch_val_loss:.4f}")
        print(f"   Val IoU:    {val_metrics['iou']:.4f}")
        print(f"   Val F1:     {val_metrics['f1']:.4f}")
        print(f"   Val Prec:   {val_metrics['precision']:.4f}")
        print(f"   Val Recall: {val_metrics['recall']:.4f}")
        print(f"   Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch + 1
            model_path = os.path.join(MODEL_DIR, "flood_model_enhanced.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_metrics': val_metrics
            }, model_path)
            print(f"   [SAVED] Best model saved! (IoU: {best_val_iou:.4f})")
        
        print("=" * 70)
    
    print(f"\n[COMPLETE] Training completed!")
    print(f"   Best Val IoU: {best_val_iou:.4f} (Epoch {best_epoch})")
    
    # Final evaluation on best model
    print("\n[EVAL] Final Evaluation on Best Model...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, "flood_model_enhanced.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    final_evaluator = FloodEvaluationMetrics(threshold=0.5)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            final_evaluator.update(outputs, targets)
    
    # Generate comprehensive report
    final_metrics = final_evaluator.generate_report(save_dir=EVAL_DIR)
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(EVAL_DIR, 'training_history.png'))
    
    print(f"\n[SAVED] All evaluation results saved to: {EVAL_DIR}/")
    
    return model, history, final_metrics

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', color='blue', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[1, 1].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model, history, metrics = train_enhanced()
    print("\n[SUCCESS] Enhanced training completed successfully!")
