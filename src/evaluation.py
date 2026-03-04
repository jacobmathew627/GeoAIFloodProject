import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class FloodEvaluationMetrics:
    """Comprehensive evaluation metrics for flood prediction"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, pred, target):
        """
        Update metrics with batch predictions
        pred: (B, 1, H, W) or (B, H, W) tensor
        target: (B, 1, H, W) or (B, H, W) tensor
        """
        pred = pred.detach().cpu().numpy().flatten()
        target = target.detach().cpu().numpy().flatten()
        
        self.predictions.extend(pred)
        self.targets.extend(target)
    
    def compute_iou(self, pred_binary=None, target_binary=None):
        """Intersection over Union (Jaccard Index)"""
        if pred_binary is None:
            pred_binary = (np.array(self.predictions) >= self.threshold).astype(int)
        if target_binary is None:
            target_binary = (np.array(self.targets) >= self.threshold).astype(int)
            
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_dice(self, pred_binary=None, target_binary=None):
        """Dice Coefficient (F1 for binary segmentation)"""
        if pred_binary is None:
            pred_binary = (np.array(self.predictions) >= self.threshold).astype(int)
        if target_binary is None:
            target_binary = (np.array(self.targets) >= self.threshold).astype(int)
            
        intersection = np.logical_and(pred_binary, target_binary).sum()
        
        if (pred_binary.sum() + target_binary.sum()) == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2 * intersection / (pred_binary.sum() + target_binary.sum())
    
    def compute_confusion_matrix(self):
        """Compute confusion matrix"""
        pred_binary = (np.array(self.predictions) >= self.threshold).astype(int)
        target_binary = (np.array(self.targets) >= self.threshold).astype(int)
        
        return confusion_matrix(target_binary, pred_binary)
    
    def compute_all_metrics(self):
        """Compute all metrics at once"""
        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)
        pred_binary = (pred_array >= self.threshold).astype(int)
        target_binary = (target_array >= self.threshold).astype(int)
        
        metrics = {}
        
        # Binary classification metrics
        metrics['iou'] = self.compute_iou(pred_binary, target_binary)
        metrics['dice'] = self.compute_dice(pred_binary, target_binary)
        metrics['precision'] = precision_score(target_binary, pred_binary, zero_division=0)
        metrics['recall'] = recall_score(target_binary, pred_binary, zero_division=0)
        metrics['f1'] = f1_score(target_binary, pred_binary, zero_division=0)
        
        # Probabilistic metrics
        try:
            metrics['roc_auc'] = roc_auc_score(target_binary, pred_array)
            metrics['pr_auc'] = average_precision_score(target_binary, pred_array)
        except:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = self.compute_confusion_matrix()
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        cm = self.compute_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Flood', 'Flood'],
                    yticklabels=['Non-Flood', 'Flood'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)
        target_binary = (target_array >= self.threshold).astype(int)
        
        fpr, tpr, thresholds = roc_curve(target_binary, pred_array)
        roc_auc = roc_auc_score(target_binary, pred_array)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve"""
        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)
        target_binary = (target_array >= self.threshold).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(target_binary, pred_array)
        pr_auc = average_precision_score(target_binary, pred_array)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return precision, recall, pr_auc
    
    def plot_calibration_curve(self, n_bins=10, save_path=None):
        """Plot calibration curve (reliability diagram)"""
        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)
        target_binary = (target_array >= self.threshold).astype(int)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        true_probs = []
        pred_probs = []
        
        for i in range(n_bins):
            mask = (pred_array >= bin_edges[i]) & (pred_array < bin_edges[i+1])
            if mask.sum() > 0:
                true_probs.append(target_binary[mask].mean())
                pred_probs.append(pred_array[mask].mean())
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(pred_probs, true_probs, 'o-', label='Model')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return pred_probs, true_probs
    
    def generate_report(self, save_dir=None):
        """Generate comprehensive evaluation report"""
        metrics = self.compute_all_metrics()
        
        print("=" * 60)
        print("FLOOD PREDICTION EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📊 Binary Classification Metrics:")
        print(f"  IoU (Jaccard):        {metrics['iou']:.4f}")
        print(f"  Dice Coefficient:     {metrics['dice']:.4f}")
        print(f"  Precision:            {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  F1 Score:             {metrics['f1']:.4f}")
        if 'specificity' in metrics:
            print(f"  Specificity:          {metrics['specificity']:.4f}")
        
        print("\n📈 Probabilistic Metrics:")
        print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:               {metrics['pr_auc']:.4f}")
        
        print("\n🔢 Confusion Matrix:")
        if 'true_positive' in metrics:
            print(f"  True Positives:       {metrics['true_positive']:,}")
            print(f"  True Negatives:       {metrics['true_negative']:,}")
            print(f"  False Positives:      {metrics['false_positive']:,}")
            print(f"  False Negatives:      {metrics['false_negative']:,}")
        
        print("=" * 60)
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Save plots
            self.plot_confusion_matrix(os.path.join(save_dir, 'confusion_matrix.png'))
            self.plot_roc_curve(os.path.join(save_dir, 'roc_curve.png'))
            self.plot_precision_recall_curve(os.path.join(save_dir, 'pr_curve.png'))
            self.plot_calibration_curve(save_path=os.path.join(save_dir, 'calibration_curve.png'))
            
            # Save metrics to file
            with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
        
        return metrics

if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing evaluation metrics...")
    
    # Simulate predictions
    np.random.seed(42)
    pred = torch.rand(100, 1, 32, 32)
    target = (torch.rand(100, 1, 32, 32) > 0.7).float()
    
    evaluator = FloodEvaluationMetrics(threshold=0.5)
    evaluator.update(pred, target)
    
    metrics = evaluator.generate_report(save_dir='evaluation_test')
    print("\nTest completed successfully!")
