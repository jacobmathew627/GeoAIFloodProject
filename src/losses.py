import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard examples and down-weights easy examples
    """
    def __init__(self, alpha=0.85, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha  # Weight for positive class (flood pixels)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal term
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss
    """
    def __init__(self, alpha=0.75, gamma=2.0, focal_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = WeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def dice_loss(self, inputs, targets, smooth=1.0):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice

class PhysicsGuidedLoss(nn.Module):
    """
    Physics-guided loss that penalizes predictions violating physical constraints
    """
    def __init__(self, base_loss, physics_weight=0.1):
        super(PhysicsGuidedLoss, self).__init__()
        self.base_loss = base_loss
        self.physics_weight = physics_weight
        
    def forward(self, inputs, targets, flow_acc=None, slope=None):
        # Base loss (Focal + Dice)
        loss = self.base_loss(inputs, targets)
        
        # Physics regularization (if flow/slope provided)
        if flow_acc is not None and slope is not None:
            # High flood risk should correlate with high flow accumulation
            # Low flood risk should correlate with steep slopes
            
            # Normalize flow and slope to [0, 1]
            flow_norm = (flow_acc - flow_acc.min()) / (flow_acc.max() - flow_acc.min() + 1e-6)
            slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
            
            # Physics expectation: high flow + low slope = high risk
            physics_risk = flow_norm * (1 - slope_norm)
            
            # Penalize large deviations from physics
            physics_penalty = F.mse_loss(inputs, physics_risk)
            
            loss = loss + self.physics_weight * physics_penalty
        
        return loss

if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    inputs = torch.rand(4, 1, 32, 32)
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
    
    # Focal Loss
    focal = WeightedFocalLoss(alpha=0.75, gamma=2.0)
    loss1 = focal(inputs, targets)
    print(f"Focal Loss: {loss1.item():.4f}")
    
    # Combined Loss
    combined = CombinedLoss(focal_weight=0.7, dice_weight=0.3)
    loss2 = combined(inputs, targets)
    print(f"Combined Loss: {loss2.item():.4f}")
    
    # Physics-Guided Loss
    flow = torch.rand(4, 1, 32, 32)
    slope = torch.rand(4, 1, 32, 32)
    physics_guided = PhysicsGuidedLoss(combined, physics_weight=0.1)
    loss3 = physics_guided(inputs, targets, flow, slope)
    print(f"Physics-Guided Loss: {loss3.item():.4f}")
