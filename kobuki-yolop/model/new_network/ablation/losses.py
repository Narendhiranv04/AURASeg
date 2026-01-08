"""
Losses for Ablation Study
=========================
Implements Focal Loss, Dice Loss, and Combined Loss with deep supervision support.

Reference:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Dice Loss: Sudre et al., "Generalised Dice overlap as a deep learning loss function" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore (default: -100)
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C, H, W)
            targets: Ground truth of shape (N, H, W)
        
        Returns:
            Focal loss value
        """
        # Get number of classes
        num_classes = inputs.size(1)
        
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding for targets
        # Handle ignore_index
        valid_mask = (targets != self.ignore_index)
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0  # Temporarily set to valid index
        
        # One-hot encode: (N, H, W) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets_clamped, num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Compute p_t
        p_t = (p * targets_one_hot).sum(dim=1)  # (N, H, W)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy (log softmax + nll)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha balancing
        # For binary: alpha for positive, (1-alpha) for negative
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_loss
        
        # Apply valid mask
        focal_loss = focal_loss * valid_mask.float()
        
        if self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Generalized Dice Loss for segmentation.
    
    DL = 1 - (2 * intersection + smooth) / (union + smooth)
    
    Args:
        smooth: Smoothing factor to prevent division by zero (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Label to ignore (default: -100)
    """
    
    def __init__(
        self, 
        smooth: float = 1.0, 
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (N, C, H, W)
            targets: Ground truth of shape (N, H, W)
        
        Returns:
            Dice loss value
        """
        num_classes = inputs.size(1)
        
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create valid mask
        valid_mask = (targets != self.ignore_index)
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets_clamped, num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).expand_as(probs).float()
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Compute intersection and union
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)
        
        # Compute Dice score per class
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice loss
        dice_loss = 1. - dice_score
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss with deep supervision support.
    
    Total Loss = sum_i(w_i * (lambda_focal * FL_i + lambda_dice * DL_i))
    
    Uses decaying schedule for supervision weights (higher weight for finer scales).
    
    Args:
        num_supervisors: Number of supervision outputs (default: 4)
        lambda_focal: Weight for focal loss (default: 1.0)
        lambda_dice: Weight for dice loss (default: 1.0)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        supervision_weights: List of weights for each supervision level.
                           If None, uses decaying schedule [0.1, 0.2, 0.3, 0.4]
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        num_supervisors: int = 4,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        supervision_weights: Optional[List[float]] = None,
        ignore_index: int = -100
    ):
        super(CombinedLoss, self).__init__()
        
        self.num_supervisors = num_supervisors
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        
        # Decaying schedule: higher weights for finer scales (later supervisors)
        if supervision_weights is None:
            # Default: [0.1, 0.2, 0.3, 0.4] for 4 supervisors
            self.supervision_weights = [0.1, 0.2, 0.3, 0.4]
        else:
            self.supervision_weights = supervision_weights
            
        # Normalize weights to sum to 1
        total = sum(self.supervision_weights)
        self.supervision_weights = [w / total for w in self.supervision_weights]
        
        # Initialize loss functions
        self.focal_loss = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(
        self, 
        predictions: Union[torch.Tensor, List[torch.Tensor]], 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: List of predictions from supervisors [(N,C,H1,W1), (N,C,H2,W2), ...]
                        Ordered from coarsest to finest. Can also be a single Tensor.
            target: Ground truth of shape (N, H, W)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components for logging
        """
        total_loss = 0.0
        loss_dict = {
            'total': 0.0,
            'focal': 0.0,
            'dice': 0.0
        }
        
        # Handle single tensor input
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]
            
        # Determine weights to use
        weights = self.supervision_weights
        if len(predictions) == 1:
            weights = [1.0]
        
        target_h, target_w = target.shape[1], target.shape[2]
        
        for i, (pred, weight) in enumerate(zip(predictions, weights)):
            # Upsample prediction to target size
            pred_upsampled = F.interpolate(
                pred, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Compute losses
            fl = self.focal_loss(pred_upsampled, target)
            dl = self.dice_loss(pred_upsampled, target)
            
            # Weighted combination
            combined = self.lambda_focal * fl + self.lambda_dice * dl
            weighted_loss = weight * combined
            
            total_loss = total_loss + weighted_loss
            
            # Log individual losses
            loss_dict[f'sup{i+1}_focal'] = fl.item()
            loss_dict[f'sup{i+1}_dice'] = dl.item()
            loss_dict[f'sup{i+1}_combined'] = combined.item()
            loss_dict[f'sup{i+1}_weighted'] = weighted_loss.item()
            
        loss_dict['total'] = total_loss.item()
        loss_dict['focal'] = sum(loss_dict.get(f'sup{i+1}_focal', 0) * w 
                                 for i, w in enumerate(self.supervision_weights))
        loss_dict['dice'] = sum(loss_dict.get(f'sup{i+1}_dice', 0) * w 
                                for i, w in enumerate(self.supervision_weights))
        
        return total_loss, loss_dict


class SingleScaleLoss(nn.Module):
    """
    Single scale loss for base ablation model (no deep supervision).
    
    Args:
        lambda_focal: Weight for focal loss
        lambda_dice: Weight for dice loss
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ignore_index: int = -100
    ):
        super(SingleScaleLoss, self).__init__()
        
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prediction: Prediction of shape (N, C, H, W)
            target: Ground truth of shape (N, H, W)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with loss components
        """
        target_h, target_w = target.shape[1], target.shape[2]
        
        # Upsample if needed
        if prediction.shape[2] != target_h or prediction.shape[3] != target_w:
            prediction = F.interpolate(
                prediction,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        fl = self.focal_loss(prediction, target)
        dl = self.dice_loss(prediction, target)
        
        total_loss = self.lambda_focal * fl + self.lambda_dice * dl
        
        loss_dict = {
            'total': total_loss.item(),
            'focal': fl.item(),
            'dice': dl.item()
        }
        
        return total_loss, loss_dict


class BoundaryWeightedLoss(nn.Module):
    """
    Boundary-Weighted Loss for RBRM Training (Idea 1)
    
    This loss emphasizes pixels near edges/boundaries while reducing
    the penalty for interior pixels. This helps RBRM focus on what
    it's designed for: boundary refinement.
    
    Loss = (Focal + Dice) * Boundary_Weight_Map
    
    Boundary weight map is computed by:
    1. Dilating the ground truth mask
    2. Eroding the ground truth mask  
    3. XOR to get boundary region
    4. Distance transform to create smooth falloff
    
    Args:
        lambda_focal: Weight for focal loss
        lambda_dice: Weight for dice loss
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        boundary_width: Width of boundary region in pixels (default: 5)
        boundary_weight: Weight multiplier for boundary pixels (default: 5.0)
        interior_weight: Weight for interior pixels (default: 0.5)
        ignore_index: Label to ignore
    """
    
    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_width: int = 5,
        boundary_weight: float = 5.0,
        interior_weight: float = 0.5,
        ignore_index: int = -100
    ):
        super(BoundaryWeightedLoss, self).__init__()
        
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.interior_weight = interior_weight
        self.ignore_index = ignore_index
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction='none'  # Get per-pixel loss
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        # Create dilation/erosion kernels
        kernel_size = 2 * boundary_width + 1
        self.register_buffer(
            'kernel', 
            torch.ones(1, 1, kernel_size, kernel_size)
        )
        self.padding = boundary_width
        
    def compute_boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary weight mask from ground truth.
        
        Args:
            target: Ground truth mask (N, H, W)
            
        Returns:
            weight_mask: Boundary weight mask (N, H, W)
        """
        # Convert to float and add channel dim
        mask = target.float().unsqueeze(1)  # (N, 1, H, W)
        
        # Dilate: max pooling
        dilated = F.max_pool2d(
            mask, 
            kernel_size=2*self.boundary_width+1, 
            stride=1, 
            padding=self.boundary_width
        )
        
        # Erode: min pooling (negate, max pool, negate)
        eroded = -F.max_pool2d(
            -mask, 
            kernel_size=2*self.boundary_width+1, 
            stride=1, 
            padding=self.boundary_width
        )
        
        # Boundary = dilated XOR eroded (difference between them)
        boundary = (dilated - eroded).abs()  # 1 at boundary, 0 elsewhere
        
        # Create weight map: high weight at boundary, lower at interior
        weight_mask = self.interior_weight + (self.boundary_weight - self.interior_weight) * boundary
        
        return weight_mask.squeeze(1)  # (N, H, W)
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            prediction: Prediction of shape (N, C, H, W)
            target: Ground truth of shape (N, H, W)
        
        Returns:
            total_loss: Boundary-weighted combined loss
            loss_dict: Dictionary with loss components
        """
        target_h, target_w = target.shape[1], target.shape[2]
        
        # Upsample if needed
        if prediction.shape[2] != target_h or prediction.shape[3] != target_w:
            prediction = F.interpolate(
                prediction,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Compute boundary weight mask
        weight_mask = self.compute_boundary_mask(target)  # (N, H, W)
        
        # Compute per-pixel focal loss
        focal_per_pixel = self.focal_loss(prediction, target)  # (N, H, W)
        
        # Apply boundary weighting to focal loss
        valid_mask = (target != self.ignore_index).float()
        weighted_focal = (focal_per_pixel * weight_mask * valid_mask).sum() / (valid_mask.sum().clamp(min=1))
        
        # Dice loss (already reduction='mean', harder to weight per-pixel)
        # We use standard dice but the focal weighting handles boundary emphasis
        dice = self.dice_loss(prediction, target)
        
        total_loss = self.lambda_focal * weighted_focal + self.lambda_dice * dice
        
        loss_dict = {
            'total': total_loss.item(),
            'focal': weighted_focal.item(),
            'dice': dice.item(),
            'boundary_weight_mean': weight_mask.mean().item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test the losses
    print("Testing Focal Loss...")
    focal = FocalLoss()
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64))
    loss = focal(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("\nTesting Dice Loss...")
    dice = DiceLoss()
    loss = dice(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    print("\nTesting Combined Loss with Deep Supervision...")
    combined = CombinedLoss(num_supervisors=4)
    preds = [
        torch.randn(2, 2, 20, 12),   # sup1 (coarsest)
        torch.randn(2, 2, 40, 24),   # sup2
        torch.randn(2, 2, 80, 48),   # sup3
        torch.randn(2, 2, 160, 96),  # sup4 (finest)
    ]
    target = torch.randint(0, 2, (2, 640, 384))
    loss, loss_dict = combined(preds, target)
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    print("\nTesting Single Scale Loss...")
    single = SingleScaleLoss()
    pred = torch.randn(2, 2, 160, 96)
    target = torch.randint(0, 2, (2, 640, 384))
    loss, loss_dict = single(pred, target)
    print(f"Single Scale Loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
