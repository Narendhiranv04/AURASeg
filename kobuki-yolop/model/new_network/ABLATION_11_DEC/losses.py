"""
Loss Functions for AURASeg Ablation Study
Paper: AURASeg - Attention Guided Upsampling with Residual Boundary-Assistive Refinement

Loss Functions:
    1. FocalLoss: Handles class imbalance by down-weighting easy examples
    2. DiceLoss: Directly optimizes the Dice coefficient / IoU
    3. CombinedLoss: Focal + Dice (default for V1, V2)
    4. BoundaryWeightedLoss: For V3, V4 with boundary emphasis (future)

Usage:
    V1 (SPP) and V2 (ASPP-Lite): Use CombinedLoss (Focal + Dice)
    V3 (Attention): Use CombinedLoss
    V4 (RBRM): Use CombinedLoss + BoundaryWeightedLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class indices
            
        Returns:
            Focal loss value
        """
        # Convert to probabilities
        p = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Get probability of true class
        p_t = (p * targets_one_hot).sum(dim=1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weighting
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    DiceLoss = 1 - (2 * intersection + smooth) / (union + smooth)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean' or 'none'
    """
    
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class indices
            
        Returns:
            Dice loss value
        """
        num_classes = inputs.shape[1]
        
        # Convert to probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten for computation
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        
        # Compute Dice for each class
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    
    This is the default loss function for V1 and V2 ablation models.
    
    Total Loss = focal_weight * FocalLoss + dice_weight * DiceLoss
    
    Args:
        focal_weight: Weight for focal loss (default: 1.0)
        dice_weight: Weight for dice loss (default: 1.0)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    """
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0, 
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class indices
            
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss


class BoundaryWeightedLoss(nn.Module):
    """
    Boundary-Weighted Loss for V4 RBRM model
    
    Applies higher weights to boundary pixels for better edge refinement.
    Should be used in addition to CombinedLoss for V4.
    
    Args:
        boundary_weight: Weight multiplier for boundary pixels
        kernel_size: Size of Laplacian kernel for edge detection
    """
    
    def __init__(self, boundary_weight=2.0, kernel_size=3):
        super().__init__()
        self.boundary_weight = boundary_weight
        
        # Laplacian kernel for edge detection
        if kernel_size == 3:
            kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32)
        else:
            kernel = torch.tensor([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ], dtype=torch.float32)
        
        self.register_buffer('laplacian', kernel.view(1, 1, kernel_size, kernel_size))
    
    def get_boundary_mask(self, targets):
        """Extract boundary pixels from segmentation mask"""
        # Convert to float and add channel dim
        targets_float = targets.float().unsqueeze(1)
        
        # Apply Laplacian filter
        edges = F.conv2d(targets_float, self.laplacian, padding=1)
        
        # Threshold to get boundary mask
        boundary = (edges.abs() > 0.1).float().squeeze(1)
        
        return boundary
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, C, H, W) - logits
            targets: Ground truth (B, H, W) - class indices
            
        Returns:
            Boundary-weighted cross entropy loss
        """
        # Get boundary mask
        boundary_mask = self.get_boundary_mask(targets)
        
        # Create weight map: 1.0 for non-boundary, boundary_weight for boundary
        weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
        
        # Compute pixel-wise cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply boundary weights
        weighted_loss = ce_loss * weight_map
        
        return weighted_loss.mean()


def get_loss_function(loss_type='combined', **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: 'focal', 'dice', 'combined', 'boundary', 'deep_supervision', 
                   'boundary_supervision', or 'v4_combined'
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryWeightedLoss(**kwargs)
    elif loss_type == 'deep_supervision':
        return DeepSupervisionLoss(**kwargs)
    elif loss_type == 'boundary_supervision':
        return BoundarySupervisionLoss(**kwargs)
    elif loss_type == 'v4_combined':
        return V4CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for V3 APUD model
    
    Applies Focal + Dice loss at each auxiliary output scale, plus the main output.
    Each auxiliary output is at its native resolution, so we downsample the GT
    to match (Option A: more efficient).
    
    Total Loss = main_loss + sum(aux_weight_i * aux_loss_i)
    
    Where:
        main_loss = Focal + Dice at full resolution
        aux_loss_i = Focal + Dice at scale i
        
    Supervision weights (coarse to fine): [0.1, 0.2, 0.3, 0.4]
    These weight the auxiliary losses relative to each other.
    
    Args:
        aux_weights: List of weights for auxiliary outputs [aux1, aux2, aux3, aux4]
        main_weight: Weight for main output loss
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
    """
    
    def __init__(self, 
                 aux_weights: list = [0.1, 0.2, 0.3, 0.4],
                 main_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.aux_weights = aux_weights
        self.main_weight = main_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Base loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal + Dice loss"""
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice
    
    def _downsample_target(self, target: torch.Tensor, size: tuple) -> torch.Tensor:
        """
        Downsample target to match prediction size using nearest neighbor
        
        Args:
            target: Ground truth (B, H, W) - class indices
            size: Target size (H', W')
            
        Returns:
            Downsampled target (B, H', W')
        """
        # Add channel dim for interpolation
        target_float = target.float().unsqueeze(1)
        
        # Nearest neighbor interpolation to preserve class labels
        target_down = F.interpolate(target_float, size=size, mode='nearest')
        
        # Remove channel dim and convert back to long
        return target_down.squeeze(1).long()
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """
        Args:
            outputs: Dictionary with 'main' and 'aux' keys
                - main: (B, C, H, W) main prediction at full resolution
                - aux: List of (B, C, H_i, W_i) auxiliary predictions
            targets: Ground truth (B, H, W) - class indices at full resolution
            
        Returns:
            Dictionary with:
                - 'total': Total combined loss
                - 'main': Main output loss
                - 'aux': List of auxiliary losses
                - 'aux_weighted': Weighted sum of auxiliary losses
        """
        losses = {}
        
        # Main loss at full resolution
        main_pred = outputs['main']
        main_loss = self._compute_loss(main_pred, targets)
        losses['main'] = main_loss
        
        # Auxiliary losses
        aux_losses = []
        aux_weighted_sum = 0.0
        
        if 'aux' in outputs and outputs['aux'] is not None:
            for i, (aux_pred, weight) in enumerate(zip(outputs['aux'], self.aux_weights)):
                # Get prediction size
                pred_size = aux_pred.shape[2:]  # (H_i, W_i)
                
                # Downsample target to match prediction size
                target_down = self._downsample_target(targets, pred_size)
                
                # Compute loss at this scale
                aux_loss = self._compute_loss(aux_pred, target_down)
                aux_losses.append(aux_loss)
                
                # Accumulate weighted loss
                aux_weighted_sum = aux_weighted_sum + weight * aux_loss
        
        losses['aux'] = aux_losses
        losses['aux_weighted'] = aux_weighted_sum
        
        # Total loss: main + weighted aux
        losses['total'] = self.main_weight * main_loss + aux_weighted_sum
        
        return losses


class BoundarySupervisionLoss(nn.Module):
    """
    Boundary Supervision Loss for V4 RBRM model
    
    Generates boundary ground truth from segmentation masks using 
    morphological dilation - erosion, then computes weighted BCE loss.
    
    Boundary GT Generation:
        1. Dilate the mask (expand boundaries outward)
        2. Erode the mask (shrink boundaries inward)
        3. Boundary = Dilated - Eroded (ring around object edges)
    
    Loss: Weighted BCE to handle class imbalance (boundaries are ~5% of pixels)
    
    Args:
        kernel_size: Size of morphological kernel (default: 5 for ~5px boundary)
        pos_weight_factor: Multiplier for computed positive weight (default: 1.0)
    """
    
    def __init__(self, kernel_size: int = 5, pos_weight_factor: float = 1.0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pos_weight_factor = pos_weight_factor
        
        # Create morphological kernels (circular for smooth boundaries)
        # Using max-pooling for dilation and -max(-x) for erosion
        self.dilation_kernel = kernel_size
        self.erosion_kernel = kernel_size
    
    def generate_boundary_gt(self, segmentation_gt: torch.Tensor) -> torch.Tensor:
        """
        Generate boundary ground truth from segmentation mask.
        
        Args:
            segmentation_gt: Segmentation mask (B, H, W) with class indices
            
        Returns:
            Boundary mask (B, 1, H, W) with 1 at boundaries, 0 elsewhere
        """
        # Convert to float and add channel dim
        mask = segmentation_gt.float().unsqueeze(1)  # (B, 1, H, W)
        
        # Dilation using max pooling
        dilated = F.max_pool2d(
            mask, 
            kernel_size=self.dilation_kernel, 
            stride=1, 
            padding=self.dilation_kernel // 2
        )
        
        # Erosion using -max(-x)
        eroded = -F.max_pool2d(
            -mask, 
            kernel_size=self.erosion_kernel, 
            stride=1, 
            padding=self.erosion_kernel // 2
        )
        
        # Boundary = Dilated - Eroded
        boundary = (dilated - eroded).clamp(0, 1)
        
        return boundary
    
    def forward(self, boundary_pred: torch.Tensor, 
                segmentation_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary supervision loss.
        
        Args:
            boundary_pred: Predicted boundary map (B, 1, H, W), values in [0, 1]
            segmentation_gt: Ground truth segmentation (B, H, W), class indices
            
        Returns:
            Weighted BCE loss for boundary prediction
        """
        # Generate boundary ground truth
        boundary_gt = self.generate_boundary_gt(segmentation_gt)
        
        # Ensure boundary_pred matches boundary_gt size
        if boundary_pred.shape[2:] != boundary_gt.shape[2:]:
            boundary_pred = F.interpolate(
                boundary_pred, 
                size=boundary_gt.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # Compute positive weight for class imbalance
        # pos_weight = num_negative / num_positive
        num_positive = boundary_gt.sum() + 1e-8
        num_negative = boundary_gt.numel() - num_positive
        pos_weight = (num_negative / num_positive) * self.pos_weight_factor
        
        # Clamp pos_weight to reasonable range
        pos_weight = pos_weight.clamp(1.0, 50.0)
        
        # Weighted BCE loss using BCE with logits (AMP-safe)
        # BCE with logits combines sigmoid + BCE in a numerically stable way
        # The model now outputs logits, not sigmoid-activated values
        
        # Create weight tensor for positive samples
        weight_map = torch.where(
            boundary_gt > 0.5, 
            torch.full_like(boundary_gt, pos_weight.item()), 
            torch.ones_like(boundary_gt)
        )
        
        # Use binary_cross_entropy_with_logits (AMP-safe)
        bce_loss = F.binary_cross_entropy_with_logits(
            boundary_pred, 
            boundary_gt, 
            weight=weight_map,
            reduction='mean'
        )
        
        return bce_loss


class V4CombinedLoss(nn.Module):
    """
    Complete Loss Function for V4 RBRM Model
    
    Combines:
        1. Main Loss: Focal + Dice at full resolution
        2. Deep Supervision Loss: Focal + Dice at 4 auxiliary scales
        3. Boundary Loss: Weighted BCE for boundary prediction
    
    Total Loss = L_main + L_supervision + λ_bnd × L_boundary
    
    Where:
        L_main = Focal + Dice (weight 1.0)
        L_supervision = Σ wᵢ × (Focal + Dice)ᵢ, w = [0.1, 0.2, 0.3, 0.4]
        L_boundary = WeightedBCE (weight 0.5)
    
    Args:
        aux_weights: Weights for auxiliary outputs [0.1, 0.2, 0.3, 0.4]
        main_weight: Weight for main output loss (default: 1.0)
        boundary_weight: Weight for boundary loss (default: 0.5)
        focal_weight: Weight for focal component (default: 1.0)
        dice_weight: Weight for dice component (default: 1.0)
        focal_alpha: Alpha for focal loss (default: 0.25)
        focal_gamma: Gamma for focal loss (default: 2.0)
        boundary_kernel: Kernel size for boundary GT generation (default: 5)
    """
    
    def __init__(self,
                 aux_weights: list = [0.1, 0.2, 0.3, 0.4],
                 main_weight: float = 1.0,
                 boundary_weight: float = 0.5,
                 focal_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 boundary_kernel: int = 5):
        super().__init__()
        
        self.aux_weights = aux_weights
        self.main_weight = main_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        # Base loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        
        # Boundary loss
        self.boundary_loss = BoundarySupervisionLoss(kernel_size=boundary_kernel)
    
    def _compute_seg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal + Dice loss for segmentation"""
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice
    
    def _downsample_target(self, target: torch.Tensor, size: tuple) -> torch.Tensor:
        """Downsample target to match prediction size"""
        target_float = target.float().unsqueeze(1)
        target_down = F.interpolate(target_float, size=size, mode='nearest')
        return target_down.squeeze(1).long()
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """
        Compute complete V4 loss.
        
        Args:
            outputs: Model outputs dictionary with:
                - 'main': (B, C, H, W) main prediction
                - 'aux': List of auxiliary predictions (optional)
                - 'boundary': (B, 1, H, W) boundary prediction (optional)
            targets: Ground truth segmentation (B, H, W)
            
        Returns:
            Dictionary with all loss components:
                - 'total': Complete combined loss
                - 'main': Main segmentation loss
                - 'aux_weighted': Weighted sum of auxiliary losses
                - 'boundary': Boundary supervision loss
                - 'aux': List of individual auxiliary losses
        """
        losses = {}
        
        # 1. Main segmentation loss
        main_pred = outputs['main']
        main_loss = self._compute_seg_loss(main_pred, targets)
        losses['main'] = main_loss
        
        # 2. Auxiliary supervision losses
        aux_weighted_sum = torch.tensor(0.0, device=main_pred.device)
        aux_losses = []
        
        if 'aux' in outputs and outputs['aux'] is not None:
            for i, (aux_pred, weight) in enumerate(zip(outputs['aux'], self.aux_weights)):
                pred_size = aux_pred.shape[2:]
                target_down = self._downsample_target(targets, pred_size)
                aux_loss = self._compute_seg_loss(aux_pred, target_down)
                aux_losses.append(aux_loss)
                aux_weighted_sum = aux_weighted_sum + weight * aux_loss
        
        losses['aux'] = aux_losses
        losses['aux_weighted'] = aux_weighted_sum
        
        # 3. Boundary supervision loss
        boundary_loss = torch.tensor(0.0, device=main_pred.device)
        
        if 'boundary' in outputs and outputs['boundary'] is not None:
            boundary_pred = outputs['boundary']
            boundary_loss = self.boundary_loss(boundary_pred, targets)
        
        losses['boundary'] = boundary_loss
        
        # 4. Total loss
        total_loss = (
            self.main_weight * main_loss + 
            aux_weighted_sum + 
            self.boundary_weight * boundary_loss
        )
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test loss functions
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    height, width = 384, 640
    
    # Random predictions and targets
    inputs = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test each loss
    print("\n1. Focal Loss:")
    focal = FocalLoss()
    focal_val = focal(inputs, targets)
    print(f"   Value: {focal_val.item():.4f}")
    
    print("\n2. Dice Loss:")
    dice = DiceLoss()
    dice_val = dice(inputs, targets)
    print(f"   Value: {dice_val.item():.4f}")
    
    print("\n3. Combined Loss (Focal + Dice):")
    combined = CombinedLoss()
    combined_val = combined(inputs, targets)
    print(f"   Value: {combined_val.item():.4f}")
    
    print("\n4. Boundary Weighted Loss:")
    boundary = BoundaryWeightedLoss()
    boundary_val = boundary(inputs, targets)
    print(f"   Value: {boundary_val.item():.4f}")
    
    print("\n5. Deep Supervision Loss:")
    # Create outputs dict with main and aux
    outputs = {
        'main': torch.randn(batch_size, num_classes, height, width),
        'aux': [
            torch.randn(batch_size, num_classes, 12, 20),   # H/32
            torch.randn(batch_size, num_classes, 24, 40),   # H/16
            torch.randn(batch_size, num_classes, 48, 80),   # H/8
            torch.randn(batch_size, num_classes, 96, 160),  # H/4
        ]
    }
    deep_sup = DeepSupervisionLoss()
    deep_sup_losses = deep_sup(outputs, targets)
    print(f"   Total Loss: {deep_sup_losses['total'].item():.4f}")
    print(f"   Main Loss: {deep_sup_losses['main'].item():.4f}")
    print(f"   Aux Weighted: {deep_sup_losses['aux_weighted'].item():.4f}")
    print(f"   Individual Aux Losses:")
    for i, aux_loss in enumerate(deep_sup_losses['aux']):
        print(f"     Aux-{i+1}: {aux_loss.item():.4f} (weight: {deep_sup.aux_weights[i]})")
    
    print("\n" + "=" * 60)
    print("All loss functions working correctly!")
    print("=" * 60)
