import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        dims = (2, 3)
        intersection = (pred_soft * target_onehot).sum(dims)
        union = pred_soft.sum(dims) + target_onehot.sum(dims)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class MorphologicalBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_float = target.unsqueeze(1).float()
        
        # Morphological operations (dilation and erosion with 3x3 kernel)
        dilated = F.max_pool2d(target_float, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target_float, kernel_size=3, stride=1, padding=1)
        
        # Boundary is difference between dilation and erosion
        boundary_gt = dilated - eroded
        
        return self.bce(pred, boundary_gt)

class WACVCombinedLoss(nn.Module):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, dice_smooth: float = 1.0):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.boundary_loss = MorphologicalBoundaryLoss()
    
    def forward(self, outputs: dict, target: torch.Tensor) -> dict:
        losses = {}
        
        # Main region loss: 0.5 * Focal + 0.5 * Dice
        main_pred = outputs['main']
        if main_pred.shape[2:] != target.shape[1:]:
            target_resized = F.interpolate(
                target.unsqueeze(1).float(), 
                size=main_pred.shape[2:], 
                mode='nearest'
            ).squeeze(1).long()
        else:
            target_resized = target
            
        main_focal = self.focal_loss(main_pred, target_resized)
        main_dice = self.dice_loss(main_pred, target_resized)
        
        losses['focal'] = main_focal
        losses['dice'] = main_dice
        losses['seg'] = 0.5 * main_focal + 0.5 * main_dice
        
        # Auxiliary deep supervision: SUM_k 0.1 * (Focal_k + Dice_k)
        if 'aux' in outputs:
            aux_loss = 0.0
            for aux_pred in outputs['aux']:
                aux_target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=aux_pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
                
                aux_focal = self.focal_loss(aux_pred, aux_target)
                aux_dice = self.dice_loss(aux_pred, aux_target)
                aux_loss += aux_focal + aux_dice
            
            losses['aux'] = 0.1 * aux_loss
        else:
            losses['aux'] = torch.tensor(0.0, device=main_pred.device)
            
        # Boundary loss: 0.2 * BCE
        if 'boundary' in outputs:
            boundary_pred = outputs['boundary']
            losses['boundary'] = 0.2 * self.boundary_loss(boundary_pred, target_resized)
        else:
            losses['boundary'] = torch.tensor(0.0, device=main_pred.device)
            
        losses['total'] = losses['seg'] + losses['aux'] + losses['boundary']
        
        return losses
