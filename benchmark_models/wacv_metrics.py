import numpy as np
import cv2

def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int = 2) -> dict:
    metrics = {}
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        ious.append(iou)
        
    metrics['iou_background'] = ious[0]
    metrics['iou_drivable'] = ious[1] if len(ious) > 1 else 0.0
    metrics['miou'] = np.mean(ious)
    
    pred_fg = (preds == 1)
    target_fg = (targets == 1)
    intersection = (pred_fg & target_fg).sum()
    dice = (2 * intersection) / (pred_fg.sum() + target_fg.sum() + 1e-6)
    metrics['dice'] = dice
    
    tp = (pred_fg & target_fg).sum()
    fp = (pred_fg & ~target_fg).sum()
    fn = (~pred_fg & target_fg).sum()
    tn = (~pred_fg & ~target_fg).sum()
    
    metrics['precision'] = tp / (tp + fp + 1e-6)
    metrics['recall'] = tp / (tp + fn + 1e-6)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-6)
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    
    return metrics

def compute_boundary_metrics(preds: np.ndarray, targets: np.ndarray, k: int = 2) -> dict:
    kernel = np.ones((3, 3), np.uint8)
    
    boundary_ious = []
    boundary_precisions = []
    boundary_recalls = []
    boundary_f1s = []
    
    for i in range(len(preds)):
        pred_binary = (preds[i] == 1).astype(np.uint8)
        target_binary = (targets[i] == 1).astype(np.uint8)
        
        pred_boundary = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel)
        target_boundary = cv2.morphologyEx(target_binary, cv2.MORPH_GRADIENT, kernel)
        
        pred_boundary = cv2.dilate(pred_boundary, kernel, iterations=k)
        target_boundary = cv2.dilate(target_boundary, kernel, iterations=k)
        
        tp = np.sum((pred_boundary > 0) & (target_boundary > 0))
        fp = np.sum((pred_boundary > 0) & (target_boundary == 0))
        fn = np.sum((pred_boundary == 0) & (target_boundary > 0))
        
        boundary_iou = tp / (tp + fp + fn + 1e-6)
        boundary_precision = tp / (tp + fp + 1e-6)
        boundary_recall = tp / (tp + fn + 1e-6)
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-6)
        
        boundary_ious.append(boundary_iou)
        boundary_precisions.append(boundary_precision)
        boundary_recalls.append(boundary_recall)
        boundary_f1s.append(boundary_f1)
    
    return {
        'boundary_iou': np.mean(boundary_ious),
        'boundary_precision': np.mean(boundary_precisions),
        'boundary_recall': np.mean(boundary_recalls),
        'boundary_f1': np.mean(boundary_f1s)
    }
