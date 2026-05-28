import torch

def mask_iou(pred_mask, true_mask, eps=1e-7):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()
    return ((intersection + eps) / (union + eps)).item()

def dice_score(pred_mask, true_mask, eps=1e-7):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    intersection = (pred_mask & true_mask).sum().float()
    total = pred_mask.sum().float() + true_mask.sum().float()
    return ((2 * intersection + eps) / (total + eps)).item()

def pixel_accuracy(logits, masks):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / max(total, 1)
