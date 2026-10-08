"""Evaluation metrics for road segmentation."""

import torch
import numpy as np


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, Dice, Accuracy, Precision, Recall."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update metrics with a new batch.

        Args:
            logits: (B, 1, H, W) raw model outputs
            targets: (B, 1, H, W) ground truth masks
        """
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()

    def compute(self) -> dict[str, float]:
        """Compute and return all metrics.

        Returns:
            Dictionary with metric names and values
        """
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        accuracy = (tp + tn + self.smooth) / (tp + tn + fp + fn + self.smooth)
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        f1 = (2 * precision * recall + self.smooth) / (precision + recall + self.smooth)

        return {
            "iou": iou,
            "dice": dice,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for numpy arrays.

    Args:
        pred: Predicted mask (H, W) or (H, W, 1)
        target: Ground truth mask (H, W) or (H, W, 1)
        threshold: Binarization threshold

    Returns:
        IoU score
    """
    pred = (pred > threshold).astype(np.uint8).flatten()
    target = (target > 0.5).astype(np.uint8).flatten()

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Dice coefficient for numpy arrays.

    Args:
        pred: Predicted mask (H, W) or (H, W, 1)
        target: Ground truth mask (H, W) or (H, W, 1)
        threshold: Binarization threshold

    Returns:
        Dice coefficient
    """
    pred = (pred > threshold).astype(np.uint8).flatten()
    target = (target > 0.5).astype(np.uint8).flatten()

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return 1.0
    return 2.0 * float(intersection) / float(total)
