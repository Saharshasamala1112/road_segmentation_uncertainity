"""Visualization utilities for segmentation results and training curves."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(
    metrics: dict[str, list[float]],
    save_path: Optional[str] = None,
):
    """Plot multiple metrics over epochs.

    Args:
        metrics: Dictionary mapping metric names to lists of values
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(next(iter(metrics.values()))) + 1)

    for name, values in metrics.items():
        ax.plot(epochs, values, label=name.upper(), linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation Metrics Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Visualize input image, ground truth, prediction, and uncertainty.

    Args:
        image: Input image (H, W, 3) in range [0, 1]
        mask: Ground truth mask (H, W) in range [0, 1]
        prediction: Predicted mask (H, W) in range [0, 1]
        uncertainty: Optional uncertainty map (H, W)
        save_path: Optional path to save the figure
    """
    n_plots = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction", fontsize=13)
    axes[2].axis("off")

    if uncertainty is not None:
        im = axes[3].imshow(uncertainty, cmap="hot")
        axes[3].set_title("Uncertainty", fontsize=13)
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_uncertainty_heatmap(
    image: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay uncertainty heatmap on input image.

    Args:
        image: Input image (H, W, 3) in range [0, 1]
        uncertainty: Uncertainty map (H, W)
        alpha: Opacity of the heatmap overlay
        save_path: Optional path to save the result

    Returns:
        Overlay image (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    im = ax.imshow(uncertainty, cmap="hot", alpha=alpha)
    ax.set_title("Uncertainty Heatmap Overlay", fontsize=14)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
