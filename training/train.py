"""Training pipeline for road segmentation model."""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel
from utils.dataset import build_dataloaders
from utils.losses import BCEDiceLoss
from utils.metrics import SegmentationMetrics
from utils.visualization import plot_training_curves, plot_metrics


class Trainer:
    """Handles model training, validation, and checkpointing."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config["training"].get("device", "cuda")
            if torch.cuda.is_available() else "cpu"
        )

        # Set seed for reproducibility
        seed = config["training"].get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build model
        model_config = config["model"]
        self.model = RoadSegmentationModel(
            encoder=model_config.get("encoder", "resnet50"),
            pretrained=model_config.get("pretrained", True),
            transformer_dim=model_config.get("transformer_dim", 512),
            transformer_heads=model_config.get("transformer_heads", 8),
            transformer_layers=model_config.get("transformer_layers", 4),
            dropout=model_config.get("dropout_rate", 0.5),
            num_classes=model_config.get("num_classes", 1),
        ).to(self.device)

        # Loss and optimizer
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"].get("learning_rate", 1e-4),
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Dataloaders
        self.train_loader, self.val_loader = build_dataloaders(config)

        # Metrics
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "val_accuracy": [],
        }

        # TensorBoard
        log_dir = config["training"].get("log_dir", "outputs/logs/")
        self.writer = SummaryWriter(log_dir)

        # Checkpoint directory
        self.checkpoint_dir = Path(config["training"].get("checkpoint_dir", "outputs/models/"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            total_loss += loss.item()
            num_batches += 1
            self.val_metrics.update(logits, masks)
            pbar.set_postfix({"loss": loss.item()})

        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / num_batches
        return metrics

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  Best model saved (val_loss: {val_loss:.4f})")

    def train(self):
        """Run the full training loop."""
        epochs = self.config["training"].get("epochs", 50)
        patience = self.config["training"].get("early_stopping_patience", 10)

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics["loss"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metrics/IoU", val_metrics["iou"], epoch)
            self.writer.add_scalar("Metrics/Dice", val_metrics["dice"], epoch)
            self.writer.add_scalar("Metrics/Accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Print summary
            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val IoU:    {val_metrics['iou']:.4f}")
            print(f"  Val Dice:   {val_metrics['dice']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            # Scheduler step
            self.scheduler.step(val_loss)

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        self.writer.close()

        # Plot final curves
        plot_training_curves(
            self.history["train_loss"],
            self.history["val_loss"],
            save_path="outputs/results/training_curves.png",
        )
        plot_metrics(
            {
                "IoU": self.history["val_iou"],
                "Dice": self.history["val_dice"],
                "Accuracy": self.history["val_accuracy"],
            },
            save_path="outputs/results/validation_metrics.png",
        )

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")


def main():
    """Entry point for training."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
