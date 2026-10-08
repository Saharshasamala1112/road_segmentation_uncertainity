"""Standalone validation script for road segmentation model."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel
from utils.dataset import RoadSegmentationDataset, build_transforms
from utils.losses import BCEDiceLoss
from utils.metrics import SegmentationMetrics


def load_model(checkpoint_path: str, device: torch.device) -> RoadSegmentationModel:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_config = config["model"]
    model = RoadSegmentationModel(
        encoder=model_config.get("encoder", "resnet50"),
        pretrained=False,
        transformer_dim=model_config.get("transformer_dim", 512),
        transformer_heads=model_config.get("transformer_heads", 8),
        transformer_layers=model_config.get("transformer_layers", 4),
        dropout=model_config.get("dropout_rate", 0.5),
        num_classes=model_config.get("num_classes", 1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validate(
    model: RoadSegmentationModel,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return metrics.

    Args:
        model: Trained model
        val_loader: Validation dataloader
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    criterion = BCEDiceLoss()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            num_batches += 1
            metrics.update(logits, masks)

    results = metrics.compute()
    results["loss"] = total_loss / num_batches
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate road segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Build validation dataset
    data_config = config["data"]
    val_transform = build_transforms({}, is_train=False)
    val_dataset = RoadSegmentationDataset(
        image_dir=data_config["val_images"],
        mask_dir=data_config["val_masks"],
        transform=val_transform,
        image_size=tuple(data_config.get("image_size", [512, 512])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
    )

    # Validate
    results = validate(model, val_loader, device)

    print("\nValidation Results:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"  {metric.upper():12s}: {value:.4f}")


if __name__ == "__main__":
    main()
