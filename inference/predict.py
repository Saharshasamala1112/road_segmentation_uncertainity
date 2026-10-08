"""Inference script with MC Dropout uncertainty estimation."""

import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel
from utils.visualization import visualize_prediction, create_uncertainty_heatmap


def load_model(checkpoint_path: str, device: torch.device) -> RoadSegmentationModel:
    """Load a trained model from checkpoint."""
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


def preprocess_image(image_path: str, image_size: tuple[int, int]) -> torch.Tensor:
    """Load and preprocess an image for inference.

    Args:
        image_path: Path to the input image
        image_size: Target (height, width)

    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]))

    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # To tensor
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image


def predict(
    model: RoadSegmentationModel,
    image: torch.Tensor,
    mc_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference with MC Dropout.

    Args:
        model: Trained model
        image: Preprocessed image tensor (1, 3, H, W)
        mc_samples: Number of MC Dropout passes

    Returns:
        Tuple of (mean_prediction, uncertainty_map, binary_mask)
    """
    device = next(model.parameters()).device
    image = image.to(device)

    mean_pred, uncertainty = model.predict_with_uncertainty(image, mc_samples)

    # Convert to numpy
    mean_pred = mean_pred.squeeze().cpu().numpy()
    uncertainty = uncertainty.squeeze().cpu().numpy()

    # Binary mask
    binary_mask = (mean_pred > 0.5).astype(np.float32)

    return mean_pred, uncertainty, binary_mask


def main():
    parser = argparse.ArgumentParser(description="Road segmentation inference with uncertainty")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/results/", help="Output directory")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of MC Dropout samples")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Preprocess
    image_size = tuple(config["data"].get("image_size", [512, 512]))
    image_tensor = preprocess_image(args.image, image_size)

    # Load original image for visualization
    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (image_size[1], image_size[0]))
    original_image = original_image.astype(np.float32) / 255.0

    # Predict
    mean_pred, uncertainty, binary_mask = predict(model, image_tensor, args.mc_samples)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.image).stem

    # Save prediction
    cv2.imwrite(
        str(output_dir / f"{stem}_prediction.png"),
        (binary_mask * 255).astype(np.uint8),
    )

    # Save uncertainty
    cv2.imwrite(
        str(output_dir / f"{stem}_uncertainty.png"),
        (uncertainty / uncertainty.max() * 255).astype(np.uint8),
    )

    # Visualize
    visualize_prediction(
        image=original_image,
        mask=binary_mask,
        prediction=mean_pred,
        uncertainty=uncertainty,
        save_path=str(output_dir / f"{stem}_visualization.png"),
    )

    create_uncertainty_heatmap(
        image=original_image,
        uncertainty=uncertainty,
        save_path=str(output_dir / f"{stem}_heatmap.png"),
    )

    print(f"Results saved to {output_dir}")
    print(f"  Mean prediction: {mean_pred.mean():.4f}")
    print(f"  Max uncertainty: {uncertainty.max():.4f}")
    print(f"  Mean uncertainty: {uncertainty.mean():.4f}")


if __name__ == "__main__":
    main()
