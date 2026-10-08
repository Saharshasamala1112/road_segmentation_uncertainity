"""Grad-CAM explainability for road segmentation model."""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int = 0) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class index

        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # For binary segmentation, use the single output channel
        if output.shape[1] == 1:
            score = output[0, 0].sum()
        else:
            score = output[0, target_class].sum()

        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


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


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for road segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/results/gradcam.png", help="Output path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Load and preprocess image
    image_size = tuple(config["data"].get("image_size", [512, 512]))
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image_float = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_float - mean) / std
    input_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Find target layer (last conv in decoder)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in model")

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)

    # Overlay on original image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * image + 0.4 * heatmap_color).astype(np.uint8)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
