"""Real-time webcam inference with MC Dropout uncertainty visualization."""

import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel


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


def preprocess_frame(frame: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    """Preprocess a webcam frame for inference."""
    frame = cv2.resize(frame, (image_size[1], image_size[0]))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    frame = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float()
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time road segmentation with uncertainty")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mc-samples", type=int, default=10, help="Number of MC Dropout samples")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image_size = tuple(config["data"].get("image_size", [512, 512]))

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")

    print("Press 'q' to quit, 's' to save current frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_tensor = preprocess_frame(frame, image_size).to(device)

        # Predict with MC Dropout
        mean_pred, uncertainty = model.predict_with_uncertainty(input_tensor, args.mc_samples)

        # Convert to numpy
        pred_mask = (mean_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        unc_map = uncertainty.squeeze().cpu().numpy()
        unc_map = (unc_map / unc_map.max() * 255).astype(np.uint8) if unc_map.max() > 0 else unc_map

        # Resize back to original frame size
        h, w = frame.shape[:2]
        pred_mask = cv2.resize(pred_mask, (w, h))
        unc_map = cv2.resize(unc_map, (w, h))

        # Create overlay
        overlay = frame.copy()
        overlay[pred_mask > 0] = (0.4 * overlay[pred_mask > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)

        # Uncertainty heatmap
        unc_color = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
        unc_overlay = cv2.addWeighted(frame, 0.6, unc_color, 0.4, 0)

        # Display
        cv2.imshow("Segmentation", overlay)
        cv2.imshow("Uncertainty", unc_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("outputs/results/webcam_frame.png", frame)
            cv2.imwrite("outputs/results/webcam_segmentation.png", overlay)
            cv2.imwrite("outputs/results/webcam_uncertainty.png", unc_overlay)
            print("Frame saved to outputs/results/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
