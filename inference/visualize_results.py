"""Visualize saved prediction results."""

import argparse
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.visualization import visualize_prediction


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation results")
    parser.add_argument("--image", type=str, required=True, help="Path to original image")
    parser.add_argument("--prediction", type=str, required=True, help="Path to prediction mask")
    parser.add_argument("--uncertainty", type=str, default=None, help="Path to uncertainty map")
    parser.add_argument("--output", type=str, default="outputs/results/visualization.png", help="Output path")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # Load prediction
    prediction = cv2.imread(args.prediction, cv2.IMREAD_GRAYSCALE)
    prediction = prediction.astype(np.float32) / 255.0

    # Load uncertainty if provided
    uncertainty = None
    if args.uncertainty:
        uncertainty = cv2.imread(args.uncertainty, cv2.IMREAD_GRAYSCALE)
        uncertainty = uncertainty.astype(np.float32) / 255.0

    # Visualize
    visualize_prediction(
        image=image,
        mask=prediction,
        prediction=prediction,
        uncertainty=uncertainty,
        save_path=args.output,
    )

    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()
