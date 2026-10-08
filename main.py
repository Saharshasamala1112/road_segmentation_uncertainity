"""Main entry point for road segmentation project."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(
        description="Uncertainty-Aware Road Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train

  # Run inference
  python main.py predict --image path/to/image.jpg --checkpoint outputs/models/best.pth

  # Launch Streamlit app
  python main.py app

  # Validate model
  python main.py validate --checkpoint outputs/models/best.pth
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/config.yaml")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--image", type=str, required=True)
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--config", type=str, default="configs/config.yaml")
    predict_parser.add_argument("--output", type=str, default="outputs/results/")
    predict_parser.add_argument("--mc-samples", type=int, default=30)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate the model")
    validate_parser.add_argument("--checkpoint", type=str, required=True)
    validate_parser.add_argument("--config", type=str, default="configs/config.yaml")

    # App command
    app_parser = subparsers.add_parser("app", help="Launch Streamlit app")

    args = parser.parse_args()

    if args.command == "train":
        from training.train import main as train_main
        train_main()

    elif args.command == "predict":
        from inference.predict import main as predict_main
        predict_main()

    elif args.command == "validate":
        from training.validate import main as validate_main
        validate_main()

    elif args.command == "app":
        import subprocess
        subprocess.run(
            ["streamlit", "run", str(Path(__file__).parent / "streamlit_app.py")]
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
