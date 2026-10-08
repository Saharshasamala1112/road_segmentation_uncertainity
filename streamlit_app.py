"""Interactive Streamlit application for uncertainty-aware road segmentation."""

import streamlit as st
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
import sys
import tempfile

sys.path.append(str(Path(__file__).resolve().parent))

from models.model import RoadSegmentationModel
from inference.predict import load_model, preprocess_image, predict

st.set_page_config(
    page_title="Uncertainty-Aware Road Segmentation",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_cached_model(checkpoint_path: str, config_path: str):
    """Cache the model to avoid reloading on every interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    return model, device


def main():
    st.title("Uncertainty-Aware Road Segmentation")
    st.markdown("""
    Upload an image to segment drivable road regions with uncertainty quantification.
    The model uses **Monte Carlo Dropout** to estimate prediction confidence.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="outputs/models/best.pth",
            help="Path to the trained model checkpoint",
        )

        config_path = st.text_input(
            "Config Path",
            value="configs/config.yaml",
            help="Path to the configuration file",
        )

        mc_samples = st.slider(
            "MC Dropout Samples",
            min_value=1,
            max_value=100,
            value=30,
            help="Number of stochastic forward passes for uncertainty estimation",
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Threshold for binary segmentation",
        )

        uncertainty_threshold = st.slider(
            "Uncertainty Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="Threshold for flagging uncertain regions",
        )

    # Main content
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image of a road scene",
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display original
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        # Run inference
        with st.spinner("Running inference..."):
            try:
                # Load config
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Load model
                model, device = load_cached_model(checkpoint_path, config_path)

                # Save uploaded image to temp file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    tmp_path = tmp.name

                # Preprocess
                image_size = tuple(config["data"].get("image_size", [512, 512]))
                image_tensor = preprocess_image(tmp_path, image_size)

                # Predict
                mean_pred, uncertainty, binary_mask = predict(model, image_tensor, mc_samples)

                # Resize to original
                h, w = image.shape[:2]
                pred_resized = cv2.resize(binary_mask, (w, h))
                unc_resized = cv2.resize(uncertainty, (w, h))

                # Create overlay
                overlay = image.copy()
                overlay[pred_resized > 0] = (
                    0.4 * overlay[pred_resized > 0] + 0.6 * np.array([0, 255, 0])
                ).astype(np.uint8)

                # Uncertainty overlay
                unc_color = cv2.applyColorMap(
                    (unc_resized / unc_resized.max() * 255).astype(np.uint8),
                    cv2.COLORMAP_JET,
                )
                unc_color = cv2.cvtColor(unc_color, cv2.COLOR_BGR2RGB)
                unc_overlay = (0.6 * image + 0.4 * unc_color).astype(np.uint8)

                # Results
                with col2:
                    st.subheader("Segmentation Result")
                    st.image(overlay, use_container_width=True)

                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Uncertainty Heatmap")
                    st.image(unc_overlay, use_container_width=True)

                with col4:
                    st.subheader("Metrics")
                    st.metric("Mean Confidence", f"{mean_pred.mean():.4f}")
                    st.metric("Max Uncertainty", f"{uncertainty.max():.4f}")
                    st.metric("Mean Uncertainty", f"{uncertainty.mean():.4f}")
                    st.metric("Road Coverage", f"{binary_mask.mean() * 100:.1f}%")

                    # Uncertainty warning
                    high_unc_ratio = (unc_resized > uncertainty_threshold).mean()
                    if high_unc_ratio > 0.3:
                        st.warning(
                            f"High uncertainty detected in {high_unc_ratio * 100:.1f}% of the image. "
                            "Proceed with caution."
                        )

            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                st.info(
                    "Make sure you have a trained model checkpoint. "
                    "Run `python training/train.py` first."
                )


if __name__ == "__main__":
    main()
