"""FastAPI backend for uncertainty-aware road segmentation."""

import io
import base64
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model import RoadSegmentationModel

app = FastAPI(title="Road Segmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model = None
device = None
config = None


def load_model(checkpoint_path: str, cfg: dict):
    """Load the segmentation model."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = cfg["model"]
    model = RoadSegmentationModel(
        encoder=model_config.get("encoder", "resnet50"),
        pretrained=False,
        transformer_dim=model_config.get("transformer_dim", 512),
        transformer_heads=model_config.get("transformer_heads", 8),
        transformer_layers=model_config.get("transformer_layers", 4),
        dropout=model_config.get("dropout_rate", 0.5),
        num_classes=model_config.get("num_classes", 1),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(__file__).resolve().parent.parent / "outputs" / "models" / "best.pth"
    load_model(str(checkpoint_path), config)
    print(f"Model loaded on {device}")


def preprocess_image(image_bytes: bytes, image_size: tuple[int, int]) -> torch.Tensor:
    """Preprocess uploaded image bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((image_size[1], image_size[0]))
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image


def tensor_to_base64(tensor: np.ndarray, is_heatmap: bool = False) -> str:
    """Convert numpy array to base64 encoded PNG."""
    if is_heatmap:
        # Apply colormap for heatmap
        tensor = (tensor / tensor.max() * 255).astype(np.uint8) if tensor.max() > 0 else (tensor * 255).astype(np.uint8)
        tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
    else:
        tensor = (tensor * 255).astype(np.uint8)

    image = Image.fromarray(tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mc_samples: int = Form(30),
):
    """Run segmentation with MC Dropout uncertainty."""
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    # Read and preprocess image
    image_bytes = await file.read()
    image_size = tuple(config["data"].get("image_size", [512, 512]))
    input_tensor = preprocess_image(image_bytes, image_size).to(device)

    # Run inference
    mean_pred, uncertainty = model.predict_with_uncertainty(input_tensor, mc_samples)

    # Convert to numpy
    mean_pred = mean_pred.squeeze().cpu().numpy()
    uncertainty = uncertainty.squeeze().cpu().numpy()
    binary_mask = (mean_pred > 0.5).astype(np.float32)

    # Create overlay
    original = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((image_size[1], image_size[0])))
    original = original.astype(np.float32) / 255.0
    overlay = (original * 255).astype(np.uint8).copy()
    overlay[binary_mask > 0] = (0.4 * overlay[binary_mask > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)

    # Encode results
    result = {
        "segmentation": tensor_to_base64(binary_mask),
        "uncertainty": tensor_to_base64(uncertainty, is_heatmap=True),
        "overlay": tensor_to_base64(overlay),
        "metrics": {
            "mean_confidence": float(mean_pred.mean()),
            "max_uncertainty": float(uncertainty.max()),
            "mean_uncertainty": float(uncertainty.mean()),
            "road_coverage": float(binary_mask.mean() * 100),
        },
    }

    return JSONResponse(result)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "device": str(device)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
