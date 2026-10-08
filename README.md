# Uncertainty-Aware Road Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/status-research--grade-2ea44f?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/architecture-CNN%20%2B%20Transformer-FF6B6B?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/uncertainty-MC%20Dropout-9B59B6?style=for-the-badge" alt="Uncertainty">
  <img src="https://img.shields.io/badge/license-MIT-27AE60?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>Probabilistic road perception for autonomous navigation in unstructured environments.</strong><br>
  Pixel-wise drivable area segmentation with calibrated uncertainty quantification.
</p>

---

## Overview

This project delivers a **hybrid CNN–Transformer segmentation model** that not only identifies drivable road regions but also estimates **prediction uncertainty** via Monte Carlo Dropout. Designed for the chaotic and unstructured nature of Indian traffic — mixed vehicles, absent lane discipline, occlusions, and extreme lighting variation — the system produces a confidence map alongside every segmentation, enabling downstream planners to make risk-aware decisions.

> **Core insight:** A segmentation model that knows when it might be wrong is safer than one that is always confidently wrong.

---

## Key Features

| Capability | Description | Status |
|---|---|---|
| **Road Segmentation** | Pixel-wise drivable area detection | Implemented |
| **Hybrid Architecture** | CNN encoder + Transformer global context + UNet decoder | Implemented |
| **Uncertainty Quantification** | Monte Carlo Dropout with variance-based confidence maps | Implemented |
| **Real-World Dataset** | Trained on IDD (Indian Driving Dataset) | Implemented |
| **Visualization Suite** | Segmentation overlays, uncertainty heatmaps, Grad-CAM | Implemented |
| **Robust Pipeline** | Handles occlusions, lighting shifts, and complex traffic scenes | Implemented |

---

## Motivation

Autonomous vehicles operating in Indian traffic face conditions that defy standard benchmarks:

- **Unstructured roads** — no clear lane markings, varying widths, informal traffic patterns
- **Mixed traffic** — cars, motorcycles, auto-rickshaws, pedestrians, and animals sharing the same space
- **Poor lane discipline** — lane changes without signaling, overtaking from either side
- **Environmental extremes** — harsh shadows, monsoon glare, dust, and nighttime driving

Traditional segmentation models output deterministic masks with no measure of confidence. When such a model encounters an unfamiliar scenario, it fails silently — producing a plausible-looking but incorrect segmentation. This project addresses that gap by coupling every prediction with an **uncertainty estimate**, flagging regions where the model lacks confidence so the planning stack can react conservatively.

---

## Architecture

```
Input Image (H x W x 3)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  Resize, Normalize, Augment
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   CNN Encoder       │  ResNet / EfficientNet backbone
│   (Local Features)  │  Multi-scale feature extraction
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Transformer       │  Self-attention for global context
│   (Global Context)  │  Captures long-range dependencies
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   UNet Decoder      │  Skip connections + upsampling
│   (Dense Prediction)│  Pixel-wise classification
└─────────┬───────────┘
          │
          ├──────────────────────────┐
          ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐
│  Segmentation   │      │  MC Dropout Head    │
│  Map (Mean)     │      │  T stochastic passes │
└─────────────────┘      └─────────┬───────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                   ┌────────────┐   ┌────────────┐
                   │    Mean    │   │  Variance  │
                   │ Prediction │   │  (Uncert.) │
                   └────────────┘   └────────────┘
```

### Model Components

| Module | Role | Details |
|---|---|---|
| **Encoder** | Local feature extraction | ResNet-50 / EfficientNet-B4 backbone, pretrained on ImageNet |
| **Transformer** | Global context modeling | Multi-head self-attention over spatial tokens |
| **Decoder** | Dense prediction | UNet-style upsampling with skip connections |
| **MC Dropout Head** | Uncertainty estimation | T forward passes with dropout enabled; mean and variance computed |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Framework | PyTorch 2.x | Deep learning engine |
| Language | Python 3.10+ | Core implementation |
| Vision | OpenCV | Image I/O and preprocessing |
| Augmentation | Albumentations | Data augmentation pipeline |
| Visualization | Matplotlib, Seaborn | Plots and heatmaps |
| UI | Streamlit | Interactive demo application |
| Experiment Tracking | TensorBoard | Training monitoring |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- 8 GB+ RAM

### Setup

```bash
git clone https://github.com/Saharshasamala1112/road_segmentation_uncertainity.git
cd road_segmentation_uncertainity

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python training/train.py --config configs/config.yaml
```

### Inference

```bash
python inference/predict.py --image path/to/image.jpg --output outputs/
```

### Interactive Demo

```bash
streamlit run streamlit_app.py
```

### Webcam Real-Time Detection

```bash
python inference/webcam_demo.py
```

---

## Results

| Metric | Value |
|---|---|
| IoU (Intersection over Union) | ~75% |
| Dice Score | ~0.80 |
| Pixel Accuracy | ~88% |
| Precision | ~85% |
| Recall | ~83% |

> Metrics are reported on the IDD validation split. Actual performance varies with scene complexity and lighting conditions.

---

## Project Structure

```
road_segmentation_uncertainity/
│
├── data/
│   ├── raw/
│   │   ├── images/              # Original dataset images
│   │   └── masks/               # Ground truth segmentation masks
│   ├── processed/
│   │   ├── train/
│   │   └── val/
│   └── splits/                  # train.txt, val.txt
│
├── models/
│   ├── encoder.py               # ResNet / EfficientNet backbone
│   ├── transformer.py           # Self-attention module
│   ├── decoder.py               # UNet upsampling layers
│   └── model.py                 # Full hybrid model definition
│
├── utils/
│   ├── dataset.py               # PyTorch Dataset and DataLoader
│   ├── augmentations.py         # Albumentations pipeline
│   ├── losses.py                # Loss functions (Dice, BCE, combined)
│   ├── metrics.py               # IoU, Dice, Accuracy
│   └── visualization.py         # Plot and heatmap helpers
│
├── training/
│   ├── train.py                 # Training loop
│   └── validate.py              # Validation and evaluation
│
├── inference/
│   ├── predict.py               # MC Dropout inference
│   ├── visualize_results.py     # Static result visualization
│   ├── webcam_demo.py           # Real-time webcam inference
│   └── gradcam.py               # Explainability via Grad-CAM
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
│
├── outputs/
│   ├── models/                  # Saved model checkpoints
│   ├── results/                 # Prediction outputs
│   └── logs/                    # Training logs
│
├── configs/
│   └── config.yaml              # Hyperparameters and paths
│
├── streamlit_app.py             # Interactive web application
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Uncertainty Quantification

The model employs **Monte Carlo Dropout** — a practical approximation to Bayesian neural networks — to estimate epistemic uncertainty:

1. At inference time, dropout layers remain **active** across T stochastic forward passes.
2. The **mean** of all passes produces the final segmentation.
3. The **variance** across passes produces an uncertainty heatmap.

High-variance regions indicate areas where the model is uncertain — typically boundaries, occlusions, or unfamiliar patterns. These regions can be flagged for conservative planning or human intervention.

---

## Roadmap

- [ ] Real-time optimization (TensorRT / ONNX export)
- [ ] Multi-class segmentation (road, sidewalk, vehicle, pedestrian)
- [ ] Video-based temporal modeling
- [ ] Edge deployment (Jetson / mobile)
- [ ] Active learning pipeline for targeted data collection
- [ ] Calibration analysis (reliability diagrams, ECE)

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **IDD Dataset** — Indian Driving Dataset for road scene understanding
- **PyTorch** community for the deep learning framework
- **OpenCV** for computer vision utilities
- **Albumentations** for the augmentation pipeline

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{road_seg_uncertainty,
  author = {Saharshasamala1112},
  title = {Uncertainty-Aware Road Segmentation},
  year = {2024},
  url = {https://github.com/Saharshasamala1112/road_segmentation_uncertainity}
}
```

---

<p align="center">
  <em>This model does not merely detect roads — it understands the limits of its own knowledge.</em>
</p>
