# Uncertainty-Aware Road Segmentation for Safe Autonomous Navigation in Indian Traffic

**MT 310 Autonomous Vehicle**  
**Assignment: Data Analytics in Autonomous Vehicles**  
**II Semester 2025-26**  
**Faculty of Science & Technology, IFHE Hyderabad**

---

## 🚗 Project Title
**Uncertainty-Aware Road Segmentation Model using India Driving Dataset (IDD)**

## 🎯 Problem Statement (Assignment Section 1)
Autonomous vehicles in unstructured Indian urban environments face significant challenges due to mixed traffic (cars, two-wheelers, auto-rickshaws), poor road markings, animals, potholes, and adverse weather.  

Traditional segmentation models often produce over-confident predictions in ambiguous situations, leading to safety risks.  

**This project develops a deep learning model that performs binary road segmentation and simultaneously estimates prediction uncertainty** (using Monte Carlo Dropout) to enable **risk-aware decision making** and safe fallback in chaotic Indian traffic conditions.

## 📊 Dataset (Assignment Section 3)
- **Name**: India Driving Dataset (IDD) Binary Segmentation  
- **Source**: Kaggle (real-world data collected in Hyderabad & Bangalore)  
- **Size**: ~6993 RGB images (512×512) with corresponding binary road masks  
- **Relevance**: Represents real unstructured Indian roads with poor lane discipline and mixed traffic

## 🧠 Model Architecture
- **Model**: Simple U-Net (Encoder-Decoder CNN)  
- **Uncertainty Estimation**: Monte Carlo Dropout  
- **Framework**: PyTorch  
- **Input**: RGB road image  
- **Output**: Binary road mask + uncertainty map

## ✨ Key Features
- Real Indian road data (not simulated)  
- Pixel-wise road segmentation  
- Uncertainty quantification for safety-critical decisions  
- Complete end-to-end pipeline in a single file  
- EDA, training, and visualization included

## 🛠️ Tech Stack
- **Language**: Python 3.11  
- **Deep Learning**: PyTorch  
- **Computer Vision**: OpenCV  
- **Data Handling**: NumPy, TorchVision  
- **Visualization**: Matplotlib  

## 📁 Project Structure

AV_IDD_Project/
├── data/raw/idd/
│   ├── image_archive/          ← IDD images
│   └── mask_archive/           ← IDD masks
├── main.py                     ← Complete single-file project (EDA + Training + Evaluation)
├── models/road_segmentation_model.pth
├── README.md
└── (other supporting files)


## 🚀 How to Run (Step-by-Step)

powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete project (EDA + Training + Evaluation)
python main.py

Note: Make sure your IDD images are placed in data/raw/idd/image_archive/ and masks in data/raw/idd/mask_archive/.
📈 Results & Insights (Assignment Section 7)

Exploratory Data Analysis with sample images and road coverage statistics
Model training with loss curve
Final evaluation showing Input Image, Ground Truth Mask, and Predicted Mask
Uncertainty estimation enables safer AV decisions in high-risk Indian traffic scenarios

📄 Assignment Sections Covered

Problem Definition
Dataset Selection
Exploratory Data Analysis (EDA)
Model Development
Model Evaluation
Results & Insights
Documentation & Presentation

📝 Future Work

Multi-class segmentation (vehicles, pedestrians, animals)
Real-time inference optimization
Integration with full AV pipeline

Author: Saharsha
Location: Telangana, India
Submission Date: 15 April 2026
