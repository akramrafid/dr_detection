# Diabetic Retinopathy Detection System

An AI-powered system for automated grading of diabetic retinopathy severity from retinal fundus images. Built with PyTorch, FastAPI, and Streamlit. The system uses EfficientNet-B5 with Grad-CAM++ explainability to provide clinically interpretable predictions.

**Course:** Python Based Project Development | **Group:** 09 | **Semester:** 10th

**Team Members:** Ovick Hassan, Shah Md. Imtiaz Chowdhury, Akram Rafid, Muntasir Adnan Eram

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Explainability](#explainability)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)

---
<img width="1919" height="1041" alt="image" src="https://github.com/user-attachments/assets/3ac77ec8-1846-4558-9c47-a6e7f8f7c15e" />
<img width="1920" height="1439" alt="screencapture-localhost-8502-2026-04-12-11_23_30" src="https://github.com/user-attachments/assets/a06e7fff-e3d9-46e6-89d1-f53ff75a5a22" />


## Overview

Diabetic Retinopathy (DR) is a leading cause of blindness worldwide. Early detection through retinal screening is critical but requires expert ophthalmologists, which creates bottlenecks in healthcare systems. This project automates DR severity grading into five levels using deep learning:

| Grade | Severity        | Clinical Action                          |
|-------|-----------------|------------------------------------------|
| 0     | No DR           | Continue regular annual screening        |
| 1     | Mild NPDR       | Follow-up in 12 months                   |
| 2     | Moderate NPDR   | Follow-up in 6 months, monitor blood sugar |
| 3     | Severe NPDR     | Urgent referral within 1 month           |
| 4     | Proliferative DR| Immediate referral required              |

The system treats this as an **ordinal regression** problem (not classification), outputting a continuous score in [0, 4] that is rounded to the nearest grade. This better captures the ordinal nature of DR severity.

---

## Key Features

- **EfficientNet-B5:** Pretrained backbone with custom regression head for ordinal DR grading
- **Ben Graham Preprocessing:** CLAHE enhancement, Gaussian illumination normalization, and circular masking
- **Grad-CAM++ Explainability:** Visual heatmaps showing exactly where the model focuses on the retina
- **Class Imbalance Handling:** Weighted random sampling to address skewed grade distribution
- **Mixed Precision Training:** FP16 training with gradient scaling for efficient GPU utilization
- **REST API:** FastAPI backend for production-ready inference
- **Interactive UI:** Streamlit dashboard for clinical image upload and analysis
- **Cloud Deployable:** Includes Procfile for Heroku/Render deployment

---

## Project Structure

```
dr_detection/
|
|-- app/
|   |-- main.py                 # FastAPI REST API server
|   |-- streamlit_app.py        # Streamlit web UI
|
|-- src/
|   |-- preprocessing.py        # Ben Graham preprocessing + train/val/test splits
|   |-- dataset.py              # PyTorch Dataset, DataLoader, augmentation pipelines
|   |-- models.py               # EfficientNet-B5 model definition
|   |-- train.py                # Training loop with early stopping and scheduling
|   |-- ensemble.py             # Model evaluation and metrics
|   |-- explainability.py       # Grad-CAM++ heatmap generation
|
|-- notebooks/
|   |-- 01_eda.ipynb            # Exploratory data analysis
|
|-- data/
|   |-- raw/                    # Original APTOS 2019 dataset
|   |-- processed/              # Ben Graham preprocessed images (456x456)
|   |-- splits/                 # Train/val/test split CSV
|
|-- checkpoints/
|   |-- efficientnet_b5_best.pth  # Best EfficientNet-B5 checkpoint
|
|-- outputs/
|   |-- class_distribution.png    # Class distribution visualization
|   |-- sample_images.png         # Sample preprocessed images
|   |-- training_history.png      # Loss and QWK curves
|   |-- confusion_matrix.png      # Confusion matrix
|   |-- efficientnet_b5_history.json
|   |-- evaluation_results.json   # Model evaluation results
|   |-- gradcam/                  # Grad-CAM++ heatmap outputs
|
|-- requirements.txt
|-- Procfile                     # Cloud deployment config
|-- .gitignore
|-- README.md
```

---

## Dataset

**Source:** [APTOS 2019 Blindness Detection Challenge](https://www.kaggle.com/c/aptos2019-blindness-detection) (Kaggle)

- **Total images:** 3,662 retinal fundus photographs
- **Format:** PNG, variable resolution
- **Labels:** 5 DR severity grades (0-4)
- **Split strategy:** Stratified 70/15/15 (train/val/test) to maintain grade distribution

The dataset exhibits significant class imbalance with Grade 0 (No DR) being the most common and Grade 3 (Severe) being the rarest. We address this with weighted random sampling during training.

---

## Preprocessing Pipeline

All images go through the **Ben Graham preprocessing pipeline**, used by top APTOS competition solutions:

1. **Dark border cropping:** Removes black borders around the fundus circle
2. **Resizing:** Standardizes all images to 456x456 pixels
3. **CLAHE enhancement:** Contrast Limited Adaptive Histogram Equalization on the green channel to enhance retinal vessel visibility
4. **Gaussian illumination normalization:** `4 * image - 4 * GaussianBlur(image) + 128` removes uneven illumination while preserving local contrast
5. **Circular masking:** Removes everything outside the circular fundus area

**Run preprocessing:**
```bash
python src/preprocessing.py
```

This processes all raw images and creates stratified train/val/test splits saved to `data/splits/splits.csv`.

---

## Model Architecture

### EfficientNet-B5

- **Backbone:** `timm` pretrained EfficientNet-B5 (ImageNet weights)
- **Input size:** 456x456 pixels
- **Feature dimension:** 2048
- **Head:** Dropout(0.3) -> Linear(2048, 256) -> ReLU -> Dropout(0.15) -> Linear(256, 1) -> Sigmoid
- **Output:** Sigmoid scaled to [0, 4] range
- **Strengths:** Excellent at detecting local features like microaneurysms and hemorrhages

The model uses:
- **Ordinal regression** (not classification) with Sigmoid output scaled to [0, 4]
- **Custom regression head** replacing the default classifier

---

## Training Strategy

| Hyperparameter         | EfficientNet-B5 |
|------------------------|-----------------|
| Image size             | 456x456         |
| Batch size             | 16              |
| Learning rate          | 1e-4            |
| Optimizer              | AdamW           |
| Weight decay           | 1e-5            |
| Scheduler              | CosineAnnealingWarmRestarts (T0=10) |
| Loss function          | SmoothL1 (Huber)|
| Early stopping patience| 7 epochs        |
| Max epochs             | 25              |
| Mixed precision        | FP16            |
| Gradient clipping      | max_norm=1.0    |

**Data Augmentation (training only):**
- Horizontal/vertical flip
- Random 90-degree rotation
- Random rotation (up to 30 degrees)
- Random brightness/contrast adjustment
- Hue/saturation/value shifts
- Grid distortion
- Coarse dropout

**Validation/test:** Only resize and ImageNet normalization (no augmentation).

**Primary metric:** Quadratic Weighted Kappa (QWK) - the official APTOS competition metric that penalizes predictions further from the true grade more heavily.

**Run training:**
```bash
python src/train.py
```

---

## Explainability

The system uses **Grad-CAM++** to generate visual heatmaps showing which regions of the retina the model focuses on when making predictions.

- **EfficientNet-B5:** Target layer is `conv_head` (last convolutional layer)

Heatmaps are overlaid on the original image using a JET colormap, making it easy for clinicians to verify the model's reasoning.

**Generate heatmaps:**
```bash
python src/explainability.py
```

---

## Results

| Metric                 | Value          |
|------------------------|----------------|
| Best Val QWK           | 0.9089         |
| Best Epoch             | 8 / 15         |
| Early Stopping         | Epoch 15       |
| Total Parameters       | 28,865,585     |
| Training Time/Epoch    | ~110s          |
| GPU                    | RTX 3060 (12.9 GB VRAM) |

The model achieves strong agreement with ophthalmologist-level grading, with a QWK score above 0.90 indicating near-expert performance.

---

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended: 6GB+ VRAM)
- [APTOS 2019 dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data) downloaded to `data/raw/`

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dr_detection.git
   cd dr_detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install `timm` and `albumentations`:**
   ```bash
   pip install timm albumentations
   ```

5. **Download the APTOS 2019 dataset** from Kaggle and place it in:
   ```
   data/raw/aptos2019-blindness-detection/
       train_images/
       train.csv
   ```

---

## Usage

### Step 1: Preprocess Images

```bash
python src/preprocessing.py
```

This will:
- Apply Ben Graham preprocessing to all 3,662 images
- Save processed images to `data/processed/train_images/`
- Create stratified train/val/test splits at `data/splits/splits.csv`

### Step 2: Train Model

```bash
python src/train.py
```

Trains EfficientNet-B5 with early stopping. Best checkpoint is saved to `checkpoints/`.

### Step 3: Evaluate Model

```bash
python src/ensemble.py
```

Runs the model on the test set and generates evaluation outputs including confusion matrix and classification report.

### Step 4: Generate Grad-CAM++ Heatmaps

```bash
python src/explainability.py
```

### Step 5: Launch the Application

**Start the FastAPI backend:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Start the Streamlit frontend (in a separate terminal):**
```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser to use the web interface.

---

## API Reference

### `POST /predict`

Upload a retinal fundus image for DR grading.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file - PNG, JPG, JPEG)

**Response:**
```json
{
    "grade": 2,
    "grade_label": "Moderate DR",
    "grade_color": "#E67E22",
    "raw_score": 1.823,
    "confidence": 0.823,
    "advice": "Moderate NPDR detected. Schedule follow-up in 6 months.",
    "heatmap_b64": "<base64 encoded Grad-CAM++ heatmap>",
    "original_b64": "<base64 encoded preprocessed image>"
}
```

### `GET /health`

Check API status.

**Response:**
```json
{
    "status": "running",
    "device": "cuda",
    "model": "EfficientNet-B5"
}
```

### `GET /`

Root endpoint.

**Response:**
```json
{
    "message": "Diabetic Retinopathy Detection API is running!"
}
```

---

## Deployment

The project includes a `Procfile` for cloud deployment on platforms like Heroku or Render:

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Make sure to upload the model checkpoint (`checkpoints/efficientnet_b5_best.pth`) to your deployment environment or use cloud storage.

---

## Technologies Used

| Category        | Technology                                      |
|-----------------|--------------------------------------------------|
| Deep Learning   | PyTorch 2.1, timm                                |
| Model           | EfficientNet-B5                                  |
| Preprocessing   | OpenCV, Albumentations                           |
| Explainability  | pytorch-grad-cam (Grad-CAM++)                    |
| API             | FastAPI, Uvicorn                                 |
| Frontend        | Streamlit                                        |
| Data Science    | NumPy, Pandas, scikit-learn                      |
| Visualization   | Matplotlib, Seaborn                              |
| Metrics         | Quadratic Weighted Kappa (Cohen's Kappa)         |
| Deployment      | Heroku/Render (Procfile)                         |

---

## License

This project is developed for academic purposes as part of the Python Based Project Development course (10th Semester).

---
