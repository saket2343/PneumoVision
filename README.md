🚀 PneumoVision

Deep Learning–Based Pneumonia Detection from Chest X-Ray Images

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/EfficientNetB0-green" alt="EfficientNetB0">
  <img src="https://img.shields.io/badge/Streamlit-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

📖 Overview

PneumoVision is an end-to-end deep learning system for pneumonia detection from chest X-ray images.

The project combines EfficientNetB0 transfer learning, medical-image preprocessing, CLAHE enhancement, data augmentation, class-weighted training, two-stage fine-tuning, validation-based threshold optimization, patient-level leakage prevention, Grad-CAM explainability, and Streamlit deployment.

The complete pipeline is designed to move from raw chest X-ray images to an evaluated and deployable pneumonia classification system.

The system includes:

🩻 Binary NORMAL / PNEUMONIA classification

🧠 EfficientNetB0 transfer learning

🔄 Two-stage training and fine-tuning

🖼️ CLAHE-based image enhancement

🔀 Albumentations data augmentation

⚖️ Class-imbalance handling through class weights

🔍 Patient-level data leakage detection and cleaning

🎯 Validation-based threshold optimization

📊 Test-set performance evaluation

🔥 Grad-CAM explainability

🌐 Interactive Streamlit application

🏗️ Architecture

                         Chest X-Ray
                              │
                              ▼
                    Dataset Integrity Check
                              │
                              ▼
                       Preprocessing
                ┌─────────────┴─────────────┐
                │                           │
          RGB Conversion                Resize
                │                           │
                └─────────────┬─────────────┘
                              ▼
                           CLAHE
                              │
                              ▼
                    Data Augmentation
                       (Train Only)
                              │
                              ▼
                      EfficientNetB0
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 Stage 1              Stage 2
              Frozen Backbone       Fine-Tuning
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Pneumonia Score
                              │
                              ▼
                  Validation Threshold Tuning
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 NORMAL             PNEUMONIA
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         Grad-CAM
                              │
                              ▼
                     Streamlit Interface

✨ Features

🩻 Pneumonia Classification

Classifies chest X-rays into:

NORMAL
PNEUMONIA

🧠 Transfer Learning

Uses a pretrained EfficientNetB0 convolutional backbone to extract visual features from chest X-ray images.

🔄 Two-Stage Training

The training pipeline consists of:

Stage 1 — Frozen Backbone

The pretrained EfficientNetB0 feature extractor is frozen while the newly added classification head learns the pneumonia-specific decision boundary.

Stage 2 — Fine-Tuning

Selected layers of the pretrained backbone are unfrozen and fine-tuned using a lower learning rate so the learned representations can adapt to chest X-ray characteristics.

🖼️ Medical Image Preprocessing

Input X-Ray
     ↓
RGB Conversion
     ↓
Resize → 224 × 224
     ↓
CLAHE Enhancement
     ↓
Normalization
     ↓
Model

🔀 Data Augmentation

Training images are augmented using an Albumentations pipeline to improve robustness to variations in:

Spatial transformations

Contrast

Orientation

Imaging conditions

Augmentation is applied only to the training split.

⚖️ Class Imbalance Handling

The training pipeline computes class weights from the training distribution:

weight(class) =
N / (number_of_classes × N_class)

These weights are passed during training to reduce the effect of class imbalance.

🎯 Threshold Optimization

Instead of automatically using a 0.5 threshold, the model performs a validation-set threshold sweep and evaluates:

Accuracy

Precision

Recall

Sensitivity

Specificity

F1-score

Balanced Accuracy

MCC

Youden's J

The selected threshold is then applied to the untouched test set.

🔥 Grad-CAM Explainability

Grad-CAM generates heatmaps showing the image regions contributing to the model's prediction.

Chest X-Ray
     ↓
Model Prediction
     ↓
Grad-CAM
     ↓
Activation Map
     ↓
Heatmap Overlay

🔍 Dataset Integrity & Leakage Prevention

The project checks for:

Unreadable images

Exact duplicate images

Cross-split duplicate images

Patient-level overlap

Train/validation leakage

Train/test leakage

Validation/test leakage

📊 Dataset

The dataset is organized into three splits:

dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Cleaned Dataset Distribution

Split

NORMAL

PNEUMONIA

Total

Train

1,213

2,684

3,897

Validation

135

389

524

Test

234

390

624

Total

1,582

3,463

5,145

🧹 Data Cleaning & Leakage Prevention

Medical datasets may contain multiple X-rays belonging to the same patient. Splitting individual images without considering patient identity can cause leakage.

PneumoVision therefore performs patient-level validation before training.

Integrity Pipeline

Dataset
   │
   ▼
Unreadable File Check
   │
   ▼
Duplicate Detection
   │
   ▼
Cross-Split Duplicate Detection
   │
   ▼
Patient-Level Overlap Detection
   │
   ▼
Clean Dataset

Final Clean Dataset Checks

Train ↔ Validation patient overlap : 0
Train ↔ Test patient overlap       : 0
Validation ↔ Test patient overlap  : 0

Cross-split duplicate groups       : 0
Unreadable images                  : 0

Run the integrity check with:

python check_data_integrity.py

🧠 Model Architecture

Input Image
     │
     ▼
EfficientNetB0
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout
     │
     ▼
Dense Classification Layers
     │
     ▼
Sigmoid Output
     │
     ▼
Pneumonia Probability

Why EfficientNetB0?

EfficientNet provides a strong balance between:

Feature extraction capability

Model size

Computational cost

Transfer-learning performance

🔬 Training Strategy

Stage 1 — Feature Extraction

Pretrained EfficientNetB0
          │
          ├── Frozen
          │
          ▼
Classification Head
          │
          ▼
Binary Prediction

The classification head first learns to distinguish NORMAL and PNEUMONIA using the pretrained visual representations.

Stage 2 — Fine-Tuning

Pretrained EfficientNetB0
          │
          ▼
Selected Layers Unfrozen
          │
          ▼
Low Learning Rate
          │
          ▼
Fine-Tuned Model

Fine-tuning allows the network to adapt its learned visual features to the specific characteristics of chest X-ray images.

🎯 Threshold Selection

The validation set is used to determine an operating threshold.

Best Model
    │
    ▼
Validation Predictions
    │
    ▼
Threshold Sweep
    │
    ├── Accuracy
    ├── Precision
    ├── Sensitivity
    ├── Specificity
    ├── F1
    ├── Balanced Accuracy
    ├── MCC
    └── Youden's J
    │
    ▼
Selected Threshold
    │
    ▼
Held-Out Test Evaluation

Important

The test set is not used during threshold selection.

This prevents test-set information from influencing the final decision threshold.

📈 Evaluation

The final test set contains:

NORMAL     : 234
PNEUMONIA  : 390
TOTAL      : 624

The evaluation reports:

Metric

Description

Accuracy

Overall classification accuracy

Precision

Reliability of positive predictions

Sensitivity

Ability to detect pneumonia

Specificity

Ability to identify normal X-rays

F1-Score

Balance between precision and recall

ROC-AUC

Overall discrimination capability

Balanced Accuracy

Average class-wise recall

MCC

Correlation between predictions and labels

Evaluation Command

python evaluate.py \
  --model-path "models/best_model.keras" \
  --threshold <SELECTED_THRESHOLD>

Example:

python evaluate.py \
  --model-path "models/best_model.keras" \
  --threshold 0.85

Generated artifacts:

outputs/
├── reports/
│   └── test_evaluation_*.json
│
└── confusion_matrix/
    └── confusion_matrix_*.png

📂 Project Structure

Pneumonia-Detection/
│
├── dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   ├── best_model.keras
│   └── final_model.keras
│
├── outputs/
│   ├── reports/
│   └── confusion_matrix/
│
├── src/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── trainer.py
│   └── utils.py
│
├── tests/
│
├── app.py
├── train.py
├── evaluate.py
├── tune_threshold.py
├── calibrate.py
├── check_data_integrity.py
├── config.py
├── requirements.txt
└── README.md

🛠️ Tech Stack

Category

Technology

Language

Python

Deep Learning

TensorFlow / Keras

Backbone

EfficientNetB0

Image Processing

OpenCV

Augmentation

Albumentations

Data Pipeline

tf.data

Explainability

Grad-CAM

Deployment

Streamlit

Testing

Pytest

Model Format

Keras .keras

🚀 Installation

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Pneumonia-Detection

2. Create Virtual Environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Windows

python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

🏋️ Training

Run the complete training pipeline:

python train.py

Training Flow

Dataset Loading
      ↓
Preprocessing
      ↓
Augmentation
      ↓
Class Weight Calculation
      ↓
Stage 1 Training
      ↓
Stage 2 Fine-Tuning
      ↓
Validation Monitoring
      ↓
Best Model Checkpoint
      ↓
Final Model

Training Options

python train.py --epochs 50 --fine-tune-epochs 15

Skip fine-tuning:

python train.py --no-fine-tune

Disable dataset caching:

python train.py --no-cache

Specify the backbone:

python train.py --backbone EfficientNetB0

🎯 Threshold Tuning

After training:

python tune_threshold.py

The script generates:

outputs/reports/
├── threshold_search.csv
├── threshold_search.json
├── threshold_recommendation.json
├── threshold_curve.png
└── probability_distribution_val.png

🌐 Streamlit Application

Launch the application with:

streamlit run app.py

Application Flow

Upload X-Ray
     ↓
Preprocessing
     ↓
Model Inference
     ↓
Pneumonia Probability
     ↓
Optimized Threshold
     ↓
NORMAL / PNEUMONIA
     ↓
Grad-CAM Visualization

The Streamlit interface provides:

Image upload

Model prediction

Pneumonia probability

Classification result

Confidence score

Grad-CAM visualization

Prediction reporting

🧪 Testing

Run the test suite:

pytest tests/ -v

Run tests with coverage:

pytest tests/ -v --cov=src --cov-report=term-missing

📁 Generated Outputs

outputs/
│
├── reports/
│   ├── data_integrity_report.json
│   ├── threshold_search.csv
│   ├── threshold_search.json
│   ├── threshold_recommendation.json
│   ├── test_evaluation_*.json
│   ├── threshold_curve.png
│   └── probability_distribution_val.png
│
└── confusion_matrix/
    └── confusion_matrix_*.png

🔬 Experiments & Future Improvements

Potential extensions include:

EfficientNetB3/B4 experiments

Focal Loss

Oversampling comparison

Probability calibration

Model ensembles

External dataset validation

Additional explainability methods

Docker deployment

CI/CD integration

Cloud deployment

⚠️ Medical Disclaimer

This project is intended only for educational and research purposes.

PneumoVision is not a medical diagnostic device.

Predictions generated by the model should not replace evaluation by a qualified healthcare professional.

Grad-CAM visualizations are intended for model interpretability and should not be considered clinical evidence.

👨‍💻 Author

Saket Pandey

Deep learning project for automated pneumonia detection from chest X-ray images using transfer learning, medical-image preprocessing, model fine-tuning, explainability, and interactive deployment.

📄 License

This project is licensed under the MIT License.
