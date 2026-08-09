🫁 PneumoVision

AI-Powered Pneumonia Detection from Chest X-Rays

<p>
<img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Backbone-EfficientNet--B0-green?style=flat-square">
<img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">
</p>

PneumoVision is an end-to-end deep learning system for binary pneumonia detection from chest X-ray images. It uses an ImageNet-pretrained EfficientNet-B0, a two-stage training strategy, validation-based threshold optimization, Grad-CAM explainability, error analysis, and a Streamlit inference application.

⚠️ Medical Disclaimer: This is a research/portfolio project and is not intended for clinical diagnosis or treatment decisions.

📌 Overview

The system takes a chest X-ray as input and produces:

NORMAL / PNEUMONIA classification

Pneumonia probability

Threshold-based final decision

Grad-CAM visual explanation

Optional Test-Time Augmentation

Downloadable prediction report through the Streamlit application

The complete workflow is:

Chest X-Ray
     │
     ▼
Image Validation
     │
     ▼
Preprocessing
 ├── Resize 224 × 224
 ├── RGB Conversion
 ├── CLAHE
 └── Normalization
     │
     ▼
Training Augmentation
     │
     ▼
EfficientNet-B0
     │
     ├── Stage 1: Frozen Backbone
     │              50 Epochs
     │
     └── Stage 2: Fine-Tuning
                    15 Epochs
     │
     ▼
Pneumonia Probability
     │
     ▼
Validation-Based Threshold
     │
     ├───────────────┐
     ▼               ▼
Prediction        Grad-CAM
     │               │
     └───────┬───────┘
             ▼
       Error Analysis
             │
             ▼
      Streamlit App

🏗️ Architecture

High-Level Architecture

                         ┌─────────────────────┐
                         │     Chest X-Ray     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Dataset / Loader   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │       Preprocessing         │
                    │                             │
                    │  Resize → RGB → CLAHE       │
                    │       → Normalize            │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     Data Augmentation       │
                    │       Training Only          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       EfficientNet-B0        │
                    │      ImageNet Pretrained     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Classification Head     │
                    │                             │
                    │ GAP → Dropout(0.4)           │
                    │ → Dense(256) → BN → Swish    │
                    │ → Dropout(0.3)               │
                    │ → Dense(128) → BN → Swish    │
                    │ → Dense(1) → Sigmoid         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                         P(PNEUMONIA | X)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             Thresholding                     Grad-CAM
                    │                             │
                    ▼                             ▼
          NORMAL / PNEUMONIA              Heatmap Overlay
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                         Streamlit Application

🧠 Model Training Architecture

Training is performed in two stages.

Stage 1 — Feature Extraction

EfficientNet-B0
     │
     ├── Backbone → FROZEN
     │
     └── Classification Head → TRAINABLE
                                │
                                ▼
                           50 Epochs
                           LR = 1e-3

The pretrained backbone remains frozen while the classification head learns the pneumonia classification task.

Stage 2 — Fine-Tuning

EfficientNet-B0
     │
     ├── Earlier Layers → FROZEN
     │
     └── Last ~40 Layers → TRAINABLE
                            │
                            ▼
                       15 Epochs
                       LR = 1e-5

The final stage adapts high-level pretrained features to chest X-ray patterns using a much smaller learning rate.

🔄 Code Flow

The project separates data processing, model construction, training, evaluation, inference, and deployment.

1. Training Flow

Entry point:

train.py

Flow:

train.py
   │
   ├── Load configuration
   │
   ├── Set random seeds
   │
   ├── Validate dataset
   │
   ├── Create train / validation datasets
   │
   ├── Apply preprocessing
   │
   ├── Apply training augmentation
   │
   ├── Calculate class weights
   │
   ├── Build EfficientNet model
   │
   ├── Stage 1
   │     └── Train classification head
   │
   ├── Stage 2
   │     └── Fine-tune last backbone layers
   │
   ├── Monitor validation ROC-AUC
   │
   └── Save best model
            │
            ▼
   models/best_model.keras

2. Threshold Optimization Flow

Entry point:

tune_threshold.py

Best Model
    │
    ▼
Validation Dataset
    │
    ▼
Generate Probabilities
    │
    ▼
Search Thresholds
0.01 → 0.99
    │
    ▼
Find Threshold
with Sensitivity ≥ 95%
    │
    ▼
Choose Highest Specificity
    │
    ▼
Save Recommendation
    │
    ▼
outputs/reports/
threshold_recommendation.json

The test set is not used during threshold selection.

3. Evaluation Flow

Entry point:

evaluate.py

Best Model
    │
    ▼
Frozen Validation Threshold
    │
    ▼
Held-Out Test Set
    │
    ▼
Predictions
    │
    ├── Accuracy
    ├── Precision
    ├── Recall / Sensitivity
    ├── Specificity
    ├── F1
    ├── Balanced Accuracy
    ├── MCC
    └── ROC-AUC
    │
    ▼
Confusion Matrix
ROC Curve
PR Curve
Error Analysis

4. Single Image Inference Flow

Entry point:

predict.py

X-Ray Image
    │
    ▼
Load Image
    │
    ▼
Preprocess
    │
    ▼
Optional TTA
    │
    ▼
Model Prediction
    │
    ▼
Pneumonia Probability
    │
    ▼
Recommended Threshold
    │
    ├───────────────┐
    ▼               ▼
NORMAL          PNEUMONIA
    │               │
    └───────┬───────┘
            ▼
         Grad-CAM

5. Streamlit Application Flow

Entry point:

app.py

User Uploads X-Ray
        │
        ▼
    app.py
        │
        ▼
Inference Engine
        │
        ├── Preprocessing
        ├── Model Prediction
        ├── Thresholding
        └── Grad-CAM
        │
        ▼
Display:
 ├── Prediction
 ├── Probability
 ├── Threshold
 ├── Heatmap
 └── PDF Report

📂 Folder Structure

PneumoVision/
│
├── 📂 dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   │
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   │
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_ModelTraining.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_GradCAM.ipynb
│
├── 📂 src/
│   ├── preprocessing.py
│   ├── augmentations.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── threshold_optimizer.py
│   ├── calibration.py
│   ├── experiment_tracker.py
│   ├── gradcam.py
│   ├── error_analysis.py
│   ├── inference.py
│   └── utils.py
│
├── 📂 tests/
│   ├── test_model.py
│   ├── test_model_preprocessing.py
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   ├── test_dataset_balancing_and_leakage.py
│   └── test_threshold_optimizer.py
│
├── 📂 models/
│   └── best_model.keras
│
├── 📂 outputs/
│   ├── confusion_matrix/
│   ├── roc/
│   ├── pr_curve/
│   ├── heatmaps/
│   ├── error_analysis/
│   └── reports/
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── tune_threshold.py
├── calibrate.py
├── check_data_integrity.py
├── compare_models.py
├── tune.py
├── export_model.py
├── config.py
├── requirements.txt
└── LICENSE

📌 Important Files

File

Purpose

train.py

Main model training pipeline

predict.py

Single-image inference

evaluate.py

Final test-set evaluation

tune_threshold.py

Validation-based threshold selection

calibrate.py

Probability calibration

app.py

Streamlit application

config.py

Central configuration

src/model.py

Model architecture

src/dataset.py

Dataset loading

src/preprocessing.py

Image preprocessing

src/augmentations.py

Training augmentation

src/trainer.py

Training logic

src/evaluator.py

Evaluation utilities

src/gradcam.py

Grad-CAM generation

src/error_analysis.py

False-positive/negative analysis

src/inference.py

Inference engine

check_data_integrity.py

Dataset validation

compare_models.py

Backbone comparison

tune.py

Hyperparameter tuning

⚙️ Configuration

Main configuration is centralized in:

config.py

Important defaults:

Backbone        = EfficientNet-B0
Input Size      = 224 × 224
Batch Size      = 32
Stage 1         = 50 epochs
Stage 2         = 15 epochs
Stage 1 LR      = 1e-3
Stage 2 LR      = 1e-5
Optimizer       = AdamW
Weight Decay    = 1e-4
Seed            = 42

🚀 Commands

Installation

Clone Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PneumoVision

Create Virtual Environment

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

Windows:

python -m venv .venv
.venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

🔍 Check Dataset

python check_data_integrity.py

🏋️ Train Model

Default two-stage training:

python train.py

Short Experiment

python train.py \
    --epochs 30 \
    --fine-tune-epochs 10

Without Fine-Tuning

python train.py --no-fine-tune

Oversampling

python train.py \
    --oversample \
    --oversample-ratio 0.5 \
    --no-class-weights

Focal Loss

python train.py \
    --loss focal \
    --focal-gamma 2.0 \
    --focal-alpha 0.25

🎯 Tune Threshold

python tune_threshold.py

The selected threshold is saved under:

outputs/reports/

📊 Evaluate Model

python evaluate.py \
    --model-path models/best_model.keras \
    --threshold <recommended-threshold>

🔮 Predict One Image

python predict.py \
    --image path/to/xray.jpg

With Test-Time Augmentation

python predict.py \
    --image path/to/xray.jpg \
    --tta

Save Grad-CAM

python predict.py \
    --image path/to/xray.jpg \
    --save-heatmap outputs/heatmaps/example.png

🌡️ Calibrate Probabilities

python calibrate.py \
    --model-path models/best_model.keras

🌐 Run Streamlit App

streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

🧪 Run Tests

pytest tests/ -v

With coverage:

pytest tests/ -v --cov=src

🧹 Code Quality

black --check .

flake8 .

🔬 Compare Backbones

python compare_models.py \
    --backbones EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3

🧠 Hyperparameter Tuning

python tune.py

🔗 Complete Command Flow

For a complete fresh experiment, run:

# 1. Validate dataset
python check_data_integrity.py

# 2. Train
python train.py

# 3. Select validation threshold
python tune_threshold.py

# 4. Evaluate on held-out test set
python evaluate.py \
    --model-path models/best_model.keras \
    --threshold <recommended-threshold>

# 5. Run the application
streamlit run app.py

🧪 Development Flow

For code changes:

# Install dependencies
pip install -r requirements.txt

# Run formatting check
black --check .

# Run linting
flake8 .

# Run tests
pytest tests/ -v

# Start application
streamlit run app.py

⚠️ Evaluation Rule

The project follows:

TRAIN
  │
  ▼
Best Model
  │
  ▼
VALIDATION
  │
  ├── Model Selection
  └── Threshold Selection
  │
  ▼
FREEZE THRESHOLD
  │
  ▼
TEST
  │
  ▼
FINAL METRICS

The test set must not be used for threshold optimization.

📄 License

This project is licensed under the MIT License.

See LICENSE for details.

<div align="center">

🫁 PneumoVision

Computer Vision · Medical Imaging · Transfer Learning · Explainable AI

</div>
