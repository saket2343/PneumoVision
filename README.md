<div align="center">

🫁 PneumoVision

AI-Powered Pneumonia Detection from Chest X-Rays

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Backbone-EfficientNet--B0-green?style=for-the-badge" alt="EfficientNet-B0">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p>
  <b>End-to-end deep learning pipeline for pneumonia classification with transfer learning, two-stage fine-tuning, threshold optimization, Grad-CAM explainability, error analysis, and Streamlit deployment.</b>
</p>

</div>

📌 Table of Contents

Overview

Key Features

System Architecture

Dataset

Preprocessing

Model Architecture

Training Strategy

Evaluation

Threshold Optimization

Explainability with Grad-CAM

Error Analysis

Calibration

Streamlit Application

Project Structure

Installation

Usage

Experiments

Testing & CI/CD

Important Engineering Fixes

Results

Limitations

Future Work

License

🔎 Overview

PneumoVision is an end-to-end computer vision system for classifying chest X-ray images into:

🟢 NORMAL

🔴 PNEUMONIA

The project goes beyond training a CNN by implementing the complete ML lifecycle:

Data Validation → Preprocessing → Augmentation → Transfer Learning → Two-Stage Fine-Tuning → Threshold Optimization → Calibration → Test Evaluation → Explainability → Error Analysis → Deployment

The system is designed with reproducibility and evaluation integrity in mind. In particular, the decision threshold is selected using validation data only, while the held-out test set is reserved for final evaluation.

⚠️ Medical Disclaimer: PneumoVision is a research/portfolio project and is not a certified medical device. It must not be used for clinical diagnosis or treatment decisions.

✨ Key Features

Area

Implementation

🧠 Model

ImageNet-pretrained EfficientNet-B0

🏋️ Training

Two-stage transfer learning + fine-tuning

🖼️ Preprocessing

Resize, RGB conversion, CLAHE, normalization

🔄 Augmentation

Conservative image augmentation on training data

⚖️ Imbalance

Class weighting, optional oversampling, focal loss

🎯 Thresholding

Validation-based threshold optimization

📊 Evaluation

Accuracy, Precision, Recall, Specificity, F1, MCC, ROC-AUC

🔥 Explainability

Grad-CAM heatmaps

🌡️ Calibration

Optional temperature scaling

🔬 Analysis

False positives, false negatives, confidence analysis

🧪 Testing

Pytest + regression tests

⚙️ CI/CD

GitHub Actions

🌐 Deployment

Streamlit web application

📄 Reporting

Downloadable PDF prediction reports

🔁 Robustness

Optional Test-Time Augmentation

🏗️ System Architecture

                         ┌──────────────────────┐
                         │      Chest X-Ray     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Image Validation     │
                         │ & Loading             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Resize 224 × 224     │
                         │ RGB Conversion       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       CLAHE          │
                         │ Contrast Enhancement │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Training Only: Augmentation │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │      EfficientNet-B0        │
                     │    ImageNet Pretrained      │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Global Average Pooling      │
                     │ Dropout → Dense → BN        │
                     │ Dropout → Dense → BN        │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Sigmoid Probability         │
                     │ P(PNEUMONIA | X-Ray)       │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Validation-Tuned Threshold  │
                     └──────────────┬──────────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  NORMAL / PNEUMONIA      Grad-CAM
                                               │
                                               ▼
                                      Visual Explanation
                                               │
                                               ▼
                                      Streamlit / PDF

📂 Dataset

The project uses a binary chest X-ray classification dataset organized as:

dataset/
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

Dataset Distribution

The current verified dataset split contains 5,856 images:

Split

NORMAL

PNEUMONIA

Total

🏋️ Train

1,214

3,495

4,709

🔧 Validation

135

388

523

🧪 Test

234

390

624

Total

1,583

4,273

5,856

The training set is naturally imbalanced toward the PNEUMONIA class, so imbalance handling is explicitly addressed during training.

🖼️ Preprocessing

Every image passes through the following deterministic preprocessing pipeline:

Input X-Ray
     │
     ▼
Image Validation
     │
     ▼
RGB Conversion
     │
     ▼
Resize → 224 × 224
     │
     ▼
CLAHE
     │
     ▼
Float32 + [0,1]
     │
     ▼
Backbone-Aware Preprocessing
     │
     ▼
Model

CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to improve local contrast.

Clip Limit : 2.0
Tile Grid  : 8 × 8

Unlike global contrast enhancement, CLAHE operates locally while limiting excessive amplification of image noise.

Data Augmentation

Augmentation is applied only to the training set.

Supported transformations include:

Random resized crop

Horizontal flip

Small affine transformations

Brightness/contrast variation

Gaussian noise

Conservative geometric perturbations

Validation and test images remain deterministic to ensure reliable evaluation.

🧠 Model Architecture

Backbone

The default backbone is:

EfficientNet-B0 pretrained on ImageNet

Transfer learning is used because the available medical-image dataset is considerably smaller than datasets typically required to train a deep CNN from scratch.

Classification Head

EfficientNet-B0
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dropout(0.40)
       │
       ▼
Dense(256)
       │
       ▼
BatchNormalization
       │
       ▼
Swish
       │
       ▼
Dropout(0.30)
       │
       ▼
Dense(128)
       │
       ▼
BatchNormalization
       │
       ▼
Swish
       │
       ▼
Dense(1)
       │
       ▼
Sigmoid

The sigmoid output represents:

P(PNEUMONIA | X-Ray)

🏋️ Training Strategy

Training is intentionally divided into two stages.

Stage 1 — Feature Extraction

50 Epochs

During the first stage:

EfficientNet-B0 backbone → FROZEN
Classification head       → TRAINABLE
Learning rate             → 1e-3

The classifier learns to map the pretrained ImageNet representations to the NORMAL/PNEUMONIA task.

Why freeze the backbone?

The pretrained convolutional features already contain useful visual representations. Training only the new classification head first allows the model to adapt to the new task without immediately modifying the pretrained representation.

Stage 2 — Fine-Tuning

15 Epochs

After the classifier has stabilized:

EfficientNet-B0
       │
       ├── Earlier layers → Frozen
       │
       └── Last ~40 layers → Trainable

The learning rate is reduced:

Stage 1 → 1e-3
Stage 2 → 1e-5

This allows the high-level visual features to adapt to chest X-ray patterns without aggressively destroying the pretrained representation.

Training Flow

ImageNet Pretrained Model
          │
          ▼
┌─────────────────────────┐
│ Stage 1                 │
│ 50 Epochs               │
│ Backbone Frozen         │
│ LR = 1e-3               │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stage 2                 │
│ 15 Epochs               │
│ Partial Fine-Tuning     │
│ LR = 1e-5               │
└────────────┬────────────┘
             │
             ▼
       Best Checkpoint
       Validation ROC-AUC

⚙️ Training Configuration

Hyperparameter

Value

Backbone

EfficientNet-B0

Input size

224 × 224 × 3

Batch size

32

Stage 1

50 epochs

Stage 2

15 epochs

Stage 1 LR

1e-3

Stage 2 LR

1e-5

Optimizer

AdamW

Weight decay

1e-4

Gradient clipping

1.0

Dropout

0.40 / 0.30

Dense layers

256 / 128

Activation

Swish

Output

Sigmoid

Primary loss

Binary Cross-Entropy

Optional loss

Focal Loss

Seed

42

Mixed precision

Enabled

Model selection

Validation ROC-AUC

⚖️ Class Imbalance Handling

The training data contains substantially more PNEUMONIA images than NORMAL images.

The project supports three approaches.

1. Class-Weighted Training

Class weights are calculated from the training distribution and passed to the training process.

2. Oversampling

Optional train-only oversampling can be enabled:

python train.py --oversample --oversample-ratio 0.5 --no-class-weights

Validation and test sets are never oversampled.

3. Focal Loss

Focal loss can be used to place more emphasis on difficult examples:

python train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25

These strategies are treated as controlled experiments rather than blindly combining every technique.

🎯 Threshold Optimization

The model produces a probability:

P(PNEUMONIA | X-Ray)

The default threshold of 0.50 is not automatically assumed to be optimal.

Instead, the project searches the validation set for an operating threshold.

Threshold Range : 0.01 → 0.99
Step Size       : 0.01

The primary selection criterion is:

Among thresholds achieving sensitivity ≥ 95%, select the threshold with the highest specificity.

Other thresholding strategies are also reported:

Youden's J

Best F1

Best Balanced Accuracy

Evaluation Protocol

                 TRAIN
                   │
                   ▼
             Model Training
                   │
                   ▼
          Best Validation AUC
                   │
                   ▼
              VALIDATION
                   │
                   ▼
          Threshold Selection
                   │
                   ▼
         ┌───────────────────┐
         │ Freeze Threshold  │
         └─────────┬─────────┘
                   │
                   ▼
                 TEST
                   │
                   ▼
           Final Evaluation

Why this matters

The test set is never used to select the threshold.

This prevents test-set leakage and gives a more honest estimate of generalization performance.

📊 Evaluation

The evaluation pipeline reports:

Classification Metrics

Accuracy

Precision

Recall / Sensitivity

Specificity

F1-score

Balanced Accuracy

Matthews Correlation Coefficient

ROC-AUC

Diagnostic Outputs

Confusion matrix

ROC curve

Precision-Recall curve

TP / TN / FP / FN counts

Probability distributions

Confidence-ranked errors

🔥 Explainability with Grad-CAM

PneumoVision integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the image regions that contributed most to the model prediction.

                 Chest X-Ray
                      │
                      ▼
                CNN Forward Pass
                      │
                      ▼
              PNEUMONIA Probability
                      │
                      ▼
              Gradient Computation
                      │
                      ▼
          Last Convolutional Features
                      │
                      ▼
                  Grad-CAM
                      │
                      ▼
             Activation Heatmap
                      │
                      ▼
              Overlay on X-Ray

This allows users to inspect the model's attention alongside its classification.

Grad-CAM is an interpretability tool, not evidence that a highlighted region is medically causal.

🔬 Error Analysis

The project performs dedicated analysis of incorrect predictions.

False Positive

Actual      → NORMAL
Prediction  → PNEUMONIA

False Negative

Actual      → PNEUMONIA
Prediction  → NORMAL

The system can generate:

False-positive montages

False-negative montages

Confidence-ranked mistakes

Prediction probability distributions

Confusion-matrix analysis

This makes it possible to inspect where and why the model fails, rather than relying only on aggregate accuracy.

🌡️ Probability Calibration

The project optionally applies temperature scaling to improve the calibration of predicted probabilities.

python calibrate.py --model-path models/best_model.keras

Calibration is fitted on the validation set.

The calibrated temperature can then be used during threshold tuning and evaluation.

The calibration step is treated as an experiment: it is retained only when it provides useful probability calibration.

🌐 Streamlit Application

The trained model is exposed through a browser-based Streamlit application.

streamlit run app.py

Application Features

Feature

Description

📤 Upload

Drag-and-drop chest X-ray

🧠 Prediction

NORMAL / PNEUMONIA

📊 Confidence

Predicted probability

🎯 Threshold

Current decision threshold

🔥 Grad-CAM

Visual explanation

🔄 TTA

Optional Test-Time Augmentation

📜 History

Session-level prediction history

🎚️ Threshold Slider

Interactive threshold exploration

📄 PDF Report

Downloadable prediction report

The application loads the recommended validation threshold from:

outputs/reports/threshold_recommendation.json

📁 Project Structure

PneumoVision/
│
├── 📂 dataset/
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
├── 📂 outputs/
│   ├── confusion_matrix/
│   ├── roc/
│   ├── pr_curve/
│   ├── heatmaps/
│   ├── error_analysis/
│   └── reports/
│
├── 📂 models/
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── train.py
├── predict.py
├── tune_threshold.py
├── calibrate.py
├── evaluate.py
├── check_data_integrity.py
├── compare_models.py
├── tune.py
├── export_model.py
├── config.py
├── requirements.txt
└── LICENSE

🚀 Installation

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PneumoVision

2. Create a Virtual Environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate

Windows

python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

▶️ Usage

1. Check Dataset Integrity

python check_data_integrity.py

Checks include:

Class distribution

Corrupted images

Exact duplicate files

Cross-split duplicates

Patient-ID overlap where detectable

2. Train

python train.py

Default training:

Stage 1 → 50 epochs
Stage 2 → 15 epochs

For a shorter experiment:

python train.py --epochs 30 --fine-tune-epochs 10

To skip fine-tuning:

python train.py --no-fine-tune

3. Tune the Decision Threshold

python tune_threshold.py

The recommended threshold is saved for downstream inference and evaluation.

4. Evaluate

python evaluate.py \
    --model-path models/best_model.keras \
    --threshold <recommended-threshold>

The test set is used only after the threshold has been frozen.

5. Single Image Prediction

python predict.py --image path/to/xray.jpg

With Test-Time Augmentation:

python predict.py \
    --image path/to/xray.jpg \
    --tta

Save a Grad-CAM heatmap:

python predict.py \
    --image path/to/xray.jpg \
    --save-heatmap outputs/heatmaps/example.png

6. Launch the Web Application

streamlit run app.py

🧪 Experiments

The repository supports controlled experiments across several dimensions.

Backbone Comparison

Supported architectures include:

EfficientNet-B0

EfficientNet-B1

EfficientNet-B2

EfficientNet-B3

DenseNet121

ResNet50

MobileNetV3Large

Example:

python compare_models.py \
    --backbones EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3

Attention Modules

Optional attention mechanisms:

SE — Squeeze-and-Excitation

CBAM — Convolutional Block Attention Module

Example:

python train.py \
    --backbone EfficientNetB3 \
    --attention se

Hyperparameter Tuning

Optuna can be used to explore:

Learning rate

Dropout

Batch size

Weight decay

Fine-tuning configuration

python tune.py

🧹 Dataset Leakage & Integrity Checks

Medical imaging datasets require careful split validation.

The project checks for:

Exact duplicates

File hashes are compared across splits.

Duplicate groups

Repeated images inside the same split can be detected.

Patient overlap

Patient identifiers are checked where they can be inferred from filenames.

Corrupted images

Unreadable or invalid image files are reported.

Exact hashing does not guarantee detection of near-duplicates such as the same X-ray saved with different compression or resolution.

🧪 Testing

Run the complete test suite:

pytest tests/ -v --cov=src

Tests cover:

Model construction

Preprocessing

Dataset loading

Dataset balancing

Leakage detection

Threshold optimization

Backbone preprocessing

Model save/load compatibility

A dedicated regression test protects against the EfficientNet preprocessing issue described below.

⚙️ CI/CD

GitHub Actions is configured under:

.github/workflows/ci.yml

The CI pipeline can automatically run:

flake8 .
black --check .
pytest tests/ -v

This ensures that changes do not silently break the ML pipeline.

🐛 Important Engineering Fixes

1. EfficientNet Double Normalization

Problem

The dataset pipeline produced:

[0, 1]

while the Keras EfficientNet implementation performs its own internal input rescaling.

Passing [0,1] directly therefore caused an unintended second normalization:

[0,1]
   ↓
/ 255
   ↓
[0, 0.0039]

This severely reduced the useful signal available to the pretrained backbone.

Observed Behavior

The model became heavily biased toward predicting:

PNEUMONIA

for a large fraction of images.

Fix

Backbone-aware preprocessing was moved into the model pipeline:

Dataset
[0,1]
  │
  ▼
× 255
  │
  ▼
[0,255]
  │
  ▼
EfficientNet preprocessing
  │
  ▼
Backbone

The preprocessing implementation is also serialization-safe so that:

Save Model → Load Model → Predict

uses the same preprocessing behavior.

2. Hard-Coded 0.5 Threshold

Problem

Earlier inference paths used:

probability >= 0.5

even when a validation-derived threshold had been selected.

Fix

The recommended threshold is now:

Selected using validation data.

Saved to the project reports.

Loaded by inference/deployment.

Used consistently during prediction.

3. Albumentations API Compatibility

The augmentation pipeline was updated to remain compatible with current Albumentations APIs, including changes to transformations such as:

RandomResizedCrop
GaussNoise

4. Keras Model Serialization

A bare Lambda(preprocess_input) can cause model serialization/loading problems.

The project uses registered custom preprocessing layers where required so that model export and reload remain reliable.

📊 Results

Evaluation Protocol

The final result should be generated using:

1. Train model
       ↓
2. Select best checkpoint using validation ROC-AUC
       ↓
3. Tune threshold on validation set
       ↓
4. Freeze threshold
       ↓
5. Evaluate exactly once on held-out test set

Final Test Results

Metric

Result

Accuracy

Run evaluate.py

Precision

Run evaluate.py

Sensitivity / Recall

Run evaluate.py

Specificity

Run evaluate.py

F1-score

Run evaluate.py

Balanced Accuracy

Run evaluate.py

MCC

Run evaluate.py

ROC-AUC

Run evaluate.py

Important: The historical pre-fix metrics are intentionally not reported as final model performance. They were obtained before correcting the EfficientNet input-preprocessing issue. The final numbers in this section should be replaced with the results from the corrected model after retraining.

📈 Recommended Results Section After Retraining

Once the corrected model has been retrained, replace the table above with the actual test-set values:

| Metric              | Test Result |
|---------------------|------------:|
| Accuracy            | XX.XX%      |
| Precision           | XX.XX%      |
| Sensitivity/Recall  | XX.XX%      |
| Specificity         | XX.XX%      |
| F1-score            | XX.XX%      |
| Balanced Accuracy   | XX.XX%      |
| MCC                 | XX.XX%      |
| ROC-AUC             | XX.XX%      |

This keeps the README scientifically honest and prevents accidentally presenting the earlier buggy run as the final result.

💡 Key Design Decisions

Why EfficientNet-B0?

EfficientNet-B0 provides a strong accuracy-to-computation trade-off and is practical for transfer learning on a moderate-sized image dataset.

Why two-stage training?

Freezing the backbone first allows the new classification head to stabilize before high-level pretrained features are gradually adapted.

Why CLAHE?

It improves local contrast while limiting excessive amplification of noise.

Why class weighting?

The training distribution is imbalanced, so class weighting helps prevent the optimization objective from being dominated by the majority class.

Why optimize the threshold?

The best operating point for a screening-oriented classifier does not necessarily occur at probability 0.5.

Why use validation data for threshold selection?

Using the test set for threshold tuning would leak information from the evaluation set and produce an overly optimistic estimate.

Why Grad-CAM?

It provides a visual explanation of the image regions contributing to the prediction, making model behavior easier to inspect.

🔮 Future Work

External validation on an independent chest X-ray dataset

Near-duplicate detection using perceptual hashing

Multi-class pneumonia classification

Uncertainty estimation with ensembles / MC Dropout

Model distillation for lightweight deployment

TFLite quantization

Improved probability calibration

Clinical validation with expert radiologist annotations

⚠️ Limitations

Not a clinical diagnostic system.The model is intended for research and educational purposes.

Dataset generalization.Performance on a public dataset does not guarantee performance across hospitals, scanners, populations, or acquisition protocols.

No external validation yet.Independent validation is required before making clinical generalization claims.

Grad-CAM limitations.A heatmap indicates influential regions but does not establish medical causality.

Threshold criterion.The 95% sensitivity target is a project-level modeling objective, not a clinically validated requirement.

📜 License

This project is licensed under the MIT License.

See LICENSE for details.

<div align="center">

🫁 PneumoVision

Computer Vision • Transfer Learning • Explainable AI • Medical Imaging

Built as an end-to-end ML engineering project.

</div>
