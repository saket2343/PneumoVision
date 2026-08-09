<div align="center">

🫁 PneumoVision
AI-Powered Pneumonia Detection from Chest X-Rays
<p> <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white"> <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"> <img src="https://img.shields.io/badge/Backbone-EfficientNet--B0-2E7D32?style=flat-square"> <img src="https://img.shields.io/badge/License-MIT-F7DF1E?style=flat-square"> </p>

An end-to-end computer vision system for detecting pneumonia from chest X-rays using transfer learning, two-stage fine-tuning, validation-based threshold optimization, Grad-CAM explainability, error analysis, and Streamlit deployment.

</div>

📑 Table of Contents
Overview

Problem Statement

Project Objectives

Key Features

End-to-End Workflow

Dataset

Data Preprocessing

Data Augmentation

Model Architecture

Training Strategy

Class Imbalance Handling

Threshold Optimization

Model Evaluation

Grad-CAM Explainability

Error Analysis

Probability Calibration

Streamlit Application

Project Structure

Installation

Running the Project

Experiments

Testing

CI/CD

Important Engineering Fixes

Results

Limitations

Future Work

License

🔎 Overview
PneumoVision is a deep-learning based chest X-ray classification system that predicts whether an X-ray belongs to one of two classes:

Class	Meaning
🟢 NORMAL	No pneumonia detected
🔴 PNEUMONIA	Pneumonia detected
The project is built as a complete ML pipeline rather than a single training notebook.

Pipeline
Chest X-Ray
    │
    ▼
Data Validation
    │
    ▼
Image Preprocessing
    │
    ├── Resize
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
    ├── Stage 1: Feature Extraction
    │
    └── Stage 2: Fine-Tuning
    │
    ▼
Pneumonia Probability
    │
    ▼
Validation-Based Threshold
    │
    ├───────────────┐
    ▼               ▼
Prediction       Grad-CAM
    │               │
    └───────┬───────┘
            ▼
     Error Analysis
            │
            ▼
   Streamlit Deployment
⚠️ Medical Disclaimer: This project is intended for research, educational, and portfolio purposes only. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

🎯 Problem Statement
Pneumonia is a respiratory infection that can manifest through patterns visible in chest radiographs. Manual interpretation of large numbers of X-rays can be time-consuming and requires trained medical professionals.

The objective of this project is to develop a computer-vision pipeline that can:

Automatically classify chest X-rays as NORMAL or PNEUMONIA.

Achieve strong sensitivity while controlling false positives.

Provide a probability score instead of only a hard classification.

Explain predictions using Grad-CAM.

Analyze model failures using false-positive and false-negative analysis.

Expose the trained model through an interactive web application.

🎯 Project Objectives
The project focuses on the following objectives:

Build a binary chest X-ray classifier.

Use ImageNet transfer learning.

Apply CLAHE-based preprocessing.

Use medically conservative augmentation.

Train using a two-stage strategy.

Handle class imbalance.

Optimize the classification threshold using validation data.

Evaluate on an untouched test set.

Generate Grad-CAM explanations.

Perform error analysis.

Support probability calibration.

Deploy through Streamlit.

Add automated tests.

Add GitHub Actions CI.

✨ Key Features
Category	Implementation
🧠 Deep Learning	EfficientNet-B0 transfer learning
🏋️ Training	50-epoch feature extraction + 15-epoch fine-tuning
🖼️ Image Processing	Resize + RGB + CLAHE + normalization
🔄 Augmentation	Training-only conservative augmentation
⚖️ Imbalance	Class weights, optional oversampling, focal loss
🎯 Decision Making	Validation-based threshold optimization
📊 Evaluation	Accuracy, Precision, Recall, Specificity, F1, MCC, ROC-AUC
🔥 Explainability	Grad-CAM
🔬 Error Analysis	FP/FN and confidence-ranked mistakes
🌡️ Calibration	Temperature scaling
🔁 Robustness	Optional Test-Time Augmentation
🌐 Deployment	Streamlit
📄 Reporting	PDF prediction reports
🧪 Testing	Pytest
⚙️ CI/CD	GitHub Actions
🔄 End-to-End Workflow
                         ┌──────────────────┐
                         │   Chest X-Ray    │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Data Validation     │
                       │ • Corrupt images    │
                       │ • Duplicates        │
                       │ • Split integrity   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Preprocessing       │
                       │ • 224 × 224         │
                       │ • RGB               │
                       │ • CLAHE             │
                       │ • Normalization     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Train Augmentation  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     EfficientNet-B0      │
                    │   ImageNet Pretrained    │
                    └────────────┬─────────────┘
                                 │
                     ┌───────────┴───────────┐
                     │                       │
                     ▼                       ▼
              ┌──────────────┐       ┌──────────────┐
              │   Stage 1    │       │   Stage 2    │
              │ 50 Epochs    │──────▶│ 15 Epochs    │
              │ Frozen       │       │ Fine-Tuning  │
              └──────────────┘       └──────┬───────┘
                                             │
                                             ▼
                                  ┌──────────────────┐
                                  │ Probability      │
                                  │ P(PNEUMONIA)     │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │ Validation       │
                                  │ Threshold        │
                                  └────────┬─────────┘
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                         Prediction                 Grad-CAM
                              │                         │
                              └────────────┬────────────┘
                                           ▼
                                     Error Analysis
                                           │
                                           ▼
                                   Streamlit Application
📊 Dataset
The dataset is organized into three independent splits:

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
The current verified split contains 5,856 X-ray images.

Split	NORMAL	PNEUMONIA	Total
🏋️ Training	1,214	3,495	4,709
🔧 Validation	135	388	523
🧪 Test	234	390	624
Total	1,583	4,273	5,856
Class Distribution
NORMAL
██████████                         1,583

PNEUMONIA
████████████████████████████       4,273
The dataset is therefore substantially imbalanced toward the pneumonia class, which is addressed during training.

🖼️ Data Preprocessing
Every X-ray passes through a deterministic preprocessing pipeline.

Original Image
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
Float32
      │
      ▼
Normalize → [0, 1]
      │
      ▼
Backbone-Specific Preprocessing
      │
      ▼
EfficientNet-B0
1. Image Validation
The pipeline checks whether images are readable and valid before training.

2. Resize
All images are resized to:

224 × 224 × 3
3. RGB Conversion
Images are converted to three-channel RGB format to match the pretrained backbone's expected input structure.

4. CLAHE
Contrast Limited Adaptive Histogram Equalization improves local image contrast.

Clip Limit : 2.0
Tile Grid  : 8 × 8
5. Normalization
Images are converted to floating-point representation and normalized to:

[0, 1]
The model then performs the appropriate preprocessing required by the selected backbone.

🔄 Data Augmentation
Augmentation is applied only to training images.

Supported transformations include:

Random resized crop

Horizontal flip

Small affine transformations

Brightness/contrast changes

Gaussian noise

Conservative geometric transformations

Important
Validation and test images are not randomly augmented.

This ensures that:

Validation = deterministic
Test       = deterministic
and evaluation reflects the actual model rather than random transformations.

🧠 Model Architecture
Backbone
The primary model uses:

EfficientNet-B0 + ImageNet pretrained weights

EfficientNet-B0 was selected because it provides a strong balance between model capacity, computational cost, and transfer-learning performance.

Classification Head
EfficientNet-B0
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dropout (0.40)
       │
       ▼
Dense (256)
       │
       ▼
Batch Normalization
       │
       ▼
Swish
       │
       ▼
Dropout (0.30)
       │
       ▼
Dense (128)
       │
       ▼
Batch Normalization
       │
       ▼
Swish
       │
       ▼
Dense (1)
       │
       ▼
Sigmoid
       │
       ▼
P(PNEUMONIA)
Output
The final sigmoid neuron produces:

P(PNEUMONIA | X-Ray)
For example:

Probability = 0.87
means the model estimates an 87% pneumonia probability before applying the decision threshold.

🏋️ Training Strategy
The model is trained in two controlled stages.

Stage 1 — Feature Extraction
Configuration
Parameter	Value
Epochs	50
Backbone	Frozen
Classification Head	Trainable
Learning Rate	1e-3
Image
  │
  ▼
EfficientNet-B0
  │
  ├── Frozen
  │
  ▼
Classification Head
  │
  └── Trainable
Purpose
The goal of Stage 1 is to allow the new classification head to learn the NORMAL/PNEUMONIA decision boundary using the pretrained visual representations.

Stage 2 — Fine-Tuning
Configuration
Parameter	Value
Epochs	15
Backbone	Partially unfrozen
Trainable layers	Last ~40 layers
Learning Rate	1e-5
EfficientNet-B0
      │
      ├── Earlier layers → Frozen
      │
      └── Last ~40 layers → Trainable
Purpose
Stage 2 adapts the high-level pretrained features to the chest X-ray domain.

The learning rate is reduced from:

1e-3 → 1e-5
to make fine-tuning gradual and stable.

Why Two Stages?
Training the entire pretrained network immediately with a large learning rate can destroy useful pretrained representations.

Instead:

Stage 1
Learn classifier
      │
      ▼
Stage 2
Adapt high-level features
      │
      ▼
Final Model
This provides a more controlled transfer-learning procedure.

⚙️ Training Configuration
Hyperparameter	Default
Backbone	EfficientNet-B0
Input	224 × 224 × 3
Batch Size	32
Stage 1	50 epochs
Stage 2	15 epochs
Stage 1 LR	1e-3
Stage 2 LR	1e-5
Optimizer	AdamW
Weight Decay	1e-4
Gradient Clipping	1.0
Dropout	0.40 / 0.30
Dense Layers	256 / 128
Activation	Swish
Loss	Binary Cross-Entropy
Optional Loss	Focal Loss
Seed	42
Mixed Precision	Enabled
Checkpoint Selection	Validation ROC-AUC
⚖️ Class Imbalance Handling
The dataset contains significantly more pneumonia images than normal images.

The project supports multiple strategies.

Class-Weighted Training
Class weights are calculated from the training distribution.

This increases the contribution of the minority class during optimization.

Oversampling
Optional training-only oversampling:

python train.py --oversample --oversample-ratio 0.5 --no-class-weights
Focal Loss
Optional focal loss:

python train.py \
    --loss focal \
    --focal-gamma 2.0 \
    --focal-alpha 0.25
These methods are evaluated as controlled experiments rather than automatically stacking every imbalance technique.

🎯 Threshold Optimization
The neural network produces a continuous probability, but deployment requires a binary decision.

Instead of assuming:

threshold = 0.50
the project determines the operating threshold from the validation set.

Search
0.01 → 0.99
step = 0.01
Primary Criterion
The main project objective is:

Among thresholds achieving sensitivity ≥ 95%, choose the threshold with the highest specificity.

Additional threshold strategies are also supported:

Youden's J

Best F1

Best Balanced Accuracy

🔐 Evaluation Integrity
The test set is never used for threshold optimization.

TRAIN
  │
  ▼
Model Training
  │
  ▼
VALIDATION
  │
  ├── Select Best Model
  └── Select Threshold
          │
          ▼
     Freeze Threshold
          │
          ▼
        TEST
          │
          ▼
   Final Evaluation
This prevents test-set information from influencing the final decision rule.

📈 Model Evaluation
The evaluation pipeline reports:

Classification Metrics
Metric	Description
Accuracy	Overall fraction of correct predictions
Precision	Fraction of positive predictions that are correct
Recall / Sensitivity	Fraction of pneumonia cases detected
Specificity	Fraction of normal cases correctly rejected
F1-score	Harmonic mean of precision and recall
Balanced Accuracy	Mean of sensitivity and specificity
MCC	Correlation between predictions and true labels
ROC-AUC	Ranking ability across thresholds
Additional Outputs
outputs/
├── confusion_matrix/
├── roc/
├── pr_curve/
├── heatmaps/
├── error_analysis/
└── reports/
🔥 Grad-CAM Explainability
PneumoVision uses Grad-CAM to visualize regions that contributed to a prediction.

Chest X-Ray
     │
     ▼
Forward Pass
     │
     ▼
Pneumonia Probability
     │
     ▼
Gradient Computation
     │
     ▼
Last Convolutional Feature Maps
     │
     ▼
Grad-CAM Weights
     │
     ▼
Activation Heatmap
     │
     ▼
Overlay on X-Ray
Why Grad-CAM?
A classification probability alone does not tell us what the model focused on.

Grad-CAM provides an additional visual signal for inspecting model behavior.

Note: Grad-CAM is an interpretability aid and should not be interpreted as clinical proof or causal evidence.

🔬 Error Analysis
The project explicitly analyzes incorrect predictions.

False Positive
Actual      → NORMAL
Predicted   → PNEUMONIA
False Negative
Actual      → PNEUMONIA
Predicted   → NORMAL
The analysis module supports:

False-positive visualization

False-negative visualization

Confidence-ranked errors

Probability distributions

Confusion-matrix analysis

Error montages

This helps identify whether errors occur primarily on:

Low-quality images

Ambiguous cases

High-confidence mistakes

Visually difficult examples

🌡️ Probability Calibration
The project supports temperature scaling to improve probability calibration.

Run:

python calibrate.py \
    --model-path models/best_model.keras
Calibration is fitted using validation data.

The calibrated temperature can then be used during threshold tuning and evaluation.

Calibration is treated as an empirical experiment and is used only when it improves probability quality.

🌐 Streamlit Application
The trained model is exposed through an interactive Streamlit interface.

Start the Application
streamlit run app.py
Application Workflow
Upload X-Ray
     │
     ▼
Preprocess Image
     │
     ▼
Model Inference
     │
     ▼
Probability
     │
     ▼
Apply Recommended Threshold
     │
     ├──────────────┐
     ▼              ▼
Prediction       Grad-CAM
     │              │
     └───────┬──────┘
             ▼
       Display Result
             │
             ▼
        PDF Report
Application Features
Feature	Description
📤 Image Upload	Upload chest X-ray
🧠 Prediction	NORMAL / PNEUMONIA
📊 Probability	Model confidence
🎯 Threshold	Active decision threshold
🔥 Grad-CAM	Visual explanation
🔄 TTA	Optional Test-Time Augmentation
📜 History	Session prediction history
🎚️ Threshold Slider	Interactive threshold analysis
📄 PDF	Download prediction report
The application reads the recommended threshold from:

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
🛠️ Installation
1. Clone Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PneumoVision
2. Create Virtual Environment
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
Windows
python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Step 1 — Check Dataset
python check_data_integrity.py
Checks:

Corrupted images

Class distribution

Duplicate files

Cross-split duplicates

Patient-ID overlap where detectable

Step 2 — Train Model
python train.py
Default:

Stage 1 → 50 epochs
Stage 2 → 15 epochs
Short experiment:

python train.py \
    --epochs 30 \
    --fine-tune-epochs 10
Without fine-tuning:

python train.py --no-fine-tune
Step 3 — Optimize Threshold
python tune_threshold.py
This generates the recommended validation threshold.

Step 4 — Evaluate
python evaluate.py \
    --model-path models/best_model.keras \
    --threshold <recommended-threshold>
Step 5 — Predict a Single X-Ray
python predict.py \
    --image path/to/xray.jpg
With TTA:

python predict.py \
    --image path/to/xray.jpg \
    --tta
With Grad-CAM output:

python predict.py \
    --image path/to/xray.jpg \
    --save-heatmap outputs/heatmaps/example.png
Step 6 — Launch Streamlit
streamlit run app.py
🧪 Experiments
The repository supports controlled experiments across different model and training configurations.

Backbone Comparison
Available backbones include:

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
Optional:

SE — Squeeze-and-Excitation

CBAM — Convolutional Block Attention Module

Example:

python train.py \
    --backbone EfficientNetB3 \
    --attention se
Hyperparameter Tuning
Optuna experiments can explore:

Learning rate

Dropout

Batch size

Weight decay

Fine-tuning configuration

python tune.py
🧹 Dataset Leakage & Integrity
Medical imaging datasets require careful split validation.

The project checks:

Exact Duplicates
File hashes are compared across dataset splits.

Within-Split Duplicates
Repeated images inside a split can be detected.

Patient Overlap
Patient identifiers are checked where they can be inferred from filenames.

Corrupt Images
Unreadable images are detected before training.

Exact hashing cannot guarantee detection of near-duplicates such as the same image saved at a different resolution or compression level.

🧪 Testing
Run:

pytest tests/ -v --cov=src
The test suite covers:

Model construction

Image preprocessing

Dataset loading

Dataset balancing

Leakage checks

Threshold optimization

Backbone preprocessing

Model save/load compatibility

A regression test specifically protects the corrected EfficientNet preprocessing pipeline.

⚙️ CI/CD
GitHub Actions is configured in:

.github/workflows/ci.yml
The CI pipeline can automatically run:

flake8 .
black --check .
pytest tests/ -v
CI Workflow
Git Push / Pull Request
          │
          ▼
     GitHub Actions
          │
     ┌────┴────┐
     ▼         ▼
  Linting    Formatting
     │         │
     └────┬────┘
          ▼
        Tests
          │
          ▼
       CI Pass
🐛 Important Engineering Fixes
1. EfficientNet Double Normalization
Original Problem
The dataset pipeline produced images in:

[0, 1]
while Keras EfficientNet also performs internal input rescaling.

This caused:

[0,1]
  │
  ▼
/255
  │
  ▼
[0, 0.0039]
The input signal was therefore unnecessarily compressed.

Observed Behavior
The model became heavily biased toward predicting:

PNEUMONIA
for a large fraction of images.

Fix
Backbone-aware preprocessing was implemented:

Dataset
[0,1]
  │
  ▼
×255
  │
  ▼
[0,255]
  │
  ▼
EfficientNet Preprocessing
  │
  ▼
Backbone
The fix is applied consistently across:

Training

Evaluation

Inference

Grad-CAM

Streamlit

2. Hard-Coded Threshold
Problem
Earlier inference logic used:

probability >= 0.5
even after validation had identified a different operating threshold.

Fix
The threshold is now:

Validation
    ↓
Threshold Optimization
    ↓
Save Recommendation
    ↓
Inference
    ↓
Streamlit
The same recommended threshold can therefore be used consistently across deployment.

3. Albumentations API Compatibility
The augmentation pipeline was updated to support the current Albumentations API, including changes to transformations such as:

RandomResizedCrop
GaussNoise
4. Keras Serialization
A bare Lambda(preprocess_input) can create serialization/loading problems.

The project uses serialization-safe custom preprocessing layers where required.

Therefore:

Save Model
    ↓
Load Model
    ↓
Inference
retains the correct preprocessing behavior.

📊 Results
Evaluation Protocol
Final results must be generated using the corrected preprocessing pipeline:

1. Train
   ↓
2. Select best validation checkpoint
   ↓
3. Optimize threshold on validation
   ↓
4. Freeze threshold
   ↓
5. Evaluate on held-out test set
Final Test Metrics
Metric	Test Result
Accuracy	XX.XX%
Precision	XX.XX%
Sensitivity / Recall	XX.XX%
Specificity	XX.XX%
F1-score	XX.XX%
Balanced Accuracy	XX.XX%
MCC	XX.XX
ROC-AUC	XX.XX
Important: Do not enter the historical pre-fix metrics into this table. Those results were produced before the EfficientNet preprocessing issue was corrected. Replace the placeholders with the metrics generated by the corrected model.

📝 Experiment Reporting Template
For every major experiment, record:

Field	Example
Backbone	EfficientNet-B0
Input Size	224 × 224
Loss	Binary Cross-Entropy
Balancing	Class Weights
Stage 1	50 epochs
Stage 2	15 epochs
Threshold	Validation-selected
Sensitivity	XX.XX%
Specificity	XX.XX%
F1	XX.XX%
ROC-AUC	XX.XX
MCC	XX.XX
This makes experiments reproducible and easier to compare.

💡 Why These Design Choices?
Why EfficientNet-B0?
It provides a strong accuracy-to-compute trade-off and is well suited for transfer learning.

Why transfer learning?
Training a deep CNN from scratch would require considerably more data. ImageNet pretraining provides useful low- and mid-level visual representations.

Why two-stage training?
The classification head can first adapt to the task before the pretrained backbone is carefully fine-tuned.

Why CLAHE?
It improves local contrast while limiting excessive contrast amplification.

Why class weighting?
The training dataset is imbalanced, so class weighting prevents the majority class from dominating optimization.

Why threshold optimization?
The best operating point for a screening-oriented classifier is not necessarily 0.5.

Why validation-only thresholding?
It prevents test-set leakage and provides a more honest final evaluation.

Why Grad-CAM?
It provides a visual explanation of the image regions influencing the prediction.

⚠️ Limitations
Not a clinical diagnostic system.
The model is intended for research and educational purposes.

Public-dataset limitation.
Strong performance on a public dataset does not guarantee clinical performance.

No external validation.
Independent datasets are required to evaluate generalization.

Dataset bias.
Dataset composition, acquisition conditions, and labeling practices may introduce bias.

Grad-CAM limitations.
Highlighted regions should not be interpreted as medically causal.

Threshold criterion.
The 95% sensitivity target is a project-level objective and not a clinically validated requirement.

🔮 Future Work
External validation on independent datasets

Near-duplicate detection using perceptual hashing

Multi-class pneumonia classification

Uncertainty estimation using ensembles / MC Dropout

Model distillation

TFLite quantization

Improved probability calibration

Clinical validation with expert radiologist annotations

📜 License
This project is licensed under the MIT License.

See LICENSE for details.

<div align="center">

🫁 PneumoVision
Medical Imaging · Computer Vision · Transfer Learning · Explainable AI

Built as an end-to-end machine learning engineering project.

</div>

