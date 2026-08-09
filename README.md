# 🚀 PneumoVision
### AI-Powered Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNetB0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

📌 Project Overview

Problem

Pneumonia detection from chest radiographs requires careful visual interpretation and can be time-consuming, particularly when screening large numbers of images. The goal of this project is to build a reproducible computer-vision pipeline that can distinguish between normal and pneumonia chest X-rays while also exposing the regions that influenced the model's prediction.

Objectives

Classify chest X-rays into NORMAL and PNEUMONIA.

Use ImageNet-pretrained EfficientNet-B0 to leverage transfer learning.

Improve robustness through CLAHE, medically conservative augmentation, dropout, batch normalization, class weighting, and fine-tuning.

Train in two controlled stages: 50 epochs of feature extraction + 15 epochs of fine-tuning by default.

Select the operating threshold using the validation set only.

Perform the final metric calculation on a held-out test set without threshold search.

Provide Grad-CAM visual explanations for individual predictions.

Analyze false positives, false negatives, confidence distributions, and confusion categories.

Provide a browser-based Streamlit application and downloadable PDF prediction reports.

Maintain reproducibility with deterministic seeds, tests, experiment logs, and GitHub Actions CI.

🏗️ End-to-End Pipeline

                    ┌─────────────────────┐
                    │     Chest X-Ray     │
                    └──────────┬──────────┘
                               ↓
                    Image validation/loading
                               ↓
                    Resize → 224 × 224
                               ↓
                    RGB conversion
                               ↓
                         CLAHE
                               ↓
                    Normalize to [0, 1]
                               ↓
              ┌────────────────────────────┐
              │ Train only: augmentation   │
              │ flip / affine / crop /     │
              │ noise / brightness etc.    │
              └─────────────┬──────────────┘
                            ↓
              Backbone-aware preprocessing
                            ↓
                 EfficientNet-B0
                 ImageNet pretrained
                            ↓
                 Global Average Pooling
                            ↓
                      Dropout 0.4
                            ↓
                   Dense 256 + BN
                            ↓
                        Swish
                            ↓
                      Dropout 0.3
                            ↓
                   Dense 128 + BN
                            ↓
                        Swish
                            ↓
                    Dense 1 + Sigmoid
                            ↓
              PNEUMONIA probability p
                            ↓
             Validation-tuned threshold
                            ↓
              NORMAL / PNEUMONIA
                    ↙             ↘
             Confidence       Grad-CAM
                                  ↓
                          Visual explanation
                                  ↓
                         Streamlit / PDF report

🧠 Model Architecture

Backbone

EfficientNet-B0, initialized with ImageNet pretrained weights.

The backbone is initially frozen so that the newly added classification head can learn the pneumonia-specific decision boundary without immediately destroying useful pretrained representations.

Classification head

EfficientNet-B0
      ↓
GlobalAveragePooling2D
      ↓
Dropout(0.4)
      ↓
Dense(256)
      ↓
BatchNormalization
      ↓
Swish
      ↓
Dropout(0.3)
      ↓
Dense(128)
      ↓
BatchNormalization
      ↓
Swish
      ↓
Dense(1)
      ↓
Sigmoid

The implementation also supports EfficientNet-B1/B2/B3, DenseNet121, ResNet50, and MobileNetV3Large for controlled backbone comparison.

Optional attention modules are available:

SE — Squeeze-and-Excitation

CBAM — Convolutional Block Attention Module

Example:

python train.py --backbone EfficientNetB3 --attention se

🔥 Two-Stage Training Strategy

Training is deliberately split into two stages.

Stage 1 — Feature extraction

Default: 50 epochs

The pretrained EfficientNet-B0 backbone is frozen.

Only the newly added classification head learns:

ImageNet backbone = frozen
Classification head = trainable
Learning rate = 1e-3

This stage lets the classifier learn how pretrained visual features map to the two target classes without aggressively modifying the pretrained representation.

Stage 2 — Fine-tuning

Default: 15 epochs

The last 40 backbone layers are unfrozen by default:

Backbone = partially trainable
Last N layers = trainable
Learning rate = 1e-5

The learning rate is reduced by two orders of magnitude so that the pretrained representation is adapted gradually rather than overwritten.

Batch-normalization layers are kept appropriately constrained during fine-tuning to protect pretrained running statistics.

Why two stages?

Stage 1
Frozen backbone
High LR
      ↓
Learn task-specific classifier
      ↓
Stage 2
Partial backbone unfreezing
Low LR
      ↓
Adapt high-level features to chest X-rays

This is generally more stable than immediately fine-tuning the entire pretrained network at a high learning rate.

⚙️ Training Configuration

Parameter

Default

Backbone

EfficientNet-B0

Input size

224 × 224 × 3

Batch size

32

Stage 1 epochs

50

Stage 2 epochs

15

Stage 1 learning rate

1e-3

Stage 2 learning rate

1e-5

Optimizer

AdamW

Weight decay

1e-4

Gradient clipping

1.0

Dropout

0.4 / 0.3

Dense layers

256 / 128

Loss

Binary Cross-Entropy

Optional loss

Binary Focal Cross-Entropy

Mixed precision

Enabled

Seed

42

Early stopping

patience = 8

LR reduction

factor = 0.5, patience = 4

Model selection

Best validation ROC-AUC

🧪 Data & Preprocessing

Dataset structure

dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

The project uses a standard chest X-ray pneumonia dataset structure with binary labels.

Latest verified split

The dataset preparation used in the project contains:

Class

Train

Validation

Test

Total

NORMAL

1,214

135

234

1,583

PNEUMONIA

3,495

388

390

4,273

Total

4,709

523

624

5,856

The test set therefore contains 624 images, with 390 pneumonia and 234 normal cases.

Preprocessing

Each image goes through:

Image loading and validation

RGB conversion

Resize to 224 × 224

CLAHE contrast enhancement

Float32 conversion

Normalization to [0, 1]

Training-only augmentation

Backbone-specific preprocessing inside the model

CLAHE

Contrast Limited Adaptive Histogram Equalization is used to enhance local contrast while limiting excessive amplification of noise.

Configuration:

clip limit = 2.0
tile grid = 8 × 8

🚨 Important Preprocessing Bug & Fix

A major preprocessing issue was identified and corrected.

The dataset pipeline outputs images in:

[0, 1]

However, Keras EfficientNet implementations contain their own internal rescaling layer:

[0, 255] → [0, 1]

Passing already-normalized [0,1] images directly into EfficientNet therefore caused double normalization.

Effect of the bug

The effective input became approximately:

[0,1] → /255 → [0, 0.0039]

This collapsed feature magnitudes and greatly reduced inter-image separability.

The observed failure mode was particularly revealing:

the model tended to predict PNEUMONIA for almost every image, even when the decision threshold was changed.

Corrected pipeline

The model now explicitly converts the external [0,1] representation back to [0,255] before passing it into backbones that require it.

Dataset output
[0,1]
   ↓
Model preprocessing block
× 255
   ↓
[0,255]
   ↓
EfficientNet internal preprocessing
   ↓
Correct feature representation

The fix is implemented inside src/model.py, keeping the preprocessing contract consistent across training, evaluation, inference, and Grad-CAM.

The fix also handles other supported backbones correctly:

EfficientNet family / MobileNetV3 → internal rescaling

ResNet50 → explicit ResNet preprocessing

DenseNet121 → explicit DenseNet preprocessing

A Keras serialization-safe custom preprocessing layer is used instead of a bare Lambda, and save/load round-trip tests are included.

🔄 Data Augmentation

Augmentation is applied only to the training split.

The validation and test pipelines remain deterministic so that evaluation measures the actual model rather than random augmentation effects.

The augmentation module supports transformations such as:

Random resized crop

Affine transformations

Horizontal flip

Small geometric perturbations

Brightness/contrast variation

Gaussian noise

The augmentation policy is intentionally moderate because aggressive transformations can create medically unrealistic radiographs.

⚖️ Class Imbalance Handling

The dataset is naturally imbalanced toward the PNEUMONIA class.

The project supports multiple controlled strategies:

1. Class-weighted loss

The default pipeline calculates class weights from the training distribution and passes them to model.fit().

2. Moderate oversampling

Optional train-only oversampling can be enabled:

python train.py --oversample --oversample-ratio 0.5 --no-class-weights

Oversampling never modifies validation or test data.

3. Focal loss

python train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25

Focal loss is treated as a controlled experiment rather than automatically combined with every other imbalance correction.

🎯 Threshold Optimization

The classifier outputs a probability:

P(PNEUMONIA | X-ray)

The default sigmoid cutoff of 0.5 is not automatically assumed to be optimal.

python tune_threshold.py searches thresholds from:

0.01 → 0.99
step = 0.01

using only the validation set.

The primary project selection rule is:

Among thresholds achieving sensitivity ≥ 95%, select the threshold with the highest specificity.

This reflects the project's modeling objective of prioritizing pneumonia sensitivity while reducing unnecessary false positives.

Other thresholds are also reported:

Youden's J

Best F1

Best balanced accuracy

Correct evaluation protocol

Training
   ↓
Best checkpoint selected using validation AUC
   ↓
Validation set
   ↓
Threshold tuning
   ↓
Freeze selected threshold
   ↓
Held-out test set
   ↓
ONE final unbiased evaluation

The test set is never used to search for the threshold.

📊 Evaluation Metrics

The evaluation module reports:

Accuracy

Precision

Recall / Sensitivity

Specificity

F1-score

Balanced accuracy

Matthews Correlation Coefficient (MCC)

ROC-AUC

Confusion matrix

TP / TN / FP / FN

Predicted probability distributions

It also generates:

ROC curve

Precision-Recall curve

Confusion matrix

Error-analysis plots

Probability distribution diagnostics

Confidence-ranked mistakes

Historical baseline before the preprocessing fix

An audit of the earlier model showed approximately:

Sensitivity: 98.5%
Specificity: 65.4%

with 81 NORMAL images incorrectly classified as PNEUMONIA.

That run was associated with the now-fixed preprocessing issue and should not be presented as the final model result.

The repository's final results should be populated only from a model retrained after the preprocessing fix and evaluated through the validation-threshold → held-out-test workflow.

🔬 Error Analysis

The project does more than report aggregate metrics.

src/error_analysis.py analyzes:

False positives

Actual = NORMAL
Predicted = PNEUMONIA

False negatives

Actual = PNEUMONIA
Predicted = NORMAL

Confidence-ranked mistakes

Mistakes can be ranked by prediction confidence to identify cases where the model is highly confident but wrong.

Normal false-positive analysis

The pipeline produces a dedicated report and montage for the most problematic NORMAL images.

This is useful because reducing false positives was a specific modeling objective after the initial baseline audit.

🌡️ Probability Calibration

The project includes optional temperature scaling.

python calibrate.py --model-path models/best_model.keras

Calibration is fitted on the validation set only.

The resulting temperature can then be supplied during threshold tuning/evaluation:

python tune_threshold.py --temperature <temperature>
python evaluate.py --threshold <threshold> --temperature <temperature>

Calibration is treated as an empirical experiment: it is only used if it actually improves probability quality rather than being assumed to help.

🔍 Grad-CAM Explainability

A prediction is accompanied by a Grad-CAM heatmap.

Chest X-ray
     ↓
Model prediction
     ↓
Gradient computation
     ↓
Last convolutional feature map
     ↓
Grad-CAM weights
     ↓
Heatmap
     ↓
Overlay on original X-ray

This allows the user to inspect where the model is focusing when making a prediction.

The implementation uses TensorFlow/Keras GradientTape directly and does not depend on a PyTorch-only Grad-CAM package.

🖥️ Streamlit Application

Run:

streamlit run app.py

The web application provides:

Drag-and-drop X-ray upload

NORMAL / PNEUMONIA prediction

Prediction confidence

Decision threshold used

Original image display

Grad-CAM explanation

Optional Test-Time Augmentation

Prediction history during the session

Threshold slider

Downloadable PDF report

The application automatically loads the validation-recommended threshold from:

outputs/reports/threshold_recommendation.json

while still allowing manual threshold exploration.

🧪 Test-Time Augmentation

TTA can be enabled from the CLI:

python predict.py --image path/to/xray.jpg --tta

or from the Streamlit interface.

The model averages predictions over multiple augmented views to obtain a more robust probability estimate.

TTA is optional and should be evaluated empirically rather than assumed to improve performance.

📁 Project Structure

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
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_ModelTraining.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_GradCAM.ipynb
│
├── src/
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
├── tests/
│   ├── test_model.py
│   ├── test_model_preprocessing.py
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   ├── test_dataset_balancing_and_leakage.py
│   └── test_threshold_optimizer.py
│
├── outputs/
│   ├── confusion_matrix/
│   ├── roc/
│   ├── pr_curve/
│   ├── heatmaps/
│   ├── error_analysis/
│   └── reports/
│
├── models/
├── .github/
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

git clone <your-repository-url>
cd Pneumonia-Detection

python -m venv .venv
source .venv/bin/activate

On Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Expected core stack:

TensorFlow
Keras
NumPy
Pandas
Scikit-learn
OpenCV
Albumentations
Pillow
Matplotlib
Streamlit
pytest
Black
Flake8

▶️ Running the Project

1. Validate the dataset

python check_data_integrity.py

This checks:

Class distribution

Corrupted/unreadable images

Exact duplicate files across splits

Duplicate groups

Patient-ID overlap where filename patterns permit detection

2. Train the model

python train.py

Default:

Stage 1 → 50 epochs
Stage 2 → 15 epochs

Shorter experiment:

python train.py --epochs 30 --fine-tune-epochs 10

Skip fine-tuning:

python train.py --no-fine-tune

3. Tune the threshold

python tune_threshold.py

This generates the recommended validation threshold and supporting diagnostics.

4. Evaluate on the test set

Use the threshold produced by validation:

python evaluate.py   --model-path models/best_model.keras   --threshold <recommended-threshold>

Do not search for the threshold on the test set.

5. Run single-image inference

python predict.py --image path/to/xray.jpg

With TTA:

python predict.py --image path/to/xray.jpg --tta

Save a Grad-CAM heatmap:

python predict.py   --image path/to/xray.jpg   --save-heatmap outputs/heatmaps/example.png

6. Launch the web application

streamlit run app.py

🧪 Experiment Framework

The repository supports controlled experiments rather than changing many variables simultaneously.

Class balancing

python train.py --oversample --oversample-ratio 0.5 --no-class-weights

Focal loss

python train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25

Fine-tuning depth

python train.py --fine-tune-at-layer -20

Backbone comparison

python compare_models.py   --backbones EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3

Hyperparameter optimization

python tune.py

Optuna support includes experiments over parameters such as:

Learning rate

Dropout

Batch size

Weight decay

📈 Experiment Tracking

tune_threshold.py can log experiment metadata and results:

python tune_threshold.py   --experiment-name exp1_baseline   --notes "EfficientNet-B0 baseline"

The experiment log records information such as:

backbone
class balancing
loss
threshold
sensitivity
specificity
F1
balanced accuracy
MCC
ROC-AUC
TP
TN
FP
FN

Saved to:

outputs/reports/experiment_log.csv

🧹 Dataset Leakage Checks

The project includes explicit integrity checks because medical datasets can contain duplicate or related images.

The integrity pipeline checks:

Cross-split exact duplicates

Within-split duplicates

Patient-ID overlap when detectable from filenames

Corrupted images

Split/class counts

Exact duplicate detection uses file hashing.

Near-duplicate detection, such as the same X-ray re-exported at a different resolution or compression level, is not currently guaranteed.

📦 Model Export

export_model.py provides an export path for deployment formats such as:

TensorFlow SavedModel

ONNX

TFLite

This creates an extension point for lightweight/edge deployment and future model optimization.

🔁 CI/CD

GitHub Actions is configured under:

.github/workflows/ci.yml

The workflow checks code quality and runs automated tests.

Typical checks include:

flake8 .
black --check .
pytest tests/ -v

This ensures changes to preprocessing, model construction, thresholding, and data handling do not silently break the pipeline.

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

Save/load model compatibility

A particularly important regression test verifies the EfficientNet preprocessing issue and ensures that the corrected model can be saved, loaded, and used for inference.

📊 Results Reporting

After retraining the corrected model, the recommended reporting workflow is:

python train.py
python tune_threshold.py
python evaluate.py     --model-path models/best_model.keras     --threshold <recommended-threshold>

Then populate the final results table from the generated test report:

Metric

Final Test Result

Accuracy

Generated by evaluate.py

Precision

Generated by evaluate.py

Recall / Sensitivity

Generated by evaluate.py

Specificity

Generated by evaluate.py

F1-score

Generated by evaluate.py

Balanced Accuracy

Generated by evaluate.py

MCC

Generated by evaluate.py

ROC-AUC

Generated by evaluate.py

Important: Do not reuse the historical pre-fix baseline numbers as the final model result. The final README metrics should come from the retrained, corrected model and the held-out test evaluation.

💡 Key Engineering Decisions

Why EfficientNet-B0?

EfficientNet-B0 provides a strong accuracy/compute trade-off and makes transfer learning practical on a moderate-sized chest X-ray dataset.

Why transfer learning?

The dataset is much smaller than the datasets typically required to train a deep CNN from scratch. ImageNet-pretrained convolutional features provide useful low- and mid-level visual representations.

Why CLAHE?

Chest radiographs can contain subtle local intensity patterns. CLAHE improves local contrast while limiting excessive contrast amplification.

Why class weights?

The training distribution is imbalanced, so class weighting prevents the optimization objective from being dominated by the majority class.

Why threshold tuning?

A sigmoid probability of 0.5 is not guaranteed to be the best operating point for an imbalanced medical screening problem. The project therefore separates probability estimation from the final decision threshold.

Why validation-only threshold tuning?

Using the test set to choose a threshold leaks information from the evaluation set and produces an optimistically biased estimate.

Why Grad-CAM?

A classification score alone does not show which part of an X-ray influenced the model. Grad-CAM provides a visual explanation that can be inspected alongside the prediction.

🛠️ Important Bugs Found and Fixed

1. EfficientNet double-normalization

Problem: [0,1] images were passed into a backbone that internally rescales them again.

Fix: Backbone-aware preprocessing was moved inside the model.

2. Threshold hard-coded to 0.5

Problem: Earlier inference paths always used:

probability >= 0.5

even after a validation-derived threshold had been selected.

Fix: InferenceEngine, predict.py, and app.py now support/load the frozen recommended threshold.

3. Albumentations 2.x API mismatch

Fresh installations of Albumentations 2.x changed constructor APIs used by the augmentation pipeline.

The project was updated for:

RandomResizedCrop(size=...)
GaussNoise(std_range=...)

and deprecated transformation usage was consolidated appropriately.

4. Keras preprocessing serialization

A bare Lambda(preprocess_input) was unsafe for Keras model serialization.

The project now uses registered custom preprocessing layers for supported backbones, allowing:

save → load → predict

to work correctly.

🧩 Stretch Goals Already Implemented

Transfer learning

Two-stage fine-tuning

CLAHE preprocessing

Class-weighted training

Optional oversampling

Focal loss

Threshold optimization

Temperature scaling

Grad-CAM

Error analysis

Test-Time Augmentation

SE attention

CBAM attention

Multi-backbone comparison

Optuna hyperparameter tuning

Model export

Streamlit deployment

PDF prediction reports

Unit/integration tests

GitHub Actions CI

Experiment tracking

Dataset leakage checks

🔮 Future Work

Multi-class classification:

Normal

Bacterial pneumonia

Viral pneumonia

Other pulmonary conditions

External validation on NIH ChestX-ray14 or another independent dataset

Perceptual-hash detection for near-duplicate images

Uncertainty estimation using MC Dropout or ensembles

Model distillation for mobile/edge inference

TFLite quantization

Better calibration and uncertainty reporting

Prospective clinical validation with expert radiologist labels

📚 Reproducibility

The project centralizes configuration in config.py.

The default seed is:

42

src/utils.py seeds Python, NumPy, and TensorFlow/Keras and requests deterministic operations where supported.

This makes experiments easier to reproduce and compare.

⚠️ Limitations

This is a research/portfolio project, not a certified medical device.

High validation/test performance on a public dataset does not establish clinical validity.

Exact duplicate detection does not guarantee detection of all near-duplicates.

The 95% sensitivity threshold is a project modeling criterion, not a clinically validated requirement.

Grad-CAM is an explanation aid, not proof that the highlighted region is medically causal.

External validation is required before making claims about generalization to other hospitals, scanners, populations, or acquisition protocols.

📜 License

This project is released under the MIT License. See LICENSE for details.

👨‍💻 Project Summary

Pneumonia Detection from Chest X-Rays is an end-to-end computer-vision project combining:

Medical Image Processing
        +
Transfer Learning
        +
Two-Stage Fine-Tuning
        +
Class-Imbalance Handling
        +
Threshold Optimization
        +
Probability Calibration
        +
Explainable AI
        +
Error Analysis
        +
Automated Testing
        +
CI/CD
        +
Streamlit Deployment

The main emphasis is not only on obtaining a classifier, but on building a reproducible, auditable, explainable, and deployable ML pipeline around it.
