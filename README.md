# 🚀 PneumoVision
### AI-Powered Pneumonia Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNetB0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

**PneumoVision** is a deep learning-based medical imaging system designed to detect **pneumonia from chest X-ray images**.

The system uses an **EfficientNetB0-based CNN** with a two-stage transfer learning strategy, image preprocessing, CLAHE enhancement, data augmentation, class-aware training, validation-based threshold optimization, and Grad-CAM explainability.

Unlike a simple binary image classifier, the pipeline focuses on building a complete and reproducible ML workflow from **data validation and preprocessing to model training, threshold selection, evaluation, and deployment through Streamlit**.

The system currently supports:

🩻 **Chest X-ray Images**  
🫁 **NORMAL / PNEUMONIA Classification**  
📊 **Model Performance Evaluation**  
🔥 **Grad-CAM Visual Explanations**  
🌐 **Streamlit Web Interface**

---

## ✨ Key Features

- 🩻 Automated pneumonia detection from chest X-rays
- 🧠 EfficientNetB0 transfer-learning backbone
- 🔄 Two-stage training:
  - Frozen-backbone classification head training
  - Selective backbone fine-tuning
- 🖼️ Image resizing and RGB preprocessing
- 🔬 CLAHE-based contrast enhancement
- 🎯 Albumentations-based data augmentation
- ⚖️ Class-weight-aware training
- 🧪 Patient-level dataset integrity validation
- 🔍 Duplicate and cross-split leakage detection
- 📈 Validation-based threshold optimization
- 📊 ROC-AUC, F1, sensitivity, specificity and MCC evaluation
- 🔥 Grad-CAM model interpretability
- 🌐 Interactive Streamlit deployment
- 📄 Automated prediction/report generation

---

## 🏗️ System Architecture

```text
                 Chest X-Ray
                      │
                      ▼
             Image Validation
                      │
                      ▼
          Preprocessing Pipeline
          ┌──────────────────────┐
          │ Resize                │
          │ RGB Conversion        │
          │ CLAHE Enhancement     │
          └──────────────────────┘
                      │
                      ▼
             Data Augmentation
                      │
                      ▼
             EfficientNetB0
                      │
             ┌────────┴────────┐
             │                 │
        Stage 1             Stage 2
     Frozen Backbone      Fine-Tuning
             │                 │
             └────────┬────────┘
                      ▼
              Pneumonia Score
                      │
                      ▼
          Validation Threshold
                      │
             ┌────────┴────────┐
             ▼                 ▼
          NORMAL          PNEUMONIA
             │                 │
             └────────┬────────┘
                      ▼
                Grad-CAM
                      │
                      ▼
             Streamlit Interface

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
│   ├── confusion_matrix/
│   └── ...
│
├── src/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── trainer.py
│   └── utils.py
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

🧹 Dataset Validation

Before training, the dataset is checked for:

Corrupted/unreadable images
Class distribution
Exact duplicate images
Cross-split duplicate leakage
Patient-level train/validation/test overlap

The final dataset uses patient-level separation so that the same pneumonia patient does not appear across multiple splits.

Current cleaned dataset:

Split	NORMAL	PNEUMONIA	Total
Train	1,213	2,684	3,897
Validation	135	389	524
Test	234	390	624

The test set is kept completely separate from model training and threshold selection.


🧠 Model

The classifier uses EfficientNetB0 with transfer learning.

Stage 1 — Classification Head

The pretrained backbone is frozen and only the newly added classification layers are trained.

Pretrained EfficientNetB0
          ↓
Global Average Pooling
          ↓
Dropout
          ↓
Dense Layers
          ↓
Binary Classification
Stage 2 — Fine-Tuning

After the classification head has learned useful task-specific representations, selected layers of the EfficientNetB0 backbone are unfrozen and fine-tuned using a lower learning rate.

This allows the model to adapt pretrained visual features to chest X-ray characteristics while reducing the risk of destroying useful pretrained representations.

🖼️ Preprocessing

Each image passes through the preprocessing pipeline before being provided to the model.

Preprocessing steps
Input X-Ray
    ↓
RGB Conversion
    ↓
Resize to 224 × 224
    ↓
CLAHE Enhancement
    ↓
Normalization
    ↓
Augmentation (training only)

CLAHE is used to improve local contrast and make relevant structures in the X-ray easier for the model to learn.

🔄 Data Augmentation

Training images are augmented using an Albumentations pipeline.

Augmentation helps the model become more robust to variations in:

Image orientation
Contrast
Spatial transformations
Imaging conditions

Validation and test images are not randomly augmented, ensuring consistent evaluation.

⚖️ Class Imbalance

The original dataset contains substantially more pneumonia images than normal images.

The training pipeline therefore computes class weights from the training distribution:

weight(class) =
N / (number_of_classes × N_class)

This prevents the model from simply favoring the majority class during optimization.

🎯 Threshold Optimization

Instead of automatically using the default binary classification threshold of 0.5, the threshold is selected using the validation set.

The threshold search evaluates:

Accuracy
Precision
Recall / Sensitivity
Specificity
F1-score
Balanced Accuracy
MCC
Youden's J

The selected threshold prioritizes maintaining high pneumonia sensitivity while improving specificity.

Important

The test set is not used during threshold selection.

The final threshold is selected from validation predictions and then applied once to the held-out test set.

🔥 Model Explainability

PneumoVision uses Grad-CAM to visualize the image regions contributing to the model's prediction.

Chest X-Ray
     ↓
Model Prediction
     ↓
Grad-CAM
     ↓
Activation Heatmap
     ↓
Overlay on X-Ray

This provides a visual explanation of which regions influenced the model's decision.

Grad-CAM provides model interpretability and should not be considered a clinical diagnosis.

📊 Evaluation Metrics

The final model is evaluated on a completely held-out test set using:

Metric	Purpose
Accuracy	Overall classification performance
Precision	Reliability of positive predictions
Sensitivity	Ability to detect pneumonia
Specificity	Ability to identify normal X-rays
F1-Score	Balance between precision and recall
ROC-AUC	Ranking/discrimination ability
Balanced Accuracy	Performance across both classes
MCC	Robust correlation-based metric

The confusion matrix is also generated automatically.

🚀 Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Pneumonia-Detection

Create a virtual environment:

python3 -m venv .venv

Activate it:

macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt
📁 Dataset Setup

Place the dataset in the following structure:

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

Before training, verify the dataset:

python check_data_integrity.py
🏋️ Training

Run the complete training pipeline:

python train.py

The training process performs:

Dataset Loading
      ↓
Preprocessing
      ↓
Augmentation
      ↓
Stage 1 Training
      ↓
Stage 2 Fine-Tuning
      ↓
Best Model Checkpoint
      ↓
Final Model

Models are saved under:

models/
├── best_model.keras
└── final_model.keras

The best_model.keras checkpoint is selected using the validation monitoring metric and should be used for final evaluation.

🎯 Tune Classification Threshold

After training:

python tune_threshold.py

This generates:

outputs/reports/
├── threshold_search.csv
├── threshold_search.json
├── threshold_recommendation.json
├── threshold_curve.png
└── probability_distribution_val.png

The threshold is selected using validation data only.

🧪 Final Test Evaluation

After obtaining the validation-selected threshold:

python evaluate.py \
  --model-path "models/best_model.keras" \
  --threshold <SELECTED_THRESHOLD>

Example:

python evaluate.py \
  --model-path "models/best_model.keras" \
  --threshold 0.85

The evaluation produces:

outputs/
├── reports/
│   └── test_evaluation_*.json
└── confusion_matrix/
    └── confusion_matrix_*.png
🌐 Streamlit Application

Launch the interactive application:

streamlit run app.py

The application allows users to:

Upload a chest X-ray
Preprocess the image
Generate pneumonia probability
Apply the optimized classification threshold
Display NORMAL/PNEUMONIA prediction
Visualize Grad-CAM explanations
Generate prediction reports
🧪 Testing

Run the project's automated tests:

pytest tests/ -v

For coverage:

pytest tests/ -v --cov=src --cov-report=term-missing
📈 Results

The model is evaluated on 624 held-out chest X-rays:

NORMAL     : 234
PNEUMONIA  : 390
TOTAL      : 624

The evaluation emphasizes clinically relevant metrics such as:

Sensitivity
Specificity
F1-score
ROC-AUC
Balanced Accuracy
MCC

Final reported metrics should be updated here using the results from the clean, patient-independent test evaluation.

🛡️ Data Leakage Prevention

A major focus of the pipeline is preventing evaluation leakage.

The dataset validation checks:

Image duplicates
       ↓
Cross-split duplicates
       ↓
Patient-level overlap
       ↓
Train / Validation / Test isolation

The cleaned dataset achieves:

Train ↔ Validation patients : 0 overlap
Train ↔ Test patients       : 0 overlap
Validation ↔ Test patients  : 0 overlap
Cross-split duplicates      : 0
Unreadable images           : 0

This ensures that the final evaluation better represents patient-independent generalization.

⚙️ Configuration

Model and training parameters can be controlled through config.py and CLI arguments.

Example:

python train.py \
  --epochs 50 \
  --fine-tune-epochs 15 \
  --backbone EfficientNetB0

Skip fine-tuning:

python train.py --no-fine-tune

Disable dataset caching:

python train.py --no-cache
📊 Project Pipeline
             DATASET
                │
                ▼
       Data Integrity Check
                │
                ▼
      Patient-Level Splitting
                │
                ▼
       Preprocessing + CLAHE
                │
                ▼
          Augmentation
                │
                ▼
        EfficientNetB0
                │
        ┌───────┴────────┐
        ▼                ▼
    Stage 1           Stage 2
  Frozen Backbone   Fine-Tuning
        │                │
        └───────┬────────┘
                ▼
          Best Model
                │
                ▼
      Validation Threshold
                │
                ▼
       Held-Out Test Set
                │
        ┌───────┴────────┐
        ▼                ▼
   Performance       Grad-CAM
    Metrics          Explainability
        │                │
        └───────┬────────┘
                ▼
           Streamlit
🔬 Future Improvements
Experiment with EfficientNetB3/B4 backbones
Improve normal-class specificity
Evaluate focal loss and oversampling strategies
Probability calibration
Stronger patient-level stratification
External dataset validation
Model ensemble experiments
Dockerized deployment
CI/CD integration
Cloud deployment
⚠️ Disclaimer

This project is intended for educational and research purposes only.

It is not a medical diagnostic device and should not be used as a substitute for evaluation by a qualified healthcare professional.

👨‍💻 Author

Saket Pandey

Built as an end-to-end deep learning project for automated pneumonia detection from chest X-ray images.

📄 License

This project is licensed under the MIT License.
