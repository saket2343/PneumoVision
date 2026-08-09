
/
README.md


🚀 PneumoVision
AI-Powered Pneumonia Detection from Chest X-Rays






📖 Overview
PneumoVision is an end-to-end deep learning system for detecting pneumonia from chest X-ray images.

The project uses a transfer-learning-based EfficientNetB0 classifier combined with medical-image preprocessing, CLAHE enhancement, data augmentation, class-weight-aware training, two-stage fine-tuning, validation-based threshold optimization, model evaluation, and Grad-CAM explainability.

The complete pipeline covers:

Dataset validation and integrity checking

Patient-level data separation

Image preprocessing

CLAHE contrast enhancement

Data augmentation

EfficientNetB0 transfer learning

Two-stage model training

Class imbalance handling

Validation-based threshold optimization

Held-out test evaluation

Grad-CAM explainability

Streamlit deployment
