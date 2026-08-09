# PneumoVision 🫁

AI-powered pneumonia detection from chest X-rays — EfficientNet-based binary classifier with Grad-CAM explainability and a Streamlit demo.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)]()
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)]()

PneumoVision is an end-to-end pipeline for classifying chest X-rays as NORMAL or PNEUMONIA. It uses an ImageNet-pretrained EfficientNet backbone, a two-stage training strategy, validation-based thresholding, Grad-CAM explanations, and a Streamlit application for interactive use.

> ⚠️ Medical disclaimer: This project is for research and educational purposes only. It is not a certified medical device and should not be used for clinical diagnosis or treatment.

---

## Demo

Add a screenshot or GIF here (recommended path: `docs/demo.gif` or `docs/screenshot.png`). A short visual helps users quickly understand the project.

## Table of Contents
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Preprocessing & Augmentation](#preprocessing--augmentation)
- [Training](#training)
- [Evaluation & Thresholding](#evaluation--thresholding)
- [Grad-CAM & Explainability](#grad-cam--explainability)
- [Inference & Streamlit App](#inference--streamlit-app)
- [Testing & CI](#testing--ci)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Quickstart

1. Clone and create a virtual environment:

```bash
git clone https://github.com/saket2343/PneumoVision.git
cd PneumoVision
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Validate dataset:

```bash
python check_data_integrity.py
```

3. Train (default two-stage training):

```bash
python train.py
```

4. Tune threshold:

```bash
python tune_threshold.py
```

5. Evaluate:

```bash
python evaluate.py --model-path models/best_model.keras --threshold <recommended-threshold>
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Usage

Predict a single image (optional TTA and heatmap):

```bash
python predict.py --image path/to/xray.jpg
python predict.py --image path/to/xray.jpg --tta
python predict.py --image path/to/xray.jpg --save-heatmap outputs/heatmaps/example.png
```

Programmatic example:

```python
from src.inference import Predictor
p = Predictor(model_path="models/best_model.keras", threshold=0.42)
prob = p.predict("path/to/xray.jpg")
print(f"Pneumonia probability: {prob:.2f}")
```

---

## Architecture

- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Classification head:
  - GlobalAveragePooling2D
  - Dropout(0.40)
  - Dense(256) -> BatchNorm -> Swish
  - Dropout(0.30)
  - Dense(128) -> BatchNorm -> Swish
  - Dense(1) -> Sigmoid
- Training strategy: two stages
  - Stage 1 (feature extraction): frozen backbone, train head, 50 epochs, lr=1e-3
  - Stage 2 (fine-tuning): unfreeze last ~40 layers, 15 epochs, lr=1e-5

---

## Dataset

Expected layout:

```
dataset/
  train/
    NORMAL/ PNEUMONIA/
  val/
    NORMAL/ PNEUMONIA/
  test/
    NORMAL/ PNEUMONIA/
```

Current verified split: 5,856 images (train 4,709 | val 523 | test 624). The dataset is imbalanced toward the PNEUMONIA class.

---

## Preprocessing & Augmentation

- Deterministic preprocessing for validation/test:
  - Convert to RGB
  - Resize to 224×224
  - CLAHE (clip limit = 2.0, tile grid = 8×8)
  - Convert to float32 and normalize to [0, 1]
  - Apply backbone-specific preprocessing (e.g., multiply by 255 if backbone expects [0,255])
- Train-only augmentations (conservative): random resized crop, horizontal flip, small affine transforms, brightness/contrast, Gaussian noise

---

## Training

Key hyperparameters and defaults are in `config.py`.

- Optimizer: AdamW, weight decay = 1e-4
- Batch size: 32
- Stage 1: 50 epochs, lr = 1e-3
- Stage 2: 15 epochs, lr = 1e-5
- Loss: Binary Cross-Entropy (optional focal loss)
- Imbalance handling: class weights, optional oversampling, focal loss
- Mixed precision: enabled
- Checkpoint selection: validation ROC-AUC

---

## Evaluation & Thresholding

- Reported metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1, Balanced Accuracy, MCC, ROC-AUC
- Threshold search: 0.01 → 0.99 (step 0.01). Primary rule: among thresholds achieving Sensitivity ≥ 95%, select the threshold with the highest Specificity. Alternatives supported: Youden's J, best F1, best balanced accuracy.
- The test set is never used for threshold selection — thresholding is performed on validation only.

---

## Grad-CAM & Explainability

Grad-CAM visualizations are available via `src/gradcam.py`. Use Grad-CAM overlays to inspect model attention on chest X-rays; note these are interpretability aids and not clinical evidence.

---

## Inference & Streamlit App

The Streamlit app (`app.py`) supports uploading an X-ray, running inference, visualizing Grad-CAM heatmaps, toggling TTA, adjusting threshold sliders, and downloading a PDF report. The recommended threshold is read from `outputs/reports/threshold_recommendation.json`.

---

## Testing & CI

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

CI is configured in `.github/workflows/ci.yml`. Recommended CI steps: black --check, flake8, pytest. Add badges to this README when CI is enabled.

---

## Contributing

Suggestions:
- Add `CONTRIBUTING.md` describing how to run experiments, test locally, coding style, and PR expectations.
- Add an issue/PR template and `CODE_OF_CONDUCT.md`.
- Require tests for changes that touch processing/training code.

If you'd like, I can create a minimal `CONTRIBUTING.md` and PR for review.

---

## Suggested improvements

- Add a short demo GIF / screenshot in `docs/` and link it in this README.
- Add CI and coverage badges near the top.
- Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and issue/PR templates.
- Add a small example notebook that runs inference on a sample image (includes Grad-CAM) so users can reproduce results quickly.
- Consider hosting a live demo (Streamlit Cloud / Heroku) and add a live demo link.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or to report issues, open an issue at: https://github.com/saket2343/PneumoVision/issues
