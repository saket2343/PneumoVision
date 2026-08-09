# Pneumonia Detection from Chest X-Rays

Deep learning system for classifying chest X-rays as **NORMAL** or **PNEUMONIA**, built with transfer learning on EfficientNet-B0, explained with Grad-CAM, and deployed as a Streamlit web app.

> ⚠️ **Disclaimer**: This is a research/portfolio project. It is **not** a certified medical device and must never be used for real clinical diagnosis. Always consult a qualified radiologist or physician.

---

## ⚠️ If you trained a model before 2026-08-08, retrain it

A real bug was found and fixed in the input preprocessing (full writeup below). Any `models/best_model.keras` / `models/final_model.keras` saved before this fix was trained on corrupted input and **must be discarded and retrained** — see [Preprocessing Fix](#preprocessing-fix-2026-08-08) below.

---

## Specificity / False-Positive Reduction Infrastructure (2026-08-09)

A second audit targeted a different problem: the trained model was biased toward predicting PNEUMONIA (baseline test run: sensitivity 98.5%, but specificity only 65.4% — 81 NORMAL images misclassified). This section documents what was actually verified in the existing code vs. what was newly built.

### A note on scope

This environment has the project's code, but not the actual 5,856-image dataset, a trained checkpoint, or a GPU — so "run 10 controlled experiments and compare validation metrics" isn't something this environment can execute end-to-end. What follows is: (1) an honest inspection of the existing pipeline against the audit's specific claims, (2) one corrected false premise, (3) a real bug found and fixed (the Streamlit app and CLI inference were silently hard-coding threshold=0.5, ignoring `tune_threshold.py` entirely), and (4) working, tested infrastructure for every experiment the audit described, so you can run them yourself and get real numbers.

### Inspection findings (verified against the actual code, not assumed)

- **Class weights**: `_compute_class_weights()` in `src/dataset.py` uses the correct formula and is passed into `model.fit()` in both training stages — confirmed correct.
- **EarlyStopping / ModelCheckpoint**: both monitor `val_auc`, `restore_best_weights=True` — confirmed correct, matches the requirement exactly.
- **Threshold selection**: already validation-only, sensitivity≥95%-constrained + max-specificity — confirmed correct, matches the requirement exactly.
- **The described `self.oversample == oversample` bug does not exist** in this codebase — there was no `oversample` parameter at all before this round. Rather than pretend to "fix" something that wasn't there, oversampling was implemented correctly from scratch (see below).
- **A real, previously-undiscovered bug**: `src/inference.py` hard-coded `pred_idx = int(prob >= 0.5)` — meaning `app.py` and `predict.py` always classified at threshold 0.5 regardless of what `tune_threshold.py` recommended. Fixed: `InferenceEngine` now takes a `threshold` parameter, and both `predict.py` and `app.py` load it automatically from `outputs/reports/threshold_recommendation.json` (with the app showing its provenance and letting you override via a slider).

### New experiment infrastructure

| Experiment | What was added |
|---|---|
| 2 — Class balancing | `PneumoniaDataset(oversample=True, oversample_ratio=...)` — train-only, seeded, replicates minority-class files up to a target ratio (not forced 50/50). `train.py --oversample --oversample-ratio 0.5 --no-class-weights` lets you test oversampling and class-weighting independently or combined (a warning fires if you combine focal loss's own class balancing with non-uniform `class_weight`, since that compounds two corrections at once). |
| 3 — Augmentation | Already moderate and medically reasonable (see inspection above) — unchanged. |
| 4 — Focal loss | `train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25`, via `tf.keras.losses.BinaryFocalCrossentropy`. |
| 5 — Fine-tuning depth | `train.py --fine-tune-at-layer -N` (was already configurable in `config.py`, now exposed as a CLI flag for quick sweeps). |
| 6 — Threshold optimization | Unchanged (already correct) — `tune_threshold.py`. |
| 7 — Calibration | New `calibrate.py`: fits temperature scaling on validation only, reports whether it actually improves NLL/Brier (doesn't assume it will), flags degenerate fits (temperature pinned at the search boundary — a sign of an over-easy/small validation set, not a real calibration factor). `--temperature` flag added to `tune_threshold.py` and `evaluate.py` to apply a chosen value. |
| 8 — NORMAL false-positive analysis | `ErrorAnalyzer.top_normal_false_positives()` + `plot_normal_fp_montage()` — wired into `tune_threshold.py`, producing `outputs/reports/top_normal_false_positives.{json,png}` automatically on every run. |
| 9 — Error analysis by confusion category | `confusion_category_probability_summary()` in `src/threshold_optimizer.py` — splits probabilities into TN/FP/TP/FN and reports percentiles for each, saved to `outputs/reports/confusion_category_probabilities.json`. |
| 10 — Model comparison | `model.py` now supports `EfficientNetB1`/`B2` (verified against the actual Keras layer graph, same internal-rescaling pattern as B0/B3) — `compare_models.py` defaults to B0–B3. |
| Patient-level & duplicate leakage | `check_data_integrity.py` now also reports patient-ID overlap across splits (parsing the `person<ID>_...` PNEUMONIA filename convention; NORMAL files are correctly reported as unmatched rather than a pattern being assumed) and full duplicate-group detection (within-split and cross-split, not just cross-split). Reports only — never deletes or moves files. |
| Experiment tracking | `src/experiment_tracker.py` + `tune_threshold.py --experiment-name <name> --notes <text>` appends one row per run to `outputs/reports/experiment_log.csv` (backbone, class balancing, loss, threshold, sensitivity, specificity, F1, balanced accuracy, MCC, ROC-AUC, TP/TN/FP/FN). |

### How to actually run the experiment plan

```bash
python check_data_integrity.py                    # confirm distribution, corrupted files, patient/duplicate leakage FIRST

# Experiment 1: baseline (current behavior, already correct — just confirm it)
python train.py
python tune_threshold.py --experiment-name exp1_baseline

# Experiment 2: class balancing variants (run ONE at a time, compare via experiment_log.csv)
python train.py --oversample --oversample-ratio 0.5 --no-class-weights
python tune_threshold.py --experiment-name exp2b_oversample_only

python train.py --no-class-weights          # C: no correction at all, as a control
python tune_threshold.py --experiment-name exp2_control_no_balancing

# Experiment 4: focal loss
python train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25
python tune_threshold.py --experiment-name exp4_focal_loss

# Experiment 5: shallower fine-tuning
python train.py --fine-tune-at-layer -20
python tune_threshold.py --experiment-name exp5_shallow_finetune

# Experiment 7: calibration (fit on validation, review before using)
python calibrate.py --model-path models/best_model.keras
python tune_threshold.py --temperature <value from calibrate.py>

# Experiment 10: backbone comparison
python compare_models.py --backbones EfficientNetB0 EfficientNetB1 EfficientNetB2 EfficientNetB3

# Compare all logged experiments
cat outputs/reports/experiment_log.csv
```

Then, once you've picked a winner from `experiment_log.csv` using validation metrics only:

```bash
python evaluate.py --model-path models/best_model.keras --threshold <winning threshold>   # the ONE final test run
streamlit run app.py   # now shows Prediction / Confidence / Threshold Used, sourced from threshold_recommendation.json
```

### What this round did NOT do

- Did not run any of the above on real data — no dataset or GPU in this environment. Every piece was integration-tested against a small synthetic dataset (a real tiny model was trained through the full pipeline: oversampling, focal loss, calibration, threshold tuning with experiment tracking, final evaluation, and CLI inference all executed successfully end-to-end and are covered by `tests/test_dataset_balancing_and_leakage.py`).
- Did not delete, move, or re-split any dataset files, per the audit brief's explicit instruction — `check_data_integrity.py` only reports.
- Did not change anything about the augmentation pipeline (Experiment 3) — it was already checked and found appropriately moderate.

---

## Problem Statement

Pneumonia is one of the leading causes of death worldwide, and early diagnosis significantly improves outcomes. Diagnosis today relies on radiologists manually reading chest X-rays — time-consuming and subject to inter-observer variability, especially in resource-limited settings. This project builds an AI-assisted classifier that is both **accurate** and **explainable**: every prediction ships with a Grad-CAM heatmap showing which regions of the X-ray drove the decision.

## Architecture

```
Chest X-ray
    │
    ▼
Image Validation
    │
    ▼
Resize (224×224) + RGB conversion
    │
    ▼
CLAHE (contrast enhancement)
    │
    ▼
Normalization ([0,255] → [0,1])          <- src/dataset.py, src/preprocessing.py output
    │
    ▼
Data Augmentation (train only)
    │
    ▼
Backbone-correct input scaling            <- INSIDE the model itself (src/model.py), see fix below
    │
    ▼
EfficientNet-B0 (ImageNet-pretrained, frozen → fine-tuned)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout(0.4) → Dense(256) → BatchNorm → Swish
    │
    ▼
Dropout(0.3) → Dense(128) → BatchNorm → Swish
    │
    ▼
Dense(1) → Sigmoid
    │
    ▼
Prediction + Confidence
    │
    ▼
Grad-CAM Explanation → Report
```

**Why EfficientNet-B0?** Compound scaling gives strong accuracy per parameter, it converges quickly on modestly sized medical datasets, and ImageNet features (edges, textures) transfer well to radiographs. **Why Swish over ReLU?** Smoother, non-monotonic gradient flow tends to help convergence and final accuracy in deeper heads. Training runs in two stages — a frozen-backbone stage to train the new head, then a low-LR fine-tuning stage that unfreezes the last N backbone layers (BatchNorm layers stay frozen to protect pretrained running statistics).

---

## Preprocessing Fix (2026-08-08)

### Root cause

The dataset pipeline (`src/dataset.py`, `src/preprocessing.py`) correctly produces float32 images normalized to **[0,1]** — that's the right representation for CLAHE, augmentation, and display. The bug was one level down: `tf.keras.applications.EfficientNetB0` **bakes its own input rescaling into the model itself** as literal layers — verified directly against the installed Keras source rather than assumed:

```
EfficientNetB0(weights=None, include_top=False).layers[:3]
  0: input_layer     InputLayer
  1: rescaling       Rescaling(scale=0.00392156862745098, offset=0.0)   # = 1/255
  2: normalization   Normalization(mean=..., variance=...)              # ImageNet stats
```

So EfficientNetB0 expects **raw [0,255] pixel values** at its input — its own `preprocess_input` function is literally a pass-through no-op, because the real work happens inside the model graph. The original code wired the pipeline's already-`[0,1]` output directly into the backbone's `input_tensor`, so every image was divided by 255 **twice**: once by the pipeline, once again by the model's own `Rescaling` layer.

**Measured impact** (feeding two very different synthetic images through EfficientNetB0 with random weights, to isolate the scaling effect from anything learned):

| | Buggy (double-scaled) | Fixed |
|---|---|---|
| Feature map magnitude | ~5×10⁻¹⁰ | ~7×10⁻⁸ (~130× larger) |
| L2 distance between two different images' features | ~3×10⁻⁶ | ~7×10⁻⁴ (~230× more separable) |

With the bug, every image collapses to a near-identical, near-zero feature map — the classification head has almost nothing to learn from except its bias term. That's exactly the observed symptom: **everything predicted PNEUMONIA regardless of threshold**, with the ROC-AUC of 0.74 coming from a faint residual signal rather than genuine feature separation.

This also affects two of the other backbones offered via `--backbone` (used by `compare_models.py`): `MobileNetV3Large` has the same baked-in `Rescaling` (needs `[0,255]`), while `ResNet50` and `DenseNet121` have **no** internal rescaling at all — they need their own explicit `preprocess_input` (BGR + ImageNet mean-subtraction for ResNet; ImageNet mean/std normalization for DenseNet) applied on `[0,255]`-range pixels. All four were broken, not just EfficientNet-B0.

### The fix

`src/model.py` now inserts a backbone-aware preprocessing block *inside the model*, right after the `Input` layer:

- `EfficientNetB0` / `EfficientNetB3` / `MobileNetV3Large` → multiply the pipeline's `[0,1]` output by 255 (`layers.Rescaling(255.0)`) and hand it straight to the backbone, whose own internal `Rescaling`/`Normalization` layers then do the rest — exactly as they're designed to.
- `ResNet50` / `DenseNet121` → multiply by 255, then apply their real `preprocess_input` function via a small registered custom layer (`ResNetPreprocess` / `DenseNetPreprocess` in `src/model.py`).

This keeps the external pipeline contract unchanged (dataset.py, augmentations.py, inference.py, gradcam.py, error_analysis.py, the notebooks — none of them needed to change), since the model itself now correctly accepts the same `[0,1]` float input it always claimed to.

**Why not just `layers.Lambda(preprocess_input)`?** That was the first attempt, and it silently breaks: Keras 3's `model.save()` / `load_model()` round-trip raises `Could not locate function 'preprocess_input'` because `Lambda` only serializes a *reference* to the function, and Keras's own `preprocess_input` functions aren't registered as serializable. This was caught by actually running a save → load → predict round-trip in this repo, not assumed — see `tests/test_model_preprocessing.py`. The fix wraps the same calls inside a proper `@tf.keras.utils.register_keras_serializable` `Layer` subclass instead, which saves/loads correctly (also verified with an explicit round-trip test).

### Everything else that was checked (Task 13 of the audit) and found correct

- **Class mapping**: `src/dataset.py`'s `_list_files_and_labels` enumerates `CONFIG.data.class_names = ("NORMAL", "PNEUMONIA")` in order, so **label 0 = NORMAL, label 1 = PNEUMONIA** everywhere (dataset, training, evaluation, Grad-CAM, error analysis). `y_pred = (y_prob >= threshold).astype(int)` correctly maps a high probability to PNEUMONIA. No reversal found anywhere.
- **Train/val/test preprocessing consistency**: `_decode_and_preprocess` (resize + CLAHE) is identical across all splits; only `_augment` branches on `training`, and the validation/test branch uses `get_val_augmentations()` (resize-only, no randomness) — confirmed no augmentation leaks into val/test.
- **Sigmoid + BinaryCrossentropy**: correct combination, no `from_logits` mismatch (the output layer applies `sigmoid` explicitly and the loss expects probabilities).
- **Class weights**: correctly upweight the minority class (`n_total / (n_classes * n_c)`); not the root cause, though on a genuinely imbalanced dataset they could mildly compound a bias already introduced by the scaling bug.
- **Grayscale/channel handling, image loading, output shapes**: all checked, no issues found.
- **Data leakage** (duplicate files across splits): can't be verified from code alone since it depends on your actual downloaded files — `check_data_integrity.py` (new) hashes every file and flags any byte-identical duplicate appearing in more than one split. Run it once against your real dataset.

No other bugs were found or fixed — per the audit brief, only changes with actual evidence behind them were made.

---

## Correct Workflow (train → tune threshold → evaluate)

```
TRAIN
  ↓
models/best_model.keras   (selected by VALIDATION AUC — never test performance)
  ↓
VALIDATION SET → tune_threshold.py → recommended threshold
  ↓
freeze threshold
  ↓
TEST SET → evaluate.py --threshold <recommended> → ONE final, unbiased number
```

```bash
python train.py
python tune_threshold.py
# tune_threshold.py prints the exact next command, e.g.:
python evaluate.py --model-path models/best_model.keras --threshold 0.68
```

`evaluate.py` never searches for a threshold itself — it only ever applies whatever `--threshold` you pass (default `0.5`, a neutral fallback for a quick sanity check, not a substitute for `tune_threshold.py`; leaving it at the default prints a reminder). The test set is only ever touched once, by this final command.

### `tune_threshold.py` — validation-only threshold selection

```bash
python tune_threshold.py
python tune_threshold.py --min-sensitivity 0.90   # override the target if 95% isn't reachable yet
```

Sweeps thresholds from 0.01 to 0.99 (step 0.01) on the **validation set only** and computes accuracy, precision, recall, sensitivity, specificity, F1, balanced accuracy, and confusion counts at each one. Because this is a screening tool, the primary selection rule is:

> Among thresholds with **sensitivity (recall on PNEUMONIA) ≥ 95%**, pick the one with the **highest specificity**.

This is a project modeling choice — missing a real pneumonia case is treated as costlier than a false alarm — not a claim that 95% is a clinically validated number. Youden's J (`sensitivity + specificity - 1`), best-F1, and best-balanced-accuracy thresholds are also computed and reported for comparison. If no threshold in the sweep reaches the sensitivity target at all, the script falls back to the Youden's J threshold and says so explicitly, rather than silently returning something misleading.

Outputs, all under `outputs/reports/`:
- `threshold_search.csv` / `.json` — the full per-threshold sweep
- `threshold_recommendation.json` — the chosen threshold + reasoning + alternatives
- `threshold_curve.png` — sensitivity/specificity/precision/F1/balanced-accuracy vs. threshold, with the recommendation marked
- `probability_distribution_val.png` — histogram of predicted probabilities by true class (validation set) — makes it visible whether the two classes actually separate anywhere, or whether probabilities are collapsed into one narrow band

### `evaluate.py` — final test-set evaluation

```bash
python evaluate.py --model-path models/best_model.keras --threshold 0.68
```

Threshold-specific outputs are saved with the threshold baked into the filename, so evaluating at multiple thresholds never overwrites a previous run:
- `outputs/reports/test_evaluation_threshold_0_68.json`
- `outputs/confusion_matrix/confusion_matrix_threshold_0_68.png`
- `outputs/error_analysis/most_confident_mistakes_threshold_0_68.png`, `least_confident_threshold_0_68.png`, `fp_fn_lists_threshold_0_68.json`

ROC curve and PR curve (`outputs/roc/roc_curve.png`, `outputs/pr_curve/pr_curve.png`) are threshold-independent (they characterize performance across *all* thresholds at once) and are saved once per run, not namespaced.

### `check_data_integrity.py` — dataset sanity checks

```bash
python check_data_integrity.py
```

Reports, from the actual files on disk (nothing assumed):
- image counts per split/class (so you can confirm your real train/val/test distribution, including the 390 PNEUMONIA / 234 NORMAL test-set figures reported against this dataset)
- corrupted/unreadable images
- byte-identical files appearing in more than one split (MD5-hash based — catches exact duplicates, e.g. the same X-ray placed in both `train/` and `test/`; does not catch near-duplicates like re-exports at a different resolution)

Saves `outputs/reports/data_integrity_report.json`.

---

## Dataset

Expected layout (a standard chest X-ray pneumonia dataset — e.g. the Kermany/Mooney "Chest X-Ray Images (Pneumonia)" dataset on Kaggle):

```
dataset/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

Download the dataset and place it in this structure before running `train.py`. The repository ships with empty placeholder directories (`.gitkeep`) so the structure is ready to go. Run `python check_data_integrity.py` once you've added your data to confirm the real class distribution and rule out corrupted files or cross-split duplicates.

*(Optional) generalization check*: evaluate the trained model against the NIH ChestX-ray14 dataset to sanity-check performance on an out-of-distribution source.

## Project Structure

```
Pneumonia-Detection/
├── dataset/                  # train/val/test splits (NORMAL/PNEUMONIA), user-provided
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_ModelTraining.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_GradCAM.ipynb
├── src/
│   ├── preprocessing.py      # resize, CLAHE, normalize, validation
│   ├── augmentations.py      # Albumentations train/val/TTA pipelines
│   ├── dataset.py            # tf.data loader (lazy, cached, prefetched, train-only oversampling)
│   ├── model.py               # EfficientNet-B0/B1/B2/B3 + SE/CBAM attention + backbone-correct preprocessing
│   ├── trainer.py            # two-stage training, callbacks, mixed precision, BCE/focal loss
│   ├── evaluator.py          # metrics + ROC/PR/confusion-matrix plots
│   ├── threshold_optimizer.py # threshold sweep, selection criteria, confusion-category breakdown, diagnostic plots
│   ├── calibration.py        # temperature scaling (validation-fitted)
│   ├── experiment_tracker.py # CSV experiment log + markdown table renderer
│   ├── gradcam.py            # Grad-CAM heatmap + overlay
│   ├── error_analysis.py     # FP/FN, confidence-ranked mistakes, top-NORMAL-FP montage
│   ├── inference.py          # single-image inference (+ TTA, threshold-aware)
│   └── utils.py              # seeding, logging, dataset/dedup/patient-leakage helpers
├── tests/                    # pytest unit tests
├── .github/workflows/ci.yml  # lint + test CI
├── app.py                    # Streamlit deployment (shows threshold used)
├── train.py                  # two-stage training (oversample/focal-loss/fine-tune-depth flags)
├── tune_threshold.py         # validation-only threshold selection + FP report + experiment tracking
├── calibrate.py               # validation-only temperature-scaling fit
├── evaluate.py                # final, unbiased test-set evaluation
├── predict.py                 # single-image inference CLI (loads frozen threshold automatically)
├── check_data_integrity.py   # distribution / corruption / duplicate-group / patient-leakage checks
├── tune.py                   # Optuna hyperparameter search
├── export_model.py           # SavedModel / ONNX / TFLite export
├── compare_models.py         # multi-backbone comparison (EfficientNetB0-B3)
├── config.py                 # single source of truth for hyperparameters/paths
└── requirements.txt
```

## Installation

```bash
git clone <your-repo-url>
cd Pneumonia-Detection
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Then download the dataset into `dataset/train|val|test/NORMAL|PNEUMONIA/` as described above.

## Training

```bash
python train.py                                   # default config (50 + 15 epochs)
python train.py --epochs 30 --fine-tune-epochs 10  # shorter run
python train.py --backbone EfficientNetB3 --attention se
python train.py --no-fine-tune                     # skip stage 2
```

Training is two-stage: frozen-backbone head training, then low-LR fine-tuning of the last backbone layers. Regularization: dropout, batch norm, early stopping (on `val_auc`), `ReduceLROnPlateau`, `ModelCheckpoint` (best `val_auc` — **never test performance**), gradient clipping, optional mixed precision, and class-weighted loss to counter class imbalance. Every run is reproducible via a fixed seed (`src/utils.set_seed`) and logs to `logs/` (CSV + TensorBoard).

## Inference (single image)

```bash
python predict.py --image path/to/xray.jpg
python predict.py --image path/to/xray.jpg --tta --save-heatmap outputs/heatmaps/example.png
```

## Deployment (Streamlit)

```bash
streamlit run app.py
```

Features: drag-and-drop upload, prediction + confidence, Grad-CAM overlay, downloadable PDF report, sidebar with model info and in-session prediction history, optional test-time augmentation toggle.

## Explainability

Every prediction is paired with a **Grad-CAM** heatmap (`src/gradcam.py`) computed against the last convolutional feature map, upsampled and alpha-blended over the original X-ray — so a viewer can see *where* the model is looking, not just *what* it predicted.

## Results

*(Placeholder — fill in after retraining with the preprocessing fix on your machine/GPU.)*

| Metric | Value | at threshold |
|---|---|---|
| Accuracy | — | — |
| Precision | — | — |
| Recall (Sensitivity) | — | — |
| Specificity | — | — |
| F1 Score | — | — |
| ROC-AUC | — | n/a (threshold-independent) |
| MCC | — | — |

Target metrics per the project spec: Accuracy/Precision/Recall/F1 > 95%, ROC-AUC > 0.98.

## Stretch Goals Included

- **Attention mechanisms** — SE and CBAM blocks (`--attention se|cbam` in `train.py`)
- **Test-Time Augmentation** — `--tta` flag in `predict.py`, toggle in the Streamlit app
- **Model comparison** — `compare_models.py` trains/evaluates EfficientNet-B0/B3, DenseNet121, ResNet50, MobileNetV3Large under identical settings (all four now correctly preprocessed — see fix above)
- **Hyperparameter optimization** — `tune.py` (Optuna: learning rate, dropout, batch size, weight decay)
- **Model export** — `export_model.py` → SavedModel, ONNX, TFLite
- **CI/CD** — `.github/workflows/ci.yml` (flake8, black, pytest)
- **Unit tests** — `tests/` (preprocessing, dataset, model, Grad-CAM, backbone-preprocessing regression tests, threshold optimizer)
- **Multi-class extension** — the architecture and loss are easiest to extend by swapping the final `Dense(1, sigmoid)` for `Dense(N, softmax)` and switching to categorical cross-entropy; left as a documented extension point rather than a shipped variant, since it needs a multi-class labeled dataset (bacterial/viral/COVID-19) that isn't part of the standard binary dataset above.

## Reproducibility

`src/utils.set_seed(seed)` seeds Python, NumPy, and TensorFlow/Keras, and requests deterministic ops. The seed is centralized in `config.py` (`TrainConfig.seed`, default 42).

## Testing

```bash
pytest tests/ -v --cov=src
```

Includes a regression test (`tests/test_model_preprocessing.py::test_double_normalization_bug_would_have_been_caught`) that directly reproduces the reported bug and asserts the fix resolves it, plus a save/load round-trip check for every backbone's preprocessing path.

## Coding Standards

Object-oriented where it aids reuse (`Trainer`, `Evaluator`, `GradCAM`, `ErrorAnalyzer`, `InferenceEngine`, `PneumoniaDataset`), type hints throughout, docstrings on every public function/class, PEP8 (enforced via `black` + `flake8` in CI), and a clean separation between training (`train.py`/`src/trainer.py`) and inference (`predict.py`/`src/inference.py`).

## Important Notes on This Repository's State

This repository was scaffolded and fixed end-to-end (every module, script, notebook, and test listed above is complete, working code) but **has not been trained on the real dataset** here — that requires the actual dataset and, in practice, a GPU. Every module — including the preprocessing fix, the Albumentations-API fix (below), and the new threshold-tuning scripts — has been unit- and integration-tested against synthetic data in this environment: a real tiny model was trained end-to-end through the fixed pipeline, and `tune_threshold.py`/`evaluate.py`/`check_data_integrity.py` were all run against it and produced correct, non-overwriting, correctly-separated results. To get real results:

1. Download the chest X-ray pneumonia dataset into `dataset/train|val|test/NORMAL|PNEUMONIA/`.
2. `pip install -r requirements.txt`
3. `python check_data_integrity.py` (confirm your actual class distribution and rule out corrupted/duplicate files)
4. `python train.py`
5. `python tune_threshold.py`
6. `python evaluate.py --model-path models/best_model.keras --threshold <recommended>`
7. `streamlit run app.py`

### A second, unrelated bug also found and fixed: Albumentations API mismatch

While integration-testing the fix above, training crashed immediately with a `pydantic` validation error inside `RandomResizedCrop`. The currently-installed Albumentations version (2.x, what `pip install albumentations` gets today) changed two transform signatures used in `src/augmentations.py`:
- `RandomResizedCrop` now takes `size=(h, w)` instead of separate `height=`/`width=` kwargs.
- `GaussNoise` now takes `std_range=(lo, hi)` (a fraction of the image's max value) instead of the old `var_limit=(lo, hi)` (a 0–255-scale variance), which no longer exists as a parameter.

Both are fixed in `src/augmentations.py` (verified against the actually-installed version's real constructor signatures via `inspect.signature`, not assumed). `ShiftScaleRotate` was also consolidated into the existing `Affine` call, since current Albumentations treats it as a deprecated special case of `Affine` and warns on every use otherwise. This would have blocked training entirely for anyone installing the project fresh, independent of the preprocessing bug.

### Known limitations

- No real accuracy/precision/recall numbers exist yet — the Results table above is a placeholder until you train on the actual dataset.
- `check_data_integrity.py`'s duplicate detection only catches byte-identical files (MD5 hash), not near-duplicates (e.g. the same image re-compressed or resized). A perceptual-hash pass would be a reasonable follow-up if you suspect that kind of leakage.
- The `sensitivity >= 95%` threshold-selection target is a project modeling choice, not a clinically validated requirement — treat it as a starting point to discuss, not a certified figure.
- Class weights were checked and found correctly implemented (they upweight the minority class), but their interaction with the now-fixed probability calibration hasn't been re-validated on the real dataset — worth a look if the recommended threshold ends up surprising after retraining.

## Future Work

- Multi-class classification (bacterial vs. viral vs. COVID-19 vs. normal)
- External validation on NIH ChestX-ray14 for generalization
- Model distillation for edge/mobile deployment (TFLite export is already wired up)
- Uncertainty quantification (e.g., MC Dropout) alongside the confidence score
- Perceptual-hash-based near-duplicate detection in `check_data_integrity.py`

## License

MIT — see `LICENSE`.
