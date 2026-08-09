"""
inference.py
-------------
Single-image inference pipeline:
    Image -> Preprocessing -> Prediction -> Probability -> Grad-CAM -> Result

Shared by predict.py (CLI) and app.py (Streamlit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.augmentations import get_tta_augmentations
from src.gradcam import GradCAM
from src.preprocessing import load_image, preprocess_image


class InferenceEngine:
    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA"),
        image_size: Tuple[int, int] = (224, 224),
        use_clahe: bool = True,
        threshold: float = 0.5,
    ):
        """
        threshold: the decision threshold applied to the PNEUMONIA
            probability. Defaults to 0.5 ONLY as a neutral fallback — for a
            real deployment this should be the value tune_threshold.py
            recommended on the validation set (see
            `src.threshold_optimizer.load_recommended_threshold`, used by
            both predict.py and app.py to load it automatically when
            available). Never silently hard-code a different value here.
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.threshold = threshold
        self.model = tf.keras.models.load_model(self.model_path)
        self.gradcam = GradCAM(self.model)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_image(image, size=self.image_size, use_clahe=self.use_clahe)

    def predict(self, image: np.ndarray, tta: bool = False, threshold: Optional[float] = None) -> dict:
        """Run the full inference pipeline on an already-loaded RGB uint8 image.

        threshold: override `self.threshold` for this call only. Defaults to
            `self.threshold` (set at construction) when not given.
        """
        threshold = self.threshold if threshold is None else threshold
        preprocessed = self._preprocess(image)

        if tta:
            probs = []
            for aug_fn in get_tta_augmentations(self.image_size):
                aug_img = aug_fn(preprocessed)
                probs.append(self._raw_predict(aug_img))
            prob = float(np.mean(probs))
        else:
            prob = self._raw_predict(preprocessed)

        pred_idx = int(prob >= threshold)
        heatmap, overlay = self.gradcam.explain(preprocessed)

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": prob if pred_idx == 1 else 1 - prob,
            "raw_probability": prob,
            "threshold_used": threshold,
            "heatmap": heatmap,
            "overlay": overlay,
            "preprocessed_image": preprocessed,
        }

    def _raw_predict(self, preprocessed_image: np.ndarray) -> float:
        batch = np.expand_dims(preprocessed_image, axis=0)
        prob = self.model.predict(batch, verbose=0)[0, 0]
        return float(prob)

    def predict_path(self, path: Union[str, Path], tta: bool = False, threshold: Optional[float] = None) -> dict:
        image = load_image(path)
        result = self.predict(image, tta=tta, threshold=threshold)
        result["source_path"] = str(path)
        return result
