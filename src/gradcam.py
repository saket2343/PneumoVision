"""
gradcam.py
----------
Explainable AI module: Grad-CAM heatmap generation and overlay, so every
prediction can show *which region* of the X-ray drove the model's decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
        )

    def _find_target_layer(self) -> str:
        for layer in reversed(self.model.layers):
            try:
                shape = layer.output.shape
            except AttributeError:
                continue
            if shape is not None and len(shape) == 4:  # conv-like feature map
                return layer.name
        raise ValueError("Could not find a 4D conv layer for Grad-CAM.")

    def compute_heatmap(self, image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """image: float32 [1, H, W, 3] preprocessed batch of size 1."""
        image_tensor = tf.convert_to_tensor(image)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor, training=False)
            loss = predictions[:, 0]  # sigmoid output for the positive class

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    @staticmethod
    def overlay_heatmap(
        original_rgb_float: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Resize + colorize the heatmap and blend it onto the original image.

        `original_rgb_float` is float32 [0,1], HxWx3.
        Returns uint8 RGB overlay image.
        """
        h, w = original_rgb_float.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        original_uint8 = np.uint8(255 * original_rgb_float)
        overlay = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    def explain(
        self,
        preprocessed_image: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full pipeline: preprocessed_image (H,W,3 float32 [0,1]) -> (heatmap, overlay)."""
        batch = np.expand_dims(preprocessed_image, axis=0)
        heatmap = self.compute_heatmap(batch)
        overlay = self.overlay_heatmap(preprocessed_image, heatmap)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return heatmap, overlay
