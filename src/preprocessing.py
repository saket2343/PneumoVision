"""
preprocessing.py
-----------------
Phase 2 of the pipeline: image validation, resizing, colour conversion,
CLAHE contrast enhancement, and normalization.

Every function operates on a single image (as a NumPy array, uint8,
HxWxC or HxW) so it can be reused identically inside the tf.data
pipeline (dataset.py), the Streamlit app (app.py), and inference
(inference.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


class CorruptedImageError(ValueError):
    """Raised when an image file cannot be read or decoded."""


def validate_image(path: Union[str, Path]) -> bool:
    """Return True if the file at `path` is a readable, non-corrupted image."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.size == 0:
        return False
    return True


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as RGB uint8. Raises CorruptedImageError on failure."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise CorruptedImageError(f"Could not read image: {path}")
    return to_rgb(img)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale, BGR, or BGRA image to 3-channel RGB uint8."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def resize_image(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize with area interpolation (best for downscaling medical images)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE on the luminance channel to boost local contrast.

    Chest X-rays often have low global contrast; CLAHE brings out fine
    structures (e.g. infiltrates) without over-amplifying noise the way
    global histogram equalization does.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def denoise(img: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    """Optional denoising: 'median' or 'gaussian'."""
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    raise ValueError(f"Unknown denoise method: {method}")


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] image to float32 [0, 1]."""
    return img.astype(np.float32) / 255.0


def preprocess_image(
    img: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Full deterministic preprocessing chain (no augmentation).

    RGB uint8 in -> RGB float32 in [0, 1] out, at target `size`.
    This is the function used identically at train, eval, and inference time.
    """
    img = to_rgb(img)
    img = resize_image(img, size)
    if use_clahe:
        img = apply_clahe(img, clip_limit, tile_grid_size)
    img = normalize(img)
    return img


def preprocess_path(
    path: Union[str, Path],
    size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Convenience: load from disk and run the full preprocessing chain."""
    img = load_image(path)
    return preprocess_image(img, size=size, use_clahe=use_clahe)
