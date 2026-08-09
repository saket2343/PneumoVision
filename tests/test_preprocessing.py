import numpy as np
import pytest

from src.preprocessing import apply_clahe, normalize, preprocess_image, resize_image, to_rgb


def _dummy_gray_image(h=100, w=120):
    return (np.random.rand(h, w) * 255).astype(np.uint8)


def _dummy_rgb_image(h=100, w=120):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_to_rgb_from_grayscale():
    img = _dummy_gray_image()
    rgb = to_rgb(img)
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3


def test_resize_image_shape():
    img = _dummy_rgb_image()
    resized = resize_image(img, (224, 224))
    assert resized.shape[:2] == (224, 224)


def test_apply_clahe_preserves_shape_and_dtype():
    img = _dummy_rgb_image()
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_normalize_range():
    img = _dummy_rgb_image()
    out = normalize(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_image_full_chain():
    img = _dummy_rgb_image(300, 400)
    out = preprocess_image(img, size=(224, 224), use_clahe=True)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0
