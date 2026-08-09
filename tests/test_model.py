import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.gradcam import GradCAM
from src.model import build_model, get_last_conv_layer_name


@pytest.fixture(scope="module")
def small_model():
    # Use a tiny input for fast CI; EfficientNetB0 requires >=32x32.
    return build_model(input_shape=(96, 96, 3), freeze_backbone=True)


def test_model_output_shape(small_model):
    x = np.random.rand(2, 96, 96, 3).astype(np.float32)
    y = small_model(x, training=False)
    assert y.shape == (2, 1)


def test_model_output_range(small_model):
    x = np.random.rand(1, 96, 96, 3).astype(np.float32)
    y = small_model(x, training=False).numpy()
    assert (y >= 0).all() and (y <= 1).all()


def test_backbone_frozen_by_default(small_model):
    trainable_backbone_layers = [l for l in small_model.layers if "block" in l.name and l.trainable]
    assert len(trainable_backbone_layers) == 0


def test_get_last_conv_layer_name(small_model):
    name = get_last_conv_layer_name(small_model)
    layer = small_model.get_layer(name)
    assert isinstance(layer, tf.keras.layers.Conv2D)


def test_gradcam_output_shapes(small_model):
    gc = GradCAM(small_model)
    img = np.random.rand(96, 96, 3).astype(np.float32)
    heatmap, overlay = gc.explain(img)
    assert heatmap.ndim == 2
    assert overlay.shape == (96, 96, 3)
    assert overlay.dtype == np.uint8
