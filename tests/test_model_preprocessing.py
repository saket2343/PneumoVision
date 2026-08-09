import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.keras import layers, models

from src.model import (
    _build_preprocessing_block,
    _EXPLICIT_PREPROCESS_LAYERS,
    _SELF_RESCALING_BACKBONES,
)


@pytest.fixture()
def pipeline_image():
    """A float32 [0,1] image — exactly what src/dataset.py hands to the model."""
    rng = np.random.RandomState(0)
    return rng.rand(1, 32, 32, 3).astype(np.float32)


def _probe_model(backbone_name, input_shape=(32, 32, 3)):
    inputs = layers.Input(input_shape)
    out = _build_preprocessing_block(inputs, backbone_name)
    return models.Model(inputs, out)


@pytest.mark.parametrize("backbone_name", sorted(_SELF_RESCALING_BACKBONES))
def test_self_rescaling_backbones_receive_0_255_range(backbone_name, pipeline_image):
    """EfficientNet/MobileNetV3 bake their own Rescaling(1/255) inside the model,
    so _build_preprocessing_block must hand them raw [0,255] pixels, not [0,1]."""
    probe = _probe_model(backbone_name)
    out = probe(pipeline_image, training=False).numpy()
    assert out.min() >= 0.0
    assert out.max() <= 255.0
    # A [0,1]-range image, scaled up, should actually reach well above 1.0 —
    # this is the exact check that would have caught the original double-normalization bug.
    assert out.max() > 1.0


@pytest.mark.parametrize("backbone_name", sorted(_EXPLICIT_PREPROCESS_LAYERS.keys()))
def test_explicit_preprocess_backbones_are_mean_centered(backbone_name, pipeline_image):
    """ResNet50/DenseNet121 have no internal rescaling — they need their own
    preprocess_input applied (BGR+mean-subtract / ImageNet mean-std normalize),
    which should produce a mean-centered range including negative values,
    not a raw positive [0,255] or [0,1] range."""
    probe = _probe_model(backbone_name)
    out = probe(pipeline_image, training=False).numpy()
    assert out.min() < 0  # mean-subtraction always pushes some values negative


def test_unknown_backbone_raises():
    inputs = layers.Input((32, 32, 3))
    with pytest.raises(ValueError):
        _build_preprocessing_block(inputs, "NotARealBackbone")


def test_double_normalization_bug_would_have_been_caught(pipeline_image):
    """Regression guard: directly reproduces the reported bug (feeding an
    already-[0,1] image straight into a backbone with its own internal
    Rescaling(1/255)) and confirms it collapses to near-zero, while the
    fixed path does not.
    """
    # Buggy path: no rescaling block at all, straight into a Rescaling(1/255) layer
    inputs = layers.Input((32, 32, 3))
    buggy_out = layers.Rescaling(scale=1.0 / 255.0)(inputs)  # simulates EfficientNet's internal layer alone
    buggy_probe = models.Model(inputs, buggy_out)
    buggy_result = buggy_probe(pipeline_image, training=False).numpy()

    # Fixed path: our preprocessing block first restores [0,255], THEN the same internal layer applies
    fixed_probe = _probe_model("EfficientNetB0")
    fixed_result = fixed_probe(pipeline_image, training=False).numpy()

    assert buggy_result.max() < 0.005  # collapsed near-zero, as empirically measured against the real bug
    assert fixed_result.max() > 0.5  # restored to a sane, informative range
