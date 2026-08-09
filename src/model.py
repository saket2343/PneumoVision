"""
model.py
--------
Phase 5 of the pipeline: EfficientNet-B0 transfer-learning architecture.

Input (224x224x3, float32 in [0,1] as produced by the dataset pipeline)
  -> backbone-specific preprocessing (see `_build_preprocessing_block`)
  -> EfficientNet-B0 (ImageNet weights, backbone frozen initially)
  -> GlobalAveragePooling2D
  -> Dropout(0.4)
  -> Dense(256) -> BatchNorm -> Swish
  -> Dropout(0.3)
  -> Dense(128) -> BatchNorm -> Swish
  -> Dense(1) -> Sigmoid

IMPORTANT — input scaling (fixed 2026-08-08, see README "Preprocessing Fix"):
`src/dataset.py` and `src/preprocessing.py` deliberately produce float32 images
normalized to [0,1] — that's the right internal representation for CLAHE,
augmentation, Grad-CAM overlay math, and display. But `tf.keras.applications`
backbones do NOT all agree on what input range they expect at the model
boundary:

  - EfficientNetB0 / EfficientNetB3 / MobileNetV3Large ship with an internal
    `Rescaling(1/255)` (+ `Normalization` for EfficientNet) baked in as their
    first layer(s). They expect raw pixel values in [0, 255]. Their
    `preprocess_input` is literally a pass-through no-op — the scaling
    happens inside the model, not before it.
  - ResNet50 / DenseNet121 have NO internal rescaling. They require calling
    their respective `keras.applications.<family>.preprocess_input` on
    [0, 255]-range pixels (BGR + ImageNet mean-subtraction for ResNet;
    ImageNet mean/std normalization for DenseNet).

Feeding our already-[0,1]-normalized pipeline output straight into
EfficientNetB0 (as the code originally did) silently re-divides it by 255
inside the model's own Rescaling layer, collapsing every image to a
near-constant, near-zero feature map (verified empirically: activation
magnitude drops ~2 orders of magnitude and cross-image separability all but
disappears). The classification head then has almost nothing to learn from
except its bias term — which is exactly the "everything predicted as one
class regardless of threshold" failure mode.

`_build_preprocessing_block` below fixes this for every supported backbone:
it always converts the pipeline's [0,1] float back to [0,255] first, then
applies the correct backbone-specific `preprocess_input` (a no-op for the
EfficientNet/MobileNetV3 family, the real mean-subtraction for ResNet/DenseNet).
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


# Backbones whose Keras implementation bakes rescaling into the model itself
# (their own `preprocess_input` is a no-op). These need [0,255] pixels and
# nothing else.
_SELF_RESCALING_BACKBONES = {"EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "MobileNetV3Large"}


@tf.keras.utils.register_keras_serializable(package="pneumonia_detection", name="ResNetPreprocess")
class ResNetPreprocess(layers.Layer):
    """Wraps `tf.keras.applications.resnet.preprocess_input` (RGB->BGR +
    ImageNet mean-subtraction, 'caffe' mode) in a real, registered Layer.

    NOTE: a bare `layers.Lambda(tf.keras.applications.resnet.preprocess_input)`
    looks equivalent but is NOT safely serializable in Keras 3 — `model.save()`
    followed by `load_model()` raises `Could not locate function
    'preprocess_input'` because Lambda only serializes a reference to the
    function, not its code, and Keras applications functions aren't
    registered as serializable. Wrapping the call inside a proper
    `@register_keras_serializable` Layer subclass (this class) avoids that
    entirely — verified empirically via an explicit save/load round-trip test.
    """

    def call(self, inputs):
        return tf.keras.applications.resnet.preprocess_input(inputs)


@tf.keras.utils.register_keras_serializable(package="pneumonia_detection", name="DenseNetPreprocess")
class DenseNetPreprocess(layers.Layer):
    """Wraps `tf.keras.applications.densenet.preprocess_input` (scale to
    [0,1] + ImageNet mean/std normalization, 'torch' mode) — see
    `ResNetPreprocess` docstring for why this can't be a bare `Lambda`.
    """

    def call(self, inputs):
        return tf.keras.applications.densenet.preprocess_input(inputs)


# Backbones that need their family-specific `preprocess_input` applied
# explicitly on [0,255]-range pixels, via a registered Layer (see above).
_EXPLICIT_PREPROCESS_LAYERS = {
    "ResNet50": ResNetPreprocess,
    "DenseNet121": DenseNetPreprocess,
}


def _build_preprocessing_block(x: tf.Tensor, backbone_name: str) -> tf.Tensor:
    """Convert the pipeline's [0,1] float input into whatever `backbone_name`
    actually expects, verified against each backbone's real Keras layer graph.
    """
    # Step 1: undo the pipeline's own [0,1] normalization — every supported
    # backbone's preprocessing (built-in or explicit) starts from [0,255].
    x = layers.Rescaling(scale=255.0, offset=0.0, name="to_pixel_range")(x)

    if backbone_name in _SELF_RESCALING_BACKBONES:
        # Rescaling/Normalization already exists inside the backbone graph —
        # nothing further to do here.
        return x

    if backbone_name in _EXPLICIT_PREPROCESS_LAYERS:
        preprocess_layer = _EXPLICIT_PREPROCESS_LAYERS[backbone_name](name="backbone_preprocess")
        return preprocess_layer(x)

    raise ValueError(
        f"No known input-preprocessing rule for backbone '{backbone_name}'. "
        "Add it to _SELF_RESCALING_BACKBONES or _EXPLICIT_PREPROCESS_LAYERS in src/model.py "
        "after checking that backbone's actual Keras layer graph."
    )


def se_block(x: tf.Tensor, reduction: int = 16, name: str = "se") -> tf.Tensor:
    """Squeeze-and-Excitation block (stretch goal: attention mechanisms).

    Recalibrates channel-wise feature responses — useful for highlighting
    channels that respond to infiltrate/consolidation patterns.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(x)
    se = layers.Dense(max(channels // reduction, 8), activation="swish", name=f"{name}_reduce")(se)
    se = layers.Dense(channels, activation="sigmoid", name=f"{name}_excite")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def cbam_block(x: tf.Tensor, reduction: int = 16, kernel_size: int = 7, name: str = "cbam") -> tf.Tensor:
    """Convolutional Block Attention Module (stretch goal): channel + spatial attention."""
    channels = x.shape[-1]

    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_ch_avg")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_ch_max")(x)
    shared_dense_1 = layers.Dense(max(channels // reduction, 8), activation="relu", name=f"{name}_ch_dense1")
    shared_dense_2 = layers.Dense(channels, name=f"{name}_ch_dense2")
    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_attn = layers.Activation("sigmoid", name=f"{name}_ch_sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attn = layers.Reshape((1, 1, channels))(channel_attn)
    x = layers.Multiply(name=f"{name}_ch_scale")([x, channel_attn])

    # Spatial attention
    avg_pool_s = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool_s = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_s, max_pool_s])
    spatial_attn = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid", name=f"{name}_sp_conv")(concat)
    x = layers.Multiply(name=f"{name}_sp_scale")([x, spatial_attn])

    return x


def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_1: float = 0.4,
    dropout_2: float = 0.3,
    dense_1: int = 256,
    dense_2: int = 128,
    freeze_backbone: bool = True,
    attention: Optional[str] = None,  # None | "se" | "cbam"
    backbone_name: str = "EfficientNetB0",
) -> tf.keras.Model:
    """Build the EfficientNet-based binary classifier described in the spec.

    `backbone_name` also supports "EfficientNetB3", "DenseNet121",
    "ResNet50", "MobileNetV3Large" for the model-comparison stretch goal.

    The model's `Input` layer expects float32 images in [0,1] — i.e. exactly
    what `src/dataset.py` / `src/preprocessing.py` already produce. The
    backbone-correct pixel-range/mean-subtraction conversion happens inside
    the model itself (see `_build_preprocessing_block`), so callers never
    need to think about it, and it's saved/loaded as part of the model.
    """
    backbones = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "EfficientNetB1": tf.keras.applications.EfficientNetB1,
        "EfficientNetB2": tf.keras.applications.EfficientNetB2,
        "EfficientNetB3": tf.keras.applications.EfficientNetB3,
        "DenseNet121": tf.keras.applications.DenseNet121,
        "ResNet50": tf.keras.applications.ResNet50,
        "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
    }
    if backbone_name not in backbones:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(backbones)}")

    inputs = layers.Input(shape=input_shape, name="input_image")
    preprocessed = _build_preprocessing_block(inputs, backbone_name)

    backbone_fn = backbones[backbone_name]
    backbone = backbone_fn(include_top=False, weights="imagenet", input_tensor=preprocessed)
    backbone.trainable = not freeze_backbone

    x = backbone.output

    if attention == "se":
        x = se_block(x, name="se_block")
    elif attention == "cbam":
        x = cbam_block(x, name="cbam_block")

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dropout(dropout_1, name="dropout_1")(x)
    x = layers.Dense(dense_1, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("swish", name="swish_1")(x)

    x = layers.Dropout(dropout_2, name="dropout_2")(x)
    x = layers.Dense(dense_2, name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("swish", name="swish_2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name=f"pneumonia_{backbone_name.lower()}")
    return model


def unfreeze_for_fine_tuning(model: tf.keras.Model, fine_tune_at_layer: int = -40) -> tf.keras.Model:
    """Unfreeze the last N layers of the backbone for a low-LR fine-tuning pass.

    BatchNorm layers are kept frozen (standard practice) to avoid destabilizing
    running statistics learned from ImageNet on a comparatively small medical dataset.
    """
    target_layers = model.layers[fine_tune_at_layer:] if abs(fine_tune_at_layer) <= len(model.layers) else model.layers

    for layer in target_layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Find the name of the last Conv2D layer — used as the Grad-CAM target layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")
