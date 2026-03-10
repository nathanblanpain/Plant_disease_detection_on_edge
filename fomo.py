"""
FOMO (Faster Objects, More Objects) + MobileNetV2
==================================================
FOMO replaces the traditional bounding-box head with a lightweight
fully-convolutional head that produces a centroid heatmap.
Each spatial cell in the output grid predicts the *probability* that
an object centre falls inside that cell – no anchor boxes, no NMS.

Architecture
------------
Input  →  MobileNetV2 backbone (feature extractor)
       →  1×1 Conv head (per-class sigmoid activation)
       →  Output heatmap  [B, H/S, W/S, num_classes]

Where S is the effective stride of the chosen backbone layer.

Usage
-----
    python fomo_mobilenetv2.py            # builds + summarises the model
    python fomo_mobilenetv2.py --train    # demo training on synthetic data
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ─────────────────────────── configuration ──────────────────────────────────

INPUT_WIDTH   = 96          # model input width  (multiples of 32 work best)
INPUT_HEIGHT  = 96          # model input height
NUM_CLASSES   = 3           # number of object classes (excluding background)
ALPHA         = 0.35        # MobileNetV2 width multiplier (0.35 / 0.5 / 1.0)
BACKBONE_LAYER = "block_6_expand_relu"   # output of this layer is used as features

# ─────────────────────────── build model ────────────────────────────────────

def build_fomo_mobilenetv2(
    input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH, 3),
    num_classes: int   = NUM_CLASSES,
    alpha: float       = ALPHA,
    backbone_layer: str = BACKBONE_LAYER,
    dropout_rate: float = 0.1,
) -> Model:
    """
    Returns a compiled FOMO model.

    Parameters
    ----------
    input_shape    : (H, W, C)  – must be (H, W, 3)
    num_classes    : number of foreground classes
    alpha          : MobileNetV2 width multiplier
    backbone_layer : name of the MobileNetV2 layer whose output feeds the head
    dropout_rate   : spatial dropout applied before the classification head

    Returns
    -------
    keras.Model (already compiled)
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # ── Backbone ────────────────────────────────────────────────────────────
    backbone = keras.applications.MobileNetV2(
        input_shape = input_shape,
        alpha        = alpha,
        include_top  = False,
        weights      = "imagenet",
    )
    # Freeze lower layers; fine-tune the top few
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    # Extract intermediate feature map
    feature_extractor = Model(
        inputs  = backbone.input,
        outputs = backbone.get_layer(backbone_layer).output,
        name    = "mobilenetv2_backbone",
    )
    features = feature_extractor(inputs, training=False)   # (B, H', W', C')

    # ── FOMO Head ────────────────────────────────────────────────────────────
    # Lightweight: just 1×1 convolutions – keeps latency tiny on MCUs
    x = layers.SpatialDropout2D(dropout_rate, name="spatial_dropout")(features)

    x = layers.Conv2D(
        filters     = 32,
        kernel_size = 1,
        padding     = "same",
        use_bias    = False,
        name        = "head_conv1",
    )(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.ReLU(6.0, name="head_relu1")(x)

    # Per-class centroid probability map  (sigmoid, not softmax → multi-label OK)
    output_map = layers.Conv2D(
        filters     = num_classes,
        kernel_size = 1,
        padding     = "same",
        activation  = "sigmoid",
        name        = "centroid_heatmap",
    )(x)                                                   # (B, H', W', num_classes)

    model = Model(inputs=inputs, outputs=output_map, name="FOMO_MobileNetV2")

    # ── Loss & metrics ───────────────────────────────────────────────────────
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = fomo_loss,
        metrics   = [
            keras.metrics.BinaryAccuracy(name="binary_acc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ─────────────────────────── FOMO loss ──────────────────────────────────────

def fomo_loss(y_true, y_pred, alpha: float = 2.0, gamma: float = 0.25):
    """
    Weighted binary focal loss tailored for FOMO's sparse heatmaps.

    Positive cells (object centres) are rare → up-weight them.
    alpha  : weight for positive (foreground) cells
    gamma  : focal modulation (set 0.0 for plain BCE)
    """
    # Focal modulation
    bce     = keras.backend.binary_crossentropy(y_true, y_pred)
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    focal   = tf.pow(1.0 - p_t, gamma)

    # Class balance
    weight  = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

    loss    = weight * focal * bce
    return tf.reduce_mean(loss)


# ─────────────────────────── centroid decoder ───────────────────────────────

def decode_centroids(
    heatmap: np.ndarray,
    threshold:   float = 0.5,
    input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH),
) -> list[dict]:
    """
    Convert a FOMO output heatmap to a list of detected centroids.

    Parameters
    ----------
    heatmap    : numpy array of shape (H', W', num_classes)
    threshold  : minimum confidence to report a detection
    input_shape: (H, W) of the *model input* (used to scale centroids)

    Returns
    -------
    List of dicts: {class_id, confidence, x_center, y_center}
    where x_center / y_center are in *pixel coordinates* of the input image.
    """
    H_map, W_map, num_classes = heatmap.shape
    H_in,  W_in               = input_shape

    stride_y = H_in / H_map
    stride_x = W_in / W_map

    detections = []
    for cls in range(num_classes):
        rows, cols = np.where(heatmap[:, :, cls] >= threshold)
        for r, c in zip(rows, cols):
            conf = float(heatmap[r, c, cls])
            detections.append({
                "class_id"  : int(cls),
                "confidence": round(conf, 4),
                "x_center"  : round(float(c * stride_x + stride_x / 2), 2),
                "y_center"  : round(float(r * stride_y + stride_y / 2), 2),
            })
    return detections


# ─────────────────────────── synthetic data ─────────────────────────────────

def make_synthetic_batch(
    batch_size:  int   = 8,
    input_shape: tuple = (INPUT_HEIGHT, INPUT_WIDTH, 3),
    num_classes: int   = NUM_CLASSES,
    output_size: tuple = (12, 12),          # matches backbone_layer output
    max_objects: int   = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a batch of random (image, heatmap) pairs for smoke-testing.
    """
    X = np.random.rand(batch_size, *input_shape).astype(np.float32)
    Y = np.zeros((batch_size, *output_size, num_classes), dtype=np.float32)

    for b in range(batch_size):
        n = np.random.randint(1, max_objects + 1)
        for _ in range(n):
            r   = np.random.randint(0, output_size[0])
            c   = np.random.randint(0, output_size[1])
            cls = np.random.randint(0, num_classes)
            Y[b, r, c, cls] = 1.0
    return X, Y


# ─────────────────────────── main ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOMO + MobileNetV2")
    parser.add_argument("--train",  action="store_true",
                        help="Run a quick training demo on synthetic data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch",  type=int, default=8)
    args = parser.parse_args()

    print("\n=== Building FOMO + MobileNetV2 ===")
    model = build_fomo_mobilenetv2()
    model.summary(line_length=90)

    # ── infer output grid size ───────────────────────────────────────────────
    dummy_in      = np.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
    dummy_out     = model.predict(dummy_in, verbose=0)
    _, H_out, W_out, _ = dummy_out.shape
    print(f"\nOutput heatmap grid : {H_out} × {W_out}  "
          f"(effective stride {INPUT_HEIGHT // H_out})")

    # ── demo decode ──────────────────────────────────────────────────────────
    synthetic_heatmap = np.random.rand(H_out, W_out, NUM_CLASSES).astype(np.float32)
    detections = decode_centroids(synthetic_heatmap, threshold=0.85)
    print(f"\nSample detections (threshold=0.85): {len(detections)} found")
    for d in detections[:5]:
        print(f"  class {d['class_id']}  conf={d['confidence']:.3f}"
              f"  centre=({d['x_center']}, {d['y_center']})")

    # ── optional training demo ───────────────────────────────────────────────
    if args.train:
        print("\n=== Training on synthetic data ===")
        X_train, Y_train = make_synthetic_batch(
            batch_size  = args.batch * 10,
            output_size = (H_out, W_out),
        )
        X_val, Y_val = make_synthetic_batch(
            batch_size  = args.batch * 2,
            output_size = (H_out, W_out),
        )

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ]

        history = model.fit(
            X_train, Y_train,
            validation_data = (X_val, Y_val),
            epochs          = args.epochs,
            batch_size      = args.batch,
            callbacks       = callbacks,
        )

        print("\nTraining complete.")
        print("Final val_loss :", round(history.history["val_loss"][-1],  4))
        print("Final val_acc  :", round(history.history["val_binary_acc"][-1], 4))

        model.save("fomo_mobilenetv2.keras")
        print("\nModel saved → fomo_mobilenetv2.keras")