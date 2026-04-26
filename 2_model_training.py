"""
=============================================================
STEP 2: MODEL BUILDING & TRAINING
Early Detection of MCI using Stacked CNN Feature Fusion
=============================================================
Architecture:
  - ResNet50    pretrained feature extractor (global structural patterns)
  - DenseNet121 pretrained feature extractor (fine-grained textures)
  - Concatenate features -> FC layers -> Binary Output (CN vs MCI)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
IMG_SIZE      = (224, 224, 3)
BATCH_SIZE    = 16
EPOCHS        = 50
LEARNING_RATE = 1e-4
NUM_CLASSES   = 2
DROPOUT_RATE  = 0.5
RANDOM_SEED   = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------
# 1. DATA AUGMENTATION  (tf.data pipeline)
# ---------------------------------------------
def augment_image(image, label):
    """Per-image augmentation ops that run inside the tf.data pipeline."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # Small random crop then resize back to 224x224 (simulates slight zoom/shift)
    crop = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * 0.95, tf.int32)
    image = tf.image.random_crop(image, size=[crop, crop, 3])
    image = tf.image.resize(image, [224, 224])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dual_input(image, label):
    """
    The model has TWO named inputs: 'resnet_input' and 'densenet_input'.
    Both receive the same MRI image. We return a dict so Keras matches by name.
    """
    return {"resnet_input": image, "densenet_input": image}, label


def create_datasets(X_train, y_train, X_val, y_val):
    """Build tf.data.Dataset objects for training and validation."""
    y_train_cat = to_categorical(y_train, NUM_CLASSES).astype(np.float32)
    y_val_cat   = to_categorical(y_val,   NUM_CLASSES).astype(np.float32)

    X_train_f = X_train.astype(np.float32)
    X_val_f   = X_val.astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_f, y_train_cat))
        .shuffle(buffer_size=len(X_train_f), seed=RANDOM_SEED)
        .map(augment_image,   num_parallel_calls=tf.data.AUTOTUNE)
        .map(make_dual_input, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_f, y_val_cat))
        .map(make_dual_input, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# ---------------------------------------------
# 2. MODEL ARCHITECTURE
# ---------------------------------------------
def build_stacked_fusion_model(input_shape=IMG_SIZE):
    """
    Dual-backbone feature fusion model.

    resnet_input  --> ResNet50   --> GAP --> 2048-d -+
                                                      +--> Concat(3072) --> FC --> CN/MCI
    densenet_input-> DenseNet121 --> GAP --> 1024-d -+
    """
    print("\n[MODEL] Building Stacked Feature Fusion Model...")

    # Each backbone gets its own Input to avoid layer-name conflicts
    resnet_input   = Input(shape=input_shape, name="resnet_input")
    densenet_input = Input(shape=input_shape, name="densenet_input")

    # Branch 1: ResNet50
    resnet_base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    resnet_base.trainable = False   # frozen in Phase 1
    resnet_features = layers.GlobalAveragePooling2D(name="resnet_gap")(
        resnet_base(resnet_input)
    )   # shape: (batch, 2048)

    # Branch 2: DenseNet121
    densenet_base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    densenet_base.trainable = False
    densenet_features = layers.GlobalAveragePooling2D(name="densenet_gap")(
        densenet_base(densenet_input)
    )   # shape: (batch, 1024)

    # Feature fusion by concatenation
    fused = layers.Concatenate(name="feature_fusion")([resnet_features, densenet_features])
    # shape: (batch, 3072)

    # Fully connected classifier
    x = layers.Dense(512, name="fc1")(fused)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)

    x = layers.Dense(128, name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop2")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = Model(
        inputs=[resnet_input, densenet_input],
        outputs=outputs,
        name="MCI_Stacked_Fusion"
    )

    print(f"  Total parameters     : {model.count_params():,}")
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Trainable parameters : {trainable:,}")

    return model, resnet_base, densenet_base


# ---------------------------------------------
# 3. COMPILE & CALLBACKS
# ---------------------------------------------
def compile_model(model, lr=LEARNING_RATE):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model


def get_callbacks(ckpt_name="best_model.keras"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=f"saved_models/{ckpt_name}",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger("logs/training_log.csv")
    ]


# ---------------------------------------------
# 4. PLOT TRAINING HISTORY
# ---------------------------------------------
def plot_history(history, tag="Phase1"):
    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss    = history.history["loss"]
    val_los = history.history["val_loss"]
    ep      = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History - {tag}", fontsize=13, fontweight="bold")

    axes[0].plot(ep, acc,     label="Train Acc", color="steelblue", lw=2)
    axes[0].plot(ep, val_acc, label="Val Acc",   color="tomato",    lw=2)
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, loss,    label="Train Loss", color="steelblue", lw=2)
    axes[1].plot(ep, val_los, label="Val Loss",   color="tomato",    lw=2)
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f"logs/training_history_{tag}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Plot saved: {fname}")


# ---------------------------------------------
# 5. FINE-TUNING (Phase 2)
# ---------------------------------------------
def fine_tune_model(model, resnet_base, densenet_base, train_ds, val_ds):
    """Unfreeze the top 30 layers of each backbone and retrain at a lower LR."""
    print("\n[FINE-TUNE] Unfreezing top 30 layers of each backbone...")
    for layer in resnet_base.layers[-30:]:
        layer.trainable = True
    for layer in densenet_base.layers[-30:]:
        layer.trainable = True

    compile_model(model, lr=LEARNING_RATE / 10)

    callbacks = get_callbacks("best_model_finetuned.keras")

    print("[FINE-TUNE] Phase 2 training...")
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )
    return history_ft


# ---------------------------------------------
# 6. MAIN
# ---------------------------------------------
def train():
    print("=" * 55)
    print("  MCI DETECTION - Model Training")
    print("=" * 55)

    # Load preprocessed arrays
    print("\n[1/4] Loading preprocessed data...")
    try:
        X_train = np.load("processed_data/X_train.npy")
        X_val   = np.load("processed_data/X_val.npy")
        y_train = np.load("processed_data/y_train.npy")
        y_val   = np.load("processed_data/y_val.npy")
    except FileNotFoundError:
        raise FileNotFoundError(
            "[ERROR] Run '1_data_preprocessing.py' first!"
        )

    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"  CN train: {sum(y_train==0)} | MCI train: {sum(y_train==1)}")

    # Build tf.data pipelines
    print("\n[2/4] Creating tf.data pipelines with augmentation...")
    train_ds, val_ds = create_datasets(X_train, y_train, X_val, y_val)

    # Build & compile
    print("\n[3/4] Building model...")
    model, resnet_base, densenet_base = build_stacked_fusion_model()
    compile_model(model)
    model.summary()

    # Phase 1: frozen backbones
    print(f"\n[4/4] Phase 1 Training (frozen backbones, {EPOCHS} max epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks("best_model.keras")
    )
    plot_history(history, "Phase1_FrozenBackbone")

    # Phase 2: fine-tune
    print("\nStarting Phase 2: Fine-tuning top backbone layers...")
    history_ft = fine_tune_model(model, resnet_base, densenet_base, train_ds, val_ds)
    plot_history(history_ft, "Phase2_FineTuning")

    print("\nTraining complete! Run '3_evaluation.py' next.")
    return model


if __name__ == "__main__":
    train()
