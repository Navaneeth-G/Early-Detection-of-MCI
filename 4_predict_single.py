"""
=============================================================
STEP 4: PREDICTION ON A SINGLE MRI IMAGE
Early Detection of MCI using Stacked CNN Feature Fusion
=============================================================

HOW TO USE:
  - In Colab / Jupyter: Set IMAGE_PATH below and run the cell
  - In terminal:        python 4_predict_single.py path/to/scan.jpg
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = ["CN (Cognitively Normal)", "MCI (Mild Cognitive Impairment)"]
IMG_SIZE    = (224, 224)

# ─────────────────────────────────────────────
# SET YOUR IMAGE PATH HERE (for Colab/Jupyter)
# ─────────────────────────────────────────────
IMAGE_PATH = "path/to/your/mri_scan.jpg"   # <-- change this


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized   = cv2.resize(img_rgb, IMG_SIZE)
    img_norm      = img_resized.astype(np.float32) / 255.0
    img_batch     = np.expand_dims(img_norm, axis=0)   # (1, 224, 224, 3)
    return img_rgb, img_batch


def load_model():
    for path in ["saved_models/best_model_finetuned.keras",
                 "saved_models/best_model.keras"]:
        if os.path.exists(path):
            print(f"[INFO] Model loaded: {path}")
            return tf.keras.models.load_model(path)
    raise FileNotFoundError("No saved model found. Run 2_model_training.py first!")


def predict(image_path):
    print("=" * 50)
    print("  MCI DETECTION - Single Image Prediction")
    print("=" * 50)

    model = load_model()
    original_img, img_batch = preprocess_image(image_path)

    # Model has two named inputs — feed the same image to both
    prob        = model.predict(
        {"resnet_input": img_batch, "densenet_input": img_batch},
        verbose=0
    )[0]
    pred_class  = np.argmax(prob)
    confidence  = prob[pred_class] * 100

    print(f"\n  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {CLASS_NAMES[pred_class]}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  CN  prob   : {prob[0]*100:.2f}%")
    print(f"  MCI prob   : {prob[1]*100:.2f}%")

    # ── Visualize ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MCI Detection - Prediction Result", fontsize=14, fontweight="bold")

    color = "tomato" if pred_class == 1 else "steelblue"

    axes[0].imshow(original_img)
    axes[0].set_title(
        f"Prediction: {CLASS_NAMES[pred_class]}\nConfidence: {confidence:.1f}%",
        fontsize=11, fontweight="bold", color=color
    )
    axes[0].axis("off")

    bars = axes[1].barh(
        ["CN", "MCI"],
        [prob[0]*100, prob[1]*100],
        color=["steelblue", "tomato"],
        edgecolor="black"
    )
    axes[1].set_xlim(0, 110)
    axes[1].set_xlabel("Probability (%)", fontsize=12)
    axes[1].set_title("Class Probabilities", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, [prob[0]*100, prob[1]*100]):
        axes[1].text(val + 1, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/single_prediction.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\n[INFO] Result saved: logs/single_prediction.png")


# ─────────────────────────────────────────────
# Entry point — works in both terminal & Colab
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Terminal usage: python 4_predict_single.py path/to/scan.jpg
    # Colab usage:    just set IMAGE_PATH above and run
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-f"):
        # Running from terminal with a path argument
        image_path = sys.argv[1]
    else:
        # Running in Colab / Jupyter — use the variable set at the top
        image_path = IMAGE_PATH

    if image_path == "path/to/your/mri_scan.jpg":
        print("=" * 55)
        print("  Please set IMAGE_PATH at the top of this script")
        print("  to your actual MRI image file path, then run again.")
        print("=" * 55)
    else:
        predict(image_path)
