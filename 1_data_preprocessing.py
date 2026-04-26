"""
=============================================================
STEP 1: DATA PREPROCESSING
Early Detection of MCI using Stacked CNN Feature Fusion
=============================================================
This script:
- Loads MRI images from ADNI dataset (CN vs MCI folders)
- Resizes and normalizes images
- Splits into train/val/test sets
- Saves processed data for model training
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION — Edit these paths!
# ─────────────────────────────────────────────
DATA_DIR = "./dataset"          # Root folder containing CN/ and MCI/ subfolders
IMG_SIZE = (224, 224)           # Required by ResNet & DenseNet (ImageNet standard)
RANDOM_SEED = 42

# Expected folder structure:
# dataset/
#   CN/     ← Cognitively Normal MRI slices (.jpg or .png)
#   MCI/    ← Mild Cognitive Impairment MRI slices


def load_images_from_folder(folder_path, label, img_size):
    """Load all images from a folder and assign a numeric label."""
    images = []
    labels = []
    skipped = 0

    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(supported_ext):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  [WARNING] Could not read: {filename}")
            skipped += 1
            continue

        # Convert BGR (OpenCV default) → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        img = cv2.resize(img, img_size)

        images.append(img)
        labels.append(label)

    print(f"  Loaded {len(images)} images | Skipped {skipped} | Label={label}")
    return images, labels


def normalize_images(images):
    """Normalize pixel values to [0, 1] range."""
    images = np.array(images, dtype=np.float32)
    images = images / 255.0
    return images


def show_sample_images(X, y, class_names, n=8):
    """Display a grid of sample images from the dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Sample MRI Images from Dataset", fontsize=14, fontweight='bold')

    indices = np.random.choice(len(X), n, replace=False)
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        ax.imshow(X[idx])
        ax.set_title(f"Class: {class_names[y[idx]]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("[INFO] Sample image grid saved as 'sample_images.png'")


def show_class_distribution(y, class_names):
    """Bar chart of class counts."""
    counts = Counter(y)
    labels = [class_names[k] for k in sorted(counts.keys())]
    values = [counts[k] for k in sorted(counts.keys())]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['steelblue', 'tomato'], edgecolor='black')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', fontweight='bold')
    plt.title("Class Distribution (CN vs MCI)", fontweight='bold')
    plt.ylabel("Number of Images")
    plt.savefig("class_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("[INFO] Class distribution chart saved as 'class_distribution.png'")


def preprocess_dataset():
    print("=" * 55)
    print("  MCI DETECTION — Data Preprocessing")
    print("=" * 55)

    # ── 1. Check folders exist ──────────────────────────────
    cn_dir  = os.path.join(DATA_DIR, "CN")
    mci_dir = os.path.join(DATA_DIR, "MCI")

    for d in [cn_dir, mci_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"\n[ERROR] Folder not found: {d}"
                f"\nMake sure your dataset folder has CN/ and MCI/ subfolders."
            )

    # ── 2. Load images ──────────────────────────────────────
    print("\n[1/5] Loading images...")
    print(f"  CN  folder → {cn_dir}")
    cn_images,  cn_labels  = load_images_from_folder(cn_dir,  label=0, img_size=IMG_SIZE)
    print(f"  MCI folder → {mci_dir}")
    mci_images, mci_labels = load_images_from_folder(mci_dir, label=1, img_size=IMG_SIZE)

    # ── 3. Combine & normalize ──────────────────────────────
    print("\n[2/5] Combining and normalizing...")
    all_images = cn_images + mci_images
    all_labels = cn_labels + mci_labels

    X = normalize_images(all_images)
    y = np.array(all_labels, dtype=np.int32)

    class_names = {0: "CN", 1: "MCI"}
    print(f"  Total samples : {len(X)}")
    print(f"  Image shape   : {X.shape[1:]}")
    print(f"  CN  count     : {sum(y == 0)}")
    print(f"  MCI count     : {sum(y == 1)}")

    # ── 4. Train / Val / Test split ─────────────────────────
    print("\n[3/5] Splitting dataset (70% train / 15% val / 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # ── 5. Save preprocessed data ───────────────────────────
    print("\n[4/5] Saving preprocessed arrays to disk...")
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/X_train.npy", X_train)
    np.save("processed_data/X_val.npy",   X_val)
    np.save("processed_data/X_test.npy",  X_test)
    np.save("processed_data/y_train.npy", y_train)
    np.save("processed_data/y_val.npy",   y_val)
    np.save("processed_data/y_test.npy",  y_test)
    print("  Saved to ./processed_data/")

    # ── 6. Visualizations ───────────────────────────────────
    print("\n[5/5] Generating visualizations...")
    show_class_distribution(y, class_names)
    show_sample_images(X, y, class_names)

    print("\n✅ Preprocessing complete! Run '2_model_training.py' next.")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    preprocess_dataset()
