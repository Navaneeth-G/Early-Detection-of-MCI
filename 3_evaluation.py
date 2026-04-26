"""
=============================================================
STEP 3: MODEL EVALUATION
Early Detection of MCI using Stacked CNN Feature Fusion
=============================================================
This script:
- Loads the best saved model
- Evaluates on the test set
- Generates: Confusion Matrix, ROC Curve, Classification Report
- Computes: Accuracy, AUC, Sensitivity, Specificity, F1-score
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score, accuracy_score
)
import tensorflow as tf

CLASS_NAMES = ["CN", "MCI"]


# ─────────────────────────────────────────────
# 1.  LOAD MODEL & TEST DATA
# ─────────────────────────────────────────────
def load_model_and_data():
    # Try fine-tuned model first, fall back to Phase 1 model
    for model_path in ["saved_models/best_model_finetuned.keras",
                        "saved_models/best_model.keras"]:
        if os.path.exists(model_path):
            print(f"[INFO] Loading model: {model_path}")
            model = tf.keras.models.load_model(model_path)
            break
    else:
        raise FileNotFoundError(
            "[ERROR] No saved model found. Run '2_model_training.py' first!"
        )

    # Load test data
    try:
        X_test = np.load("processed_data/X_test.npy")
        y_test = np.load("processed_data/y_test.npy")
    except FileNotFoundError:
        raise FileNotFoundError(
            "[ERROR] Test data not found. Run '1_data_preprocessing.py' first!"
        )

    print(f"  Test samples : {len(X_test)}")
    print(f"  CN  samples  : {sum(y_test==0)}")
    print(f"  MCI samples  : {sum(y_test==1)}")

    return model, X_test, y_test


# ─────────────────────────────────────────────
# 2.  PREDICTIONS
# ─────────────────────────────────────────────
def get_predictions(model, X_test):
    print("\n[INFO] Running predictions on test set...")
    # Model expects [resnet_input, densenet_input] — same image for both
    y_prob = model.predict([X_test, X_test], verbose=1)   # Shape: (N, 2)
    y_pred = np.argmax(y_prob, axis=1)           # Convert to class indices
    y_prob_mci = y_prob[:, 1]                    # Probability of MCI (positive class)
    return y_pred, y_prob_mci


# ─────────────────────────────────────────────
# 3.  METRICS SUMMARY
# ─────────────────────────────────────────────
def compute_metrics(y_test, y_pred, y_prob_mci):
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for MCI
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for CN

    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob_mci)
    roc_auc = auc(fpr, tpr)

    print(f"\n  Accuracy       : {acc*100:.2f}%")
    print(f"  AUC-ROC        : {roc_auc:.4f}")
    print(f"  Sensitivity    : {sensitivity*100:.2f}%  (MCI correctly identified)")
    print(f"  Specificity    : {specificity*100:.2f}%  (CN correctly identified)")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"\n  True Positives (MCI→MCI) : {tp}")
    print(f"  True Negatives (CN→CN)   : {tn}")
    print(f"  False Positives (CN→MCI) : {fp}")
    print(f"  False Negatives (MCI→CN) : {fn}")

    print("\n─── Detailed Classification Report ───")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return cm, fpr, tpr, roc_auc, acc, sensitivity, specificity


# ─────────────────────────────────────────────
# 4.  VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]);  ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')

    # Annotate each cell
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("[INFO] Confusion matrix saved: logs/confusion_matrix.png")


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curve — MCI Detection", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/roc_curve.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("[INFO] ROC curve saved: logs/roc_curve.png")


def plot_metrics_bar(acc, roc_auc, sensitivity, specificity):
    metrics = ["Accuracy", "AUC-ROC", "Sensitivity\n(MCI Recall)", "Specificity\n(CN Recall)"]
    values  = [acc, roc_auc, sensitivity, specificity]
    colors  = ['steelblue', 'seagreen', 'tomato', 'darkorange']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val*100:.1f}%", ha='center', fontweight='bold', fontsize=11)
    plt.ylim(0, 1.15)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Summary", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/metrics_summary.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("[INFO] Metrics bar chart saved: logs/metrics_summary.png")


# ─────────────────────────────────────────────
# 5.  SHOW MISCLASSIFIED SAMPLES
# ─────────────────────────────────────────────
def show_misclassified(X_test, y_test, y_pred, y_prob_mci, n=8):
    wrong_idx = np.where(y_test != y_pred)[0]
    if len(wrong_idx) == 0:
        print("\n[INFO] No misclassified samples — perfect test accuracy!")
        return

    n = min(n, len(wrong_idx))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Misclassified Samples", fontsize=13, fontweight='bold', color='red')

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue
        idx = wrong_idx[i]
        ax.imshow(X_test[idx])
        true_lbl = CLASS_NAMES[y_test[idx]]
        pred_lbl = CLASS_NAMES[y_pred[idx]]
        conf     = y_prob_mci[idx] if y_pred[idx] == 1 else 1 - y_prob_mci[idx]
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf*100:.1f}%)",
                     fontsize=9, color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("logs/misclassified_samples.png", dpi=100, bbox_inches='tight')
    plt.show()
    print(f"[INFO] {len(wrong_idx)} misclassified samples visualized.")


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
def evaluate():
    print("=" * 55)
    print("  MCI DETECTION — Evaluation")
    print("=" * 55)

    os.makedirs("logs", exist_ok=True)

    model, X_test, y_test = load_model_and_data()
    y_pred, y_prob_mci    = get_predictions(model, X_test)
    cm, fpr, tpr, roc_auc, acc, sensitivity, specificity = compute_metrics(
        y_test, y_pred, y_prob_mci
    )

    print("\n[INFO] Generating plots...")
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_metrics_bar(acc, roc_auc, sensitivity, specificity)
    show_misclassified(X_test, y_test, y_pred, y_prob_mci)

    print("\n✅ Evaluation complete! All plots saved in ./logs/")


if __name__ == "__main__":
    evaluate()
