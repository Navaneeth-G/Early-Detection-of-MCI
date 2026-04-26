# 🧠 Early Detection of MCI using Stacked CNN Feature Fusion

### *Published Research. Two AI Brains. One Mission — Detect Memory Loss Before It Starts.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-RAET'26-purple.svg)]()

---

## 📋 Overview

A professional-grade medical AI system for the early detection of **Mild Cognitive Impairment (MCI)** — often the earliest detectable stage of Alzheimer's disease — using brain MRI scans.

This project was presented at the **RAET'26 National Conference** and implements a novel **Stacked CNN Feature Fusion** architecture that combines two powerful neural networks to capture both global brain structure and fine-grained textural details.

---

## 🏗️ Architecture

```
        Input MRI (224x224)
               │
    ┌──────────┴──────────┐
    ▼                     ▼
🧠 ResNet50          🔍 DenseNet121
(Global Structure)   (Fine-grained Textures)
    │                     │
    └──────────┬──────────┘
               ▼
        ⚡ Feature Fusion
         (Concatenation)
               │
               ▼
         🧬 Dense Layers
               │
               ▼
      🏥 Binary Classification
         ┌─────┴─────┐
         ▼           ▼
   ✅ CN (Normal)  ⚠️ MCI (Impaired)
```

| Component | Role |
|-----------|------|
| **🧠 ResNet50** | Captures global structural patterns — overall brain shape, ventricle size |
| **🔍 DenseNet121** | Captures fine-grained textures — subtle tissue changes invisible to the human eye |
| **⚡ Feature Fusion** | Concatenates features from both backbones before final classification |
| **🔄 Two-Phase Training** | Phase 1: Frozen ImageNet weights for stability → Phase 2: Fine-tune top 30 layers for domain adaptation |

---

## 💡 What Makes This Project Unique

✔ **Dual-backbone CNN** — ResNet50 + DenseNet121 feature fusion (rare in student projects)  
✔ **Custom MRI validation layer** — Rejects non-MRI images automatically  
✔ **Medical-grade evaluation** — Sensitivity, Specificity, AUC-ROC (not just accuracy)  
✔ **Published research** — Presented at RAET'26 National Conference  
✔ **End-to-end system** — Training → Validation → Deployment (Streamlit)  
✔ **Real-world impact** — Early Alzheimer's detection before symptoms worsen  

> *Most student projects stop at model training. This one goes further — into validation, deployment, and clinical relevance.*

---

## 📂 Project Structure

```
📁 Early-Detection-of-MCI/
├── 📄 1_data_preprocessing.py   ← MRI resizing, normalization, train/test split
├── 📄 2_model_training.py       ← Dual-backbone fusion + training loop
├── 📄 3_evaluation.py           ← Confusion matrix, ROC curve, metrics
├── 📄 4_predict_single.py       ← Single image prediction (command line)
├── 🌐 app.py                    ← Streamlit web dashboard
├── 🛡️ mri_validator.py          ← Input validation — rejects non-MRI images
├── 📄 .gitignore                ← Excludes dataset, models, venv
├── 📄 requirements.txt          ← All dependencies
├── 📄 LICENSE                   ← MIT License
└── 📄 README.md                 ← This file
```

---

## 🛡️ MRI Validator — A Unique Feature

Most medical AI projects skip input validation. This project doesn't.

The `mri_validator.py` module checks every uploaded image:

| Check | Method |
|-------|--------|
| **🔲 Grayscale verification** | Confirms image is medical-style monochrome |
| **📊 Pixel histogram analysis** | Compares distribution against known MRI patterns |
| **🎯 Central region intensity** | Real MRIs have specific brightness in the brain region |

> *Upload a cat photo? 🐱 It gets rejected. Upload a brain MRI? 🧠 It passes through.*

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Navaneeth-G/Early-Detection-of-MCI.git
cd Early-Detection-of-MCI
pip install -r requirements.txt
```

### 2. Obtain the Dataset

This project uses the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset.

> ⚠️ The dataset is **NOT included** in this repository. ADNI data is governed by strict patient privacy regulations and cannot be redistributed.

**To obtain access:**
1. Visit [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Create an account and submit a data use application
3. Once approved, download MRI scans for CN (Cognitively Normal) and MCI subjects
4. Organize as:

```
dataset/
├── CN/      ← Normal brain MRIs
└── MCI/     ← MCI brain MRIs
```

### 3. Run the Pipeline

```bash
# Step 1: Preprocess the data
python 1_data_preprocessing.py

# Step 2: Train the model (takes several hours on GPU)
python 2_model_training.py

# Step 3: Evaluate performance
python 3_evaluation.py

# Step 4: Launch the web app
streamlit run app.py
```

### 4. Single Image Prediction

```bash
python 4_predict_single.py --image path/to/mri_scan.jpg
```

---

## 📊 Results

> ⚠️ **Phase 1 — Code & Architecture Complete.**
> Full model training requires ADNI dataset access.
> Results below are expected based on similar dual-backbone CNN architectures
> in published literature. Will be updated with actual metrics after
> ADNI dataset acquisition and training.

| Metric | Expected Range |
|--------|---------------|
| 🎯 Accuracy | 90–94% |
| 🔬 Sensitivity (Recall) | 92–96% |
| 🛡️ Specificity | 88–92% |
| 📈 AUC-ROC | 0.94–0.97 |

### 🧠 Why Sensitivity Matters Most

In MCI detection, **false negatives are dangerous**. Missing an early sign of cognitive decline delays treatment. This architecture is designed to prioritize **high sensitivity** — catching cases early, even at the cost of a few false positives.

---

## 📸 Project Screenshots

> Screenshots of the Streamlit dashboard will be added after deployment.
> Confusion matrix, ROC curve, and sample predictions will be added after ADNI dataset training.

---

## 📊 Evaluation Metrics

The model is evaluated using **medical-grade metrics** — not just accuracy:

| Metric | Why It Matters |
|--------|---------------|
| 🎯 Accuracy | Overall correctness |
| 🔬 Sensitivity (Recall) | How many MCI cases we caught — critical, because false negatives mean missed diagnoses |
| 🛡️ Specificity | How many healthy people we correctly identified — prevents unnecessary alarm |
| 📈 AUC-ROC | Overall discriminatory power of the model |
| 📋 Confusion Matrix | Shows exactly where the model gets confused |

> In medical AI, a **false negative** (missing a real MCI case) is far more dangerous than a false positive. Our evaluation prioritizes Sensitivity.

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| 🧠 Deep Learning | TensorFlow, Keras |
| 👁️ Computer Vision | OpenCV, Pillow |
| 📊 Data Processing | NumPy, Scikit-learn |
| 📉 Visualization | Matplotlib |
| 🌐 Web Interface | Streamlit |
| 🛡️ Validation | Custom pixel histogram analysis |

---

## 🔮 Future Work

- 🧬 Integrate multimodal data (PET scans + MRI)
- 🔥 Add Explainable AI with Grad-CAM heatmaps
- 🤖 Upgrade to Vision Transformers (ViT)
- ☁️ Deploy as a cloud API for remote access
- 🏥 Expand to multi-class classification (CN vs MCI vs Alzheimer's)
- 🌍 Validate on external hospital datasets

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> ⚠️ **Medical Disclaimer:** This tool is for **research purposes only**. It is not FDA-approved and should not be used for clinical diagnosis. Always consult a qualified neurologist.

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@conference{NavaneethG2026RAET,
  title     = {Advancing Early Diagnosis: Predicting Mild Cognitive Impairment
               Progression Using Deep Learning on MRI Features},
  author    = {Navaneeth G},
  booktitle = {Second National Conference on Recent Advances in Engineering
               and Technology (RAET'26)},
  year      = {2026}
}
```

---

## 📬 Contact

| Platform | Link |
|----------|------|
| 📧 Email | navaneethgnambiar@gmail.com |
| 💼 LinkedIn | [linkedin.com/in/navaneeth-g-b1a735278](https://linkedin.com/in/navaneeth-g-b1a735278) |
| 🐙 GitHub | [github.com/Navaneeth-G](https://github.com/Navaneeth-G) |

---

## 🔗 Related Work

This project is part of a broader healthcare AI portfolio.

- [GitHub Profile](https://github.com/Navaneeth-G)
- [LinkedIn](https://linkedin.com/in/navaneeth-g-b1a735278)

---

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b><br>
  <i>Built with ❤️ by Navaneeth G — One MRI scan at a time.</i>
</p>
