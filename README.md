# ⚽ Football Player Roles Classification using CNN-Based Deep Learning

> Classifying football players into **Goalkeeper, Defender, Midfielder, and Forward** roles from portrait images using custom and transfer learning CNN models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/yashbedekar07/football-player-role-classification-ml-notebook)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Explainability (Grad-CAM & SHAP)](#-explainability-grad-cam--shap)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [References](#-references)

---

## 🔍 Project Overview

In modern football, player analysis is a critical part of team strategy, scouting, and recruitment. This project explores whether a **Convolutional Neural Network (CNN)** can learn visual patterns from player portrait images alone — such as body type, build, and posture — to accurately predict a player's **positional role**.

This is a **4-class image classification** problem:

| Label | Role |
|-------|------|
| `GK` | Goalkeeper |
| `DEF` | Defender |
| `MID` | Midfielder |
| `FWD` | Forward |

Three models are trained and compared:
- ✅ **Custom CNN** (built from scratch)
- ✅ **MobileNetV2** (transfer learning + fine-tuning)
- ✅ **ResNet50** (transfer learning + fine-tuning)

Model decisions are explained visually using **Grad-CAM** and **SHAP** pixel attribution.

---

## 📦 Dataset

**Source:** [FIFA 22 Complete Player Dataset — Stefano Leone (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)

Player face portrait images are downloaded programmatically from the `player_face_url` field in `players_22.csv`. Each image is a standardized PNG portrait at a consistent resolution, lighting, and background.

### Dataset Summary

| Attribute | Details |
|-----------|---------|
| Total Images (after balancing) | 3,200 |
| Number of Classes | 4 |
| Images per Class | 800 (balanced) |
| Image Format | PNG (RGB) |
| Input Resolution | 224 × 224 pixels |
| Total Players in Raw CSV | ~19,239 |

### Raw Class Distribution (Before Balancing)

| Role | Player Count | % of Total |
|------|-------------|-----------|
| Goalkeeper | 2,132 | 11.1% |
| Defender | 6,394 | 33.2% |
| Midfielder | 7,033 | 36.6% |
| Forward | 3,680 | 19.1% |

> **Imbalance handling:** Stratified sampling was applied to cap each class at **800 images**, producing a perfectly balanced dataset of 3,200 images.

### Train / Validation / Test Split (70 / 15 / 15)

| Split | GK | DEF | MID | FWD | Total |
|-------|----|-----|-----|-----|-------|
| Train (70%) | 560 | 560 | 560 | 560 | 2,240 |
| Validation (15%) | 120 | 120 | 120 | 120 | 480 |
| Test (15%) | 120 | 120 | 120 | 120 | 480 |

---

## 🔧 Methodology

### 1. Data Collection & Role Mapping
Player positions from the FIFA 22 CSV (`GK`, `CB`, `RB`, `CM`, `CAM`, `ST`, `LW`, etc.) are mapped to one of four high-level roles. Players with null positions or missing face URLs are discarded.

### 2. Image Downloading & Validation
Images are downloaded in parallel using a threaded pipeline with retry logic. Each download is validated with PIL (Pillow) to discard corrupt files.

### 3. Dataset Balancing & Splitting
Stratified sampling limits each class to 800 images. The balanced dataset is split with `sklearn.model_selection.train_test_split` using stratification.

### 4. Data Augmentation Pipeline (tf.data)
Training images are augmented on-the-fly using Keras augmentation layers:
- Random horizontal flip
- Random rotation (±8°)
- Random zoom (±10%)
- Random contrast adjustment (±10%)

### 5. Model Training & Evaluation
Three models are trained; transfer learning models use a **two-phase strategy**:
- **Phase 1 (Feature Extraction):** Base model frozen, classification head trained for up to 8 epochs
- **Phase 2 (Fine-Tuning):** Last 30 layers unfrozen, trained at reduced LR (`1e-5`) for up to 5 epochs

### 6. Explainability
Model decisions are inspected using **Grad-CAM** (region-level heatmaps) and **SHAP** (pixel-level attribution).

---

## 🏗️ Model Architectures

### Custom CNN (Built from Scratch)

| Layer / Block | Configuration | Output Shape |
|---------------|--------------|--------------|
| Input | 224 × 224 × 3 RGB | 224 × 224 × 3 |
| Rescaling | Normalize to [0, 1] | 224 × 224 × 3 |
| Block 1: Conv2D ×2 | 32 filters, 3×3, ReLU, BatchNorm | 224 × 224 × 32 |
| Block 1: MaxPool + Dropout | 2×2 pool, 15% | 112 × 112 × 32 |
| Block 2: Conv2D ×2 | 64 filters, 3×3, ReLU, BatchNorm | 112 × 112 × 64 |
| Block 2: MaxPool + Dropout | 2×2 pool, 20% | 56 × 56 × 64 |
| Block 3: Conv2D ×3 | 128 filters, 3×3, ReLU, BatchNorm | 56 × 56 × 128 |
| Block 3: MaxPool + Dropout | 2×2 pool, 25% | 28 × 28 × 128 |
| GlobalAveragePooling2D | — | 128 |
| Dense + Dropout | 128 neurons, ReLU, 40% | 128 |
| Output (Softmax) | 4 classes | 4 |

> **Total Trainable Parameters:** 453,412 &nbsp;|&nbsp; **Optimizer:** Adam &nbsp;|&nbsp; **Loss:** Categorical Cross-Entropy

### Transfer Learning Models

| Model | Base | Fine-Tuned Layers | Parameters |
|-------|------|-------------------|-----------|
| ResNet50 | ResNet50 (ImageNet) | Last 30 layers | ~23.6M |
| MobileNetV2 | MobileNetV2 (ImageNet) | Last 30 layers | ~2.3M |

Both transfer models use the same classification head:
`GlobalAveragePooling2D → Dropout(0.4) → Dense(4, softmax)`

---

## 📊 Results

### Final Model Comparison (Test Set)

| Model | Test Accuracy | Macro F1-Score | Parameters |
|-------|-------------|----------------|-----------|
| **Custom CNN** | **46.44%** | 0.4290 | 453,412 |
| MobileNetV2 | 44.73% | **0.4424** | ~2.3M |
| ResNet50 | 40.17% | 0.2429 | ~23.6M |

> All models exceed the **25% random baseline**, confirming that meaningful visual patterns exist in player portrait images — despite the inherent visual ambiguity of the classification task.

### Key Observations

- The **Custom CNN** achieves the highest test accuracy (46.44%), generalizing well on this small, visually ambiguous dataset while avoiding the over-parameterization risk of deeper architectures.
- **MobileNetV2** achieves the best Precision (0.4807) and Macro F1 (0.4424), offering the best trade-off between accuracy and reliability.
- **ResNet50** underperforms relative to its size, likely due to the small dataset size and insufficient fine-tuning epochs.
- The most frequent confusion occurs between **Midfielder and Forward** roles — expected, given the visual similarity of players in these positions.

---

## 🔍 Explainability (Grad-CAM & SHAP)

**Grad-CAM** heatmaps show which image regions the model attends to when making predictions:
- ✅ *Correctly classified* images show focused attention on the **upper body, jersey, and shoulder regions**
- ❌ *Misclassified* images show diffuse or background-directed attention, reflecting the model's uncertainty

**SHAP pixel attribution** confirms that:
- No single dominant pixel region drives predictions across all models
- Attribution is broadly distributed, consistent with the visual ambiguity of this classification task

---

## 📁 Project Structure

```
football-role-classification-ml/
│
├── notebooks/
│   └── Football_Player_Role_Classification_ML_Notebook.ipynb
│
├── data/                        # (Not included — downloaded at runtime)
│   ├── GK/
│   ├── DEF/
│   ├── MID/
│   └── FWD/
│
├── outputs/
│   ├── figures/                 # Training curves, confusion matrices, Grad-CAM, SHAP
│   └── models/                  # Saved model weights (optional)
│
├── README.md
└── requirements.txt
```

---

## 🚀 Setup & Usage

### Requirements

```bash
pip install tensorflow keras scikit-learn pillow matplotlib seaborn shap
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Running the Notebook

This project is designed to run on **Kaggle Notebooks** with GPU acceleration.

1. Open the notebook on Kaggle: [Football Player Role Classification ML Notebook](https://www.kaggle.com/code/yashbedekar07/football-player-role-classification-ml-notebook)
2. Attach the [FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) as an input dataset
3. Enable GPU accelerator (P100 recommended)
4. Run all cells in order

### To Run Locally

```bash
git clone https://github.com/yashbedekar-DS/football-role-classification-ml.git
cd football-role-classification-ml
jupyter notebook notebooks/Football_Player_Role_Classification_ML_Notebook.ipynb
```

> ⚠️ Local execution requires a stable internet connection for the image download step and a CUDA-enabled GPU for reasonable training times.

---

## 📚 References

- **FIFA 22 Complete Player Dataset** — Stefano Leone, Kaggle (2022). https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Howard, A. G. et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.
- TensorFlow/Keras Documentation — https://www.tensorflow.org/api_docs

---

## 👤 Author

**Yash David Bedekar**  
MSc Data Science — University of Europe for Applied Sciences, Potsdam  
📧 [GitHub Profile](https://github.com/yashbedekar-DS) &nbsp;|&nbsp; 🏆 [Kaggle Profile](https://www.kaggle.com/yashbedekar07)

---

*Machine Learning Course Project — June 2026*
