# ⚽ Football Player Role Classification using CNN-Based Deep Learning

> Classifying football players into **Goalkeeper, Defender, Midfielder, and Forward** roles from portrait images using custom and transfer learning CNN models — with Grad-CAM and SHAP explainability analysis.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/yashbedekar07/football-player-role-classification-cnn-ml)
[![Live Web App](https://img.shields.io/badge/Web%20App-Live-brightgreen?logo=react)](https://football-webapp.base44.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📄 Overleaf Report | [https://www.overleaf.com/8691872315rbmpkhwgkqqv#391ec0](https://www.overleaf.com/8691872315rbmpkhwgkqqv#391ec0) |
| 📓 Kaggle Notebook | [football-player-role-classification-cnn-ml](https://www.kaggle.com/code/yashbedekar07/football-player-role-classification-cnn-ml) |
| 📦 Kaggle Dataset | [Elite World Cup 2022 Players Image Dataset](https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset) |
| 💻 GitHub Repository | [football-role-classification-ml](https://github.com/yashbedekar-DS/football-role-classification-ml) |
| 🌐 Live Web App | [https://football-webapp.base44.app/](https://football-webapp.base44.app/) |

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Phases](#-project-phases)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Explainability (Grad-CAM & SHAP)](#-explainability-grad-cam--shap)
- [Web Application](#-web-application)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Deliverables](#-deliverables)
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

Model decisions are explained visually using **Grad-CAM** (region-level heatmaps) and **SHAP** (pixel-level attribution).

> ⚠️ **Honest Finding:** None of the three models reached the 70% accuracy target. The best result was Custom CNN at 46.44%. This is reported as a genuine academic finding — the task is inherently difficult because football player portraits are visually ambiguous across positional roles.

---

## 📅 Project Phases

This project was completed in three phases as part of course **PS26-DSC01 – Machine Learning 120 B** at the University of Europe for Applied Sciences, Potsdam.

| Phase | Title | Due Date | Status |
|-------|-------|----------|--------|
| Phase 1 | Topic Shortlisting and Selection | May 17, 2026 | ✅ Completed |
| Phase 2 | Proposal, Code & Implementation | June 7, 2026 | ✅ Completed |
| Phase 3 | Final Report & Presentation | June 28, 2026 | ✅ Submitted |

---

## 📦 Dataset

**Source:** [Elite World Cup 2022 Players Image Dataset — Peter Kibuchi (Kaggle)](https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset)

Player face portrait images were downloaded programmatically from the `player_face_url` field in `players_22.csv`. Each image is a standardized PNG portrait at consistent resolution, lighting, and background.

### Dataset Summary

| Attribute | Details |
|-----------|---------|
| Total Images (after balancing) | 3,200 |
| Images Used (after validation) | 2,325 |
| Number of Classes | 4 |
| Images per Class | ~800 (balanced) |
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

> **Imbalance handling:** Stratified sampling was applied to cap each class at 800 images, producing a balanced dataset of 3,200 images. After image validation, 2,325 images remained.

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

### 6. Explainability (XAI)
Model decisions are inspected using:
- **Grad-CAM** — gradient-weighted class activation maps for region-level heatmaps
- **SHAP** — pixel-level attribution maps showing what drives each prediction

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

> **Total Parameters:** 848,932 &nbsp;|&nbsp; **Optimizer:** Adam &nbsp;|&nbsp; **Loss:** Categorical Cross-Entropy

### Transfer Learning Models

| Model | Base | Fine-Tuned Layers | Parameters |
|-------|------|-------------------|-----------|
| MobileNetV2 | MobileNetV2 (ImageNet) | Last 30 layers | 2,586,948 |
| ResNet50 | ResNet50 (ImageNet) | Last 30 layers | 24,638,852 |

Both transfer models use the same classification head:
`GlobalAveragePooling2D → Dropout(0.4) → Dense(4, softmax)`

---

## 📊 Results

### Final Model Comparison (Test Set)

| Model | Test Accuracy | Macro F1-Score | Parameters | Training Time |
|-------|-------------|----------------|-----------|--------------|
| **Custom CNN** | **46.44%** | 0.4290 | 848,932 | 9.90 min |
| MobileNetV2 | 44.73% | **0.4424** | 2,586,948 | **7.87 min** |
| ResNet50 | 40.17% | 0.2429 | 24,638,852 | 9.82 min |

> All models exceed the **25% random baseline**, confirming that meaningful visual patterns exist in player portrait images — despite the inherent visual ambiguity of the classification task.

### Key Observations

- The **Custom CNN** achieves the highest test accuracy (46.44%), generalizing well on this small, visually ambiguous dataset while avoiding the over-parameterization risk of deeper architectures.
- **MobileNetV2** achieves the best Macro F1-Score (0.4424) and fastest training time (7.87 min), offering the best trade-off between accuracy and efficiency.
- **ResNet50** underperforms relative to its size (24.6M parameters), likely due to the small dataset size and insufficient fine-tuning epochs.
- The most frequent confusion occurs between **Midfielder and Forward** roles — expected, given the visual similarity of players in these positions.
- The **70% accuracy target was not reached** by any model. This is an honest finding: football player portrait images are visually ambiguous across positional roles, and the task may require additional contextual signals (kit number, tactical context) for higher accuracy.

---

## 🔍 Explainability (Grad-CAM & SHAP)

### Grad-CAM
Gradient-weighted Class Activation Maps show which image regions the model attends to when making predictions:
- ✅ **Correctly classified** images show focused attention on the **upper body, jersey, and shoulder regions**
- ❌ **Misclassified** images show diffuse or background-directed attention, reflecting the model's uncertainty

### SHAP Pixel Attribution
SHAP pixel attribution confirms that:
- No single dominant pixel region drives predictions across all models
- Attribution is broadly distributed, consistent with the visual ambiguity of this classification task
- All three models (Custom CNN, MobileNetV2, ResNet50) were analysed with SHAP

---

## 🌐 Web Application

An interactive React web application was built to demonstrate the project end-to-end.

**Live URL:** [https://football-webapp.base44.app/](https://football-webapp.base44.app/)

| Feature | Description |
|---------|-------------|
| 🏠 Home | Project overview and key results summary |
| 🤖 Predict Role | Simulate player role prediction with confidence scores |
| 🔥 Grad-CAM Viewer | Interactive heatmap visualisation for all three models |
| 📊 Model Comparison | Side-by-side accuracy, F1, parameters, and training time |
| 📁 Dataset Explorer | Browse the dataset structure and class distribution |
| 📈 Research Analytics | Training curves and performance charts |
| ℹ️ About | Project background and academic context |

**Tech Stack:** React 18 · Vite · Recharts · Base44 Hosting

---

## 📁 Project Structure

```
football-role-classification-ml/
│
├── notebooks/
│   └── Football_Player_Role_Classification_CNN_ML.ipynb
│
├── data/                        # (Not included — downloaded at runtime)
│   ├── GK/
│   ├── DEF/
│   ├── MID/
│   └── FWD/
│
├── outputs/
│   ├── figures/
│   │   ├── fig1_dataset_samples.png
│   │   ├── fig2_model_workflow.png
│   │   ├── fig2b_class_distribution.png
│   │   ├── fig3_training_curves.png
│   │   ├── fig4_confusion_matrices.png
│   │   ├── fig5_gradcam_correct.png
│   │   ├── fig6_gradcam_misclassified.png
│   │   ├── fig7_shap_custom_cnn.png
│   │   ├── fig7_shap_mobilenetv2.png
│   │   ├── fig7_shap_resnet50.png
│   │   └── fig8_final_comparison.png
│   ├── csv/
│   │   ├── CustomCNN_history.csv
│   │   ├── MobileNetV2_history.csv
│   │   ├── ResNet50_history.csv
│   │   ├── model_comparison.csv
│   │   ├── dataset_summary.csv
│   │   └── final_comparison_table.csv
│   └── models/                  # Saved model weights (optional)
│
├── web-app/                     # React web application source
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── README.md
│
├── report/
│   ├── main.tex                 # Overleaf LaTeX source (elsarticle format)
│   └── references.bib
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

1. Open the notebook on Kaggle: [Football Player Role Classification CNN ML](https://www.kaggle.com/code/yashbedekar07/football-player-role-classification-cnn-ml)
2. Attach the [FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) as an input dataset
3. Enable GPU accelerator (P100 recommended)
4. Run all cells in order

### To Run Locally

```bash
git clone https://github.com/yashbedekar-DS/football-role-classification-ml.git
cd football-role-classification-ml
pip install -r requirements.txt
jupyter notebook notebooks/Football_Player_Role_Classification_CNN_ML.ipynb
```

> ⚠️ Local execution requires a stable internet connection for the image download step and a CUDA-enabled GPU for reasonable training times.

### Running the Web App Locally

```bash
cd web-app
npm install
npm run dev
# Open http://localhost:5173
```

---

## 📦 Deliverables

All Phase 3 deliverables submitted on June 28, 2026:

| Deliverable | Format | Status |
|-------------|--------|--------|
| Academic Report | LaTeX / Overleaf (elsarticle) | ✅ Submitted |
| Final PDF Report | PDF exported from Overleaf | ✅ Submitted |
| Presentation Slides | PPTX (18 slides) | ✅ Submitted |
| Presentation Video | 10-minute narrated walkthrough | ✅ Submitted |
| Kaggle Notebook | Python / TensorFlow / Keras | ✅ Public |
| GitHub Repository | Full source code + figures | ✅ Public |
| Live Web App | React · Base44 Hosting | ✅ Live |
| Submission Document | Word (.docx) | ✅ Submitted |

---

## 📚 References

- **FIFA 22 Complete Player Dataset** — Stefano Leone, Kaggle (2022). https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
- **Elite World Cup 2022 Players Image Dataset** — Peter Kibuchi, Kaggle. https://www.kaggle.com/datasets/peterkibuchi/world-cup-2022-elite-players-image-dataset
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

*Machine Learning Course Project (PS26-DSC01) — Phase 3 Final Submission — June 2026*