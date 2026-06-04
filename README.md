# ⚽ Football Player Roles Classification using CNN

> **Machine Learning Project — Phase II**  
> Classifying football player positions (Goalkeeper, Defender, Midfielder, Forward) from official FIFA 22 player portrait images using Convolutional Neural Networks.

---

## 📌 Project Overview

This project applies deep learning and computer vision to automatically classify football players into one of four positional roles based solely on their official FIFA 22 player portrait images. Three models are trained and compared: a custom-built CNN, ResNet50, and MobileNetV2, with explainability provided through Grad-CAM and SHAP visualizations.

| Detail | Info |
|---|---|
| **Student** | Yash David Bedekar |
| **Course** | Machine Learning |
| **Dataset** | FIFA 22 Complete Player Dataset (Kaggle) |
| **Framework** | TensorFlow 2.x / Keras |
| **Environment** | Kaggle Notebook |
| **Submission** | Phase 2 — Proposal & Code Implementation |

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📓 Kaggle Notebook | [football-player-roles-classification-ml](https://www.kaggle.com/code/yashbedekar07/football-player-roles-classification-ml) |
| 📦 Dataset | [FIFA 22 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) |
| 🐙 GitHub Repo | [football-role-classification-ml](https://github.com/yashbedekar-DS/football-role-classification-ml) |

---

## 🗂️ Repository Structure

```
football-role-classification-ml/
│
├── notebook/
│   └── football-player-roles-classification-ml.ipynb   # Main Kaggle notebook
│
├── figures/
│   ├── role_counts_before_download.pdf                  # Class distribution chart
│   ├── CustomCNN_gradcam.pdf                            # Grad-CAM heatmaps (Custom CNN)
│   ├── ResNet50_gradcam.pdf                             # Grad-CAM heatmaps (ResNet50)
│   ├── MobileNetV2_gradcam.pdf                          # Grad-CAM heatmaps (MobileNetV2)
│   └── shap_interpretability_matrix.pdf                 # SHAP pixel attribution matrix
│
├── images/                                              # Downloaded player portrait images
│   ├── Goalkeeper/   (800 images)
│   ├── Defender/     (800 images)
│   ├── Midfielder/   (800 images)
│   └── Forward/      (800 images)
│
└── README.md
```

> **Note:** The `images/` folder contains a sample of the downloaded player portraits. Full dataset download happens automatically when you run the notebook on Kaggle with the FIFA 22 dataset attached.

---

## 📊 Dataset

- **Source:** FIFA 22 Complete Player Dataset (`players_22.csv`)
- **Total raw players:** ~19,239
- **Images used (after balancing):** 3,200 (800 per class)
- **Image type:** PNG player portrait / face images
- **Input resolution:** 224 × 224 × 3 (RGB)

### Class Distribution (Raw CSV)

| Role | Raw Count | % of Total |
|---|---|---|
| Goalkeeper | 2,132 | 11.1% |
| Defender | 6,394 | 33.2% |
| Midfielder | 7,033 | 36.6% |
| Forward | 3,680 | 19.1% |

### Position → Role Mapping

| FIFA Positions | Mapped Role |
|---|---|
| GK | Goalkeeper |
| CB, LB, RB, LCB, RCB, LWB, RWB | Defender |
| CM, CDM, CAM, LM, RM, LCM, RCM, LDM, RDM, LAM, RAM | Midfielder |
| ST, CF, LW, RW, LF, RF, LS, RS | Forward |

### Train / Validation / Test Split (Stratified)

| Split | Per Class | Total |
|---|---|---|
| Train (70%) | 560 | 2,240 |
| Validation (15%) | 120 | 480 |
| Test (15%) | 120 | 480 |

---

## 🧠 Models

### 1. Custom CNN (Built from Scratch)

```
Input (224×224×3) → Rescaling (÷255)
→ Block 1: Conv2D(32)×2 + BatchNorm + MaxPool + Dropout(15%)
→ Block 2: Conv2D(64)×2 + BatchNorm + MaxPool + Dropout(20%)
→ Block 3: Conv2D(128)×3 + BatchNorm + MaxPool + Dropout(25%)
→ GlobalAveragePooling2D
→ Dense(128, ReLU) + Dropout(40%)
→ Dense(4, Softmax)
```

- **Parameters:** 453,412
- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy

### 2. ResNet50 (Transfer Learning)

- Pre-trained on ImageNet (top excluded)
- Phase 1: Feature extraction (base frozen, 5 epochs)
- Phase 2: Fine-tuning last 30 layers (5 more epochs, lr=1e-5)
- **Parameters:** ~23.6M

### 3. MobileNetV2 (Transfer Learning)

- Pre-trained on ImageNet (top excluded)
- Phase 1: Feature extraction (base frozen, 4 epochs)
- Phase 2: Fine-tuning last 30 layers (4 more epochs, lr=1e-5)
- **Parameters:** ~2.3M

---

## 📈 Results

| Model | Test Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| **ResNet50** | **39.17%** | **0.390** | ~23.6M |
| MobileNetV2 | 35.63% | 0.358 | ~2.3M |
| CustomCNN | 33.75% | 0.300 | 453K |

> All models exceeded the random baseline of 25%, confirming that some signal exists in player portrait images. The relatively modest accuracy reflects the inherent visual ambiguity of classifying roles from headshot images without pose or action context.

---

## 🔍 Explainability

### Grad-CAM
Gradient-weighted Class Activation Maps were generated for all three models, highlighting which regions of the player portrait activated most strongly during classification. Heatmaps show the model tends to focus on the upper body and face structure.

### SHAP
SHAP pixel attribution was applied using ResNet50 as the active model. The `PartitionExplainer` with an inpainting masker assigns pixel-level contribution scores — red regions positively influence the prediction, blue regions suppress it.

All figures are saved as PDF in the `/figures` directory.

---

## 🔄 Data Augmentation

Applied only during training:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])
```

---

## ▶️ How to Run

### Option 1: Kaggle Notebook (Recommended)

1. Open the notebook at: https://www.kaggle.com/code/yashbedekar07/football-player-roles-classification-ml
2. Click **Copy & Edit** to fork your own version
3. Add the FIFA 22 dataset:
   - Go to **+ Add Data** → search for `fifa-22-complete-player-dataset` by `stefanoleone992`
   - Attach it to your notebook session
4. Click **Run All** — the notebook will auto-detect the dataset path and begin downloading player images in parallel
5. All figures, tables, and model checkpoints are saved to `/kaggle/working/`

### Option 2: Local Environment

```bash
# Clone the repository
git clone https://github.com/yashbedekar-DS/football-role-classification-ml.git
cd football-role-classification-ml

# Install dependencies
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn pillow requests shap

# Download the dataset manually from Kaggle and place players_22.csv in the working directory

# Open the notebook
jupyter notebook notebook/football-player-roles-classification-ml.ipynb
```

> **Hardware Note:** A GPU is strongly recommended. The notebook was developed and tested on a Kaggle P100 GPU instance. CPU-only execution will be significantly slower.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building, training, and evaluation |
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Visualization and plotting |
| `scikit-learn` | Train/val/test splitting, metrics |
| `Pillow (PIL)` | Image downloading and validation |
| `requests` | Parallel image downloading |
| `shap` | SHAP pixel attribution explainability |

---

## 📋 Output Files Generated

After running the notebook, the following outputs are saved:

```
/kaggle/working/
├── figures/
│   ├── role_counts_before_download.pdf
│   ├── CustomCNN_gradcam.pdf
│   ├── ResNet50_gradcam.pdf
│   ├── MobileNetV2_gradcam.pdf
│   └── shap_interpretability_matrix.pdf
│
├── tables/
│   ├── train_split.csv
│   ├── validation_split.csv
│   ├── test_split.csv
│   ├── model_comparison.csv
│   ├── CustomCNN_confusion_matrix.csv
│   ├── ResNet50_confusion_matrix.csv
│   └── MobileNetV2_confusion_matrix.csv
│
└── models/
    └── [model checkpoint files]
```

---

## 📚 References

- Stefano Leone (2022). *FIFA 22 Complete Player Dataset*. Kaggle.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
- Lundberg, S. M. & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. NeurIPS 2017.

---

## 👤 Author

**Yash David Bedekar**  
Machine Learning Course Project — Phase II  
Kaggle: [@yashbedekar07](https://www.kaggle.com/yashbedekar07) | GitHub: [@yashbedekar-DS](https://github.com/yashbedekar-DS)
