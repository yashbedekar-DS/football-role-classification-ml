# Football Player Role Classification

This project uses a CNN-based image classification workflow to predict a football player's role from a portrait image.

The notebook behind this project is:

- [football-player-role-classification-cnn-ml.ipynb](./notebook/football-player-role-classification-cnn-ml.ipynb)

It compares three approaches:

- Custom CNN
- MobileNetV2 transfer learning
- ResNet50 transfer learning

The task is to classify each player into one of four roles:

- Goalkeeper
- Defender
- Midfielder
- Forward

## Project Idea

The main goal of the notebook is to test how far image-based role classification can go when the model only sees a player portrait rather than a full match image.

This is a difficult problem because role clues are often subtle. A portrait may show kit details, facial appearance, or image style, but it does not directly show field position or in-game action. That is why the project is useful for comparison, explainability, and model analysis, even though the accuracy is not expected to be extremely high.

## Notebook Workflow

The notebook follows this pipeline:

1. Load the World Cup 2022 elite players dataset.
2. Map FIFA position groups into four role labels.
3. Organize images into train, validation, and test folders.
4. Build data generators with augmentation for training.
5. Train three models:
   - a custom CNN from scratch
   - MobileNetV2 with transfer learning
   - ResNet50 with transfer learning
6. Evaluate all models on the held-out test set.
7. Save plots, tables, and model checkpoints into the `results/` folder.
8. Generate explainability outputs using Grad-CAM and SHAP.

## Dataset Structure

The notebook creates a balanced split across the four roles.

| Role | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| GK | 173 | 37 | 38 | 248 |
| DEF | 278 | 60 | 60 | 398 |
| MID | 525 | 112 | 113 | 750 |
| FWD | 650 | 139 | 140 | 929 |

The notebook also confirms the final image counts:

- Train: 1626 images
- Validation: 348 images
- Test: 351 images

## Role Mapping

The notebook maps FIFA position groups into the four role classes:

| FIFA Position Group | Role |
|---|---|
| GK | Goalkeeper |
| CB, LB, RB, LCB, RCB, LWB, RWB | Defender |
| CM, CDM, CAM, LM, RM, LCM, RCM, LDM, RDM, LAM, RAM | Midfielder |
| ST, CF, LW, RW, LF, RF, LS, RS | Forward |

## Models Used

### Custom CNN

The custom CNN is a smaller model built from scratch. It uses:

- convolution layers
- batch normalization
- max pooling
- dropout
- global average pooling
- softmax classification output

This model is useful as a baseline because it learns directly from the dataset without pretrained weights.

### MobileNetV2

MobileNetV2 is used as a pretrained feature extractor. It is smaller and faster than ResNet50, and it usually works well when the dataset is not very large.

### ResNet50

ResNet50 is the largest model in the notebook. It has more parameters and stronger representation capacity, but it also takes longer to train and can overfit more easily on smaller datasets.

## Output Files

The notebook writes its outputs into `results/`.

### Figures

- `fig1_dataset_samples.png` shows sample player images from each class.
- `fig2_model_workflow.png` shows the end-to-end project workflow.
- `fig2b_class_distribution.png` shows the class distribution across train, validation, and test splits.
- `fig3_training_curves.png` shows training and validation curves.
- `fig4_confusion_matrices.png` shows normalized confusion matrices for the three models.
- `fig5_gradcam_correct.png` shows Grad-CAM examples for correct predictions.
- `fig6_gradcam_misclassified.png` shows Grad-CAM examples for wrong predictions.
- `fig7_shap_custom_cnn.png` shows SHAP explanations for the custom CNN.
- `fig7_shap_mobilenetv2.png` shows SHAP explanations for MobileNetV2.
- `fig7_shap_resnet50.png` shows SHAP explanations for ResNet50.
- `fig8_final_comparison.png` shows the final comparison chart.

### Tables

- `dataset_summary.csv` stores the final role counts for each split.
- `CustomCNN_history.csv` stores the epoch-by-epoch training history for the custom CNN.
- `MobileNetV2_history.csv` stores the epoch-by-epoch training history for MobileNetV2.
- `ResNet50_history.csv` stores the epoch-by-epoch training history for ResNet50.
- `model_comparison.csv` stores the test metrics for each model.
- `final_comparison_table.csv` stores the summary table used in the final report section.

### Model Files

The `.keras` files are saved checkpoints so that the trained models can be loaded again without retraining.

## Accuracy Table Explanation

The most important result in the notebook is the final comparison table. It reports:

- `Test Accuracy (%)`: the percentage of correct predictions on the test set
- `Precision`: how reliable the model's positive predictions are
- `Recall`: how many true samples the model found
- `F1-Score`: the balance between precision and recall
- `Train Time (min)`: how long training took
- `Parameters`: how large the model is

### Accuracy Results Captured in the Notebook Output

The notebook output shows these final test results:

| Model | Test Accuracy (%) | Precision | Recall | F1-Score | Train Time (min) | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 39.89 | 0.1591 | 0.3989 | 0.2275 | 4.51 | 848,932 |
| MobileNetV2 | 49.00 | 0.5071 | 0.4900 | 0.4815 | 8.29 | 2,586,948 |
| ResNet50 | 40.46 | 0.4012 | 0.4046 | 0.2442 | 10.10 | 24,638,852 |

### Why the Table Looks Like This

- MobileNetV2 performs best in the notebook run because it gives the highest accuracy and strongest F1-score.
- The custom CNN trains faster and is simpler, but it does not separate the four role classes as well as MobileNetV2.
- ResNet50 has the most parameters, but more parameters do not automatically mean better performance when the dataset is relatively small.

The values are close enough to show that this is a challenging four-class problem. The models are learning some useful image patterns, but the portraits do not contain full tactical context.

## Why the Accuracy Is Not Very High

The notebook's accuracy is modest for a few important reasons:

- the input is only a portrait image, not a full match scene
- role information is indirect and visually subtle
- different roles can look similar in portrait-only images
- the dataset is much smaller than typical large-scale vision datasets
- transfer learning helps, but it cannot replace missing context

This is also why the notebook includes confusion matrices, Grad-CAM, and SHAP, not just accuracy.

## Explainability Outputs

The project uses two explainability methods:

- Grad-CAM highlights the areas of the image that most influence the prediction.
- SHAP estimates how image regions contribute to each class decision.

These outputs are useful because they help show whether the model is focusing on meaningful football-related cues or just learning background and texture patterns.

## What the Project Files Mean

### `notebook/`

Contains the main notebook that performs preprocessing, training, evaluation, and explainability.

### `results/figures/`

Contains all charts and visual explanations created by the notebook.

### `results/tables/`

Contains CSV summaries for the dataset split, training histories, and final metrics.

### `results/models/`

Contains the saved trained model checkpoints.

### `role_dataset/`

Contains the organized train, validation, and test image folders created by the notebook.

## Reproducibility Note

If the notebook is rerun, the exact metric values may change slightly because of:

- random weight initialization
- data shuffling
- augmentation randomness
- GPU and training nondeterminism

So the values in the notebook output cells are the most direct record of that run. The CSV tables in `results/tables/` may reflect a later rerun if the notebook was executed again after the displayed outputs were captured.

## How to Run

### In Kaggle

1. Open the notebook in Kaggle.
2. Attach the FIFA 22 player dataset.
3. Run all cells.
4. Check the `results/` folder for generated figures, tables, and model files.

### Locally

```bash
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn pillow requests shap opencv-python-headless
jupyter notebook notebook/football-player-role-classification-cnn-ml.ipynb
```

## References

- FIFA 22 World Cup elite players image dataset
- He et al., Deep Residual Learning for Image Recognition
- Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks
- Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks
- Lundberg and Lee, A Unified Approach to Interpreting Model Predictions

## Author

Yash David Bedekar

Machine Learning Course Project

GitHub: [@yashbedekar-DS](https://github.com/yashbedekar-DS)
