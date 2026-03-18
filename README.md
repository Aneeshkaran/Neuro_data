# NeuroFaces: Facial Analysis for Neurological Disease Classification

A machine learning pipeline for detecting and classifying neurological conditions — **ALS (Amyotrophic Lateral Sclerosis)**, **Post-Stroke (PS)**, and **Healthy Control (HC)** — using facial landmark analysis from video data.

---

## Overview

This project analyzes facial motion from videos in the Neuroface dataset to extract clinically meaningful features and train classifiers that can distinguish between neurological conditions. The pipeline covers the full workflow from raw video exploration to model evaluation.

---

## Repository Structure

```
├── video_dataset_analysis_of_neuroface_data.ipynb   # Dataset exploration & statistics
├── facial_landmark_detection.ipynb                  # Visualizing ground-truth landmarks
├── feature_extraction.ipynb                         # Computing facial motion features
├── Training_ML_model_neurofaces_dataset.ipynb       # Model training & evaluation
└── Training_ML_model_neurofaces_dataset_using__FAN_.ipynb  # Landmark detection using FAN
```

---

## Notebooks

### 1. `video_dataset_analysis_of_neuroface_data.ipynb`
Exploratory analysis of the NeuroFace video dataset.

- Loads video metadata for all three subject groups: **ALS** (11 subjects, 76 videos), **PS** (14 subjects, 105 videos), and **HC** (11 subjects, 80 videos)
- Investigates task coverage across subjects (9 facial tasks per subject: BLOW, BROW, BIGSMILE, KISS, etc.)
- Identifies missing videos and data imbalance (261 expected vs 216 actual videos)
- Visualizes subject and task distributions using Seaborn/Matplotlib

### 2. `facial_landmark_detection.ipynb`
Visualization of 68-point ground-truth facial landmarks on video frames.

- Reads landmark `.txt` files and bounding box annotations
- Overlays landmarks and connecting lines on extracted video frames using OpenCV
- Visualizes individual landmark indices to aid feature design
- Uses Google Colab's `cv2_imshow` for inline display

### 3. `feature_extraction.ipynb`
Computes biomechanical facial features from landmark sequences across all subjects and tasks.

Key features extracted:
| Feature | Description |
|---|---|
| `LLpath` | Total path length of the lower lip (landmark 31 → 58) |
| `Mouth Area` | Convex hull area of inner lip landmarks across frames |
| `Mouth Width (W)` | Euclidean distance between lip corners (landmarks 49, 55) |
| `LLdisp` | Lower lip displacement over time |
| `vW / vLL` | Velocity of mouth width and lip displacement |
| `Eccentricity (e)` | Elliptical shape measure of mouth opening |
| `Symmetry (r_symm)` | Pearson correlation of left/right lip corner displacements |

All features are computed per subject per task and exported to `feature_extraction.csv` for downstream modeling.

### 4. `Training_ML_model_neurofaces_dataset.ipynb`
Trains and evaluates ML classifiers on the extracted features to distinguish ALS, PS, and HC.

- Loads `feature_extraction.csv` and drops PS subjects (for binary classification experiments)
- Trains three models with hyperparameter tuning via `GridSearchCV`:
  - **Logistic Regression**
  - **SVM with Linear Kernel**
  - **SVM with RBF Kernel**
- Uses **Leave-One-Group-Out (LOGO)** cross-validation (grouped by subject)
- Reports accuracy, sensitivity, and specificity per task per model
- Produces grouped bar charts comparing model performance across tasks

### 5. `Training_ML_model_neurofaces_dataset_using__FAN_.ipynb`
Replicates the landmark detection step using the **Face Alignment Network (FAN)** instead of ground-truth annotations.

- Installs and initializes `face-alignment` (FAN) with GPU/CPU support via PyTorch
- Extracts frames from videos and runs FAN to predict 68 facial landmarks
- Saves predicted landmarks for all subjects across ALS, PS, and HC categories
- Evaluates prediction quality by computing **NRMSE%** (Normalized Root Mean Squared Error) against ground-truth landmarks
- Plots NRMSE% threshold curves to summarize detection accuracy across ~3,300 frames

---

## Dataset

This project uses the **NeuroFace dataset**, a clinical video dataset of patients performing standardized facial tasks. Each video filename encodes:
- Subject ID (e.g., `A002`)
- Session number
- Disease category (`ALS`, `PS`, `HC`)
- Task name (e.g., `BLOW`, `KISS`, `NORMAL`, `BIGSMILE`)

Landmark annotations are provided as `.txt` files with 68 (x, y) coordinates per frame. Bounding box ground truths are in `*_gt.txt` files.

> **Note:** The dataset is not included in this repository. Access it via the official NeuroFace project.

---

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
opencv-python
torch
face-alignment
imageio
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy opencv-python torch face-alignment imageio
```

---

## Getting Started

1. Mount your Google Drive (if using Colab) and set the dataset path:
   ```python
   base_dir = "/content/drive/MyDrive/neuro_data/Neuroface"
   ```

2. Run notebooks in order:
   ```
   video_dataset_analysis → facial_landmark_detection → feature_extraction →
   Training_ML_model_neurofaces_dataset (or ...using__FAN_)
   ```

3. After `feature_extraction.ipynb`, the `feature_extraction.csv` output is required as input to the training notebook.

---

## Results

Models are evaluated per facial task using Leave-One-Subject-Out cross-validation. Metrics include per-task accuracy, sensitivity, and specificity, compared across Logistic Regression, Linear SVM, and RBF SVM.

FAN-based landmark detection achieves strong agreement with ground-truth annotations, with the majority of frames falling below 2% NRMSE.

---

## License

Please refer to the NeuroFace dataset's terms of use before using or distributing this work.
