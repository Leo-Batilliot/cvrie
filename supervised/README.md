# Supervised — Brain Tumor Classification

> Classifies brain MRI images into 4 tumor categories using scikit-learn.

---

## Objective

Train a machine learning model that, given a brain MRI image, predicts one of:

| Label | Description |
|-------|-------------|
| `glioma` | Malignant brain tumor |
| `meningioma` | Usually benign meningeal tumor |
| `notumor` | Healthy brain (no tumor) |
| `pituitary` | Pituitary gland tumor |

---

## Dataset

- **Source:** Brain MRI Tumor Classification (Kaggle)
- **Total:** 7023 images
- **Split:** Training (~5712) / Testing (~1311)
- **Index file:** `dataset/dataset_info.csv` — columns: `image_path, split, class_name, class_label, filename`
- **Structure:** `dataset/{Training|Testing}/{class_name}/{filename}.jpg`

Class distribution (balanced):

| Class | Count |
|-------|-------|
| glioma | 1621 |
| meningioma | 1645 |
| notumor | 2000 |
| pituitary | 1757 |

---

## Project Structure

```
supervised/
├── dataset/
│   └── dataset_info.csv      # Image index (path, class, split)
├── model/
│   └── best_model.pkl        # Saved best model (joblib)
├── notebook/
│   ├── supervised_ai.ipynb   # Main notebook: training + evaluation
│   ├── observation.ipynb     # Dataset exploration
│   ├── clean_dataset.ipynb   # Preprocessing analysis
│   └── analyse.ipynb         # Deep analysis
├── src/
│   ├── utils.py              # Image loading + pixel feature extraction
│   └── graph.py              # Visualization utilities
└── requirements.txt
```

---

## Setup

```bash
# From the project root
source venv/bin/activate
jupyter lab
```

Then open `supervised/notebook/supervised_ai.ipynb`.

---

## How It Works

### 1. Feature Extraction (`src/utils.py`)

Images are resized to a configurable `target_size` (default 224×224), converted to RGB,
flattened to a pixel vector, and normalized to [0, 1].

```python
from pathlib import Path
from supervised.src.utils import load_data

SUPERVISED_ROOT = Path(__file__).resolve().parent  # or use the notebook path setup
CSV_PATH = str(SUPERVISED_ROOT / "dataset" / "dataset_info.csv")
DATASET_DIR = str(SUPERVISED_ROOT / "dataset")

# Full quality
X_train, y_train, X_test, y_test = load_data(CSV_PATH, DATASET_DIR)

# Fast mode — 49× fewer features, much faster training
X_train, y_train, X_test, y_test = load_data(CSV_PATH, DATASET_DIR, target_size=(32, 32))
```

### 2. Model Comparison (`notebook/supervised_ai.ipynb`)

Four models are trained and compared side-by-side, with accuracy and training time measured for each:

| Model | Accuracy (224×224) | Notes |
|-------|-------------------|-------|
| **Random Forest** | ~93% | Best model — selected for tuning |
| Logistic Regression | ~90% | Strong linear baseline |
| Decision Tree | ~87% | Non-linear, tends to overfit |
| K-Nearest Neighbors | ~85% | Sensitive to high dimensionality |

Two charts are produced: accuracy comparison and training time per model.

### 3. Hyperparameter Tuning

`RandomizedSearchCV` (cv=5) searches over Random Forest hyperparameters.
`n_iter` is controlled by the `n_iter_search` variable set in the config cell.

### 4. Evaluation

- Confusion matrix (raw counts, Blues colormap)
- Classification report — precision, recall, F1-score per class (grouped bar chart)

---

## Training Time Estimates

Training time scales with image resolution. The config cell at the top of the notebook
lets you switch modes without editing any other cell.

| Mode | Resolution | Features | Data loading | Model comparison | RandomizedSearchCV |
|------|-----------|----------|--------------|-----------------|-------------------|
| **Full** | 224×224 | 150 528 | ~15 min | ~30–60 min | several hours (n_iter=200) |
| **Medium** | 64×64 | 12 288 | ~5 min | ~10–20 min | ~30–60 min (n_iter=50) |
| **Fast** | 32×32 | 3 072 | ~3 min | ~3–5 min | ~10–15 min (n_iter=20) |

> These estimates assume a standard laptop CPU. Results will be faster with more cores
> (the pipeline uses `n_jobs=-1` to parallelize where possible).

---

## Fast Mode

In the notebook, find the **Configuration** cell near the top and set:

```python
FAST_MODE = True
```

This automatically sets:
- `target_size = (32, 32)` — 49× fewer features than 224×224
- `n_iter_search = 20` — fewer hyperparameter combinations to search

Fast mode is useful to verify the full pipeline works end-to-end before committing
to a long training run. Accuracy will be slightly lower (~85% RF vs ~93% at full size)
because 32×32 images lose fine structural detail.

---

## Running

### Run the main notebook

```bash
jupyter lab notebook/supervised_ai.ipynb
```

Execute cells top to bottom. The tuning cell saves the best model to `model/best_model.pkl`.

### Load a saved model

```python
import joblib
from pathlib import Path

model = joblib.load(Path(__file__).resolve().parent / "model" / "best_model.pkl")
prediction = model.predict(X_test)
```

---

## Requirements

```
matplotlib
pillow
numpy
scikit-learn
joblib
```
