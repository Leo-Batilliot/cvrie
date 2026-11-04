# CVRIE — Brain Tumor Detection & Patient Clustering

> AI medical system in two parts: brain MRI classification (supervised) and patient symptom clustering (unsupervised).

---

## Overview

**CVRIE** is an AI-assisted medical triage system designed to:

1. **Classify brain MRI scans** into 4 tumor categories using image classification (supervised learning).
2. **Group patients by symptom similarity** using NLP clustering on free-form testimonies (unsupervised learning).

The system operates without deep learning frameworks — only `scikit-learn`, `pandas`, `numpy`, and `matplotlib` are used.

---

## Project Structure

```
G-AIA-400-LIL-4-1-cvrie-1/
├── supervised/
│   ├── dataset/              # Brain MRI image dataset (CSV index)
│   ├── model/                # Saved trained models (.pkl)
│   ├── notebook/
│   │   ├── supervised_ai.ipynb   # Main notebook: models + evaluation
│   │   ├── observation.ipynb     # Dataset exploration
│   │   ├── clean_dataset.ipynb   # Preprocessing analysis
│   │   └── analyse.ipynb         # Deep analysis
│   ├── src/
│   │   ├── utils.py          # Image loading + feature extraction
│   │   └── graph.py          # Visualizations
│   ├── requirements.txt
│   └── README.md
│
├── unsupervised/
│   ├── dataset/
│   │   └── student_dataset.csv   # 1011 patient testimonies
│   ├── notebooks/
│   │   └── unsupervised_ai.ipynb # Main notebook: clustering + evaluation
│   ├── src/
│   │   └── clustering.py     # Full NLP clustering pipeline
│   ├── requirements.txt
│   └── README.md
│
└── README.md                 # You are here
```

---

## Prerequisites

**Python 3.10+** is required.

```bash
# Linux / macOS
python3 --version

# Windows
python --version
```

If not installed: https://www.python.org/downloads/

> On Windows, check **"Add Python to PATH"** during installation.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/EpitechPGE2-2025/G-AIA-400-LIL-4-1-cvrie-1.git
cd G-AIA-400-LIL-4-1-cvrie-1
```

### 2. Navigate to a part

```bash
cd supervised    # or
cd unsupervised
```

### 3. Create a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install jupyterlab
```

### 5. Launch JupyterLab

```bash
jupyter lab
```

Then open the main notebook in `notebook/` or `notebooks/`.

---

## Parts

### Supervised — Brain Tumor Classification

Trains machine learning models to classify brain MRI images into 4 categories:
- `glioma`, `meningioma`, `notumor`, `pituitary`

Four models are compared (Random Forest, Logistic Regression, Decision Tree, KNN).
Best initial accuracy: **Random Forest ~93%**. After RandomizedSearchCV tuning: **86.4%** on the test set.

See [supervised/README.md](supervised/README.md) for full details.

### Unsupervised — Patient Symptom Clustering

Groups 1011 patient testimonies into medical condition clusters using:
- TF-IDF vectorization + TruncatedSVD (LSA, 50 components)
- KMeans with Elbow method for K selection
- Silhouette Score evaluation + dominant words per cluster

See [unsupervised/README.md](unsupervised/README.md) for full details.

---

## Constraints

This project intentionally uses only:

| Library | Usage |
|---------|-------|
| `scikit-learn` | ML models, metrics, preprocessing |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | All visualizations |
| `pillow` | Image loading (supervised only) |

No deep learning (TensorFlow, PyTorch), no NLP libraries (NLTK, spaCy).

---

## Authors

- **Gabin Wilson**
- **Léo Batilliot**

## Dataset

The MRI images (`supervised/dataset/Training` and `supervised/dataset/Testing`) are not included in this public copy. Only the CSV index is kept.
