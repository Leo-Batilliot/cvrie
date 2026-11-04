# Unsupervised — Patient Symptom Clustering

> Groups 1011 patient testimonies into medical condition clusters using NLP and unsupervised learning.

---

## Objective

Automatically group patients by symptom similarity from free-form text testimonies — with no labels. CVRIE uses this to route incoming patients to the right care pathway before a doctor intervenes.

The dataset contains a masked `color` column (`0x000000`) that will be revealed during the defense to validate cluster quality against ground truth.

---

## Dataset

- **File:** `dataset/student_dataset.csv`
- **Size:** 1011 patient testimonies
- **Language:** English
- **Columns:** `id`, `color` (masked hex label or empty), `testimony`

Examples of conditions present in the data:
- Thermal burns, ORL infections, respiratory issues
- Musculoskeletal injuries (upper and lower limb), dermatology / skin allergies
- Neurological symptoms, metabolic disorders (diabetes, hypertension)
- Ear infections, chest tightness, dyspnea / anemia

---

## Project Structure

```
unsupervised/
├── dataset/
│   └── student_dataset.csv       # Raw patient testimonies
├── notebooks/
│   └── unsupervised_ai.ipynb     # Main notebook: full pipeline + results
├── src/
│   └── clustering.py             # Complete NLP clustering pipeline
└── requirements.txt
```

---

## Setup

```bash
# From the project root
source venv/bin/activate
jupyter lab
```

Then open `notebooks/unsupervised_ai.ipynb`.

---

## Pipeline Overview

```
Raw testimony
     │
     ▼
clean_text()              → lowercase, remove punctuation, stopwords,
                            custom medical noise words, lightweight stemming
     │
     ▼
TF-IDF Vectorizer         → sublinear_tf, ngrams (1–3), min_df=2, max_df=0.85,
                            max_features=15 000
     │
     ▼
TruncatedSVD (LSA)        → 50 latent semantic components
     │
     ▼
L2 Normalization          → unit-sphere projection (KMeans ≈ cosine distance)
     │
     ▼
find_best_k()             → Elbow 2nd-derivative + Silhouette, K ∈ [5, 12]
     │
     ▼
run_full_analysis()       → KMeans + Agglomerative + GMM comparison,
     │                       3 evaluation metrics, dominant words
     │
     ├── COARSE  K=7      → 7 disease families (department routing)
     └── DETAILED K=auto  → 10 sub-conditions (fine-grained triage)
     │
     ▼
compare_granularities_tsne() → shared t-SNE: coarse | detailed | true colors
     │
     ▼
plot_nmf_topics()         → NMF topic bar charts (disease-level interpretation)
```

---

## How It Works

### 1. Text Preprocessing

`clean_text()` in `src/clustering.py`:
1. Lowercases and removes non-alphabetic characters
2. Removes sklearn's English stopwords + tokens shorter than 3 characters
3. Applies `simple_stem()` (lightweight suffix-stripping stemmer)
4. Removes `_CUSTOM_STOPS`: generic narrative verbs, time words, over-broad medical terms,
   cross-cluster ambiguous terms (`walk`, `tir`…), and stemmer artifacts

### 2. Vectorization, Dimensionality Reduction & Normalization

`preprocessing()` returns a tuple `(X_reduced, X_tfidf, vectorizer)`:

| Step | Details |
|------|---------|
| TF-IDF | sublinear_tf, min_df=2, max_df=0.85, ngram_range=(1,3), max_features=15 000 |
| TruncatedSVD | 50 components — projects sparse TF-IDF onto latent semantic axes |
| **L2 normalization** | Unit-sphere projection → Euclidean distance ≈ cosine similarity |

> **Why 50 components, not 100?**
> 100 components capture noise axes that inflate intra-cluster variance and produce
> negative silhouette coefficients. 50 gives ~70 % higher Silhouette Score on this dataset.

> **Why L2 normalization?**
> TF-IDF vectors have very different norms. After normalization, KMeans effectively
> uses cosine distance — the gold standard for text clustering.

### 3. K Selection

`find_best_k()` evaluates K from 5 to 12:
- **Elbow method** (second derivative of inertia) — primary selector; more reliable
  than Silhouette for medical text where the Silhouette curve is monotonically increasing.
- **Silhouette Score** — reported at each K as a quality benchmark.

### 4. Multi-Granularity Clustering

`run_full_analysis()` wraps the full pipeline for a given K.
Two granularities are run and compared:

| Granularity | K | Silhouette | Use case |
|---|---|---|---|
| **Coarse** | 7 | ~0.079 | Route patient to the right hospital department |
| **Detailed** | 10 (auto) | ~0.097 | Identify the likely sub-condition |

Three algorithms are compared at each K:

| Algorithm | Mechanism |
|---|---|
| **KMeans** | Iterative centroid update — best with L2-normalized vectors |
| **Agglomerative (Ward)** | Bottom-up hierarchical merging — no initialization randomness |
| **GMM** | Probabilistic mixture of Gaussians — handles ellipsoidal clusters |

### 5. Evaluation

Three complementary metrics are reported for each algorithm:

| Metric | Range | Optimum |
|---|---|---|
| **Silhouette Score** | [-1, 1] | → +1 |
| **Davies-Bouldin Index** | ≥ 0 | → 0 |
| **Calinski-Harabasz** | ≥ 0 | → ∞ |

### 6. Visualization

| Function | Output |
|---|---|
| `plot_elbow_silhouette()` | Side-by-side Elbow + Silhouette curves with best K marked |
| `compare_granularities_tsne()` | t-SNE: coarse \| detailed \| true colors (shared 2D layout) |
| `plot_silhouette_per_cluster()` | Per-sample silhouette bars grouped by cluster |
| `plot_metrics_comparison()` | Grouped bar chart — all algorithms × all metrics |
| `plot_nmf_topics()` | Bar-chart grid — top terms per NMF topic |

### 7. Medical Interpretation

`word_dominant_tfidf()` extracts dominant terms via the **TF-IDF centroid method**
(mean TF-IDF vector per cluster → top terms by centroid weight), which is more
accurate than raw frequency counting.

`extract_nmf_topics()` / `plot_nmf_topics()` provide a complementary view using
**Non-negative Matrix Factorization**: each topic is a weighted sum of vocabulary
terms, directly readable as a medical condition fingerprint.

---

## Identified Condition Clusters

### Coarse (K=7) — disease families

| Cluster | Condition family |
|---|---|
| C0 | Skin allergies / Hives |
| C1 | Musculoskeletal (general) |
| C2 | Respiratory |
| C3 | Upper respiratory infections (ORL) |
| C4 | Ear infections |
| C5 | Thermal burns |
| C6 | Neurological / Vascular / Metabolic |

### Detailed (K=10) — sub-conditions

| Cluster | Sub-condition |
|---|---|
| C0 | Migraines / Visual headaches |
| C1 | Upper limb injuries (arm, elbow, wrist) |
| C2 | Skin allergies / Urticaria |
| C3 | Ear infections |
| C4 | Thermal burns |
| C5 | Chest tightness / Cardiac-like |
| C6 | ORL infections (throat, nose) |
| C7 | Dyspnea / Anemia / Dizziness |
| C8 | Metabolic disorders (diabetes, hypertension) |
| C9 | Lower limb injuries (knee, ankle, foot) |

---

## Running

### Notebook (recommended)

```bash
jupyter lab notebooks/unsupervised_ai.ipynb
```

Execute all cells top to bottom. All metrics, plots, and cluster keywords are displayed inline.

### Script

```bash
cd unsupervised
python src/clustering.py
```

---

## Evaluation Metrics — Typical Ranges

| Metric | Typical text range | This project |
|---|---|---|
| Silhouette | 0.05 – 0.15 | ~0.08 (K=7) / ~0.10 (K=10) |
| Davies-Bouldin | 2 – 5 | ~3.2 / ~2.9 |

> Low Silhouette scores on medical text are expected — patient testimonies share vocabulary
> across conditions. The medical interpretability of clusters (dominant words + NMF topics)
> matters more than the raw score.
>
> Some border-case patients always have negative silhouette: "burning itch" (allergy)
> and "burning sensation" (burn) are identical in token space. This is a fundamental
> bag-of-words limitation — a semantic model (e.g. BERT) would resolve it.

---

## Requirements

```
pandas
scikit-learn
numpy
matplotlib
```
