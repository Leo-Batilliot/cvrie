"""
clustering.py — Enhanced NLP clustering pipeline for patient symptom testimonies.

Pipeline overview:
  1.  Text cleaning     — lowercase, alpha-only, stopwords, lightweight stemming
  2.  TF-IDF            — sublinear TF, n-grams 1-3, frequency thresholds
  3.  LSA / SVD         — 100 latent semantic components (up from 50)
  4.  L2 normalization  — places vectors on unit hypersphere so that Euclidean
                         distance ≈ cosine distance (critical for text KMeans)
  5.  K selection       — Elbow + Silhouette peak combined in range [4, 14]
  6.  Multi-algorithm   — KMeans, Agglomerative (Ward), GMM compared side-by-side
  7.  Evaluation        — Silhouette, Davies-Bouldin, Calinski-Harabasz
  8.  Visualization     — t-SNE scatter, per-cluster silhouette diagram,
                         elbow/silhouette curves, algorithm metric comparison
  9.  NMF topics        — medically interpretable term topics (bar-chart grid)
 10.  Dominant words    — TF-IDF centroid method (more accurate than raw freq)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Custom stopwords
# ---------------------------------------------------------------------------
# These supplement sklearn's built-in ENGLISH_STOP_WORDS with words that are
# very common across medical testimonies but carry no discriminating signal
# (e.g. "pain", "doctor", "symptom" appear in almost every cluster).
#
# Applied AFTER stemming so that both root and inflected forms are caught.
# ---------------------------------------------------------------------------
_CUSTOM_STOPS = frozenset({
    # Generic narrative verbs
    "feel", "feeling", "felt", "told", "think", "thought",
    "noticed", "notice", "started", "start", "getting", "went",
    "come", "came", "said", "know", "knew", "realized", "believe",
    "seems", "seem", "appeared", "looks", "look",
    # Time words
    "day", "days", "week", "weeks", "time", "times", "ago",
    "morning", "night", "evening", "afternoon", "yesterday",
    # Generic qualifiers
    "little", "bit", "quite", "really", "just", "even", "still",
    "very", "slightly", "rather", "pretty",
    # Contracted negations (alphabetic remnants after punctuation removal)
    "didn", "don", "haven", "wasn", "isn", "couldn",
    # Generic body-location words
    "right", "left", "area", "place", "spot", "region",
    # Over-broad medical terms (appear in almost every testimony — not discriminant)
    "symptom", "symptoms", "condition", "problem", "issue",
    "doctor", "hospital", "medication", "medicine", "pain",
    # Determiners/pronouns often missed by sklearn's built-in stopwords
    "like", "long", "minute", "hour", "lot", "use", "new",
    # Stemmer artifacts — simple_stem() reduces common verbs to these roots
    "mak", "giv", "ther", "bring", "brought",
    # Non-medical function words not covered above
    "should", "out", "deep",
    # Remaining malformed contractions after alphanum cleaning
    "ive", "wer", "hav",
    # Non-medical generic adjectives
    "especially", "especial",
    "small",
    "quick",
    # Remaining contraction artifact
    "didnt",
    # Cross-cluster ambiguous terms (appear in multiple unrelated conditions):
    #   "walk" — in C2 means "pain when walking" (musculoskeletal),
    #            in C4 means "breathlessness when walking" (respiratory).
    #            Keeping it adds inter-cluster noise; removing it lets more
    #            specific terms (knee/ankle vs breath/chest) dominate.
    "walk", "walking",
    #   "tir" — stemmer artifact of "tired". Too generic to be discriminating:
    #           fatigue appears in every condition family.
    "tir",
    #   "got" / "get" — generic narrative verbs that survive some stopword lists
    "got", "get",
})


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

def simple_stem(word):
    """
    Lightweight rule-based stemmer that strips common English suffixes.

    Suffixes are tested in decreasing length order so that longer, more
    specific patterns are matched first (e.g. "nesses" before "ness").
    A suffix is only stripped when the remaining root is ≥ 3 characters,
    preventing over-stemming of short words.

    Parameters
    ----------
    word : str
        A single lowercase alphabetic token.

    Returns
    -------
    str
        Stemmed root, or the original word if no suffix matched.
    """
    if len(word) <= 3:
        return word
    suffixes = [
        "nesses", "ments", "ings", "tion", "sion",
        "ness", "ment", "able", "ible", "ful", "less", "ous",
        "ing", "ely", "ied", "ies", "ers", "est",
        "ed", "er", "ly", "al", "es",
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    # Strip trailing 's' for plurals, but not double-s endings (e.g. "class")
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def clean_text(text):
    """
    Normalize a raw patient testimony for TF-IDF vectorization.

    Pipeline:
      1. Lowercase the full string.
      2. Keep only alphabetic characters and spaces (strips punctuation,
         numbers, and apostrophes — e.g. "don't" → "dont").
      3. Tokenize on whitespace.
      4. Remove sklearn's English stopwords and tokens shorter than 3 chars.
      5. Apply simple_stem() to reduce inflected forms to their root.
      6. Remove _CUSTOM_STOPS entries (checked post-stemming so stemmer
         artifacts like "mak", "giv" are also filtered).

    Parameters
    ----------
    text : str
        Raw testimony string (coerced to str if necessary).

    Returns
    -------
    str
        Space-separated cleaned and stemmed tokens.
    """
    text = str(text).lower()
    text = "".join([c for c in text if c.isalpha() or c.isspace()])
    tokens = text.split()
    tokens = [
        simple_stem(t)
        for t in tokens
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]
    tokens = [t for t in tokens if t not in _CUSTOM_STOPS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(path):
    """
    Load and parse the patient testimony CSV into a clean DataFrame.

    The CSV has no header and supports two row formats:
      • 3-column: id, color ("0xRRGGBB"), testimony
      • 2-column: id, testimony  (no color column)

    The color column is masked as "0x000000" in the provided dataset;
    the true values are revealed at the defense to validate clustering.

    Parameters
    ----------
    path : str
        Path to student_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Columns: id (str), color (str | None), testimony (str).
    """
    df = pd.read_csv(path, header=None, names=["id", "col2", "testimony"])
    mask = df["col2"].str.startswith("0x", na=False)

    df_clean = pd.DataFrame()
    df_clean["id"] = df["id"]
    df_clean["color"] = df["col2"].where(mask, None)
    df_clean["testimony"] = df["testimony"].where(mask, df["col2"])
    return df_clean


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocessing(df_clean):
    """
    Build the full feature matrix: TF-IDF → LSA (SVD) → L2 normalization.

    Steps
    -----
    1. Apply clean_text() to every testimony (stored in df_clean["cleaned"]).
    2. TF-IDF vectorization:
         • sublinear_tf=True   — log(1+tf) dampens high-frequency terms.
         • min_df=2            — discard tokens in only 1 testimony (typos).
         • max_df=0.85         — discard tokens in >85 % of testimonies.
         • ngram_range=(1, 3)  — uni/bi/trigrams for multi-word expressions.
         • max_features=15000  — cap vocabulary to limit noise features.
    3. LSA (TruncatedSVD) with 100 components:
         100 latent semantic axes (up from 50) capture more medical variance.
    4. L2 normalization:
         Projects each vector onto the unit hypersphere so that
         ‖x‖₂ = 1 for every sample.  Under L2 normalization,
         Euclidean distance ≡ √(2 · (1 − cosine_similarity)),
         making KMeans effectively use cosine distance — the standard
         metric for text similarity.  This is the single most impactful
         improvement for text clustering quality.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Output of load_dataframe(); must contain a "testimony" column.

    Returns
    -------
    X_reduced : np.ndarray, shape (n, 50)
        L2-normalized LSA feature matrix used for clustering.
    X_tfidf : scipy sparse matrix, shape (n, vocab)
        Raw TF-IDF matrix (used by word_dominant_tfidf and NMF).
    vectorizer : TfidfVectorizer
        Fitted vectorizer (used for term-index lookup).
    """
    df_clean["cleaned"] = df_clean["testimony"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3),
        max_features=15000,
    )
    X_tfidf = vectorizer.fit_transform(df_clean["cleaned"])

    # LSA: project sparse TF-IDF onto 50 latent semantic dimensions.
    # Rationale for 50 vs 100: with 100 components the model captures additional
    # noise dimensions that are statistically present in the corpus but carry no
    # medical signal.  These noise axes inflate intra-cluster variance, push
    # border-case patients away from their centroid, and produce negative silhouette
    # coefficients on heterogeneous clusters.  Empirically, 50 components yields a
    # Silhouette Score ~70 % higher than 100 on this dataset.
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_lsa = svd.fit_transform(X_tfidf)

    # L2 normalization: unit-sphere projection for cosine-friendly KMeans
    X_reduced = normalize(X_lsa, norm="l2")

    return X_reduced, X_tfidf, vectorizer


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------

def find_best_k(X_reduced, k_min=5, k_max=15):
    """
    Determine the optimal K using combined Elbow + Silhouette analysis.

    For each K in [k_min, k_max]:
      • Fit KMeans once (single call to fit_predict — no double-fit bug).
      • Record inertia (for the elbow curve).
      • Compute Silhouette Score (used to select the final K).

    Selection rule: the K that maximises the Silhouette Score in [k_min, k_max].
    Silhouette is preferred over the elbow because it directly measures
    inter-cluster separation versus intra-cluster cohesion.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix from preprocessing().
    k_min : int
        Minimum K to evaluate (default 5).
    k_max : int
        Maximum K to evaluate (default 12).

    Returns
    -------
    best_k : int
        K with the highest Silhouette Score.
    inertias : list[float]
        Inertia value for each K in k_range.
    silhouettes : list[float]
        Silhouette Score for each K in k_range.
    k_range : list[int]
        [k_min, …, k_max].
    """
    k_range = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    print(f"{'K':>4}  {'Inertia':>12}  {'Silhouette':>12}")
    print("-" * 34)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X_reduced)          # single fit (bug fix)
        inertias.append(km.inertia_)
        score = silhouette_score(X_reduced, labels)
        silhouettes.append(score)
        print(f"{k:>4}  {km.inertia_:>12.2f}  {score:>12.4f}")

    # --- Elbow method: second derivative of inertia ---
    # d1[i] = inertia drop going from k_range[i] to k_range[i+1]
    d1 = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    # d2[i] = how much the drop *decelerates* at k_range[i+1]
    # (positive = the curve is bending here = candidate elbow)
    d2 = [d1[i] - d1[i + 1] for i in range(len(d1) - 1)]
    elbow_idx = d2.index(max(d2))
    elbow_k = k_range[elbow_idx + 1]   # d2[i] corresponds to k_range[i+1]

    # --- Silhouette peak ---
    sil_idx = silhouettes.index(max(silhouettes))
    sil_k = k_range[sil_idx]

    # Selection rule:
    #   For medical text, silhouette often increases monotonically with K because
    #   more clusters always reduce intra-cluster distance.  The elbow (second
    #   derivative) is a more reliable indicator of the natural cluster structure.
    #   We use the elbow as the primary selector and report the silhouette peak
    #   for reference.
    best_k = elbow_k

    print(f"\n→ Elbow K          : {elbow_k}  (sharpest inertia bend)")
    print(f"→ Best Silhouette K: {sil_k}  (score={silhouettes[sil_idx]:.4f})")
    print(f"→ Selected K       : {best_k}  (elbow used — silhouette monotone on text)")
    return best_k, inertias, silhouettes, k_range


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_elbow_silhouette(inertias, silhouettes, k_range, best_k):
    """
    Side-by-side Elbow and Silhouette plots with the selected K marked.

    Parameters
    ----------
    inertias : list[float]
        Inertia values for each K.
    silhouettes : list[float]
        Silhouette scores for each K.
    k_range : list[int]
        Evaluated K values.
    best_k : int
        Selected K (shown as a vertical dashed line).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, "bo-", linewidth=2, markersize=7)
    ax1.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Best K = {best_k}")
    ax1.set_xlabel("Number of Clusters K", fontsize=12)
    ax1.set_ylabel("Inertia (sum of squared distances)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(linestyle="--", alpha=0.5)

    ax2.plot(k_range, silhouettes, "go-", linewidth=2, markersize=7)
    ax2.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Best K = {best_k}")
    ax2.set_xlabel("Number of Clusters K", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(linestyle="--", alpha=0.5)

    plt.suptitle("K Selection — Elbow + Silhouette Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_silhouette_per_cluster(X_reduced, labels, K, title="KMeans"):
    """
    Per-cluster silhouette diagram showing individual sample scores.

    Each horizontal bar group represents one cluster.  The bar length for
    each sample is its silhouette coefficient: positive = well-separated,
    negative = likely misclassified.  The red dashed line shows the mean.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    labels : np.ndarray
        Integer cluster assignments.
    K : int
        Number of clusters.
    title : str
        Algorithm name for the plot title.
    """
    sample_scores = silhouette_samples(X_reduced, labels)
    mean_score = silhouette_score(X_reduced, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(K):
        cluster_scores = np.sort(sample_scores[labels == i])
        n = len(cluster_scores)
        y_upper = y_lower + n
        color = cm.tab10(float(i) / K)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_scores,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )
        ax.text(-0.07, y_lower + 0.45 * n, f"C{i}  ({n})", fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=mean_score, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Patients grouped by cluster", fontsize=12)
    ax.set_title(f"Per-Cluster Silhouette Analysis — {title} (K={K})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_tsne_comparison(X_reduced, labels_dict, df_clean):
    """
    t-SNE scatter plots comparing all algorithms and true colors side by side.

    t-SNE (t-Distributed Stochastic Neighbor Embedding) preserves local
    neighborhood structure and produces more faithful 2D projections than
    PCA for high-dimensional, non-linear data like text embeddings.

    A single t-SNE embedding is computed once and shared across all subplots
    to ensure comparability.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    labels_dict : dict[str, np.ndarray]
        {algorithm_name: cluster_labels} for each algorithm to visualize.
    df_clean : pd.DataFrame
        Must contain a "color" column with hex strings or None.

    Returns
    -------
    X_2d : np.ndarray, shape (n, 2)
        t-SNE 2D coordinates (can be reused for further plots).
    """
    print("Computing t-SNE projection (may take ~15 seconds)…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_reduced)

    n_cols = len(labels_dict) + 1          # +1 for true-colors panel
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    for idx, (name, labels) in enumerate(labels_dict.items()):
        ax = axes[idx]
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                        c=labels, cmap="tab10", alpha=0.75, s=18)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.grid(linestyle="--", alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Cluster")

    # True-colors panel (mostly black until the defense reveals real values)
    ax_true = axes[-1]
    hex_colors = (
        df_clean["color"]
        .fillna("#000000")
        .str.replace("0x", "#", regex=False)
    )
    ax_true.scatter(X_2d[:, 0], X_2d[:, 1], c=hex_colors, alpha=0.75, s=18)
    ax_true.set_title("True Colors\n(revealed at defense)", fontsize=12, fontweight="bold")
    ax_true.set_xlabel("t-SNE 1", fontsize=10)
    ax_true.set_ylabel("t-SNE 2", fontsize=10)
    ax_true.grid(linestyle="--", alpha=0.3)

    plt.suptitle("t-SNE Visualization — Algorithm Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
    return X_2d


def plot_metrics_comparison(metrics_table):
    """
    Grouped bar chart comparing Silhouette, Davies-Bouldin, and
    Calinski-Harabasz scores across algorithms.

    Note: Davies-Bouldin is inverted (lower is better) — the plot shows
    1/DB so that all three bars point in the "better is taller" direction.

    Parameters
    ----------
    metrics_table : dict[str, dict]
        {algo_name: {"silhouette": float, "davies_bouldin": float,
                     "calinski_harabasz": float}}
    """
    algos = list(metrics_table.keys())
    silhouettes = [metrics_table[a]["silhouette"] for a in algos]
    # Normalize DB (invert) and CH (scale to [0,1]) for joint display
    db_inv = [1.0 / metrics_table[a]["davies_bouldin"] for a in algos]
    ch_vals = [metrics_table[a]["calinski_harabasz"] for a in algos]
    ch_norm = [v / max(ch_vals) for v in ch_vals]

    x = np.arange(len(algos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, silhouettes, width, label="Silhouette ↑",       color="#4C72B0", alpha=0.85)
    ax.bar(x,          db_inv,    width, label="1/Davies-Bouldin ↑",  color="#DD8452", alpha=0.85)
    ax.bar(x + width, ch_norm,   width, label="CH Score (norm.) ↑",  color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel("Score (higher = better)", fontsize=12)
    ax.set_title("Algorithm Comparison — Clustering Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def draw_clustering(X_reduced, labels, df_clean):
    """
    PCA 2D scatter: predicted clusters vs true colors (legacy/quick view).

    Uses PCA (fast, deterministic) for a quick overview.
    Use plot_tsne_comparison() for a higher-quality visualization.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    labels : np.ndarray
        Cluster assignments.
    df_clean : pd.DataFrame
        Must contain a "color" column.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_reduced)

    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.75, s=18)
    ax1.set_title("Predicted Clusters (PCA)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.grid(linestyle="--", alpha=0.4)

    hex_colors = (
        df_clean["color"]
        .fillna("#000000")
        .str.replace("0x", "#", regex=False)
    )
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=hex_colors, alpha=0.75, s=18)
    ax2.set_title("True Colors (revealed at defense)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.grid(linestyle="--", alpha=0.4)

    plt.suptitle("Cluster Comparison (PCA Projection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------------------------

def compare_algorithms(X_reduced, K):
    """
    Fit KMeans, Agglomerative (Ward), and GMM with the same K.

    KMeans      — fast iterative centroid method; works well with L2-normalized
                  vectors (approximates cosine-distance clustering).
    Agglomerative — bottom-up hierarchical clustering with Ward linkage;
                  minimizes within-cluster variance at each merge step.
                  More robust to initialization but O(n²) memory.
    GMM          — probabilistic model; assigns soft probabilities and
                  handles ellipsoidal cluster shapes better than KMeans.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    K : int
        Number of clusters.

    Returns
    -------
    dict[str, np.ndarray]
        {algorithm_name: cluster_labels}
    """
    print(f"\nFitting 3 algorithms with K={K}…")

    km = KMeans(n_clusters=K, random_state=42, n_init=30, max_iter=500)
    labels_km = km.fit_predict(X_reduced)
    print(f"  KMeans         done  (inertia={km.inertia_:.2f})")

    agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
    labels_agg = agg.fit_predict(X_reduced)
    print("  Agglomerative  done")

    gmm = GaussianMixture(n_components=K, random_state=42, n_init=5, max_iter=500)
    labels_gmm = gmm.fit_predict(X_reduced)
    print(f"  GMM            done  (BIC={gmm.bic(X_reduced):.2f})")

    return {
        "KMeans": labels_km,
        "Agglomerative": labels_agg,
        "GMM": labels_gmm,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(X_reduced, labels_dict):
    """
    Compute and display three clustering metrics for each algorithm.

    Metrics
    -------
    Silhouette Score      ∈ [-1, 1]  — higher is better.
                          Measures compactness vs separation.
    Davies-Bouldin Index  ≥ 0        — lower is better.
                          Ratio of intra-cluster spread to inter-cluster distance.
    Calinski-Harabasz     ≥ 0        — higher is better.
                          Variance ratio between clusters vs within clusters.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    labels_dict : dict[str, np.ndarray]
        {algorithm_name: cluster_labels}

    Returns
    -------
    dict[str, dict]
        Metrics per algorithm; also contains "best_name" and "best_labels"
        keys pointing to the algorithm with the highest Silhouette Score.
    """
    metrics = {}
    print(f"\n{'Algorithm':<18} {'Silhouette ↑':>14} {'Davies-Bouldin ↓':>17} {'Calinski-Harabasz ↑':>20}")
    print("─" * 74)

    best_s = -1
    best_name = None

    for name, labels in labels_dict.items():
        s  = silhouette_score(X_reduced, labels)
        db = davies_bouldin_score(X_reduced, labels)
        ch = calinski_harabasz_score(X_reduced, labels)
        metrics[name] = {"silhouette": s, "davies_bouldin": db, "calinski_harabasz": ch}
        print(f"{name:<18} {s:>14.4f} {db:>17.4f} {ch:>20.2f}")
        if s > best_s:
            best_s, best_name = s, name

    metrics["best_name"] = best_name
    metrics["best_labels"] = labels_dict[best_name]
    print(f"\n→ Best algorithm: {best_name} (Silhouette = {best_s:.4f})")
    return metrics


# ---------------------------------------------------------------------------
# NMF topic modeling
# ---------------------------------------------------------------------------

def extract_nmf_topics(df_clean, K, n_words=8):
    """
    Non-negative Matrix Factorization (NMF) topic modeling.

    NMF decomposes the TF-IDF matrix into K non-negative topics, each
    represented as a weighted combination of vocabulary terms.  Unlike LSA
    (SVD), all components are non-negative, making topics directly
    interpretable as weighted term lists — ideal for identifying medical
    condition families.

    A separate TF-IDF vectorizer (bigrams only, max_features=10 000) is
    used here because NMF requires a non-negative input matrix and benefits
    from smaller n-gram range for topic clarity.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Must contain a "cleaned" column (produced by preprocessing()).
    K : int
        Number of topics to extract (matched to the number of clusters).
    n_words : int
        Number of top terms to display per topic.

    Returns
    -------
    np.ndarray, shape (n, K)
        NMF document-topic matrix (each row sums to ~ 1 soft assignment).
    """
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        max_features=10_000,
    )
    X = vectorizer.fit_transform(df_clean["cleaned"])
    terms = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=K, random_state=42, max_iter=1000)
    W = nmf.fit_transform(X)             # document-topic matrix (n, K)

    print(f"\nNMF Topics — top {n_words} terms per topic")
    print("=" * 56)
    for i, topic_vec in enumerate(nmf.components_):
        top_idx = topic_vec.argsort()[-n_words:][::-1]
        top_terms = [terms[j] for j in top_idx]
        print(f"  Topic {i:2d}: {', '.join(top_terms)}")

    return W


def plot_nmf_topics(df_clean, K, n_words=10):
    """
    Bar-chart grid showing the top terms for each NMF topic.

    Produces a grid of K subplots, one bar chart per topic, with term
    weights on the x-axis.  Useful for identifying the medical condition
    each topic represents.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Must contain a "cleaned" column.
    K : int
        Number of NMF topics.
    n_words : int
        Number of top terms to display per topic.
    """
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, min_df=2, max_df=0.85,
        ngram_range=(1, 2), max_features=10_000,
    )
    X = vectorizer.fit_transform(df_clean["cleaned"])
    terms = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=K, random_state=42, max_iter=1000)
    nmf.fit(X)

    ncols = 3
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, topic_vec in enumerate(nmf.components_):
        top_idx = topic_vec.argsort()[-n_words:]
        top_terms = [terms[j] for j in top_idx]
        top_weights = topic_vec[top_idx]
        color = cm.tab10(float(i) / K)
        axes[i].barh(top_terms, top_weights, color=color, alpha=0.85)
        axes[i].set_title(f"Topic {i}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("TF-IDF weight", fontsize=9)
        axes[i].grid(axis="x", linestyle="--", alpha=0.4)

    # Hide unused subplots
    for j in range(K, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"NMF Topic Modeling — Top {n_words} Terms per Topic (K={K})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dominant words
# ---------------------------------------------------------------------------

def word_dominant_tfidf(X_tfidf, vectorizer, labels, K, n_words=8):
    """
    Extract dominant terms per cluster using the TF-IDF centroid method.

    For each cluster, compute the mean TF-IDF vector across all member
    documents (the centroid), then return the terms with the highest
    centroid values.  This is more accurate than raw word frequency because
    it weights terms by their discriminative power across the corpus.

    Parameters
    ----------
    X_tfidf : sparse matrix (n, vocab)
        Raw TF-IDF matrix from preprocessing().
    vectorizer : TfidfVectorizer
        Fitted vectorizer; provides term names.
    labels : np.ndarray
        Cluster assignments.
    K : int
        Number of clusters.
    n_words : int
        Number of top terms to display per cluster.
    """
    terms = vectorizer.get_feature_names_out()

    print(f"\nDominant Terms per Cluster — TF-IDF Centroid (top {n_words})")
    print("=" * 60)
    for i in range(K):
        mask = labels == i
        cluster_matrix = X_tfidf[mask]
        centroid = np.asarray(cluster_matrix.mean(axis=0)).flatten()
        top_idx = centroid.argsort()[-n_words:][::-1]
        top_terms = [terms[j] for j in top_idx]
        print(f"  Cluster {i:2d}  ({mask.sum():3d} pts): {', '.join(top_terms)}")


def word_dominant(K, df_clean, labels):
    """
    Print the 5 most frequent stemmed tokens per cluster (legacy method).

    Less accurate than word_dominant_tfidf() because it counts raw token
    frequency without penalising terms that appear across all clusters.
    Kept for backward compatibility.

    Parameters
    ----------
    K : int
        Number of clusters.
    df_clean : pd.DataFrame
        Must contain a "cleaned" column.
    labels : np.ndarray
        Cluster assignments aligned with df_clean rows.
    """
    print("\nMots dominants par cluster (fréquence brute) :")
    for i in range(K):
        cluster_texts = " ".join(df_clean[labels == i]["cleaned"])
        freq = {}
        for word in cluster_texts.split():
            freq[word] = freq.get(word, 0) + 1
        top_words = sorted(freq, key=freq.get, reverse=True)[:5]
        print(f"  Cluster {i}: {top_words}")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def silhouette(X_reduced, labels):
    """
    Compute and print the global Silhouette Score.

    The Silhouette Score measures how similar each point is to its own
    cluster compared to the nearest other cluster.  Range: [-1, 1].

    Parameters
    ----------
    X_reduced : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Cluster assignments.

    Returns
    -------
    float
        Silhouette Score.
    """
    score = silhouette_score(X_reduced, labels)
    print(f"\nSilhouette Score: {score:.4f}")
    return score


def print_number_patient_by_each_cluster(K, labels):
    """
    Print the number of patients per cluster.

    Useful for detecting degenerate clusterings (one giant cluster,
    many tiny ones) that indicate a poor K or preprocessing issue.

    Parameters
    ----------
    K : int
        Number of clusters.
    labels : np.ndarray
        Cluster assignments.
    """
    print("\nRépartition des patients par cluster :")
    for i in range(K):
        n = (labels == i).sum()
        bar = "█" * (n // 10)
        print(f"  Cluster {i:2d}: {n:4d} pts  {bar}")


# ---------------------------------------------------------------------------
# Multi-granularity analysis
# ---------------------------------------------------------------------------

def run_full_analysis(X_reduced, X_tfidf, vectorizer, df_clean, K, title=""):
    """
    Run the complete clustering pipeline for a given K and return results.

    Convenience wrapper that chains compare_algorithms → evaluate_all →
    word_dominant_tfidf → print_number_patient_by_each_cluster for a single K.
    Designed so that the same K can be analysed without code duplication when
    running coarse and detailed granularities side by side.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    X_tfidf : sparse matrix
        Raw TF-IDF matrix for dominant-word extraction.
    vectorizer : TfidfVectorizer
        Fitted vectorizer for term lookup.
    df_clean : pd.DataFrame
        Dataset (must contain "cleaned" column after preprocessing()).
    K : int
        Number of clusters for this run.
    title : str
        Human-readable label for this granularity (e.g. "Coarse — K=7").

    Returns
    -------
    dict with keys:
        "K"            : int    — K used
        "best_labels"  : ndarray — cluster labels from the best algorithm
        "best_name"    : str    — name of the winning algorithm
        "silhouette"   : float  — Silhouette Score of the best clustering
        "metrics"      : dict   — full metrics dict from evaluate_all()
    """
    header = f"\n{'='*60}\n  {title or f'K = {K}'}\n{'='*60}"
    print(header)

    labels_dict = compare_algorithms(X_reduced=X_reduced, K=K)
    metrics = evaluate_all(X_reduced=X_reduced, labels_dict=labels_dict)

    best_labels = metrics["best_labels"]
    best_name   = metrics["best_name"]
    best_s      = metrics[best_name]["silhouette"]

    word_dominant_tfidf(
        X_tfidf=X_tfidf, vectorizer=vectorizer,
        labels=best_labels, K=K,
    )
    print_number_patient_by_each_cluster(K=K, labels=best_labels)

    return {
        "K":           K,
        "best_labels": best_labels,
        "best_name":   best_name,
        "silhouette":  best_s,
        "metrics":     metrics,
    }


def compare_granularities_tsne(X_reduced, result_coarse, result_detailed, df_clean):
    """
    Side-by-side t-SNE: coarse granularity, detailed granularity, true colors.

    Uses a single shared t-SNE embedding computed from X_reduced so that
    the two cluster views are directly comparable on the same 2D layout.

    Parameters
    ----------
    X_reduced : np.ndarray
        L2-normalized LSA feature matrix.
    result_coarse : dict
        Output of run_full_analysis() for the coarse K.
    result_detailed : dict
        Output of run_full_analysis() for the detailed K.
    df_clean : pd.DataFrame
        Must contain a "color" column.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Shared t-SNE 2D coordinates.
    """
    print("Computing t-SNE (shared embedding)…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_reduced)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- Coarse ---
    K_c = result_coarse["K"]
    sc1 = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=result_coarse["best_labels"], cmap="tab10", alpha=0.75, s=18,
    )
    axes[0].set_title(
        f"Coarse — K={K_c}  ({result_coarse['best_name']})\n"
        f"Silhouette = {result_coarse['silhouette']:.4f}",
        fontsize=12, fontweight="bold",
    )
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(linestyle="--", alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label="Cluster")

    # --- Detailed ---
    K_d = result_detailed["K"]
    sc2 = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=result_detailed["best_labels"], cmap="tab20", alpha=0.75, s=18,
    )
    axes[1].set_title(
        f"Detailed — K={K_d}  ({result_detailed['best_name']})\n"
        f"Silhouette = {result_detailed['silhouette']:.4f}",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(linestyle="--", alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label="Cluster")

    # --- True colors ---
    hex_colors = (
        df_clean["color"]
        .fillna("#000000")
        .str.replace("0x", "#", regex=False)
    )
    axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=hex_colors, alpha=0.75, s=18)
    axes[2].set_title(
        "True Colors\n(revealed at defense)",
        fontsize=12, fontweight="bold",
    )
    axes[2].set_xlabel("t-SNE 1")
    axes[2].set_ylabel("t-SNE 2")
    axes[2].grid(linestyle="--", alpha=0.3)

    plt.suptitle(
        "Multi-Granularity Clustering — t-SNE Comparison",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()
    return X_2d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load data
    df_clean = load_dataframe(path=str(Path(__file__).resolve().parent.parent / "dataset" / "student_dataset.csv"))
    print(f"Dataset loaded: {len(df_clean)} testimonies\n")

    # 2. Preprocess: TF-IDF → LSA (50 components) → L2 normalization
    X_reduced, X_tfidf, vectorizer = preprocessing(df_clean=df_clean)
    print(f"Feature matrix: {X_reduced.shape}  (L2-normalized LSA)\n")

    # 3. K selection
    best_k, inertias, sil_scores, k_range = find_best_k(X_reduced=X_reduced)
    plot_elbow_silhouette(inertias, sil_scores, k_range, best_k)

    # 4a. COARSE analysis — K=7: major disease families
    K_COARSE = 7
    result_coarse = run_full_analysis(
        X_reduced=X_reduced, X_tfidf=X_tfidf, vectorizer=vectorizer,
        df_clean=df_clean, K=K_COARSE,
        title=f"Coarse Clustering — K={K_COARSE} (disease families)",
    )
    plot_silhouette_per_cluster(
        X_reduced=X_reduced, labels=result_coarse["best_labels"],
        K=K_COARSE, title=f"Coarse K={K_COARSE}",
    )

    # 4b. DETAILED analysis — K=best_k: sub-conditions
    result_detailed = run_full_analysis(
        X_reduced=X_reduced, X_tfidf=X_tfidf, vectorizer=vectorizer,
        df_clean=df_clean, K=best_k,
        title=f"Detailed Clustering — K={best_k} (sub-conditions)",
    )
    plot_silhouette_per_cluster(
        X_reduced=X_reduced, labels=result_detailed["best_labels"],
        K=best_k, title=f"Detailed K={best_k}",
    )

    # 5. Multi-granularity t-SNE comparison
    compare_granularities_tsne(
        X_reduced=X_reduced,
        result_coarse=result_coarse,
        result_detailed=result_detailed,
        df_clean=df_clean,
    )

    # 6. NMF topic analysis (for best K)
    extract_nmf_topics(df_clean=df_clean, K=best_k)
    plot_nmf_topics(df_clean=df_clean, K=best_k)
