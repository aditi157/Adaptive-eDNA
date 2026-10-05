"""
B0 baseline clustering: fixed K=10 K-Means on autoencoder embeddings.

Terminology note: K-Means is FITTED on embeddings (unsupervised), not
"trained" in the neural-network sense. If ground-truth labels are
available (e.g. a constructed labelled benchmark), they are never passed
to K-Means -- they are only used afterwards, for external evaluation.

  Internal evaluation (no labels needed):
    - Silhouette score        (higher is better)
    - Davies-Bouldin index    (lower is better)

  External evaluation (only if labels are available, used strictly for
  evaluation -- never for fitting or for choosing K):
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
"""

from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from scripts.config import BASELINE_K, RANDOM_SEED


def fit_kmeans(embeddings: np.ndarray, k: int = BASELINE_K, seed: int = RANDOM_SEED):
    """Fit K-Means with a fixed K on embeddings. Returns (labels, model)."""
    n_clusters = min(k, len(embeddings))
    if n_clusters < 1:
        raise ValueError("Cannot cluster zero embeddings.")
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels, model


def internal_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Silhouette and Davies-Bouldin require at least 2 clusters and fewer
    clusters than samples; return None instead of raising if that isn't
    satisfied (e.g. a tiny toy run).
    """
    n_clusters = len(set(labels))
    n_samples = len(labels)
    if n_clusters < 2 or n_clusters >= n_samples:
        return {"silhouette": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(embeddings, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
    }


def external_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """
    Compare predicted clusters against hidden ground-truth labels.
    Only meaningful when a labelled benchmark dataset is used -- real,
    BLAST-unassigned environmental reads typically have no ground truth.
    """
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
    }


def cluster_size_table(labels: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}
