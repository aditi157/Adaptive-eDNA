"""
AESCTA adaptive-K stage.

Instead of forcing the 32-D embeddings into a fixed K=10, fit K-Means for
every candidate K (default 2..10), score each one, and select K* = the K
with the highest weighted score.

Per-K metrics (no labels needed):
    - Silhouette score       (higher is better)
    - Davies-Bouldin index   (lower  is better)
    - Shannon diversity      (reported for each K; NOT part of the weighted score)

Weighted score (higher is better), normalised ACROSS the K values tested:
    silhouette_norm = (S - S_min) / (S_max - S_min)
    dbi_norm        = (DBI_max - DBI) / (DBI_max - DBI_min)     <- flipped: lower DBI scores higher
    weighted_score  = W_SILHOUETTE * silhouette_norm + W_DAVIES_BOULDIN * dbi_norm

Because of the min-max scaling, the score says "which K is best relative to
the other Ks for THIS dataset"; it is not comparable between runs/datasets.
Ties are broken in favour of the smaller K.

Shannon diversity is shown per K but deliberately excluded from the score:
it tends to rise with K simply because more clusters means more categories,
so using it would bias selection towards large K.
"""

from math import log
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from scripts.config import RANDOM_SEED

K_MIN = 2
K_MAX = 10
W_SILHOUETTE = 0.5
W_DAVIES_BOULDIN = 0.5


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. If all values are equal, return 0.5 for each."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def shannon_from_labels(labels: np.ndarray) -> float:
    """Shannon index H' = -sum(p_i * ln p_i) over cluster relative abundances."""
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-sum(x * log(x) for x in p if x > 0))


def adaptive_k_search(
    embeddings: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    w_silhouette: float = W_SILHOUETTE,
    w_dbi: float = W_DAVIES_BOULDIN,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, int, Dict[int, np.ndarray]]:
    """
    Fit K-Means for every K in [k_min, k_max] and select K*.

    Returns
    -------
    metrics_df : one row per K: silhouette, davies_bouldin, silhouette_norm,
                 dbi_norm, weighted_score, shannon_index, selected.
    best_k     : the selected K*.
    labels_by_k: {K: cluster labels}, so K* labels are reused without
                 refitting. Same seed/n_init as the baseline K-Means, so the
                 K=10 row matches the baseline's K=10 metrics.
    """
    n = len(embeddings)
    k_max_eff = min(k_max, n - 1)  # silhouette/DBI need K <= n-1
    if k_max_eff < k_min:
        raise ValueError(
            f"Need at least {k_min + 1} sequences for adaptive K search (got {n})."
        )

    rows, labels_by_k = [], {}
    for k in range(k_min, k_max_eff + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        # Duplicate points can make K-Means return fewer clusters than asked;
        # silhouette/DBI are undefined for <2 clusters, so skip such K.
        if len(set(labels)) < 2:
            continue
        labels_by_k[k] = labels
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(embeddings, labels)),
            "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
            "shannon_index": shannon_from_labels(labels),
            "evenness": shannon_from_labels(labels) / np.log(len(set(labels))),
        })
    if not rows:
        raise ValueError("No valid K produced at least 2 clusters.")

    df = pd.DataFrame(rows)
    df["silhouette_norm"] = _minmax(df["silhouette"].to_numpy())
    df["dbi_norm"] = 1.0 - _minmax(df["davies_bouldin"].to_numpy())
    df["weighted_score"] = w_silhouette * df["silhouette_norm"] + w_dbi * df["dbi_norm"]

    best_idx = df.sort_values(["weighted_score", "k"], ascending=[False, True]).index[0]
    best_k = int(df.loc[best_idx, "k"])
    df["selected"] = df["k"] == best_k

    return (
        df[["k", "silhouette", "davies_bouldin", "silhouette_norm", "dbi_norm",
                        "weighted_score", "shannon_index", "evenness", "selected"]],
        best_k,
        labels_by_k,
    )
