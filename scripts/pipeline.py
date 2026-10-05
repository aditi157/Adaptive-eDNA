"""
B0 baseline pipeline orchestration.

    Input FASTA/FASTQ
          |
          v
     Primary BLAST
          |
          v
     BLAST filtering
          |
          v
  Assigned / BLAST-unassigned
          |
          v
   4-mer frequency features (256-D)
          |
          v
   train/test split (autoencoder)
          |
          v
   Autoencoder training + test-set MSE
          |
          v
   32-D latent embeddings (all sequences)
          |
          v
      K-Means, K=10 (fitted, not "trained")
          |
          v
   Internal metrics (Silhouette, DBI)
   [+ External metrics (ARI, NMI) if a labelled benchmark is supplied]
          |
          v
   Abundance + Shannon diversity
          |
          v
   Adaptive K search (K=2..10): Silhouette + DBI -> weighted score -> K*
   -> abundance + Shannon for K*   [runs alongside the fixed-K=10 baseline]

This module is the single source of truth for the baseline so that
app.py (Streamlit) and experiments/run_baseline.py (CLI) cannot drift
apart.
"""

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from scripts import autoencoder as ae
from scripts import baseline_clustering as bc
from scripts.abundance import calculate_abundance
from scripts.adaptive_kmeans import adaptive_k_search
from scripts.config import (
    BASELINE_K,
    BLAST_DB_PREFIX,
    INPUT_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    ensure_dirs,
)
from scripts.kmer_features import build_kmer_matrix
from scripts.parse_blast import parse_blast_results
from scripts.reproducibility import set_seed
from scripts.utils import ensure_fasta, extract_unassigned_reads, run_blast


def run_baseline_pipeline(
    input_path: str,
    k: int = BASELINE_K,
    seed: int = RANDOM_SEED,
    labels_csv: Optional[str] = None,
    progress_cb=None,
) -> dict:
    """
    Run the full B0 baseline end to end and write all result artifacts to
    scripts.config.RESULTS_DIR.

    Parameters
    ----------
    input_path : path to a user-supplied FASTA or FASTQ file.
    k          : fixed number of K-Means clusters (baseline default: 10).
    seed       : random seed for the autoencoder split/init and K-Means.
    labels_csv : optional path to a CSV with columns [id, label] giving
                 ground-truth group membership for a labelled benchmark
                 dataset. If provided, ARI/NMI are additionally reported.
                 Labels are NEVER passed to the autoencoder or K-Means --
                 only used afterwards for external evaluation.
    progress_cb: optional callable(str) invoked with a short status string
                 after each pipeline stage (used by the Streamlit app).

    Returns
    -------
    dict summary of key results (also written to
    data/results/baseline_summary.json).
    """
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    ensure_dirs()
    set_seed(seed)
    summary = {"k": k, "seed": seed, "timings_sec": {}}

    # 1. Normalize input to FASTA -------------------------------------------------
    t0 = time.time()
    fasta_path = ensure_fasta(input_path, INPUT_DIR)
    summary["timings_sec"]["input_normalization"] = round(time.time() - t0, 3)
    report("Input normalized to FASTA")

    # 2. Primary BLAST -------------------------------------------------------------
    blast_out = os.path.join(RESULTS_DIR, "blast_results.tsv")
    t0 = time.time()
    run_blast(fasta_path, BLAST_DB_PREFIX, blast_out)
    summary["timings_sec"]["primary_blast"] = round(time.time() - t0, 3)
    report("Primary BLAST completed")

    # 3. Filter BLAST hits -----------------------------------------------------------
    blast_filtered = os.path.join(RESULTS_DIR, "blast_filtered.csv")
    best_hits = parse_blast_results(blast_out, blast_filtered)
    summary["n_assigned_by_blast"] = int(best_hits["query_id"].nunique())
    report("BLAST results filtered")

    # 4. Extract BLAST-unassigned reads ----------------------------------------------
    unassigned_fasta = os.path.join(RESULTS_DIR, "unassigned_reads.fasta")
    n_unassigned = extract_unassigned_reads(fasta_path, blast_filtered, unassigned_fasta)
    summary["n_unassigned"] = n_unassigned
    report(f"Extracted {n_unassigned} BLAST-unassigned reads")

    if n_unassigned == 0:
        summary["warning"] = "No unassigned reads; downstream ML stages skipped."
        with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # 5. k-mer features ------------------------------------------------------------
    ids, seqs, features = build_kmer_matrix(unassigned_fasta)
    summary["n_features"] = int(features.shape[1])

    # 6. Train/test split for the autoencoder -----------------------------------------
    train_feat, test_feat, train_idx, test_idx = ae.split_features(features, seed=seed)
    report(f"Split into {len(train_feat)} train / {len(test_feat)} test sequences")

    # 7. Train autoencoder on TRAIN ONLY ----------------------------------------------
    t0 = time.time()
    model, loss_history = ae.train_autoencoder(train_feat, seed=seed)
    summary["timings_sec"]["autoencoder_training"] = round(time.time() - t0, 3)
    train_mse = loss_history[-1] if loss_history else None
    report("Autoencoder training complete")

    # 8. Evaluate on held-out TEST set -------------------------------------------------
    test_mse = ae.evaluate_autoencoder(model, test_feat)
    # Context for the raw MSE: k-mer frequencies are tiny (~0.004), so MSE is
    # tiny almost regardless of model quality. Compare against a "dumb" model
    # that ignores the input and always predicts the mean training vector.
    #   relative_mse = model MSE / mean-predictor MSE   (<1 = better than the mean)
    #   r2           = 1 - relative_mse                 (fraction of variance reconstructed)
    train_mean = train_feat.mean(axis=0)
    mean_pred_train = float(np.mean((train_feat - train_mean) ** 2))
    mean_pred_test = float(np.mean((test_feat - train_mean) ** 2))
    rel_train = (train_mse / mean_pred_train) if (train_mse is not None and mean_pred_train > 0) else None
    rel_test = (test_mse / mean_pred_test) if mean_pred_test > 0 else None
    summary["autoencoder"] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "mean_predictor_train_mse": mean_pred_train,
        "mean_predictor_test_mse": mean_pred_test,
        "relative_train_mse": rel_train,
        "relative_test_mse": rel_test,
        "train_r2": (1 - rel_train) if rel_train is not None else None,
        "test_r2": (1 - rel_test) if rel_test is not None else None,
        "test_over_train_ratio": (test_mse / train_mse) if train_mse else None,
    }
    report(f"Autoencoder test MSE = {test_mse:.6f}")

    pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "train_mse": loss_history}) \
        .to_csv(os.path.join(RESULTS_DIR, "autoencoder_train_metrics.csv"), index=False)
    pd.DataFrame([{"n_test_sequences": len(test_feat), "test_mse": test_mse,
                   "mean_predictor_test_mse": mean_pred_test,
                   "relative_test_mse": rel_test,
                   "test_r2": summary["autoencoder"]["test_r2"]}]) \
        .to_csv(os.path.join(RESULTS_DIR, "autoencoder_test_metrics.csv"), index=False)
    ae.save_model(model, os.path.join(MODELS_DIR, "autoencoder.pt"))

    # 9. Embeddings for ALL unassigned sequences (for clustering) ----------------------
    embeddings = ae.get_embeddings(model, features)
    pd.DataFrame(embeddings, index=ids).to_csv(
        os.path.join(RESULTS_DIR, "embeddings.csv"), index_label="id"
    )

    # 10. Fixed-K K-Means (fitted, not trained) ----------------------------------------
    t0 = time.time()
    labels, kmeans_model = bc.fit_kmeans(embeddings, k=k, seed=seed)
    summary["timings_sec"]["kmeans_fit"] = round(time.time() - t0, 3)

    clusters_df = pd.DataFrame({"id": ids, "sequence": seqs, "cluster": labels})
    clusters_df.to_csv(os.path.join(RESULTS_DIR, "baseline_clusters.csv"), index=False)
    report(f"K-Means (K={k}) fitted")

    # 11. Internal clustering metrics --------------------------------------------------
    metrics = bc.internal_metrics(embeddings, labels)
    metrics["k"] = int(len(set(labels)))
    metrics["n_sequences"] = int(len(labels))
    sizes = bc.cluster_size_table(labels)
    metrics["largest_cluster"] = max(sizes.values())
    metrics["smallest_cluster"] = min(sizes.values())

    # 12. Optional external metrics (labelled benchmark only) --------------------------
    if labels_csv:
        label_df = pd.read_csv(labels_csv).set_index("id")
        aligned_true = label_df.reindex(ids)["label"]
        if aligned_true.isna().any():
            missing = int(aligned_true.isna().sum())
            report(f"Warning: {missing} sequences missing ground-truth labels; "
                   "external metrics skipped.")
        else:
            ext = bc.external_metrics(aligned_true.values, labels)
            metrics.update(ext)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, "baseline_cluster_metrics.csv"), index=False
    )
    summary["cluster_metrics"] = metrics
    report("Clustering metrics computed")

    # 13. Abundance + Shannon diversity -------------------------------------------------
    abundance_df, shannon_index = calculate_abundance(
        os.path.join(RESULTS_DIR, "baseline_clusters.csv"),
        os.path.join(RESULTS_DIR, "baseline_abundance.csv"),
    )
    summary["shannon_diversity"] = shannon_index
    report(f"Shannon diversity index = {shannon_index:.4f}")

    # 14. Adaptive K search (K = 2..10) -- weighted Silhouette + DBI ---------------------
    t0 = time.time()
    if len(embeddings) < 3:
        summary["adaptive_k"] = {"warning": "Fewer than 3 sequences; adaptive K search skipped."}
        report("Adaptive K search skipped (fewer than 3 sequences)")
    else:
        k_table, best_k, labels_by_k = adaptive_k_search(embeddings, seed=seed)
        k_table.to_csv(os.path.join(RESULTS_DIR, "adaptive_k_metrics.csv"), index=False)

        pd.DataFrame({"id": ids, "sequence": seqs, "cluster": labels_by_k[best_k]}) \
            .to_csv(os.path.join(RESULTS_DIR, "adaptive_clusters.csv"), index=False)
        adaptive_abundance_df, adaptive_shannon = calculate_abundance(
            os.path.join(RESULTS_DIR, "adaptive_clusters.csv"),
            os.path.join(RESULTS_DIR, "adaptive_abundance.csv"),
        )
        best_row = k_table[k_table["k"] == best_k].iloc[0]
        summary["adaptive_k"] = {
            "selected_k": best_k,
            "k_range_tested": [int(k_table["k"].min()), int(k_table["k"].max())],
            "weighted_score": float(best_row["weighted_score"]),
            "silhouette": float(best_row["silhouette"]),
            "davies_bouldin": float(best_row["davies_bouldin"]),
            "shannon_diversity": adaptive_shannon,
        }
        report(f"Adaptive K search complete -- selected K* = {best_k}")
    summary["timings_sec"]["adaptive_k_search"] = round(time.time() - t0, 3)

    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
