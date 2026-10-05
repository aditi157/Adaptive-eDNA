"""
eDNA Biodiversity Analysis Platform -- B0 Fixed-K Baseline

    Input FASTA/FASTQ -> BLAST -> filter -> unassigned reads ->
    4-mer features -> autoencoder (train/test) -> 32-D embeddings ->
    K-Means (K=10) -> abundance -> Shannon diversity

The fixed-K=10 baseline is kept as-is. On top of it, this version adds the
AESCTA adaptive-K stage (K=2..10, weighted Silhouette + DBI). Cluster
confidence, representative-sequence selection and secondary BLAST are not
part of this app yet.
"""

import os

import pandas as pd
import streamlit as st
import math

from scripts.adaptive_kmeans import W_DAVIES_BOULDIN, W_SILHOUETTE
from scripts.config import BASELINE_K, INPUT_DIR, RANDOM_SEED, RESULTS_DIR, ensure_dirs
from scripts.pipeline import run_baseline_pipeline
from scripts.visualization import plot_cluster_abundance, plot_clusters_2d

ensure_dirs()

st.set_page_config(page_title="eDNA Adaptive K Selection", layout="wide")
st.title("eDNA Biodiversity Analysis Platform")
st.caption("Adaptive K selection")

with st.sidebar:
    st.header("Baseline settings")
    k = st.number_input("K (number of clusters)", min_value=2, max_value=50, value=BASELINE_K)
    seed = st.number_input("Random seed", min_value=0, value=RANDOM_SEED)
    
    st.divider()
    st.subheader("Optional: labelled benchmark")
    labels_file = st.file_uploader(
        "CSV with columns 'id,label' (hidden from clustering, used only "
        "for ARI/NMI evaluation)",
        type=["csv"],
    )

uploaded_file = st.file_uploader("Upload eDNA FASTA/FASTQ file", type=["fasta", "fa", "fastq", "fq"])

if uploaded_file:
    input_path = os.path.join(INPUT_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    labels_path = None
    if labels_file:
        labels_path = os.path.join(INPUT_DIR, labels_file.name)
        with open(labels_path, "wb") as f:
            f.write(labels_file.getbuffer())

    if st.button("Run Baseline Pipeline"):
        status = st.empty()

        def progress_cb(msg):
            status.info(msg)

        with st.spinner("Running baseline pipeline..."):
            try:
                summary = run_baseline_pipeline(
                    input_path, k=int(k), seed=int(seed),
                    labels_csv=labels_path, progress_cb=progress_cb,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

        st.success("Baseline pipeline completed")

        st.subheader("Run summary")
        st.json(summary)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("BLAST-assigned reads", summary.get("n_assigned_by_blast", "-"))
        with col2:
            st.metric("BLAST-unassigned reads", summary.get("n_unassigned", "-"))

        if summary.get("n_unassigned", 0) > 0:
            ae_metrics = summary.get("autoencoder", {})

            def sci(v):
                return f"{v:.3e}" if v is not None else "-"

            col3, col4 = st.columns(2)
            with col3:
                st.metric("Autoencoder train MSE", sci(ae_metrics.get("train_mse")))
            with col4:
                st.metric("Autoencoder test MSE", sci(ae_metrics.get("test_mse")))

            # Raw MSE is tiny because 4-mer frequencies are tiny (~0.004 each),
            # so also show it relative to a "predict the mean" model.
            rel = ae_metrics.get("relative_test_mse")
            r2 = ae_metrics.get("test_r2")
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Mean-predictor test MSE", sci(ae_metrics.get("mean_predictor_test_mse")))
            with col6:
                st.metric("Test MSE vs mean-predictor", f"{rel:.3f}x" if rel is not None else "-")
            with col7:
                st.metric("Variance reconstructed (test R\u00b2)", f"{r2 * 100:.1f}%" if r2 is not None else "-")
            
            # _h = summary.get("shannon_diversity", 0)
            # _k = summary.get("k", 0)
            # _j = (_h / math.log(_k)) if _k and _k > 1 else None
            # s1, s2 = st.columns(2)
            # s1.metric("Baseline Shannon diversity index", f"{_h:.4f}")
            # s2.metric("Baseline evenness (Shannon / ln K)", f"{_j:.3f}" if _j is not None else "-")
            # st.subheader("Filtered BLAST matches")
            # st.dataframe(
            #     pd.read_csv(os.path.join(RESULTS_DIR, "blast_filtered.csv")),
            #     column_config={"evalue": st.column_config.NumberColumn("evalue", format="%.2e")},
            # )

            # st.header("Baseline (fixed K)")
            # st.subheader(f"Baseline clusters (K={summary['k']})")
            # st.dataframe(pd.read_csv(os.path.join(RESULTS_DIR, "baseline_clusters.csv")))

            # st.subheader("Cluster visualization")
            # plot_clusters_2d(
            #     os.path.join(RESULTS_DIR, "embeddings.csv"),
            #     os.path.join(RESULTS_DIR, "baseline_clusters.csv"),
            # )

            # st.subheader("Clustering metrics")
            # st.dataframe(pd.read_csv(os.path.join(RESULTS_DIR, "baseline_cluster_metrics.csv")))

            # st.subheader("Abundance")
            # abundance_csv = os.path.join(RESULTS_DIR, "baseline_abundance.csv")
            # st.dataframe(pd.read_csv(abundance_csv).drop(columns=["shannon_index"], errors="ignore"))
            # plot_cluster_abundance(abundance_csv)
            
            st.subheader("Filtered BLAST matches")
            blast_df = pd.read_csv(os.path.join(RESULTS_DIR, "blast_filtered.csv"))
            # Display order: put evalue in 4th position (display only; the CSV is unchanged)
            front = ["query_id", "subject_id", "perc_identity", "evalue"]
            blast_df = blast_df[front + [c for c in blast_df.columns if c not in front]]
            st.dataframe(
                blast_df,
                column_config={"evalue": st.column_config.NumberColumn("evalue", format="%.2e")},
            )

            # ---------------- Adaptive K ----------------
            st.divider()
            st.header("Adaptive K selection")
            adaptive = summary.get("adaptive_k", {})
            if "selected_k" not in adaptive:
                st.warning(adaptive.get("warning", "Adaptive K search did not run."))
            else:
                best_k = adaptive["selected_k"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Selected K*", best_k)
                c2.metric("Weighted score at K*", f"{adaptive['weighted_score']:.4f}")
                c3.metric("Shannon diversity at K*", f"{adaptive['shannon_diversity']:.4f}")

                st.subheader(f"Clustering metrics for each K ({adaptive['k_range_tested'][0]}\u2013{adaptive['k_range_tested'][1]})")
                k_table = pd.read_csv(os.path.join(RESULTS_DIR, "adaptive_k_metrics.csv"))
                float_cols = [c for c in k_table.columns if c not in ("k", "selected")]
                st.dataframe(
                    k_table.style.apply(
                        lambda r: ["background-color: #d4edda" if r["selected"] else "" for _ in r], axis=1
                    ).format({c: "{:.4f}" for c in float_cols}),
                    use_container_width=True, hide_index=True,
                )
                st.markdown(
                    "**How the weighted score is calculated** (min and max are taken across the K values tested):"
                )
                st.latex(r"\text{silhouette\_norm} = \frac{S - S_{min}}{S_{max} - S_{min}}")
                st.latex(r"\text{dbi\_norm} = \frac{DBI_{max} - DBI}{DBI_{max} - DBI_{min}}")
                st.latex(
                    rf"\text{{weighted\_score}} = {W_SILHOUETTE}\times\text{{silhouette\_norm}} "
                    rf"+ {W_DAVIES_BOULDIN}\times\text{{dbi\_norm}}"
                )

                st.subheader(f"Adaptive clusters (K*={best_k})")
                st.dataframe(pd.read_csv(os.path.join(RESULTS_DIR, "adaptive_clusters.csv")))

                st.subheader(f"Cluster visualization (K*={best_k})")
                plot_clusters_2d(
                    os.path.join(RESULTS_DIR, "embeddings.csv"),
                    os.path.join(RESULTS_DIR, "adaptive_clusters.csv"),
                    title=f"Adaptive K-Means clusters, K*={best_k} (PCA projection of 32-D embeddings)",
                )

                st.subheader(f"Abundance (K*={best_k})")
                adaptive_abundance_csv = os.path.join(RESULTS_DIR, "adaptive_abundance.csv")
                st.dataframe(pd.read_csv(adaptive_abundance_csv).drop(columns=["shannon_index"], errors="ignore"))
                plot_cluster_abundance(
                    adaptive_abundance_csv, title=f"Cluster Relative Abundance (K*={best_k})"
                )
        else:
            st.warning("No BLAST-unassigned reads were found, so no clustering was performed.")
