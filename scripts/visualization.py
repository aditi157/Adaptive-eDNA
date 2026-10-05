"""
Plotly/Streamlit visualization helpers.

Fix applied relative to the original prototype: the file previously
defined plot_cluster_abundance() TWICE. The second definition silently
overrode the first and called fig.show(), which opens a separate browser
tab/tries to launch a renderer instead of drawing inside the Streamlit
app. There is now a single definition that renders inline via
st.plotly_chart.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA


def plot_cluster_abundance(abundance_csv: str, title: str = "Cluster Relative Abundance") -> None:
    """Bar chart of relative abundance per cluster, rendered in Streamlit."""
    df = pd.read_csv(abundance_csv)
    fig = px.bar(
        df, x="cluster", y="relative_abundance",
        title=title,
        labels={"cluster": "Cluster ID", "relative_abundance": "Relative Abundance"},
        text="count",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def plot_clusters_2d(embeddings_csv: str, clusters_csv: str,
                     title: str = "K-Means clusters (PCA projection of 32-D embeddings)") -> None:
    """
    Scatter plot of the actual K-Means clusters.

    The 32-D autoencoder embeddings are projected to 2-D with PCA (for
    display only -- clustering was done in the full 32-D space, so clusters
    can overlap more in this 2-D view than they really do). Each point is a
    sequence, coloured by cluster; the X markers are cluster centroids.
    """
    emb = pd.read_csv(embeddings_csv).set_index("id")
    clusters = pd.read_csv(clusters_csv).set_index("id")["cluster"]
    emb = emb.loc[clusters.index]

    n_comp = 2 if emb.shape[0] >= 2 and emb.shape[1] >= 2 else 1
    if n_comp < 2:
        st.info("Not enough data points/dimensions for a 2-D cluster plot.")
        return
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(emb.values)
    var = pca.explained_variance_ratio_ * 100

    df = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "cluster": clusters.astype(str).values, "id": clusters.index,
    })
    order = sorted(df["cluster"].unique(), key=int)
    fig = px.scatter(
        df, x="PC1", y="PC2", color="cluster", hover_name="id",
        category_orders={"cluster": order},
        title=title,
        labels={"PC1": f"PC1 ({var[0]:.1f}% variance)", "PC2": f"PC2 ({var[1]:.1f}% variance)"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))

    cent = df.groupby("cluster")[["PC1", "PC2"]].mean().reset_index()
    for _, r in cent.iterrows():
        fig.add_scatter(
            x=[r["PC1"]], y=[r["PC2"]], mode="markers+text",
            text=[r["cluster"]], textposition="top center",
            marker=dict(symbol="x", size=13, color="black", line=dict(width=1, color="white")),
            showlegend=False, hoverinfo="skip",
        )
    st.plotly_chart(fig, use_container_width=True)
