"""
Relative abundance and Shannon diversity, computed from cluster
assignments.

Fix applied relative to the original prototype: the original
calculate_abundance() computed shannon_index internally but never
returned it, even though app.py assigned its return value to a variable
and printed it (always printing None). It now actually returns the value.
"""

from math import log
from typing import Tuple

import pandas as pd
from collections import Counter


def calculate_abundance(cluster_csv: str, abundance_output_csv: str) -> Tuple[pd.DataFrame, float]:
    """
    Read a CSV with a 'cluster' column, compute per-cluster count and
    relative abundance, compute the Shannon diversity index over clusters,
    write the abundance table to abundance_output_csv, and return
    (abundance_dataframe, shannon_index).
    """
    cluster_df = pd.read_csv(cluster_csv)
    cluster_counts = Counter(cluster_df["cluster"])
    total = sum(cluster_counts.values())

    abundance_data = []
    for cluster, count in sorted(cluster_counts.items()):
        abundance_data.append({
            "cluster": cluster,
            "count": count,
            "relative_abundance": count / total,
        })

    shannon_index = -sum(
        row["relative_abundance"] * log(row["relative_abundance"])
        for row in abundance_data
        if row["relative_abundance"] > 0
    )

    df_abundance = pd.DataFrame(abundance_data)
    df_abundance["shannon_index"] = shannon_index
    df_abundance.to_csv(abundance_output_csv, index=False)

    return df_abundance, shannon_index
