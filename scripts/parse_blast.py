"""
Parse raw BLASTN outfmt-6 output and keep the single best hit per query
that passes the identity / e-value thresholds.
"""

import os

import pandas as pd

from scripts.config import BLAST_EVALUE_MAX, BLAST_PIDENT_MIN

BLAST_COLUMNS = [
    "query_id", "subject_id", "perc_identity", "alignment_length",
    "mismatches", "gap_opens", "q_start", "q_end", "s_start", "s_end",
    "evalue", "bit_score",
]


def parse_blast_results(
    blast_file: str,
    output_file: str,
    evalue_max: float = BLAST_EVALUE_MAX,
    pident_min: float = BLAST_PIDENT_MIN,
) -> pd.DataFrame:
    """
    Read blast_file (outfmt 6, no header), filter by evalue/identity, keep
    the best hit per query_id, and write the result to output_file.
    """
    df = pd.read_csv(blast_file, sep="\t", names=BLAST_COLUMNS)

    filtered_df = df[
        (df["evalue"] <= evalue_max) & (df["perc_identity"] >= pident_min)
    ]
    filtered_df = filtered_df.sort_values(
        ["query_id", "evalue", "bit_score"], ascending=[True, True, False]
    )
    best_hits = filtered_df.drop_duplicates(subset=["query_id"], keep="first")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    best_hits.to_csv(output_file, index=False)
    return best_hits
