"""
Run the frozen B0 baseline from the command line and print a summary.

Usage:
    python experiments/run_baseline.py --input data/input/your_sample.fasta
    python experiments/run_baseline.py --input data/input/your_sample.fastq --k 10 --seed 42
    python experiments/run_baseline.py --input data/input/benchmark.fasta --labels data/input/benchmark_labels.csv

The labelled-benchmark CSV must have columns: id,label
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import BASELINE_K, RANDOM_SEED  # noqa: E402
from scripts.pipeline import run_baseline_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the B0 fixed-K eDNA baseline.")
    parser.add_argument("--input", required=True, help="Path to a FASTA or FASTQ file.")
    parser.add_argument("--k", type=int, default=BASELINE_K, help="Fixed number of clusters.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--labels", default=None,
                         help="Optional CSV (columns: id,label) for a labelled benchmark; "
                              "enables ARI/NMI external evaluation.")
    args = parser.parse_args()

    summary = run_baseline_pipeline(
        args.input, k=args.k, seed=args.seed, labels_csv=args.labels,
        progress_cb=print,
    )

    print("\n=== Baseline summary ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
