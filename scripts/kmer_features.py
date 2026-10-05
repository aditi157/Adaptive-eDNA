"""
Turn nucleotide sequences into fixed-length k-mer frequency vectors.

This is the "4-mer 256-D" representation stage referenced throughout the
project spec. It sits upstream of the autoencoder and is shared, unchanged,
between the baseline and AESCTA.
"""

import itertools
from collections import Counter
from typing import List, Tuple

import numpy as np
from Bio import SeqIO

from scripts.config import KMER_K

BASES = ["A", "C", "G", "T"]


def all_kmers(k: int = KMER_K) -> List[str]:
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def seq_to_kmer_vector(seq: str, k: int = KMER_K) -> np.ndarray:
    """
    Return the normalized k-mer frequency vector for a single sequence.
    Sequences shorter than k, or sequences with only ambiguous bases,
    produce an all-zero vector.
    """
    seq = seq.upper()
    kmer_counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    kmers = all_kmers(k)
    vector = np.array([kmer_counts.get(kmer, 0) for kmer in kmers], dtype=float)
    total = vector.sum()
    return vector / total if total > 0 else vector


def build_kmer_matrix(fasta_path: str, k: int = KMER_K) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Read every sequence from fasta_path and return:
      - list of sequence ids
      - list of raw sequence strings
      - (n_sequences, 4^k) feature matrix
    """
    ids, seqs = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq))

    if len(seqs) == 0:
        raise ValueError(f"No sequences found in {fasta_path}.")

    matrix = np.array([seq_to_kmer_vector(s, k) for s in seqs])
    return ids, seqs, matrix
