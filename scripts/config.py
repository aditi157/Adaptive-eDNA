"""
Central configuration for the B0 fixed-K eDNA clustering baseline.

Keeping every "knob" in one place makes the experiment reproducible and
makes it obvious, later, exactly which settings AESCTA inherits unchanged
from the baseline (representation pipeline) versus which settings AESCTA
replaces (fixed K -> adaptive K, no confidence -> confidence, etc).
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, NOT hard-coded to one machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
BLAST_DB_DIR = os.path.join(DATA_DIR, "blast_db")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BLAST_DB_PREFIX = os.path.join(BLAST_DB_DIR, "18S_fungal_sequences")

# ---------------------------------------------------------------------------
# BLAST filtering thresholds
# ---------------------------------------------------------------------------
BLAST_EVALUE_MAX = 1e-5
BLAST_PIDENT_MIN = 90.0

# ---------------------------------------------------------------------------
# k-mer feature representation
# ---------------------------------------------------------------------------
KMER_K = 4                     # 4-mer -> 4^4 = 256-D feature vector

# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
AE_LATENT_DIM = 32
AE_HIDDEN_DIM = 128
AE_EPOCHS = 100
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3
AE_TEST_SIZE = 0.2              # held-out fraction, never used for training

# ---------------------------------------------------------------------------
# Baseline clustering (fixed K)
# ---------------------------------------------------------------------------
BASELINE_K = 10

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (INPUT_DIR, RESULTS_DIR, BLAST_DB_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)
