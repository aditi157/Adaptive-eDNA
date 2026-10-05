# eDNA Baseline (B0) -- Fixed-K Clustering Pipeline

This is the **frozen baseline** for the eDNA project: fixed K=10 K-Means
clustering of BLAST-unassigned reads on autoencoder embeddings. It is
meant to be run, evaluated, and locked before any AESCTA (adaptive /
confidence-guided) work is layered on top, so that the final comparison
is credible: *same representation pipeline, different decision layer.*

```
FASTA/FASTQ
    |
    v
Primary BLAST
    |
    v
BLAST filtering (evalue <= 1e-5, identity >= 90%)
    |
    v
Assigned / BLAST-unassigned
    |
    v
4-mer frequency features (256-D)
    |
    v
Train/test split
    |
    v
Autoencoder training (train MSE) -> 32-D latent space
    |                                      |
    v                                      v
Test-set reconstruction (test MSE)   Embeddings (all sequences)
                                            |
                                            v
                                     K-Means, K=10 (fitted, not trained)
                                            |
                                +-----------+-----------+
                                v                       v
                       Internal metrics          External metrics
                       (Silhouette, DBI)          (ARI, NMI -- only if a
                                                    labelled benchmark is
                                                    supplied)
                                |
                                v
                     Abundance + Shannon diversity
```

## What is explicitly NOT in this baseline

- Adaptive K search (K=2..10)
- Cluster confidence / reliability scoring
- Representative-sequence selection
- Secondary BLAST
- Putative taxonomic affiliation reporting

Those are the AESCTA additions, built as a separate phase on top of this
frozen baseline -- not mixed into it.

## What changed vs. the original prototype

- Removed `classifier.py` (unused, no training implementation, not
  wired into the app).
- Removed the placeholder `annotations.py` (`taxonomy`/`ecological_role`
  columns were hard-coded to `"Unknown"`; not real annotation).
- Removed `__pycache__/` and stale `data/results/*.csv` outputs -- these
  should always be regenerated fresh, not shipped alongside source.
- Replaced hard-coded, machine-specific paths
  (`C:/Users/.../edna_streamlit/...`) with project-relative paths
  (`scripts/config.py`).
- Fixed FASTQ handling: the uploader accepted `.fastq`, but the parser
  always called `SeqIO.parse(..., "fasta")`, which silently produced zero
  sequences for FASTQ input. FASTQ is now converted to FASTA first
  (`scripts/utils.py::ensure_fasta`).
- Removed duplicate imports and duplicated annotation/abundance calls in
  `app.py`.
- Fixed `calculate_abundance()`: it computed the Shannon index internally
  but never returned it, so `app.py` always printed `None`. It now
  returns `(abundance_dataframe, shannon_index)`.
- Fixed `visualization.py`: `plot_cluster_abundance()` was defined twice;
  the second definition called `fig.show()`, which does not render inside
  Streamlit. There is now a single definition using `st.plotly_chart`.
- Added a genuine train/test split for the autoencoder (previously it
  trained and evaluated on the exact same sequences). Train MSE and
  held-out test MSE are both reported.
- Added clustering evaluation: Silhouette and Davies-Bouldin (no labels
  needed), plus optional ARI/NMI if a labelled benchmark is supplied.
  Labels are never passed to the autoencoder or K-Means -- only used
  afterward, for evaluation.
- Fixed random seeds across Python, NumPy, and PyTorch
  (`scripts/reproducibility.py`) so repeated baseline runs are
  comparable.

## Project layout

```
edna_baseline/
|-- app.py                     Streamlit UI
|-- requirements.txt
|-- README.md
|-- scripts/
|   |-- config.py              All paths, thresholds, fixed seed, K=10
|   |-- reproducibility.py     set_seed()
|   |-- utils.py               BLAST runner, FASTA/FASTQ normalization
|   |-- parse_blast.py         BLAST filtering
|   |-- kmer_features.py       4-mer feature extraction
|   |-- autoencoder.py         Model, train/test split, train+eval
|   |-- baseline_clustering.py Fixed-K K-Means + internal/external metrics
|   |-- abundance.py           Relative abundance + Shannon diversity
|   |-- visualization.py       Plotly/Streamlit charts
|   \-- pipeline.py            Orchestrates the full run (shared by
|                               app.py and experiments/run_baseline.py)
|-- experiments/
|   \-- run_baseline.py        CLI entry point (no Streamlit needed)
|-- data/
|   |-- blast_db/              BLAST database (see its README.md --
|   |                          excluded here due to size, ~268 MB)
|   |-- input/                 Sample FASTA/FASTQ files
|   \-- results/               Generated on each run (see below)
\-- models/
    \-- autoencoder.pt         Saved after each run
```

## Running it

### Option A -- Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a FASTA/FASTQ file (and, optionally, a `id,label` CSV for a
labelled benchmark) and click "Run Baseline Pipeline".

### Option B -- command line (for scripted/repeated experiment runs)

```bash
python experiments/run_baseline.py --input data/input/output1.fasta
python experiments/run_baseline.py --input data/input/SRR34865974_1.fastq --k 10 --seed 42
python experiments/run_baseline.py --input data/input/benchmark.fasta --labels data/input/benchmark_labels.csv
```

Both require the `blastn` binary (NCBI BLAST+) on your PATH, and the
BLAST database copied into `data/blast_db/` (see that folder's
`README.md`).

## Output artifacts (`data/results/`)

| File | Contents |
| --- | --- |
| `blast_results.tsv` | Raw primary BLAST hits |
| `blast_filtered.csv` | Best hit per query passing thresholds |
| `unassigned_reads.fasta` | Reads with no acceptable BLAST hit |
| `autoencoder_train_metrics.csv` | Per-epoch training MSE |
| `autoencoder_test_metrics.csv` | Held-out test-set MSE |
| `embeddings.csv` | 32-D latent embedding per sequence |
| `baseline_clusters.csv` | Sequence -> cluster (K=10) assignment |
| `baseline_cluster_metrics.csv` | K, Silhouette, DBI, cluster sizes, (ARI/NMI if labelled) |
| `baseline_abundance.csv` | Relative abundance and Shannon index per cluster |
| `baseline_summary.json` | Everything above, plus stage timings |

## Reproducibility

All randomness (train/test split, autoencoder initialization/shuffling,
K-Means initialization) is controlled by a single seed
(`scripts/config.py::RANDOM_SEED`, default 42). Keep the seed, dataset,
and thresholds fixed across baseline runs you intend to compare.

## Freeze this before building AESCTA

Once you have a baseline run you're satisfied with, copy this whole
folder to something like `baseline/` (frozen) before starting AESCTA
development in a separate `aescta/` folder. That way "what exactly was
my baseline?" always has an unambiguous answer when you write up the
comparison.

## Building a labelled benchmark (optional, for ARI/NMI)

`experiments/build_benchmark.py` builds a labelled benchmark FASTA + CSV
directly from your existing `18S_fungal_sequences` BLAST database.

Note: that database is a *type/reference-strain* database -- almost every
species has exactly one sequence in it (check yourself with
`blastdbcmd -db data/blast_db/18S_fungal_sequences -info`). So the script
doesn't pull 20-50 *distinct real* sequences per species (that data
doesn't exist in this DB) -- instead, for each chosen species, it takes
that species' one real reference sequence and simulates multiple reads
from it (random ~301bp fragments + a small substitution rate), the same
way real amplicon sequencing produces many reads from one organism's gene
region. This is a semi-synthetic benchmark grounded in real reference
sequences -- state that plainly in your report.

```bash
python experiments/build_benchmark.py
# or customize:
python experiments/build_benchmark.py --n-species 6 --reads-per-species 30 \
    --read-length 301 --mutation-rate 0.02 --seed 42
# or hand-pick species:
python experiments/build_benchmark.py --species "Trichophyton rubrum,Candida albicans,Aspergillus niger"
```

This writes `data/input/benchmark.fasta` and
`data/input/benchmark_labels.csv`, ready to pass straight into
`experiments/run_baseline.py --labels ...` or the Streamlit sidebar.

Important BLAST+ gotcha this script works around: `blastdbcmd`'s species
lookup only finds `taxdb.bti`/`taxdb.btd` via the `BLASTDB` environment
variable (or current directory) -- not just because they sit next to your
`-db` path. If you ever call `blastdbcmd` directly and get `N/A` for every
species name, that's why.

## Internal vs. external clustering evaluation

- **Internal** (no ground truth): Silhouette, Davies-Bouldin -- always
  computed.
- **External** (ground truth available): supply a labelled benchmark CSV
  (`id,label`) via the sidebar/`--labels`. Labels are hidden from
  clustering and used only afterward to compute ARI/NMI against the
  predicted cluster IDs. Real, BLAST-unassigned environmental reads
  typically have no such ground truth, so this is optional and mainly
  intended for a constructed benchmark dataset of known sequences.


## Dashboard updates
- Cluster scatter plot (PCA projection of the 32-D embeddings, centroids marked).
- Autoencoder MSE shown in scientific notation, plus the mean-predictor MSE, the ratio to it, and test R-squared (variance reconstructed).
- Autoencoder training-loss graph removed.
- Shannon diversity index moved to the top (after BLAST counts and MSE values) and removed from the abundance table.
- BLAST e-values displayed in scientific notation.

## Adaptive K (added on top of the baseline)
`scripts/adaptive_kmeans.py` fits K-Means for K=2..10 on the same 32-D embeddings (same seed/n_init as the baseline,
so the K=10 row equals the baseline result). Per K it reports Silhouette, DBI, their min-max normalised values, a
weighted score (0.5 x silhouette_norm + 0.5 x dbi_norm) and the Shannon index (not part of the score). K* = highest
weighted score (ties -> smaller K). Outputs in `results/`: `adaptive_k_metrics.csv`, `adaptive_clusters.csv`,
`adaptive_abundance.csv`. The dashboard shows the per-K table, K* abundance table/graph and K* cluster plot.
Not yet implemented: cluster confidence, representative sequences, secondary BLAST, baseline-vs-adaptive comparison.
