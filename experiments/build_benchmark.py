"""
Build a labelled benchmark dataset from the existing 18S fungal BLAST
database, for external clustering evaluation (ARI / NMI).

WHY THIS APPROACH
------------------
The 18S_fungal_sequences database is a *type/reference-strain* database:
almost every species has exactly ONE sequence in it (verify yourself with
`blastdbcmd -db <db> -info`). That means you cannot literally "pull 20-50
real sequences per species" out of it -- there isn't that redundancy to
pull from.

Instead, this script:
  1. Picks N distinct, named fungal species that already have a reference
     sequence in your BLAST database.
  2. For each chosen species, extracts its one real reference sequence via
     `blastdbcmd`.
  3. Simulates multiple reads per species by cutting random fragments out
     of that reference (matching your real data's ~301bp read length) and
     applying a small substitution rate -- mimicking how real amplicon
     sequencing produces many overlapping reads from one organism's gene
     region, with the error rate real sequencers also have.

This gives you a labelled benchmark grounded in real reference sequences,
with the known caveat that it is semi-synthetic (simulated reads, not
literally multiple distinct specimens). State that plainly in your
report -- it's scientifically honest and still a legitimate benchmark
for sanity-checking ARI/NMI.

REQUIREMENTS
------------
- NCBI BLAST+ installed and on PATH (you said you already have this).
- The `18S_fungal_sequences` BLAST database present (you already have
  this, in data/blast_db/).
- Biopython (`pip install biopython`) -- same requirement as the rest of
  the baseline.

USAGE
-----
    python experiments/build_benchmark.py

    # customize:
    python experiments/build_benchmark.py \
        --db data/blast_db/18S_fungal_sequences \
        --n-species 6 \
        --reads-per-species 30 \
        --read-length 301 \
        --mutation-rate 0.02 \
        --seed 42 \
        --out-fasta data/input/benchmark.fasta \
        --out-labels data/input/benchmark_labels.csv

    # or hand-pick species yourself instead of auto-selecting:
    python experiments/build_benchmark.py \
        --species "Trichophyton rubrum,Candida albicans,Aspergillus niger"

Then run the baseline against it with external evaluation enabled:

    python experiments/run_baseline.py --input data/input/benchmark.fasta \
        --labels data/input/benchmark_labels.csv
"""

import argparse
import os
import random
import re
import subprocess
import sys


def run_blastdbcmd(args, db_prefix=None):
    """
    Run blastdbcmd with the given argument list and return stdout text.

    IMPORTANT: blastdbcmd's species-name lookup (%S) only finds
    taxdb.bti/taxdb.btd if they're on the BLASTDB environment variable
    path (or the current working directory) -- NOT simply because they
    sit next to the -db prefix. Without this, %S silently returns "N/A"
    for every sequence instead of erroring, which is easy to miss. We set
    BLASTDB here so this works regardless of where the script is run from.
    """
    env = os.environ.copy()
    if db_prefix:
        env["BLASTDB"] = os.path.dirname(os.path.abspath(db_prefix)) or "."

    try:
        result = subprocess.run(
            ["blastdbcmd"] + args, capture_output=True, text=True, check=True, env=env
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find 'blastdbcmd'. It ships with NCBI BLAST+ "
            "alongside blastn -- make sure the same install/PATH that "
            "gives you blastn also gives you blastdbcmd."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"blastdbcmd failed: {exc.stderr.strip()}"
        ) from exc
    return result.stdout


def list_species(db_prefix: str, min_length: int):
    """
    Return a list of (species_name, accession, length) for every sequence
    in the DB, restricted to:
      - species length >= min_length (so we can cut fragments of
        read_length out of it)
      - "clean" species names: no square brackets (provisional genus
        placement), no 'sp.'/'cf.'/'aff.' (uncertain identification)
    One entry per species (if a species happens to have more than one
    reference sequence, the longest is kept).
    """
    raw = run_blastdbcmd(["-db", db_prefix, "-entry", "all", "-outfmt", "%S|%a|%l"], db_prefix)

    best_per_species = {}
    bad_name_pattern = re.compile(r"\[|\bsp\.|\bcf\.|\baff\.", re.IGNORECASE)
    n_a_count = 0

    for line in raw.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        species, accession, length_str = parts
        species = species.strip()
        try:
            length = int(length_str)
        except ValueError:
            continue

        if species in ("N/A", ""):
            n_a_count += 1
            continue
        if length < min_length:
            continue
        if bad_name_pattern.search(species):
            continue

        if species not in best_per_species or length > best_per_species[species][1]:
            best_per_species[species] = (accession, length)

    if n_a_count > 0 and not best_per_species:
        raise RuntimeError(
            f"blastdbcmd returned 'N/A' for species name on all {n_a_count} entries. "
            "This means taxdb.bti/taxdb.btd could not be found for the taxonomy "
            "lookup. Make sure taxdb.bti and taxdb.btd sit in the same folder as "
            "your BLAST db files (they should already, from your original project)."
        )

    return [(sp, acc, length) for sp, (acc, length) in best_per_species.items()]


def fetch_sequence(db_prefix: str, accession: str) -> str:
    """Fetch the raw nucleotide sequence for one accession from the DB."""
    seq = run_blastdbcmd(["-db", db_prefix, "-entry", accession, "-outfmt", "%s"], db_prefix)
    return seq.strip().upper()


def simulate_reads(ref_seq: str, n_reads: int, read_length: int,
                    mutation_rate: float, rng: random.Random):
    """
    Cut n_reads random fragments of read_length out of ref_seq and apply
    independent random substitutions at mutation_rate per base, to mimic
    real amplicon sequencing reads (random start positions + sequencer
    error) drawn from one organism's gene region.

    If ref_seq is shorter than read_length, the whole sequence is used for
    every read (only mutations are applied, no fragmenting).
    """
    bases = "ACGT"
    reads = []
    max_start = max(len(ref_seq) - read_length, 0)

    for _ in range(n_reads):
        if max_start > 0:
            start = rng.randint(0, max_start)
            fragment = ref_seq[start:start + read_length]
        else:
            fragment = ref_seq

        mutated = list(fragment)
        for i, base in enumerate(mutated):
            if base not in bases:
                continue  # leave ambiguity codes (N, etc.) untouched
            if rng.random() < mutation_rate:
                choices = [b for b in bases if b != base]
                mutated[i] = rng.choice(choices)

        reads.append("".join(mutated))

    return reads


def slugify(species: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Build a labelled benchmark from the 18S fungal BLAST DB."
    )
    parser.add_argument("--db", default="data/blast_db/18S_fungal_sequences",
                         help="BLAST DB prefix (path without file extension).")
    parser.add_argument("--species", default=None,
                         help="Comma-separated list of species names to use "
                              "instead of auto-selecting (must exist in the DB).")
    parser.add_argument("--n-species", type=int, default=6,
                         help="Number of species to auto-select if --species is not given.")
    parser.add_argument("--reads-per-species", type=int, default=30,
                         help="Number of simulated reads to generate per species.")
    parser.add_argument("--read-length", type=int, default=301,
                         help="Length of each simulated read (301 matches your real SRR data).")
    parser.add_argument("--mutation-rate", type=float, default=0.02,
                         help="Per-base substitution probability (sequencing-error-like noise).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-fasta", default="data/input/benchmark.fasta")
    parser.add_argument("--out-labels", default="data/input/benchmark_labels.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.species:
        wanted = [s.strip() for s in args.species.split(",") if s.strip()]
        candidates = list_species(args.db, min_length=1)
        by_name = {sp: (acc, length) for sp, acc, length in candidates}
        chosen = []
        for sp in wanted:
            if sp not in by_name:
                print(f"WARNING: '{sp}' not found in the DB (or too short/ambiguous "
                      f"name) -- skipping.", file=sys.stderr)
                continue
            acc, length = by_name[sp]
            if length < args.read_length:
                print(f"WARNING: '{sp}' reference is only {length}bp, shorter than "
                      f"--read-length={args.read_length}. Reads for this species will "
                      f"all be the full (unfragmented) sequence instead.", file=sys.stderr)
            chosen.append((sp, acc, length))
    else:
        candidates = list_species(args.db, min_length=args.read_length)
        if len(candidates) < args.n_species:
            raise SystemExit(
                f"Only found {len(candidates)} species with a clean name and "
                f"length >= {args.read_length}bp; lower --read-length or "
                f"--n-species, or pass --species explicitly."
            )
        chosen = rng.sample(candidates, args.n_species)

    if not chosen:
        raise SystemExit("No species selected -- nothing to do.")

    print("Selected species:")
    for sp, acc, length in chosen:
        print(f"  - {sp}  (accession {acc}, {length}bp)")

    os.makedirs(os.path.dirname(args.out_fasta) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)

    with open(args.out_fasta, "w") as fasta_f, open(args.out_labels, "w") as labels_f:
        labels_f.write("id,label\n")

        for species, accession, length in chosen:
            ref_seq = fetch_sequence(args.db, accession)
            reads = simulate_reads(
                ref_seq, args.reads_per_species, args.read_length,
                args.mutation_rate, rng,
            )

            slug = slugify(species)
            for i, read in enumerate(reads, start=1):
                read_id = f"{slug}_{i:03d}"
                fasta_f.write(f">{read_id} source={accession} species=\"{species}\"\n")
                fasta_f.write(read + "\n")
                labels_f.write(f"{read_id},{species}\n")

    n_total = len(chosen) * args.reads_per_species
    print(f"\nWrote {n_total} simulated reads across {len(chosen)} species.")
    print(f"  FASTA:  {args.out_fasta}")
    print(f"  Labels: {args.out_labels}")
    print(
        "\nRun the baseline against it with external evaluation:\n"
        f"  python experiments/run_baseline.py --input {args.out_fasta} "
        f"--labels {args.out_labels}"
    )


if __name__ == "__main__":
    main()
