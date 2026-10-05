"""
General I/O utilities for the baseline pipeline.

Fixes applied relative to the original prototype:
  - No hard-coded machine-specific paths (see scripts/config.py instead).
  - FASTQ inputs are properly converted to FASTA before anything downstream
    touches them (the original code accepted .fastq in the uploader but the
    parser only ever called SeqIO.parse(..., "fasta"), which silently
    produced zero sequences for FASTQ input).
"""

import os
import subprocess

import pandas as pd
from Bio import SeqIO


def ensure_fasta(input_path: str, output_dir: str) -> str:
    """
    Given a user-uploaded file that is either FASTA or FASTQ, return the
    path to a FASTA file that downstream steps (BLAST, k-mer featurization)
    can consume.

    If the input is already FASTA, it is returned unchanged. If it is
    FASTQ, it is converted (quality scores are dropped; only sequences and
    IDs are kept) and written to <output_dir>/<basename>.converted.fasta.
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".fasta", ".fa", ".fna"):
        return input_path

    if ext in (".fastq", ".fq"):
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        fasta_path = os.path.join(output_dir, f"{base}.converted.fasta")
        count = SeqIO.convert(input_path, "fastq", fasta_path, "fasta")
        if count == 0:
            raise ValueError(
                f"No sequences could be read from {input_path} as FASTQ."
            )
        return fasta_path

    raise ValueError(
        f"Unsupported input extension '{ext}'. Expected one of: "
        ".fasta, .fa, .fna, .fastq, .fq"
    )


def run_blast(input_fasta: str, blast_db: str, output_path: str) -> None:
    """
    Run a primary BLASTN search of input_fasta against blast_db, writing
    tabular (outfmt 6) results to output_path.

    Requires the `blastn` binary (NCBI BLAST+) to be installed and on PATH.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    missing_db_files = [
        ext for ext in (".nhr", ".nin", ".nsq")
        if not os.path.exists(blast_db + ext)
    ]
    if missing_db_files:
        raise RuntimeError(
            f"BLAST database '{blast_db}' looks incomplete -- missing "
            f"{', '.join(blast_db + e for e in missing_db_files)}. "
            "Copy ALL of the database files (not just some) into data/blast_db/ "
            "-- see data/blast_db/README.md for the full list."
        )

    cmd = [
        "blastn",
        "-query", input_fasta,
        "-db", blast_db,
        "-out", output_path,
        "-outfmt", "6",
        "-max_target_seqs", "5",
        "-num_threads", "4",
    ]
    try:
        # capture_output so that if blastn fails, we can surface its ACTUAL
        # error message instead of just "non-zero exit status N", which is
        # useless for diagnosing what actually went wrong.
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find the 'blastn' executable. Install NCBI BLAST+ "
            "and make sure it is on your PATH before running the baseline."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"blastn failed (exit code {exc.returncode}). "
            f"blastn's error message was:\n{stderr}\n\n"
            "Common causes: an incomplete/corrupted BLAST database in "
            "data/blast_db/, or a file path containing characters BLAST+ "
            "dislikes on Windows (spaces, parentheses)."
        ) from exc


def extract_unassigned_reads(
    all_fasta: str, blast_filtered_csv: str, output_fasta: str
) -> int:
    """
    Write every sequence from all_fasta whose id does NOT appear in the
    filtered BLAST hits to output_fasta. Returns the number of unassigned
    sequences written.
    """
    assigned_ids = set(pd.read_csv(blast_filtered_csv)["query_id"].unique())
    os.makedirs(os.path.dirname(output_fasta) or ".", exist_ok=True)

    n_written = 0
    with open(output_fasta, "w") as out_f:
        for record in SeqIO.parse(all_fasta, "fasta"):
            if record.id not in assigned_ids:
                SeqIO.write(record, out_f, "fasta")
                n_written += 1
    return n_written
