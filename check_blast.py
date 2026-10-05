import os, time
import pandas as pd
from scripts.parse_blast import BLAST_COLUMNS
from scripts.config import BLAST_EVALUE_MAX, BLAST_PIDENT_MIN, BLAST_DB_PREFIX, RESULTS_DIR

path = os.path.join(RESULTS_DIR, "blast_results.tsv")
print("file:", path)
print("size (bytes):", os.path.getsize(path))
print("last modified:", time.ctime(os.path.getmtime(path)))
print("thresholds: evalue <=", BLAST_EVALUE_MAX, "| identity >=", BLAST_PIDENT_MIN)

df = pd.read_csv(path, sep="\t", names=BLAST_COLUMNS)
print("\nrows:", len(df), "| unique reads with a hit:", df.query_id.nunique())
print(df.head(3).to_string())
print(df[["perc_identity", "evalue"]].describe())
print("pass e-value:", (df.evalue <= BLAST_EVALUE_MAX).sum(),
      "| pass identity:", (df.perc_identity >= BLAST_PIDENT_MIN).sum(),
      "| pass both:", ((df.evalue <= BLAST_EVALUE_MAX) & (df.perc_identity >= BLAST_PIDENT_MIN)).sum())

print("\nBLAST database files:")
d = os.path.dirname(BLAST_DB_PREFIX)
for f in sorted(os.listdir(d)):
    print(f"  {f}: {os.path.getsize(os.path.join(d, f)):,} bytes")