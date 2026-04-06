"""Aggregate a Raven Pro selection table by merging consecutive same-cluster selections.

Reads an existing Raven selection table and writes a new one where immediately
consecutive selections with the same Cluster label are merged into a single
selection. The Score is averaged across merged selections.

Output is saved as <stem>_aggregate.txt next to the input file.

Usage:
    uv run python scripts/aggregate_raven.py <raven.txt>
"""

import sys
from pathlib import Path

# --- args ---------------------------------------------------------------------

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

in_path = Path(sys.argv[1])
out_path = in_path.with_name(in_path.stem + "_aggregate.txt")

# --- read ---------------------------------------------------------------------

with open(in_path) as f:
    lines = f.readlines()

header = lines[0].rstrip("\n").split("\t")
rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in lines[1:] if line.strip()]

if not rows:
    print("No selections found.")
    sys.exit(0)

# --- aggregate ----------------------------------------------------------------

merged = []
cur = dict(rows[0])
scores = [float(cur["Score"])]

for row in rows[1:]:
    if row["Cluster"] == cur["Cluster"]:
        cur["End Time (s)"] = row["End Time (s)"]
        scores.append(float(row["Score"]))
    else:
        cur["Score"] = f"{sum(scores) / len(scores):.4f}"
        merged.append(cur)
        cur = dict(row)
        scores = [float(cur["Score"])]

cur["Score"] = f"{sum(scores) / len(scores):.4f}"
merged.append(cur)

# --- write --------------------------------------------------------------------

with open(out_path, "w") as f:
    f.write("\t".join(header) + "\n")
    for i, row in enumerate(merged, start=1):
        row["Selection"] = str(i)
        f.write("\t".join(row[col] for col in header) + "\n")

print(f"Aggregated {len(rows)} → {len(merged)} selections → {out_path}")
