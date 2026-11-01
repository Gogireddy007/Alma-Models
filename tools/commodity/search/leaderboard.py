"""Print the DEV results table (markdown) from runs/commodity/search/results.tsv, every row, sorted by score."""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from run_exp import TSV
df = pd.read_csv(TSV, sep="\t")
df["key"] = pd.to_numeric(df.score, errors="coerce").fillna(9)
cols = ["name", "arch", "abl", "params", "steps", "score", "ci5", "ci95", "ratio1", "ratio5", "ratio20", "da1", "da5",
        "da20", "volda", "cov80_20", "score_d", "ratio_d", "da_d", "status"]
t = df.sort_values("key")[cols].fillna("")
print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
for _, r in t.iterrows():
    print("| " + " | ".join(str(r[c]) for c in cols) + " |")
