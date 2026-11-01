"""Summarise results_b.tsv: EM rows grouped by name (mean over seeds), rows after the RT-4 annotation only."""
import sys, csv, collections
import numpy as np
rows = list(csv.DictReader(open("runs/fin/search_b/results_b.tsv"), delimiter="\t"))
k = max(i for i, r in enumerate(rows) if r["kind"] == "annot")
g = collections.defaultdict(list)
for r in rows[k + 1:]:
    if r["fold"] == "EM" and r["status"] == "ok" and r["kind"] in ("wm_b",):
        g[r["name"]].append((float(r["P1_all"]), float(r["P3_all"]), float(r["P1_heldout"])))
pref = sys.argv[1] if len(sys.argv) > 1 else ""
out = sorted(((np.mean([x[0] for x in v]), n, len(v), np.mean([x[1] for x in v]), np.mean([x[2] for x in v]),
               np.std([x[0] for x in v])) for n, v in g.items() if n.startswith(pref)), reverse=True)
for m, n, c, q, h, s in out[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print(f"{m:.4f} sd {s:.4f} n{c} QS {q:.4f} ho {h:.4f} {n}")
print("total rows", len(rows) - 1, "EM rows", sum(r["fold"] == "EM" for r in rows))
