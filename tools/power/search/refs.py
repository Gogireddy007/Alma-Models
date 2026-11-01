"""Reference rows for the search metric (naive baselines; no training)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_data as W, wm_models as M
from run import score, naive_epf, append, BT
D = W.PowerData("dev"); bt = W.valid_days(D, *BT, 14); ql = M.QL.numpy()
def naive(lag):
    out = np.zeros((len(bt), 13, 24), np.float32)
    for n, i in enumerate(bt):
        diffs = D.P[i - 364:i] - D.P[i - 364 - lag:i - lag]
        out[n] = D.P[i - lag][None] + np.nanquantile(diffs, ql, axis=0)
    return out
for name, q in [("ref_naive_epf", naive_epf(D, bt)), ("ref_naive_d", naive(1)), ("ref_naive_w", naive(7))]:
    r = score(D, bt, q); r.update(name=name, arch="ref", status="ok", note="reference"); r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    append(r); print(r)
