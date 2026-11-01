"""F1 sanity (auditor): P1 of reference models refit at 2024-06-30 on the F1 window, and B/A F1 preds re-scored from
raw labels (own code). DEV only."""
import os, sys, glob
import torch  # noqa: F401  (lead A3 import order)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
import rt_arch as RA
H, DT = R.H, R.DT
sys.path.insert(0, os.path.join(R.FIN, "search"))
import common as C
import models as M
y, syms, ho = RA.raw_labels()
P = C.dev_panel("crypto"); idx = C.bt_days(P); days = pd.DatetimeIndex(P.days[idx])
Y = y.loc[days, syms].to_numpy()
def acc(p, zs):
    Yz = y.loc[days, list(zs)].to_numpy()
    d = np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan)
    ok = np.isfinite(Yz)
    d = np.where(ok & ~np.isfinite(d), 0.0, d)          # missing -> DEV-majority 'down' (A4 rule)
    return float(((d == Yz) & ok).sum() / ok.sum())
for mn in ["har", "har_roll", "lgbm", "rv_persist"]:
    with C.fit_end_patch(C.FOLD_FIT_END):
        m = M.get(mn)
    r = H.run(P, m, idx, online=False)
    if np.isfinite(r["p_up"]).any():
        p = r["p_up"]
    else:
        ref = np.stack([P.labels(int(i))["ref"] for i in idx])
        med = r["q"][:, :, H.QS.index(0.5)]
        p = np.where(np.isfinite(med), (med > ref).astype(float), np.nan)
    print(f"ref {mn}: F1 P1 {acc(p, r['symbols']):.4f}", flush=True)
for f in sorted(glob.glob(os.path.join(R.ROOT, "runs/fin/search/preds/*F1*_crypto.npz")) +
                glob.glob(os.path.join(R.ROOT, "runs/fin/search/preds/*2024-06-30_crypto.npz"))):
    z = np.load(f)
    if "p_up" in z.files:
        print(f"{os.path.basename(f)}: F1 P1 {acc(z['p_up'], z['symbols']):.4f}")
