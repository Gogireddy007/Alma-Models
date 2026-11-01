"""
Alma-Live Gas FIXED harness pieces (never edited by the autoresearch loop; red team LRT-17): fold construction,
normalisation, input windows, inference and the scoring pinball. wm_core.py (the editable file) imports these.
"""
import os

import numpy as np
import torch

WD = __import__(os.environ.get("LIVE_SRC", "wm_store"))
QL = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ------------------------------------------------------------------ data windows
class Norm:
    def __init__(self, X, Y, idx):
        self.xm = X[idx].mean(0); self.xs = X[idx].std(0) + 1e-6
        flags = [WD.FEATS.index(f) for f in ["dow0", "dow1", "dow2", "dow3", "dow4"]]
        self.xm[flags] = 0; self.xs[flags] = 1
        self.ym = np.nanmean(Y[idx], 0); self.ys = np.nanstd(Y[idx], 0) + 1e-6

    def x(self, X):
        return ((X - self.xm) / self.xs).astype(np.float32)

    def y(self, Y):
        return ((Y - self.ym) / self.ys).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ["xm", "xs", "ym", "ys"]}


def windows(P, rows, L, norm):
    Xn = norm.x(P["X"])
    W = np.stack([Xn[i - L + 1:i + 1] for i in rows])
    return (torch.tensor(W), torch.tensor(P["U"][rows]), torch.tensor(P["ST"][rows]),
            torch.tensor(norm.y(P["Y"][rows])))


def fold_rows(P, L, kind, year=None):
    D = P["D"].astype("datetime64[D]")
    LAST = P["LAST"].astype("datetime64[D]")
    ok = np.arange(len(D)) >= L - 1
    if kind == "dev":
        start = np.datetime64(f"{year}-07-01"); end = np.datetime64(f"{year + 1}-06-30")
    elif kind == "val":
        start = np.datetime64(WD.VAL_START.date()); end = np.datetime64(WD.VAL_END.date())
    tr = np.where(ok & (LAST < start) & (D < start))[0]
    te = np.where(ok & (D >= start) & (D <= end))[0]
    if kind == "dev":   # LRT-15: no DEV test row may have a label that ends after DEV_END
        te = te[LAST[te] <= np.datetime64(WD.DEV_END.date())]
    assert P["LAST"][tr].astype("datetime64[D]").max() < start
    return tr, te


def predict(m, P, rows, norm, L, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            Xw, U, ST, _ = windows(P, rows[i:i + bs], L, norm)
            out.append(m(Xw, U, ST, sample=False).numpy())
    return np.concatenate(out)          # standardized units


def pinball_np(q, y):
    """q [N, T, NQ], y [N, T] -> per-sample per-target pinball [N, T] (NaN where y missing)."""
    e = y[..., None] - q
    l = np.maximum(QL * e, (QL - 1) * e).mean(-1)
    return np.where(np.isfinite(y), l, np.nan)
