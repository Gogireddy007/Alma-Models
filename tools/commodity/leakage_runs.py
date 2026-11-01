"""Follow-up to leakage_scan.py: for TRAIN datasets with scale-free 6-gram hits, the longest run of CONSECUTIVE
matching grams per dataset (a copied segment of length L gives a run of about L-6 grams; isolated hits are
coincidences on a coarse price lattice), with an example. Vectorised: grams hashed to int64.

    nice -n 10 .venv/bin/python tools/commodity/leakage_runs.py monash_fred_md monash_traffic monash_weather dominick
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "commodity"))
import data as D   # noqa: E402

N = 6
MUL = np.array([(1000003 ** k) % (1 << 61) for k in range(N)], dtype=np.int64)   # products wrap mod 2^64, consistently


def ints(y, kind):
    y = np.asarray(y, np.float64)
    y = y[np.isfinite(y)]
    if kind == "ratio":
        y = y[y > 0]
        if len(y) < N + 1:
            return None
        return np.round(np.diff(np.log(y)) * 1000).astype(np.int64)
    return np.round(y * 100).astype(np.int64)


def hashes(r):
    if r is None or len(r) < N:
        return None, None
    w = np.lib.stride_tricks.sliding_window_view(r, N)
    with np.errstate(over="ignore"):
        h = (w * MUL).sum(1)
    return h, w.max(1) != w.min(1)


def refs():
    out = {}
    for name in ("wti", "brent", "hh", "inv"):
        d, v = D.load(name)
        s = pd.Series(v, index=pd.DatetimeIndex(d))
        series = {f"{name}_raw": s.values}
        if name != "inv":
            series[f"{name}_monthly"] = s.resample("MS").mean().dropna().values
            series[f"{name}_weekly"] = s.resample("W").mean().dropna().values
        for k, x in series.items():
            for kind in ("ratio", "level"):
                h, var = hashes(ints(x, kind))
                out[(k, kind)] = np.unique(h[var])
    return out


def longest(mask):
    if not mask.any():
        return 0, -1
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    dd = np.diff(m)
    st, en = np.where(dd == 1)[0], np.where(dd == -1)[0]
    j = np.argmax(en - st)
    return int(en[j] - st[j]), int(st[j])


R = refs()
for name in sys.argv[1:]:
    df = pd.read_parquet(os.path.join(ROOT, "data", "raw", f"{name}.parquet"))
    cols = ["target"] if "target" in df.columns else [c for c in df.columns if c not in ("id", "timestamp") and df[c].dtype == object]
    best = {}
    for c in cols:
        for si, v in enumerate(df[c]):
            for kind in ("ratio", "level"):
                r = ints(v, kind)
                h, var = hashes(r)
                if h is None:
                    continue
                for key, ref in R.items():
                    if key[1] != kind:
                        continue
                    mask = np.isin(h, ref) & var
                    L, st = longest(mask)
                    if L > best.get(key, (0,))[0]:
                        best[key] = (L, si, (r[st:st + N] / (1000 if kind == "ratio" else 100)).tolist())
    for key, (L, si, ex) in sorted(best.items()):
        print(f"{name:16s} vs {key[0]:14s} {key[1]:5s}: longest run of consecutive matching 6-grams = {L:3d} "
              f"(series #{si}); first gram {np.round(ex, 3).tolist()}", flush=True)
    del df
