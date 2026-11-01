"""Red-team helpers (leak auditor; files in tools/fin/redteam/ are the auditor's only). DEV only.

perturbed_load(cut=..., heldout=..., mode=...) monkeypatches data.load so that every Panel built afterwards sees a
corrupted copy of the open files:
  - rows dated > cut (daily: t > cut; hourly: hour start >= cut + 1 day) are corrupted, and/or
  - every row of the held-out coins is corrupted.
mode 'mult' multiplies numeric fields by U(0.3, 3); mode 'rand' replaces them by random values (incl. NaN <-> number).
Nothing is ever loaded with allow_test=True.
"""
import os
import sys
import contextlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIN))
sys.path.insert(0, FIN)
import data as DT        # noqa: E402
import harness as H      # noqa: E402

SKIP = {"symbol", "t", "last_bar"}
_ORIG_LOAD = DT.load


def corrupt(df, rows, rng, mode="mult"):
    df = df.copy()
    rows = np.asarray(rows, bool)
    if not rows.any():
        return df
    for c in df.columns:
        if c in SKIP or not np.issubdtype(df[c].dtype, np.number):
            continue
        x = df[c].to_numpy(np.float64, copy=True)
        if mode == "mult":
            x[rows] = x[rows] * rng.uniform(0.3, 3.0, rows.sum())
        else:
            v = rng.lognormal(0, 1, rows.sum()) * np.nanmedian(np.abs(x)) if np.isfinite(x).any() else rng.random(rows.sum())
            v[rng.random(rows.sum()) < 0.05] = np.nan
            x[rows] = v
        df[c] = x.astype(df[c].dtype) if df[c].dtype.kind == "f" else x
    return df


@contextlib.contextmanager
def perturbed_load(cut=None, heldout=False, mode="mult", seed=0):
    rng = np.random.default_rng(seed)
    _, ho = DT.universe()

    def load(kind="crypto_daily", allow_test=False):
        assert not allow_test, "red team never loads sealed rows"
        df = _ORIG_LOAD(kind, False)
        rows = np.zeros(len(df), bool)
        if cut is not None:
            c = pd.Timestamp(cut)
            if kind == "crypto_hourly":
                rows |= (df["t"] >= c + pd.Timedelta(days=1)).to_numpy()
            else:
                rows |= (df["t"] > c).to_numpy()
        if heldout and kind.startswith("crypto"):
            rows |= df["symbol"].isin(ho).to_numpy()
        return corrupt(df, rows, rng, mode)
    DT.load = load
    try:
        yield
    finally:
        DT.load = _ORIG_LOAD


def out_of(res):
    if "pos" in res:
        return res["pos"]
    parts = [res[k].reshape(len(res["days"]), -1) for k in ("p_up", "q", "p_ret_up") if k in res]
    return np.concatenate(parts, 1)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)
