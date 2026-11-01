"""Commodity world-model search: data loading with hard date guards (docs/commodity_search.md).

DEV  = every observation dated <= 2025-06-30 (training + model selection).
VAL  = 2025-07-01 .. 2025-10-31 (confirm ONE frozen candidate, once).
Nothing dated after 2025-10-31 is ever requested from FRED (the request itself carries coed=), and every loader
asserts on the max date of what it returns. TEST belongs to the commodity lead.

FRED series are fetched with fredgraph.csv?id=<ID>&cosd=1980-01-01&coed=<cap> and cached per split under
runs/commodity/search/data/<split>/<ID>.csv. The DEV cache and the VAL cache are separate files, so a DEV run
cannot read a VAL value even by accident.
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "runs", "commodity", "search", "data")
DEV_END = pd.Timestamp("2025-06-30")
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-10-31")
CAPS = {"dev": DEV_END, "val": VAL_END}

# daily FRED commodity series (WTI first: the target). Products / gas / propane carry crack and term information.
SERIES = ["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "DGASNYH", "DGASUSGULF", "DHOILNYH", "DJFUELUSGULF",
          "DPROPANEMBTX", "DDFUELNYH", "DDFUELUSGULF", "DDFUELLA"]
TARGET = "DCOILWTICO"


def _guard(df, split):
    cap = CAPS[split]
    mx = df.index.max()
    assert mx <= cap, f"DATA GUARD: {split} data reaches {mx.date()} > {cap.date()}"
    assert mx <= VAL_END, "DATA GUARD: nothing after 2025-10-31 may be loaded"
    return df


def fetch(sid, split="dev"):
    assert split in CAPS
    cap = CAPS[split]
    d = os.path.join(DATA, split)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.csv")
    if not os.path.exists(path):
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1980-01-01"
               f"&coed={cap.strftime('%Y-%m-%d')}")
        raw = urllib.request.urlopen(url, timeout=60).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", sid]
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= cap]                      # belt and braces: FRED honours coed, we re-cut anyway
        df.to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df[sid] = pd.to_numeric(df[sid], errors="coerce")
    return _guard(df, split)


def load_panel(split="dev", series=None):
    """Daily panel on WTI trading days (rows where WTI is observed), other series forward-filled from their own
    past only (ffill never looks ahead). Returns DataFrame of PRICES, index = date."""
    series = series or SERIES
    frames = []
    for s in series:
        try:
            frames.append(fetch(s, split))
        except Exception as e:  # a series FRED does not serve is skipped and reported
            print(f"[wm_data] skip {s}: {e}")
    df = pd.concat(frames, axis=1).sort_index()
    keep = df[TARGET].notna() & (df.index >= pd.Timestamp("1986-01-02"))   # WTI observed (before any fill)
    df = df.ffill(limit=10)                             # causal fill of the other series (<= 10 days stale)
    df = df[keep]
    return _guard(df, split)


def log_panel(split="dev", series=None, floor=1.0):
    """Log prices. WTI printed -37 on 2020-04-20; prices are floored at `floor` before the log (affects only
    that week in training data; never the 2023+ backtest). Returns (dates, names, X[T, N]) with NaN kept."""
    df = load_panel(split, series)
    X = np.log(np.maximum(df.values.astype(np.float64), floor))
    return df.index, list(df.columns), X
