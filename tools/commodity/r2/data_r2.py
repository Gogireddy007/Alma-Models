"""Round-2 commodity data loader with hard date guards (docs/commodity_r2_prereg.md).

Universe: daily FRED refined-product / NGL spot series (NOT WTI, Brent, Henry Hub, whose round-1 TEST is being
consumed). Each series is requested from FRED with coed=<cap> in the request itself, re-cut, cached per split under
runs/commodity/r2/data/<split>/, and every loader asserts on the max date it returns.

    split "dev": cap 2025-06-30 (training, all search/selection)
    split "val": cap 2025-10-31 (used ONCE for the frozen choice)
    TEST (>= 2025-11-01) is never requested by this module. tools/commodity/r2/test_r2.py has its own loader,
    to be run once, later, by someone else.
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "runs", "commodity", "r2", "data")
DEV_END = pd.Timestamp("2025-06-30")
VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-10-31")
CAPS = {"dev": DEV_END, "val": VAL_END}
TRAIN_CUT = pd.Timestamp("2015-01-01")      # search models: every training target strictly before this date
BT_START = pd.Timestamp("2015-01-01")       # DEV backtest origins 2015-01-01 .. 2025-06-30

SERIES = ["DGASNYH", "DHOILNYH", "DJFUELUSGULF", "DPROPANEMBTX", "DGASUSGULF", "DDFUELNYH", "DDFUELUSGULF",
          "DDFUELLA"]
NAMES = {"DGASNYH": "NY Harbor conventional gasoline", "DHOILNYH": "NY Harbor No.2 heating oil",
         "DJFUELUSGULF": "US Gulf Coast kerosene-type jet fuel", "DPROPANEMBTX": "Mont Belvieu TX propane",
         "DGASUSGULF": "US Gulf Coast conventional gasoline", "DDFUELNYH": "NY Harbor ULSD",
         "DDFUELUSGULF": "US Gulf Coast ULSD", "DDFUELLA": "Los Angeles ULSD (CARB)"}
# pooled panel for the PRIMARY target (vol direction) and the two SECONDARY (CRPS) series (manager, 2026-10-31)
POOL5 = ["DHOILNYH", "DJFUELUSGULF", "DGASNYH", "DGASUSGULF", "DPROPANEMBTX"]
SECONDARY = ["DHOILNYH", "DJFUELUSGULF"]
GASOLINE = ["DGASNYH", "DGASUSGULF"]
EXO = ["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"]          # INPUT covariates only, never targets, dates <= 2025-10-31
FORBIDDEN = {"DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"}     # round-1 TEST series: never loaded by round 2


def _guard(df, split):
    cap = CAPS[split]
    mx = df.index.max()
    assert mx <= cap, f"DATA GUARD: {split} data reaches {mx.date()} > {cap.date()}"
    assert mx <= VAL_END, "DATA GUARD: nothing after 2025-10-31 may be loaded in round 2"
    return df


def fetch(sid, split="dev", exo=False):
    assert split in CAPS, split
    assert exo or sid not in FORBIDDEN, f"{sid} is a round-1 TEST series; round 2 uses it only as an exo input"
    if exo:
        assert sid in EXO
    cap = CAPS[split]
    d = os.path.join(DATA, split + ("_exo" if exo else ""))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.csv")
    if not os.path.exists(path):
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1980-01-01"
               f"&coed={cap.strftime('%Y-%m-%d')}")
        raw = urllib.request.urlopen(url, timeout=60).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", sid]
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= cap]
        df.to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df[sid] = pd.to_numeric(df[sid], errors="coerce")
    return _guard(df, split)


def load_series(sid, split="dev"):
    """(dates [T] datetime64, log price [T]) with missing days dropped (no fill). Non-positive prices: none in this
    universe (asserted)."""
    df = fetch(sid, split).dropna()
    v = df[sid].values.astype(np.float64)
    assert (v > 0).all(), f"{sid}: non-positive price"
    return df.index.values, np.log(v)


def load_all(split="dev", series=None):
    return {s: load_series(s, split) for s in (series or SERIES)}


def load_exo(split="dev"):
    """WTI / Brent / Henry Hub as INPUT covariates (capped like every split, never past 2025-10-31). Non-positive
    prints (WTI 2020-04-20, -36.98) are DROPPED (treated as a missing day), pre-registered. Returns
    {sid: (dates, log price)} on each series' own calendar."""
    out = {}
    for s in EXO:
        df = fetch(s, split, exo=True).dropna()
        df = df[df[s] > 0]
        out[s] = (df.index.values, np.log(df[s].values.astype(np.float64)))
    return out
