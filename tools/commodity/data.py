"""
Commodity data: raw files -> clean series, manifest (sha256 + download date), split constants, and the
training-window helper that can never emit a target past TRAIN_END.

    .venv/bin/python tools/commodity/data.py build      # parse data/commodity/raw/* -> data/commodity/*.csv + manifest

Series (all daily series are FRED, business days, missing days dropped, no forward fill):
    wti     DCOILWTICO   WTI Cushing spot, USD/bbl         (primary)
    brent   DCOILBRENTEU Brent Europe spot, USD/bbl        (secondary)
    hh      DHHNGSP      Henry Hub natural gas spot, USD/MMBtu (secondary)
    inv     WCESTUS1     US commercial crude stocks excl. SPR, thousand bbl, weekly (EIA; not on FRED)

Splits (docs/commodity_prereg.md):
    TRAIN/DEV  <= 2025-06-30   (DEV backtest origins 2019-01-01..2025-06-30, model selection)
    VALIDATION 2025-07-01..2025-10-31 (confirm the frozen candidate; debugging)
    TEST       >= 2025-11-01   (run exactly once, all models together)
"""
import os
import re
import sys
import json
import hashlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DDIR = os.path.join(ROOT, "data", "commodity")
RAW = os.path.join(DDIR, "raw")

TRAIN_END = pd.Timestamp("2025-06-30")
DEV_START = pd.Timestamp("2019-01-01")
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

SERIES = {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU", "hh": "DHHNGSP", "inv": "WCESTUS1"}
DAILY = ("wti", "brent", "hh")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _parse_fred(path):
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")       # FRED marks holidays with "." or empty
    return df.dropna().reset_index(drop=True)


_MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def _parse_eia_leaf(path):
    """EIA LeafHandler weekly table: a B6 cell 'YYYY-Mon' starts a row, then (B5 'MM/DD', B3 'value') pairs."""
    html = open(path, encoding="latin-1").read()
    cells = re.findall(r"<td class='(B6|B5|B3)'>([^<]*)</td>", html)
    rows, year, mon, md = [], None, None, None
    for cls, txt in cells:
        t = txt.replace("&nbsp;", "").strip()
        if cls == "B6":
            year, mon = int(t[:4]), _MON[t[5:8]]
        elif cls == "B5":
            md = t or None
        elif cls == "B3" and md and t:
            m, d = int(md[:2]), int(md[3:5])
            y = year + 1 if m < mon else year                          # defensive; never triggered in practice
            rows.append((pd.Timestamp(y, m, d), float(t.replace(",", ""))))
            md = None
    df = pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates("date").sort_values("date")
    return df.reset_index(drop=True)


def build(download_date):
    man = {"download_date": download_date, "files": {}, "series": {}}
    for name, sid in SERIES.items():
        if name == "inv":
            raw = os.path.join(RAW, "WCESTUS1_leaf.html")
            src = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W"
            df = _parse_eia_leaf(raw)
        else:
            raw = os.path.join(RAW, f"{sid}.csv")
            src = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            df = _parse_fred(raw)
        out = os.path.join(DDIR, f"{name}.csv")
        df.to_csv(out, index=False, date_format="%Y-%m-%d")
        man["files"][os.path.relpath(raw, ROOT)] = {"sha256": sha256(raw), "source": src}
        man["series"][name] = {"id": sid, "csv": os.path.relpath(out, ROOT), "sha256": sha256(out), "n": len(df),
                               "first": str(df.date.iloc[0].date()), "last": str(df.date.iloc[-1].date()),
                               "n_nonpositive": int((df.value <= 0).sum()),
                               "n_test": int((df.date >= TEST_START).sum()),
                               "n_val": int(((df.date >= VAL_START) & (df.date <= VAL_END)).sum())}
    json.dump(man, open(os.path.join(DDIR, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


def load(name):
    df = pd.read_csv(os.path.join(DDIR, f"{name}.csv"), parse_dates=["date"])
    return df.date.to_numpy(), df.value.to_numpy(dtype=np.float64)


def train_windows(name, context, horizon, stride=1, end=TRAIN_END, start=None, log=False):
    """Training windows from data up to `end` (default 2025-06-30) ONLY.

    Returns (X [n, context], Y [n, horizon], origin_dates [n]); X[i] are the `context` observations strictly before
    origin i, Y[i] the `horizon` observations from origin i on. Hard assertion: no target date is after TRAIN_END,
    whatever `end` the caller passes (end > TRAIN_END raises). Values are prices (log prices if log=True).
    """
    end = pd.Timestamp(end)
    if end > TRAIN_END:
        raise AssertionError(f"train_windows: end={end.date()} is after TRAIN_END={TRAIN_END.date()}")
    d, v = load(name)
    keep = d <= np.datetime64(end)
    d, v = d[keep], v[keep]
    if log:
        if (v <= 0).any():                         # WTI 2020-04-20 = -37.63: the caller must pick a start after it
            raise ValueError("train_windows(log=True): non-positive values in range; pass start= after them")
        v = np.log(v)
    if start is not None:
        s = np.searchsorted(d, np.datetime64(pd.Timestamp(start)))
        d, v = d[s:], v[s:]
    idx = np.arange(context, len(v) - horizon + 1, stride)
    X = np.stack([v[i - context:i] for i in idx]) if len(idx) else np.zeros((0, context))
    Y = np.stack([v[i:i + horizon] for i in idx]) if len(idx) else np.zeros((0, horizon))
    od = d[idx] if len(idx) else d[:0]
    if len(idx):
        last_target = d[idx[-1] + horizon - 1]
        assert last_target <= np.datetime64(TRAIN_END), f"leak: target date {last_target} > {TRAIN_END.date()}"
    return X, Y, od


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else "2026-10-31")
