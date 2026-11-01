"""
Alma-Fin data layer (docs/fin_prereg.md sections 1-3 are binding).

    nice -n 10 .venv/bin/python tools/fin/data.py build        # raw zips/json -> processed files + build report

Processed files (all in data/fin/, sha256-pinned in run_eval.DATA_PINS):
  crypto_hourly.parquet   one row per (symbol, UTC hour): open, high, low, close, volume, quote_volume, count,
                          taker_buy_volume, taker_buy_quote_volume, n5 (number of 5-min bars present), rv5 (sum of
                          squared 5-min log returns inside the hour), funding (sum of funding rates SETTLED at
                          times t with hour_start < t <= hour_end, i.e. known at the hour's close).
  crypto_daily.parquet    one row per (symbol, UTC day D): ohlc (00:00-24:00), quote_volume, taker_buy ratio, n5,
                          rv (label RV, sum of squared 5-min log returns of the bars opening 00:05..23:55; the first
                          return is close(00:05 bar) / close(00:00 bar)), fill (the 00:05-00:10 bar's (O+H+L+C)/4),
                          funding_00 / funding_08 / funding_16 / funding_sum (rates settled during the day),
                          last_bar (bool: last day with data = delisting/last snapshot day).
  etf_daily.parquet       one row per (symbol, NYSE session): open, high, low, close, adjclose, volume, rv
                          (Garman-Klass range variance of the session), fill = the session's OPEN (the decision for
                          session D is taken at the close of session D-1).
  *_SEALED.parquet        the same tables for days >= 2025-11-01 (TEST). They are built and hashed, never read
                          except by run_eval strict TEST mode (load(..., allow_test=True)); nothing is printed.

Gaps: a missing 5-min bar is not filled; RV of a day with fewer than MIN_BARS (274 of the 287 label returns, ~95%) is NaN (label
and RV feature excluded; listed in build_report.json). Returns across a gap span the gap. Outliers: 5-min |log
return| > 0.25 are counted and listed (kept unchanged; labels are never clipped). Delistings: a symbol's last day
is flagged; the harness closes any position at that day's last close (docs/fin_prereg.md section 3).
"""
import io
import os
import sys
import json
import glob
import zipfile
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "fin")
RAW = os.path.join(DATA, "raw")
UNIVERSE = os.path.join(DATA, "universe.json")
FILES = {k: os.path.join(DATA, f"{k}.parquet") for k in
         ("crypto_hourly", "crypto_daily", "etf_daily", "crypto_hourly_SEALED", "crypto_daily_SEALED",
          "etf_daily_SEALED")}
REPORT = os.path.join(DATA, "build_report.json")

TRAIN_END = pd.Timestamp("2025-06-30")          # last DEV decision day / last trainable label day
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-31")
TEST_START, TEST_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-30")
MIN_BARS = 274
OUTLIER = 0.25
ETFS = ["SPY", "QQQ", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def universe():
    u = json.load(open(UNIVERSE))
    return u["train"], u["heldout"]


# --------------------------------------------------------------------------- raw readers
def _read_kline_zip(path):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    head = raw.split(b"\n", 1)[0]
    df = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith(b"open_time") else None)
    df = df.iloc[:, :12]
    df.columns = COLS
    ot = df["open_time"].astype("int64")
    df["t"] = pd.to_datetime(ot, unit="us" if ot.max() > 10 ** 14 else "ms", utc=True).dt.tz_localize(None)
    return df.drop(columns=["open_time", "close_time", "ignore"])


def read_5m(sym):
    fs = sorted(glob.glob(os.path.join(RAW, "um", "klines_5m", sym, "*.zip")))
    if not fs:
        raise FileNotFoundError(sym)
    df = pd.concat([_read_kline_zip(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
    assert (df["t"].dt.minute % 5 == 0).all() and (df["t"].dt.second == 0).all(), sym
    return df


def read_funding(sym):
    fs = sorted(glob.glob(os.path.join(RAW, "um", "fundingRate", sym, "*.zip")))
    out = []
    for f in fs:
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0])
        head = raw.split(b"\n", 1)[0]
        d = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith(b"calc_time") else None)
        d = d.iloc[:, [0, d.shape[1] - 1]]
        d.columns = ["calc_time", "rate"]
        out.append(d)
    if not out:
        return pd.DataFrame({"t": pd.Series([], dtype="datetime64[ns]"), "rate": []})
    d = pd.concat(out, ignore_index=True)
    ct = d["calc_time"].astype("int64")
    # settlement time: calc_time rounded DOWN to the minute (Binance stamps a few ms after the settlement instant)
    t = pd.to_datetime(ct, unit="us" if ct.max() > 10 ** 14 else "ms", utc=True).dt.tz_localize(None).dt.floor("min")
    return pd.DataFrame({"t": t, "rate": d["rate"].astype(float)}).drop_duplicates("t").sort_values("t")


# --------------------------------------------------------------------------- crypto build
def build_symbol(sym, rep):
    m = read_5m(sym)
    lc = np.log(m["close"].to_numpy(float))
    r = np.diff(lc, prepend=np.nan)                   # return of bar k = log C_k - log C_{k-1} (previous PRESENT bar)
    m["r2"] = r ** 2
    big = np.abs(r) > OUTLIER
    rep["outliers_5m"][sym] = [str(x) for x in m.loc[big, "t"].tolist()]
    m["hour"] = m["t"].dt.floor("h")
    m["day"] = m["t"].dt.floor("D")
    agg = dict(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
               volume=("volume", "sum"), quote_volume=("quote_volume", "sum"), count=("count", "sum"),
               taker_buy_volume=("taker_buy_volume", "sum"), taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
               n5=("close", "size"), rv5=("r2", "sum"))
    h = m.groupby("hour").agg(**agg).reset_index().rename(columns={"hour": "t"})
    fu = read_funding(sym)
    # funding settled in (hour_start, hour_end] -> known at hour close. Settlements at exactly hh:00 belong to the
    # hour ENDING at hh:00 (hour starting hh-1).
    fu_h = fu.assign(t=(fu["t"] - pd.Timedelta(seconds=1)).dt.floor("h")).groupby("t")["rate"].sum()
    h["funding"] = h["t"].map(fu_h).fillna(0.0)
    h.insert(0, "symbol", sym)
    # daily
    m["min"] = (m["t"] - m["day"]).dt.total_seconds() // 60
    lab = m[m["min"] >= 5]                             # bars opening 00:05..23:55
    d = m.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), quote_volume=("quote_volume", "sum"),
                             taker_buy_quote_volume=("taker_buy_quote_volume", "sum"), n5=("close", "size"))
    d["n_lab"] = lab.groupby("day")["r2"].size()
    d["rv"] = lab.groupby("day")["r2"].sum(min_count=1)
    first_ret_ok = m[m["min"] == 5].set_index("day")["r2"].notna()
    d.loc[d["n_lab"].fillna(0) < MIN_BARS, "rv"] = np.nan
    fb = m[m["min"] == 5].set_index("day")
    d["fill"] = (fb["open"] + fb["high"] + fb["low"] + fb["close"]) / 4.0
    fu["day"] = (fu["t"] - pd.Timedelta(seconds=1)).dt.floor("D")   # 00:00 settlement belongs to the previous day
    # ... except the settlement at the TEST boundary (2025-11-01 00:00): it is a TEST-period value; VAL positions are
    # marked at the 2025-10-31 close, so it is not assigned to the open 2025-10-31 row
    fu.loc[fu["t"] == TEST_START, "day"] = TEST_START
    fu["slot"] = ((fu["t"] - fu["day"]).dt.total_seconds() // 3600).astype(int)
    d["funding_sum"] = fu.groupby("day")["rate"].sum()
    for slot, nm in ((8, "funding_08"), (16, "funding_16"), (24, "funding_24")):
        d[nm] = fu[fu["slot"] == slot].groupby("day")["rate"].sum()
    d["funding_n"] = fu.groupby("day")["rate"].size()
    d[["funding_sum", "funding_n"]] = d[["funding_sum", "funding_n"]].fillna(0.0)
    d = d.reset_index().rename(columns={"day": "t"})
    d["last_bar"] = False
    d.loc[d.index[-1], "last_bar"] = True
    d.insert(0, "symbol", sym)
    rep["gaps"][sym] = {"days": int(len(d)), "first_day": str(d["t"].min().date()), "last_day": str(d["t"].max().date()),
                        "days_rv_nan": [str(x.date()) for x in d.loc[d["rv"].isna(), "t"]][:400],
                        "n_days_rv_nan": int(d["rv"].isna().sum()), "n_missing_5m_bars": int(288 * len(d) - len(m)),
                        "days_without_fill_bar": int(d["fill"].isna().sum()), "funding_events": int(len(fu))}
    del first_ret_ok
    return h, d


# --------------------------------------------------------------------------- ETF build
def build_etf(sym):
    j = json.load(open(os.path.join(RAW, "etf", f"{sym}.json")))["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    adj = j["indicators"].get("adjclose", [{}])[0].get("adjclose")
    tz = j["meta"].get("exchangeTimezoneName", "America/New_York")
    t = pd.to_datetime(np.asarray(j["timestamp"], "int64"), unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    d = pd.DataFrame({"t": t, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                      "volume": q["volume"], "adjclose": adj if adj else q["close"]}).astype(
        {"open": float, "high": float, "low": float, "close": float, "adjclose": float})
    d = d.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    lo, lh, lc_ = np.log(d["open"]), np.log(d["high"]), np.log(d["low"])
    lcl = np.log(d["close"])
    d["rv"] = 0.5 * (lh - lc_) ** 2 - (2 * np.log(2) - 1) * (lcl - lo) ** 2          # Garman-Klass
    d["fill"] = d["open"]                         # decision for session D is taken at the close of D-1; fill = open of D
    d["adj_factor"] = d["adjclose"] / d["close"]
    d["last_bar"] = False
    d.insert(0, "symbol", sym)
    return d


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build():
    train, held = universe()
    rep = {"outliers_5m": {}, "gaps": {}, "min_bars": MIN_BARS, "outlier_threshold": OUTLIER}
    H, Dd = [], []
    for s in train + held:
        h, d = build_symbol(s, rep)
        H.append(h)
        Dd.append(d)
        print(f"{s}: {len(d)} days, rv NaN {rep['gaps'][s]['n_days_rv_nan']}, "
              f"outliers {len(rep['outliers_5m'][s])}", flush=True)
    H = pd.concat(H, ignore_index=True)
    Dd = pd.concat(Dd, ignore_index=True)
    E = pd.concat([build_etf(s) for s in ETFS], ignore_index=True) if os.path.isdir(os.path.join(RAW, "etf")) else None
    cut = TEST_START
    # a day-D row is TEST if D >= 2025-11-01; an hourly row if its hour starts >= 2025-11-01 00:00. Every field of a
    # day-D row lies inside day D (crypto fill = 00:05 bar of D; ETF fill = open of D).
    parts = {"crypto_hourly": H, "crypto_daily": Dd}
    if E is not None:
        E.loc[E["t"] == E["t"].max(), "last_bar"] = False
        parts["etf_daily"] = E
    for k, df in parts.items():
        open_ = df[df["t"] < cut].copy()
        sealed = df[df["t"] >= cut]
        if k in ("crypto_daily", "etf_daily"):
            # last pre-TEST row: its 'fill' (00:05 bar of the NEXT day / next open) lies in TEST -> sealed field
            last = open_.groupby("symbol")["t"].transform("max") == open_["t"]
            nxt_in_test = open_["symbol"].isin(sealed["symbol"].unique())
            open_.loc[last & ~nxt_in_test, "last_bar"] = True
            open_.loc[last & nxt_in_test, "last_bar"] = False
        open_.to_parquet(FILES[k], index=False)
        sealed.to_parquet(FILES[k + "_SEALED"], index=False)
        rep.setdefault("rows", {})[k] = int(len(open_))
        rep.setdefault("rows_sealed", {})[k] = int(len(sealed))
    rep["sha256"] = {os.path.relpath(f, ROOT): sha(f) for f in FILES.values() if os.path.exists(f)}
    rep["cross_check"] = "TEST rows are counted only; no TEST value is printed or summarised."
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(json.dumps(rep["rows"]), json.dumps(rep["rows_sealed"]))
    for k, v in rep["sha256"].items():
        print(v, k)


# --------------------------------------------------------------------------- loaders (the only API models get)
def load(kind="crypto_daily", allow_test=False):
    """Open (pre-TEST) table; with allow_test=True (run_eval strict TEST mode only) the sealed TEST rows are appended.
    Asserts the open table holds nothing dated >= 2025-11-01."""
    df = pd.read_parquet(FILES[kind])
    assert df["t"].max() < TEST_START, f"{kind}: open file holds TEST-period rows"
    if allow_test:
        if os.environ.get("ALMA_FIN_TEST_GO") != "1":
            raise PermissionError("sealed TEST data: only run_eval.py --split test --i-have-auditor-go may load it")
        df = pd.concat([df, pd.read_parquet(FILES[kind + "_SEALED"])], ignore_index=True)
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
