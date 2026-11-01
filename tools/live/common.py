"""
Alma-Live Gas: shared paths, clocks, exchange calendar, hashing, the append-only record files and the data guard.

Record schema (every point-in-time file under data/live/pit/<source>.csv, one row per (series, obs, vintage)):
    source        source id (storage, px, cpc_fcst, cpc_obs, om_fcst, cot, edgar_filing, edgar_fact, eia_api)
    series        series id within the source (e.g. 'l48', 'NG=F.open', 'hdd', 'NE.hdd.d3')
    obs           ISO date (or datetime) of the period the value describes (forecasts: the valid day)
    value         float
    published_at  UTC ISO datetime when the value became knowable (from the source's release schedule; conservative)
    fetched_at    UTC ISO datetime when this pipeline obtained it (backfill: the fetch time of the snapshot)
    vintage       'first_release' | 'current_backfill' | 'revised' | 'snapshot' | 'issue:<id>' ...
    extra         short JSON / text (form type, accession, issue id, ...) or ''
Invariants (asserted on write, tested in test_infoset.py): published_at <= fetched_at; files are append-only.
"""
import os
import io
import sys
import csv
import json
import time
import hashlib
import datetime as dt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DDIR = os.path.join(ROOT, "data", "live")
RAW = os.path.join(DDIR, "raw")
PIT = os.path.join(DDIR, "pit")
MANIFEST = os.path.join(DDIR, "manifest.json")
UPDATE_LOG = os.path.join(DDIR, "update_log.jsonl")
RUNS = os.path.join(ROOT, "runs", "live")
GAS = os.path.join(ROOT, "data", "gas")            # read-only reuse of the Alma-Gas raw archive
ET = "America/New_York"
COLS = ["source", "series", "obs", "value", "published_at", "fetched_at", "vintage", "extra"]

# splits (decision days; docs/live_prereg.md)
DEV_END = pd.Timestamp("2024-06-30")
VAL_START, VAL_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-10-31")
EXPOSED_START = pd.Timestamp("2025-11-01")          # prior project's TEST exposed HH prices from here: NOT clean


def split_of(day):
    day = pd.Timestamp(day)
    if day <= DEV_END:
        return "dev"
    if day <= VAL_END:
        return "val"
    return "exposed"


# --------------------------------------------------------------------------- clocks
def utcnow():
    return pd.Timestamp.now(tz="UTC").floor("s")


def iso(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("naive timestamp")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def et(day, hh, mm=0):
    """Wall-clock time hh:mm US/Eastern on calendar day `day` (DST-aware) as a UTC Timestamp."""
    d = pd.Timestamp(day)
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm).tz_localize(ET).tz_convert("UTC")


# Manager decision (2026-11-01, after docs/live_gas_research.md): decide at 15:30 ET with data published by then,
# fill at the 16:00 ET close of the same session plus slippage; primary instrument UNG (BOIL secondary, NG futures
# research-only until the forward curve snapshotter has history). SINGLE SOURCE OF TRUTH for every component.
DECISION_HH, DECISION_MM = 15, 30
FILL_HH, FILL_MM = 16, 0


def decision_time(day):
    """The daily decision time: 15:30 ET on trading day `day` (UTC Timestamp)."""
    return et(day, DECISION_HH, DECISION_MM)


def fill_time(day):
    """Fill time of the decision taken on `day`: the 16:00 ET close of the same session (strictly after decision)."""
    return et(day, FILL_HH, FILL_MM)


def trading_days(start="2010-01-01", end="2026-12-31"):
    """Exchange trading days (NYSE/NYMEX full sessions) in [start, end]."""
    return [d for d in pd.date_range(start, end, freq="B") if d not in HOLIDAYS]


def decision_times(start="2010-01-01", end="2026-12-31"):
    """pd.Series: trading day -> decision time (UTC)."""
    days = trading_days(start, end)
    return pd.Series([decision_time(d) for d in days], index=pd.DatetimeIndex(days))


# --------------------------------------------------------------------------- exchange calendar (NYMEX / NYSE-like)
def _good_friday(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, month, day) - pd.Timedelta(days=2)


def _nth_wd(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    d += pd.Timedelta(days=(wd - d.weekday()) % 7)
    return d + pd.Timedelta(days=7 * (n - 1))


def _last_wd(y, m, wd):
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - wd) % 7)


def _observed(d, sat_to_fri=True):
    if d.weekday() == 5:
        return d - pd.Timedelta(days=1) if sat_to_fri else None
    if d.weekday() == 6:
        return d + pd.Timedelta(days=1)
    return d


def _holidays():
    """NYSE/NYMEX full-day closures (rule-based) + known special closures."""
    out = set()
    for y in range(1995, 2036):
        nyd = _observed(pd.Timestamp(y, 1, 1), sat_to_fri=False)   # NYSE does not close Fri Dec 31
        days = [nyd, _nth_wd(y, 2, 0, 3), _good_friday(y), _last_wd(y, 5, 0),
                _observed(pd.Timestamp(y, 7, 4)), _nth_wd(y, 9, 0, 1), _nth_wd(y, 11, 3, 4),
                _observed(pd.Timestamp(y, 12, 25))]
        if y >= 1998:
            days.append(_nth_wd(y, 1, 0, 3))
        if y >= 2022:
            days.append(_observed(pd.Timestamp(y, 6, 19)))
        out |= {d for d in days if d is not None}
    out |= {pd.Timestamp(x) for x in ("2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11",
                                      "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09")}
    return out


HOLIDAYS = _holidays()


def is_bday(d):
    d = pd.Timestamp(d).normalize()
    return d.weekday() < 5 and d not in HOLIDAYS


def bdays(start, end):
    return [d for d in pd.date_range(start, end, freq="D") if is_bday(d)]


def add_bdays(d, n):
    d = pd.Timestamp(d).normalize()
    step = 1 if n >= 0 else -1
    k = 0
    while k < abs(n):
        d += pd.Timedelta(days=step)
        if is_bday(d):
            k += 1
    return d


MONTH_CODE = "FGHJKMNQUVXZ"


def ng_last_trade(year, month):
    """NYMEX Henry Hub (NG) last trading day of contract (year, month): 3 business days before the first calendar
    day of the delivery month. Calendar only (no prices)."""
    first = pd.Timestamp(year, month, 1)
    return add_bdays(first, -3)


ROLL_BDAYS = 3      # live paper rule: roll out of the front contract ROLL_BDAYS business days before its last trade


def ng_contracts_around(day, n=3):
    """The next n NG contracts (year, month, last_trade, symbol) whose last trade day >= day."""
    day = pd.Timestamp(day).normalize()
    out, y, m = [], day.year, day.month
    while len(out) < n:
        lt = ng_last_trade(y, m)
        if lt >= day:
            out.append((y, m, lt, f"NG{MONTH_CODE[m - 1]}{y % 100:02d}.NYM"))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def ng_active_contract(day):
    """Point-in-time roll rule (calendar only): on trading day `day` the active (held) contract is the nearest one
    whose last trade day is more than ROLL_BDAYS business days after `day`. Deterministic, no prices used."""
    day = pd.Timestamp(day).normalize()
    for y, m, lt, sym in ng_contracts_around(day, 3):
        if add_bdays(day, ROLL_BDAYS) < lt:
            return {"year": y, "month": m, "last_trade": lt, "symbol": sym}
    raise RuntimeError("no active contract")


def yahoo_front_switch_days(start, end):
    """Sessions on which Yahoo's continuous NG=F shows a NEW contract (first session after a last trade day)."""
    out = []
    y, m = pd.Timestamp(start).year - 1, 1
    while True:
        lt = ng_last_trade(y, m)
        if lt > pd.Timestamp(end) + pd.Timedelta(days=40):
            break
        out.append(add_bdays(lt, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return [d for d in out if pd.Timestamp(start) <= d <= pd.Timestamp(end)]


# --------------------------------------------------------------------------- hashing + manifest
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, ROOT)


def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return {"raw": {}, "pit": {}, "unavailable": {}}


def save_manifest(man):
    os.makedirs(DDIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def save_raw(source, name, body, url, man, fetched_at=None):
    """Store a raw response append-only under data/live/raw/<source>/<stamp>_<name>; skip if identical to the
    latest snapshot of the same (source, name). Returns (path, sha, is_new)."""
    fetched_at = fetched_at or utcnow()
    sha = sha256_bytes(body)
    prev = [k for k, v in man["raw"].items() if v.get("source") == source and v.get("name") == name]
    if prev:
        last = max(prev, key=lambda k: man["raw"][k]["fetched_at"])
        if man["raw"][last]["sha256"] == sha:
            return os.path.join(ROOT, last), sha, False
    d = os.path.join(RAW, source)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{pd.Timestamp(fetched_at).strftime('%Y%m%dT%H%M%SZ')}_{name}")
    with open(path, "wb") as f:
        f.write(body)
    man["raw"][rel(path)] = {"source": source, "name": name, "url": url, "fetched_at": iso(fetched_at),
                             "sha256": sha, "bytes": len(body)}
    return path, sha, True


# --------------------------------------------------------------------------- append-only PIT files
def pit_path(source):
    return os.path.join(PIT, f"{source}.csv")


def _key(df):
    return (df["series"].astype(str) + "|" + df["obs"].astype(str) + "|" + df["vintage"].astype(str) + "|" +
            df["published_at"].astype(str) + "|" + df["extra"].astype(str))


def validate(df):
    assert list(df.columns) == COLS, df.columns
    pa = pd.to_datetime(df["published_at"], utc=True)
    fa = pd.to_datetime(df["fetched_at"], utc=True)
    assert pa.notna().all() and fa.notna().all(), "published_at / fetched_at must be set"
    bad = pa > fa
    assert not bad.any(), f"published_at > fetched_at in {int(bad.sum())} rows: {df[bad].head(3).to_dict('records')}"


def append_records(source, df, man):
    """Append rows whose (series, obs, vintage, published_at) key is new. Verifies the existing file is unchanged
    since the last manifest pin (append-only). Returns the number of rows appended."""
    os.makedirs(PIT, exist_ok=True)
    path = pit_path(source)
    df = df[COLS].copy()
    df["value"] = df["value"].astype(float)
    validate(df)
    r = rel(path)
    if os.path.exists(path):
        pin = man["pit"].get(r)
        if pin is not None:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                head = f.read(pin["bytes"])
            if size < pin["bytes"] or sha256_bytes(head) != pin["sha256"]:
                raise RuntimeError(f"{r}: existing content differs from the manifest pin (append-only violated)")
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        seen = set(_key(old))
        new = df[~_key(df.astype(str)).isin(seen)]
    else:
        new = df
    new = new.drop_duplicates(subset=["series", "obs", "vintage", "published_at", "extra"])
    if len(new):
        exists = os.path.exists(path)
        buf = io.StringIO()
        new.to_csv(buf, index=False, header=not exists, float_format="%.10g", quoting=csv.QUOTE_MINIMAL)
        with open(path, "a") as f:
            f.write(buf.getvalue())
    if os.path.exists(path):
        man["pit"][r] = {"sha256": sha256_file(path), "bytes": os.path.getsize(path),
                         "rows": int(sum(1 for _ in open(path)) - 1), "updated_at": iso(utcnow())}
    return len(new)


def records(source, published_at=None, fetched_at=None, **cols):
    """Build a record frame from column arrays."""
    n = len(next(iter(cols.values())))
    out = pd.DataFrame({c: cols.get(c, [""] * n) for c in COLS if c in cols or c in ("vintage", "extra")})
    out["source"] = source
    out["published_at"] = published_at if published_at is not None else cols["published_at"]
    out["fetched_at"] = iso(fetched_at) if fetched_at is not None else cols["fetched_at"]
    for c in COLS:
        if c not in out:
            out[c] = ""
    return out[COLS]


# --------------------------------------------------------------------------- data guard for models
GUARDED = [os.path.realpath(DDIR), os.path.realpath(GAS), os.path.realpath(os.path.join(ROOT, "data", "fin")),
           os.path.realpath(RUNS)]


class _Guard:
    active = False
    violations = []


def _audit(event, args):
    if _Guard.active and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        try:
            rp = os.path.realpath(p)
        except Exception:
            rp = p
        if any(rp.startswith(g) for g in GUARDED) and "/ckpt/" not in rp:
            _Guard.violations.append(rp)
            raise PermissionError(f"model tried to open {rp}: models get data only through ctx = asof(t) "
                                  "(data/live, data/gas, data/fin, runs/live are guarded; */ckpt/ is allowed)")


sys.addaudithook(_audit)


class no_data_access:
    def __enter__(self):
        _Guard.active = True

    def __exit__(self, *a):
        _Guard.active = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
