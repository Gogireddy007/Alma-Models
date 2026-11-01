"""
Alma-Gas data: raw files (tools/gas/fetch.py, data/gas/manifest.json) -> a daily panel indexed by Henry Hub spot
trade dates, plus the information-set-safe context builder (docs/gas_prereg.md section 2 is binding).

    nice -n 10 .venv/bin/python tools/gas/data.py build      # writes data/gas/daily.csv, storage.csv, weather.csv,
                                                             # build_report.json

Conventions
  * Index: the dates on which EIA RNGWHHD (Henry Hub spot, the source of FRED DHHNGSP) has a value (empty / NA days
    dropped, never filled; FRED's copy has ~1,900 blanks before 2007 and is used only as a cross-check). Row j = date
    d[j]. All panel values in row j are KNOWN AT THE CLOSE OF d[j] (the forecast time of an origin whose last
    observation is row j):
      hh        Henry Hub spot of d[j]
      stor, stor_chg, stor_dev5  the latest Lower-48 weekly storage report RELEASED on or before d[j]
                (level, weekly change, level minus the 5-year mean of the same report week); stor_week = its week
      hdd, cdd  population-weighted degree days of calendar day d[j]-1 (observed weather only up to the day before)
      hdd7, cdd7  sums over calendar days d[j]-7 .. d[j]-1
  * Storage release (availability) date A(F) of the report for week ending Friday F:
      - listed alternate date in EIA's holiday schedule (data/gas/raw/ngs_schedule.html, 2025-2026), else
      - F+10 (the Monday after) if any US federal holiday falls on F+3..F+7 (holiday week; conservative, covers the
        longest observed delay, Dec 2025: Monday 12:00), else
      - F+6 (Thursday, 10:30 ET).
    A report is usable for an origin whose last observation date t satisfies A(F) <= t (release before the close).
  * Storage values: EIA's first-release ("original_data", revisions.xls) values where available (weeks from
    2015-06-19), current vintage before (ngshistory.xls, from 2010-01-01). Weekly change = first-release level minus
    the previous week's first-release level.
  * Non-positive prices: none expected for HH; any log transform floors at 0.01 USD/MMBtu (all windows).
"""
import os
import re
import sys
import json
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from cities import CITIES, BASE_F  # noqa: E402

DDIR = os.path.join(ROOT, "data", "gas")
RAW = os.path.join(DDIR, "raw")
DAILY = os.path.join(DDIR, "daily.csv")

TRAIN_END = pd.Timestamp("2025-06-30")         # DEV: every target date <= this
DEV_BT_START = pd.Timestamp("2022-07-01")       # harness DEV backtest: first target dates 2022-07-01 .. 2025-06-30
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")
STORAGE_START = pd.Timestamp("2010-01-01")
COVS = ["stor", "stor_chg", "stor_dev5", "hdd", "cdd", "hdd7", "cdd7"]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --------------------------------------------------------------------------- parsers (raw files are data only)
def parse_fred(path):
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


_MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def parse_eia_daily(path):
    """EIA dnav daily table: a B6 cell 'YYYY Mon-DD to Mon-DD' starts a week (Mon..Fri), then 5 B3 value cells."""
    html = open(path, encoding="latin-1").read()
    rows = []
    for m in re.finditer(r"<tr>\s*<td class='B6'>([^<]*)</td>((?:\s*<td class='B3'>[^<]*</td>){5})", html):
        lab = m.group(1).replace("&nbsp;", " ").strip()
        y, mon, day = int(lab[:4]), _MON[lab[5:8]], int(lab[9:11].replace("-", " ").strip())
        start = pd.Timestamp(y, mon, day)
        vals = re.findall(r"<td class='B3'>([^<]*)</td>", m.group(2))
        for k, v in enumerate(vals):
            v = v.replace("&nbsp;", "").replace(",", "").strip()
            if v and v.upper() != "NA":
                rows.append((start + pd.Timedelta(days=k), float(v)))
    return pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def parse_storage():
    cur = pd.read_excel(os.path.join(RAW, "ngshistory.xls"), sheet_name=0, header=None, engine="xlrd")
    cur = cur[pd.to_datetime(cur[0], errors="coerce").notna()].iloc[:, [0, 9]]
    cur.columns = ["week", "l48_current"]
    cur["week"] = pd.to_datetime(cur["week"])
    org = pd.read_excel(os.path.join(RAW, "ngs_revisions.xls"), sheet_name=0, header=None, engine="xlrd")
    org = org[pd.to_datetime(org[0], errors="coerce").notna()].iloc[:, [0, 9]]
    org.columns = ["week", "l48_first"]
    org["week"] = pd.to_datetime(org["week"])
    s = cur.merge(org, on="week", how="left").sort_values("week").reset_index(drop=True)
    s["l48_current"] = s["l48_current"].astype(float)
    s["l48_first"] = s["l48_first"].astype(float)
    s["vintage"] = np.where(s["l48_first"].notna(), "first_release", "current")
    s["l48"] = s["l48_first"].fillna(s["l48_current"])
    assert (s["week"].dt.dayofweek == 4).all(), "storage weeks must end on Fridays"
    assert (s["week"].diff().dropna() == pd.Timedelta(days=7)).all(), "storage weeks must be consecutive"
    return s


def parse_schedule():
    """Alternate release dates from EIA's holiday schedule page -> list of Timestamps."""
    txt = re.sub(r"<[^>]+>", " ", open(os.path.join(RAW, "ngs_schedule.html"), encoding="latin-1").read())
    txt = " ".join(txt.split())
    out = []
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                         r" (\d{1,2}), (\d{4})", txt):
        out.append(pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}"))
    return sorted(set(out))


SCHEDULE_YEARS = (2025, 2026)      # years covered by data/gas/raw/ngs_schedule.html (authoritative for them)


def release_dates(weeks, schedule):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = set(USFederalHolidayCalendar().holidays("1990-01-01", "2030-12-31"))
    hol |= {pd.Timestamp("2025-01-09")}                     # National Day of Mourning (in the 2025 schedule)
    out, rule, used = [], [], set()
    for F in weeks:
        std = F + pd.Timedelta(days=6)
        alt = sorted([a for a in schedule if -3 <= (a - std).days <= 5 and a not in used],
                     key=lambda a: abs((a - std).days))
        if alt:
            used.add(alt[0])
            out.append(alt[0]), rule.append("schedule")
        elif SCHEDULE_YEARS[0] <= std.year <= SCHEDULE_YEARS[1]:
            out.append(std), rule.append("thursday_schedule_period")
        elif any((F + pd.Timedelta(days=k)) in hol for k in range(3, 8)):
            out.append(F + pd.Timedelta(days=10)), rule.append("holiday_week_conservative")
        else:
            out.append(std), rule.append("thursday")
    return out, rule


def parse_weather():
    cols, wsum = {}, sum(w for _, _, w in CITIES.values())
    for name in CITIES:
        d = json.load(open(os.path.join(RAW, f"weather_{name}.json")))
        s = pd.Series(d["daily"]["temperature_2m_mean"], index=pd.to_datetime(d["daily"]["time"]), dtype=float)
        cols[name] = s
    w = pd.DataFrame(cols)
    w.index.name = "date"
    T = w.to_numpy()
    wt = np.array([CITIES[c][2] for c in w.columns]) / wsum
    hdd = np.maximum(0, BASE_F - T)
    cdd = np.maximum(0, T - BASE_F)
    w["hdd_pw"] = np.where(np.isfinite(T).all(1), hdd @ wt, np.nan)
    w["cdd_pw"] = np.where(np.isfinite(T).all(1), cdd @ wt, np.nan)
    return w


# --------------------------------------------------------------------------- panel (pure function of the sources)
def make_panel(hh, stor, wx):
    """hh: DataFrame (date, value); stor: DataFrame (week, l48, avail); wx: DataFrame indexed by calendar date
    with hdd_pw, cdd_pw. Returns the daily panel (see module doc). Every column of row j uses only information
    available at the close of d[j]; tools/gas/test_infoset.py re-derives the panel from perturbed sources to test it."""
    p = hh.rename(columns={"value": "hh"}).sort_values("date").reset_index(drop=True)
    d = p["date"]
    dv = d.to_numpy()
    # storage: latest report with avail <= d[j]
    s = stor.sort_values("week").reset_index(drop=True)
    lvl = s["l48"].to_numpy(float)
    chg = np.concatenate([[np.nan], np.diff(lvl)])
    wk = s["week"]
    dev5 = np.full(len(s), np.nan)
    wkv = wk.to_numpy(dtype="datetime64[D]")
    for k in range(len(s)):
        prev = []
        for yrs in range(1, 6):
            t = wkv[k] - np.timedelta64(364 * yrs, "D")
            m = np.searchsorted(wkv, t - np.timedelta64(3, "D"))
            if m < k and abs((wkv[m] - t).astype(int)) <= 3:
                prev.append(lvl[m])
        if len(prev) == 5:
            dev5[k] = lvl[k] - np.mean(prev)
    av = s["avail"].to_numpy(dtype="datetime64[ns]")
    order = np.argsort(av, kind="stable")
    assert (np.diff(av[order]).astype("timedelta64[D]").astype(int) >= 0).all()
    # for each row: the report with the latest week among those available
    pos = np.searchsorted(av[order], dv, side="right") - 1
    best = np.full(len(p), -1)
    run, ptr = -1, 0
    for j in range(len(p)):
        while ptr <= pos[j]:
            run = max(run, order[ptr])
            ptr += 1
        best[j] = run
    ok = best >= 0
    p["stor"] = np.where(ok, lvl[np.maximum(best, 0)], np.nan)
    p["stor_chg"] = np.where(ok, chg[np.maximum(best, 0)], np.nan)
    p["stor_dev5"] = np.where(ok, dev5[np.maximum(best, 0)], np.nan)
    p["stor_week"] = pd.Series(np.where(ok, wk.to_numpy()[np.maximum(best, 0)], np.datetime64("NaT")))
    p["stor_avail"] = pd.Series(np.where(ok, av[np.maximum(best, 0)], np.datetime64("NaT")))
    # weather: calendar day d-1, and d-7..d-1
    h, c = wx["hdd_pw"], wx["cdd_pw"]
    h7, c7 = h.rolling(7, min_periods=7).sum(), c.rolling(7, min_periods=7).sum()
    prev = d - pd.Timedelta(days=1)
    p["hdd"], p["cdd"] = prev.map(h).astype(float), prev.map(c).astype(float)
    p["hdd7"], p["cdd7"] = prev.map(h7).astype(float), prev.map(c7).astype(float)
    p["wx_date"] = prev
    return p


def sources():
    hh = parse_eia_daily(os.path.join(RAW, "rngwhhdd.htm"))
    st = parse_storage()
    st["avail"], st["avail_rule"] = release_dates(st["week"], parse_schedule())
    wx = parse_weather()
    return hh, st, wx


def build():
    hh, st, wx = sources()
    p = make_panel(hh, st, wx)
    p.to_csv(DAILY, index=False, date_format="%Y-%m-%d")
    st.to_csv(os.path.join(DDIR, "storage.csv"), index=False, date_format="%Y-%m-%d")
    wx.to_csv(os.path.join(DDIR, "weather.csv"), date_format="%Y-%m-%d")
    # cross-check FRED vs EIA spot (dates <= VAL_END only; TEST values are not inspected)
    hf = parse_fred(os.path.join(RAW, "DHHNGSP.csv"))
    m = hf.merge(hh, on="date", suffixes=("_fred", "_eia"))
    m = m[m.date <= VAL_END]
    seen = p[p.date <= VAL_END]
    rep = {
        "daily_csv_sha256": sha256(DAILY),
        "rows": int(len(p)), "first": str(p.date.iloc[0].date()), "last": str(p.date.iloc[-1].date()),
        "rows_dev": int((p.date <= TRAIN_END).sum()),
        "rows_val": int(((p.date >= VAL_START) & (p.date <= VAL_END)).sum()),
        "rows_test": int((p.date >= TEST_START).sum()),
        "fred_missing_days_dropped": int(pd.to_numeric(pd.read_csv(os.path.join(RAW, "DHHNGSP.csv")).iloc[:, 1],
                                                       errors="coerce").isna().sum()),
        "hh_nonpositive_through_val": int((seen.hh <= 0).sum()),
        "fred_vs_eia_spot_through_val": {"common_days": int(len(m)),
                                         "n_differ_gt_0.005": int((abs(m.value_fred - m.value_eia) > 0.005).sum())},
        "eia_days_without_fred_value_through_val": int((~seen.date.isin(hf.date)).sum()),
        "futures": "EIA RNGC1/RNGC2 downloaded but NOT used: EIA stopped publishing them in 2024 (no values on any "
                   "2024-2025 spot day)",
        "storage_weeks": int(len(st)), "storage_first_week": str(st.week.iloc[0].date()),
        "storage_last_week": str(st.week.iloc[-1].date()),
        "storage_vintage_counts": st.vintage.value_counts().to_dict(),
        "storage_avail_rule_counts": st.avail_rule.value_counts().to_dict(),
        "storage_schedule_rows": st[st.avail_rule == "schedule"][["week", "avail"]].astype(str).values.tolist(),
        "weather_first": str(wx.index[0].date()), "weather_last": str(wx.index[-1].date()),
        "weather_missing_days_through_val": int(wx.loc[:VAL_END, "hdd_pw"].isna().sum()),
    }
    json.dump(rep, open(os.path.join(DDIR, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# --------------------------------------------------------------------------- loading and the context builder
_CACHE = {}


def panel():
    """-> (dates [n] datetime64[D], hh [n], C [n, len(COVS)], extra DataFrame)."""
    if "p" not in _CACHE:
        p = pd.read_csv(DAILY, parse_dates=["date", "stor_week", "stor_avail", "wx_date"])
        _CACHE["p"] = (p["date"].to_numpy().astype("datetime64[D]"), p["hh"].to_numpy(float),
                       p[COVS].to_numpy(float), p)
    return _CACHE["p"]


def build_context(i, ctx, dates=None, y=None, C=None, extra=None):
    """Inputs for an origin at panel index i: the forecast is made at the close of dates[i-1]; step k targets
    y[i-1+k]. Returns dict(y [ctx], covs [ctx, len(COVS)], ctx_dates [ctx], origin_date, first_target_date).
    Assertions enforce the information set with explicit dates."""
    if dates is None:
        dates, y, C, extra = panel()
    lo = i - ctx
    if lo < 0:
        raise ValueError(f"not enough history for origin {i}")
    t = dates[i - 1]
    cd = dates[lo:i]
    assert (cd <= t).all() and i < len(dates) + 1
    if i < len(dates):
        assert dates[i] > t, "first target must be after the forecast time"
    if extra is not None:
        sa = extra["stor_avail"].to_numpy()[lo:i].astype("datetime64[D]")
        sw = extra["stor_week"].to_numpy()[lo:i].astype("datetime64[D]")
        okk = ~np.isnat(sa)
        # storage: every report used was released on or before its row date, hence before the forecast time
        assert (sa[okk] <= cd[okk]).all() and (sa[okk] <= t).all(), "storage report used before its release"
        assert (sw[okk] + np.timedelta64(5, "D") <= sa[okk]).all(), "storage report released before week end + 5 days"
        wd = extra["wx_date"].to_numpy()[lo:i].astype("datetime64[D]")
        assert (wd < cd).all() and (wd <= t - np.timedelta64(1, "D")).all(), "weather of the forecast day used"
    return {"y": y[lo:i].copy(), "covs": C[lo:i].copy(), "ctx_dates": cd, "origin_date": t,
            "first_target_date": dates[i] if i < len(dates) else None}


# --------------------------------------------------------------------------- training helpers (DEV only)
def train_panel(end=None):
    """Panel rows with date <= TRAIN_END (2025-06-30); a later `end` raises. Use for any fitting / training."""
    end = TRAIN_END if end is None else pd.Timestamp(end)
    if end > TRAIN_END:
        raise ValueError(f"training data may not extend past {TRAIN_END.date()} (asked {end.date()})")
    dates, y, C, extra = panel()
    m = dates <= np.datetime64(end.date(), "D")
    out = (dates[m], y[m].copy(), C[m].copy(), extra[m].reset_index(drop=True))
    assert out[0].max() <= np.datetime64(TRAIN_END.date(), "D")
    return out


def train_windows(ctx, horizon, stride=1, start=None, end=None):
    """X [n, ctx] (hh), XC [n, ctx, len(COVS)], Y [n, horizon], target_dates [n, horizon]: every target <= 2025-06-30."""
    dates, y, C, extra = train_panel(end)
    lo = ctx if start is None else max(ctx, int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start).date(), "D"))))
    X, XC, Y, T = [], [], [], []
    for i in range(lo, len(dates) - horizon + 1, stride):
        w = build_context(i, ctx, dates, y, C, extra)
        X.append(w["y"]), XC.append(w["covs"]), Y.append(y[i:i + horizon]), T.append(dates[i:i + horizon])
    T = np.array(T)
    assert len(T) == 0 or T.max() <= np.datetime64(TRAIN_END.date(), "D"), "training target after TRAIN_END"
    return np.array(X), np.array(XC), np.array(Y), T


def split_origins(split, dates=None, min_ctx=1024):
    """Origin indices i of a split: the first target date dates[i] lies in the split window (DEV backtest:
    2022-07-01 .. 2025-06-30). Scoring additionally requires the h-step target date to lie in the window."""
    if dates is None:
        dates = panel()[0]
    if split == "dev":
        lo, hi = DEV_BT_START, TRAIN_END
    elif split == "val":
        lo, hi = VAL_START, VAL_END
    elif split == "test":
        lo, hi = TEST_START, pd.Timestamp(dates[-1])
    else:
        raise ValueError(split)
    m = (dates >= np.datetime64(lo.date(), "D")) & (dates <= np.datetime64(hi.date(), "D"))
    idx = np.where(m)[0]
    return idx[idx >= min_ctx]


def split_end(split, dates=None):
    if dates is None:
        dates = panel()[0]
    return {"dev": np.datetime64(TRAIN_END.date(), "D"), "val": np.datetime64(VAL_END.date(), "D"),
            "test": dates[-1]}[split]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
