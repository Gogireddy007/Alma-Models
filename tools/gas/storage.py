"""
Alma-Gas weekly storage task (PRIMARY target, docs/gas_prereg.md sections 2-4): weekly table, CPC degree days
(observed + archived 7-day forecasts), origins and the information-set-safe storage context builder.

    nice -n 10 .venv/bin/python tools/gas/storage.py build     # data/gas/weekly.csv, cpc_obs.csv, cpc_fcst.csv

Definitions
  * Week W ends Friday F_W; its report is released at A_W (data.py release_dates; Thursday F_W+6 10:30 ET normally).
    Release datetime used: A_W at 12:00 ET (the latest standard time of any alternate release).
  * dS_W = first-print level(W) - first-print level(W-1) (Bcf; current vintage before 2015-06-19), per series
    (l48 and the 5 regions). avg5_W = mean of dS over the same report week of the 5 previous years (weeks F_W - 364k
    +- 3 days, k = 1..5), computed by us. dev_W = dS_W - avg5_W. Label y_W = 1 if dev_W > 0, 0 if < 0, excluded if 0.
  * Origins for target week W:
      thu: A_{W-1} at 17:00 ET (close of the day the previous report came out; normally Thursday F_W - 1)
      fri: max(F_W, A_{W-1}) at 17:00 ET (close of the last day of the report week)
    Both are strictly before A_W (asserted).
  * Weather for the 7 days of week W (Sat F_W-6 .. Fri F_W), CONUS rows of CPC files: gas-utility-weighted HDD,
    population-weighted CDD. Day x uses the OBSERVED value iff x <= origin date - 1; otherwise the CPC FORECAST of
    the latest issue i usable at the origin: i < origin date, or i == origin date and the file's HTTP Last-Modified
    <= the origin datetime (issues are posted ~10:00-11:00 ET). Normals: mean of the observed value on the same
    month-day over the 10 previous calendar years (causal).
"""
import os
import re
import sys
import glob
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

ROOT = D.ROOT
RAW = D.RAW
WEEKLY = os.path.join(D.DDIR, "weekly.csv")
CPC_OBS = os.path.join(D.DDIR, "cpc_obs.csv")
CPC_FCST = os.path.join(D.DDIR, "cpc_fcst.csv")
SERIES = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]
ORIGINS = ("thu", "fri")
TZ = "America/New_York"
HIST_WEEKS = 520            # weekly history handed to models (10 years; models crop)
NORMAL_YEARS = 10


# --------------------------------------------------------------------------- weekly table
def _sheet(fn):
    x = pd.read_excel(os.path.join(RAW, fn), sheet_name=0, header=None, engine="xlrd")
    x = x[pd.to_datetime(x[0], errors="coerce").notna()].iloc[:, [0, 2, 3, 4, 5, 6, 9]]
    x.columns = ["week", "east", "midwest", "mountain", "pacific", "south_central", "l48"]
    x["week"] = pd.to_datetime(x["week"])
    return x.set_index("week").astype(float)


def levels():
    """(first-print-where-available levels by week [DataFrame, index week, columns SERIES], release table)."""
    cur, org = _sheet("ngshistory.xls"), _sheet("ngs_revisions.xls")
    st = D.parse_storage()
    st["avail"], st["avail_rule"] = D.release_dates(st["week"], D.parse_schedule())
    lvl = org.reindex(cur.index).combine_first(cur)            # first print where available
    assert np.allclose(lvl["l48"].to_numpy(), st.set_index("week")["l48"].to_numpy())
    return lvl, st


def weekly_from(lvl, st):
    """Weekly table from levels: dS, causal avg5 (same week, 5 previous years), dev. Pure function (tested)."""
    w = st[["week", "avail", "avail_rule", "vintage"]].set_index("week")
    wk = lvl.index
    match = []
    for k in range(len(wk)):
        mm = []
        for yrs in range(1, 6):
            m = np.where(np.abs((wk - (wk[k] - pd.Timedelta(days=364 * yrs))).days) <= 3)[0]
            mm.append(m[0] if len(m) == 1 else -1)
        match.append(mm)
    match = np.array(match)
    for s in SERIES:
        v = lvl[s].to_numpy(float)
        ds = np.concatenate([[np.nan], np.diff(v)])
        a5 = np.full(len(v), np.nan)
        for k in range(len(v)):
            mm = match[k]
            if (mm >= 0).all() and (mm < k).all() and np.isfinite(ds[mm]).all():
                a5[k] = ds[mm].mean()
        w[f"lvl_{s}"], w[f"ds_{s}"], w[f"avg5_{s}"], w[f"dev_{s}"] = v, ds, a5, ds - a5
    return w.reset_index()


def weekly_table():
    return weekly_from(*levels())


# --------------------------------------------------------------------------- CPC degree days
def _read_cpc(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    hdr = next(l for l in lines if l.startswith("Region|"))
    row = next(l for l in lines if l.startswith("CONUS|"))
    days = [c for c in hdr.split("|")[1:] if re.fullmatch(r"\d{8}", c)]
    vals = row.split("|")[1:1 + len(days)]
    return [(pd.Timestamp(d), float(v) if v.strip() not in ("", "-999", "-9999") else np.nan) for d, v in zip(days, vals)]


def cpc_obs():
    out = {}
    for prod, col in (("UtilityGas.Heating.txt", "hdd"), ("Population.Cooling.txt", "cdd")):
        rows = []
        for f in sorted(glob.glob(os.path.join(RAW, "cpc", "obs", "*", prod))):
            rows += _read_cpc(f)
        s = pd.Series(dict(rows)).sort_index()
        out[col] = s
    o = pd.DataFrame(out)
    o.index.name = "date"
    return o


BAD_FCST = []


def cpc_fcst():
    man = json.load(open(os.path.join(D.DDIR, "manifest_cpc.json")))["files"]
    rows = []
    for key, meta in man.items():
        if "/cpc/fcst/" not in key:
            continue
        prod = os.path.basename(key)
        col = "hdd" if prod.startswith("UtilityGas") else "cdd"
        parts = key.split("/")
        issue = pd.Timestamp(f"{parts[-4]}-{parts[-3]}-{parts[-2]}")
        lm = pd.to_datetime(meta.get("last_modified"), utc=True) if meta.get("last_modified") else pd.NaT
        txt = open(os.path.join(ROOT, key), encoding="latin-1").read()
        m = re.search(r"Issued (\d{4})UTC (\d{8})", txt)
        if not m or "CONUS|" not in txt:
            BAD_FCST.append(key)                    # malformed / empty file: unusable (counted in the build report)
            continue
        assert pd.Timestamp(m.group(2)) == issue, f"{key}: issue date mismatch"
        for d, v in _read_cpc(os.path.join(ROOT, key)):
            rows.append((issue, col, d, v, lm))
    f = pd.DataFrame(rows, columns=["issue", "var", "day", "value", "last_modified"])
    f = f.pivot_table(index=["issue", "day"], columns="var", values="value", aggfunc="first").reset_index()
    lm = pd.DataFrame(rows, columns=["issue", "var", "day", "value", "last_modified"]).groupby("issue")["last_modified"].max()
    f["last_modified"] = f["issue"].map(lm)
    return f.sort_values(["issue", "day"]).reset_index(drop=True)


def build():
    w = weekly_table()
    w.to_csv(WEEKLY, index=False, date_format="%Y-%m-%d")
    o = cpc_obs()
    o.to_csv(CPC_OBS, date_format="%Y-%m-%d")
    f = cpc_fcst()
    f.to_csv(CPC_FCST, index=False)
    seen = w[w.avail <= D.VAL_END]
    rep = {"weekly_csv_sha256": D.sha256(WEEKLY), "cpc_obs_csv_sha256": D.sha256(CPC_OBS),
           "cpc_fcst_csv_sha256": D.sha256(CPC_FCST), "weeks": int(len(w)),
           "weeks_with_avg5_l48": int(w.avg5_l48.notna().sum()),
           "weeks_dev_released": int((w.avail <= D.TRAIN_END).sum()),
           "weeks_val_released": int(((w.avail >= D.VAL_START) & (w.avail <= D.VAL_END)).sum()),
           "weeks_test_released": int((w.avail >= D.TEST_START).sum()),
           "dev_l48_zero_through_val": int((seen.dev_l48 == 0).sum()),
           "cpc_obs_first": str(o.index[0].date()), "cpc_obs_last": str(o.index[-1].date()),
           "cpc_obs_missing_through_val": int(o.loc[:D.VAL_END].isna().any(axis=1).sum()),
           "cpc_fcst_malformed_files": BAD_FCST,
           "cpc_fcst_missing_404": len(json.load(open(os.path.join(D.DDIR, "manifest_cpc.json")))["missing"]),
           "cpc_fcst_issues": int(f.issue.nunique()), "cpc_fcst_first": str(f.issue.min().date()),
           "cpc_fcst_last": str(f.issue.max().date()),
           "cpc_fcst_issue_days_missing_2014_through_val": int(len(pd.date_range("2014-01-01", D.VAL_END))
                                                              - f[f.issue <= D.VAL_END].issue.nunique()),
           "cpc_fcst_lastmod_same_day_share": float(np.mean(
               [(pd.Timestamp(lm).tz_convert(TZ).date() == i.date()) if pd.notna(lm) else False
                for i, lm in f.groupby("issue")["last_modified"].first().items()]))}
    json.dump(rep, open(os.path.join(D.DDIR, "build_report_storage.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# --------------------------------------------------------------------------- loading + context
_C = {}


def load():
    if "w" not in _C:
        w = pd.read_csv(WEEKLY, parse_dates=["week", "avail"])
        o = pd.read_csv(CPC_OBS, parse_dates=["date"]).set_index("date")
        f = pd.read_csv(CPC_FCST, parse_dates=["issue", "day"])
        f["last_modified"] = pd.to_datetime(f["last_modified"], utc=True)
        _C["w"], _C["o"], _C["f"] = w, o, f
    return _C["w"], _C["o"], _C["f"]


def o7_of(o):
    """Observed 7-day sums over d-6..d (NaN unless all 7 days exist); cached per frame object."""
    if o is _C.get("o") and "o7" in _C:
        return _C["o7"]
    r7 = o[["hdd", "cdd"]].rolling("7D").sum()
    n7 = o.hdd.rolling("7D").count()
    r7[n7 < 7] = np.nan
    if o is _C.get("o"):
        _C["o7"] = r7
    return r7


def at_et(day, hh, mm=0):
    return pd.Timestamp(day.date()).replace(hour=hh, minute=mm).tz_localize(TZ)


def release_dt(w, k):
    return at_et(w.avail.iloc[k], 12)


def origin_dt(w, k, kind):
    """Origin datetime (tz-aware, ET) for target week index k."""
    prev = w.avail.iloc[k - 1]
    if kind == "thu":
        day = prev
    elif kind == "fri":
        day = max(w.week.iloc[k], prev)
    else:
        raise ValueError(kind)
    return at_et(day, 17)


def normals(o, days):
    """Causal climatology: mean observed value on the same month-day over the NORMAL_YEARS previous years."""
    out = []
    for x in days:
        vals = []
        for y in range(1, NORMAL_YEARS + 1):
            try:
                t = x.replace(year=x.year - y)
            except ValueError:                       # Feb 29
                t = x.replace(year=x.year - y, day=28)
            if t in o.index:
                vals.append(o.loc[t].to_numpy(float))
        out.append(np.nanmean(vals, axis=0) if vals else [np.nan, np.nan])
    return np.array(out, float)


def week_weather(k, odt, w=None, o=None, f=None):
    """[7, 2] HDD/CDD for the days of target week k as known at origin odt, the source of each day
    (0 = observed, j >= 1 = forecast with lead j days from its issue), and the issue date used."""
    if w is None:
        w, o, f = load()
    F = w.week.iloc[k]
    days = [F - pd.Timedelta(days=6 - j) for j in range(7)]
    od = pd.Timestamp(odt.date())
    usable = f[(f.issue < od) | ((f.issue == od) & (f.last_modified <= odt.tz_convert("UTC")))]
    issue = usable.issue.max() if len(usable) else pd.NaT
    fx = usable[usable.issue == issue].set_index("day") if pd.notna(issue) else None
    vals, src = np.full((7, 2), np.nan), np.full(7, -1)
    for j, x in enumerate(days):
        if x <= od - pd.Timedelta(days=1):
            if x in o.index:
                vals[j] = o.loc[x, ["hdd", "cdd"]].to_numpy(float)
                src[j] = 0
        elif fx is not None and x in fx.index:
            vals[j] = fx.loc[x, ["hdd", "cdd"]].to_numpy(float)
            src[j] = (x - issue).days + 1
    return vals, src, issue, days


def target_weeks(split, w=None, series="l48"):
    """Indices k of target weeks whose release date lies in the split window (DEV backtest: 2019-07-01 ..
    2025-06-30) and whose label is defined."""
    if w is None:
        w = load()[0]
    if split == "dev":
        lo, hi = pd.Timestamp("2019-07-01"), D.TRAIN_END
    elif split == "val":
        lo, hi = D.VAL_START, D.VAL_END
    elif split == "test":
        lo, hi = D.TEST_START, w.avail.max()
    elif split == "train":
        lo, hi = pd.Timestamp("2014-01-15"), D.TRAIN_END
    else:
        raise ValueError(split)
    m = (w.avail >= lo) & (w.avail <= hi) & w[f"avg5_{series}"].notna()
    return np.where(m.to_numpy())[0]


def build_context(k, kind, w=None, o=None, f=None, series=SERIES):
    """Storage-task inputs for target week k at origin kind ('thu' | 'fri'). Everything returned is known at the
    origin; assertions check it with explicit timestamps. No value of week k (or later) is included."""
    if w is None:
        w, o, f = load()
    odt = origin_dt(w, k, kind)
    assert odt < release_dt(w, k), "origin must precede the target release"
    od = pd.Timestamp(odt.date())
    # released history: weeks whose release datetime <= origin -> exactly the weeks before k
    rel = np.array([release_dt(w, j) <= odt for j in range(max(0, k - HIST_WEEKS - 2), k + 1)])
    lo = max(0, k - HIST_WEEKS)
    assert rel[-1] == False and rel[-2] == True, "history must end at week k-1"   # noqa: E712
    hist = {s: w[f"ds_{s}"].to_numpy(float)[lo:k] for s in series}
    hdev = {s: w[f"dev_{s}"].to_numpy(float)[lo:k] for s in series}
    hlvl = {s: w[f"lvl_{s}"].to_numpy(float)[lo:k] for s in series}
    assert (w.avail.iloc[lo:k] <= od).all()
    # weather of the target week, and the weather of the history weeks (observed, all known)
    vals, src, issue, days = week_weather(k, odt, w, o, f)
    for j, x in enumerate(days):
        if src[j] == 0:
            assert x <= od - pd.Timedelta(days=1), "observed weather of the origin day or later"
        elif src[j] >= 1:
            assert issue < od or issue == od, "forecast issued after the origin"
    nrm = normals(o, days)
    obs_hist = o.loc[:od - pd.Timedelta(days=1)].iloc[-HIST_WEEKS * 7:]
    assert obs_hist.index.max() <= od - pd.Timedelta(days=1)
    o7 = o7_of(o)
    hw = w.week.iloc[lo:k]
    assert hw.max() <= od - pd.Timedelta(days=1)          # history weeks ended before the origin day
    seg = o7.reindex(hw)
    wk_hdd, wk_cdd = seg.hdd.to_numpy(float), seg.cdd.to_numpy(float)
    hh_dates, hh, _, _ = D.panel()
    m = hh_dates <= np.datetime64(od.date(), "D")
    return {
        "k": int(k), "kind": kind, "origin": str(odt), "target_week": str(w.week.iloc[k].date()),
        "target_release": str(release_dt(w, k)),
        "avg5": {s: float(w[f"avg5_{s}"].iloc[k]) for s in series},     # deterministic from past years
        "ds_hist": hist, "dev_hist": hdev, "lvl_hist": hlvl,
        "hist_weeks": w.week.iloc[lo:k].dt.strftime("%Y-%m-%d").to_numpy(),
        "hist_hdd": np.array(wk_hdd), "hist_cdd": np.array(wk_cdd),
        "wx": vals, "wx_src": src, "wx_issue": None if pd.isna(issue) else str(issue.date()),
        "wx_normal": nrm, "wx_days": [str(x.date()) for x in days],
        "obs_daily": obs_hist[["hdd", "cdd"]].to_numpy(float), "obs_daily_dates": obs_hist.index.strftime("%Y-%m-%d").to_numpy(),
        "hh": hh[m][-1024:], "hh_dates": hh_dates[m][-1024:],
    }


def label(k, series="l48", w=None):
    if w is None:
        w = load()[0]
    dv = float(w[f"dev_{series}"].iloc[k])
    return dv, (None if dv == 0 or not np.isfinite(dv) else int(dv > 0))


def train_weeks(kind_unused=None):
    """Target weeks usable for fitting: released <= 2025-06-30 and from 2014-01-15 (CPC forecast archive)."""
    w = load()[0]
    ks = target_weeks("train", w)
    assert (w.avail.iloc[ks] <= D.TRAIN_END).all()
    return ks


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
