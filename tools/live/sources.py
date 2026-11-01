"""
Alma-Live Gas sources: fetch (raw snapshot, append-only, sha256 in data/live/manifest.json) + parse into point-in-time
records (common.COLS). Every parser assigns published_at from the source's RELEASE SCHEDULE (conservative where the
schedule is uncertain) and never from the observation date alone. docs/live_leakage.md lists each rule and risk.

Sources (all free; no keys):
  storage       EIA WNGSR Lower-48 + 5 regions: first print (revisions.xls original_data) at its release, revised
                vintage at the NEXT report's release (33-week resubmission: release of week 2026-04-17), printed net
                change as its own series (FP(W) - level(W-1) known at A(W)); pre-2015-06-19 weeks only exist in the
                current vintage ('current_backfill', revision risk disclosed). Release: Thu 10:30 ET; EIA holiday
                schedule alternates at 12:00 ET; pre-2025 holiday weeks F+10 12:00 ET (conservative).
  px            Yahoo chart daily OHLCV for NG=F (continuous front), UNG, BOIL (+ the live NG contracts). Values
                stored UNADJUSTED (split adjustment is re-applied as of t from split records <= t). Bar of session d
                published d 18:00 ET (after NG Globex 17:00 ET and ETF 16:00 ET closes).
  cpc_fcst      CPC 7-day CONUS degree-day forecasts as issued (gas-weighted HDD, pop-weighted CDD); published_at =
                HTTP Last-Modified if within 3 days of the issue date, else issue + 1 day 17:00 UTC.
  cpc_obs       CPC observed daily CONUS degree days (REALIZED weather, never used as a forecast): day D at D+1
                18:00 UTC (yearly files hold final values: small revision risk).
  om_fcst       Open-Meteo Previous-Runs API (GFS), hourly temperature_2m_previous_day{1..7} for the 10 Alma-Gas cities
                -> population-weighted HDD/CDD of 3 demand regions (NE, MW, S) per valid UTC day and lead k;
                published_at = (valid day - (k-1)) 06:00 UTC. Available from 2021-04 only (earlier: unavailable).
  cot           CFTC disaggregated futures-only, NYMEX Henry Hub NG (023651): Tuesday positions at Friday 15:30 ET;
                federal holiday Tue..Fri -> next business day 15:30 ET; shutdown backlogs -> conservative catch-up.
  edgar_filing  SEC EDGAR submissions of gas producers / LNG exporters / utilities / pipelines: forms 10-K, 10-Q, 8-K
                (+/A); published_at = acceptanceDateTime interpreted as US/Eastern (EDGAR's 'Z' suffix is ET).
  edgar_fact    XBRL company-concept facts (Revenues, NaturalGasProductionRevenue, ...) published at the accepting
                filing's acceptance time (or the filed date 23:59 ET when the accession is not in the submissions).
  rigs          Baker Hughes: rigcount.bakerhughes.com returns HTTP 403 (bot protection, not bypassed) -> unavailable.
  eia_api       only if env EIA_API_KEY is set (never requested); otherwise skipped.
"""
import os
import re
import sys
import json
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

UA_WEB = "Mozilla/5.0 (alma-live-gas research)"
# SEC fair-access policy asks for a descriptive User-Agent with contact info: operators set SEC_USER_AGENT
SEC_UA = os.environ.get("SEC_USER_AGENT", "AlmaLiveGas research bot (set SEC_USER_AGENT to add a contact)")


def http_get(url, ua=UA_WEB, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 404):
                raise
            time.sleep(2 * (k + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise last


# =========================================================================== EIA storage
EIA_FILES = {"ngshistory.xls": "https://ir.eia.gov/ngs/ngshistory.xls",
             "revisions.xls": "https://ir.eia.gov/ngs/revisions.xls",
             "schedule.html": "https://ir.eia.gov/ngs/schedule.html"}
STOR_SERIES = ["east", "midwest", "mountain", "pacific", "south_central", "l48"]
RESUB_FIRST, RESUB_LAST, RESUB_REPORT_WEEK = pd.Timestamp("2025-08-29"), pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")


def _sheet(path):
    x = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    x = x[pd.to_datetime(x[0], errors="coerce").notna()].iloc[:, [0, 2, 3, 4, 5, 6, 9]]
    x.columns = ["week"] + STOR_SERIES
    x["week"] = pd.to_datetime(x["week"])
    return x.set_index("week").astype(float)


def _schedule(path):
    txt = re.sub(r"<[^>]+>", " ", open(path, encoding="latin-1").read())
    txt = " ".join(txt.split())
    out = []
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                         r" (\d{1,2}), (\d{4})", txt):
        out.append(pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}"))
    return sorted(set(out))


SCHEDULE_YEARS = (2025, 2026)


def storage_release(weeks, schedule):
    """Release datetime (UTC) and rule for each report week ending Friday F (see module doc)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = set(USFederalHolidayCalendar().holidays("1990-01-01", "2035-12-31")) | {pd.Timestamp("2025-01-09"),
                                                                              pd.Timestamp("2018-12-05")}
    out, rule, used = [], [], set()
    for F in weeks:
        std = F + pd.Timedelta(days=6)
        alt = sorted([a for a in schedule if -3 <= (a - std).days <= 5 and a not in used],
                     key=lambda a: abs((a - std).days))
        if alt and alt[0] != std:
            used.add(alt[0])
            out.append(C.et(alt[0], 12, 0)), rule.append("schedule_alternate_1200ET")
        elif alt:
            used.add(alt[0])
            out.append(C.et(std, 10, 30)), rule.append("schedule_thursday")
        elif SCHEDULE_YEARS[0] <= std.year <= SCHEDULE_YEARS[1]:
            out.append(C.et(std, 10, 30)), rule.append("thursday_schedule_period")
        elif any((F + pd.Timedelta(days=k)) in hol for k in range(4, 8)):     # Tue..Fri holiday in release week
            out.append(C.et(F + pd.Timedelta(days=10), 12, 0)), rule.append("holiday_week_conservative")
        else:
            out.append(C.et(std, 10, 30)), rule.append("thursday")
    return out, rule


def fetch_storage(man):
    paths = {}
    for name, url in EIA_FILES.items():
        body, _ = http_get(url)
        p, sha, new = C.save_raw("storage", name, body, url, man)
        paths[name] = p
    return paths


def parse_storage(paths, fetched_at):
    cur = _sheet(paths["ngshistory.xls"])
    org = _sheet(paths["revisions.xls"])
    sched = _schedule(paths["schedule.html"])
    weeks = cur.index
    assert (weeks.dayofweek == 4).all() and (np.diff(weeks.values).astype("timedelta64[D]").astype(int) == 7).all()
    rel, rule = storage_release(weeks, sched)
    rel = pd.Series(rel, index=weeks)
    rule = pd.Series(rule, index=weeks)
    last_org = org.index.max()
    rows = []
    fa = C.iso(fetched_at)
    level_known = {}            # (series, week) -> [(published_at, value)]
    for s in STOR_SERIES:
        for i, w in enumerate(weeks):
            c = cur.at[w, s]
            if w in org.index:
                first, vin = org.at[w, s], "first_release"
            elif w > last_org:
                first, vin = c, "first_release"          # no revision recorded after the file's last week
            else:
                first, vin = c, "current_backfill"       # pre-2015-06-19: only the current vintage exists
            pa = rel[w]
            rows.append((s, w, first, pa, vin, rule[w]))
            level_known.setdefault((s, w), []).append((pa, first))
            if vin == "first_release" and np.isfinite(c) and abs(c - first) > 1e-9:
                if RESUB_FIRST <= w <= RESUB_LAST:
                    rpa = rel[RESUB_REPORT_WEEK]
                else:
                    rpa = rel[weeks[i + 1]] if i + 1 < len(weeks) else pd.Timestamp(fetched_at)
                rows.append((s, w, c, rpa, "revised", "revision_at_next_report"))
                level_known[(s, w)].append((rpa, c))
    out = pd.DataFrame(rows, columns=["series", "week", "value", "pa", "vintage", "extra"])
    # printed net change: FP(W) - level(W-1) as known at A(W)
    chg = []
    for s in STOR_SERIES:
        for i, w in enumerate(weeks[1:], start=1):
            pa = rel[w]
            fp = [v for p, v in level_known[(s, w)] if p == pa][0]
            prev = [v for p, v in sorted(level_known[(s, weeks[i - 1])]) if p <= pa]
            if prev:
                chg.append((f"{s}.chg_printed", w, fp - prev[-1], pa, "first_release"
                            if w >= pd.Timestamp("2015-06-19") else "current_backfill", "printed_net_change"))
    out = pd.concat([out, pd.DataFrame(chg, columns=out.columns)], ignore_index=True)
    return C.records("storage", series=out["series"].tolist(), obs=out["week"].dt.strftime("%Y-%m-%d").tolist(),
                     value=out["value"].tolist(), published_at=[C.iso(p) for p in out["pa"]], fetched_at=fa,
                     vintage=out["vintage"].tolist(), extra=out["extra"].tolist())


# =========================================================================== prices (Yahoo chart)
PX_SYMBOLS = ["NG=F", "UNG", "BOIL"]
YURL = ("https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1={p1}&period2={p2}&interval=1d"
        "&events=div%2Csplit&includeAdjustedClose=true")
BAR_PUBLISH_HH = 18          # bar of session d published d 18:00 ET


def fetch_px(man, symbols=None, start="2000-01-01"):
    symbols = symbols or (PX_SYMBOLS + [c[3] for c in C.ng_contracts_around(pd.Timestamp.now(), 2)])
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp.now(tz="UTC").timestamp()) + 86400
    out = {}
    for s in symbols:
        url = YURL.format(s=urllib.request.quote(s), p1=p1, p2=p2)
        try:
            body, _ = http_get(url)
        except Exception as e:  # noqa: BLE001
            C.log(f"px {s}: {e}")
            continue
        out[s] = C.save_raw("px", s.replace("=", "_") + ".json", body, url, man)[0]
        time.sleep(1.0)
    return out


def parse_px(path, symbol, fetched_at):
    d = json.load(open(path))["chart"]["result"][0]
    # session date in the EXCHANGE time zone with DST of each bar (meta.gmtoffset is today's offset only: using it
    # shifted every summer futures bar, stamped 00:00 ET = 04:00Z, to the previous calendar day)
    tzname = d["meta"].get("exchangeTimezoneName") or "America/New_York"
    ts = pd.to_datetime(np.array(d["timestamp"]), unit="s", utc=True).tz_convert(tzname).tz_localize(None).normalize()
    q = d["indicators"]["quote"][0]
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")}, index=ts).astype(float)
    df = df[~df.index.duplicated(keep="last")]
    splits = d.get("events", {}).get("splits", {})
    sp = pd.Series({pd.Timestamp(int(v["date"]), unit="s", tz="UTC").tz_convert(tzname).tz_localize(None).normalize():
                    v["numerator"] / v["denominator"] for v in splits.values()}, dtype=float).sort_index()
    # Yahoo OHLC are split-adjusted for every split up to the fetch: unadjust -> raw traded prices
    fac = np.ones(len(df))
    for sd, r in sp.items():
        fac[df.index < sd] *= r
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * fac
    df["volume"] = df["volume"] / fac
    fa = pd.Timestamp(fetched_at)
    today = fa.tz_convert(C.ET).normalize().tz_localize(None)
    df = df[df.index <= today].dropna(subset=["open", "close"])
    pub = [C.et(x, BAR_PUBLISH_HH) for x in df.index]
    keep = np.array([p <= fa for p in pub])                    # today's bar before 18:00 ET is not final: skip
    df, pub = df[keep], [p for p, k in zip(pub, keep) if k]
    rows = []
    for c in ("open", "high", "low", "close", "volume"):
        rows.append(pd.DataFrame({"series": f"{symbol}.{c}", "obs": df.index.strftime("%Y-%m-%d"),
                                  "value": df[c].to_numpy(), "published_at": [C.iso(p) for p in pub]}))
    if len(sp):
        # a split is public no later than its ex-date open; published at ex-date 00:00 ET
        rows.append(pd.DataFrame({"series": f"{symbol}.split", "obs": sp.index.strftime("%Y-%m-%d"),
                                  "value": sp.to_numpy(), "published_at": [C.iso(C.et(x, 0)) for x in sp.index]}))
    r = pd.concat(rows, ignore_index=True)
    r = r[pd.to_datetime(r["published_at"], utc=True) <= fa]
    return C.records("px", series=r["series"].tolist(), obs=r["obs"].tolist(), value=r["value"].tolist(),
                     published_at=r["published_at"].tolist(), fetched_at=fetched_at,
                     vintage=["snapshot"] * len(r), extra=[""] * len(r))


# =========================================================================== CPC degree days
def _read_cpc(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    hdr = next(l_ for l_ in lines if l_.startswith("Region|"))
    row = next(l_ for l_ in lines if l_.startswith("CONUS|"))
    days = [c for c in hdr.split("|")[1:] if re.fullmatch(r"\d{8}", c)]
    vals = row.split("|")[1:1 + len(days)]
    return [(pd.Timestamp(d), float(v) if v.strip() not in ("", "-999", "-9999") else np.nan) for d, v in zip(days, vals)]


def _cpc_manifests():
    out = {}
    for mp in (os.path.join(C.GAS, "manifest_cpc.json"), os.path.join(C.DDIR, "manifest_cpc.json")):
        if os.path.exists(mp):
            out.update(json.load(open(mp))["files"])
    return out


def parse_cpc_fcst(fetched_floor=None):
    man = _cpc_manifests()
    rows, bad = [], []
    for key, meta in sorted(man.items()):
        if "/cpc/fcst/" not in key:
            continue
        prod = os.path.basename(key)
        var = "hdd" if prod.startswith("UtilityGas") else "cdd"
        parts = key.split("/")
        issue = pd.Timestamp(f"{parts[-4]}-{parts[-3]}-{parts[-2]}")
        path = os.path.join(C.ROOT, key)
        if not os.path.exists(path):
            continue
        txt = open(path, encoding="latin-1").read()
        m = re.search(r"Issued (\d{4})UTC (\d{8})", txt)
        if not m or "CONUS|" not in txt or pd.Timestamp(m.group(2)) != issue:
            bad.append(key)
            continue
        lm = pd.to_datetime(meta.get("last_modified"), utc=True) if meta.get("last_modified") else pd.NaT
        iss_utc = issue.tz_localize("UTC")
        if pd.notna(lm) and pd.Timedelta(0) <= lm - iss_utc <= pd.Timedelta(days=3):
            pa, rule = lm, "last_modified"
        else:
            pa, rule = iss_utc + pd.Timedelta(days=1, hours=17), "issue_plus_1d_1700Z"
        fa = pd.to_datetime(meta.get("fetched_utc") or meta.get("downloaded_utc"), utc=True)
        if pd.isna(fa):
            fa = pd.Timestamp.now(tz="UTC")
        fa = max(fa, pa) if fa < pa else fa
        for d, v in _read_cpc(path):
            if np.isfinite(v):
                rows.append((var, d.strftime("%Y-%m-%d"), v, C.iso(pa), C.iso(fa), f"issue:{issue:%Y-%m-%d}", rule))
    df = pd.DataFrame(rows, columns=["series", "obs", "value", "published_at", "fetched_at", "vintage", "extra"])
    df["source"] = "cpc_fcst"
    return df[C.COLS], bad


def parse_cpc_obs():
    """Observed daily CONUS degree days from the latest snapshot of each yearly file (live first, then gas)."""
    man = _cpc_manifests()
    best = {}
    for key, meta in man.items():
        if "/cpc/obs/" not in key:
            continue
        parts = key.split("/")
        y, prod = parts[-2] if parts[-1].endswith(".txt") else parts[-2], os.path.basename(key).split(".txt")[0] + ".txt"
        fa = meta.get("fetched_utc") or meta.get("downloaded_utc") or ""
        k = (y, prod)
        if k not in best or fa > best[k][1]:
            best[k] = (key, fa)
    rows = []
    for (y, prod), (key, fa) in sorted(best.items()):
        var = "hdd" if prod.startswith("UtilityGas") else "cdd"
        fat = pd.to_datetime(fa, utc=True)
        for d, v in _read_cpc(os.path.join(C.ROOT, key)):
            pa = d.tz_localize("UTC") + pd.Timedelta(days=1, hours=18)
            if np.isfinite(v) and pa <= fat:
                rows.append((var, d.strftime("%Y-%m-%d"), v, C.iso(pa), C.iso(fat), "final_file", ""))
    df = pd.DataFrame(rows, columns=["series", "obs", "value", "published_at", "fetched_at", "vintage", "extra"])
    df["source"] = "cpc_obs"
    return df[C.COLS]


# =========================================================================== Open-Meteo previous runs (as-issued GFS)
from importlib import util as _iu  # noqa: E402
_spec = _iu.spec_from_file_location("gas_cities", os.path.join(C.ROOT, "tools", "gas", "cities.py"))
_gc = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_gc)
CITIES, BASE_F = _gc.CITIES, _gc.BASE_F
REGIONS = {"NE": ["new_york", "philadelphia", "boston", "pittsburgh"], "MW": ["chicago", "detroit", "minneapolis"],
           "S": ["atlanta", "dallas", "houston"]}
OM_START = "2021-04-01"
OM_LEADS = range(1, 8)
OM_URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={vars}"
          "&start_date={s}&end_date={e}&models=gfs_seamless&timezone=GMT")


def fetch_om(man, start=OM_START, end=None):
    end = end or (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=0)).strftime("%Y-%m-%d")
    vars_ = ",".join(f"temperature_2m_previous_day{k}" for k in OM_LEADS)
    out = {}
    for city, (lat, lon, _w) in CITIES.items():
        url = OM_URL.format(lat=lat, lon=lon, vars=vars_, s=start, e=end)
        body, _ = http_get(url, timeout=180)
        out[city] = C.save_raw("om_fcst", f"{city}_{start}_{end}.json", body, url, man)[0]
        time.sleep(1.5)
    return out


def parse_om(paths, fetched_at):
    temps = {}
    for city, p in paths.items():
        h = json.load(open(p))["hourly"]
        t = pd.to_datetime(h["time"])
        temps[city] = pd.DataFrame({k: h[f"temperature_2m_previous_day{k}"] for k in OM_LEADS}, index=t).astype(float)
    fa = pd.Timestamp(fetched_at)
    rows = []
    for reg, cities in REGIONS.items():
        w = np.array([CITIES[c][2] for c in cities])
        for k in OM_LEADS:
            hours = pd.concat([temps[c][k] for c in cities], axis=1)
            day = hours.groupby(hours.index.normalize())
            cnt = day.count().min(axis=1)
            tmean_c = day.mean()
            tf = tmean_c * 9 / 5 + 32
            ok = (cnt >= 20)
            tfw = (tf.to_numpy() * w).sum(1) / w.sum()
            hdd = np.maximum(0, BASE_F - tf.to_numpy()) @ w / w.sum()
            cdd = np.maximum(0, tf.to_numpy() - BASE_F) @ w / w.sum()
            for j, D in enumerate(tf.index):
                if not ok.iloc[j] or not np.isfinite(tfw[j]):
                    continue
                pa = D.tz_localize("UTC") - pd.Timedelta(days=k - 1) + pd.Timedelta(hours=6)
                if pa > fa:
                    continue
                for var, v in (("hdd", hdd[j]), ("cdd", cdd[j])):
                    rows.append((f"{reg}.{var}.d{k}", D.strftime("%Y-%m-%d"), round(float(v), 4), C.iso(pa),
                                 f"lead:{k}"))
    df = pd.DataFrame(rows, columns=["series", "obs", "value", "published_at", "vintage"])
    return C.records("om_fcst", series=df["series"].tolist(), obs=df["obs"].tolist(), value=df["value"].tolist(),
                     published_at=df["published_at"].tolist(), fetched_at=fetched_at, vintage=df["vintage"].tolist(),
                     extra=["gfs_seamless"] * len(df))


# =========================================================================== CFTC COT
COT_URL = ("https://publicreporting.cftc.gov/resource/72hh-3qpy.json?cftc_contract_market_code=023651"
           "&$limit=5000&$order=report_date_as_yyyy_mm_dd")
COT_FIELDS = {"open_interest_all": "oi", "m_money_positions_long_all": "mm_long",
              "m_money_positions_short_all": "mm_short", "prod_merc_positions_long": "pm_long",
              "prod_merc_positions_short": "pm_short", "swap_positions_long_all": "sw_long",
              "swap__positions_short_all": "sw_short", "nonrept_positions_long_all": "nr_long",
              "nonrept_positions_short_all": "nr_short"}
# government shutdowns delayed COT; reports dated in the window are treated as published at a conservative catch-up
COT_SHUTDOWNS = [(pd.Timestamp("2013-10-01"), pd.Timestamp("2013-10-15"), pd.Timestamp("2013-11-22")),
                 (pd.Timestamp("2018-12-24"), pd.Timestamp("2019-01-22"), pd.Timestamp("2019-03-08")),
                 (pd.Timestamp("2025-09-30"), pd.Timestamp("2025-11-11"), pd.Timestamp("2026-01-30"))]


def cot_release(tuesday):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    tue = pd.Timestamp(tuesday)
    for a, b, catch in COT_SHUTDOWNS:
        # backlog published chronologically after each shutdown: every Tuesday from the shutdown start up to the
        # catch-up date is treated as published at the catch-up date (red-team LRT-10)
        if a <= tue < catch:
            normal = C.et(tue + pd.Timedelta(days=3), 15, 30)
            return max(normal, C.et(catch, 15, 30)), "shutdown_catchup_conservative"
    hol = set(USFederalHolidayCalendar().holidays(tue - pd.Timedelta(days=7), tue + pd.Timedelta(days=14)))
    fri = tue + pd.Timedelta(days=3)
    if any((tue + pd.Timedelta(days=k)) in hol for k in range(0, 4)):
        d = fri + pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in hol:
            d += pd.Timedelta(days=1)
        return C.et(d, 15, 30), "holiday_next_business_day"
    return C.et(fri, 15, 30), "friday_1530ET"


def fetch_cot(man):
    body, _ = http_get(COT_URL)
    return C.save_raw("cot", "cot_023651.json", body, COT_URL, man)[0]


def parse_cot(path, fetched_at):
    data = json.load(open(path))
    rows = []
    fa = pd.Timestamp(fetched_at)
    for d in data:
        tue = pd.Timestamp(d["report_date_as_yyyy_mm_dd"][:10])
        pa, rule = cot_release(tue)
        if pa > fa:
            pa = fa
            rule += "|capped_at_fetch"
        for f, s in COT_FIELDS.items():
            if f in d:
                rows.append((s, tue.strftime("%Y-%m-%d"), float(d[f]), C.iso(pa), rule))
    df = pd.DataFrame(rows, columns=["series", "obs", "value", "published_at", "extra"])
    return C.records("cot", series=df["series"].tolist(), obs=df["obs"].tolist(), value=df["value"].tolist(),
                     published_at=df["published_at"].tolist(), fetched_at=fetched_at,
                     vintage=["snapshot"] * len(df), extra=df["extra"].tolist())


# =========================================================================== SEC EDGAR
EDGAR_TICKERS = {"producer": ["EQT", "EXE", "CTRA", "AR", "RRC", "CNX", "CRK"],
                 "lng": ["LNG", "CQP", "VG", "NEXT"],
                 "utility": ["ATO", "NI", "OGS", "SWX", "SR", "NFG"],
                 "pipeline": ["KMI", "WMB"]}
EDGAR_FORMS = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"}
EDGAR_CONCEPTS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NaturalGasProductionRevenue"]


def fetch_edgar(man):
    body, _ = http_get("https://www.sec.gov/files/company_tickers.json", ua=SEC_UA)
    C.save_raw("edgar", "company_tickers.json", body, "https://www.sec.gov/files/company_tickers.json", man)
    tick = {v["ticker"]: int(v["cik_str"]) for v in json.loads(body).values()}
    out = {"submissions": {}, "concepts": {}}
    for grp, ts in EDGAR_TICKERS.items():
        for t in ts:
            if t not in tick:
                C.log(f"edgar: {t} not in company_tickers")
                continue
            cik = f"{tick[t]:010d}"
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            b, _ = http_get(url, ua=SEC_UA)
            out["submissions"][t] = (grp, C.save_raw("edgar", f"sub_{t}.json", b, url, man)[0])
            time.sleep(0.3)
            for con in EDGAR_CONCEPTS:
                url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{con}.json"
                try:
                    b, _ = http_get(url, ua=SEC_UA)
                except urllib.error.HTTPError:
                    continue
                out["concepts"][(t, con)] = C.save_raw("edgar", f"concept_{t}_{con}.json", b, url, man)[0]
                time.sleep(0.3)
    return out


def _edgar_ts(s):
    # EDGAR acceptanceDateTime carries a 'Z' but is US/Eastern wall time: localize as ET (conservative either way)
    t = pd.Timestamp(s.replace("Z", "").split(".")[0])
    return t.tz_localize(C.ET, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def parse_edgar(got, fetched_at):
    fa = pd.Timestamp(fetched_at)
    rows, acc = [], {}
    for t, (grp, p) in got["submissions"].items():
        r = json.load(open(p))["filings"]["recent"]
        for form, adt, an, rd in zip(r["form"], r["acceptanceDateTime"], r["accessionNumber"], r["reportDate"]):
            pa = _edgar_ts(adt)
            acc[an] = pa
            if form in EDGAR_FORMS and pa <= fa:
                rows.append(("edgar_filing", f"{grp}.{t}.{form}", pa.strftime("%Y-%m-%d"), 1.0, C.iso(pa),
                             json.dumps({"accn": an, "report_date": rd, "form": form})))
    for (t, con), p in got["concepts"].items():
        units = json.load(open(p)).get("units", {})
        for u, facts in units.items():
            for f in facts:
                if f.get("form") not in EDGAR_FORMS or not f.get("fp"):
                    continue
                pa = acc.get(f["accn"]) or C.et(pd.Timestamp(f["filed"]), 23, 59)
                if pa > fa:
                    continue
                per = f.get("end")
                rows.append(("edgar_fact", f"{t}.{con}", per, float(f["val"]), C.iso(pa),
                             json.dumps({"accn": f["accn"], "start": f.get("start"), "fp": f.get("fp"),
                                         "fy": f.get("fy"), "form": f.get("form"), "unit": u})))
    df = pd.DataFrame(rows, columns=["source", "series", "obs", "value", "published_at", "extra"])
    df["fetched_at"] = C.iso(fa)
    df["vintage"] = np.where(df["source"] == "edgar_fact", "as_filed", "filing")
    return {s: g[C.COLS].drop_duplicates(subset=["series", "obs", "published_at", "extra"]) for s, g in df.groupby("source")}


# =========================================================================== Baker Hughes / EIA API
def probe_rigs():
    try:
        http_get("https://rigcount.bakerhughes.com/na-rig-count", tries=1)
        return True, "reachable (parser not implemented)"
    except Exception as e:  # noqa: BLE001
        return False, f"unavailable: {e}"


def eia_api_available():
    return bool(os.environ.get("EIA_API_KEY"))
