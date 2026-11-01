"""
Download raw data for Alma-Gas (free, no API key). Raw responses are stored unmodified in data/gas/raw/, with url,
download time (UTC), bytes and sha256 in data/gas/manifest.json. Downloaded files are untrusted data: they are only
parsed by tools/gas/data.py (pandas / regex), never executed.

    nice -n 10 .venv/bin/python tools/gas/fetch.py [--until 2026-10-31] [--only hh_fred,weather]

Sources
  hh_fred      FRED DHHNGSP: Henry Hub natural gas spot price, USD/MMBtu, daily (EIA via FRED; "." / empty = no trade)
  hh_eia       EIA RNGWHHD (same series at the source, daily table) -- cross-check of hh_fred
  ng_c1, ng_c2 EIA RNGC1 / RNGC2: NYMEX Henry Hub futures, contract 1 / 2 settlement, USD/MMBtu, daily
  storage      EIA ngshistory.xls: Weekly Natural Gas Storage Report history (Lower 48 + 5 regions, Bcf, week
               ending Friday; current vintage); revisions.xls: EIA's revision log; schedule.html: release schedule
               (alternate holiday release dates, 2025-2026); nw2 leaf table: Lower-48 weekly series (cross-check)
  cpc_obs      NOAA CPC observed daily degree days, yearly files: UtilityGas.Heating (HDD weighted by gas-heating
               households) and Population.Cooling (CDD), 9 Census divisions + CONUS (preliminary values, next-day)
  cpc_fcst     NOAA CPC 7-day degree-day FORECASTS (genuine ex-ante archive, from NDFD), one file per issue day
               2014-01-01 .. --until, same two products; HTTP Last-Modified recorded per file
               (separate manifest data/gas/manifest_cpc.json: ~9,400 small files)
  weather      Open-Meteo archive API (ERA5 / ERA5-Land reanalysis), daily mean 2 m temperature (deg F) for 10
               fixed demand cities (tools/gas/data.py CITIES); observed, not forecasts
"""
import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "gas", "raw")
MANIFEST = os.path.join(ROOT, "data", "gas", "manifest.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cities import CITIES  # noqa: E402

EIA = "https://www.eia.gov/dnav/ng/hist/"
FILES = {
    "hh_fred": [("DHHNGSP.csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP")],
    "hh_eia": [("rngwhhdd.htm", EIA + "rngwhhdd.htm")],
    "ng_c1": [("rngc1d.htm", EIA + "rngc1d.htm")],
    "ng_c2": [("rngc2d.htm", EIA + "rngc2d.htm")],
    "storage": [("ngshistory.xls", "https://ir.eia.gov/ngs/ngshistory.xls"),
                ("ngs_revisions.xls", "https://ir.eia.gov/ngs/revisions.xls"),
                ("ngs_schedule.html", "https://ir.eia.gov/ngs/schedule.html"),
                ("nw2_epg0_swo_r48_bcfw.htm", EIA + "nw2_epg0_swo_r48_bcfw.htm")],
}
WEATHER_START = "1997-01-01"


def weather_files(until):
    out = []
    for name, (lat, lon, _w) in CITIES.items():
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f&start_date=%s&end_date=%s"
               "&daily=temperature_2m_mean&temperature_unit=fahrenheit&timezone=America%%2FNew_York"
               % (lat, lon, WEATHER_START, until))
        out.append((f"weather_{name}.json", url))
    return out


CPC = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/"
CPC_PRODUCTS = ("UtilityGas.Heating.txt", "Population.Cooling.txt")
CPC_OBS_START, CPC_FCST_START = 1997, "2014-01-01"


def get(url, tries=8, with_headers=False, allow_404=False):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"}),
                                        timeout=180) as r:
                body = r.read()
                return (body, dict(r.headers)) if with_headers else body
        except urllib.error.HTTPError as e:
            if allow_404 and e.code == 404:
                return (None, {}) if with_headers else None
            wait = 60 * (k + 1) if e.code == 429 else 5 * (k + 1)
            print(f"  retry {k + 1} in {wait}s: {e}", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 60 * (k + 1) if "429" in str(e) else 5 * (k + 1)
            print(f"  retry {k + 1} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def needed_issue_days():
    """Issue days the storage task can use: each origin day (thu = release day of the previous report, fri =
    max(week end, that release day)) and the day before it (fallback when the same-day file was posted after the
    origin). Derived from data/gas/storage.csv (release dates only; no values)."""
    import datetime as dt
    import pandas as pd
    st = pd.read_csv(os.path.join(ROOT, "data", "gas", "storage.csv"), parse_dates=["week", "avail"])
    days = set()
    for k in range(1, len(st)):
        prev = st.avail.iloc[k - 1]
        for o in (prev, max(st.week.iloc[k], prev)):
            for lag in (0, 1):
                days.add((o - pd.Timedelta(days=lag)).date())
    return days


def fetch_cpc(until, which):
    """CPC degree days: observed yearly files and daily forecast issues; raw text stored under data/gas/raw/cpc/."""
    import datetime as dt
    mp = os.path.join(ROOT, "data", "gas", "manifest_cpc.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {"files": {}, "missing": {}}
    jobs = []
    if "cpc_obs" in which:
        for y in range(CPC_OBS_START, int(until[:4]) + 1):
            for prod in CPC_PRODUCTS:
                jobs.append((f"cpc/obs/{y}/{prod}", f"{CPC}daily_data/{y}/{prod}", y == int(until[:4])))
    if "cpc_fcst" in which:
        d, end = dt.date.fromisoformat(CPC_FCST_START), dt.date.fromisoformat(until)
        need = needed_issue_days()
        while d <= end:
            if d not in need:
                d += dt.timedelta(days=1)
                continue
            for prod in CPC_PRODUCTS:
                rel = f"cpc/fcst/{d:%Y/%m/%d}/{prod}"
                jobs.append((rel, f"{CPC}daily_forecasts_7day/{d:%Y/%m/%d}/{prod}", False))
            d += dt.timedelta(days=1)
    todo = [(rel, url) for rel, url, refresh in jobs
            if refresh or not ((f"data/gas/raw/{rel}" in man["files"] and os.path.exists(os.path.join(ROOT, "data/gas/raw", rel)))
                               or f"data/gas/raw/{rel}" in man["missing"])]

    def one(job):
        rel, url = job
        body, hdr = get(url, with_headers=True, allow_404=True)
        return rel, url, body, hdr, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    from concurrent.futures import ThreadPoolExecutor
    n = 0
    with ThreadPoolExecutor(3) as ex:                      # main thread writes files and the manifest
        for rel, url, body, hdr, ts in ex.map(one, todo):
            key = f"data/gas/raw/{rel}"
            if body is None:
                man["missing"][key] = {"url": url, "checked_utc": ts}
            else:
                dst = os.path.join(ROOT, key)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "wb").write(body)
                man["files"][key] = {"url": url, "downloaded_utc": ts, "last_modified": hdr.get("Last-Modified"),
                                     "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            n += 1
            if n % 500 == 0:
                json.dump(man, open(mp, "w"), indent=0, sort_keys=True)
                print(f"cpc: {n}/{len(todo)} fetched (last {rel})", flush=True)
    json.dump(man, open(mp, "w"), indent=0, sort_keys=True)
    print(f"cpc: done, {len(man['files'])} files, {len(man['missing'])} missing", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="2026-10-30", help="last weather day requested (Open-Meteo end_date)")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"files": {}}
    man["note"] = ("raw files exactly as downloaded; sha256 of the bytes on disk. Licences: FRED/EIA US public domain; "
                   "Open-Meteo CC BY 4.0 (ERA5: Copernicus C3S).")
    groups = dict(FILES, weather=weather_files(a.until))
    names = [s for s in a.only.split(",") if s] or list(groups) + ["cpc_obs", "cpc_fcst"]
    cpc = [g for g in names if g.startswith("cpc")]
    names = [g for g in names if not g.startswith("cpc")]
    for g in names:
        for fn, url in groups[g]:
            body = get(url)
            if len(body) < 500:
                raise RuntimeError(f"{fn}: suspiciously short response ({len(body)} bytes): {body[:200]!r}")
            dst = os.path.join(RAW, fn)
            open(dst, "wb").write(body)
            man["files"][f"data/gas/raw/{fn}"] = {
                "group": g, "url": url, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            print(f"{g}: {fn} {len(body)} bytes", flush=True)
            json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)
            time.sleep(2 if g == "weather" else 1)
    json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)
    if cpc:
        fetch_cpc(a.until, cpc)


if __name__ == "__main__":
    sys.exit(main())
