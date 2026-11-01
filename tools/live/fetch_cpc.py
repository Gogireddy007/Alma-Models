"""
CPC 7-day degree-day forecast issues (NDFD-based, CONUS gas-utility-weighted HDD and population-weighted CDD) for
every calendar day, plus the yearly observed files. Issues already in the Alma-Gas archive (data/gas/raw/cpc, read
only) are not re-downloaded; the rest go to data/live/raw/cpc/{fcst/YYYY/MM/DD,obs/YYYY}/<product> with url, fetch
time, HTTP Last-Modified and sha256 in data/live/manifest_cpc.json. Idempotent (skips files already held; the
current year's observed files and issues of the last 3 days are refreshed as new snapshots).

    nice -n 10 .venv/bin/python tools/live/fetch_cpc.py [--start 2014-01-01] [--end today]
"""
import os
import sys
import json
import time
import argparse
import datetime as dt
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CPC = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/"
PRODUCTS = ("UtilityGas.Heating.txt", "Population.Cooling.txt")
MAN = os.path.join(C.DDIR, "manifest_cpc.json")
GAS_MAN = os.path.join(C.GAS, "manifest_cpc.json")


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "alma-live-gas research (curl/8.7.1)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, {}
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None, {"error": "failed"}


def load():
    return json.load(open(MAN)) if os.path.exists(MAN) else {"files": {}, "missing": {}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--obs", action="store_true", help="also refresh the observed yearly files")
    a = ap.parse_args(argv)
    end = dt.date.fromisoformat(a.end) if a.end else dt.datetime.now(dt.timezone.utc).date()
    man = load()
    gas = json.load(open(GAS_MAN))["files"] if os.path.exists(GAS_MAN) else {}
    jobs = []
    d = dt.date.fromisoformat(a.start)
    recent = end - dt.timedelta(days=3)
    while d <= end:
        for p in PRODUCTS:
            relp = f"cpc/fcst/{d:%Y/%m/%d}/{p}"
            have_gas = f"data/gas/raw/{relp}" in gas
            have = f"data/live/raw/{relp}" in man["files"]
            missing = f"data/live/raw/{relp}" in man["missing"] and d < recent
            if not (have_gas or have or missing) or (d >= recent and not have):
                jobs.append((relp, f"{CPC}daily_forecasts_7day/{d:%Y/%m/%d}/{p}"))
        d += dt.timedelta(days=1)
    if a.obs:
        for y in range(1997, end.year + 1):
            for p in PRODUCTS:
                relp = f"cpc/obs/{y}/{p}"
                if y >= end.year - 1 or (f"data/live/raw/{relp}" not in man["files"] and f"data/gas/raw/{relp}" not in gas):
                    jobs.append((relp, f"{CPC}daily_data/{y}/{p}"))
    C.log(f"cpc: {len(jobs)} files to fetch")

    def one(job):
        relp, url = job
        body, hdr = get(url)
        return relp, url, body, hdr, C.iso(C.utcnow())

    n = 0
    with ThreadPoolExecutor(3) as ex:
        for relp, url, body, hdr, fa in ex.map(one, jobs):
            key = f"data/live/raw/{relp}"
            if body is None:
                if "error" not in hdr:
                    man["missing"][key] = {"url": url, "checked_utc": fa}
                continue
            path = os.path.join(C.ROOT, key)
            if relp.startswith("cpc/obs/") and os.path.exists(path):
                # observed yearly files are revised/extended: keep every snapshot
                path = path + "." + fa.replace(":", "").replace("-", "")
                key = C.rel(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(body)
            man["files"][key] = {"url": url, "fetched_utc": fa, "last_modified": hdr.get("Last-Modified"),
                                 "sha256": C.sha256_bytes(body), "bytes": len(body)}
            man["missing"].pop(key, None)
            n += 1
            if n % 200 == 0:
                json.dump(man, open(MAN, "w"), indent=0, sort_keys=True)
                C.log(f"cpc: {n}/{len(jobs)} (last {relp})")
    json.dump(man, open(MAN, "w"), indent=0, sort_keys=True)
    C.log(f"cpc: done, fetched {n}, held {len(man['files'])}, missing {len(man['missing'])}")
    return n


if __name__ == "__main__":
    main()
