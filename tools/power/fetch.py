"""
Download raw data for Alma-Power from the Fraunhofer ISE Energy-Charts public API (https://api.energy-charts.info,
no key). Raw responses are stored unmodified, one JSON per (series, calendar year), with url, download time (UTC)
and sha256 in data/power/manifest.json.

    nice -n 10 .venv/bin/python tools/power/fetch.py [--until 2026-10-31] [--only price_de_lu,...]

Series (start/end query parameters are local Europe/Berlin dates, end inclusive; we de-duplicate by unix_seconds):
  price_de_at_lu   /price?bzn=DE-AT-LU     2015-01 .. 2018-09   (old zone; DE-AT split on 2018-10-01)
  price_de_lu      /price?bzn=DE-LU        2018-10 .. latest    (hourly until 2025-09-30, 15-min from 2025-10-01)
  public_power     /public_power?country=de                     (15-min actual net generation by type, load)
  fc_{load,solar,wind_onshore,wind_offshore}  /public_power_forecast?forecast_type=day-ahead
                   (TSO day-ahead forecasts; NO publication-time / vintage field -> NOT used by any pre-registered
                    model, see docs/power_prereg.md section 2; kept only for documentation)
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
RAW = os.path.join(ROOT, "data", "power", "raw")
MANIFEST = os.path.join(ROOT, "data", "power", "manifest.json")
API = "https://api.energy-charts.info"

SERIES = {
    "price_de_at_lu": ("/price?bzn=DE-AT-LU", "2015-01", "2018-09"),
    "price_de_lu": ("/price?bzn=DE-LU", "2018-10", None),
    "public_power": ("/public_power?country=de", "2015-01", None),
    "fc_load": ("/public_power_forecast?country=de&production_type=load&forecast_type=day-ahead", "2015-01", None),
    "fc_solar": ("/public_power_forecast?country=de&production_type=solar&forecast_type=day-ahead", "2015-01", None),
    "fc_wind_onshore": ("/public_power_forecast?country=de&production_type=wind_onshore&forecast_type=day-ahead",
                        "2015-01", None),
    "fc_wind_offshore": ("/public_power_forecast?country=de&production_type=wind_offshore&forecast_type=day-ahead",
                         "2015-01", None),
}


def months(a, b):
    y, m = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    while (y, m) <= (yb, mb):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def month_range(y, m, until):
    import calendar
    s = f"{y:04d}-{m:02d}-01"
    e = f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
    return s, min(e, until)


def get(url, tries=12):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "alma-power-research"}),
                                        timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            wait = min(300, 30 * (k + 1)) if "429" in str(e) else 5 * (k + 1)
            print(f"  retry {k + 1} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"source": API, "files": {}}
    man["license"] = ("Energy-Charts (Fraunhofer ISE); price data CC BY 4.0 from Bundesnetzagentur | SMARD.de; "
                      "see license_info in each price file")
    names = [s for s in a.only.split(",") if s] or list(SERIES)
    last_month = a.until[:7]
    for name in names:
        path, m0, m1 = SERIES[name]
        y0, y1 = int(m0[:4]), int((m1 or last_month)[:4])
        os.makedirs(os.path.join(RAW, name), exist_ok=True)
        for y in range(y0, y1 + 1):
            rel = f"data/power/raw/{name}/{y:04d}.json"
            dst = os.path.join(ROOT, rel)
            recent = m1 is None and y == y1
            if os.path.exists(dst) and rel in man["files"] and not recent:
                continue
            s = max(f"{y:04d}-01-01", m0 + "-01")
            e = min(f"{y:04d}-12-31", a.until) if m1 is None else min(f"{y:04d}-12-31", month_range(*map(int, m1.split("-")), a.until)[1])
            url = f"{API}{path}&start={s}&end={e}"
            body = get(url)
            try:
                d = json.loads(body)
                n = len(d.get("unix_seconds") or [])
            except Exception:
                d, n = None, 0
            if not isinstance(d, dict):
                print(f"{name} {y}: non-JSON / error response ({body[:120]!r})", flush=True)
                man.setdefault("errors", {})[rel] = {"url": url, "body": body[:300].decode("utf8", "replace"),
                                                     "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                continue
            open(dst, "wb").write(body)
            man["files"][rel] = {"url": url, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "n_points": n}
            man.get("errors", {}).pop(rel, None)
            print(f"{name} {y}: {n} points", flush=True)
            json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)
            time.sleep(15)
    json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
