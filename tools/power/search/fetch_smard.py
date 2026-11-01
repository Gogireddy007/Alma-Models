"""Architect's split-limited copy of hourly DE-LU price and German load / wind / solar actuals from the SMARD.de
chart-data API (Bundesnetzagentur; the same source Energy-Charts cites for prices). Used because the Energy-Charts API
was rate-limiting (HTTP 429) while the lead fetched; the lead's data is used for the harness.

Weekly chunks are downloaded and TRUNCATED IN MEMORY to timestamps < split end + 1 day (Europe/Berlin) before they are
written, so the DEV cache holds nothing after 2025-06-30 and the VAL cache nothing after 2025-10-31.
    runs/power/search/smard/<split>/<series>.json  = {"unix_ms": [...], "value": [...], "files": {url: sha256}}
"""
import hashlib, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
B = "https://www.smard.de/app/chart_data"
SER = {"price": (4169, "DE-LU"), "load": (410, "DE"), "wind_on": (4067, "DE"), "wind_off": (1225, "DE"),
       "solar": (4068, "DE")}
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
END = {"dev": "2025-07-01", "val": "2025-11-01"}[split]          # exclusive, local midnight
assert END <= "2025-11-01"
end_ms = int(datetime.fromisoformat(END).replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp() * 1000)
start_ms = int(datetime(2018, 9, 24, tzinfo=ZoneInfo("Europe/Berlin")).timestamp() * 1000)
out = os.path.join(ROOT, "runs", "power", "search", "smard", split)
os.makedirs(out, exist_ok=True)


def get(url):
    for k in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "alma-power-research"}), timeout=60).read()
        except Exception as e:
            print("retry", url, e, flush=True); time.sleep(3 * (k + 1))
    raise RuntimeError(url)


for name, (f, reg) in SER.items():
    idx = json.loads(get(f"{B}/{f}/{reg}/index_hour.json"))["timestamps"]
    idx = [t for t in idx if start_ms - 8 * 86400000 <= t < end_ms]
    urls = [f"{B}/{f}/{reg}/{f}_{reg}_hour_{t}.json" for t in idx]
    with ThreadPoolExecutor(3) as ex:
        bodies = list(ex.map(get, urls))
    pts, files = {}, {}
    for u, body in zip(urls, bodies):
        files[u] = hashlib.sha256(body).hexdigest()
        for t, v in json.loads(body)["series"]:
            if t < end_ms and v is not None:          # truncation at the split end, before anything is stored
                pts[t] = v
    ts = sorted(pts)
    assert not ts or ts[-1] < end_ms
    json.dump({"unix_ms": ts, "value": [pts[t] for t in ts], "files": files, "end_exclusive": END},
              open(os.path.join(out, name + ".json"), "w"))
    print(name, len(ts), datetime.fromtimestamp(ts[0] / 1000), datetime.fromtimestamp(ts[-1] / 1000), flush=True)
