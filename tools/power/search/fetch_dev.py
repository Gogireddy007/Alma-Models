"""Architect's own copy of the Energy-Charts raw data, fetched with the split end IN THE REQUEST (DEV: 2025-06-30;
VAL months 2025-07..2025-10 only when --split val is run, once, at the end). Same JSON format as the lead's
data/power/raw (tools/power/fetch.py); stored under runs/power/search/raw/<split>/<series>/YYYY-MM.json."""
import calendar, hashlib, json, os, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
API = "https://api.energy-charts.info"
SER = {"price_de_at_lu": ("/price?bzn=DE-AT-LU", "2015-01", "2018-09"),
       "price_de_lu": ("/price?bzn=DE-LU", "2018-10", None),
       "public_power": ("/public_power?country=de", "2015-01", None)}
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
END = {"dev": "2025-06", "val": "2025-10"}[split]
FIRST = {"dev": None, "val": "2025-07"}[split]
assert END <= "2025-10"
out = os.path.join(ROOT, "runs", "power", "search", "raw", split)
man = {}
for name, (path, m0, m1) in SER.items():
    y, m = map(int, m0.split("-"))
    os.makedirs(os.path.join(out, name), exist_ok=True)
    while f"{y:04d}-{m:02d}" <= (m1 or END):
        tag = f"{y:04d}-{m:02d}"
        if FIRST is None or tag >= FIRST:
            dst = os.path.join(out, name, tag + ".json")
            if not os.path.exists(dst):
                e = f"{tag}-{calendar.monthrange(y, m)[1]:02d}"
                url = f"{API}{path}&start={tag}-01&end={e}"
                for k in range(5):
                    try:
                        body = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "alma-power-research"}), timeout=120).read()
                        json.loads(body); break
                    except Exception as ex:
                        print("retry", url, ex, flush=True); time.sleep(5 * (k + 1))
                open(dst, "wb").write(body)
                man[f"{name}/{tag}"] = {"url": url, "sha256": hashlib.sha256(body).hexdigest()}
                print(name, tag, len(body), flush=True)
                time.sleep(0.3)
        m += 1
        if m == 13: y, m = y + 1, 1
json.dump(man, open(os.path.join(out, f"manifest_{int(time.time())}.json"), "w"), indent=1)
