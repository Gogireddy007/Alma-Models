"""
Download daily OHLCV for the 13 US ETFs of the Alma-Fin cross-market transfer arm (SPY, QQQ and the 11 SPDR sector
ETFs) from the public Yahoo Finance chart endpoint (no key; unofficial). Stooq is not used: it now serves a
JavaScript bot check, which we do not bypass. Raw JSON responses are stored unmodified in data/fin/raw/etf/<SYM>.json;
url, UTC download time and sha256 go into data/fin/manifest.json (section "files"), exactly like the Binance files.
The raw snapshot is pinned by sha256 (Yahoo may revise; our pinned copy is what every run uses).

    nice -n 10 .venv/bin/python tools/fin/fetch_etf.py [--end 2026-10-31]
"""
import os
import sys
import json
import time
import hashlib
import argparse
import calendar
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import ROOT, load_manifest, save_manifest   # noqa: E402

ETFS = ["SPY", "QQQ", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1={p1}&period2={p2}&interval=1d"
       "&events=div%2Csplit&includeAdjustedClose=true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-10-31")
    a = ap.parse_args()
    p1 = calendar.timegm(time.strptime(a.start, "%Y-%m-%d"))
    p2 = calendar.timegm(time.strptime(a.end, "%Y-%m-%d")) + 86400
    man = load_manifest()
    os.makedirs(os.path.join(ROOT, "data", "fin", "raw", "etf"), exist_ok=True)
    for s in ETFS:
        url = URL.format(s=s, p1=p1, p2=p2)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (alma-fin-research)"})
        body = urllib.request.urlopen(req, timeout=60).read()
        d = json.loads(body)
        n = len(d["chart"]["result"][0]["timestamp"])
        rel = f"data/fin/raw/etf/{s}.json"
        open(os.path.join(ROOT, rel), "wb").write(body)
        man["files"][rel] = {"url": url, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "n_points": n}
        print(f"{s}: {n} rows", flush=True)
        time.sleep(2)
    save_manifest(man)


if __name__ == "__main__":
    sys.exit(main())
