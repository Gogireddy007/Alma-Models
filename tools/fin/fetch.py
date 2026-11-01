"""
Download raw Binance public market data for Alma-Fin from https://data.binance.vision (no key, no account).
Raw monthly zips are stored unmodified under data/fin/raw/<market>/<kind>/<SYMBOL>/<file>.zip; each file's url,
UTC download time, sha256, byte size, S3 LastModified and the result of Binance's own .CHECKSUM (sha256) are
recorded in data/fin/manifest.json. A file whose sha256 differs from Binance's CHECKSUM is rejected (not stored).

    nice -n 10 .venv/bin/python tools/fin/fetch.py list   --market um                      # symbol listing (S3)
    nice -n 10 .venv/bin/python tools/fin/fetch.py get    --market um --kind klines --interval 1h \
               --symbols BTCUSDT,ETHUSDT --start 2020-01 --end 2026-10
    nice -n 10 .venv/bin/python tools/fin/fetch.py get    --market um --kind fundingRate --symbols ...

markets: um = USD-M futures (data/futures/um), spot = spot (data/spot).
kinds: klines (needs --interval), fundingRate (um only), premiumIndexKlines / markPriceKlines (um, --interval).
Kline columns (Binance): open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_volume, taker_buy_quote_volume, ignore. Spot files from 2025-01 use microsecond timestamps (handled in
data.py, not here). Only COMPLETE calendar months are downloaded (monthly files are published after month end).
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
import threading
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "fin", "raw")
MANIFEST = os.path.join(ROOT, "data", "fin", "manifest.json")
LISTING = os.path.join(ROOT, "data", "fin", "listing_{market}.json")
BASE = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = {"um": "data/futures/um/monthly/", "spot": "data/spot/monthly/"}
LOCK = threading.Lock()


def get(url, tries=8):
    for k in range(tries):
        try:
            url = urllib.parse.quote(url, safe=":/?&=%")
            req = urllib.request.Request(url, headers={"User-Agent": "alma-fin-research"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = 30 * (k + 1) if e.code in (403, 418, 429) else 3 * (k + 1)
            print(f"  retry {k + 1} in {wait}s: {url}: {e}", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  retry {k + 1}: {url}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def s3_list(prefix, delimiter="/"):
    """All CommonPrefixes and Contents (key, size, last_modified) under prefix (paginated)."""
    prefixes, contents, marker = [], [], ""
    while True:
        q = urllib.parse.quote
        url = f"{S3}?prefix={q(prefix)}&delimiter={q(delimiter)}" + (f"&marker={q(marker)}" if marker else "")
        body = get(url).decode()
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix>", body)
        contents += re.findall(r"<Key>(.*?)</Key><LastModified>(.*?)</LastModified>.*?<Size>(\d+)</Size>", body)
        if "<IsTruncated>true</IsTruncated>" not in body:
            break
        nm = re.search(r"<NextMarker>(.*?)</NextMarker>", body)
        marker = nm.group(1) if nm else (contents[-1][0] if contents else prefixes[-1])
    return prefixes, contents


def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return {"source": BASE, "license": "Binance public market data (data.binance.vision), free, no key",
            "files": {}, "missing": {}}


def save_manifest(man):
    tmp = MANIFEST + ".tmp"
    json.dump(man, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def cmd_list(a):
    """Every symbol under <market>/klines/ with the months available for interval a.interval (first/last month =
    listing/delisting proxies), from the S3 bucket listing (it keeps delisted symbols)."""
    root = PREFIX[a.market] + "klines/"
    syms, _ = s3_list(root)
    syms = [p[len(root):-1] for p in syms]
    if a.quote:
        syms = [s for s in syms if s.endswith(a.quote)]
    print(f"{len(syms)} symbols ending {a.quote!r}", flush=True)
    out = {}

    def one(s):
        _, c = s3_list(f"{root}{s}/{a.interval}/")
        ms = sorted({re.search(r"(\d{4}-\d{2})\.zip$", k).group(1): (lm, int(sz)) for k, lm, sz in c
                     if k.endswith(".zip")}.items())
        return s, {"months": [m for m, _ in ms], "last_modified": {m: v[0] for m, v in ms},
                   "bytes": {m: v[1] for m, v in ms}}
    with ThreadPoolExecutor(a.threads) as ex:
        for k, (s, v) in enumerate(ex.map(one, syms)):
            out[s] = v
            if k % 50 == 0:
                print(f"  {k}/{len(syms)} {s}: {v['months'][:1]}..{v['months'][-1:]}", flush=True)
    rec = {"market": a.market, "interval": a.interval, "listed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": f"{S3}?prefix={root}", "symbols": out}
    json.dump(rec, open(LISTING.format(market=a.market), "w"), indent=1, sort_keys=True)
    print(f"-> {LISTING.format(market=a.market)}")


def months(a, b):
    y, m = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    while (y, m) <= (yb, mb):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def days(a, b):
    import datetime as dt
    d, e = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    while d <= e:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def cmd_get(a):
    man = load_manifest()
    syms = [s for s in a.symbols.split(",") if s]
    if a.symbols_file:
        syms += [l.strip() for l in open(a.symbols_file) if l.strip() and not l.startswith("#")]
    listing = {}
    lp = LISTING.format(market=a.market)
    if os.path.exists(lp):
        listing = json.load(open(lp))["symbols"]
    kind = a.kind
    sub = f"{kind}/{{s}}/{a.interval}/" if kind != "fundingRate" else f"{kind}/{{s}}/"
    stem = f"{{s}}-{a.interval}-{{m}}.zip" if kind != "fundingRate" else f"{{s}}-fundingRate-{{m}}.zip"
    jobs = []
    for s in syms:
        avail = set(listing.get(s, {}).get("months", [])) if (kind == "klines" and listing) else None
        for m in (days(a.start, a.end) if a.daily else months(a.start, a.end)):
            if avail is not None and not a.daily and m not in avail:
                continue
            pre = PREFIX[a.market].replace("/monthly/", "/daily/") if a.daily else PREFIX[a.market]
            rel_url = pre + sub.format(s=s) + stem.format(s=s, m=m)
            rel = os.path.join("data", "fin", "raw", a.market, kind if kind == "fundingRate" else f"{kind}_{a.interval}",
                               s, stem.format(s=s, m=m))
            if rel in man["files"] and os.path.exists(os.path.join(ROOT, rel)):
                continue
            if rel in man["missing"] and not a.retry_missing:
                continue
            jobs.append((rel_url, rel))
    print(f"{len(jobs)} files to fetch", flush=True)
    done = [0]

    def one(job):
        rel_url, rel = job
        url = BASE + rel_url
        body = get(url)
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if body is None:
            with LOCK:
                man["missing"][rel] = {"url": url, "checked_utc": t, "status": 404}
            return
        ck = get(url + ".CHECKSUM")
        sha = hashlib.sha256(body).hexdigest()
        ck_sha = ck.decode().split()[0] if ck else None
        if ck_sha is not None and ck_sha != sha:
            with LOCK:
                man.setdefault("rejected", {})[rel] = {"url": url, "sha256": sha, "binance_checksum": ck_sha, "time_utc": t}
            print(f"  REJECT checksum mismatch {rel}", flush=True)
            return
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(body)
        with LOCK:
            man["files"][rel] = {"url": url, "downloaded_utc": t, "sha256": sha, "bytes": len(body),
                                 "binance_checksum_ok": ck_sha is not None}
            man["missing"].pop(rel, None)
            done[0] += 1
            if done[0] % 200 == 0:
                save_manifest(man)
                print(f"  {done[0]}/{len(jobs)} {rel}", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, jobs))
    save_manifest(man)
    print(f"fetched {done[0]} files; manifest has {len(man['files'])} files, {len(man['missing'])} missing")


def cmd_funding_api(a):
    """Funding settlements of a month that has no monthly archive yet (2026-10) from the public REST endpoint
    https://fapi.binance.com/fapi/v1/fundingRate (no key). Raw JSON stored unmodified, recorded in the manifest."""
    import calendar
    man = load_manifest()
    syms = [l.strip() for l in open(a.symbols_file) if l.strip()]
    y, m = map(int, a.month.split("-"))
    t0 = calendar.timegm((y, m, 1, 0, 0, 0)) * 1000
    t1 = calendar.timegm((y + (m == 12), m % 12 + 1, 1, 0, 0, 0)) * 1000 - 1
    for s in syms:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}&startTime={t0}&endTime={t1}&limit=1000"
        body = get(url)
        rel = f"data/fin/raw/um/fundingRate_api/{s}/{s}-fundingRate-{a.month}.json"
        os.makedirs(os.path.dirname(os.path.join(ROOT, rel)), exist_ok=True)
        open(os.path.join(ROOT, rel), "wb").write(body)
        man["files"][rel] = {"url": url, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                             "n_points": len(json.loads(body))}
        time.sleep(0.5)
    save_manifest(man)
    print(f"funding_api {a.month}: {len(syms)} symbols")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    l = sp.add_parser("list")
    l.add_argument("--market", default="um", choices=list(PREFIX))
    l.add_argument("--interval", default="1d")
    l.add_argument("--quote", default="USDT")
    l.add_argument("--threads", type=int, default=6)
    g = sp.add_parser("get")
    g.add_argument("--market", default="um", choices=list(PREFIX))
    g.add_argument("--kind", default="klines")
    g.add_argument("--interval", default="1h")
    g.add_argument("--symbols", default="")
    g.add_argument("--symbols-file", default="")
    g.add_argument("--start", default="2019-09")
    g.add_argument("--end", required=True)
    g.add_argument("--threads", type=int, default=6)
    g.add_argument("--retry-missing", action="store_true")
    g.add_argument("--daily", action="store_true", help="daily files (start/end = YYYY-MM-DD), for the last month")
    fa = sp.add_parser("funding_api")
    fa.add_argument("--symbols-file", required=True)
    fa.add_argument("--month", default="2026-10")
    a = ap.parse_args()
    {"list": cmd_list, "get": cmd_get, "funding_api": cmd_funding_api}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
