"""Rebuild (cold start) or top up (daily) the public data for all four projects. Idempotent.

    .venv/bin/python tools/deploy/refresh_data.py [--today 2026-11-05]

Cold start: every pinned raw file of the committed manifests is downloaded and checked against its sha256 pin before
anything new is added (exit 1 on a mismatch). Then the newly published data is fetched and the processed tables are
built. The fetchers rewrite data/*/manifest*.json; this script copies the new manifests to
runs/deploy_daily/manifests/<today>/ and restores the committed (pinned) versions, so the pins never drift.
Open-Meteo weather is not fetched (it rate-limits shared egress IPs); see tools/deploy/gas_daily_noweather.py.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
MANIFESTS = ["data/commodity/manifest.json", "data/power/manifest.json", "data/gas/manifest.json",
             "data/gas/manifest_cpc.json", "data/fin/manifest.json"]


def sh(*args, check=True):
    print("+", " ".join(args), flush=True)
    return subprocess.run([PY, *args], cwd=ROOT, check=check).returncode


def get(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (alma-research)"})
    body = urllib.request.urlopen(req, timeout=120).read()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(body)
    print(f"  {os.path.relpath(dst, ROOT)}: {len(body)} bytes", flush=True)


def oil(today):
    for sid in ("DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"):
        get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", os.path.join(ROOT, f"data/commodity/raw/{sid}.csv"))
    get("https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W",
        os.path.join(ROOT, "data/commodity/raw/WCESTUS1_leaf.html"))
    sh("tools/commodity/data.py", "build", today)


def power(today):
    sh("tools/deploy/fetch_from_manifest.py", "data/power/manifest.json", "--only", "/price_de", check=False)
    sh("tools/deploy/fetch_from_manifest.py", "data/power/manifest.json", "--only", "/public_power/", check=False)
    tomorrow = (dt.date.fromisoformat(today) + dt.timedelta(days=1)).isoformat()
    sh("tools/power/fetch.py", "--only", "price_de_lu,public_power", "--until", tomorrow)
    sh("tools/power/data.py", "build")


def gas(today):
    sh("tools/gas/fetch.py", "--only", "storage,hh_fred,hh_eia,ng_c1,ng_c2", "--until", today)
    sh("tools/deploy/gas_daily_noweather.py")                         # also writes storage.csv (release dates)
    sh("tools/deploy/fetch_from_manifest.py", "data/gas/manifest_cpc.json")
    code = ("import sys, datetime as dt; sys.path.insert(0, 'tools/gas'); sys.argv = ['x']; import fetch as F; "
            "o = F.needed_issue_days; t = dt.date.fromisoformat(sys.argv_today); "
            "F.needed_issue_days = lambda: o() | {t - dt.timedelta(days=k) for k in range(0, 10)}; "
            "F.fetch_cpc(sys.argv_today, ['cpc_obs', 'cpc_fcst'])").replace("sys.argv_today", repr(today))
    subprocess.run([PY, "-c", code], cwd=ROOT, check=True)        # forecast issues of the coming origins too
    sh("tools/gas/storage.py", "build")


def fin(today):
    sh("tools/deploy/fetch_from_manifest.py", "data/fin/manifest.json", "--only", "/um/")
    if not os.path.exists(os.path.join(ROOT, "data/fin/listing_um.json")):
        sh("tools/fin/fetch.py", "list", "--market", "um")
    uni = os.path.join(ROOT, "data/fin/universe.json")
    if not os.path.exists(uni):
        cands = sorted(os.listdir(os.path.join(ROOT, "data/fin/raw/um/klines_1d")))
        open(os.path.join(ROOT, "data/fin/universe_candidates.txt"), "w").write("\n".join(cands) + "\n")
        sh("tools/fin/universe.py")
    u = json.load(open(uni))
    syms = ",".join(u["train"] + u["heldout"])
    t = dt.date.fromisoformat(today)
    last_month = (t.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
    for kind, extra in (("klines", ["--interval", "5m"]), ("fundingRate", [])):   # complete months since Oct 2026
        sh("tools/fin/fetch.py", "get", "--market", "um", "--kind", kind, *extra, "--symbols", syms,
           "--start", "2026-10", "--end", last_month, check=False)
    yday = (t - dt.timedelta(days=1)).isoformat()
    sh("tools/fin/fetch.py", "get", "--market", "um", "--kind", "klines", "--interval", "5m", "--symbols", syms,
       "--daily", "--start", "2026-10-28", "--end", yday, check=False)       # this month, one file per day
    sh("tools/fin/data.py", "build")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    ap.add_argument("--only", default="oil,power,gas,fin")
    a = ap.parse_args()
    keep = os.path.join(ROOT, "runs", "deploy_daily", "manifests", a.today)
    os.makedirs(keep, exist_ok=True)
    failed = []
    try:
        for p in a.only.split(","):
            try:
                {"oil": oil, "power": power, "gas": gas, "fin": fin}[p](a.today)
            except Exception as e:  # noqa: BLE001  (one source down must not hide the others; reported below)
                print(f"!! {p} refresh failed: {type(e).__name__}: {e}", flush=True)
                failed.append(p)
    finally:
        for m in MANIFESTS:                     # keep only entries that are new or differ from the committed pins
            src = os.path.join(ROOT, m)
            if not os.path.exists(src):
                continue
            cur = json.load(open(src)).get("files", {})
            pinned = subprocess.run(["git", "show", f"HEAD:{m}"], cwd=ROOT, capture_output=True, text=True).stdout
            pin = json.loads(pinned).get("files", {}) if pinned else {}
            delta = {k: v for k, v in cur.items() if pin.get(k, {}).get("sha256") != v.get("sha256")}
            json.dump({"note": f"entries new or changed vs the committed {m}", "files": delta},
                      open(os.path.join(keep, m.replace("/", "_")), "w"), indent=1, sort_keys=True)
        subprocess.run(["git", "checkout", "--", *MANIFESTS], cwd=ROOT, check=False)
    print(f"refresh done; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
