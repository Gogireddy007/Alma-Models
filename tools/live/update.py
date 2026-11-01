"""
Alma-Live Gas updater: fetch every source, append NEW point-in-time records (data/live/pit/<source>.csv,
append-only, prefix-pinned in data/live/manifest.json), log what arrived and when (data/live/update_log.jsonl).
Idempotent: unchanged raw responses are not re-stored and already-held records are not re-appended. Safe to run on a
schedule (docs/live_ops.md); a lock file prevents overlapping runs.

    nice -n 10 .venv/bin/python tools/live/update.py                 # all sources
    nice -n 10 .venv/bin/python tools/live/update.py --only px,cot   # subset
    nice -n 10 .venv/bin/python tools/live/update.py --snap-curve    # also snapshot the NG forward curve now

Sources: storage px cpc om cot edgar curve (rigs: probed, unavailable; eia_api: only with EIA_API_KEY).
"""
import os
import sys
import json
import time
import fcntl
import argparse
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import sources as S  # noqa: E402

ALL = ["storage", "px", "cpc", "om", "cot", "edgar", "curve"]


def _log(entry):
    os.makedirs(C.DDIR, exist_ok=True)
    with open(C.UPDATE_LOG, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _summ(df):
    if df is None or not len(df):
        return {}
    pa = pd.to_datetime(df["published_at"], utc=True)
    return {"max_published_at": C.iso(pa.max()), "min_published_at": C.iso(pa.min()),
            "series": int(df["series"].nunique())}


def run_storage(man, now):
    paths = S.fetch_storage(man)
    return {"storage": S.parse_storage(paths, now)}


def run_px(man, now):
    paths = S.fetch_px(man)
    out = []
    for sym, p in paths.items():
        out.append(S.parse_px(p, sym, now))
    df = pd.concat(out, ignore_index=True)
    # a bar that changed since we first stored it (Yahoo revision) is appended as a new vintage published at THIS
    # fetch (never back-dated); unchanged bars are de-duplicated by append_records
    p = C.pit_path("px")
    if os.path.exists(p):
        old = pd.read_csv(p, dtype={"series": str, "obs": str}).drop_duplicates(["series", "obs"], keep="last")
        m = df.merge(old[["series", "obs", "value"]], on=["series", "obs"], how="left", suffixes=("", "_old"))
        chg = m["value_old"].notna() & ~pd.Series(np.isclose(m["value"], m["value_old"], rtol=1e-9, atol=1e-12))
        df.loc[chg.to_numpy(), "published_at"] = C.iso(now)
        df.loc[chg.to_numpy(), "vintage"] = "revised_snapshot"
    return {"px": df}


def run_cpc(man, now):
    import fetch_cpc
    fetch_cpc.main(["--start", (now - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), "--obs"])
    f, bad = S.parse_cpc_fcst()
    o = S.parse_cpc_obs()
    if bad:
        man.setdefault("notes", {})["cpc_bad_files"] = len(bad)
    return {"cpc_fcst": f, "cpc_obs": o}


def run_om(man, now):
    paths = S.fetch_om(man)
    return {"om_fcst": S.parse_om(paths, now)}


def run_cot(man, now):
    return {"cot": S.parse_cot(S.fetch_cot(man), now)}


def run_edgar(man, now):
    return S.parse_edgar(S.fetch_edgar(man), now)


def run_curve(man, now):
    """Forward-only NG curve snapshot: Yahoo quote of the next 6 NG contracts (regularMarketPrice and time). Each
    point is published at its own regularMarketTime (exchange-reported) and never before this fetch's start."""
    rows = []
    for y, m, lt, sym in C.ng_contracts_around(now.tz_convert(C.ET).tz_localize(None), 6):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1d&interval=1m"
        try:
            body, _ = S.http_get(url)
        except Exception as e:  # noqa: BLE001
            C.log(f"curve {sym}: {e}")
            continue
        C.save_raw("curve", f"{sym}.json", body, url, man, fetched_at=now)
        meta = json.loads(body)["chart"]["result"][0]["meta"]
        px, mt = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
        if px is None or mt is None:
            continue
        pa = min(pd.Timestamp(mt, unit="s", tz="UTC"), now)
        rows.append((f"{sym}.last", C.iso(pa), float(px), C.iso(pa), f"last_trade:{lt:%Y-%m-%d}"))
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=["series", "obs", "value", "published_at", "extra"])
    return {"curve": C.records("curve", series=df["series"].tolist(), obs=df["obs"].tolist(),
                               value=df["value"].tolist(), published_at=df["published_at"].tolist(), fetched_at=now,
                               vintage=["snapshot"] * len(df), extra=df["extra"].tolist())}


RUN = {"storage": run_storage, "px": run_px, "cpc": run_cpc, "om": run_om, "cot": run_cot, "edgar": run_edgar,
       "curve": run_curve}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--snap-curve", action="store_true")
    a = ap.parse_args(argv)
    names = [s for s in a.only.split(",") if s] or [n for n in ALL if n != "curve"]
    if a.snap_curve and "curve" not in names:
        names.append("curve")
    os.makedirs(C.DDIR, exist_ok=True)
    lock = open(os.path.join(C.DDIR, ".update.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another update is running; exit")
        return 1
    man = C.load_manifest()
    run_id = C.iso(C.utcnow())
    ok = True
    for n in names:
        now = C.utcnow()
        t0 = time.time()
        try:
            got = RUN[n](man, now)
            for src, df in got.items():
                k = C.append_records(src, df, man)
                ent = {"run": run_id, "source": src, "fetched_at": C.iso(now), "status": "ok", "n_parsed": int(len(df)),
                       "n_appended": int(k), "secs": round(time.time() - t0, 1),
                       "pit_sha256": man["pit"].get(C.rel(C.pit_path(src)), {}).get("sha256"), **_summ(df)}
                _log(ent)
                C.log(f"{src}: parsed {len(df)}, appended {k}")
        except Exception as e:  # noqa: BLE001
            ok = False
            _log({"run": run_id, "source": n, "fetched_at": C.iso(now), "status": "error", "error": repr(e)[:500]})
            C.log(f"{n}: ERROR {e!r}")
            traceback.print_exc()
        C.save_manifest(man)
    avail, why = S.probe_rigs()
    man["unavailable"]["rigs"] = {"checked_at": C.iso(C.utcnow()), "status": why}
    if not S.eia_api_available():
        man["unavailable"]["eia_api"] = {"checked_at": C.iso(C.utcnow()), "status": "EIA_API_KEY not set (skipped)"}
    C.save_manifest(man)
    _log({"run": run_id, "source": "_run", "status": "ok" if ok else "partial", "finished_at": C.iso(C.utcnow()),
          "manifest_sha256": C.sha256_file(C.MANIFEST)})
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
