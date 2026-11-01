"""
Leakage scan: does Alma's real training data (the 14 TSAR_GIFT_CLEAN TRAIN datasets stored in the checkpoint cfg)
contain any oil / gas price or crude-stock series, or any data from the VALIDATION/TEST period?

    nice -n 10 .venv/bin/python tools/commodity/leakage_scan.py > runs/commodity/leakage_scan.txt

For every TRAIN dataset (data/raw/<name>.parquet, the file bench.load_train reads), one at a time:
  - timestamp range (if the file has timestamps), number of series and points;
  - 6-gram matching against the commodity series: (a) exact levels rounded to 2 decimals, (b) scale-free
    consecutive log-ratios rounded to 3 decimals (catches rescaled copies), for wti/brent/hh daily, their monthly and
    weekly means, and the weekly inventory. All-zero / constant grams are ignored.
Also reports monash_fred_md (excluded from TRAIN, bench.TRAIN_EXCLUDED) date range and its oil column, if present.
Reads no VAL/TEST bench data.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "commodity"))
import data as D   # noqa: E402

N = 6


def grams(y, kind):
    y = np.asarray(y, np.float64)
    y = y[np.isfinite(y)]
    if kind == "level":
        r = np.round(y, 2)
    else:
        y = y[y > 0]
        if len(y) < N + 1:
            return set()
        r = np.round(np.diff(np.log(y)), 3)
    if len(r) < N:
        return set()
    w = np.lib.stride_tricks.sliding_window_view(r, N)
    keep = w.max(1) != w.min(1)                     # drop constant grams (zeros, flat runs)
    if not keep.any():
        return set()
    return set(map(bytes, np.ascontiguousarray(w[keep]).view(np.uint8).reshape(keep.sum(), -1)))


def oil_refs():
    refs = {}
    for name in ("wti", "brent", "hh", "inv"):
        d, v = D.load(name)
        s = pd.Series(v, index=pd.DatetimeIndex(d))
        refs[f"{name}_raw"] = s.values
        if name != "inv":
            refs[f"{name}_monthly"] = s.resample("MS").mean().dropna().values
            refs[f"{name}_weekly"] = s.resample("W").mean().dropna().values
    return {k: {"level": grams(v, "level"), "ratio": grams(v, "ratio")} for k, v in refs.items()}


def main():
    ck = torch.load(os.path.join(ROOT, "data/ckpt/wm_ssm_none_s0.pt"), map_location="cpu", weights_only=False)["cfg"]
    names = ck["train_datasets"]
    print(f"checkpoint wm_ssm_none_s0 cfg: bench_extra={ck['bench_extra']} train_series={ck['train_series']} "
          f"GEP={'TSAR_BENCH_GEP' in ck['env']} GIFT_CLEAN={ck['env'].get('TSAR_GIFT_CLEAN')}")
    print(f"TRAIN datasets ({len(names)}): {names}")
    refs = oil_refs()
    report = {}
    for name in names + ["monash_fred_md"]:
        path = os.path.join(ROOT, "data", "raw", f"{name}.parquet")
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        df = pd.read_parquet(path)
        rng = None
        if "timestamp" in df.columns:
            ts = df["timestamp"].explode()
            ts = pd.to_datetime(ts, errors="coerce").dropna()
            rng = (str(ts.min().date()), str(ts.max().date())) if len(ts) else None
        cols = ["target"] if "target" in df.columns else [c for c in df.columns
                                                           if c not in ("id", "timestamp") and df[c].dtype == object]
        hits = {k: {"level": 0, "ratio": 0} for k in refs}
        n_series = n_pts = 0
        for c in cols:
            for v in df[c]:
                y = np.asarray(v, np.float64)
                n_series += 1
                n_pts += len(y)
                gl, gr = grams(y, "level"), grams(y, "ratio")
                for k, rf in refs.items():
                    hits[k]["level"] += len(gl & rf["level"])
                    hits[k]["ratio"] += len(gr & rf["ratio"])
        tot = {k: v for k, v in hits.items() if v["level"] or v["ratio"]}
        report[name] = {"timestamps": rng, "series": n_series, "points": n_pts, "gram_hits": tot,
                        "columns": [c for c in df.columns][:12]}
        print(f"{name:32s} series={n_series:6d} points={n_pts:10d} timestamps={rng} hits={tot or 'none'}", flush=True)
        if name == "monash_fred_md" and "id" in df.columns:
            print(f"  monash_fred_md ids containing OIL: {[i for i in df['id'].astype(str) if 'OIL' in i.upper()]}")
        del df
    os.makedirs(os.path.join(ROOT, "runs", "commodity"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "runs", "commodity", "leakage_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
