"""
Leakage review for the fev-bench tasks outside our VAL/TEST splits: does Alma's real training corpus (TSAR_GIFT_CLEAN=1
TRAIN, read from the checkpoint's stored config) contain the same series?

Dataset level (fev's rule, "trained on this dataset"): m5 (x3) and ercot (x4) are TRAIN datasets -> flagged.
Content level: tools/overlap_scan.scan_pairs (lag-searching Pearson r of first differences, r >= 0.95 over >= N aligned
points; the method behind bench.TRAIN_DROP) between every full series of the probe task (context + all targets) and
every series (full, before subsampling and TRAIN_DROP) of the TRAIN datasets of the same frequency class. Monthly
TRAIN series are also aggregated to quarterly sums for the quarterly probes (tourism is published both ways).

    nice -n 10 uv run tools/fev_full_overlap.py --out runs/fev_full/overlap_scan.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import bench as BN            # noqa: E402
import overlap_scan as OS     # noqa: E402

# probe task -> (fev parquet config, target columns or None = "target"/single, freq class, TRAIN datasets to scan,
#                min overlap)
PROBES = {
    "australian_tourism": ("australian_tourism", None, "Q",
                           ["monash_tourism_quarterly", "monash_m3_quarterly", "monash_tourism_monthly@Q",
                            "monash_m1_monthly@Q", "monash_m3_monthly@Q"], 30),
    "hospital": ("hospital", None, "M", ["monash_m1_monthly", "monash_m3_monthly", "monash_tourism_monthly",
                                         "monash_cif_2016"], 36),
    "fred_md_2025": ("fred_md_2025", "__ALL__", "M", ["monash_m1_monthly", "monash_m3_monthly",
                                                      "monash_tourism_monthly", "monash_cif_2016"], 60),
    "fred_qd_2025": ("fred_qd_2025", "__ALL__", "Q", ["monash_m3_quarterly", "monash_tourism_quarterly",
                                                      "monash_m3_monthly@Q", "monash_m1_monthly@Q"], 40),
    "proenfo_gfc12": ("proenfo_gfc12", None, "H", ["ercot", "monash_traffic"], 168),
    "proenfo_gfc14": ("proenfo_gfc14", None, "H", ["ercot", "monash_traffic"], 168),
    "proenfo_gfc17": ("proenfo_gfc17", None, "H", ["ercot", "monash_traffic"], 168),
    "favorita_transactions_1D": ("favorita_transactions_1D", None, "D", ["nn5", "monash_weather"], 60),
    "favorita_transactions_1W": ("favorita_transactions_1W", None, "W", ["monash_nn5_weekly", "dominick"], 60),
    "kdd_cup_2022_30T": ("kdd_cup_2022_30T", None, "30min", ["monash_australian_electricity"], 96),
}
TARGET = {"favorita_transactions_1D": "transactions", "favorita_transactions_1W": "transactions",
          "kdd_cup_2022_30T": "Patv"}


def probe_series(cfg, cols, name):
    p = os.path.join(ROOT, "data", "fev_full", "raw", cfg + ".parquet")
    df = pd.read_parquet(p)
    if cols == "__ALL__":
        cols = [c for c in df.columns if c not in ("id", "timestamp") and isinstance(df[c].iloc[0], np.ndarray)]
    else:
        cols = [TARGET.get(name, "target")]
    out = []
    for _, r in df.iterrows():
        for c in cols:
            out.append((f"{r['id']}/{c}" if len(cols) > 1 else str(r["id"]), np.asarray(r[c], np.float64)))
    return out


def train_series(name):
    agg = name.endswith("@Q")
    base = name[:-2] if agg else name
    spec = {n: (n, s, f, h) for n, s, f, h in BN.TRAIN_LOCAL}[base]
    df = pd.read_parquet(BN._train_file(spec[0], False))
    ys = BN._target_arrays(df)
    del df
    out = []
    for k, y in enumerate(ys):
        y = np.asarray(y, np.float64)
        if agg:
            n = len(y) // 3 * 3
            if n < 6:
                continue
            for off in range(3):                 # the quarter boundary is unknown for anonymous series
                m = len(y[off:]) // 3 * 3
                out.append(((name, k, off), y[off:off + m].reshape(-1, 3).sum(1)))
        else:
            out.append(((name, k), y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "fev_full", "overlap_scan.json"))
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import torch
    stored = torch.load(os.path.join(ROOT, "data", "ckpt", "wm_ssm_none_s0.pt"), map_location="cpu",
                        weights_only=False)["cfg"]
    trained = set(stored["train_datasets"])
    print("TRAIN datasets (checkpoint cfg):", sorted(trained), flush=True)
    rep = {"train_datasets": sorted(trained), "thr": a.thr, "tasks": {}}
    for name, (cfg, cols, fc, tr, mo) in PROBES.items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        probes = probe_series(cfg, cols, name)
        hits, scanned = [], []
        for tn in tr:
            if tn.split("@")[0] not in trained:
                continue
            scanned.append(tn)
            T = train_series(tn)
            for pid, tid, r, n, lag in OS.scan_pairs(probes, T, min_overlap=mo, thr=a.thr, chunk=48):
                hits.append({"probe": pid, "train": list(tid), "r": round(r, 4), "n": n, "lag": lag})
            del T
        rep["tasks"][name] = {"probes": len(probes), "scanned": scanned, "min_overlap": mo, "hits": hits}
        print(f"{name:28s} probes={len(probes):5d} scanned={scanned} hits={len(hits)} ({time.time() - t0:.0f}s)",
              flush=True)
        for h in hits[:20]:
            print("    ", h, flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
