"""DEV split for architecture screening (Hana Kim, tools/search.py). TRAIN families only; VAL/TEST never read.

Why: hundreds of search configs selected on the 30 bench VAL tasks would overfit VAL (winner's curse,
audit_v3 H-4). Screens instead hold out whole TRAIN source families, train without them, and score them
zero-shot with the exact bench metrics (bench.evaluate's scoring loop: wql_rel, mase_rel, cov80_error vs
seasonal naive, task and family geometric means).

Used by train_search.py behind TSAR_DEV=1 (default off):
    holdout(by_dataset)   pops DEV_DATASETS from the training pool (before `datasets = list(by_dataset)`)
    evaluate(predict_fn)  scores DEV_TASKS; any split argument is ignored, VAL/TEST are never loaded

DEV families (5 sources, 9 tasks; 4 of 9 tasks are H/30min with >= 2048 points of history, about the VAL
share of 10/30 sub-daily tasks, so a context-length or long-rollout effect is visible to the proxy):
    monash_traffic                (H, 862 x 17544, daily + weekly seasonality)        h=168, h=24
    monash_australian_electricity (30min, 5 x ~230k, daily + weekly)                  h=96, h=336
    nn5 + monash_nn5_weekly       (same ATM source: D and W)                          h=28, h=13
    monash_tourism_monthly/_quarterly (same source: M and Q)                          h=12, h=8
    monash_car_parts              (M, 74% zeros: intermittent, like the retail VAL tasks) h=12
Each task scores the last `windows` non-overlapping windows of every (subsampled) series; contexts are the
<= 2048 preceding points, as in bench._build_task. Training then keeps 11 TRAIN sets, ercot (H) among them.
None of these is a VAL/TEST family, an EXCLUDED family, or a TRAIN_DROP set (asserted at import).

TSAR_GIFT_CLEAN=1 (Tanaka 2026-10-30, default off; docs/gift_clean_corpus.md): monash_car_parts is a GIFT-Eval test
dataset (bench.GIFT_TEST_OVERLAP), so it must not drive selection, and bench.load_train no longer trains on it. Under
the flag the car_parts task is replaced by
    dominick                      (W, store x UPC retail sales, 100k series, median 272 points, 42% zeros) h=8
which keeps the intermittent-retail role (car_parts: 74% zeros) from a GIFT-free source and leaves m5 in training.
Hourly coverage is unchanged: traffic (H) and aus_electricity (30min) stay the 4 sub-daily DEV tasks (4 of 9).
Training-side sub-daily data under DEV is ercot alone with or without the flag; TSAR_BENCH_EXTRA=1 (once stamped)
adds the GIFT-free hourly/30min extras (pedestrian, rideshare, taxi_30min, uber_tlc_hourly, wind_farms_hourly).
With the flag off, DEV_TASK_SPECS and every DEV task are unchanged. Coordinated with Kim: flag-guarded edit only.
"""
import os, sys, time, zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as BN  # noqa: E402

# (task, family, dataset, horizon, windows, season)
DEV_TASK_SPECS = [
    ("traffic_1H", "traffic", "monash_traffic", 168, 2, 24),
    ("traffic_1H/short", "traffic", "monash_traffic", 24, 4, 24),
    ("ausel_30T", "aus_electricity", "monash_australian_electricity", 96, 20, 48),
    ("ausel_30T/long", "aus_electricity", "monash_australian_electricity", 336, 10, 48),
    ("nn5_1D", "nn5", "nn5", 28, 4, 7),
    ("nn5_1W", "nn5", "monash_nn5_weekly", 13, 2, 1),
    ("tourism_1M", "tourism", "monash_tourism_monthly", 12, 2, 12),
    ("tourism_1Q", "tourism", "monash_tourism_quarterly", 8, 2, 4),
    ("car_parts_1M", "car_parts", "monash_car_parts", 12, 1, 12),
]
GIFT_CLEAN = BN.gift_clean()
if GIFT_CLEAN:   # car_parts is GIFT-Eval test data: replace it with a GIFT-free intermittent retail family
    DEV_TASK_SPECS = [s for s in DEV_TASK_SPECS if s[2] != "monash_car_parts"] + [
        ("dominick_1W", "dominick", "dominick", 8, 4, 1)]
SUBDAY = {"traffic_1H", "traffic_1H/short", "ausel_30T", "ausel_30T/long"}
DEV_DATASETS = sorted({s[2] for s in DEV_TASK_SPECS})
MAX_FORECASTS = 300   # per task (bench uses 1000; DEV is a screen)
SEED = 20261030

_FREQ = {n: f for n, _, f, _ in BN.TRAIN_LOCAL}
_vt = {t[1] for t in BN.VAL_TASKS + BN.TEST_TASKS} | {t[2] for t in BN.VAL_TASKS + BN.TEST_TASKS}
assert set(DEV_DATASETS) <= set(_FREQ), "DEV sets must be TRAIN_LOCAL sets"
assert not (set(DEV_DATASETS) | {s[1] for s in DEV_TASK_SPECS}) & (_vt | set(BN.EXCLUDED_FAMILIES)), \
    "a DEV family collides with a VAL/TEST/excluded family"
assert not set(DEV_DATASETS) & set(BN.TRAIN_DROP), "DEV sets must not be TRAIN_DROP sets (VAL/TEST overlap)"
assert not (GIFT_CLEAN and set(DEV_DATASETS) & set(BN.GIFT_TEST_OVERLAP)), "TSAR_GIFT_CLEAN: a DEV set is GIFT test data"

_TASKS = None


def holdout(by_dataset):
    """Remove every DEV set from a {dataset: [series]} training pool, in place."""
    for d in DEV_DATASETS:
        by_dataset.pop(d, None)
    return by_dataset


def _build(spec, cache):
    name, fam, ds, H, W, m = spec
    if ds not in cache:
        cache[ds] = [y for y in BN._target_arrays(pd.read_parquet(BN._train_file(ds, False)))
                     if np.isfinite(y).sum() >= BN.MIN_CONTEXT]
    series = cache[ds]
    rng = np.random.default_rng(SEED + zlib.crc32(name.encode()))
    n_keep = max(1, MAX_FORECASTS // W)
    idx = np.sort(rng.permutation(len(series))[:n_keep]) if len(series) > n_keep else np.arange(len(series))
    ctxs, tgts, sids, wins = [], [], [], []
    for i in idx:
        y = series[i]
        first = len(y) - W * H
        for k in range(W):
            c = first + k * H
            if c < 0 or c + H > len(y):
                continue
            ctx = y[max(0, c - BN.MAX_CONTEXT):c]
            fin = np.flatnonzero(np.isfinite(ctx))
            if len(fin) < BN.MIN_CONTEXT:
                continue
            ctxs.append(ctx[fin[0]:].copy())
            tgts.append(y[c:c + H].copy())
            sids.append(str(i))
            wins.append(k)
    return {"name": name, "family": fam, "config": f"dev:{ds}", "horizon": H, "season": m, "freq": _FREQ[ds],
            "contexts": ctxs, "targets": np.stack(tgts) if tgts else np.zeros((0, H), np.float32),
            "series_id": sids, "window": np.asarray(wins)}


def load_dev():
    global _TASKS
    if _TASKS is None:
        cache = {}
        _TASKS = [_build(s, cache) for s in DEV_TASK_SPECS]
    return _TASKS


def evaluate(predict_fn, split=None, verbose=True, names=None, tasks=None):
    """bench.evaluate on the DEV tasks (identical scoring code). split/names/tasks are ignored: DEV only."""
    t0 = time.time()
    dev = load_dev()
    assert dev and all(t["config"].startswith("dev:") for t in dev)
    # tasks= is always given, so bench.evaluate never loads a split; "val" only satisfies its split assert
    r = BN.evaluate(predict_fn, split="val", verbose=False, tasks=dev)
    r["split"] = "dev"
    df = r["per_dataset"]
    sub = df[df["dataset"].isin(SUBDAY)]
    r["wql_rel_subday"] = BN._gmean(sub["wql_rel"]) if len(sub) else float("nan")
    if verbose:
        with pd.option_context("display.width", 160, "display.precision", 3, "display.max_rows", 200):
            print(df.drop(columns=["sn_wql", "sn_mase"]).to_string(index=False))
        print(f"dev: wql_rel={r['wql_rel']:.4f} mase_rel={r['mase_rel']:.4f} cov80_error={r['cov80_error']:.4f} "
              f"family-gmean wql_rel={r['wql_rel_family']:.4f} subday wql_rel={r['wql_rel_subday']:.4f} "
              f"({r['n_tasks']} tasks, {r['n_families']} families, {time.time() - t0:.1f}s)", flush=True)
    return r


if __name__ == "__main__":   # task summary: uv run tools/devsplit.py
    for t in load_dev():
        L = [len(c) for c in t["contexts"]]
        print(f"{t['name']:18s} {t['freq']:6s} h={t['horizon']:4d} n={len(L):4d} series={len(set(t['series_id'])):4d} "
              f"ctx median={int(np.median(L))} min={min(L)}")
