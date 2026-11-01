"""
Fixed benchmark harness for time-series autoresearch. DO NOT MODIFY during a research run.

Downloads a fixed suite of public forecasting datasets (the Chronos zero-shot benchmark
collection on Hugging Face), cuts each series into train / val / test windows, and defines
the one evaluation function every model is scored with: our model, the statistical
baselines and the foundation models.

Usage (one-time, a few minutes, ~250 MB download):
    uv run prepare.py
"""

import os
import io
import time
import pickle

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 240          # seconds of training per experiment (wall clock)
QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
MAX_EVAL_SERIES = 300      # series scored per dataset (fixed random sample)
MAX_TRAIN_SERIES = 2000    # series per dataset exposed for training
MAX_CONTEXT = 1024         # history kept before the val window
MIN_CONTEXT = 10           # series with less history than this are dropped
SEED = 1234

# (name, horizon, season, freq) - horizons follow the Chronos / fev benchmark conventions.
DATASETS = [
    ("monash_tourism_monthly",        24, 12, "M"),
    ("monash_tourism_quarterly",       8,  4, "Q"),
    ("monash_m1_monthly",             18, 12, "M"),
    ("monash_m3_monthly",             18, 12, "M"),
    ("monash_m3_quarterly",            8,  4, "Q"),
    ("m4_quarterly",                   8,  4, "Q"),
    ("monash_cif_2016",               12, 12, "M"),
    ("monash_hospital",               12, 12, "M"),
    ("monash_car_parts",              12, 12, "M"),
    ("monash_fred_md",                12, 12, "M"),
    ("exchange_rate",                 30,  5, "B"),
    ("nn5",                           56,  7, "D"),
    ("monash_nn5_weekly",              8,  1, "W"),
    ("monash_covid_deaths",           30,  1, "D"),
    ("monash_weather",                30,  1, "D"),
    ("m5",                            28,  7, "D"),
    ("dominick",                       8,  1, "W"),
    ("ercot",                         24, 24, "H"),
    ("monash_traffic",                24, 24, "H"),
    ("monash_australian_electricity", 48, 48, "30min"),
]

# Datasets held out entirely: never returned by load_train_series(), never scored on the
# val split. They are scored only on the final "holdout" split to measure zero-shot
# generalization to unseen domains (finance, macro, energy, retail, intermittent demand).
HOLDOUT = {"exchange_rate", "monash_fred_md", "monash_australian_electricity",
           "dominick", "monash_car_parts"}

HF_URL ="https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/{name}/train-00000-of-00001.parquet"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
SUITE_PATH = os.path.join(CACHE_DIR, "suite.pkl")

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _download(name):
    path = os.path.join(RAW_DIR, f"{name}.parquet")
    if os.path.exists(path):
        return path
    os.makedirs(RAW_DIR, exist_ok=True)
    url = HF_URL.format(name=name)
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path + ".tmp", "wb") as f:
                f.write(r.content)
            os.replace(path + ".tmp", path)
            return path
        except Exception as e:  # network hiccups
            wait = 2 ** attempt
            print(f"  download failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not download {name}")


def _build_dataset(name, horizon, season, freq, rng):
    df = pd.read_parquet(_download(name))
    target_col = "target"
    series = []
    for target in df[target_col]:
        y = np.asarray(target, dtype=np.float32)
        if len(y) - 2 * horizon < MIN_CONTEXT:
            continue
        y = y[-(MAX_CONTEXT + 2 * horizon):]
        series.append(y)
    order = rng.permutation(len(series))
    series = [series[i] for i in order[:MAX_TRAIN_SERIES]]
    return {
        "name": name, "horizon": horizon, "season": season, "freq": freq,
        "series": series,                       # full kept series: context + val + test
        "n_eval": min(MAX_EVAL_SERIES, len(series)),  # first n_eval series are scored
    }


def prepare():
    rng = np.random.default_rng(SEED)
    suite = []
    for name, horizon, season, freq in DATASETS:
        t0 = time.time()
        ds = _build_dataset(name, horizon, season, freq, rng)
        suite.append(ds)
        print(f"{name:32s} series={len(ds['series']):5d} eval={ds['n_eval']:4d} "
              f"H={horizon:3d} m={season:3d}  ({time.time() - t0:.1f}s)")
    with open(SUITE_PATH, "wb") as f:
        pickle.dump(suite, f)
    print(f"saved suite to {SUITE_PATH}")


def load_suite():
    if not os.path.exists(SUITE_PATH):
        raise FileNotFoundError("benchmark data missing: run `uv run prepare.py` first")
    with open(SUITE_PATH, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# What a model may train on
# ---------------------------------------------------------------------------

def load_train_series():
    """
    History of every series up to (not including) the val window. This is the only real
    data train.py may learn from. Synthetic data generated inside train.py is also allowed.
    Returns a list of dicts: {dataset, target, horizon, season, freq}.
    """
    out = []
    for ds in load_suite():
        if ds["name"] in HOLDOUT:
            continue
        h = ds["horizon"]
        for y in ds["series"]:
            out.append({"dataset": ds["name"], "target": y[:-2 * h], "horizon": h,
                        "season": ds["season"], "freq": ds["freq"]})
    return out


def fill_nans(x):
    """Forward-fill NaNs, then back-fill, then zero. For models that cannot take NaNs."""
    x = pd.Series(x).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    return x

# ---------------------------------------------------------------------------
# Evaluation (the ground truth metric)
# ---------------------------------------------------------------------------

def _windows(ds, split):
    """val: second-to-last horizon. test / holdout: last horizon."""
    h = ds["horizon"]
    ctxs, tgts = [], []
    for y in ds["series"][:ds["n_eval"]]:
        if split == "val":
            ctxs.append(y[:-2 * h]); tgts.append(y[-2 * h:-h])
        else:
            ctxs.append(y[:-h]); tgts.append(y[-h:])
    return ctxs, np.stack(tgts)


def _seasonal_naive(ctx, h, m):
    c = fill_nans(ctx)
    if len(c) < m:
        m = 1
    reps = int(np.ceil(h / m))
    return np.tile(c[-m:], reps)[:h]


def _scale(ctx, m):
    c = ctx[~np.isnan(ctx)]
    if len(c) <= m:
        m = 1
    if len(c) <= m:
        return np.nan
    d = np.abs(c[m:] - c[:-m])
    s = d.mean()
    return s if s > 0 else np.nan


def _score(q_pred, target, ctxs, m):
    """q_pred: [n, Q, H]. Returns (wql, mase) for one dataset."""
    mask = ~np.isnan(target)
    y = np.nan_to_num(target)
    err = y[:, None, :] - q_pred                                   # [n, Q, H]
    pinball = np.maximum(QUANTILES[None, :, None] * err, (QUANTILES[None, :, None] - 1) * err)
    pinball = pinball * mask[:, None, :]
    denom = np.abs(y * mask).sum()
    wql = (2 * pinball.sum(axis=(0, 2)) / max(denom, 1e-8)).mean()
    med = q_pred[:, len(QUANTILES) // 2, :]
    scales = np.array([_scale(c, m) for c in ctxs])
    ae = (np.abs(y - med) * mask).sum(1) / np.maximum(mask.sum(1), 1)
    mase = np.nanmean(ae / scales)
    inside = (y >= q_pred[:, 0, :]) & (y <= q_pred[:, -1, :])
    coverage80 = (inside & mask).sum() / max(mask.sum(), 1)   # nominal 0.80
    return float(wql), float(mase), float(coverage80)


def evaluate(predict_fn, split="val", verbose=True):
    """
    Score a forecaster on every dataset in the suite.

    predict_fn(contexts, horizon, season, freq) -> np.ndarray [n_series, 9, horizon]
        contexts: list of 1-D float32 arrays (may contain NaN), oldest value first.
        Must return the 9 quantiles in QUANTILES order.

    Splits:
        val     - development datasets, second-to-last window. Use this for iteration.
        test    - development datasets, last window. Confirmatory, in-domain.
        holdout - held-out datasets, last window. Confirmatory, zero-shot generalization.
    test and holdout only run with TSAR_SPLIT=test, so they cannot be used by accident.

    Returns a dict. The headline numbers are geometric means over datasets of the ratio
    to Seasonal Naive (lower is better, < 1.0 beats Seasonal Naive):
        wql_rel     - probabilistic accuracy (weighted quantile loss, ~CRPS)
        mase_rel    - point accuracy of the median
        cov80_error - mean |coverage of the 10-90% interval - 0.80| (calibration)
    """
    assert split in ("val", "test", "holdout"), split
    if split != "val" and os.environ.get("TSAR_SPLIT") != "test":
        raise PermissionError(f"{split} split is reserved: set TSAR_SPLIT=test to score it")
    rows = []
    for ds in load_suite():
        if (ds["name"] in HOLDOUT) != (split == "holdout"):
            continue
        h, m = ds["horizon"], ds["season"]
        ctxs, tgts = _windows(ds, split)
        t0 = time.time()
        q = np.asarray(predict_fn(ctxs, h, m, ds["freq"]), dtype=np.float32)
        secs = time.time() - t0
        assert q.shape == (len(ctxs), len(QUANTILES), h), f"{ds['name']}: bad shape {q.shape}"
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.sort(q, axis=1)  # enforce monotone quantiles
        wql, mase, cov = _score(q, tgts, ctxs, m)
        sn = np.stack([_seasonal_naive(c, h, m) for c in ctxs])
        sn_q = np.repeat(sn[:, None, :], len(QUANTILES), axis=1)
        sn_wql, sn_mase, _ = _score(sn_q, tgts, ctxs, m)
        rows.append({"dataset": ds["name"], "wql": wql, "mase": mase,
                     "wql_rel": wql / max(sn_wql, 1e-8), "mase_rel": mase / max(sn_mase, 1e-8),
                     "cov80": cov, "predict_seconds": secs})
    df = pd.DataFrame(rows)
    gmean = lambda s: float(np.exp(np.log(np.clip(s, 1e-6, None)).mean()))
    result = {
        "split": split,
        "wql_rel": gmean(df["wql_rel"]),
        "mase_rel": gmean(df["mase_rel"]),
        "cov80_error": float((df["cov80"] - 0.8).abs().mean()),
        "predict_seconds": float(df["predict_seconds"].sum()),
        "per_dataset": df,
    }
    if verbose:
        with pd.option_context("display.width", 120, "display.precision", 3):
            print(df.to_string(index=False))
    return result


if __name__ == "__main__":
    prepare()
