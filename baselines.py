"""
Industry baselines our model has to beat, scored with the exact same harness (prepare.evaluate).

    uv run baselines.py                      # all available baselines, val split
    uv run baselines.py auto_ets chronos2    # a subset
    TSAR_SPLIT=test uv run baselines.py --split test     # confirmatory, run rarely
    TSAR_SPLIT=test uv run baselines.py --split holdout

Results append to baselines/results.tsv; per-dataset scores go to baselines/per_dataset/.
Models whose package is not installed are skipped with a message.
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd

import prepare as P

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
STATS_CONTEXT = 512   # input window for statistical models (documented in BENCHMARK.md)

# ---------------------------------------------------------------------------
# Statistical baselines (statsforecast)
# ---------------------------------------------------------------------------

LEVELS = [20, 40, 60, 80]
_Q_COLS = ["lo-80", "lo-60", "lo-40", "lo-20", None, "hi-20", "hi-40", "hi-60", "hi-80"]


def _stats_model(make):
    def predict(ctxs, h, m, freq):
        from statsforecast import StatsForecast
        rows = []
        for i, c in enumerate(ctxs):
            c = P.fill_nans(c)[-STATS_CONTEXT:]
            rows.append(pd.DataFrame({"unique_id": i, "ds": np.arange(len(c)), "y": c}))
        df = pd.concat(rows)
        model = make(m)
        sf = StatsForecast(models=[model], freq=1, n_jobs=-1, fallback_model=None)
        fc = sf.forecast(df=df, h=h, level=LEVELS).sort_values(["unique_id", "ds"])
        name = repr(model)
        out = np.empty((len(ctxs), 9, h), dtype=np.float32)
        for j, col in enumerate(_Q_COLS):
            v = fc[name if col is None else f"{name}-{col}"].to_numpy()
            out[:, j, :] = v.reshape(len(ctxs), h)
        return out
    return predict


def _stats_registry():
    from statsforecast.models import AutoETS, AutoTheta, AutoARIMA, Naive, SeasonalNaive
    return {
        "naive": _stats_model(lambda m: Naive()),
        "seasonal_naive_pi": _stats_model(lambda m: SeasonalNaive(season_length=m)),
        "auto_ets": _stats_model(lambda m: AutoETS(season_length=m)),
        "auto_theta": _stats_model(lambda m: AutoTheta(season_length=m)),
        "auto_arima": _stats_model(lambda m: AutoARIMA(season_length=m)),
    }

# ---------------------------------------------------------------------------
# Time-series foundation models (zero-shot)
# ---------------------------------------------------------------------------

def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _chronos(model_id, batch=64):
    state = {}

    def predict(ctxs, h, m, freq):
        import torch
        from chronos import BaseChronosPipeline
        if "pipe" not in state:
            state["pipe"] = BaseChronosPipeline.from_pretrained(model_id, device_map=_device(),
                                                                torch_dtype=torch.float32)
            state["params"] = sum(p.numel() for p in state["pipe"].model.parameters())
        pipe = state["pipe"]
        qs = P.QUANTILES.tolist()
        out = []
        for s in range(0, len(ctxs), batch):
            chunk = [torch.tensor(c[-2048:], dtype=torch.float32) for c in ctxs[s:s + batch]]
            res = pipe.predict_quantiles(chunk, prediction_length=h, quantile_levels=qs)
            q = res[0]
            if isinstance(q, list):            # Chronos-2 returns one tensor per input
                q = torch.stack([t.reshape(-1, h, len(qs))[0] for t in q])
            q = q.reshape(len(chunk), h, len(qs))
            out.append(q.permute(0, 2, 1).float().cpu().numpy())
        return np.concatenate(out)

    predict.state = state
    return predict


def _tirex():
    state = {}

    def predict(ctxs, h, m, freq):
        import torch
        from tirex import load_model
        if "model" not in state:
            state["model"] = load_model("NX-AI/TiRex", device=_device())
        out = []
        for s in range(0, len(ctxs), 64):
            chunk = [torch.tensor(c[-2048:], dtype=torch.float32) for c in ctxs[s:s + 64]]
            q, _ = state["model"].forecast(context=chunk, prediction_length=h)
            out.append(np.asarray(q, dtype=np.float32).transpose(0, 2, 1))
        return np.concatenate(out)

    predict.state = state
    return predict


def _timesfm():
    state = {}

    def predict(ctxs, h, m, freq):
        import timesfm
        if "model" not in state:
            mdl = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            mdl.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256,
                                               normalize_inputs=True, use_continuous_quantile_head=True))
            state["model"] = mdl
        _, q = state["model"].forecast(horizon=h, inputs=[P.fill_nans(c)[-1024:] for c in ctxs])
        return np.asarray(q, dtype=np.float32)[:, :, 1:10].transpose(0, 2, 1)  # drop mean column

    predict.state = state
    return predict


def _timestral(model_id, batch=64):
    state = {}
    want = P.QUANTILES.tolist()

    def predict(ctxs, h, m, freq):
        import torch
        from timestral import TimestralForecaster
        if "model" not in state:
            state["model"] = TimestralForecaster.from_pretrained(model_id)  # CPU, per the README
            mod = getattr(state["model"], "model", None)
            if mod is not None:
                state["params"] = sum(p.numel() for p in mod.parameters())
        out = []
        for s in range(0, len(ctxs), batch):
            chunk = ctxs[s:s + batch]
            L = max(len(c) for c in chunk)
            x = np.full((len(chunk), L), np.nan, dtype=np.float32)   # left-pad with NaN = missing
            for i, c in enumerate(chunk):
                x[i, L - len(c):] = c
            fc = state["model"].forecast(x, prediction_length=h)
            levels = [round(float(l), 4) for l in fc.quantile_levels]
            idx = [levels.index(round(q, 4)) for q in want]
            q = np.asarray(fc.quantiles)                              # (B, H, 21) or (B, 1, H, 21)
            q = q.reshape(len(chunk), h, len(levels))[:, :, idx]
            out.append(q.transpose(0, 2, 1).astype(np.float32))
        return np.concatenate(out)

    predict.state = state
    return predict


def registry():
    reg = {}
    try:
        reg.update(_stats_registry())
    except ImportError as e:
        print(f"skipping statistical baselines: {e}")
    reg.update({
        "chronos_bolt_small": _chronos("amazon/chronos-bolt-small"),
        "chronos_bolt_base": _chronos("amazon/chronos-bolt-base"),
        "chronos2": _chronos("amazon/chronos-2"),
        "tirex": _tirex(),
        "timesfm_2p5": _timesfm(),
        "timestral_5m": _timestral("mistralai/Timestral-5M"),
        "timestral_28m": _timestral("mistralai/Timestral-28M"),
        "timestral_209m": _timestral("mistralai/Timestral-209M"),
    })
    return reg

# ---------------------------------------------------------------------------

def run(name, fn, split):
    t0 = time.time()
    try:
        r = P.evaluate(fn, split=split, verbose=False)
    except ImportError as e:
        print(f"{name:20s} skipped (not installed: {e.name})")
        return None
    total = time.time() - t0
    params = getattr(fn, "state", {}).get("params")
    os.makedirs(os.path.join(OUT_DIR, "per_dataset"), exist_ok=True)
    r["per_dataset"].to_csv(os.path.join(OUT_DIR, "per_dataset", f"{name}_{split}.csv"), index=False)
    row = {"model": name, "split": split, "wql_rel": round(r["wql_rel"], 4),
           "mase_rel": round(r["mase_rel"], 4), "cov80_error": round(r["cov80_error"], 4),
           "predict_seconds": round(r["predict_seconds"], 1), "total_seconds": round(total, 1),
           "params_M": round(params / 1e6, 1) if params else "",
           "date": time.strftime("%Y-%m-%d")}
    path = os.path.join(OUT_DIR, "results.tsv")
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not os.path.exists(path))
    print(f"{name:20s} wql_rel={row['wql_rel']:.4f} mase_rel={row['mase_rel']:.4f} "
          f"cov80_err={row['cov80_error']:.3f} predict={row['predict_seconds']}s")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--split", default="val", choices=["val", "test", "holdout"])
    args = ap.parse_args()
    reg = registry()
    names = args.models or list(reg)
    for n in names:
        if n not in reg:
            sys.exit(f"unknown baseline {n}; choose from {list(reg)}")
        run(n, reg[n], args.split)
