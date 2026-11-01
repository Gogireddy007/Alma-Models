"""
Model registry for the commodity harness (interface: tools/commodity/README.md).

    predict(context: np.ndarray, horizon: int, quantiles: list[float]) -> np.ndarray [len(quantiles), horizon]

Every factory is lazy. Settings here are the pre-registered ones (docs/commodity_prereg.md section 5); do not change
them after the pre-registration commit without a dated amendment in that file.
"""
import os
import sys
import glob
import hashlib
import importlib.util

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE9 = [round(0.1 * i, 1) for i in range(1, 10)]
FM_CONTEXT = 1024          # foundation models: last 1024 observations
STATS_CONTEXT = 512        # statsforecast models
RW_WINDOW = 250            # Gaussian random walk: sigma from the last 250 log returns


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _fill(want, have_levels, have_q):
    """have_q [n, len(have_levels), h] -> [n, len(want), h]; levels not provided -> NaN."""
    hv = [round(float(x), 4) for x in have_levels]
    out = np.full((have_q.shape[0], len(want), have_q.shape[2]), np.nan, dtype=np.float64)
    for j, q in enumerate(want):
        if round(float(q), 4) in hv:
            out[:, j] = have_q[:, hv.index(round(float(q), 4))]
    return out


def _single(batch_fn):
    def predict(context, horizon, quantiles, **kw):
        return batch_fn([context], horizon, quantiles, **kw)[0]
    predict.batch = batch_fn
    return predict


# --------------------------------------------------------------------------- simple baselines
def rw_gauss():
    """Driftless Gaussian random walk in log price; sigma = std of the last 250 daily log returns of the context."""
    def batch(ctxs, h, qs):
        out = np.empty((len(ctxs), len(qs), h))
        k = np.sqrt(np.arange(1, h + 1))
        z = norm.ppf(qs)
        for i, c in enumerate(ctxs):
            c = np.asarray(c, float)
            pos = c[-(RW_WINDOW + 1):]
            pos = pos[pos > 0]
            s = np.std(np.diff(np.log(pos)), ddof=1)
            last = c[-1]
            if last > 0:
                out[i] = last * np.exp(z[:, None] * s * k[None, :])
            else:                                               # never in VAL/TEST; DEV 2020-04 only
                out[i] = last + z[:, None] * s * abs(last) * k[None, :]
        return out
    p = _single(batch)
    p.meta = {"sigma_window": RW_WINDOW}
    return p


def _stats(kind):
    LV = [20, 40, 60, 80, 90, 95]
    lv_q = {}
    for L in LV:
        lv_q[round((1 - L / 100) / 2, 4)] = f"lo-{L}"
        lv_q[round(1 - (1 - L / 100) / 2, 4)] = f"hi-{L}"

    def batch(ctxs, h, qs):
        season = 1
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, AutoARIMA
        rows = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(c[-STATS_CONTEXT:])),
                              "y": np.asarray(c[-STATS_CONTEXT:], float)}) for i, c in enumerate(ctxs)]
        model = AutoETS(season_length=season) if kind == "ets" else AutoARIMA(season_length=season)
        sf = StatsForecast(models=[model], freq=1, n_jobs=1)
        fc = sf.forecast(df=pd.concat(rows), h=h, level=LV).sort_values(["unique_id", "ds"])
        name = repr(model)
        out = np.full((len(ctxs), len(qs), h), np.nan)
        for j, q in enumerate(qs):
            col = name if round(q, 4) == 0.5 else (f"{name}-{lv_q[round(q, 4)]}" if round(q, 4) in lv_q else None)
            if col is not None:
                out[:, j] = fc[col].to_numpy().reshape(len(ctxs), h)
        return out
    p = _single(batch)
    p.meta = {"context": STATS_CONTEXT, "season_length": "1 (daily and weekly)"}
    return p


# --------------------------------------------------------------------------- foundation models (CPU)
def _chronos(model_id, native):
    st = {}

    def batch(ctxs, h, qs):
        import torch
        from chronos import BaseChronosPipeline
        if "pipe" not in st:
            st["pipe"] = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        ask = [q for q in qs if native is None or round(q, 4) in native]
        out = []
        for s in range(0, len(ctxs), 32):
            chunk = [torch.tensor(np.asarray(c[-FM_CONTEXT:], np.float32)) for c in ctxs[s:s + 32]]
            q, _ = st["pipe"].predict_quantiles(chunk, prediction_length=h, quantile_levels=ask)
            if isinstance(q, list):
                q = torch.stack([t.reshape(-1, h, len(ask))[0] for t in q])
            out.append(q.reshape(len(chunk), h, len(ask)).permute(0, 2, 1).double().numpy())
        return _fill(qs, ask, np.concatenate(out))
    p = _single(batch)
    p.meta = {"hf": model_id, "context": FM_CONTEXT, "device": "cpu"}
    return p


def _tirex():
    st = {}

    def batch(ctxs, h, qs):
        import torch
        from tirex import load_model
        if "m" not in st:
            st["m"] = load_model("NX-AI/TiRex", device="cpu")
        out = []
        for s in range(0, len(ctxs), 32):
            chunk = [torch.tensor(np.asarray(c[-FM_CONTEXT:], np.float32)) for c in ctxs[s:s + 32]]
            q, _ = st["m"].forecast(context=chunk, prediction_length=h)
            out.append(np.asarray(q, dtype=np.float64).transpose(0, 2, 1))
        return _fill(qs, NATIVE9, np.concatenate(out))
    p = _single(batch)
    p.meta = {"hf": "NX-AI/TiRex", "context": FM_CONTEXT, "device": "cpu"}
    return p


def _timestral(model_id):
    st = {}

    def batch(ctxs, h, qs):
        from timestral import TimestralForecaster
        if "m" not in st:
            st["m"] = TimestralForecaster.from_pretrained(model_id)
        out = []
        for s in range(0, len(ctxs), 32):
            chunk = [np.asarray(c[-FM_CONTEXT:], np.float32) for c in ctxs[s:s + 32]]
            L = max(len(c) for c in chunk)
            x = np.full((len(chunk), L), np.nan, dtype=np.float32)
            for i, c in enumerate(chunk):
                x[i, L - len(c):] = c
            fc = st["m"].forecast(x, prediction_length=h)
            lev = [round(float(v), 4) for v in fc.quantile_levels]
            q = np.asarray(fc.quantiles).reshape(len(chunk), h, len(lev)).transpose(0, 2, 1)
            out.append(_fill(qs, lev, q))
        return np.concatenate(out)
    p = _single(batch)
    p.meta = {"hf": model_id, "context": FM_CONTEXT, "device": "cpu",
              "contamination": "released 2026-10-15: training data may cover the TEST period"}
    return p


KRONOS_REPO = "/Users/srimanarayana/MIKO/kronos/repo"     # official github.com/shiyu-coder/Kronos @ 67b630e
KRONOS_SAMPLES, KRONOS_CONTEXT = 10, 256     # amendment A1 (prereg): 20/512 took 372 s per origin


def _kronos():
    st = {}

    def batch(ctxs, h, qs, dates):
        import torch
        import pandas as pd
        if "m" not in st:
            sys.path.insert(0, KRONOS_REPO)
            from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps
            st.update(tok=KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval(),
                      m=Kronos.from_pretrained("NeoQuasar/Kronos-small").eval(),
                      ari=auto_regressive_inference, cts=calc_time_stamps)
        out = np.empty((len(ctxs), len(qs), h))
        for i, (c, d) in enumerate(zip(ctxs, dates)):
            torch.manual_seed(0)
            np.random.seed(0)
            c = np.asarray(c[-KRONOS_CONTEXT:], np.float32)
            d = pd.DatetimeIndex(d[-KRONOS_CONTEXT:])
            fut = pd.bdate_range(d[-1] + pd.Timedelta(days=1), periods=h)  # business-day stamps for the future
            x = np.stack([c, c, c, c, np.zeros_like(c), np.zeros_like(c)], 1)        # o=h=l=c, no volume
            mu, sd = x.mean(0), x.std(0)
            xn = np.clip((x - mu) / (sd + 1e-5), -5, 5)
            xs = st["cts"](pd.Series(d)).values.astype(np.float32)
            ys = st["cts"](pd.Series(fut)).values.astype(np.float32)
            S = KRONOS_SAMPLES
            with torch.no_grad():
                pr = st["ari"](st["tok"], st["m"], torch.from_numpy(np.repeat(xn[None], S, 0)),
                               torch.from_numpy(np.repeat(xs[None], S, 0)), torch.from_numpy(np.repeat(ys[None], S, 0)),
                               KRONOS_CONTEXT, h, 5, 1.0, 0, 0.9, 1, False)     # T=1, top_k=0, top_p=0.9, 1 sample/row
            close = pr[:, -h:, 3] * (sd[3] + 1e-5) + mu[3]                       # [S, h]
            out[i] = np.quantile(close, qs, axis=0)
        return out

    def predict(context, horizon, quantiles, dates=None):
        return batch([context], horizon, quantiles, [dates])[0]
    predict.batch = batch
    predict.wants_dates = True
    predict.meta = {"hf": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                    "code": KRONOS_REPO + " @ 67b630e", "samples": KRONOS_SAMPLES, "context": KRONOS_CONTEXT,
                    "T": 1.0, "top_p": 0.9, "top_k": 0, "seed": 0, "ohlc": "open=high=low=close=price, volume=0"}
    return predict


# --------------------------------------------------------------------------- Alma (frozen train_wm checkpoints)
def _alma(ckpt, tta="0"):
    """Runs train_wm.py itself in eval-only mode (light path of tools/fev_full_alma.py: TRAIN not loaded, stored
    config verified) and captures its predict(ctxs, h, m_season, freq) -> [n, 9, h] (levels 0.1..0.9)."""
    st = {}
    path = os.path.join(ROOT, ckpt)
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)       # audit B1: refuse incomplete checkpoints
    if ck.get("complete") is not True or ck.get("step") != (ck.get("budget") or {}).get("steps"):
        raise RuntimeError(f"{ckpt}: incomplete checkpoint (step={ck.get('step')}, complete={ck.get('complete')}, "
                           f"budget={ck.get('budget')}); refused")
    del ck

    def load():
        import torch
        torch.set_num_threads(2)
        saved = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("TSAR_")]}
        os.environ.update({"TSAR_EVAL_ONLY": path, "TSAR_DEVICE": "cpu"})
        if tta != "0":
            os.environ["TSAR_WM_TTA"] = tta
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import bench as BN
        import fev_full_alma as FA
        cap = {}
        orig = BN.evaluate

        def grab(predict_fn, split="val", verbose=True, names=None, tasks=None):
            cap["fn"] = predict_fn
            nan = float("nan")
            return {"wql_rel": nan, "mase_rel": nan, "cov80_error": nan, "predict_seconds": 0.0,
                    "wql_rel_family": nan, "wql_rel_subday": nan}
        BN.evaluate = grab
        src = open(os.path.join(ROOT, "train_wm.py")).read()
        g = {"__name__": "__alma__", "__file__": os.path.join(ROOT, "train_wm.py"), "__builtins__": __builtins__}
        argv = sys.argv
        sys.argv = [os.path.join(ROOT, "train_wm.py")]
        try:
            exec(compile(FA._light(src, path), g["__file__"], "exec"), g)
        finally:
            sys.argv = argv
            BN.evaluate = orig
            for k in [k for k in os.environ if k.startswith("TSAR_")]:
                del os.environ[k]
            os.environ.update(saved)
        st["fn"], st["sha"] = cap["fn"], g.get("CK_SHA")

    def batch(ctxs, h, qs, freq="B"):
        if "fn" not in st:
            load()
        season = {"B": 5, "W": 52}[freq]
        q = st["fn"]([np.asarray(c[-FM_CONTEXT:], np.float32) for c in ctxs], h, season, freq)
        return _fill(qs, NATIVE9, np.sort(np.asarray(q, np.float64), axis=1))
    p = _single(batch)
    p.wants_freq = True
    p.meta = {"ckpt": ckpt, "ckpt_sha256": _sha(path), "tta": tta, "context": "256 (train_wm CONTEXT crop)",
              "season": "5 daily (B), 52 weekly (W)"}
    return p


REGISTRY = {
    "rw_gauss": rw_gauss,
    "auto_ets": lambda: _stats("ets"),
    "auto_arima": lambda: _stats("arima"),
    "chronos2": lambda: _chronos("amazon/chronos-2", None),
    "chronos_bolt_base": lambda: _chronos("amazon/chronos-bolt-base", set(NATIVE9)),
    "tirex": _tirex,
    "timestral_5m": lambda: _timestral("mistralai/Timestral-5M"),
    "kronos_small": _kronos,
}
for _s in (0, 1, 2, 3, 4):
    for _abl in ("none", "direct"):
        _ck = f"data/ckpt/wm_ssm_{_abl}_s{_s}.pt"
        if os.path.exists(os.path.join(ROOT, _ck)):
            _nm = f"alma_s{_s}" if _abl == "none" else f"alma_direct_s{_s}"
            REGISTRY[_nm] = (lambda ck: (lambda: _alma(ck)))(_ck)
            REGISTRY[_nm + "_tta"] = (lambda ck: (lambda: _alma(ck, "1")))(_ck)

# architect modules: tools/commodity/wm_*.py defining register(REGISTRY)
LOAD_ERRORS = {}
for _f in sorted(glob.glob(os.path.join(HERE, "wm_*.py"))):
    _spec = importlib.util.spec_from_file_location(os.path.basename(_f)[:-3], _f)
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "register"):
            _mod.register(REGISTRY)
    except Exception as _e:                      # a broken architect module must not break VAL/DEV debugging ...
        print(f"models.py: skipping {_f}: {type(_e).__name__}: {_e}", file=sys.stderr)
        LOAD_ERRORS[_f] = f"{type(_e).__name__}: {_e}"   # ... but run_eval's strict (TEST) mode aborts on any entry


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
