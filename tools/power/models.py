"""
Model registry for Alma-Power (interface: tools/power/README.md; settings fixed in docs/power_prereg.md section 5).

    predict(context_prices [T], context_covariates [T, 4] | None, horizon=24, quantiles=QS) -> [len(quantiles), 24]

Every factory is lazy. Batch signature: predict.batch(prices_list, covs_list, horizon, quantiles, **kw) -> [n, nq, 24].
"""
import os
import sys
import glob
import hashlib
import warnings
import importlib.util

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE9 = [round(0.1 * i, 1) for i in range(1, 10)]
FM_CONTEXT = 2048           # hours
NAIVE_WINDOW = 364          # days for empirical naive intervals
CAL_DAYS = 728              # LEAR / ARX calibration window (days)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _fill(want, have_levels, have_q):
    hv = [round(float(x), 4) for x in have_levels]
    out = np.full((have_q.shape[0], len(want), have_q.shape[2]), np.nan, dtype=np.float64)
    for j, q in enumerate(want):
        if round(float(q), 4) in hv:
            out[:, j] = have_q[:, hv.index(round(float(q), 4))]
    return out


def _single(batch_fn):
    def predict(context_prices, context_covariates=None, horizon=24, quantiles=None, **kw):
        kw2 = {k: [v] for k, v in kw.items()}
        return batch_fn([context_prices], [context_covariates], horizon, quantiles, **kw2)[0]
    predict.batch = batch_fn
    return predict


def _days(x):
    return np.asarray(x, float).reshape(-1, 24)


# --------------------------------------------------------------------------- naive baselines
def _naive(kind):
    """kind: 'd' (same slot yesterday), 'w' (same slot last week), 'epf' (w on Mon/Sat/Sun, else d)."""
    def one(P, lag, qs):
        med = P[-lag]
        W = P[-(NAIVE_WINDOW + lag):]
        diff = W[lag:] - W[:-lag]                                    # [<=364, 24]
        out = np.empty((len(qs), 24))
        for h in range(24):
            dh = diff[:, h][np.isfinite(diff[:, h])]
            out[:, h] = med[h] + np.quantile(dh, qs) if len(dh) > 20 else np.nan
        out[[j for j, q in enumerate(qs) if round(q, 4) == 0.5]] = med     # median exactly = the naive value
        return out

    def batch(prices, covs, h, qs, dates=None):
        res = []
        for j, p in enumerate(prices):
            P = _days(p)
            if kind == "d":
                lag = 1
            elif kind == "w":
                lag = 7
            else:
                wd = int(((np.datetime64(dates[j]["target_day"], "D").astype(int) + 3) % 7))   # Mon=0
                lag = 7 if wd in (0, 5, 6) else 1
            res.append(one(P, lag, qs))
        return np.stack(res)
    p = _single(batch)
    p.wants_dates = kind == "epf"
    p.uses_covariates = False
    p.meta = {"naive": kind, "interval_window_days": NAIVE_WINDOW}
    return p


# --------------------------------------------------------------------------- LEAR / ARX
def _vst_fit(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return med, (mad if mad > 1e-6 else (np.nanstd(x) + 1e-6))


def _wd(day):
    return int((np.datetime64(day, "D").astype(int) + 3) % 7)


def _lear_features(P, C, i, use_cov):
    """Features for target day index i of the day panel P [n,24] / C [n,24,4] (context-relative indices)."""
    f = [P[i - 1], P[i - 2], P[i - 3], P[i - 7]]
    if use_cov:
        f += [C[i - 2, :, k] for k in range(C.shape[2])] + [C[i - 7, :, k] for k in range(C.shape[2])]
    return np.concatenate(f)


def _lear(use_cov=True):
    def batch(prices, covs, h, qs, dates=None):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):                       # single-threaded BLAS: bit-reproducible
            return _batch(prices, covs, h, qs, dates)

    def _batch(prices, covs, h, qs, dates=None):
        from sklearn.linear_model import LassoLarsIC, Lasso
        res = []
        for j, p in enumerate(prices):
            P = _days(p)
            n = len(P)
            C = np.asarray(covs[j], float).reshape(n, 24, -1)
            ctx_days = dates[j]["ctx_dates"]
            tday = dates[j]["target_day"]
            # extended panel: append the target day (unknown prices) so that i = n is the target
            cal = range(n - CAL_DAYS, n)
            Xp = np.stack([_lear_features(P, C, i, use_cov) for i in list(cal) + [n]])   # last row = target
            wd = np.array([_wd(ctx_days[i]) for i in cal] + [_wd(tday)])
            D7 = np.eye(7)[wd]
            Y = P[n - CAL_DAYS:n]                                                       # [CAL, 24]
            # VST: prices with the calibration-window price stats, each covariate with its own
            pm, ps = _vst_fit(P[n - CAL_DAYS:n])
            Xt = Xp.copy()
            npf = 96
            Xt[:, :npf] = np.arcsinh((Xp[:, :npf] - pm) / ps)
            if use_cov:
                for k in range(C.shape[2]):
                    cm, cs = _vst_fit(C[n - CAL_DAYS:n, :, k])
                    for blk in range(2):
                        a = npf + (blk * C.shape[2] + k) * 24
                        Xt[:, a:a + 24] = np.arcsinh((Xp[:, a:a + 24] - cm) / cs)
            Xt = np.hstack([Xt, D7])
            Yt = np.arcsinh((Y - pm) / ps)
            ok_rows = np.isfinite(Xt[:-1]).all(1)
            x_target = Xt[-1]
            if not np.isfinite(x_target).all():
                x_target = np.where(np.isfinite(x_target), x_target, 0.0)          # missing input -> neutral (0 = median)
            out = np.empty((len(qs), 24))
            for s in range(24):
                ok = ok_rows & np.isfinite(Yt[:, s])
                Xs, ys = Xt[:-1][ok], Yt[ok, s]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        a = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xs, ys).alpha_
                    except Exception:
                        a = 1e-3
                    m = Lasso(alpha=max(a, 1e-6), max_iter=5000).fit(Xs, ys)
                pt = m.predict(x_target[None])[0]
                r = ys - m.predict(Xs)
                out[:, s] = np.sinh(pt + np.quantile(r, qs)) * ps + pm
                out[[k for k, q in enumerate(qs) if round(q, 4) == 0.5], s] = np.sinh(pt) * ps + pm
            res.append(out)
        return np.stack(res)
    p = _single(batch)
    p.wants_dates = True
    p.uses_covariates = use_cov
    p.meta = {"calibration_days": CAL_DAYS, "use_cov": use_cov, "vst": "asinh((x-median)/MAD)",
              "alpha": "LassoLarsIC(aic) -> Lasso", "intervals": "in-sample residual quantiles"}
    return p


def _arx():
    def batch(prices, covs, h, qs, dates=None):
        res = []
        for j, p in enumerate(prices):
            P = _days(p)
            n = len(P)
            C = np.asarray(covs[j], float).reshape(n, 24, -1)
            ctx_days, tday = dates[j]["ctx_dates"], dates[j]["target_day"]
            pm, ps = _vst_fit(P[n - CAL_DAYS:n])
            T = lambda x: np.arcsinh((x - pm) / ps)          # noqa: E731
            stats = {}
            for k in range(C.shape[2]):
                stats[k] = _vst_fit(C[n - CAL_DAYS:n, :, k])
            TC = lambda x, k: np.arcsinh((x - stats[k][0]) / stats[k][1])   # noqa: E731
            wind = C[:, :, 1] + C[:, :, 2]
            wm, ws = _vst_fit(wind[n - CAL_DAYS:n])
            out = np.empty((len(qs), 24))
            idx = list(range(n - CAL_DAYS, n)) + [n]
            wd = np.eye(7)[[_wd(ctx_days[i]) for i in idx[:-1]] + [_wd(tday)]]
            for s in range(24):
                X = np.stack([[T(P[i - 1, s]), T(P[i - 2, s]), T(P[i - 7, s]), T(np.nanmin(P[i - 1])),
                               T(np.nanmax(P[i - 1])), T(P[i - 1, 23]), TC(C[i - 2, s, 0], 0),
                               np.arcsinh((wind[i - 2, s] - wm) / ws), TC(C[i - 2, s, 3], 3)] for i in idx])
                X = np.hstack([X, wd])
                y = T(P[n - CAL_DAYS:n, s])
                ok = np.isfinite(X[:-1]).all(1) & np.isfinite(y)
                beta, *_ = np.linalg.lstsq(X[:-1][ok], y[ok], rcond=None)
                r = y[ok] - X[:-1][ok] @ beta
                sd = np.std(r, ddof=X.shape[1])
                xt = np.where(np.isfinite(X[-1]), X[-1], 0.0)
                pt = xt @ beta
                out[:, s] = np.sinh(pt + norm.ppf(qs) * sd) * ps + pm
            res.append(out)
        return np.stack(res)
    p = _single(batch)
    p.wants_dates = True
    p.uses_covariates = True
    p.meta = {"calibration_days": CAL_DAYS, "intervals": "gaussian residual sd (transformed scale)"}
    return p


# --------------------------------------------------------------------------- foundation models (CPU)
def _clean_ctx(p):
    c = np.asarray(p[-FM_CONTEXT:], np.float32)
    return c


def _chronos(model_id, native, use_cov=False):
    st = {}

    def batch(prices, covs, h, qs):
        import torch
        from chronos import BaseChronosPipeline
        if "pipe" not in st:
            torch.manual_seed(0)
            st["pipe"] = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        ask = [q for q in qs if native is None or round(q, 4) in native]
        out = []
        for s in range(0, len(prices), 16):
            if use_cov:
                chunk = []
                for p, c in zip(prices[s:s + 16], covs[s:s + 16]):
                    c = np.asarray(c, np.float32)
                    sh = np.full_like(c, np.nan)
                    sh[24:] = c[:-24]                                # row t carries the value at t-24h (<= D-2)
                    sh = sh[-FM_CONTEXT:]
                    chunk.append({"target": _clean_ctx(p),
                                  "past_covariates": {f"c{k}": sh[:, k] for k in range(sh.shape[1])}})
            else:
                chunk = [torch.tensor(_clean_ctx(p)) for p in prices[s:s + 16]]
            q, _ = st["pipe"].predict_quantiles(chunk, prediction_length=h, quantile_levels=ask)
            if isinstance(q, list):
                q = torch.stack([t.reshape(-1, h, len(ask))[0] for t in q])
            out.append(q.reshape(len(chunk), h, len(ask)).permute(0, 2, 1).double().numpy())
        return _fill(qs, ask, np.concatenate(out))
    p = _single(batch)
    p.uses_covariates = use_cov
    p.meta = {"hf": model_id, "context_hours": FM_CONTEXT, "device": "cpu",
              "covariates": "4 fundamentals shifted 24h (past only)" if use_cov else None}
    return p


def _tirex():
    st = {}

    def batch(prices, covs, h, qs):
        import torch
        from tirex import load_model
        if "m" not in st:
            st["m"] = load_model("NX-AI/TiRex", device="cpu")
        out = []
        for s in range(0, len(prices), 16):
            chunk = [torch.tensor(_clean_ctx(p)) for p in prices[s:s + 16]]
            q, _ = st["m"].forecast(context=chunk, prediction_length=h)
            out.append(np.asarray(q, dtype=np.float64).transpose(0, 2, 1))
        return _fill(qs, NATIVE9, np.concatenate(out))
    p = _single(batch)
    p.uses_covariates = False
    p.meta = {"hf": "NX-AI/TiRex", "context_hours": FM_CONTEXT, "device": "cpu"}
    return p


def _timestral(model_id):
    st = {}

    def batch(prices, covs, h, qs):
        from timestral import TimestralForecaster
        if "m" not in st:
            st["m"] = TimestralForecaster.from_pretrained(model_id)
        out = []
        for s in range(0, len(prices), 16):
            chunk = [_clean_ctx(p) for p in prices[s:s + 16]]
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
    p.uses_covariates = False
    p.meta = {"hf": model_id, "context_hours": FM_CONTEXT, "device": "cpu",
              "contamination": "released 2026-10-15: training data may cover the TEST period"}
    return p


KRONOS_REPO = "/Users/srimanarayana/MIKO/kronos/repo"     # official github.com/shiyu-coder/Kronos @ 67b630e
KRONOS_SAMPLES, KRONOS_CONTEXT = 10, 256


def _kronos():
    st = {}

    def batch(prices, covs, h, qs, dates):
        import torch
        import pandas as pd
        if "m" not in st:
            sys.path.insert(0, KRONOS_REPO)
            from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps
            st.update(tok=KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval(),
                      m=Kronos.from_pretrained("NeoQuasar/Kronos-small").eval(),
                      ari=auto_regressive_inference, cts=calc_time_stamps)
        out = np.empty((len(prices), len(qs), h))
        for i, (c, d) in enumerate(zip(prices, dates)):
            torch.manual_seed(0)
            np.random.seed(0)
            c = np.asarray(c[-KRONOS_CONTEXT:], np.float32)
            t0 = pd.Timestamp(d["target_day"])                       # slot timestamps (local clock, naive)
            past = pd.date_range(t0 - pd.Timedelta(hours=KRONOS_CONTEXT), periods=KRONOS_CONTEXT, freq="h")
            fut = pd.date_range(t0, periods=h, freq="h")
            x = np.stack([c, c, c, c, np.zeros_like(c), np.zeros_like(c)], 1)
            mu, sd = x.mean(0), x.std(0)
            xn = np.clip((x - mu) / (sd + 1e-5), -5, 5)
            xs = st["cts"](pd.Series(past)).values.astype(np.float32)
            ys = st["cts"](pd.Series(fut)).values.astype(np.float32)
            S = KRONOS_SAMPLES
            with torch.no_grad():
                pr = st["ari"](st["tok"], st["m"], torch.from_numpy(np.repeat(xn[None], S, 0)),
                               torch.from_numpy(np.repeat(xs[None], S, 0)), torch.from_numpy(np.repeat(ys[None], S, 0)),
                               KRONOS_CONTEXT, h, 5, 1.0, 0, 0.9, 1, False)
            close = pr[:, -h:, 3] * (sd[3] + 1e-5) + mu[3]
            out[i] = np.quantile(close, qs, axis=0)
        return out
    p = _single(batch)
    p.wants_dates = True
    p.uses_covariates = False
    p.meta = {"hf": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
              "code": KRONOS_REPO + " @ 67b630e", "samples": KRONOS_SAMPLES, "context_hours": KRONOS_CONTEXT,
              "T": 1.0, "top_p": 0.9, "top_k": 0, "seed": 0, "ohlc": "open=high=low=close=price, volume=0"}
    return p


# --------------------------------------------------------------------------- Alma (frozen train_wm checkpoints)
def _alma(ckpt):
    """train_wm.py in eval-only mode (light path of tools/fev_full_alma.py), as in tools/commodity/models.py;
    season 24, freq 'H'. Returns levels 0.1..0.9."""
    st = {}
    path = os.path.join(ROOT, ckpt)
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("complete") is not True or ck.get("step") != (ck.get("budget") or {}).get("steps"):
        raise RuntimeError(f"{ckpt}: incomplete checkpoint (step={ck.get('step')}, complete={ck.get('complete')}); refused")
    del ck

    def load():
        import torch
        torch.set_num_threads(2)
        saved = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("TSAR_")]}
        os.environ.update({"TSAR_EVAL_ONLY": path, "TSAR_DEVICE": "cpu"})
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
        st["fn"] = cap["fn"]

    def batch(prices, covs, h, qs):
        if "fn" not in st:
            load()
        q = st["fn"]([np.asarray(p[-FM_CONTEXT:], np.float32) for p in prices], h, 24, "H")
        return _fill(qs, NATIVE9, np.sort(np.asarray(q, np.float64), axis=1))
    p = _single(batch)
    p.uses_covariates = False
    p.meta = {"ckpt": ckpt, "ckpt_sha256": _sha(path), "context": "256 (train_wm CONTEXT crop)", "season": 24,
              "freq": "H"}
    return p


REGISTRY = {
    "naive_d": lambda: _naive("d"),
    "naive_w": lambda: _naive("w"),
    "naive_epf": lambda: _naive("epf"),
    "lear": lambda: _lear(True),
    "lear_nocov": lambda: _lear(False),
    "arx": _arx,
    "chronos2": lambda: _chronos("amazon/chronos-2", None),
    "chronos2_cov": lambda: _chronos("amazon/chronos-2", None, use_cov=True),
    "chronos_bolt_base": lambda: _chronos("amazon/chronos-bolt-base", set(NATIVE9)),
    "tirex": _tirex,
    "timestral_5m": lambda: _timestral("mistralai/Timestral-5M"),
    "kronos_small": _kronos,
    "alma_s0": lambda: _alma("data/ckpt/wm_ssm_none_s0.pt"),
    "alma_direct_s0": lambda: _alma("data/ckpt/wm_ssm_direct_s0.pt"),
}

LOAD_ERRORS = {}
for _f in sorted(glob.glob(os.path.join(HERE, "wm_*.py"))):
    _spec = importlib.util.spec_from_file_location("power_" + os.path.basename(_f)[:-3], _f)
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "register"):
            _mod.register(REGISTRY)
    except Exception as _e:
        print(f"power/models.py: skipping {_f}: {type(_e).__name__}: {_e}", file=sys.stderr)
        LOAD_ERRORS[_f] = f"{type(_e).__name__}: {_e}"


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
