"""
Alma-Gas model registry (docs/gas_prereg.md section 5 fixes every setting; changes need a dated amendment).

Two task interfaces; a registered factory returns an object implementing one or both:

  PRICE (daily HH spot):   obj.batch(ctxs, h, qs) -> array [n, len(qs), h]
      ctxs: list of dicts from data.build_context (keys y, covs, ctx_dates, origin_date); step k = k-th future
      panel row. Quantiles of the price level (USD/MMBtu).
  STORAGE (weekly EIA):    obj.storage(ctxs, qs, series) -> array [n, len(qs)]
      ctxs: list of dicts from storage.build_context; quantiles of dS_W (Bcf) of `series` for the target week.

Optional attributes: obj.meta (dict, recorded), obj.uses_covariates, obj.stride (forecast every stride-th origin).
Architect modules: tools/gas/wm_*.py defining register(REGISTRY) are imported at the end (import errors are recorded
in LOAD_ERRORS; strict TEST mode aborts on any). Models receive only the context dicts: no file access at inference.
"""
import os
import sys
import glob
import hashlib
import importlib.util

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
NATIVE9 = [round(0.1 * i, 1) for i in range(1, 10)]
FM_CONTEXT = 1024          # daily: last 1024 observations
STATS_CONTEXT = 512
RW_WINDOW = 250
WK_CONTEXT = 520           # weekly storage series: last 520 weeks (10 years)
KRONOS_REPO = "/Users/srimanarayana/MIKO/kronos/repo"     # official github.com/shiyu-coder/Kronos @ 67b630e
KRONOS_SAMPLES, KRONOS_CONTEXT, KRONOS_STRIDE = 10, 256, 5


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _fill(want, have_levels, have_q):
    hv = [round(float(x), 4) for x in have_levels]
    out = np.full((have_q.shape[0], len(want), have_q.shape[2]), np.nan, dtype=np.float64)
    for j, q in enumerate(want):
        if round(float(q), 4) in hv:
            out[:, j] = have_q[:, hv.index(round(float(q), 4))]
    return out


class Model:
    def __init__(self, meta=None, **kw):
        self.meta = meta or {}
        for k, v in kw.items():
            setattr(self, k, v)


# =========================================================================== generic univariate forecasters
# f(series_list, h, qs, freq) -> [n, len(qs), h]; used by both tasks (price: hh levels; storage: weekly dS)
def _u_rw(series, h, qs, freq):
    out = np.empty((len(series), len(qs), h))
    k = np.sqrt(np.arange(1, h + 1))
    z = norm.ppf(qs)
    for i, c in enumerate(series):
        c = np.asarray(c, float)
        if freq == "B":                                 # log random walk on prices
            pos = c[-(RW_WINDOW + 1):]
            pos = pos[pos > 0]
            s = np.std(np.diff(np.log(pos)), ddof=1)
            out[i] = max(c[-1], 0.01) * np.exp(z[:, None] * s * k[None, :])
        else:
            raise ValueError("rw only for prices")
    return out


def _u_stats(kind):
    LV = [20, 40, 60, 80, 90, 95]
    lv_q = {}
    for L in LV:
        lv_q[round((1 - L / 100) / 2, 4)] = f"lo-{L}"
        lv_q[round(1 - (1 - L / 100) / 2, 4)] = f"hi-{L}"

    def f(series, h, qs, freq):
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, AutoARIMA
        ctx = STATS_CONTEXT if freq == "B" else WK_CONTEXT
        rows = [pd.DataFrame({"unique_id": i, "ds": np.arange(len(c[-ctx:])), "y": np.asarray(c[-ctx:], float)})
                for i, c in enumerate(series)]
        model = AutoETS(season_length=1) if kind == "ets" else AutoARIMA(season_length=1)
        fc = StatsForecast(models=[model], freq=1, n_jobs=1).forecast(df=pd.concat(rows), h=h, level=LV)
        fc = fc.sort_values(["unique_id", "ds"])
        name = repr(model)
        out = np.full((len(series), len(qs), h), np.nan)
        for j, q in enumerate(qs):
            col = name if round(q, 4) == 0.5 else (f"{name}-{lv_q[round(q, 4)]}" if round(q, 4) in lv_q else None)
            if col is not None:
                out[:, j] = fc[col].to_numpy().reshape(len(series), h)
        return out
    return f


def _u_chronos(model_id, native):
    st = {}

    def f(series, h, qs, freq, past_cov=None, fut_cov=None):
        import torch
        from chronos import BaseChronosPipeline
        if "pipe" not in st:
            torch.manual_seed(0)
            st["pipe"] = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        ctx = FM_CONTEXT if freq == "B" else WK_CONTEXT
        ask = [q for q in qs if native is None or round(q, 4) in native]
        out = []
        for s in range(0, len(series), 16):
            if past_cov is not None:
                chunk = []
                for j in range(s, min(s + 16, len(series))):
                    item = {"target": np.asarray(series[j][-ctx:], np.float32),
                            "past_covariates": {kk: np.asarray(v[-ctx:], np.float32) for kk, v in past_cov[j].items()}}
                    if fut_cov is not None:
                        item["future_covariates"] = {kk: np.asarray(v, np.float32) for kk, v in fut_cov[j].items()}
                    chunk.append(item)
            else:
                chunk = [torch.tensor(np.asarray(c[-ctx:], np.float32)) for c in series[s:s + 16]]
            q, _ = st["pipe"].predict_quantiles(chunk, prediction_length=h, quantile_levels=ask)
            if isinstance(q, list):
                q = torch.stack([t.reshape(-1, h, len(ask))[0] for t in q])
            out.append(q.reshape(len(chunk), h, len(ask)).permute(0, 2, 1).double().numpy())
        return _fill(qs, ask, np.concatenate(out))
    return f


def _u_tirex():
    st = {}

    def f(series, h, qs, freq):
        import torch
        from tirex import load_model
        if "m" not in st:
            st["m"] = load_model("NX-AI/TiRex", device="cpu")
        ctx = FM_CONTEXT if freq == "B" else WK_CONTEXT
        out = []
        for s in range(0, len(series), 16):
            chunk = [torch.tensor(np.asarray(c[-ctx:], np.float32)) for c in series[s:s + 16]]
            q, _ = st["m"].forecast(context=chunk, prediction_length=h)
            out.append(np.asarray(q, dtype=np.float64).transpose(0, 2, 1))
        return _fill(qs, NATIVE9, np.concatenate(out))
    return f


def _u_timestral(model_id):
    st = {}

    def f(series, h, qs, freq):
        from timestral import TimestralForecaster
        if "m" not in st:
            st["m"] = TimestralForecaster.from_pretrained(model_id)
        ctx = FM_CONTEXT if freq == "B" else WK_CONTEXT
        out = []
        for s in range(0, len(series), 16):
            chunk = [np.asarray(c[-ctx:], np.float32) for c in series[s:s + 16]]
            L = max(len(c) for c in chunk)
            x = np.full((len(chunk), L), np.nan, dtype=np.float32)
            for i, c in enumerate(chunk):
                x[i, L - len(c):] = c
            fc = st["m"].forecast(x, prediction_length=h)
            lev = [round(float(v), 4) for v in fc.quantile_levels]
            q = np.asarray(fc.quantiles).reshape(len(chunk), h, len(lev)).transpose(0, 2, 1)
            out.append(_fill(qs, lev, q))
        return np.concatenate(out)
    return f


def _u_alma(ckpt):
    """train_wm.py eval-only light path (as tools/commodity/models.py and tools/power/models.py); refuses incomplete
    checkpoints. Season 5 (freq B) or 52 (freq W). Levels 0.1..0.9."""
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

    def f(series, h, qs, freq):
        if "fn" not in st:
            load()
        season = {"B": 5, "W": 52}[freq]
        ctx = FM_CONTEXT if freq == "B" else WK_CONTEXT
        q = st["fn"]([np.asarray(c[-ctx:], np.float32) for c in series], h, season, freq)
        return _fill(qs, NATIVE9, np.sort(np.asarray(q, np.float64), axis=1))
    f.meta = {"ckpt": ckpt, "ckpt_sha256": _sha(path), "context": "256 (train_wm CONTEXT crop)",
              "season": "5 (price, B) / 52 (storage, W)"}
    return f


def _kronos_price():
    st = {}

    def batch(ctxs, h, qs):
        import torch
        import pandas as pd
        if "m" not in st:
            sys.path.insert(0, KRONOS_REPO)
            from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps
            st.update(tok=KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval(),
                      m=Kronos.from_pretrained("NeoQuasar/Kronos-small").eval(),
                      ari=auto_regressive_inference, cts=calc_time_stamps)
        out = np.empty((len(ctxs), len(qs), h))
        for i, w in enumerate(ctxs):
            torch.manual_seed(0)
            np.random.seed(0)
            c = np.asarray(w["y"][-KRONOS_CONTEXT:], np.float32)
            d = pd.DatetimeIndex(w["ctx_dates"][-KRONOS_CONTEXT:])
            fut = pd.bdate_range(d[-1] + pd.Timedelta(days=1), periods=h)
            x = np.stack([c, c, c, c, np.zeros_like(c), np.zeros_like(c)], 1)
            mu, sd = x.mean(0), x.std(0)
            xn = np.clip((x - mu) / (sd + 1e-5), -5, 5)
            xs = st["cts"](pd.Series(d)).values.astype(np.float32)
            ys = st["cts"](pd.Series(fut)).values.astype(np.float32)
            S = KRONOS_SAMPLES
            with torch.no_grad():
                pr = st["ari"](st["tok"], st["m"], torch.from_numpy(np.repeat(xn[None], S, 0)),
                               torch.from_numpy(np.repeat(xs[None], S, 0)), torch.from_numpy(np.repeat(ys[None], S, 0)),
                               KRONOS_CONTEXT, h, 5, 1.0, 0, 0.9, 1, False)
            close = pr[:, -h:, 3] * (sd[3] + 1e-5) + mu[3]
            out[i] = np.quantile(close, qs, axis=0)
        return out
    return Model(meta={"hf": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                       "code": KRONOS_REPO + " @ 67b630e", "samples": KRONOS_SAMPLES, "context": KRONOS_CONTEXT,
                       "T": 1.0, "top_p": 0.9, "top_k": 0, "seed": "0 per origin",
                       "ohlc": "open=high=low=close=price, volume=0"},
                 batch=batch, stride=KRONOS_STRIDE)


# =========================================================================== wrappers: generic -> task interfaces
def wrap(f, meta, price=True, storage=True, transform_storage="ds"):
    m = Model(meta=dict(meta, **getattr(f, "meta", {})), uses_covariates=False)
    if price:
        m.batch = lambda ctxs, h, qs: np.asarray(f([c["y"] for c in ctxs], h, qs, "B"), float)
    if storage:
        def st(ctxs, qs, series):
            q = np.asarray(f([c["ds_hist"][series] for c in ctxs], 1, qs, "W"), float)[:, :, 0]
            return np.sort(q, axis=1) if np.isfinite(q).all() else q
        m.storage = st
    return m


def _chronos2_cov():
    """Chronos-2 with covariates. Storage: past = weekly observed HDD/CDD of the history weeks, future = the target
    week's HDD/CDD as known at the origin (observed days + CPC forecast days). Price: past = storage dev5 and hdd7,
    cdd7 columns of the panel (all known at their row date)."""
    f = _u_chronos("amazon/chronos-2", None)

    def st(ctxs, qs, series):
        pc, fc = [], []
        for c in ctxs:
            wx = c["wx"].copy()
            bad = ~np.isfinite(wx)
            wx[bad] = c["wx_normal"][bad]
            pc.append({"hdd": np.nan_to_num(c["hist_hdd"], nan=float(np.nanmean(c["hist_hdd"]))),
                       "cdd": np.nan_to_num(c["hist_cdd"], nan=float(np.nanmean(c["hist_cdd"])))})
            fc.append({"hdd": np.array([wx[:, 0].sum()]), "cdd": np.array([wx[:, 1].sum()])})
        q = f([c["ds_hist"][series] for c in ctxs], 1, qs, "W", pc, fc)[:, :, 0]
        return np.sort(q, axis=1)

    def batch(ctxs, h, qs):
        import data as D
        ix = [D.COVS.index(k) for k in ("stor_dev5", "hdd7", "cdd7")]
        pc = []
        for c in ctxs:
            X = c["covs"][:, ix]
            pc.append({f"c{j}": np.nan_to_num(X[:, j], nan=0.0) for j in range(X.shape[1])})
        return np.asarray(f([c["y"] for c in ctxs], h, qs, "B", pc, None), float)
    return Model(meta={"hf": "amazon/chronos-2", "covariates": "storage: weekly HDD/CDD past + target-week future; "
                       "price: stor_dev5, hdd7, cdd7 past only (NaN -> 0)"}, storage=st, batch=batch,
                 uses_covariates=True)


# =========================================================================== storage baselines
def _q_from(center, resid, qs):
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    return center + np.quantile(resid, qs) if len(resid) >= 10 else np.full(len(qs), np.nan)


def _woy_dist(hist_weeks, target_week):
    import pandas as pd
    hw = pd.to_datetime(hist_weeks)
    t = pd.Timestamp(target_week)
    doy = (hw.dayofyear.to_numpy() - t.dayofyear) % 365
    doy = np.minimum(doy, 365 - doy)
    return doy, (t - hw).days.to_numpy()


def st_climatology():
    """Median of past deviations of the same report week (+-7 days of year, all previous years in the context);
    quantiles: avg5 + empirical quantiles of past deviations within +-14 days of year."""
    def st(ctxs, qs, series):
        out = []
        for c in ctxs:
            doy, age = _woy_dist(c["hist_weeks"], c["target_week"])
            dv = c["dev_hist"][series]
            near = (doy <= 14) & (age > 300) & np.isfinite(dv)
            med = np.median(dv[(doy <= 7) & (age > 300) & np.isfinite(dv)]) if near.any() else 0.0
            q = c["avg5"][series] + np.quantile(dv[near], qs) if near.sum() >= 10 else np.full(len(qs), np.nan)
            q[qs.index(0.5)] = c["avg5"][series] + med
            out.append(np.sort(q))
        return np.array(out)
    return Model(meta={"rule": "same-week past deviations"}, storage=st)


def st_persistence():
    """dev_W = dev_{W-1}: median avg5 + dev_{W-1}; quantiles from past one-week changes of dev."""
    def st(ctxs, qs, series):
        out = []
        for c in ctxs:
            dv = c["dev_hist"][series]
            out.append(_q_from(c["avg5"][series] + dv[-1], np.diff(dv[-260:]), qs))
        return np.array(out)
    return Model(meta={"rule": "dev persistence"}, storage=st)


def st_seasonal_naive():
    """dS_W = dS of the same week one year earlier (nearest week, 364 +- 3 days); quantiles from past 52-week
    differences of dS."""
    def st(ctxs, qs, series):
        out = []
        for c in ctxs:
            ds = c["ds_hist"][series]
            doy, age = _woy_dist(c["hist_weeks"], c["target_week"])
            j = np.where(np.abs(age - 364) <= 3)[0]
            if not len(j):
                out.append(np.full(len(qs), np.nan))
                continue
            d52 = ds[52:] - ds[:-52]
            out.append(_q_from(ds[j[0]], d52[-260:], qs))
        return np.array(out)
    return Model(meta={"rule": "last year's same-week dS"}, storage=st)


def wx_features(c):
    """[HDD anomaly, CDD anomaly] of the target week (sum over 7 days of (value - causal normal)); days without a
    value use the normal (anomaly 0). Values: observed days + CPC forecast days as known at the origin."""
    wx, nrm = c["wx"], c["wx_normal"]
    an = np.where(np.isfinite(wx), wx - nrm, 0.0)
    return np.nan_to_num(an.sum(0), nan=0.0)


class _OLSWx(Model):
    """dev_W ~ 1 + HDD anomaly + CDD anomaly + dev_{W-1}, OLS fitted ONCE on DEV target weeks released
    2014-01-15 .. 2025-06-30 (storage.train_weeks), separately for each origin kind and series, with the features
    built by storage.build_context at that origin; quantiles = point + empirical quantiles of the in-sample residuals."""

    def __init__(self, use_lag=True):
        super().__init__(meta={"fit": "DEV weeks released 2014-01-15..2025-06-30", "lag": use_lag})
        self.use_lag, self.fits = use_lag, {}

    def _x(self, c, series):
        x = [1.0, *wx_features(c)]
        if self.use_lag:
            x.append(c["dev_hist"][series][-1])
        return np.array(x, float)

    def _fit(self, kind, series):
        import storage as S
        key = (kind, series)
        if key not in self.fits:
            if "ctx" not in self.__dict__:
                self.ctx = {}
            X, y = [], []
            w = S.load()[0]
            for k in S.train_weeks():
                if (kind, k) not in self.ctx:
                    self.ctx[(kind, k)] = S.build_context(int(k), kind)
                c = self.ctx[(kind, k)]
                dv = float(w[f"dev_{series}"].iloc[k])
                xx = self._x(c, series)
                if np.isfinite(dv) and np.isfinite(xx).all():
                    X.append(xx), y.append(dv)
            X, y = np.array(X), np.array(y)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.fits[key] = (beta, y - X @ beta, len(y))
            self.meta[f"n_fit_{kind}_{series}"] = len(y)
            self.meta[f"beta_{kind}_{series}"] = [round(float(b), 4) for b in beta]
        return self.fits[key]

    def storage(self, ctxs, qs, series):
        out = []
        for c in ctxs:
            beta, res, _ = self._fit(c["kind"], series)
            out.append(_q_from(c["avg5"][series] + self._x(c, series) @ beta, res, qs))
        return np.array(out)


def st_arima_dev():
    f = _u_stats("arima")

    def st(ctxs, qs, series):
        q = f([c["dev_hist"][series][np.isfinite(c["dev_hist"][series])] for c in ctxs], 1, qs, "W")[:, :, 0]
        return q + np.array([c["avg5"][series] for c in ctxs])[:, None]
    return Model(meta={"model": "AutoARIMA on the deviation series, season 1, last 520 weeks"}, storage=st)


# =========================================================================== price: fundamentals regression
class _PxFund(Model):
    """h-day log return ~ 1 + stor_dev5/100 + stor_chg/100 + hdd7/100 + cdd7/100 (panel row of the origin, all known
    at its close), OLS per horizon fitted once on DEV origins whose targets are <= 2025-06-30 (from 2015-01-01, when
    stor_dev5 exists); quantiles = point + empirical in-sample residual quantiles, in log space."""

    def __init__(self):
        super().__init__(meta={"fit": "DEV origins 2015-01-01.. with targets <= 2025-06-30"}, uses_covariates=True)
        self.fit = None

    def _feat(self, covs_row):
        import data as D
        r = {k: covs_row[D.COVS.index(k)] for k in ("stor_dev5", "stor_chg", "hdd7", "cdd7")}
        return np.array([1.0, r["stor_dev5"] / 100, r["stor_chg"] / 100, r["hdd7"] / 100, r["cdd7"] / 100])

    def _train(self, H):
        import data as D
        dates, y, C, extra = D.train_panel()
        ly = np.log(np.maximum(y, 0.01))
        s = int(np.searchsorted(dates, np.datetime64("2015-01-01")))
        self.fit = []
        for h in range(1, H + 1):
            X, Y = [], []
            for j in range(s, len(y) - h):            # origin row j (close of dates[j]), target row j+h <= TRAIN_END
                x = self._feat(C[j])
                if np.isfinite(x).all():
                    X.append(x), Y.append(ly[j + h] - ly[j])
            X, Y = np.array(X), np.array(Y)
            b = np.linalg.lstsq(X, Y, rcond=None)[0]
            self.fit.append((b, Y - X @ b))
        self.meta["n_fit_h1"] = len(self.fit[0][1])

    def batch(self, ctxs, h, qs):
        if self.fit is None or len(self.fit) < h:
            self._train(h)
        out = np.full((len(ctxs), len(qs), h), np.nan)
        for i, c in enumerate(ctxs):
            x = self._feat(c["covs"][-1])
            last = np.log(max(c["y"][-1], 0.01))
            if not np.isfinite(x).all():
                continue
            for k in range(h):
                b, res = self.fit[k]
                out[i, :, k] = np.exp(last + x @ b + np.quantile(res, qs))
        return out


# =========================================================================== registry
def _fm(name, factory, meta, price=True, storage=True):
    return lambda: wrap(factory(), meta, price=price, storage=storage)


REGISTRY = {
    # price-only statistical baselines
    "rw_gauss": lambda: wrap(_u_rw, {"sigma_window": RW_WINDOW}, storage=False),
    "auto_ets": lambda: wrap(_u_stats("ets"), {"context": "512 daily"}, storage=False),
    "auto_arima": lambda: wrap(_u_stats("arima"), {"context": "512 daily"}, storage=False),
    "px_fund_ols": _PxFund,
    # storage-only baselines
    "st_climatology": st_climatology,
    "st_persistence": st_persistence,
    "st_seasonal_naive": st_seasonal_naive,
    "st_ols_wx": lambda: _OLSWx(True),
    "st_ols_wx_nolag": lambda: _OLSWx(False),
    "st_arima_dev": st_arima_dev,
    # foundation models (both tasks; storage = univariate weekly dS series)
    "chronos2": _fm("chronos2", lambda: _u_chronos("amazon/chronos-2", None), {"hf": "amazon/chronos-2"}),
    "chronos2_cov": _chronos2_cov,
    "chronos_bolt_base": _fm("cbb", lambda: _u_chronos("amazon/chronos-bolt-base", set(NATIVE9)),
                             {"hf": "amazon/chronos-bolt-base"}),
    "tirex": _fm("tirex", _u_tirex, {"hf": "NX-AI/TiRex"}),
    "timestral_5m": _fm("ts5", lambda: _u_timestral("mistralai/Timestral-5M"),
                        {"hf": "mistralai/Timestral-5M", "contamination": "released 2026-10-15: may cover TEST"}),
    "kronos_small": _kronos_price,
    "alma_s0": lambda: wrap(_u_alma("data/ckpt/wm_ssm_none_s0.pt"), {}),
    "alma_direct_s0": lambda: wrap(_u_alma("data/ckpt/wm_ssm_direct_s0.pt"), {}),
}

LOAD_ERRORS = {}
for _f in sorted(glob.glob(os.path.join(HERE, "wm_*.py"))):
    _spec = importlib.util.spec_from_file_location("gas_" + os.path.basename(_f)[:-3], _f)
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "register"):
            _mod.register(REGISTRY)
    except Exception as _e:
        print(f"gas/models.py: skipping {_f}: {type(_e).__name__}: {_e}", file=sys.stderr)
        LOAD_ERRORS[_f] = f"{type(_e).__name__}: {_e}"


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
