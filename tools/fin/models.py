"""
Alma-Fin model registry (docs/fin_prereg.md section 5). Interface: tools/fin/README.md.

REGISTRY[name] = factory -> object with .kind ('forecaster' | 'agent'), .predict(ctx) or .act(ctx), optional
.observe(rec) (online), optional .meta (dict, recorded). Factories may fit on DEV through harness.train_view();
predict/act/observe run under the data-access guard and must use only their arguments.

Architect modules tools/fin/wm_*.py and tools/fin/agents/*.py are imported automatically if they define
register(REGISTRY); an import error is recorded in LOAD_ERRORS and aborts strict TEST mode.
"""
import os
import sys
import glob
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch  # noqa: F401  imported BEFORE lightgbm on purpose: lightgbm-first makes later torch ops segfault on this
#              Mac, and torch-first is safe with LightGBM n_jobs=1 (prereg A3; red-team RT-1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import harness as H     # noqa: E402
import metrics as MT    # noqa: E402

QS = H.QS
ZQ = norm.ppf(QS)
VOL_TARGET = {"crypto": 0.40 / np.sqrt(365), "etf": 0.10 / np.sqrt(252)}   # daily vol target of vol-scaled rules
FM_CONTEXT = 512


def _nanmean(x, axis=0):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(x, axis)


def ref_of(ctx):
    return _nanmean(ctx["daily"]["logrv"][-H.LOOKBACK:], 0)


def gauss_q(mu, sd):
    return mu[:, None] + np.asarray(sd)[..., None] * ZQ[None, :] if np.ndim(sd) else mu[:, None] + sd * ZQ[None, :]


# --------------------------------------------------------------------------- DEV training view
def train_view(market="crypto", group="train"):
    """Panel restricted (by assertion) to decision days <= 2025-06-30 and, for crypto, to the training coins."""
    P = H.Panel(market)
    end = P.index(H.DT.TRAIN_END) + (1 if np.datetime64(H.DT.TRAIN_END, "ns") in P.days else 0)
    keep = [s for s in P.symbols if group is None or P.group[s] == group or market == "etf"]
    js = [P.symbols.index(s) for s in keep]
    for f in P.daily:
        P.daily[f] = P.daily[f][:end][:, js]
    if P.hourly is not None:
        for f in P.hourly:
            P.hourly[f] = P.hourly[f][:24 * end][:, js]
    P.days, P.symbols = P.days[:end], keep
    P.last_bar, P.alive, P.cost = P.last_bar[:end][:, js], P.alive[:end][:, js], P.cost[js]
    assert P.days.max() <= np.datetime64(H.DT.TRAIN_END, "ns")
    return P


def har_design(lr, extra=None):
    """lr [n, m] log RV -> X [n, m, k] with features known BEFORE day t (rows t use days <= t-1)."""
    n, m = lr.shape
    X = np.full((n, m, 4), np.nan)
    X[:, :, 0] = 1.0
    for t in range(22, n):
        X[t, :, 1] = lr[t - 1]
        X[t, :, 2] = _nanmean(lr[t - 5:t], 0)
        X[t, :, 3] = _nanmean(lr[t - 22:t], 0)
    if extra is not None:
        X = np.concatenate([X, extra], 2)
    return X


def harx_extra(daily):
    n, m = daily["logrv"].shape
    E = np.full((n, m, 3), np.nan)
    E[1:, :, 0] = np.abs(daily["funding"][:-1]) * 1e3 if "funding" in daily else 0.0
    E[1:, :, 1] = (daily["taker_ratio"][:-1] - 0.5) * 10 if "taker_ratio" in daily else 0.0
    E[1:, :, 2] = np.minimum(daily["ret"][:-1], 0) * 10
    return np.nan_to_num(E, nan=0.0)


def ols(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(-1)
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res = y[ok] - X[ok] @ b
    return b, res


# --------------------------------------------------------------------------- forecasters
class Forecaster:
    kind = "forecaster"
    meta = {}


def majority():
    P = train_view()
    ys = [P.labels(i)["y_vol"] for i in range(30, len(P.days))]
    p = float(np.nanmean(np.concatenate(ys)))
    f = Forecaster()
    f.predict = lambda ctx: {"p_up": np.full(len(ctx["symbols"]), p)}
    f.meta = {"dev_up_fraction": p}
    return f


def rv_persist():
    P = train_view()
    d = np.diff(P.daily["logrv"], axis=0).ravel()
    eq = np.nanquantile(d, QS)
    f = Forecaster()

    def predict(ctx):
        last = ctx["daily"]["logrv"][-1]
        return {"q": last[:, None] + eq[None, :]}
    f.predict = predict
    f.meta = {"residual_quantiles": "DEV pooled daily changes of log RV (training coins)"}
    return f


def dow_onehot(days, k=7):
    """Weekday one-hot of the TARGET (decision) days [n] -> [n, k] (Mon = column 0). Calendar facts are known in advance."""
    wd = pd.DatetimeIndex(np.asarray(days, dtype="datetime64[ns]")).weekday.to_numpy()
    return np.eye(7)[wd][:, 7 - k:] if k < 7 else np.eye(7)[wd]


def har(extra=False, rolling=False, dow=False):
    """HAR-RV on log RV. pooled: one fit on DEV training coins (fixed). rolling: per-asset refit every day on the
    asset's own trailing 365 context days (the per-asset / rolling re-estimation baseline). dow: + 6 weekday dummies
    of the target day (Tue..Sun; Mon = base), prereg A5."""
    b0 = sd0 = None
    if not rolling:
        P = train_view()
        ex = harx_extra(P.daily) if extra else None
        if dow:
            D6 = np.repeat(dow_onehot(P.days, 6)[:, None, :], P.daily["logrv"].shape[1], 1)
            ex = D6 if ex is None else np.concatenate([ex, D6], 2)
        X = har_design(P.daily["logrv"], ex)
        b0, res = ols(X.reshape(-1, X.shape[-1]), P.daily["logrv"].ravel())
        sd0 = float(res.std())
    f = Forecaster()

    def predict(ctx):
        lr = ctx["daily"]["logrv"]
        n, m = lr.shape
        X = har_design(lr, harx_extra(ctx["daily"]) if extra else None) if rolling else None
        k = 4 + (3 if extra else 0) + (6 if dow else 0)
        x_next = np.empty((m, k))
        x_next[:, 0] = 1
        x_next[:, 1] = lr[-1]
        x_next[:, 2] = _nanmean(lr[-5:], 0)
        x_next[:, 3] = _nanmean(lr[-22:], 0)
        if extra:
            x_next[:, 4:] = np.nan_to_num(np.stack([np.abs(ctx["daily"]["funding"][-1]) * 1e3,
                                                    (ctx["daily"]["taker_ratio"][-1] - 0.5) * 10,
                                                    np.minimum(ctx["daily"]["ret"][-1], 0) * 10], 1))
        if dow:
            x_next[:, k - 6:] = dow_onehot([ctx["day"]], 6)[0][None, :]
        if not rolling:
            mu = x_next @ b0
            return {"q": gauss_q(mu, sd0)}
        mu, sd = np.full(m, np.nan), np.full(m, np.nan)
        for j in range(m):
            Xj, yj = X[-365:, j], lr[-365:, j]
            ok = np.isfinite(yj) & np.isfinite(Xj).all(-1)
            if ok.sum() < 60:
                continue
            b, res = ols(Xj, yj)
            mu[j], sd[j] = x_next[j] @ b, res.std()
        return {"q": mu[:, None] + sd[:, None] * ZQ[None, :]}
    f.predict = predict
    f.meta = {"extra": extra, "rolling": rolling, "dow": dow, "beta": None if b0 is None else b0.tolist(), "resid_sd": sd0}
    return f


def us_holiday(days):
    """1 if the day is a US federal holiday (observed), from pandas' rule-based USFederalHolidayCalendar (fixed rules
    published in advance, incl. Juneteenth from 2021; no data). Calendar facts are known at decision time."""
    global _HOL
    if _HOL is None:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        _HOL = USFederalHolidayCalendar().holidays(start="2015-01-01", end="2027-12-31")
    d = pd.DatetimeIndex(np.asarray(days, dtype="datetime64[ns]"))
    return d.isin(_HOL).astype(float)


_HOL = None


def har_intra_dow():
    """HAR + reference + weekday + US-holiday + intraday recency (prereg A5; red-team RT-15 spec): pooled OLS on the
    DEV training coins, target log RV(D), regressors known at D 00:00: 1, log RV(D-1), mean 5d, mean 22d,
    ref = mean log RV(D-7..D-1), 6 weekday dummies (Tue..Sun) of D, US federal holiday flag of D, log of the summed
    5-min RV of hours 18-24 UTC of D-1, log of the 5-min RV of hour 23-24 UTC of D-1. Gaussian residual quantiles.
    Crypto only (needs hourly 5-min RV)."""
    P = train_view()
    lr = P.daily["logrv"]
    n, m = lr.shape
    hr = P.hourly["rv5"].astype(np.float64)

    def rows(t_idx, lr_, hr_, day_list):
        X = np.full((len(t_idx), m_ := lr_.shape[1], 15), np.nan)
        for r, t in enumerate(t_idx):
            X[r, :, 0] = 1.0
            X[r, :, 1] = lr_[t - 1]
            X[r, :, 2] = _nanmean(lr_[t - 5:t], 0)
            X[r, :, 3] = _nanmean(lr_[t - 22:t], 0)
            X[r, :, 4] = lr_[t - 7:t].mean(0)
            X[r, :, 5:11] = dow_onehot([day_list[r]], 6)[0][None, :]
            X[r, :, 11] = us_holiday([day_list[r]])[0]
            h6 = hr_[24 * t - 6:24 * t]
            h1 = hr_[24 * t - 1]
            X[r, :, 12] = np.where(np.isfinite(h6).all(0), np.log(np.maximum(np.nansum(h6, 0), 1e-12)), np.nan)
            X[r, :, 13] = np.where(np.isfinite(h1), np.log(np.maximum(h1, 1e-12)), np.nan)
        return X[:, :, :14]
    ts = np.arange(30, n)
    X = rows(ts, lr, hr, P.days[ts])
    Y = lr[ts]
    b0, res = ols(X.reshape(-1, X.shape[-1]), Y.ravel())
    sd0 = float(res.std())
    f = Forecaster()

    def predict(ctx):
        lr_ = ctx["daily"]["logrv"]
        hr_ = ctx["hourly"]["rv5"].astype(np.float64)
        L = lr_.shape[0]
        assert hr_.shape[0] >= 24 and ctx["hours"][-1] + np.timedelta64(1, "h") == np.datetime64(ctx["decision_time"], "ns")
        # hourly context ends at D 00:00, daily at D-1: align both so that index L is the target day
        hpad = np.full((24 * L, hr_.shape[1]), np.nan)
        hpad[-hr_.shape[0]:] = hr_[-24 * L:]
        x = rows([L], np.concatenate([lr_, np.full((1, lr_.shape[1]), np.nan)]),
                 np.concatenate([hpad, np.full((24, hr_.shape[1]), np.nan)]), [ctx["day"]])[0]
        mu = x @ b0
        return {"q": gauss_q(mu, sd0)}
    f.predict = predict
    f.meta = {"beta": b0.tolist(), "resid_sd": sd0, "n_params": int(len(b0)),
              "regressors": ["1", "lrv_d1", "mean5", "mean22", "ref7", "dow_tue..sun(6)", "us_holiday",
                             "log_rv_18_24h_d1", "log_rv_23_24h_d1"]}
    return f


def garch():
    """GARCH(1,1) Gaussian per asset, refit daily on the trailing 365 daily close-to-close returns of the context;
    log RV median = log sigma^2_next + mean(logRV - log sigma^2) over the trailing 365 days (calibration in context)."""
    f = Forecaster()

    def predict(ctx):
        r = ctx["daily"]["ret"][-365:]
        lr = ctx["daily"]["logrv"][-365:]
        m = r.shape[1]
        mu, sd = np.full(m, np.nan), np.full(m, np.nan)
        for j in range(m):
            ok = np.isfinite(r[:, j])
            if ok.sum() < 120:
                continue
            rj = r[ok, j] - r[ok, j].mean()
            th = MT.garch_fit(rj)
            s2 = MT.garch_filter(rj, th, rj.var())
            gap = lr[ok, j] - np.log(np.maximum(s2[:-1], 1e-12))
            gap = gap[np.isfinite(gap)]
            mu[j] = np.log(max(s2[-1], 1e-12)) + gap.mean()
            sd[j] = gap.std()
        return {"q": mu[:, None] + sd[:, None] * ZQ[None, :]}
    f.predict = predict
    f.meta = {"window": 365, "calibration": "in-context mean log gap"}
    return f


def lgb_features(daily, t=None):
    """Features for day t (rows <= t-1) from a daily dict; t=None -> next day after the arrays."""
    lr, ret = daily["logrv"], daily["ret"]
    n = lr.shape[0]
    end = n if t is None else t
    f = [lr[end - 1], _nanmean(lr[end - 7:end], 0), _nanmean(lr[end - 30:end], 0),
         lr[end - 1] - _nanmean(lr[end - 7:end], 0), ret[end - 1], np.nansum(ret[end - 7:end], 0),
         np.nansum(ret[end - 30:end], 0), np.log(np.maximum(daily["high"][end - 1], 1e-12) /
                                                 np.maximum(daily["low"][end - 1], 1e-12))]
    if "taker_ratio" in daily:
        f += [daily["taker_ratio"][end - 1] - 0.5, daily["funding"][end - 1] * 1e3,
              np.log(np.maximum(daily["quote_volume"][end - 1], 1)) -
              _nanmean(np.log(np.maximum(daily["quote_volume"][end - 30:end], 1)), 0)]
    else:
        f += [np.full(lr.shape[1], np.nan)] * 3
    return np.stack(f, 1)


def lgbm(target="vol", dow=False):
    """dow: + weekday one-hot (7) of the target day (prereg A5)."""
    import lightgbm as lgb
    P = train_view()
    Xs, ys, ms = [], [], []
    for i in range(40, len(P.days)):
        lab = P.labels(i)
        x = lgb_features(P.daily, i)
        if dow:
            x = np.concatenate([x, np.repeat(dow_onehot([P.days[i]]), x.shape[0], 0)], 1)
        Xs.append(x)
        ys.append(lab["y_vol"] if target == "vol" else lab["y_ret"])
        ms.append(lab["logrv"] - lab["ref"])
    X, y, mres = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ms)
    ok = np.isfinite(y)
    prm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1, deterministic=True)
    clf = lgb.LGBMClassifier(**prm).fit(X[ok], y[ok].astype(int))
    reg = None
    if target == "vol":
        ok2 = np.isfinite(mres) & np.isfinite(X).any(1)
        reg = lgb.LGBMRegressor(**prm).fit(X[ok2], mres[ok2])
        sd = float(np.nanstd(mres[ok2] - reg.predict(X[ok2])))
    f = Forecaster()

    def predict(ctx):
        x = lgb_features(ctx["daily"])
        if dow:
            x = np.concatenate([x, np.repeat(dow_onehot([ctx["day"]]), x.shape[0], 0)], 1)
        p = clf.predict_proba(x)[:, 1]
        if target == "vol":
            mu = ref_of(ctx) + reg.predict(x)
            return {"p_up": p, "q": mu[:, None] + sd * ZQ[None, :]}
        return {"p_ret_up": p}
    f.predict = predict
    f.meta = {"params": prm, "n_train": int(ok.sum()), "target": target, "dow": dow}
    return f


def lgbm_asset(market="crypto"):
    """Per-asset LightGBM (the 'specific' twin of portability): one classifier + one regressor per asset, fitted on
    that asset's OWN DEV history (decision days <= 2025-06-30), same features/params as lgbm. ETFs: per ETF on its
    own DEV history (the only place an ETF series is ever fitted, and only for this per-asset baseline)."""
    import lightgbm as lgb
    P = train_view(market, group=None)
    prm = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8,
               subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1, deterministic=True)
    fits = {}
    for j, s in enumerate(P.symbols):
        Xs, ys, ms = [], [], []
        for i in range(40, len(P.days)):
            lab = P.labels(i)
            Xs.append(lgb_features(P.daily, i)[j])
            ys.append(lab["y_vol"][j])
            ms.append(lab["logrv"][j] - lab["ref"][j])
        X, y, mr = np.stack(Xs), np.array(ys), np.array(ms)
        ok, ok2 = np.isfinite(y), np.isfinite(mr)
        if ok.sum() < 200:
            continue
        clf = lgb.LGBMClassifier(**prm).fit(X[ok], y[ok].astype(int))
        reg = lgb.LGBMRegressor(**prm).fit(X[ok2], mr[ok2])
        fits[s] = (clf, reg, float(np.std(mr[ok2] - reg.predict(X[ok2]))))
    f = Forecaster()

    def predict(ctx):
        x = lgb_features(ctx["daily"])
        ref = ref_of(ctx)
        m = len(ctx["symbols"])
        p, q = np.full(m, np.nan), np.full((m, len(QS)), np.nan)
        for j, s in enumerate(ctx["symbols"]):
            if s in fits:
                clf, reg, sd = fits[s]
                p[j] = clf.predict_proba(x[j:j + 1])[0, 1]
                q[j] = ref[j] + reg.predict(x[j:j + 1])[0] + sd * ZQ
        return {"p_up": p, "q": q}
    f.predict = predict
    f.meta = {"params": prm, "assets_fitted": sorted(fits), "market": market}
    return f


# --------------------------------------------------------------------------- foundation models (zero-shot, log RV)
def _ctx_series(ctx):
    lr = ctx["daily"]["logrv"][-FM_CONTEXT:]
    out = []
    for j in range(lr.shape[1]):
        x = lr[:, j]
        ok = np.nonzero(np.isfinite(x))[0]
        if len(ok) == 0:
            out.append(np.zeros(8, np.float32))
            continue
        x = x[ok[0]:]
        idx = np.arange(len(x))
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good])               # interior gaps: linear interpolation (inputs only)
        out.append(x.astype(np.float32))
    return out


def _fm(load_fn, name, contaminated=None):
    st = {}
    f = Forecaster()

    def predict(ctx):
        if "m" not in st:
            st["m"] = load_fn()
        q = st["m"](_ctx_series(ctx))                       # [m, len(QS)]
        return {"q": q}
    f.predict = predict
    f.meta = {"hf": name, "context_days": FM_CONTEXT, "input": "daily log RV", "contamination": contaminated}
    return f


def _fill(have_levels, q):
    """q [m, L] at have_levels -> [m, len(QS)] (NaN where not native)."""
    out = np.full((q.shape[0], len(QS)), np.nan)
    for k, lv in enumerate(QS):
        hits = [i for i, h in enumerate(have_levels) if abs(h - lv) < 1e-6]
        if hits:
            out[:, k] = q[:, hits[0]]
    return out


def _chronos_loader(model_id, native):
    def load():
        import torch
        from chronos import BaseChronosPipeline
        torch.manual_seed(0)
        pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        ask = [q for q in QS if native is None or round(q, 4) in native]

        def run(series):
            qq, _ = pipe.predict_quantiles([torch.tensor(s) for s in series], prediction_length=1, quantile_levels=ask)
            if isinstance(qq, list):
                qq = torch.stack([t.reshape(-1, 1, len(ask))[0] for t in qq])
            return _fill(ask, qq.reshape(len(series), len(ask)).double().numpy())
        return run
    return load


NATIVE9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _tirex_loader():
    import torch
    from tirex import load_model
    m = load_model("NX-AI/TiRex", device="cpu")

    def run(series):
        q, _ = m.forecast(context=[torch.tensor(s) for s in series], prediction_length=1)
        return _fill(NATIVE9, np.asarray(q, np.float64).reshape(len(series), 9))
    return run


def _timestral_loader(model_id):
    def load():
        from timestral import TimestralForecaster
        m = TimestralForecaster.from_pretrained(model_id)

        def run(series):
            L = max(len(s) for s in series)
            x = np.full((len(series), L), np.nan, np.float32)
            for i, s in enumerate(series):
                x[i, L - len(s):] = s
            fc = m.forecast(x, prediction_length=1)
            lev = [round(float(v), 4) for v in fc.quantile_levels]
            return _fill(lev, np.asarray(fc.quantiles).reshape(len(series), len(lev)))
        return run
    return load


KRONOS_REPO = "/Users/srimanarayana/MIKO/kronos/repo"     # official github.com/shiyu-coder/Kronos @ 67b630e
KRONOS_SAMPLES, KRONOS_CONTEXT = 10, 256


def kronos_small():
    """Kronos-small on the coin's DAILY OHLCV K-lines (its native input), 1-step-ahead, 10 samples; predicted log
    Parkinson variance of each sampled bar, mapped to log RV by the in-context mean gap (log RV - log Parkinson) of
    the last 256 days; quantiles over samples. Contamination: pretrained on data up to June 2024 (saw DEV)."""
    st = {}
    f = Forecaster()

    def predict(ctx):
        import torch
        import pandas as pd
        if "m" not in st:
            sys.path.insert(0, KRONOS_REPO)
            from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps
            st.update(tok=KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval(),
                      m=Kronos.from_pretrained("NeoQuasar/Kronos-small").eval(),
                      ari=auto_regressive_inference, cts=calc_time_stamps)
        d = ctx["daily"]
        m = d["close"].shape[1]
        out = np.full((m, len(QS)), np.nan)
        vol = d["quote_volume"] if "quote_volume" in d else d["volume"]
        for j in range(m):
            o, h, l, c = [d[k][-KRONOS_CONTEXT:, j] for k in ("open", "high", "low", "close")]
            v = vol[-KRONOS_CONTEXT:, j]
            ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
            if ok.sum() < 60:
                continue
            first = np.nonzero(ok)[0][0]
            sl = slice(first, None)
            x = np.stack([o[sl], h[sl], l[sl], c[sl], v[sl], v[sl] * c[sl]], 1).astype(np.float32)
            x = pd.DataFrame(x).ffill().to_numpy(np.float32)
            L = len(x)
            torch.manual_seed(0)
            np.random.seed(0)
            mu_, sd_ = x.mean(0), x.std(0)
            xn = np.clip((x - mu_) / (sd_ + 1e-5), -5, 5)
            past = pd.Series(pd.DatetimeIndex(ctx["days"][-L:]))
            fut = pd.Series([ctx["day"]])
            xs = st["cts"](past).values.astype(np.float32)
            ys = st["cts"](fut).values.astype(np.float32)
            S = KRONOS_SAMPLES
            with torch.no_grad():
                pr = st["ari"](st["tok"], st["m"], torch.from_numpy(np.repeat(xn[None], S, 0)),
                               torch.from_numpy(np.repeat(xs[None], S, 0)), torch.from_numpy(np.repeat(ys[None], S, 0)),
                               KRONOS_CONTEXT, 1, 5, 1.0, 0, 0.9, 1, False)
            bar = pr[:, -1, :4] * (sd_[:4] + 1e-5) + mu_[:4]
            hi_, lo_ = np.maximum(bar[:, 1], bar[:, [0, 3]].max(1)), np.minimum(bar[:, 2], bar[:, [0, 3]].min(1))
            lpk = np.log(np.maximum((np.log(np.maximum(hi_, 1e-12) / np.maximum(lo_, 1e-12))) ** 2 / (4 * np.log(2)), 1e-12))
            hist_pk = (np.log(h[sl] / l[sl])) ** 2 / (4 * np.log(2))
            gap = d["logrv"][-KRONOS_CONTEXT:, j][sl] - np.log(np.maximum(hist_pk, 1e-12))
            gap = gap[np.isfinite(gap)]
            out[j] = np.quantile(lpk + (gap.mean() if len(gap) else 0.0), QS)
        return {"q": out}
    f.predict = predict
    f.meta = {"hf": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
              "code": KRONOS_REPO + " @ 67b630e", "samples": KRONOS_SAMPLES, "context_days": KRONOS_CONTEXT,
              "T": 1.0, "top_p": 0.9, "seed": 0, "input": "daily OHLCV (volume=quote volume for crypto)",
              "contamination": "pretrained on K-lines up to 2024-06 (incl. crypto): saw DEV, not VAL/TEST"}
    return f


# --------------------------------------------------------------------------- agents (trading baselines)
class Agent:
    kind = "agent"
    meta = {}


def _agent(fn, meta):
    a = Agent()
    a.act = fn
    a.meta = meta
    return a


def buy_hold():
    return _agent(lambda ctx: np.ones(len(ctx["symbols"])), {"rule": "long 1 in every asset"})


def tsmom(k):
    def act(ctx):
        r = np.nansum(ctx["daily"]["ret"][-k:], 0)
        return np.sign(r)
    return _agent(act, {"rule": f"sign of the past {k}-day log return"})


def meanrev():
    return _agent(lambda ctx: -np.sign(np.nan_to_num(ctx["daily"]["ret"][-1])),
                  {"rule": "minus sign of the previous day's return"})


def voltgt_tsmom(k=30):
    """sign(k-day return) x min(1, sigma* / sigma_HAR); sigma_HAR from the pooled DEV HAR median log RV
    (crypto: 5-min RV of a 24h day; ETF: Garman-Klass session variance)."""
    st = {"har": har()}                   # fitted here: act() runs under the data-access guard

    def act(ctx):
        mu = st["har"].predict(ctx)["q"][:, QS.index(0.5)]
        sig = np.sqrt(np.exp(mu))
        tgt = VOL_TARGET[ctx["market"]]
        return np.sign(np.nansum(ctx["daily"]["ret"][-k:], 0)) * np.minimum(1.0, tgt / np.maximum(sig, 1e-6))
    return _agent(act, {"rule": f"vol-targeted TSMOM {k}d, target daily vol {VOL_TARGET}", "vol_model": "har (pooled DEV)"})


def lgbm_policy():
    st = {"c": lgbm("ret"), "v": har()}

    def act(ctx):
        p = st["c"].predict(ctx)["p_ret_up"]
        sig = np.sqrt(np.exp(st["v"].predict(ctx)["q"][:, QS.index(0.5)]))
        return np.sign(p - 0.5) * np.minimum(1.0, VOL_TARGET[ctx["market"]] / np.maximum(sig, 1e-6))
    return _agent(act, {"rule": "sign(P_lgbm(ret up) - 0.5), vol-targeted with pooled HAR"})


REGISTRY = {
    "majority": majority,
    "rv_persist": rv_persist,
    "har": lambda: har(),
    "harx": lambda: har(extra=True),
    "har_roll": lambda: har(rolling=True),
    "garch": garch,
    "lgbm": lambda: lgbm("vol"),
    "har_dow": lambda: har(dow=True),
    "lgbm_dow": lambda: lgbm("vol", dow=True),
    "har_intra_dow": har_intra_dow,
    "lgbm_asset": lgbm_asset,
    "lgbm_asset_etf": lambda: lgbm_asset("etf"),
    "chronos2": lambda: _fm(_chronos_loader("amazon/chronos-2", None), "amazon/chronos-2",
                            "released 2025-10-17 (before TEST; corpus cut-off not public, may overlap VAL)"),
    "chronos_bolt_base": lambda: _fm(_chronos_loader("amazon/chronos-bolt-base", set(NATIVE9)),
                                     "amazon/chronos-bolt-base", "released 2024-11 (before VAL/TEST)"),
    "tirex": lambda: _fm(_tirex_loader, "NX-AI/TiRex", "released 2025-05 (before VAL/TEST)"),
    "timestral_5m": lambda: _fm(_timestral_loader("mistralai/Timestral-5M"), "mistralai/Timestral-5M",
                                "CONTAMINATED: released 2026-10-15; training data may include TEST"),
    "kronos_small": kronos_small,
    # agents
    "buy_hold": buy_hold,
    "tsmom_1": lambda: tsmom(1),
    "tsmom_7": lambda: tsmom(7),
    "tsmom_30": lambda: tsmom(30),
    "meanrev_1": meanrev,
    "voltgt_tsmom": voltgt_tsmom,
    "lgbm_policy": lgbm_policy,
}

LOAD_ERRORS = {}
for _f in sorted(glob.glob(os.path.join(HERE, "wm_*.py")) + glob.glob(os.path.join(HERE, "agents", "*.py"))):
    if os.path.basename(_f).startswith("_"):
        continue
    _spec = importlib.util.spec_from_file_location("fin_" + os.path.basename(_f)[:-3], _f)
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
        if hasattr(_mod, "register"):
            _mod.register(REGISTRY)
    except Exception as _e:
        print(f"fin/models.py: skipping {_f}: {type(_e).__name__}: {_e}", file=sys.stderr)
        LOAD_ERRORS[_f] = f"{type(_e).__name__}: {_e}"


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
