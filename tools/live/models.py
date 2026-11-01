"""
Alma-Live Gas model registry: rule baselines, fitted baselines (ridge, LightGBM single-threaded, DQN, PPO in torch),
zero-shot price models (Chronos-2, TiRex; offline HF cache only) and auto-imported architect models
(tools/live/wm_*.py exposing register(REGISTRY)). Interface: tools/live/README.md.

Every model object:
    name, meta (dict)
    predict(ctx) -> {"p_up": P(UNG close(d)->close(d+1) > 0), "mu": E[log return], optional "q": {level: value}}
    act(ctx)     -> {"UNG": w, "BOIL": w}  target weights in [-1, 1]
    observe(outcome)  (optional, continual) outcome = {"day", "labels": {...}, "available_at"}
    fit(X, y)    (optional, fitted baselines; the harness passes DEV-only features / labels)
Models never read files: paper.py / run_eval.py run predict/act/observe under common.no_data_access().
"""
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from store import FEATURES, storage_expectation, _ng_series  # noqa: E402,F401

REGISTRY = {}
LOAD_ERRORS = {}


def _sig(x):
    return {"UNG": float(np.sign(x)) if np.isfinite(x) else 0.0}


class Rule:
    """A rule baseline: signal(ctx) -> position in [-1, 1] on UNG; p_up = 0.5 + 0.1 * signal (direction only)."""

    def __init__(self, name, signal, meta=None):
        self.name, self.signal, self.meta = name, signal, meta or {"kind": "rule"}

    def predict(self, ctx):
        s = self.signal(ctx)
        s = 0.0 if not np.isfinite(s) else float(np.clip(s, -1, 1))
        return {"p_up": 0.5 + 0.1 * s, "mu": np.nan}

    def act(self, ctx):
        s = self.signal(ctx)
        return {"UNG": 0.0 if not np.isfinite(s) else float(np.clip(s, -1, 1))}


def _f(name):
    return lambda ctx: ctx["features"].get(name, np.nan)


def _sign(fn):
    return lambda ctx: float(np.sign(fn(ctx))) if np.isfinite(fn(ctx)) else 0.0


def _seasonal(ctx):
    # fixed a-priori seasonal rule (no fitting): long UNG into the withdrawal-season run-up (Sep-Nov), short in the
    # shoulder months when injections dominate (Mar-May), flat otherwise
    m = ctx["day"].month
    return 1.0 if m in (9, 10, 11) else (-1.0 if m in (3, 4, 5) else 0.0)


def _weather_rev(ctx):
    r = ctx["features"].get("wx_rev", np.nan)
    if not np.isfinite(r) or abs(r) < 1.0:
        return 0.0
    return float(np.sign(r))           # demand revised up (more HDD+CDD) -> long


def _storage_surprise(ctx):
    s = ctx["features"].get("stor_surprise", np.nan)
    if not np.isfinite(s) or abs(s) < 2.0:
        return 0.0
    return float(-np.sign(s))          # smaller build / larger draw than the naive 5-yr expectation -> long


REGISTRY.update({
    "flat": lambda: Rule("flat", lambda c: 0.0),
    "buy_hold": lambda: Rule("buy_hold", lambda c: 1.0),
    "short_ung_carry": lambda: Rule("short_ung_carry", lambda c: -1.0),
    "tsmom_5": lambda: Rule("tsmom_5", _sign(_f("ung_r5"))),
    "tsmom_20": lambda: Rule("tsmom_20", _sign(_f("ung_r20"))),
    "tsmom_60": lambda: Rule("tsmom_60", _sign(_f("ung_r60"))),
    "seasonal": lambda: Rule("seasonal", _seasonal),
    "weather_rev": lambda: Rule("weather_rev", _weather_rev),
    "storage_surprise": lambda: Rule("storage_surprise", _storage_surprise),
})


# --------------------------------------------------------------------------- fitted baselines (same features)
class _Fitted:
    kind = "fitted"

    def __init__(self, name):
        self.name, self.meta = name, {"kind": "fitted", "features": FEATURES}
        self.med = None

    def _x(self, F):
        x = np.array([[F.get(k, np.nan) for k in FEATURES]], float) if isinstance(F, dict) else np.asarray(F, float)
        return x

    def predict(self, ctx):
        mu = float(self._mu(self._x(ctx["features"]))[0])
        p = 1 / (1 + np.exp(-mu / max(self.sd, 1e-6) * 1.6))
        return {"p_up": float(p), "mu": mu}

    def act(self, ctx):
        mu = self.predict(ctx)["mu"]
        return {"UNG": float(np.sign(mu)) if abs(mu) > self.thr else 0.0}


class RidgeB(_Fitted):
    def __init__(self, alpha=10.0):
        super().__init__("ridge")
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.mu_, self.sd_ = Xi.mean(0), Xi.std(0) + 1e-9
        Z = (Xi - self.mu_) / self.sd_
        A = Z.T @ Z + self.alpha * len(Z) * 0.01 * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        self.b = y.mean()
        self.sd = float(y.std())
        self.thr = 0.0
        return self

    def _mu(self, X):
        Xi = np.where(np.isfinite(X), X, self.med)
        return ((Xi - self.mu_) / self.sd_) @ self.w + self.b


class LGBMB(_Fitted):
    def __init__(self):
        super().__init__("lgbm")

    def fit(self, X, y):
        import lightgbm as lgb
        X, y = np.asarray(X, float), np.asarray(y, float)
        ok = np.isfinite(y)
        self.m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                   n_jobs=1, random_state=0, verbose=-1, deterministic=True, force_row_wise=True)
        self.m.fit(X[ok], y[ok])
        self.sd = float(y[ok].std())
        self.thr = 0.0
        return self

    def _mu(self, X):
        return self.m.predict(X)


REGISTRY["ridge"] = RidgeB
REGISTRY["lgbm"] = LGBMB


# --------------------------------------------------------------------------- RL baselines (torch, tiny)
class _RLBase(_Fitted):
    """State = standardized features + previous position; actions {-1, 0, +1} on UNG; reward = a * r - cost * |a - a_prev|
    with cost = 10 bp per side. Trained on DEV only by the harness."""
    ACTIONS = np.array([-1.0, 0.0, 1.0])
    COST = 10e-4

    def _prep(self, X, fit=False):
        X = np.asarray(X, float)
        if fit:
            self.med = np.nan_to_num(np.nanmedian(X, 0))
            Xi = np.where(np.isfinite(X), X, self.med)
            self.mu_, self.sd_ = Xi.mean(0), Xi.std(0) + 1e-9
        Xi = np.where(np.isfinite(X), X, self.med)
        return np.clip((Xi - self.mu_) / self.sd_, -5, 5).astype(np.float32)

    def act(self, ctx):
        import torch
        z = self._prep(self._x(ctx["features"]))
        s = torch.tensor(np.concatenate([z, [[self.prev]]], 1), dtype=torch.float32)
        with torch.no_grad():
            a = int(self._greedy(s))
        self.prev = float(self.ACTIONS[a])
        return {"UNG": self.prev}

    def predict(self, ctx):
        return {"p_up": np.nan, "mu": np.nan}

    def observe(self, outcome):
        return 0


class DQNB(_RLBase):
    def __init__(self, epochs=30, seed=0):
        super().__init__("dqn")
        self.epochs, self.seed, self.prev = epochs, seed, 0.0

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ok = np.isfinite(np.asarray(y, float))
        Z = self._prep(np.asarray(X, float)[ok], fit=True)
        r = np.asarray(y, float)[ok].astype(np.float32)
        d = Z.shape[1] + 1
        self.q = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 3))
        opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        gamma = 0.9
        for ep in range(self.epochs):
            prev = rng.choice(self.ACTIONS, size=len(Z))
            s = torch.tensor(np.concatenate([Z, prev[:, None]], 1), dtype=torch.float32)
            # one-step targets for every action (state transition deterministic in features, position = action)
            rew = self.ACTIONS[None, :] * r[:, None] - self.COST * np.abs(self.ACTIONS[None, :] - prev[:, None])
            with torch.no_grad():
                nxt = np.concatenate([Z[1:], Z[-1:]], 0)
                vn = []
                for a in self.ACTIONS:
                    sn = torch.tensor(np.concatenate([nxt, np.full((len(Z), 1), a)], 1), dtype=torch.float32)
                    vn.append(self.q(sn).max(1).values.numpy())
                vn = np.stack(vn, 1)
            tgt = torch.tensor(rew * 100 + gamma * vn, dtype=torch.float32)
            for idx in np.array_split(rng.permutation(len(Z)), max(1, len(Z) // 256)):
                loss = ((self.q(s[idx]) - tgt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.q.eval()
        return self

    def _greedy(self, s):
        return self.q(s).argmax(1).item()


class PPOB(_RLBase):
    def __init__(self, iters=60, seed=0):
        super().__init__("ppo")
        self.iters, self.seed, self.prev = iters, seed, 0.0

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ok = np.isfinite(np.asarray(y, float))
        Z = self._prep(np.asarray(X, float)[ok], fit=True)
        r = np.asarray(y, float)[ok]
        d = Z.shape[1] + 1
        self.pi = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 3))
        self.v = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
        opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=3e-4)
        T = 128
        for it in range(self.iters):
            starts = rng.integers(0, max(1, len(Z) - T), size=8)
            S, A, R, LP = [], [], [], []
            for s0 in starts:
                prev = 0.0
                for t in range(s0, min(s0 + T, len(Z))):
                    st = np.concatenate([Z[t], [prev]]).astype(np.float32)
                    with torch.no_grad():
                        logits = self.pi(torch.tensor(st))
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                    act = self.ACTIONS[a.item()]
                    rew = (act * r[t] - self.COST * abs(act - prev)) * 100
                    S.append(st), A.append(a.item()), R.append(rew), LP.append(dist.log_prob(a).item())
                    prev = act
            S = torch.tensor(np.array(S))
            A = torch.tensor(A)
            Rt = torch.tensor(R, dtype=torch.float32)
            LP = torch.tensor(LP)
            for _ in range(4):
                adv = Rt - self.v(S).squeeze(1).detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                dist = torch.distributions.Categorical(logits=self.pi(S))
                ratio = torch.exp(dist.log_prob(A) - LP)
                l_pi = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean() - 0.01 * dist.entropy().mean()
                l_v = ((self.v(S).squeeze(1) - Rt) ** 2).mean()
                opt.zero_grad()
                (l_pi + 0.5 * l_v).backward()
                opt.step()
        return self

    def _greedy(self, s):
        return self.pi(s).argmax(1).item()


REGISTRY["dqn"] = DQNB
REGISTRY["ppo"] = PPOB


# --------------------------------------------------------------------------- zero-shot price models (offline)
class ZeroShot:
    """Chronos-2 / TiRex zero-shot on the UNG log close (ctx prices through d-1). Forecast horizon 2 steps (d, d+1):
    the decision return close(d)->close(d+1) is step 2 minus step 1. Loaded from the local HF cache only
    (HF_HUB_OFFLINE=1); unavailable -> raises at construction (recorded in LOAD_ERRORS)."""
    CTX = 512

    def __init__(self, which):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        torch.manual_seed(0)
        self.name, self.which = which, which
        if which == "chronos2":
            from chronos import BaseChronosPipeline
            self.m = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
        else:
            from tirex import load_model
            self.m = load_model("NX-AI/TiRex", device="cpu")
        self.meta = {"kind": "zero_shot", "hf": {"chronos2": "amazon/chronos-2", "tirex": "NX-AI/TiRex"}[which],
                     "contamination": "Chronos-2 (Oct 2025) / TiRex (May 2025) may include VAL-period data"}

    def predict(self, ctx):
        import torch
        u = ctx["px"]["UNG"]["close"].dropna()
        y = np.log(u.to_numpy()[-self.CTX:]).astype(np.float32)
        if self.which == "chronos2":
            q, mean = self.m.predict_quantiles([torch.tensor(y)], prediction_length=2, quantile_levels=[0.1, 0.5, 0.9])
            q = q[0] if isinstance(q, list) else q
            med = np.asarray(q).reshape(-1, 2, 3)[0, :, 1]
            lo, hi = np.asarray(q).reshape(-1, 2, 3)[0, 1, 0], np.asarray(q).reshape(-1, 2, 3)[0, 1, 2]
        else:
            q, mean = self.m.forecast(context=[torch.tensor(y)], prediction_length=2)
            qq = np.asarray(q, dtype=np.float64)[0]            # [2, 9]
            med, lo, hi = qq[:, 4], qq[1, 0], qq[1, 8]
        mu = float(med[1] - med[0])
        sd = max(float(hi - lo) / 2.56, 1e-6)
        from scipy.stats import norm
        return {"p_up": float(norm.cdf(mu / sd)), "mu": mu}

    def act(self, ctx):
        return {"UNG": float(np.sign(self.predict(ctx)["mu"]))}


REGISTRY["chronos2"] = lambda: ZeroShot("chronos2")
REGISTRY["tirex"] = lambda: ZeroShot("tirex")

RULES = ["buy_hold", "short_ung_carry", "tsmom_5", "tsmom_20", "tsmom_60", "seasonal", "weather_rev", "storage_surprise"]
FITTED = ["ridge", "lgbm", "dqn", "ppo"]
ZEROSHOT = ["chronos2", "tirex"]


def _autoload():
    for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm_*.py"))):
        mod = os.path.basename(p)[:-3]
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "register"):
                m.register(REGISTRY)
        except Exception as e:  # noqa: BLE001
            LOAD_ERRORS[mod] = repr(e)


_autoload()


def load(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; known: {sorted(REGISTRY)}; load errors: {LOAD_ERRORS}")
    return REGISTRY[name]()


# =========================================================================== forecast baselines (primary targets)
# predict(ctx) keys for the three pre-registered targets (docs/live_prereg.md):
#   "ung_q":      {level: quantile of the UNG log return close(d)->close(d+1)}
#   "rv5_q":      {level: quantile of log RV5, RV5 = sum of squared UNG close-to-close log returns d+1..d+5}
#   "stor_dev_q": {level: quantile of dev(W) = printed net change(W) - naive 5-yr expectation, for the NEXT
#                  unreleased report W = ctx['next_storage']['week']}
QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _ung_ret(ctx):
    u = ctx["px"].get("UNG")
    if u is None or not len(u):
        return pd.Series(dtype=float)
    return np.log(u["close"] / u["close"].shift(1)).dropna()


def past_storage_devs(ctx, years=5):
    """dev(W) for the released weeks of the last `years` years, from ctx's storage table (values known at t)."""
    st = ctx["storage"]
    if not len(st) or "l48.chg_printed" not in st:
        return pd.Series(dtype=float)
    wk = st.index[st.index >= st.index[-1] - pd.Timedelta(days=366 * years)]
    out = {}
    for w in wk:
        e = storage_expectation(st, w)
        if np.isfinite(e) and np.isfinite(st.at[w, "l48.chg_printed"]):
            out[w] = st.at[w, "l48.chg_printed"] - e
    return pd.Series(out, dtype=float)


def _past_log_rv5(r, n=500):
    r2 = (r ** 2).to_numpy()
    if len(r2) < 30:
        return np.array([]), np.array([])
    rv5 = np.array([r2[i + 1:i + 6].sum() for i in range(len(r2) - 5)])
    pred = np.array([5 * r2[max(0, i - 19):i + 1].mean() for i in range(len(r2) - 5)])
    ok = (rv5 > 0) & (pred > 0)
    return np.log(rv5[ok])[-n:], np.log(pred[ok])[-n:]


class FcHist:
    """Climatology: empirical quantiles of the last 500 UNG returns, of the last 500 log RV5, of the last 5 years
    of storage deviations (all computed inside ctx)."""
    name = "fc_hist"
    meta = {"kind": "forecast_baseline"}

    def predict(self, ctx):
        r = _ung_ret(ctx).iloc[-500:]
        out = {"ung_q": {q: float(np.quantile(r, q)) for q in QS} if len(r) > 50 else {}}
        lrv, _ = _past_log_rv5(_ung_ret(ctx))
        out["rv5_q"] = {q: float(np.quantile(lrv, q)) for q in QS} if len(lrv) > 50 else {}
        dv = past_storage_devs(ctx)
        out["stor_dev_q"] = {q: float(np.quantile(dv, q)) for q in QS} if len(dv) > 50 else {}
        out["p_up"] = float((r > 0).mean()) if len(r) else np.nan
        out["mu"] = 0.0
        return out

    def act(self, ctx):
        return {"UNG": 0.0}


class FcEwma:
    """Volatility-scaled: return ~ N(0, EWMA(0.94) variance); log RV5 = log(5 x mean r^2 of last 20 days) + empirical
    quantiles of its past errors; storage dev = 0.5 x last dev + empirical quantiles of past errors of that rule."""
    name = "fc_ewma"
    meta = {"kind": "forecast_baseline"}

    def predict(self, ctx):
        from scipy.stats import norm
        r = _ung_ret(ctx)
        out = {}
        if len(r) > 60:
            v = r.iloc[-250:].pow(2).ewm(alpha=0.06).mean().iloc[-1]
            out["ung_q"] = {q: float(norm.ppf(q) * np.sqrt(v)) for q in QS}
            lrv, lpred = _past_log_rv5(r)
            cur = np.log(5 * (r.iloc[-20:] ** 2).mean())
            e = lrv - lpred
            out["rv5_q"] = {q: float(cur + np.quantile(e, q)) for q in QS} if len(e) > 50 else {}
        dv = past_storage_devs(ctx)
        if len(dv) > 60:
            e = dv.iloc[1:].to_numpy() - 0.5 * dv.iloc[:-1].to_numpy()
            c = 0.5 * dv.iloc[-1]
            out["stor_dev_q"] = {q: float(c + np.quantile(e, q)) for q in QS}
        out["p_up"], out["mu"] = 0.5, 0.0
        return out

    def act(self, ctx):
        return {"UNG": 0.0}


REGISTRY["fc_hist"] = FcHist
REGISTRY["fc_ewma"] = FcEwma
FORECASTERS = ["fc_hist", "fc_ewma"]
