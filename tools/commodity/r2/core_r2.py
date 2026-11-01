"""Round-2 shared machinery (docs/commodity_r2_prereg.md is binding).

Conventions (identical for every model):
  * Each series lives on its own trading calendar (FRED missing days dropped, never forward-filled).
  * Origin DATES of a split = the intersection calendar of the five POOL5 products with date in [start, end].
    For series c the model sees its observations dated strictly before the origin date; t_c = index of the last
    of them. Horizon h = the h-th future observation of that series (own calendar).
  * A (series, origin, h) triple is scored only if t_c + h exists in the split's data (so nothing after the split
    cap is ever scored).
  * Every forecast is an array [11, H] of quantiles QL = (0.025, 0.1, ..., 0.9, 0.975) of the log change
    lp[t+k] - lp[t], k = 1..H (H = 10). Models without the outer levels put NaN there.
Targets:
  vol h (PRIMARY, h in {5, 10}, pooled over POOL5): past = RMS of the h daily log returns ending at t,
       fut = RMS of the h returns t+1..t+h, y = 1[fut > past], ties excluded. Quantile-model rule (all quantile
       models, incl. world models, twins, foundation models, Gaussian RW baselines): sigma_hat = mean_{k<=h}
       (q.9_k - q.1_k) / (2 * 1.28155 * sqrt(k)); predict 1 iff sigma_hat > past.
  crps h (SECONDARY, h in {5, 10}, DHOILNYH and DJFUELUSGULF): QS = mean pinball over the 9 levels 0.1..0.9 of
       the log change at h; coverage of [q.1, q.9] and [q.025, q.975].
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import data_r2 as D          # noqa: E402
import metrics as MT         # noqa: E402  (round-1 statistics, reused unchanged)

QL = np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975])
I9 = list(range(1, 10))
Q9 = QL[I9]
ZL = stats.norm.ppf(QL)
I10, I90, ILO, IHI = 1, 9, 0, 10
H = 10
HS = (5, 10)
ZQ = 1.2815516
OUT = os.path.join(D.ROOT, "runs", "commodity", "r2")


# ------------------------------------------------------------------------------------------------ extras
def cal_feats(dates):
    """Calendar features known ex ante for the return ending on each date: Wednesday (EIA weekly petroleum status
    report day), sin/cos of day of year (seasonal vol). [T, 3]"""
    d = pd.DatetimeIndex(dates)
    doy = d.dayofyear.values / 365.25 * 2 * np.pi
    return np.stack([(d.dayofweek.values == 2).astype(float), np.sin(doy), np.cos(doy)], 1)


def future_cal(last_date, Hh=H):
    """Calendar features of the next Hh business days after last_date (weekday calendar; holidays are not
    known to the model, a disclosed approximation). [Hh, 3]"""
    fd = np.busday_offset(np.datetime64(pd.Timestamp(last_date).date()), np.arange(1, Hh + 1), roll="forward")
    return cal_feats(fd.astype("datetime64[ns]"))


def exo_feats(dates, X):
    """Lagged exogenous inputs for rows on `dates`: for each of WTI, Brent, Henry Hub, the latest daily log return
    on that series' own calendar dated STRICTLY BEFORE the row's date (lag >= 1 day; as-of join, no value fill;
    NaN/mask 0 if the latest such return is more than 5 calendar days old), scaled by 1/0.03 and clipped to +-10,
    plus its mask. [T, 6]"""
    out = []
    q = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    for s in D.EXO:
        d, lp = X[s]
        r = pd.DataFrame({"date": pd.DatetimeIndex(d[1:]), "r": np.diff(lp)})
        m = pd.merge_asof(q, r, on="date", allow_exact_matches=False, tolerance=pd.Timedelta(days=5))
        v = m["r"].values
        ok = np.isfinite(v)
        out += [np.clip(np.where(ok, v, 0) / 0.03, -10, 10), ok.astype(float)]
    return np.stack(out, 1)


class Series:
    """One product: dates, log prices, extras [T, 9] (cal 3 + exo 6), aligned with the RETURN ending at row t."""

    def __init__(self, sid, dates, lp, X=None):
        self.sid, self.dates, self.lp = sid, dates, lp
        ex = [cal_feats(dates)]
        ex.append(exo_feats(dates, X) if X is not None else np.zeros((len(dates), 6)))
        self.ex = np.concatenate(ex, 1).astype(np.float32)

    def train_origins(self, cut, L, Hh=H):
        c = int(np.searchsorted(self.dates, np.datetime64(cut)))
        o = np.arange(L, c - Hh)
        assert len(o) and self.dates[o[-1] + Hh] < np.datetime64(cut)
        return o


def load(split, series=None, exo=True):
    S = D.load_all(split, series or D.POOL5)
    X = D.load_exo(split) if exo else None
    return {s: Series(s, d, lp, X) for s, (d, lp) in S.items()}


def origin_dates(S, split):
    start, end = (D.BT_START, D.DEV_END) if split == "dev" else (D.VAL_START, D.VAL_END)
    cal = None
    for s in D.POOL5:
        d = set(S[s].dates[(S[s].dates >= np.datetime64(start)) & (S[s].dates <= np.datetime64(end))])
        cal = d if cal is None else cal & d
    return np.array(sorted(cal))


def t_index(ser, odates):
    """index of the last observation strictly before each origin date."""
    return np.searchsorted(ser.dates, odates) - 1


# ------------------------------------------------------------------------------------------------ baselines
def _gjr_nll(th, r):
    w, a, g, b, nu = th
    if w <= 0 or a < 0 or g < 0 or b < 0 or a + g / 2 + b >= 0.999 or nu <= 2.1 or nu > 60:
        return 1e12
    s2 = np.empty(len(r))
    s2[0] = r.var()
    for t in range(1, len(r)):
        s2[t] = w + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
    sc2 = s2 * (nu - 2) / nu
    ll = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
          - 0.5 * np.log(sc2) - (nu + 1) / 2 * np.log1p(r * r / (nu * sc2)))
    return -ll.sum()


def fit_params(S):
    """Per series, from returns dated before 2015-01-01 (TRAIN): GARCH(1,1)-Gaussian, GJR-GARCH(1,1)-t, HAR and
    log-HAR OLS coefficients per h (own and pooled). Cached in runs/commodity/r2/baseline_params.json."""
    path = os.path.join(OUT, "baseline_params.json")
    P = json.load(open(path)) if os.path.exists(path) else {}
    changed = False
    for s, ser in S.items():
        if s in P:
            continue
        r = np.diff(ser.lp)
        rr = r[ser.dates[1:] < np.datetime64(D.TRAIN_CUT)][-5000:]
        g = MT.garch_fit(rr)
        v0 = rr.var()
        res = optimize.minimize(_gjr_nll, [v0 * 0.05, 0.05, 0.05, 0.88, 6.0], args=(rr,), method="Nelder-Mead",
                                options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-5})
        P[s] = {"garch": list(g), "gjr": [float(x) for x in res.x]}
        changed = True
    for h in HS:
        key = f"har{h}"
        if all(key in P[s] for s in S) and f"pooled_har{h}" in P:
            continue
        Xs, Ys = [], []
        for s, ser in S.items():
            X, Y = har_xy(ser, h, train=True)
            P[s][key] = np.linalg.lstsq(X, Y, rcond=None)[0].tolist()
            P[s]["log" + key] = np.linalg.lstsq(_logx(X), np.log(Y + 1e-10), rcond=None)[0].tolist()
            Xs.append(X)
            Ys.append(Y)
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        P[f"pooled_har{h}"] = np.linalg.lstsq(X, Y, rcond=None)[0].tolist()
        P[f"pooled_loghar{h}"] = np.linalg.lstsq(_logx(X), np.log(Y + 1e-10), rcond=None)[0].tolist()
        changed = True
    if changed:
        os.makedirs(OUT, exist_ok=True)
        json.dump(P, open(path, "w"), indent=1)
    return P


def _logx(X):
    return np.concatenate([X[:, :1], np.log(X[:, 1:] + 1e-10)], 1)


def har_regressors(r2, t):
    """r2[j] = squared return ending at row j+1. At origin t (last row t): RV_d = r2[t-1], RV_w = mean of the last
    5, RV_m = mean of the last 22."""
    cs = np.concatenate([[0], np.cumsum(r2)])
    d = r2[t - 1]
    w = (cs[t] - cs[t - 5]) / 5
    m = (cs[t] - cs[t - 22]) / 22
    return np.stack([np.ones_like(d), d, w, m], 1)


def har_xy(ser, h, train):
    r2 = np.diff(ser.lp) ** 2
    cs = np.concatenate([[0], np.cumsum(r2)])
    c = int(np.searchsorted(ser.dates, np.datetime64(D.TRAIN_CUT)))
    t = np.arange(25, c - h)
    X = har_regressors(r2, t)
    Y = (cs[t + h] - cs[t]) / h
    return X, Y


def gauss_q(sig_cum):
    """[n, H] cumulative sd -> [n, 11, H] Gaussian quantiles."""
    return ZL[None, :, None] * sig_cum[:, None, :]


def baselines(ser, t, P):
    """Baseline forecasts at origins t of one series. Returns (qdict: name -> [n, 11, H] quantiles,
    vdict: h -> name -> predicted vol class [n] for the non-quantile vol baselines)."""
    r = np.diff(ser.lp)
    k = np.arange(1, H + 1)
    q = {}
    sig = np.array([np.sqrt(np.mean(r[i - 250:i] ** 2)) for i in t])
    q["rw_gauss"] = gauss_q(sig[:, None] * np.sqrt(k)[None])
    s2 = np.empty(len(r) + 1)
    s2[0] = np.mean(r[:250] ** 2)
    for j in range(len(r)):
        s2[j + 1] = 0.94 * s2[j] + 0.06 * r[j] ** 2
    q["ewma"] = gauss_q(np.sqrt(s2[t])[:, None] * np.sqrt(k)[None])
    w, a, b = P[ser.sid]["garch"]
    g = MT.garch_filter(r, (w, a, b), np.var(r[:500]))
    lr = w / (1 - a - b)
    vk = lr + (g[t][:, None] - lr) * (a + b) ** (k - 1)[None]
    q["garch"] = gauss_q(np.sqrt(np.cumsum(vk, 1)))
    w, a, gm, b, nu = P[ser.sid]["gjr"]
    s = np.empty(len(r) + 1)
    s[0] = np.var(r[:500])
    for j in range(len(r)):
        s[j + 1] = w + (a + gm * (r[j] < 0)) * r[j] ** 2 + b * s[j]
    pers = a + gm / 2 + b
    lr = w / (1 - pers)
    vk = lr + (s[t][:, None] - lr) * pers ** (k - 1)[None]
    tq = stats.t.ppf(QL, nu) * np.sqrt((nu - 2) / nu)
    q["gjr_t"] = tq[None, :, None] * np.sqrt(np.cumsum(vk, 1))[:, None, :]
    v = {}
    r2 = r ** 2
    X = har_regressors(r2, t)
    for h in HS:
        past = past_rv(ser.lp, t, h) ** 2
        v[h] = {}
        for nm, coef, lg in (("har", P[ser.sid][f"har{h}"], 0), ("loghar", P[ser.sid][f"loghar{h}"], 1),
                             ("har_pooled", P[f"pooled_har{h}"], 0), ("loghar_pooled", P[f"pooled_loghar{h}"], 1)):
            pred = (_logx(X) if lg else X) @ np.asarray(coef)
            v[h][nm] = (pred > (np.log(past + 1e-10) if lg else past)).astype(int)
        # mean reversion: predict "down" iff past-h RV > trailing 1-year (252 obs) median of the h-day RV
        cs = np.concatenate([[0], np.cumsum(r2)])
        rv_h = np.full(len(r2) + 1, np.nan)
        rv_h[h:] = (cs[h:] - cs[:-h]) / h                     # rv_h[j] = mean of r2[j-h..j-1] (ends at row j)
        med = np.array([np.nanmedian(rv_h[i - 251:i + 1]) for i in t])
        v[h]["meanrev"] = (past <= med).astype(int)
        v[h]["majority"] = np.full(len(t), majority_vol(ser, h))
    return q, v


def majority_vol(ser, h):
    c = int(np.searchsorted(ser.dates, np.datetime64(D.TRAIN_CUT)))
    t = np.arange(h + 1, c - h)
    y, ok = vol_actual(ser.lp, t, h)
    return int(y[ok].mean() >= 0.5)


# ------------------------------------------------------------------------------------------------ targets
def past_rv(lp, t, h):
    r = np.diff(lp)
    return np.array([np.sqrt(np.mean(r[i - h:i] ** 2)) for i in t])


def fut_rv(lp, t, h):
    r = np.diff(lp)
    return np.array([np.sqrt(np.mean(r[i:i + h] ** 2)) for i in t])


def vol_actual(lp, t, h):
    p, f = past_rv(lp, t, h), fut_rv(lp, t, h)
    return (f > p).astype(int), f != p


def vol_class_from_q(q, lp, t, h):
    k = np.arange(1, h + 1)
    sh = ((q[:, I90, :h] - q[:, I10, :h]) / (2 * ZQ * np.sqrt(k))[None]).mean(1)
    return (sh > past_rv(lp, t, h)).astype(int)


def qs(q, lp, t, h):
    y = lp[t + h] - lp[t]
    return MT.pinball(y, q[:, I9, h - 1], Q9)


def coverage(q, lp, t, h):
    y = lp[t + h] - lp[t]
    c80 = (y >= q[:, I10, h - 1]) & (y <= q[:, I90, h - 1])
    if np.isnan(q[:, ILO, h - 1]).any():
        return float(c80.mean()), float("nan")
    c95 = (y >= q[:, ILO, h - 1]) & (y <= q[:, IHI, h - 1])
    return float(c80.mean()), float(c95.mean())


# ------------------------------------------------------------------------------------------------ tests
def test_pooled_vol(loss_a, loss_b, ok, h):
    """loss_* [n_dates, n_series] 0/1 losses (1 = wrong), ok mask. Date-clustered test: on origin dates whose
    position is divisible by 2h (non-overlapping past+future windows), average (a - b) over the scoreable series of
    that date, then a one-sided DM-HLN (lag 0) / paired t on those date means (H1: A has lower loss). Cross-series
    correlation is absorbed by averaging within a date. Returns dict."""
    n = loss_a.shape[0]
    st = 2 * h
    pos = np.arange(n)
    acc_a = 1 - loss_a[ok].mean()
    acc_b = 1 - loss_b[ok].mean()
    dd = np.where(ok, loss_a - loss_b, np.nan)
    dm = np.nanmean(dd, 1)
    sub = (pos % st == 0) & np.isfinite(dm)
    m, s, _ = MT.dm_hln(dm[sub], 1, lag=0)
    p1 = float(stats.t.cdf(s, sub.sum() - 1)) if np.isfinite(s) else (0.5 if m == 0 else float("nan"))
    full = np.isfinite(dm)
    lo, hi = MT.block_boot_ci(-dm[full], 2 * st)
    return {"acc_a": float(acc_a), "acc_b": float(acc_b), "diff": float(acc_a - acc_b), "p1": p1,
            "n_indep": int(sub.sum()), "n_pairs": int(ok.sum()), "ci_lo": lo, "ci_hi": hi}


def test_crps(qs_a, qs_b, ok, h):
    """One series. diff = mean(QS_b - QS_a)/mean(QS_b) (relative skill of A over B) on all ok origins; one-sided
    DM-HLN (lag 0) on the non-overlapping subsample (positions divisible by h)."""
    n = len(qs_a)
    pos = np.arange(n)
    sub = ok & (pos % h == 0)
    m, s, _ = MT.dm_hln(qs_a[sub] - qs_b[sub], 1, lag=0)
    p1 = float(stats.t.cdf(s, sub.sum() - 1)) if np.isfinite(s) else float("nan")
    lo, hi = MT.block_boot_ci((qs_b - qs_a)[ok] / qs_b[ok].mean(), 2 * h)
    return {"skill": float(1 - qs_a[ok].mean() / qs_b[ok].mean()), "p1": p1, "n_indep": int(sub.sum()),
            "n": int(ok.sum()), "ci_lo": lo, "ci_hi": hi}


def p_to_z(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(stats.norm.isf(p))
