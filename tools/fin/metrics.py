"""Statistics for Alma-Fin (docs/fin_prereg.md section 6). Re-uses tools/power/metrics.py (-> tools/commodity/metrics.py:
Wilson, binomial, Pesaran-Timmermann, DM-HLN, McNemar, circular block bootstrap, Holm, pinball, GARCH) and adds
trading statistics: Sharpe, PSR, DSR, max drawdown, turnover, paired block-bootstrap Sharpe difference."""
import os
import importlib.util

import numpy as np
from scipy import stats

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "power", "metrics.py")
_spec = importlib.util.spec_from_file_location("power_metrics", _p)
_pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pm)

wilson, binom_p, pesaran_timmermann, dm_hln = _pm.wilson, _pm.binom_p, _pm.pesaran_timmermann, _pm.dm_hln
mcnemar_exact, holm, pinball = _pm.mcnemar_exact, _pm.holm, _pm.pinball
boot_idx, boot_mean_ci, boot_ratio_ci = _pm.boot_idx, _pm.boot_mean_ci, _pm.boot_ratio_ci
garch_fit, garch_filter = _pm._cm.garch_fit, _pm._cm.garch_filter
BLOCK_DAYS, B, SEED = 7, 2000, 0
EULER = 0.5772156649015329


def sharpe(r, periods):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def psr(r, sr_star=0.0):
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012) on per-period returns; sr_star per period."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / den))


def dsr(r, k_trials, var_sr_trials):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): PSR against the expected maximum per-period Sharpe of
    k_trials independent trials whose per-period Sharpe estimates have variance var_sr_trials."""
    if k_trials is None or k_trials < 1 or var_sr_trials is None or not np.isfinite(var_sr_trials):
        return np.nan
    if k_trials == 1:
        return psr(r, 0.0)
    z1 = stats.norm.ppf(1 - 1.0 / k_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (k_trials * np.e))
    sr0 = np.sqrt(var_sr_trials) * ((1 - EULER) * z1 + EULER * z2)
    return psr(r, sr0)


def max_drawdown(r):
    r = np.nan_to_num(np.asarray(r, float))
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((1 - eq / peak).max()) if len(eq) else np.nan


def trading_summary(r, turnover, periods, k_trials=None, var_sr_trials=None):
    """r: daily net portfolio returns; turnover: daily mean |delta position| per asset."""
    r = np.asarray(r, float)
    ok = np.isfinite(r)
    rr = r[ok]
    sr = sharpe(rr, periods)
    idx = boot_idx(len(rr)) if len(rr) > 2 else None
    lo = hi = np.nan
    if idx is not None:
        m, s = rr[idx].mean(1), rr[idx].std(1, ddof=1)
        bs = m / np.where(s > 0, s, np.nan) * np.sqrt(periods)
        lo, hi = float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))
    ann = float((np.prod(1 + rr) ** (periods / max(1, len(rr))) - 1)) if len(rr) else np.nan
    mdd = max_drawdown(rr)
    return {"n_days": int(len(rr)), "sharpe": sr, "sharpe_boot_lo": lo, "sharpe_boot_hi": hi,
            "psr0": psr(rr), "dsr": dsr(rr, k_trials, var_sr_trials), "ann_return": ann,
            "ann_vol": float(rr.std(ddof=1) * np.sqrt(periods)) if len(rr) > 1 else np.nan,
            "max_drawdown": mdd, "calmar": float(ann / mdd) if mdd and mdd > 0 else np.nan,
            "turnover": float(np.nanmean(turnover)) if len(turnover) else np.nan,
            "total_return": float(np.prod(1 + rr) - 1) if len(rr) else np.nan,
            "hit_days": float((rr > 0).mean()) if len(rr) else np.nan}


def sharpe_diff_test(ra, rb, periods):
    """Paired circular block bootstrap (7-day blocks, 2000 draws, seed 0) of SR(a) - SR(b) on common days.
    Returns (diff, lo, hi, two-sided p = 2 * min(P(diff* <= 0), P(diff* >= 0)))."""
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    ok = np.isfinite(ra) & np.isfinite(rb)
    ra, rb = ra[ok], rb[ok]
    if len(ra) < 10:
        return (np.nan,) * 4
    idx = boot_idx(len(ra))

    def sr(x):
        s = x.std(1, ddof=1)
        return x.mean(1) / np.where(s > 0, s, np.nan) * np.sqrt(periods)
    d = sr(ra[idx]) - sr(rb[idx])
    d0 = sharpe(ra, periods) - sharpe(rb, periods)
    p = float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
    return float(d0), float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975)), p


def wilson_eff(k, n, n_eff):
    """Wilson CI of k/n using the effective sample size n_eff (design effect for cross-sectional correlation)."""
    if n == 0:
        return (np.nan, np.nan)
    return wilson(k / n * n_eff, n_eff)
