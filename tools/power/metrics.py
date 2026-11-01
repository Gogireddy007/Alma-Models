"""Statistics for Alma-Power (docs/power_prereg.md section 6). Re-uses tools/commodity/metrics.py (Wilson, binomial,
Pesaran-Timmermann, DM-HLN, McNemar, circular block bootstrap, Holm, pinball) and adds day-block helpers."""
import os
import importlib.util

import numpy as np

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "commodity", "metrics.py")
_spec = importlib.util.spec_from_file_location("commodity_metrics", _p)
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)

wilson, binom_p, pesaran_timmermann, dm_hln = _cm.wilson, _cm.binom_p, _cm.pesaran_timmermann, _cm.dm_hln
mcnemar_exact, block_boot_ci, holm, pinball = _cm.mcnemar_exact, _cm.block_boot_ci, _cm.holm, _cm.pinball

BLOCK_DAYS, B, SEED = 7, 2000, 0


def boot_idx(n, block=BLOCK_DAYS, B=B, seed=SEED):
    """Circular block bootstrap index matrix [B, n] over days."""
    b = max(1, min(int(block), n))
    nb = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(B, nb))
    return ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(B, -1)[:, :n]


def boot_ratio_ci(num, den, block=BLOCK_DAYS):
    """CI of sum(num)/sum(den) with whole days resampled (num, den per day)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n < 2:
        return (np.nan, np.nan)
    idx = boot_idx(n, block)
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def boot_mean_ci(x, block=BLOCK_DAYS):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = boot_idx(len(x), block)
    m = x[idx].mean(1)
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))
