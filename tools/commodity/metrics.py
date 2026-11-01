"""Statistics for the commodity evaluation (docs/commodity_prereg.md section 6)."""
import numpy as np
from scipy import stats, optimize


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - half, c + half)


def binom_p(k, n):
    return float(stats.binomtest(int(k), int(n), 0.5).pvalue) if n else np.nan


def pesaran_timmermann(pred, actual):
    """PT (1992) test of predictive direction; pred/actual in {0,1}. Returns (stat, one-sided p)."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    n = len(pred)
    if n < 10:
        return np.nan, np.nan
    p_hat = np.mean(pred == actual)
    py, px = actual.mean(), pred.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    den = v_hat - v_star
    if den <= 0:
        return np.nan, np.nan            # degenerate (e.g. a constant forecast): PT undefined
    s = (p_hat - p_star) / np.sqrt(den)
    return float(s), float(1 - stats.norm.cdf(s))


def dm_hln(d, h, lag=None):
    """Diebold-Mariano on loss differential d (model minus competitor; negative = model better), Newey-West
    (Bartlett) HAC variance with lag h-1 (or `lag`), Harvey-Leybourne-Newbold small-sample correction for horizon h,
    t(n-1) reference. Returns (mean d, stat, two-sided p)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan
    m = d.mean()
    e = d - m
    L = max(0, h - 1) if lag is None else int(lag)
    var = e @ e / n
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return float(m), np.nan, (1.0 if m == 0 else np.nan)
    s = m / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    s *= corr
    return float(m), float(s), float(2 * stats.t.sf(abs(s), n - 1))


def mcnemar_exact(wrong_a, wrong_b):
    """Exact two-sided McNemar (binomial on discordant pairs). Returns (n01 = a wrong & b right, n10, p)."""
    wa, wb = np.asarray(wrong_a, bool), np.asarray(wrong_b, bool)
    n01, n10 = int((wa & ~wb).sum()), int((~wa & wb).sum())
    p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return n01, n10, p


def block_boot_ci(hits, block, B=2000, seed=0, alpha=0.05):
    """Circular block bootstrap percentile CI of the mean of a hit series (block length `block`)."""
    x = np.asarray(hits, float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    b = max(1, min(int(block), n))
    nb = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    xx = np.concatenate([x, x[:b]])
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B, -1)[:, :n]
    m = xx[idx].mean(1)
    return (float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)))


def holm(pvals):
    """Holm step-down adjusted p-values (NaN kept)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(ok)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def pinball(y, q, levels):
    """y [n], q [n, L], levels [L] -> mean over levels of the pinball loss, per row [n]."""
    lv = np.asarray(levels)[None, :]
    diff = y[:, None] - q
    return np.mean(np.maximum(lv * diff, (lv - 1) * diff), axis=1)


# --------------------------------------------------------------------------- GARCH(1,1), Gaussian, zero mean
def garch_fit(r):
    r = np.asarray(r, float)
    v0 = r.var()

    def nll(th):
        w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        s2 = np.empty_like(r)
        s2[0] = v0
        for t in range(1, len(r)):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + r * r / s2)
    res = optimize.minimize(nll, [v0 * 0.05, 0.08, 0.9], method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-6})
    return tuple(float(x) for x in res.x)


def garch_filter(r, th, s2_0):
    """Conditional variance s2[t] = var(r[t] | r[:t]) for t = 0..len(r) (last entry = one step beyond)."""
    w, a, b = th
    s2 = np.empty(len(r) + 1)
    s2[0] = s2_0
    for t in range(len(r)):
        s2[t + 1] = w + a * r[t] ** 2 + b * s2[t]
    return s2


def garch_mean_var(s2_next, th, k=20):
    w, a, b = th
    lr = w / (1 - a - b)
    ph = (a + b) ** np.arange(k)
    return lr + (s2_next - lr) * ph.mean()
