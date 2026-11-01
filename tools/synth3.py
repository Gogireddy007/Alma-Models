"""
synth3: a vectorised synthetic prior and a causal augmentation pipeline for real windows.

    generate(rng, n, length) -> y float32 [n, length] (NaN = missing / left pad), period int64 [n]
    augment(win, rng, period=None) -> win' (or (win', period') when period is given)

numpy only, CPU, no Python loop over rows or time steps (every recursion is either a cumsum or an
FFT convolution with a closed-form impulse response). Rows are served from per-family pools of
POOL_ROWS fresh rows (each row used once) to amortise numpy call overhead; see docs/synth3.md.

Generator families (one row = one family; mixture weights ~ Dirichlet(10) per call, so no family
dominates any batch). References are in docs/synth3.md.
    gp        KernelSynth-style GP: sums/products of RBF, rational-quadratic, periodic, locally
              periodic, linear kernels, sampled with random Fourier features (no O(n^3) Cholesky)
    season    multi-seasonal profiles with frequency-realistic period sets (5min 288/2016 ...
              quarterly 4), harmonic or table or peaky shapes, weekday factors, additive or
              multiplicative, plus trend
    arima     SARIMA(2,d,2)(1,D,0)_s: AR roots (real or complex), MA terms, seasonal AR / unit
              root, d in {0,1,2}, via exact linear filtering in the FFT domain
    ets       damped local-linear-trend ETS with evolving (stochastic-amplitude) seasonality,
              additive or multiplicative error
    regime    Markov regime switching (2-3 states) of mean / volatility / drift, OU smoothing
    trend     piecewise-linear trend with changepoints, logistic saturation (Prophet-like),
              exponential growth/decay, optional seasonality
    rw        random walk with drift, Student-t steps, integrated (smooth) RW, geometric RW
    vol       stochastic-volatility (GARCH-like clustering) returns or prices
    count     intermittent demand: zero-inflated Poisson / negative binomial with seasonal and
              trending rates
    pattern   sawtooth, square / duty-cycle, random steps, decaying spike trains, triangle, spikes
    causal    CauKer-like random causal DAG over 3-5 latent signals with lagged edges and random
              nonlinearities
Post-processing on every row (each with modest probability): heteroscedastic / heavy-tailed
observation noise, positivity shift, censoring at a floor, quantisation, lognormal scale and
offset, random missing values and missing runs, random left padding (shorter series).

Run `uv run --with matplotlib tools/synth3.py` for the speed benchmark, diagnostics and
docs/synth3_examples.png (without matplotlib only the PNG is skipped).
"""

import numpy as np

FAMILIES = ("gp", "season", "arima", "ets", "regime", "trend", "rw", "vol", "count", "pattern", "causal")

# sampling frequency -> (primary period, secondary period or 0, weight). The primary period is the
# fev-bench / GIFT seasonality convention used for the seasonal-naive reference.
FREQS = (("5min", 288, 2016, 0.10), ("10min", 144, 1008, 0.07), ("15min", 96, 672, 0.10),
         ("30min", 48, 336, 0.05), ("H", 24, 168, 0.20), ("D", 7, 365.25, 0.17),
         ("W", 52, 0, 0.11), ("M", 12, 0, 0.13), ("Q", 4, 0, 0.07))
_FP1 = np.array([f[1] for f in FREQS], dtype=np.int64)
_FP2 = np.array([f[2] for f in FREQS], dtype=np.float64)
_FW = np.array([f[3] for f in FREQS]); _FW = _FW / _FW.sum()
_SUBDAILY = np.array([f[0] in ("5min", "10min", "15min", "30min", "H") for f in FREQS])


# ---------------------------------------------------------------------------
# helpers (all [m, L] vectorised)
# ---------------------------------------------------------------------------

def _u(rng, lo, hi, m):
    return rng.uniform(lo, hi, (m, 1))


def _logu(rng, lo, hi, m):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), (m, 1)))


def _std(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


def _nfft(L):
    return 1 << int(np.ceil(np.log2(2 * L)))


def _filt(u, *hs):
    """Causal linear filter: u convolved with impulse responses hs (rows), exact (zero-padded FFT)."""
    L = u.shape[1]
    nf = _nfft(L)
    F = np.fft.rfft(u, nf)
    for h in hs:
        F = F * np.fft.rfft(h, nf)
    return np.fft.irfft(F, nf)[:, :L]


def _ar1(u, phi):
    """x_t = phi x_{t-1} + u_t (phi: [m, 1])."""
    k = np.arange(u.shape[1])[None, :]
    return _filt(u, phi ** k)


def _smooth_rw(rng, m, L, knots=None):
    """Slow random curve with unit-ish scale: random walk on a few knots, linearly interpolated."""
    K = knots or 6
    pts = np.cumsum(rng.normal(0, 1, (m, K + 1)), 1) / np.sqrt(K)
    x = np.linspace(0, K, L)
    i = np.minimum(x.astype(int), K - 1)
    w = x - i
    return pts[:, i] * (1 - w) + pts[:, i + 1] * w


_TDF = np.array([2.5, 3.0, 4.0, 6.0, 10.0])
_TPOOL = None


def _tpool():
    """Cached pool of unit-variance Student-t draws (5 dfs x 2^18, 10 MB): rows read random
    contiguous slices, which is much cheaper than standard_t per batch."""
    global _TPOOL
    if _TPOOL is None:
        g = np.random.default_rng(12345)
        _TPOOL = np.stack([np.clip(g.standard_t(d, 1 << 18), -30, 30) * np.sqrt((d - 2) / d) for d in _TDF])
    return _TPOOL


def _noise(rng, m, L, scale=None):
    """Observation noise: Gaussian (1/2) or Student-t with df in 2.5..10 (1/2), homoscedastic or
    heteroscedastic (slow log-volatility curve, 1/3). Unit-ish scale times `scale`."""
    e = rng.normal(0, 1, (m, L))
    tr = rng.random(m) < 0.5
    k = int(tr.sum())
    if k:
        pool = _tpool()
        st = rng.integers(0, pool.shape[1] - L, k)
        e[tr] = pool[rng.integers(0, len(_TDF), k)[:, None], st[:, None] + np.arange(L)] * rng.choice([-1.0, 1.0], (k, 1))
    het = rng.random(m) < 0.35
    k = int(het.sum())
    if k:
        e[het] *= np.exp(0.6 * _smooth_rw(rng, k, L))
    return e if scale is None else e * scale


def _freq(rng, m):
    """Sample a sampling frequency per row: primary period p1 (int), secondary p2 (float, 0 = none)."""
    f = rng.choice(len(FREQS), m, p=_FW)
    return f, _FP1[f], _FP2[f]


def _fourier(rng, t, p, W, off):
    """Random Fourier series sum_k W[:, k-1] (a_k cos(2 pi k (t+off)/p) + b_k sin(...)), a, b ~ N(0, 1).
    Integer periods (the common case) are evaluated once per distinct period on a one-period grid and
    gathered, so the cost is O(m L) regardless of the number of harmonics; other periods use the
    complex rotation z^k."""
    m, L, K = len(p), len(t), W.shape[1]
    a = rng.normal(0, 1, (m, K)) * W
    b = rng.normal(0, 1, (m, K)) * W
    out = np.empty((m, L))
    p = np.asarray(p, dtype=np.float64)
    integer = (p == np.round(p)) & (p <= 4096)
    k = np.arange(1, K + 1)[:, None]
    for pv in np.unique(p[integer]):
        rows = np.where(integer & (p == pv))[0]
        P = int(pv)
        ang = 2 * np.pi * k * np.arange(P)[None, :] / P                 # [K, P]
        tab = a[rows] @ np.cos(ang) + b[rows] @ np.sin(ang)             # [r, P]
        idx = (t[None, :].astype(np.int64) + off[rows, None].astype(np.int64)) % P
        out[rows] = np.take_along_axis(tab, idx, 1)
    rows = np.where(~integer)[0]
    if len(rows):
        z = np.exp(1j * (2 * np.pi * (t[None, :] + off[rows, None]) / p[rows, None]))
        zk = z.copy()
        acc = np.zeros((len(rows), L))
        for j in range(K):
            acc += a[rows, j:j + 1] * zk.real + b[rows, j:j + 1] * zk.imag
            if j < K - 1:
                zk *= z
        out[rows] = acc
    return out


def _harmonics(rng, t, p, K, decay, off):
    """Random Fourier-series profile with period p ([m]) and K harmonics ([m]), amplitude k^-decay."""
    k = np.arange(1, int(K.max()) + 1)[None, :]
    return _fourier(rng, t, p, (k <= K[:, None]) * k ** -decay, off)


def _profile(rng, t, p, off, kmax=6):
    """One seasonal profile of period p per row: smooth harmonics, sharp table (p <= 12) or peaky
    (exp-transformed, like business-hour or rush-hour shapes). Unit std over the window."""
    m, L = len(p), len(t)
    s = np.empty((m, L))
    tab = (p <= 12) & (rng.random(m) < 0.4)
    h = ~tab
    if h.any():
        ph = p[h]
        K = np.minimum(rng.integers(1, kmax + 1, len(ph)), np.maximum(ph // 2, 1)).astype(int)
        s[h] = _std(_harmonics(rng, t, ph.astype(float), K, rng.uniform(0.3, 2.0, (len(ph), 1)), off[h]))
    if tab.any():
        k = int(tab.sum())
        idx = (t[None, :] + off[tab, None]).astype(np.int64) % p[tab, None]
        s[tab] = _std(np.take_along_axis(rng.normal(0, 1, (k, 12)), idx, 1) + 1e-6 * rng.normal(0, 1, (k, L)))
    pk = rng.random(m) < 0.3
    if pk.any():
        s[pk] = _std(np.exp(np.clip(rng.uniform(0.8, 2.0, (int(pk.sum()), 1)) * s[pk], -10, 10)))
    return s


def _seasonal(rng, m, L):
    """Frequency-realistic multi-seasonal component (unit std) and the primary period.
    Sub-daily: intraday profile x weekday factor (optionally weekend dip) + weekly harmonics.
    Daily: day-of-week profile + annual harmonics (365.25). Weekly/monthly/quarterly: one period."""
    f, p1, p2 = _freq(rng, m)
    t = np.arange(L, dtype=np.float64)
    off = rng.integers(0, 4096, m).astype(np.float64)
    s = _profile(rng, t, p1, off)
    sub = _SUBDAILY[f] & (rng.random(m) < 0.8)
    if sub.any():
        k = int(sub.sum())
        day = ((t[None, :] + off[sub, None]) // p1[sub, None]).astype(np.int64) % 7
        fac = 1 + 0.15 * rng.normal(0, 1, (k, 7))
        fac[:, 5:] = np.where(rng.random((k, 1)) < 0.6, rng.uniform(0.2, 1.1, (k, 1)), fac[:, 5:])
        dayfac = np.take_along_axis(fac, day, 1)
        lvl = rng.uniform(0.0, 1.5, (k, 1))        # how much the weekday factor scales the day level
        s[sub] = (s[sub] + lvl) * dayfac - lvl * dayfac.mean(1, keepdims=True)
    two = (p2 > 0) & (rng.random(m) < 0.7)
    if two.any():
        k = int(two.sum())
        s2 = _std(_harmonics(rng, t, p2[two], rng.integers(1, 4, k), rng.uniform(0.5, 1.5, (k, 1)), off[two]))
        s[two] += rng.uniform(0.1, 0.8, (k, 1)) * s2
    return _std(s), p1


# ---------------------------------------------------------------------------
# families: each returns (y [m, L] float64, period [m] int64)
# ---------------------------------------------------------------------------

def _stat_gp(rng, m, L, ell_lo, ell_hi):
    """Stationary GP draws (RBF, or rational-quadratic as a gamma scale mixture of 3 RBFs) by
    spectral synthesis on a >=2L-periodic grid (circulant-embedding approximation, O(L log L)).
    Unit std per row."""
    nf = _nfft(L)
    fr = np.fft.rfftfreq(nf)[None, :]
    ell = _logu(rng, ell_lo, ell_hi, m)
    rq = rng.random((m, 1)) < 0.4
    alpha = _logu(rng, 0.2, 5, m)
    S = np.zeros((m, fr.shape[1]))
    for _ in range(3):
        li = np.where(rq, ell / np.sqrt(rng.gamma(alpha, 1 / alpha)), ell)
        S += li * np.exp(np.maximum(-2 * np.pi ** 2 * li ** 2 * fr ** 2, -50))
    Z = (rng.normal(0, 1, S.shape) + 1j * rng.normal(0, 1, S.shape)) * np.sqrt(S)
    st = rng.integers(0, nf - L + 1, (m, 1))
    y = np.take_along_axis(np.fft.irfft(Z, nf), st + np.arange(L)[None, :], 1)
    return _std(y)


def _per_gp(rng, m, L, p, ell=None):
    """Periodic-kernel GP draw: exact random Fourier series with harmonic weights
    q_k ~ exp(-k^2 l^2 / 2) (the large-argument form of the Bessel weights of the MacKay periodic
    kernel). Unit std."""
    t = np.arange(L, dtype=np.float64)
    ell = rng.uniform(0.15, 1.5, (m, 1)) if ell is None else ell
    K = np.clip(np.minimum(np.ceil(3 / ell[:, 0]), np.maximum(np.asarray(p) // 2, 1)), 1, 12)
    k = np.arange(1, 13)[None, :]
    W = np.exp(-(k * ell) ** 2 / 4) * (k <= K[:, None])
    return _std(_fourier(rng, t, p, W, rng.integers(0, 4096, m).astype(np.float64)))


def _gp(rng, m, L):
    """KernelSynth-style composite GP. Kernel bank: RBF and RQ (short/medium lengthscale A, a second
    independent one C), slow RBF envelope E, periodic P1 (frequency-realistic period) and P2 (second
    period), linear Lin. A random composition of 1-4 terms, each a single kernel or a product
    (P1*E locally periodic, P1*P2, A*Lin, C*E), plus an optional linear trend. Products are
    realised as products of independent draws, which has exactly the product covariance
    (non-Gaussian marginals, same second-order structure)."""
    t = np.arange(L, dtype=np.float64)
    _, p1, p2 = _freq(rng, m)
    rnd_p = rng.integers(3, max(4, L // 3), m)
    p = np.where((rng.random(m) < 0.75) & (p1 <= L // 2), p1, rnd_p).astype(np.float64)
    pb = np.where((p2 > 0) & (p2 < 4 * L) & (rng.random(m) < 0.5), p2, np.maximum(np.round(p / rng.integers(2, 5, m)), 2))
    A = _stat_gp(rng, m, L, 2, L / 2)
    C = _stat_gp(rng, m, L, L / 20, 1.5 * L)
    E = _stat_gp(rng, m, L, L / 4, 1.5 * L)
    P1 = _per_gp(rng, m, L, p)
    P2 = _per_gp(rng, m, L, pb)
    lin = t[None, :] / L - rng.uniform(-0.5, 1.5, (m, 1))
    terms = [A, C, P1, P2, P1 * (1 + E), P1 * P2, A * lin, C * E]
    prob = np.array([0.35, 0.35, 0.5, 0.2, 0.25, 0.1, 0.1, 0.1])
    on = rng.random((m, len(terms))) < prob
    none = ~on.any(1)
    on[none, rng.integers(0, len(terms), int(none.sum()))] = True
    w = on * _logu(rng, 0.2, 1, m) ** rng.uniform(0, 2, (m, len(terms)))
    y = sum(w[:, i:i + 1] * f for i, f in enumerate(terms))
    y = y + (rng.random((m, 1)) < 0.3) * rng.normal(0, 1, (m, 1)) * lin * y.std(1, keepdims=True) * 2
    y = y + rng.normal(0, 1, (m, L)) * _logu(rng, 1e-3, 0.2, m) * y.std(1, keepdims=True)   # white kernel
    perw = w[:, 2] + w[:, 4] + w[:, 5]
    domin = (perw > 0.5 * w.max(1)) & (perw > 0)
    return y, np.where(domin, p, 1).astype(np.int64)


def _season(rng, m, L):
    t = np.arange(L, dtype=np.float64)
    s, p1 = _seasonal(rng, m, L)
    tr = np.where(rng.random((m, 1)) < 0.5, rng.normal(0, 1, (m, 1)) * t / L,
                  rng.normal(0, 0.7, (m, 1)) * _smooth_rw(rng, m, L, 4))
    amp = _logu(rng, 0.3, 3, m)
    noise = _noise(rng, m, L, _logu(rng, 0.03, 0.6, m))
    add = tr + amp * s + noise
    # multiplicative: positive level x exp(seasonal); noise multiplicative or additive
    lvl = np.exp(0.5 * tr)
    mult = lvl * np.exp(0.5 * amp * s / (1 + 0.3 * amp)) * np.where(rng.random((m, 1)) < 0.5, np.exp(0.3 * noise), 1) \
        + np.where(rng.random((m, 1)) < 0.5, 0.1 * noise, 0)
    y = np.where(rng.random((m, 1)) < 0.4, mult, add)
    return y, p1


def _arima(rng, m, L):
    B = 128
    Lb = L + B
    k = np.arange(Lb)[None, :]
    order = rng.integers(0, 3, (m, 1))                      # AR order 0, 1, 2
    cplx = rng.random((m, 1)) < 0.4
    rho, th = rng.uniform(0.5, 0.98, (m, 1)), rng.uniform(0.05, np.pi, (m, 1))
    r1 = np.where(cplx, rho * np.exp(1j * th), rng.uniform(-0.8, 0.99, (m, 1)))
    r2 = np.where(cplx, np.conj(r1), rng.uniform(-0.6, 0.9, (m, 1)))
    r1 = np.where(order == 0, 0, r1)
    r2 = np.where(order <= 1, 0, r2)
    r2 = np.where(np.abs(r1 - r2) < 1e-3, r2 + 2e-3, r2)
    h1 = np.real((r1 ** (k + 1) - r2 ** (k + 1)) / (r1 - r2))
    # MA(2)
    th1, th2 = rng.uniform(-0.8, 0.8, (2, m, 1)) * (rng.random((2, m, 1)) < 0.5)
    hma = np.zeros((m, Lb)); hma[:, :1] = 1; hma[:, 1:2] = th1; hma[:, 2:3] = th2
    # seasonal AR / seasonal unit root at the frequency's period
    _, p1, _ = _freq(rng, m)
    s = np.where(p1 <= Lb // 3, p1, 1)[:, None]
    Phi = np.where(rng.random((m, 1)) < 0.25, 1.0, rng.uniform(0.3, 0.95, (m, 1)))
    seas = (rng.random((m, 1)) < 0.5) & (s > 1)
    hs = np.where((k % s == 0) & seas, Phi ** (k // s), 0.0)
    hs[:, 0] = 1.0
    e = _noise(rng, m, Lb)
    x = _filt(e, h1, hma, hs)[:, B:]
    d = rng.choice(3, (m, 1), p=[0.5, 0.4, 0.1])
    x1 = np.cumsum(x, 1)
    x2 = np.cumsum(x1, 1) / np.sqrt(L)
    y = np.select([d == 0, d == 1], [x, x1], x2)
    y = y + rng.normal(0, 1, (m, 1)) * (d == 1) * np.arange(L) * rng.uniform(0, 0.3, (m, 1))
    return y, np.where(seas[:, 0], s[:, 0], 1).astype(np.int64)


def _ets(rng, m, L):
    t = np.arange(L, dtype=np.float64)
    e = _noise(rng, m, L)
    alpha, beta = _u(rng, 0.05, 0.6, m), _u(rng, 0.0, 0.15, m) * (rng.random((m, 1)) < 0.6)
    phi = np.where(rng.random((m, 1)) < 0.5, 1.0, _u(rng, 0.8, 0.98, m))
    b0 = rng.normal(0, 0.05, (m, 1)) * (rng.random((m, 1)) < 0.6)
    b = _ar1(beta * e, phi) + b0 * phi ** t                  # damped slope
    lvl = np.cumsum(np.concatenate([np.zeros((m, 1)), b[:, :-1]], 1) + alpha * e, 1)
    s, p1 = _seasonal(rng, m, L)
    has_s = rng.random((m, 1)) < 0.75
    gam = _u(rng, 0.0, 0.6, m)                               # evolving seasonal amplitude
    s = s * np.exp(gam * _smooth_rw(rng, m, L, 8)) * _logu(rng, 0.3, 3, m) * has_s
    sig = _logu(rng, 0.05, 0.5, m)
    add = lvl + s + sig * e
    mult = np.exp(_u(rng, 0.1, 1.0, m) * _std(lvl + 1e-9 * t)) * (1 + 0.25 * np.tanh(s)) * (1 + 0.3 * sig * e)
    y = np.where(rng.random((m, 1)) < 0.35, mult, add)
    return y, np.where(has_s[:, 0], p1, 1).astype(np.int64)


def _regime(rng, m, L):
    K = rng.integers(2, 4, (m, 1))
    psw = _logu(rng, 0.002, 0.05, m)
    sw = rng.random((m, L)) < psw
    sw[:, 0] = False
    seg = np.cumsum(sw, 1)                                    # segment id
    st = np.cumsum(rng.integers(1, 3, (m, L)), 1) % K         # regime per segment, always changes
    st[:, 0] = rng.integers(0, 3, m) % K[:, 0]
    reg = np.take_along_axis(st, seg, 1)
    mus = rng.normal(0, 1, (m, 3)) * _logu(rng, 0.5, 4, m)
    sig = _logu(rng, 0.05, 1, m) * np.exp(rng.normal(0, 0.8, (m, 3)))
    dr = rng.normal(0, 0.03, (m, 3))
    what = rng.integers(0, 4, (m, 1))                         # 0 mean, 1 vol, 2 mean+vol, 3 drift
    mu_t = np.where((what == 0) | (what == 2), np.take_along_axis(mus, reg, 1), 0)
    sg_t = np.where((what == 1) | (what == 2), np.take_along_axis(sig, reg, 1), sig[:, :1])
    phi = 1 - _logu(rng, 0.01, 0.5, m)                        # OU mean reversion
    x = _ar1((1 - phi) * mu_t + sg_t * np.sqrt(1 - phi ** 2) * _noise(rng, m, L), phi)
    x = x + mu_t[:, :1] * phi ** np.arange(L)                 # start near the first regime mean
    drift = np.cumsum(np.take_along_axis(dr, reg, 1), 1)
    x = np.where(what == 3, drift + 0.3 * x, x)
    s, p1 = _seasonal(rng, m, L)
    has_s = rng.random((m, 1)) < 0.3
    x = x + has_s * s * x.std(1, keepdims=True) * _u(rng, 0.2, 1, m)
    return x, np.where(has_s[:, 0], p1, 1).astype(np.int64)


def _trend(rng, m, L):
    t = np.arange(L, dtype=np.float64)
    kmax = 5
    k = rng.integers(0, kmax + 1, (m, 1))
    cps = np.sort(rng.uniform(0, L, (m, kmax)), 1)
    cps = np.where(np.arange(kmax)[None, :] < k, cps, np.inf)
    segi = (t[None, :, None] >= cps[:, None, :]).sum(2)
    slopes = rng.normal(0, 1, (m, kmax + 1)) / L * _logu(rng, 1, 10, m)
    pl = np.cumsum(np.take_along_axis(slopes, segi, 1), 1)
    t0, r = rng.uniform(-0.3, 1.3, (m, 1)) * L, _logu(rng, 2, 30, m) / L
    logi = 1 / (1 + np.exp(-np.clip(r * (t - t0), -50, 50))) * rng.choice([-1, 1], (m, 1))
    ex = np.expm1(np.clip(rng.uniform(-3, 3, (m, 1)) * t / L, -30, 30))
    kind = rng.integers(0, 3, (m, 1))
    base = _std(np.select([kind == 0, kind == 1], [pl, logi + 0.3 * pl], ex + 0.2 * pl) + 1e-9 * t)
    s, p1 = _seasonal(rng, m, L)
    has_s = rng.random((m, 1)) < 0.5
    a = _u(rng, 0.1, 0.8, m) * has_s
    mult = rng.random((m, 1)) < 0.4
    y = np.where(mult, (base - base.min(1, keepdims=True) + 0.5) * (1 + 0.5 * a * np.tanh(s)), base + a * s)
    y = y + _noise(rng, m, L, _logu(rng, 0.01, 0.3, m))
    return y, np.where(has_s[:, 0], p1, 1).astype(np.int64)


def _rw(rng, m, L):
    e = _noise(rng, m, L)
    drift = rng.normal(0, 0.1, (m, 1)) * (rng.random((m, 1)) < 0.6)
    x = np.cumsum(e + drift, 1)
    kind = rng.integers(0, 3, (m, 1))
    smooth = np.cumsum(_std(x), 1) / np.sqrt(L)               # integrated RW
    geo = np.exp(np.clip(x * _logu(rng, 0.003, 0.03, m), -20, 20))
    return np.select([kind == 0, kind == 1], [x, smooth], geo), np.ones(m, np.int64)


def _vol(rng, m, L):
    """Stochastic volatility (log-variance AR(1)): volatility clustering like GARCH, heavy tails."""
    phi = _u(rng, 0.85, 0.995, m)
    h = _ar1(rng.normal(0, 1, (m, L)) * _u(rng, 0.1, 0.5, m) * np.sqrt(1 - phi ** 2), phi) * 3
    r = np.exp(h / 2) * _noise(rng, m, L) * _logu(rng, 0.003, 0.03, m)
    kind = rng.random((m, 1))
    price = np.exp(np.clip(np.cumsum(r + rng.normal(0, 3e-4, (m, 1)), 1), -20, 20))
    y = np.where(kind < 0.3, r, np.where(kind < 0.5, np.abs(r), price))
    return y, np.ones(m, np.int64)


def _count(rng, m, L):
    lam = _logu(rng, 0.05, 30, m)
    s, p1 = _seasonal(rng, m, L)
    has_s = rng.random((m, 1)) < 0.5
    rate = lam * np.exp(has_s * _u(rng, 0.2, 1.2, m) * s + _u(rng, 0, 0.8, m) * _smooth_rw(rng, m, L, 4))
    nb = rng.random((m, 1)) < 0.5                                   # negative binomial via gamma mixing
    kdisp = _logu(rng, 0.3, 10, m)
    rate = np.where(nb, rate * rng.gamma(np.broadcast_to(kdisp, (m, L)), 1) / kdisp, rate)
    pocc = np.where(rng.random((m, 1)) < 0.6, _u(rng, 0.05, 0.7, m), 1.0)   # zero inflation
    y = (rng.random((m, L)) < pocc) * rng.poisson(np.minimum(rate, 1e6))
    return y.astype(np.float64), np.where(has_s[:, 0], p1, 1).astype(np.int64)


def _pattern(rng, m, L):
    t = np.arange(L, dtype=np.float64)
    _, p1, _ = _freq(rng, m)
    pr = rng.integers(3, max(4, L // 4), m)
    p = np.where((rng.random(m) < 0.6) & (p1 <= L // 2), p1, pr)
    off = rng.integers(0, 4096, (m, 1))
    ph = ((t[None, :] + off) % p[:, None]) / p[:, None]           # phase in [0, 1)
    saw = ph * np.where(rng.random((m, 1)) < 0.5, 1, -1)
    sq = (ph < _u(rng, 0.1, 0.9, m)).astype(float)
    tri = 1 - 2 * np.abs(ph - 0.5)
    spk = np.exp(-ph * p[:, None] / _logu(rng, 0.3, 10, m))       # decaying spike train
    jumps = (rng.random((m, L)) < _logu(rng, 0.003, 0.05, m)) * rng.normal(0, 1, (m, L))
    steps = np.cumsum(jumps, 1)
    rsp = (rng.random((m, L)) < _logu(rng, 0.005, 0.05, m)) * rng.exponential(1, (m, L)) * rng.choice([-1, 1], (m, 1))
    kind = rng.integers(0, 6, (m, 1))
    y = np.select([kind == 0, kind == 1, kind == 2, kind == 3, kind == 4], [saw, sq, tri, spk, steps], rsp)
    y = _std(y + 1e-9 * rng.normal(0, 1, (m, L)))
    extra = rng.random((m, 1)) < 0.4                              # add a second, different pattern
    y = y + extra * _u(rng, 0.2, 0.7, m) * np.where(kind == 4, rsp * 3, steps)
    y = y + _noise(rng, m, L, _logu(rng, 0.005, 0.3, m))
    periodic = (kind[:, 0] <= 3)
    return y, np.where(periodic, p, 1).astype(np.int64)


_ACTS = (lambda x: x, np.tanh, lambda x: np.maximum(x, 0), np.sin, lambda x: x * x / 2,
         lambda x: 1 / (1 + np.exp(-2 * x)), np.abs)


def _causal(rng, m, L):
    """CauKer-like: J latent root signals (seasonal, AR/OU, RW, steps, sinusoid), a random DAG in
    topological order, node_i = act_i(sum_j W_ij node_j[t - lag_ij] + b_i) + eps * root_i, output
    a random node (usually the last) or a random linear read-out of nodes."""
    J = 5
    t = np.arange(L, dtype=np.float64)
    s, p1 = _seasonal(rng, m, L)
    roots = np.stack([s, _per_gp(rng, m, L, rng.integers(3, max(4, L // 3), m).astype(float)),
                      _std(_ar1(rng.normal(0, 1, (m, L)), _u(rng, 0.5, 0.99, m))),
                      _std(np.cumsum(rng.normal(0, 1, (m, L)), 1)),
                      _std(np.cumsum((rng.random((m, L)) < 0.01) * rng.normal(0, 1, (m, L)), 1) + 1e-6 * t)], 1)
    roots = roots[:, rng.permutation(J)]
    n_nodes = rng.integers(3, J + 1, (m, 1))
    nodes = np.zeros((m, J, L))
    for i in range(J):
        x = roots[:, i]
        if i:
            W = rng.normal(0, 1, (m, i)) * (rng.random((m, i)) < 0.6)
            lag = rng.integers(0, 12, (m, i)) * (rng.random((m, i)) < 0.4)
            agg = np.zeros((m, L))
            for j in range(i):
                lj = lag[:, j:j + 1]
                idx = np.maximum(t[None, :].astype(np.int64) - lj, 0)
                agg += W[:, j:j + 1] * np.take_along_axis(nodes[:, j], idx, 1)
            a = rng.integers(0, len(_ACTS), m)
            z = agg + rng.normal(0, 0.3, (m, 1))
            out = np.zeros((m, L))
            for ai, fn in enumerate(_ACTS):
                sel = a == ai
                if sel.any():
                    out[sel] = fn(z[sel])
            x = _std(out + rng.uniform(0.05, 1, (m, 1)) * x + 1e-9 * rng.normal(0, 1, (m, L)))
        nodes[:, i] = x
    last = np.take_along_axis(nodes, (n_nodes - 1)[:, :, None].repeat(L, 2), 1)[:, 0]
    mix = np.einsum("mj,mjl->ml", rng.normal(0, 1, (m, J)) * (np.arange(J)[None] < n_nodes), nodes)
    y = np.where(rng.random((m, 1)) < 0.7, last, mix)
    y = y + _noise(rng, m, L, _logu(rng, 0.01, 0.2, m))
    # report a period only if the output still carries the root season clearly (checked cheaply:
    # lag-p autocorrelation of the output)
    pp = np.minimum(p1, L // 2)
    yc = y - y.mean(1, keepdims=True)
    idx = np.minimum(t[None, :].astype(np.int64) + pp[:, None], L - 1)
    valid = t[None, :] < (L - pp[:, None])
    ac = (yc * np.take_along_axis(yc, idx, 1) * valid).sum(1) / ((yc ** 2).sum(1) + 1e-9) * L / np.maximum(L - pp, 1)
    return y, np.where((ac > 0.5) & (pp == p1), p1, 1).astype(np.int64)


_GEN = {"gp": _gp, "season": _season, "arima": _arima, "ets": _ets, "regime": _regime, "trend": _trend,
        "rw": _rw, "vol": _vol, "count": _count, "pattern": _pattern, "causal": _causal}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _postprocess(rng, y, is_count):
    """Row-level post-processing; continuous rows are standardised first, count rows keep their
    integer values. Each step touches only the rows it fires on."""
    n, L = y.shape
    c = ~is_count
    y[c] = _std(y[c])
    sd = y.std(1)

    def pick(p, base=c):
        return np.where(base & (rng.random(n) < p))[0]

    r = pick(0.25)                                   # extra heteroscedastic / heavy-tailed noise
    if len(r):
        y[r] += _noise(rng, len(r), L, _logu(rng, 0.02, 0.3, len(r)))
    pos = pick(0.4)                                  # positivity shift (most real series are > 0)
    if len(pos):
        y[pos] += -y[pos].min(1, keepdims=True) + rng.exponential(0.5, (len(pos), 1))
    cen = pick(0.05)                                 # censoring at a floor
    if len(cen):
        y[cen] = np.maximum(y[cen] - np.quantile(y[cen], rng.uniform(0.05, 0.4), axis=1, keepdims=True), 0)
    shifted = np.zeros(n, bool); shifted[pos] = True; shifted[cen] = True
    sc = np.exp(rng.normal(0, 2, n))
    off = np.where(shifted | (rng.random(n) < 0.5), 0, rng.normal(0, 3, n))
    y[c] = (y[c] + off[c, None]) * sc[c, None]
    q = pick(0.08)                                   # quantisation (rounded reporting)
    if len(q):
        step = y[q].std(1, keepdims=True) * _logu(rng, 0.02, 0.3, len(q)) + 1e-12
        y[q] = np.round(y[q] / step) * step
    tt = np.arange(L)[None, :]
    mi = pick(0.08, True)                            # iid missing values
    if len(mi):
        y[mi] = np.where(rng.random((len(mi), L)) < _logu(rng, 0.01, 0.15, len(mi)), np.nan, y[mi])
    ru = pick(0.05, True)                            # one missing run
    if len(ru):
        st, ln = rng.integers(0, L, (len(ru), 1)), rng.integers(2, max(3, L // 8), (len(ru), 1))
        y[ru] = np.where((tt >= st) & (tt < st + ln) & (tt < L - 1), np.nan, y[ru])
    return y


def _make(rng, f, c, L):
    """c finished rows (post-processed, no left padding) of family index f."""
    y, per = _GEN[FAMILIES[f]](rng, c, L)
    bad = ~np.isfinite(y).all(1)
    if bad.any():                                            # never emit inf; fall back to noise
        y[bad] = rng.normal(0, 1, (int(bad.sum()), L)); per[bad] = 1
    y = _postprocess(rng, y, np.full(c, FAMILIES[f] == "count"))
    return y.astype(np.float32), np.where(per >= L, 1, per).astype(np.int64)


POOL_ROWS = 512          # rows generated per family refill: amortises numpy call overhead
_POOL = {}               # (family, length) -> [y, period, next row]; ~1.3 MB per family at L=576


def _take(rng, f, c, L):
    key = (f, L)
    buf = _POOL.get(key)
    if buf is None or buf[2] + c > len(buf[0]):
        if len(_POOL) > 8 * len(FAMILIES):
            _POOL.clear()
        y, per = _make(rng, f, max(POOL_ROWS, c), L)
        buf = _POOL[key] = [y, per, 0]
    i = buf[2]
    buf[2] = i + c
    return buf[0][i:i + c], buf[1][i:i + c]


def generate(rng, n, length, short_frac=0.5, min_len=16, weights=None, return_family=False, pool=True):
    """n synthetic series of `length` points (right-aligned; NaN = missing or left pad).
    Returns y float32 [n, length], period int64 [n] (dominant season, 1 = none) and, with
    return_family, the family index per row (into FAMILIES).
    pool=True serves rows from per-family buffers of POOL_ROWS fresh rows, refilled with `rng` when
    exhausted; every row is used once. Family weights are drawn per call either way, so batches
    stay diverse; pool=False generates exactly n rows now (slower for small n)."""
    L = int(length)
    y = np.zeros((n, L), dtype=np.float32)
    per = np.ones(n, dtype=np.int64)
    fam = np.zeros(n, dtype=np.int64)
    if n == 0:
        return (y, per, fam) if return_family else (y, per)
    w = rng.dirichlet(np.full(len(FAMILIES), 10.0)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    counts = rng.multinomial(n, w)
    i = 0
    for f, c in enumerate(counts):
        if c:
            y[i:i + c], per[i:i + c] = _take(rng, f, c, L) if pool else _make(rng, f, c, L)
            fam[i:i + c] = f
            i += c
    lens = np.where(rng.random(n) < short_frac, rng.integers(min(min_len, L), L + 1, n), L)
    y[np.arange(L)[None, :] < (L - lens)[:, None]] = np.nan
    order = rng.permutation(n)
    out = (y[order], per[order])
    return out + (fam[order],) if return_family else out


def _expanding_std(x):
    """Strictly causal std at each t from observed values at times < t (NaN-aware)."""
    o = ~np.isnan(x)
    x0 = np.where(o, x, 0.0)
    c = np.cumsum(o, 1); s1 = np.cumsum(x0, 1); s2 = np.cumsum(x0 * x0, 1)
    c, s1, s2 = (np.concatenate([np.zeros((x.shape[0], 1)), a[:, :-1]], 1) for a in (c, s1, s2))
    mu = s1 / np.maximum(c, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mu * mu, 0))
    return np.where(c >= 2, sd, np.nan)


AUG_P = {"amp": 0.25, "censor": 0.10, "spike": 0.15, "missing": 0.12, "reagg": 0.10, "shift": 0.12}


def augment(win, rng, period=None, probs=None, min_obs=32):
    """Augment real windows. win: float32 [n, L], right-aligned, NaN = pad/missing. Each transform
    fires independently per row with probability probs[name] (AUG_P by default). All statistics
    used are causal (computed from values strictly before the point they affect, or from a prefix
    that is only applied after that prefix), and random positions/curves are independent of the data,
    so no transform leaks future values into the past.
      amp      multiply by a smooth positive random curve (piecewise-linear in log space)
      censor   cap (or floor) values after a prefix at a quantile of that prefix
      spike    add sparse spikes scaled by the causal expanding std
      missing  NaN runs and iid gaps (padding untouched)
      reagg    sum or mean over non-overlapping blocks of k (aligned to the last point), result
               right-aligned and left-padded; period -> period/k if divisible, else 1
      shift    add 1-2 level shifts scaled by the causal std just before the shift
    Returns win' (float32), or (win', period') if period is given."""
    P = dict(AUG_P, **(probs or {}))
    x = np.array(win, dtype=np.float64, copy=True)
    n, L = x.shape
    per = None if period is None else np.array(period, dtype=np.int64, copy=True)
    if n == 0:
        return (x.astype(np.float32), per) if per is not None else x.astype(np.float32)
    t = np.arange(L)[None, :]
    obs = ~np.isnan(x)
    nobs = obs.sum(1)
    first = np.where(nobs > 0, obs.argmax(1), L)
    span = L - first                                      # length incl. internal gaps
    fire = {k: (rng.random(n) < p) & (nobs >= min_obs) for k, p in P.items()}

    r = fire["reagg"]
    if r.any():
        ks = np.array([2, 3, 4, 6, 12, 24])
        k = ks[rng.integers(0, len(ks), n)]
        k = np.where(span // k >= min_obs, k, 1)
        for kv in np.unique(k[r]):
            if kv == 1:
                continue
            rows = np.where(r & (k == kv))[0]
            nb = L // kv
            blk = x[rows, L - nb * kv:].reshape(len(rows), nb, kv)
            ob = ~np.isnan(blk)
            cnt = ob.sum(2)
            tot = np.where(ob, blk, 0).sum(2)
            use_sum = rng.random((len(rows), 1)) < 0.5
            val = np.where(use_sum, tot * kv / np.maximum(cnt, 1), tot / np.maximum(cnt, 1))
            val = np.where(cnt > 0, val, np.nan)
            new = np.full((len(rows), L), np.nan)
            new[:, L - nb:] = val
            x[rows] = new
            if per is not None:
                p = per[rows]
                per[rows] = np.where((p % kv == 0) & (p // kv > 1), p // kv, 1)
        obs = ~np.isnan(x)
        nobs = obs.sum(1)
        first = np.where(nobs > 0, obs.argmax(1), L)
        span = L - first

    a = fire["amp"]
    if a.any():
        m = int(a.sum())
        K = 8
        knots = np.cumsum(rng.normal(0, 1, (m, K + 1)), 1) * rng.uniform(0.05, 0.25, (m, 1))
        xs = np.linspace(0, K, L)
        i = np.minimum(xs.astype(int), K - 1)
        w = xs - i
        x[a] *= np.exp(knots[:, i] * (1 - w) + knots[:, i + 1] * w)

    c = fire["censor"]
    if c.any():
        rows = np.where(c)[0]
        pre = np.maximum((span[rows] * rng.uniform(0.2, 0.5, len(rows))).astype(int), 16)
        upper = rng.random(len(rows)) < 0.6
        q = np.where(upper, rng.uniform(0.7, 0.99, len(rows)), rng.uniform(0.01, 0.3, len(rows)))
        for j, rw in enumerate(rows):                    # per-row quantile of a variable prefix
            seg = x[rw, first[rw]:first[rw] + pre[j]]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 8:
                continue
            thr = np.quantile(seg, q[j])
            tail = x[rw, first[rw] + pre[j]:]
            x[rw, first[rw] + pre[j]:] = np.minimum(tail, thr) if upper[j] else np.maximum(tail, thr)

    sd = _expanding_std(x) if (fire["spike"].any() or fire["shift"].any()) else None
    s = fire["spike"]
    if s.any():
        m = int(s.sum())
        pos = (rng.random((m, L)) < rng.uniform(0.002, 0.02, (m, 1))) & obs[s] & (t >= first[s, None] + 8)
        mag = rng.normal(0, 1, (m, L)) * rng.uniform(2, 8, (m, 1)) * np.nan_to_num(sd[s])
        x[s] = x[s] + pos * mag

    h = fire["shift"]
    if h.any():
        rows = np.where(h)[0]
        for rep in range(2):
            keep = rows if rep == 0 else rows[rng.random(len(rows)) < 0.3]
            if not len(keep):
                break
            tau = first[keep] + (rng.uniform(0.15, 0.95, len(keep)) * span[keep]).astype(int)
            tau = np.minimum(tau, L - 1)
            base = np.nan_to_num(sd[keep, tau])
            delta = rng.normal(0, 1, len(keep)) * rng.uniform(1, 4, len(keep)) * base
            x[keep] += (t >= tau[:, None]) * delta[:, None]

    mi = fire["missing"]
    if mi.any():
        m = int(mi.sum())
        st = first[mi, None] + (rng.random((m, 3)) * span[mi, None]).astype(int)
        ln = rng.geometric(1 / rng.uniform(3, 30, (m, 3)))
        on = rng.random((m, 3)) < np.array([1.0, 0.5, 0.25])
        gap = ((t[:, :, None] >= st[:, None, :]) & (t[:, :, None] < (st + ln)[:, None, :]) & on[:, None, :]).any(2)
        gap |= (rng.random((m, 1)) < 0.3) & (rng.random((m, L)) < 0.05)
        gap[:, -1] = False                               # keep the last point (an origin anchor)
        x[mi] = np.where(gap, np.nan, x[mi])

    x = x.astype(np.float32)
    return (x, per) if per is not None else x


# ---------------------------------------------------------------------------
# benchmark + diagnostics
# ---------------------------------------------------------------------------

def _spectral_check(y, per):
    """Share of (detrended) periodogram power at harmonics k/p (k<=4, +-1 bin) for seasonal rows,
    versus the same statistic computed for non-seasonal rows at a random period."""
    L = y.shape[1]
    full = ~np.isnan(y).any(1)
    t = np.arange(L)
    def share(rows, ps):
        out, hit = [], []
        for r, p in zip(rows, ps):
            v = y[r].astype(np.float64)
            v = v - np.polyval(np.polyfit(t, v, 1), t)
            pw = np.abs(np.fft.rfft(v)) ** 2
            pw[0] = 0
            bins = np.unique([int(round(k * L / p)) + d for k in range(1, 5) for d in (-1, 0, 1)
                              if 0 < round(k * L / p) + d < len(pw)])
            out.append(pw[bins].sum() / (pw.sum() + 1e-12))
            hit.append(int(pw.argmax()) in set(bins))
        return np.median(out) if out else np.nan, np.mean(hit) if hit else np.nan
    rs = np.where(full & (per > 1) & (per <= L // 2))[0]
    rn = np.where(full & (per == 1))[0]
    rnd = np.random.default_rng(0).choice([7, 12, 24, 52, 96], len(rn))
    return share(rs, per[rs]), share(rn, rnd), len(rs), len(rn)


if __name__ == "__main__":
    import os
    import time
    from collections import Counter

    rng = np.random.default_rng(0)
    L = 576
    for _ in range(3):
        generate(rng, 128, L)
    ts, tc = [], []
    for _ in range(200):
        t0, c0 = time.perf_counter(), time.process_time()
        generate(rng, 128, L)
        ts.append(time.perf_counter() - t0); tc.append(time.process_time() - c0)
    print(f"generate(128, {L}) with pool: mean wall {1e3 * np.mean(ts):.1f} ms, mean CPU {1e3 * np.mean(tc):.1f} ms "
          f"(amortised over 200 calls incl. refills); median wall {1e3 * np.median(ts):.1f} ms (p90 {1e3 * np.quantile(ts, 0.9):.1f}), "
          f"median CPU {1e3 * np.median(tc):.1f} ms; load avg {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores")
    for f in FAMILIES:
        w = np.array([x == f for x in FAMILIES], float)
        tf = []
        for _ in range(3):
            t0 = time.process_time(); generate(rng, 512, L, weights=w, pool=False); tf.append(time.process_time() - t0)
        print(f"   only {f:8s}: {1e3 * np.median(tf) / 4:6.1f} ms CPU per 128 rows (no pool, 512-row calls)")
    real, _ = generate(rng, 128, L)
    ta = []
    for _ in range(30):
        t0 = time.perf_counter(); augment(real, rng, period=np.full(128, 24)); ta.append(time.perf_counter() - t0)
    print(f"augment(128, {L}) at default probs: median {1e3 * np.median(ta):.1f} ms")
    ta = []
    for _ in range(10):
        t0 = time.perf_counter(); augment(real, rng, probs={k: 1.0 for k in AUG_P}); ta.append(time.perf_counter() - t0)
    print(f"augment(128, {L}) every transform on every row: median {1e3 * np.median(ta):.1f} ms")

    y, per, fam = generate(np.random.default_rng(1), 8192, L, return_family=True)
    print("\nshare by family:", {f: round(float(np.mean(fam == i)), 3) for i, f in enumerate(FAMILIES)})
    print(f"finite or NaN only: {bool(np.all(np.isfinite(y) | np.isnan(y)))}; "
          f"fraction NaN {np.isnan(y).mean():.3f} (left pad + missing); rows with internal gaps "
          f"{np.mean([np.isnan(r[np.argmax(~np.isnan(r)):]).any() for r in y]):.3f}")
    c = Counter(per.tolist())
    print("period distribution:", ", ".join(f"{k}:{v / len(per):.3f}" for k, v in sorted(c.items(), key=lambda z: -z[1])[:14]))
    print("period>1 by family:", {f: round(float(np.mean(per[fam == i] > 1)), 2) for i, f in enumerate(FAMILIES)})
    absmax = np.nanmax(np.abs(y), 1)
    print(f"|y| max per row: median {np.median(absmax):.3g}, p99 {np.quantile(absmax, 0.99):.3g}, max {absmax.max():.3g}")
    yc, pc = generate(np.random.default_rng(2), 4096, L, short_frac=0.0)
    (s_seas, hit_s), (s_null, hit_n), ns, nn = _spectral_check(yc, pc)
    print(f"spectral sanity (complete rows): power share at harmonics of the reported period "
          f"{s_seas:.2f} (argmax hits {hit_s:.2f}, n={ns}) vs non-seasonal rows at a random period "
          f"{s_null:.2f} (argmax hits {hit_n:.2f}, n={nn})")
    a, pa = augment(yc[:2048], np.random.default_rng(3), period=pc[:2048])
    print(f"augment: rows changed {np.mean(~np.all((a == yc[:2048].astype(np.float32)) | (np.isnan(a) & np.isnan(yc[:2048])), 1)):.2f}, "
          f"period changed {np.mean(pa != pc[:2048]):.3f}, NaN frac {np.isnan(a).mean():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping PNG (uv run --with matplotlib tools/synth3.py)")
        raise SystemExit
    g = np.random.default_rng(4)
    rows = []
    for i, f in enumerate(FAMILIES):                   # 2-3 examples per family, + 3 augmented
        w = np.eye(len(FAMILIES))[i]
        yy, pp = generate(g, 3 if i < 5 else 2, L, short_frac=0.15, weights=w)
        rows += [(f, yy[j], pp[j]) for j in range(len(yy))]
    yy, pp = generate(g, 3, L, short_frac=0.0, weights=np.eye(len(FAMILIES))[1])
    aa, pa2 = augment(yy, g, period=pp, probs={k: 0.5 for k in AUG_P})
    rows += [("augmented", aa[j], pa2[j]) for j in range(3)]
    fig, axes = plt.subplots(6, 5, figsize=(22, 15))
    for ax, (f, v, p) in zip(axes.flat, rows):
        ax.plot(v, lw=0.7)
        ax.set_title(f"{f}  period={p}", fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "synth3_examples.png")
    fig.savefig(out, dpi=80)
    print("saved", out)
