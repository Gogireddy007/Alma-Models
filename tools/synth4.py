"""
synth4: gap-targeted synthetic prior + causal time-stretch of real high-frequency windows.

    generate(rng, n, length, freq_weights=None) -> y float32 [n, length] (NaN = missing / left pad),
                                                  period int64 [n]
    stretch(y, k, rng) -> causal k-times upsampled copy of a real series (period becomes m * k)
    gap_weights(real_share, syn_frac, target=TARGET) -> synthetic frequency weights that complement
                                                  the real rows toward a target frequency mix

Why: the real TRAIN split has no 5/10/15-minute data at all (only 5 series at
30 min), 53% of real batch rows are monthly/quarterly series shorter than one training window, and
synth3 puts only ~10% of its rows on periods >= 96, half of them left-padded so that the long
season is invisible. synth4 is frequency-conditioned: each row first draws a sampling frequency
(from `freq_weights`, by default the gap-filling prior), then a family that is realistic at that
frequency. Long-season rows are never padded below two full cycles when the window allows it.

Families (all numpy-vectorised over rows and time, reusing synth3 helpers):
    longseason  intraday/weekly profile whose SHAPE drifts (mixture of two profiles with a slow
                weight), per-day level AR(1), weekday factor and weekend dip, amplitude drift
    multiseas   2-3 superposed seasonalities: primary, secondary (weekly / annual), a harmonic
                sub-period and an incommensurate non-integer period (tidal-like)
    solar       clipped diurnal bell (exact zeros at night), daylength drift, Markov clear/cloudy
                days times fast cloud OU; daily rows = annual cycle x cloudiness
    intermittent Croston-type demand: seasonal occurrence probability, lognormal/NB sizes,
                obsolescence (fade-out) or new-product ramp-up
    cumulative  running totals of non-negative counts (epidemic waves or demand), with optional
                weekend non-reporting (stair steps)
    epidemic    1-4 overlapping epidemic waves (logistic-derivative bumps), weekly reporting
                effect, negative-binomial observation; optional annual (flu) cycle
    regseason   Markov regime per cycle (day or week): 2-3 different seasonal profiles and levels
                (workday / holiday / outage), sticky
    bursty      seasonal baseline + self-exciting bursts with heavy-tailed sizes and exponential
                decay, or counter-reset sawtooth, or plateau steps (observability-like)
References: see the report that introduced train_data.py (lit: TBATS, Croston, Prophet, KernelSynth, TempoPFN).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth3 as S3  # noqa: E402

FAMILIES = ("longseason", "multiseas", "solar", "intermittent", "cumulative", "epidemic", "regseason", "bursty")
FAM_W = np.array([0.18, 0.13, 0.10, 0.13, 0.07, 0.07, 0.14, 0.18])

# frequency -> primary period (fev/GIFT seasonality convention), secondary period (0 = none),
# observations per day (0 = coarser than daily)
FREQS = (("5min", 288, 2016, 288), ("10min", 144, 1008, 144), ("15min", 96, 672, 96),
         ("30min", 48, 336, 48), ("H", 24, 168, 24), ("D", 7, 365.25, 1),
         ("W", 52, 0, 0), ("M", 12, 0, 0), ("Q", 4, 0, 0), ("Y", 1, 0, 0))
FNAMES = [f[0] for f in FREQS]
_P1 = np.array([f[1] for f in FREQS], np.int64)
_P2 = np.array([f[2] for f in FREQS], np.float64)
_SUB = np.array([f[3] > 1 for f in FREQS])          # sub-daily

# Target share of ALL training rows (real + synthetic) per frequency. A fixed, documented design
# prior (not fitted to VAL scores): roughly the frequency mix of public zero-shot suites
# (GIFT-Eval / fev-bench, where >40% of configurations are sub-daily) shrunk toward uniform.
TARGET = {"5min": 0.10, "10min": 0.08, "15min": 0.12, "30min": 0.07, "H": 0.20, "D": 0.18,
          "W": 0.10, "M": 0.08, "Q": 0.04, "Y": 0.03}
DEFAULT_FW = np.array([TARGET[f] for f in FNAMES]); DEFAULT_FW /= DEFAULT_FW.sum()

# which frequencies each family may use (masks over FREQS)
_ALLOWED = {
    "longseason":   _SUB | np.isin(FNAMES, ["D"]),
    "multiseas":    _SUB | np.isin(FNAMES, ["D", "W"]),
    "solar":        _SUB | np.isin(FNAMES, ["D"]),
    "intermittent": np.ones(len(FREQS), bool),
    "cumulative":   np.isin(FNAMES, ["H", "D", "W"]),
    "epidemic":     np.isin(FNAMES, ["D", "W"]),
    "regseason":    _SUB | np.isin(FNAMES, ["D"]),
    "bursty":       _SUB | np.isin(FNAMES, ["D"]),
}
NONNEG = {"solar", "cumulative"}                     # keep zeros/monotonicity: no standardise/offset
COUNT = {"intermittent", "epidemic"}                 # integer valued

_u, _logu, _std, _ar1, _filt, _noise, _smooth_rw = S3._u, S3._logu, S3._std, S3._ar1, S3._filt, S3._noise, S3._smooth_rw


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def _cycle_idx(t, off, p):
    """index of the cycle (e.g. day) each time step falls in: [m, L] int."""
    return ((t[None, :] + off[:, None]) // p[:, None]).astype(np.int64)


def _per_cycle(rng, m, n_cyc, phi_lo=0.3, phi_hi=0.95):
    """AR(1) sequence over cycles (day-level effects), [m, n_cyc], unit-ish std."""
    phi = _u(rng, phi_lo, phi_hi, m)
    return _ar1(rng.normal(0, 1, (m, n_cyc)) * np.sqrt(1 - phi ** 2), phi)


def _markov_states(rng, m, n, K, p_sw):
    """sticky Markov chain over n steps with K[m,1] states and switch prob p_sw[m,1]: [m, n] int."""
    sw = rng.random((m, n)) < p_sw
    sw[:, 0] = False
    st = np.cumsum(sw * rng.integers(1, 3, (m, n)), 1)
    return (st + rng.integers(0, 3, (m, 1))) % K


# ---------------------------------------------------------------------------
# families: (rng, m, L, fi [m] freq index) -> (y [m, L] float64, period [m] int64)
# ---------------------------------------------------------------------------

def _longseason(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1, p2 = _P1[fi], _P2[fi]
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    a = S3._profile(rng, t, p1, off)
    b = S3._profile(rng, t, p1, off)
    lam = _sigmoid(1.5 * _smooth_rw(rng, m, L, 3) + _u(rng, -2, 2, m)) * (rng.random((m, 1)) < 0.7)
    s = _std((1 - lam) * a + lam * b)                                   # slowly drifting profile shape
    day = _cycle_idx(t, off, p1)
    d0 = day[:, :1]
    nd = int((day - d0).max()) + 1
    lvl = np.take_along_axis(_per_cycle(rng, m, nd), day - d0, 1)       # day-to-day level AR(1)
    wk = _SUB[fi] & (rng.random(m) < 0.8)                               # weekday factor, weekend dip
    fac = 1 + 0.12 * rng.normal(0, 1, (m, 7))
    fac[:, 5:] = np.where(rng.random((m, 1)) < 0.6, _u(rng, 0.3, 1.0, m), fac[:, 5:])
    dfac = np.where(wk[:, None], np.take_along_axis(fac, day % 7, 1), 1.0)
    amp = np.exp(0.3 * _smooth_rw(rng, m, L, 4))                        # amplitude drift
    a_s, a_l = _logu(rng, 0.5, 3, m), _logu(rng, 0.05, 0.6, m)
    tr = rng.normal(0, 0.5, (m, 1)) * t / L * (rng.random((m, 1)) < 0.4)
    add = a_s * amp * s * dfac + a_l * lvl + tr + (dfac - 1) * a_s
    mult = np.exp(0.4 * (add - add.mean(1, keepdims=True)) / (add.std(1, keepdims=True) + 1e-9))
    y = np.where(rng.random((m, 1)) < 0.35, mult, add)
    y = y + _noise(rng, m, L, _logu(rng, 0.02, 0.4, m) * y.std(1, keepdims=True))
    return y, p1.copy()


def _multiseas(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1, p2 = _P1[fi].astype(np.float64), _P2[fi]
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    y = _logu(rng, 0.5, 2, m) * S3._profile(rng, t, p1.astype(np.int64), off)
    # secondary: weekly / annual where defined, else a random longer integer period
    q = np.where(p2 > 0, p2, np.round(p1 * rng.uniform(2.5, 6, m)))
    K = rng.integers(1, 4, m)
    y += _logu(rng, 0.2, 1.2, m) * _std(S3._harmonics(rng, t, q, K, _u(rng, 0.5, 1.5, m), off))
    # harmonic sub-period (half day / half week) and an incommensurate non-integer period
    half = rng.random((m, 1)) < 0.4
    y += half * _logu(rng, 0.1, 0.6, m) * _std(S3._harmonics(rng, t, np.maximum(p1 / 2, 2), np.ones(m, int), _u(rng, 1, 1, m), off))
    inc = rng.random((m, 1)) < 0.5
    pin = p1 * rng.uniform(0.35, 0.8, m) + rng.uniform(0.01, 0.99, m)  # never an integer ratio
    y += inc * _logu(rng, 0.1, 0.8, m) * _std(S3._harmonics(rng, t, pin, np.ones(m, int), _u(rng, 1, 1, m), off))
    y += 0.3 * rng.normal(0, 1, (m, 1)) * _smooth_rw(rng, m, L, 4)
    y += _noise(rng, m, L, _logu(rng, 0.02, 0.3, m) * y.std(1, keepdims=True))
    return y, _P1[fi].copy()


def _solar(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1 = _P1[fi].astype(np.float64)
    sub = _SUB[fi]
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    y = np.zeros((m, L))
    if sub.any():
        r = np.where(sub)[0]
        k = len(r)
        pp = p1[r][:, None]
        ph = ((t[None, :] + off[r, None]) % pp) / pp                     # time of day in [0, 1)
        dayn = _cycle_idx(t, off[r], p1[r])
        yr = 2 * np.pi * dayn / 365.25 + rng.uniform(0, 2 * np.pi, (k, 1))
        half = _u(rng, 0.17, 0.3, k) + _u(rng, 0, 0.07, k) * np.sin(yr)   # half day length / day
        noon = 0.5 + _u(rng, -0.04, 0.04, k)
        x = (ph - noon + half) / (2 * half)
        bell = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0) ** _u(rng, 1, 2.2, k)
        d0 = dayn[:, :1]
        nd = int((dayn - d0).max()) + 1
        cl = _markov_states(rng, k, nd, np.full((k, 1), 2), _u(rng, 0.1, 0.5, k))
        dayf = np.where(cl == 0, _u(rng, 0.8, 1.0, k) + 0.05 * rng.normal(0, 1, (k, nd)),
                        _u(rng, 0.1, 0.6, k) * np.exp(0.3 * rng.normal(0, 1, (k, nd))))
        dayf = np.take_along_axis(np.clip(dayf, 0.02, 1.1), dayn - d0, 1)
        fast = np.minimum(np.exp(_ar1(rng.normal(0, 1, (k, L)) * _u(rng, 0.02, 0.25, k), _u(rng, 0.7, 0.97, k))), 1.0)
        y[r] = bell * dayf * fast * _logu(rng, 1, 1000, k)
    if (~sub).any():                                                    # daily totals
        r = np.where(~sub)[0]
        k = len(r)
        ann = 1 + _u(rng, 0.2, 0.8, k) * np.sin(2 * np.pi * (t[None, :] + off[r, None]) / 365.25)
        cl = _markov_states(rng, k, L, np.full((k, 1), 2), _u(rng, 0.1, 0.5, k))
        dayf = np.where(cl == 0, _u(rng, 0.8, 1.0, k), _u(rng, 0.1, 0.6, k) * np.exp(0.3 * rng.normal(0, 1, (k, L))))
        y[r] = ann * np.clip(dayf, 0.02, 1.1) * _logu(rng, 1, 1000, k)
    return y, np.where(sub, _P1[fi], 1).astype(np.int64)                # fev: solar daily m = 1


def _intermittent(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1 = _P1[fi]
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    has_s = (p1 > 1) & (p1 < L) & (rng.random(m) < 0.6)
    prof = S3._profile(rng, t, np.maximum(p1, 2), off)
    p0 = _logu(rng, 0.03, 0.6, m)
    logit = np.log(p0 / (1 - p0)) + has_s[:, None] * _u(rng, 0.5, 2.0, m) * prof
    life = rng.random((m, 1))                                             # 0.2 fade-out, 0.2 ramp-up
    tc, w = rng.uniform(0.2, 1.2, (m, 1)) * L, _logu(rng, 0.02, 0.2, m) * L
    logit += np.where(life < 0.2, -6 * _sigmoid((t - tc) / w), np.where(life < 0.4, -6 * _sigmoid(-(t - tc + 0.5 * L) / w), 0))
    occ = rng.random((m, L)) < _sigmoid(logit)
    mu = _logu(rng, 1, 50, m)
    lump = _logu(rng, 0.2, 3, m)                                          # NB dispersion: small = lumpy
    size = 1 + rng.poisson(np.minimum(mu * rng.gamma(np.broadcast_to(lump, (m, L)), 1) / lump, 1e6))
    return (occ * size).astype(np.float64), np.where(has_s, p1, 1).astype(np.int64)


def _waves(rng, m, L):
    """sum of 1-4 epidemic waves (logistic-derivative bumps), centres may lie beyond the window."""
    t = np.arange(L, dtype=np.float64)[None, :]
    y = np.zeros((m, L))
    nw = rng.integers(1, 5, (m, 1))
    for j in range(4):
        c = rng.uniform(-0.2, 1.3, (m, 1)) * L
        w = _logu(rng, 0.02, 0.25, m) * L
        z = (t - c) / w
        y += (j < nw) * _logu(rng, 0.1, 1, m) * np.exp(-z) / (1 + np.exp(-np.clip(z, -50, 50))) ** 2 * (np.abs(z) < 50)
    return y / (y.max(1, keepdims=True) + 1e-12)


def _epidemic(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    daily = np.array(FNAMES)[fi] == "D"
    lam = _waves(rng, m, L) * _logu(rng, 5, 5000, m) + _logu(rng, 0.1, 20, m)
    flu = (~daily) & (rng.random(m) < 0.5)                                # weekly: annual flu season
    lam = np.where(flu[:, None], lam * np.exp(_u(rng, 0.5, 2, m) * np.cos(2 * np.pi * (t + rng.integers(0, 52, (m, 1))) / 52.18)), lam)
    rep = daily & (rng.random(m) < 0.7)                                   # weekday reporting effect
    fac = np.exp(0.25 * rng.normal(0, 1, (m, 7)))
    fac[:, 5:] *= np.where(rng.random((m, 1)) < 0.5, _u(rng, 0.2, 0.8, m), 1)
    off = rng.integers(0, 7, (m, 1))
    lam = np.where(rep[:, None], lam * np.take_along_axis(fac, (t.astype(int)[None, :] + off) % 7, 1), lam)
    k = _logu(rng, 2, 50, m)
    y = rng.poisson(np.minimum(lam * rng.gamma(np.broadcast_to(k, (m, L)), 1) / k, 1e7))
    return y.astype(np.float64), np.where(rep, 7, 1).astype(np.int64)


def _cumulative(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    inc = np.where(rng.random((m, 1)) < 0.6, _waves(rng, m, L) * _logu(rng, 1, 1000, m),
                   _logu(rng, 0.1, 50, m) * np.exp(0.5 * _smooth_rw(rng, m, L, 4)))
    inc = rng.poisson(np.minimum(inc, 1e7)).astype(np.float64)
    daily = np.array(FNAMES)[fi] == "D"
    hold = daily & (rng.random(m) < 0.3)                                   # no reports on weekends:
    wk = ((t[None, :] + rng.integers(0, 7, (m, 1))) % 7) >= 5              # backlog lands on Monday
    y = np.cumsum(inc, 1) + _logu(rng, 1, 1e4, m) * (rng.random((m, 1)) < 0.5)
    if hold.any():
        yh = np.where(wk[hold], np.nan, y[hold])
        idx = np.where(~np.isnan(yh), np.arange(L)[None, :], 0)
        np.maximum.accumulate(idx, 1, out=idx)
        y[hold] = np.take_along_axis(np.where(np.isnan(yh), 0, yh), idx, 1)
    return y, np.ones(m, np.int64)


def _regseason(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1 = _P1[fi]
    cyc = np.where(_SUB[fi], p1, 7)                                        # regime per day or per week
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    profs = [S3._profile(rng, t, p1, off) for _ in range(3)]
    ci = _cycle_idx(t, off, cyc)
    c0 = ci[:, :1]
    nc = int((ci - c0).max()) + 1
    K = rng.integers(2, 4, (m, 1))
    st = np.take_along_axis(_markov_states(rng, m, nc, K, _logu(rng, 0.05, 0.4, m)), ci - c0, 1)
    amps = _logu(rng, 0.1, 2, m * 3).reshape(m, 3) * np.array([1, 1, 1])
    lv = rng.normal(0, 1, (m, 3)) * _logu(rng, 0.2, 2, m)
    lv[:, 0] = 0
    y = np.select([st == 0, st == 1], [profs[0], profs[1]], profs[2]) * np.take_along_axis(amps, st, 1)
    y = y + np.take_along_axis(lv, st, 1) + 0.3 * rng.normal(0, 1, (m, 1)) * _smooth_rw(rng, m, L, 3)
    y = y + _noise(rng, m, L, _logu(rng, 0.02, 0.3, m) * y.std(1, keepdims=True))
    return y, p1.copy()


def _bursty(rng, m, L, fi):
    t = np.arange(L, dtype=np.float64)
    p1 = _P1[fi]
    off = rng.integers(0, 1 << 14, m).astype(np.float64)
    base = 1 + _logu(rng, 0.05, 0.6, m) * S3._profile(rng, t, p1, off) * (rng.random((m, 1)) < 0.7)
    base = base + 0.2 * _smooth_rw(rng, m, L, 4) * (rng.random((m, 1)) < 0.5)
    # self-exciting bursts: immigrants, then one offspring generation through an exponential kernel
    k_dec = np.exp(-np.arange(L)[None, :] / _logu(rng, 1, 30, m))
    mu = _logu(rng, 0.002, 0.05, m)
    ev = (rng.random((m, L)) < mu).astype(np.float64)
    exc = _filt(ev, k_dec)
    ev2 = ev + (rng.random((m, L)) < np.clip(_u(rng, 0, 0.3, m) * exc, 0, 0.9))
    size = (rng.pareto(_u(rng, 1.2, 3, m), (m, L)) + 1) * _logu(rng, 0.2, 5, m)
    bursts = _filt(ev2 * size, k_dec)
    # counter-reset sawtooth: a cumulative counter that resets at random times
    rs = np.cumsum(rng.random((m, L)) < _logu(rng, 0.002, 0.03, m), 1)
    cnt = np.cumsum(np.abs(base) * _logu(rng, 0.01, 0.2, m), 1)
    first = np.zeros_like(cnt)
    chg = np.diff(rs, axis=1, prepend=rs[:, :1]) > 0
    first = np.where(chg, cnt, np.nan); first[:, 0] = 0
    idx = np.where(~np.isnan(first), np.arange(L)[None, :], 0); np.maximum.accumulate(idx, 1, out=idx)
    saw = cnt - np.take_along_axis(np.nan_to_num(first), idx, 1)
    # plateaus: random step levels (deploys / capacity changes)
    steps = np.cumsum((rng.random((m, L)) < _logu(rng, 0.002, 0.02, m)) * rng.normal(0, 1, (m, L)), 1)
    kind = rng.random((m, 1))
    y = np.where(kind < 0.6, base + bursts, np.where(kind < 0.8, saw + 0.1 * bursts, base + 0.5 * steps + 0.3 * bursts))
    y = y + _noise(rng, m, L, _logu(rng, 0.005, 0.15, m) * y.std(1, keepdims=True))
    has_s = (kind[:, 0] < 0.6) | (kind[:, 0] >= 0.8)
    return y, np.where(has_s, p1, 1).astype(np.int64)


_GEN = {"longseason": _longseason, "multiseas": _multiseas, "solar": _solar, "intermittent": _intermittent,
        "cumulative": _cumulative, "epidemic": _epidemic, "regseason": _regseason, "bursty": _bursty}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _freq_for(rng, f, c, fw):
    w = np.where(_ALLOWED[FAMILIES[f]], fw, 0.0)
    w = w / w.sum() if w.sum() > 0 else _ALLOWED[FAMILIES[f]] / _ALLOWED[FAMILIES[f]].sum()
    return rng.choice(len(FREQS), c, p=w)


def _make(rng, f, c, L, fw):
    name = FAMILIES[f]
    fi = _freq_for(rng, f, c, fw)
    y, per = _GEN[name](rng, c, L, fi)
    bad = ~np.isfinite(y).all(1)
    if bad.any():
        y[bad] = rng.normal(0, 1, (int(bad.sum()), L)); per[bad] = 1
    if name in NONNEG:                       # synth3 post-processing in its count mode: missing only
        y = S3._postprocess(rng, y, np.ones(c, bool)) * np.exp(rng.normal(0, 1.5, (c, 1)))
    else:
        y = S3._postprocess(rng, y, np.full(c, name in COUNT))
    per = np.where(per >= L, 1, per)
    return y.astype(np.float32), per.astype(np.int64), _P1[fi]


POOL_ROWS = 256
_POOL = {}


def _take(rng, f, c, L, fw):
    key = (f, L, fw.tobytes())
    buf = _POOL.get(key)
    if buf is None or buf[3] + c > len(buf[0]):
        if len(_POOL) > 4 * len(FAMILIES):
            _POOL.clear()
        buf = _POOL[key] = [*_make(rng, f, max(POOL_ROWS, c), L, fw), 0]
    i = buf[3]
    buf[3] = i + c
    return buf[0][i:i + c], buf[1][i:i + c], buf[2][i:i + c]


def generate(rng, n, length, freq_weights=None, weights=None, short_frac=0.25, min_len=16,
             min_cycles=2, return_family=False, pool=True):
    """n rows of `length` points (right-aligned; NaN = missing or left pad), period int64 [n].
    freq_weights: weights over FREQS (default DEFAULT_FW); weights: over FAMILIES (default FAM_W
    jittered by Dirichlet(20)). Rows that get left-padded keep at least min_cycles full cycles of
    the frequency's primary period when the window allows it, so long seasons stay visible."""
    L = int(length)
    fw = DEFAULT_FW if freq_weights is None else np.asarray(freq_weights, float) / np.sum(freq_weights)
    y = np.zeros((n, L), np.float32)
    per = np.ones(n, np.int64)
    p1 = np.ones(n, np.int64)
    fam = np.zeros(n, np.int64)
    if n == 0:
        return (y, per, fam) if return_family else (y, per)
    w = rng.dirichlet(20 * FAM_W) if weights is None else np.asarray(weights, float) / np.sum(weights)
    counts = rng.multinomial(n, w)
    i = 0
    for f, c in enumerate(counts):
        if c:
            y[i:i + c], per[i:i + c], p1[i:i + c] = _take(rng, f, c, L, fw) if pool else _make(rng, f, c, L, fw)
            fam[i:i + c] = f
            i += c
    lo = np.minimum(np.maximum(min_len, min_cycles * p1 + 1), L)
    lens = np.where(rng.random(n) < short_frac, rng.integers(lo, L + 1), L)
    y[np.arange(L)[None, :] < (L - lens)[:, None]] = np.nan
    o = rng.permutation(n)
    out = (y[o], per[o])
    return out + (fam[o],) if return_family else out


def gap_weights(real_share, syn_frac, target=None, floor=0.02):
    """Synthetic frequency weights that complement the real rows: per frequency f,
    max(target_f - (1 - syn_frac) * real_share_f, 0) / syn_frac, plus a small floor so every
    frequency keeps some synthetic support. real_share: {freq name: share of real rows}."""
    target = TARGET if target is None else target
    g = np.array([max(target[f] - (1 - syn_frac) * real_share.get(f, 0.0), 0.0) for f in FNAMES])
    g = g / max(syn_frac, 1e-9) + floor
    return g / g.sum()


def generate_sdd(rng, n, length, freq_weights=None, short_frac=0.25, min_len=16, min_cycles=2):
    """Rows with an EXACT conditional law for Synthetic Data Distillation (lit_review #5, 2609.09586):
        y_t = g_t + e_t,  e_t = phi e_{t-1} + sig * eps_t,  eps ~ N(0, 1) iid (stationary start)
    with g a known deterministic signal (frequency-conditioned seasonal profile x weekday factor +
    linear/logistic trend, all drawn up front) and (phi, sig) known. Given the parameters and the
    history up to origin t, y_{t+h} ~ N(g_{t+h} + phi^h (y_t - g_t), sig^2 (1 - phi^2h) / (1 - phi^2)).
    No post-processing other than an affine map (exact) and left padding (does not change the law).
    Returns y float32 [n, L] (NaN = pad), period int64 [n], g float32 [n, L], phi float32 [n],
    sig float32 [n]. Note: the law conditions on the generator parameters (as SDD does); a forecaster
    must also infer them, so the distilled target is slightly sharper than the Bayes predictive at
    short contexts."""
    L = int(length)
    fw = DEFAULT_FW if freq_weights is None else np.asarray(freq_weights, float) / np.sum(freq_weights)
    fi = rng.choice(len(FREQS), n, p=fw)
    t = np.arange(L, dtype=np.float64)
    p1 = _P1[fi]
    off = rng.integers(0, 1 << 14, n).astype(np.float64)
    s = S3._profile(rng, t, np.maximum(p1, 2), off)
    has_s = (p1 > 1) & (rng.random(n) < 0.85)
    day = _cycle_idx(t, off, np.maximum(p1, 1))
    fac = 1 + 0.15 * rng.normal(0, 1, (n, 7))
    dfac = np.where((_SUB[fi] & (rng.random(n) < 0.6))[:, None], np.take_along_axis(fac, day % 7, 1), 1.0)
    tr = np.where(rng.random((n, 1)) < 0.5, rng.normal(0, 1, (n, 1)) * t / L,
                  rng.normal(0, 1, (n, 1)) / (1 + np.exp(-np.clip(_logu(rng, 2, 20, n) / L * (t - rng.uniform(0, 1.2, (n, 1)) * L), -50, 50))))
    g = has_s[:, None] * _logu(rng, 0.3, 2, n) * s * dfac + tr
    phi = rng.uniform(0, 0.97, n)
    sig = _logu(rng, 0.03, 0.6, n)[:, 0]
    e = S3._ar1(rng.normal(0, 1, (n, L)) * sig[:, None], phi[:, None])
    e += (rng.normal(0, 1, n) * sig / np.sqrt(1 - phi ** 2))[:, None] * phi[:, None] ** t[None, :]   # stationary start
    y = g + e
    sc = np.exp(rng.normal(0, 2, n))[:, None]
    of = rng.normal(0, 3, n)[:, None] * (rng.random((n, 1)) < 0.5)
    y, g, sig = (y + of) * sc, (g + of) * sc, sig * sc[:, 0]
    lo = np.minimum(np.maximum(min_len, min_cycles * p1 + 1), L)
    lens = np.where(rng.random(n) < short_frac, rng.integers(lo, L + 1), L)
    y[np.arange(L)[None, :] < (L - lens)[:, None]] = np.nan
    per = np.where(has_s & (p1 < L), p1, 1).astype(np.int64)
    return y.astype(np.float32), per, g.astype(np.float32), phi.astype(np.float32), sig.astype(np.float32)


def stretch(y, k, rng):
    """Causal k-times upsampling of a real series y [n] (NaN-aware) into a series of k*(n-1) points.
    Point t (in new time) lies in original interval j = t // k + 1 and linearly interpolates between
    y[j-1] and y[j] with weight (t % k) / k. The value at new time t therefore depends on original
    points up to index j-1 + (t % k > 0); in new time those lie at or before t, so no future value
    enters any point (it is a delayed ramp, not centred interpolation). A small AR(1) innovation
    with scale proportional to the CAUSAL running mean |diff| is added so the fine scale is not
    piecewise linear. Returns float32 [k * (n - 1)]."""
    y = np.asarray(y, np.float64)
    n = len(y)
    if n < 3:
        return y.astype(np.float32)
    t = np.arange(k * (n - 1))
    j0, fr = t // k, (t % k) / k
    a, b = y[j0], y[j0 + 1]
    # delayed by one original step: the ramp toward y[j0+1] is emitted during (j0+1)k .. (j0+2)k
    z = np.r_[np.full(k, np.nan), (a + fr * (b - a))[:-k]]
    d = np.abs(np.diff(y))
    dd = np.where(np.isfinite(d), d, 0)
    cum = np.cumsum(dd) / np.maximum(np.cumsum(np.isfinite(d)), 1)     # running mean |diff| up to j
    # cum[j] involves y[j+1], whose ramp is emitted from new time (j+1)k + 1: delay by 2k
    sc = np.r_[np.full(2 * k, np.nan), np.repeat(cum, k)][:len(z)]
    phi = rng.uniform(0.3, 0.9)
    e = S3._ar1(rng.normal(0, 1, (1, len(z))), np.array([[phi]]))[0] * np.sqrt(1 - phi ** 2)
    z = z + rng.uniform(0.05, 0.4) * np.nan_to_num(sc) * e / np.sqrt(k)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# diagnostics: uv run --no-project --python 3.12 --with matplotlib --with numpy --with pandas --with pyarrow python tools/synth4.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from collections import Counter

    rng = np.random.default_rng(0)
    L = 320
    for _ in range(3):
        generate(rng, 64, L)
    tc = []
    for _ in range(50):
        c0 = time.process_time(); generate(rng, 64, L); tc.append(time.process_time() - c0)
    print(f"generate(64, {L}) pooled: mean CPU {1e3 * np.mean(tc):.1f} ms/call (incl. refills); "
          f"load avg {os.getloadavg()[0]:.0f}")
    for i, f in enumerate(FAMILIES):
        c0 = time.process_time(); _make(rng, i, 256, L, DEFAULT_FW)
        print(f"   {f:12s} {1e3 * (time.process_time() - c0) / 2:6.1f} ms CPU per 128 rows")
    y, per, fam = generate(np.random.default_rng(1), 4096, L, return_family=True)
    print("family shares:", {f: round(float(np.mean(fam == i)), 3) for i, f in enumerate(FAMILIES)})
    print(f"finite-or-NaN: {bool(np.all(np.isfinite(y) | np.isnan(y)))}  NaN frac {np.isnan(y).mean():.3f}  "
          f"zero frac {np.mean(y == 0):.3f}")
    c = Counter(per.tolist())
    print("period dist:", ", ".join(f"{k}:{v / len(per):.3f}" for k, v in sorted(c.items(), key=lambda z: -z[1])[:12]))
    vis = np.array([np.isfinite(r).sum() for r in y])
    for p in (96, 144, 288):
        r = per == p
        if r.any():
            print(f"  period {p}: {r.mean():.3f} of rows, median observed {np.median(vis[r]):.0f} points "
                  f"({np.median(vis[r]) / p:.2f} cycles)")
    yc, pc = generate(np.random.default_rng(2), 2048, 640, short_frac=0.0)
    (s1, h1), (s0, h0), ns, nn = S3._spectral_check(yc, pc)
    print(f"spectral check (L=640): harmonic power share {s1:.2f} (argmax hit {h1:.2f}, n={ns}) vs null {s0:.2f} ({h0:.2f}, n={nn})")
    z = stretch(np.sin(np.arange(200) * 2 * np.pi / 24), 4, rng)
    zz = z[np.isfinite(z)]
    print(f"stretch sanity: len {len(z)}, lag-96 autocorr {np.corrcoef(zz[:-96], zz[96:])[0, 1]:.3f} (expect ~1)")
    # causality test: changing y[j] must not change stretch output before new time (j)*k
    y0 = rng.normal(0, 1, 50); y1 = y0.copy(); y1[30] += 10
    s0_, s1_ = stretch(y0, 4, np.random.default_rng(9)), stretch(y1, 4, np.random.default_rng(9))
    first = np.where(~np.isclose(np.nan_to_num(s0_), np.nan_to_num(s1_)))[0][0]
    print(f"stretch causality: perturbing original index 30 first changes new index {first} "
          f"(the ramp toward y[30] starts after new time {30 * 4}; must be > it)")
    assert first > 30 * 4

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("no matplotlib: PNG skipped")
    g = np.random.default_rng(4)
    rows = []
    for i, f in enumerate(FAMILIES):
        yy, pp, ff = generate(g, 3, 640, weights=np.eye(len(FAMILIES))[i], short_frac=0.1, return_family=True, pool=False)
        rows += [(f, yy[j], pp[j]) for j in range(3)]
    stretched = []
    try:
        import pandas as pd
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_parquet(os.path.join(root, "data", "raw", "monash_traffic.parquet"))
        col = "target" if "target" in df.columns else [c for c in df.columns if df[c].dtype == object][0]
        for kk, sidx in ((4, 3), (6, 100), (12, 400)):
            src = np.asarray(df[col].iloc[sidx], np.float64)[5000:5000 + 640 // kk + 2]
            stretched.append((f"traffic H x{kk} (period {24 * kk})", stretch(src, kk, g)[-640:], 24 * kk))
    except Exception as e:  # noqa: BLE001
        print("stretch demo skipped:", e)
    rows += stretched
    nr = int(np.ceil(len(rows) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(18, 2.0 * nr))
    for ax, (f, v, p) in zip(axes.flat, rows):
        ax.plot(v, lw=0.6)
        ax.set_title(f"{f}  period={p}", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "synth4_examples.png")
    fig.savefig(out, dpi=70)
    print("saved", out)
