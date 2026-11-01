"""
synth5: "world prior" -- series emitted by simple latent dynamical systems, so the world model sees
state dynamics (regimes, attractors, latent intensities, volatility states), not only curve shapes.

    generate(rng, n, length, weights=None, short_frac=0.25, min_len=16, min_cycles=2,
             return_family=False, pool=True) -> y float32 [n, length] (NaN = missing / left pad),
                                                period int64 [n] (dominant integer season, 1 = none)

Families (one row = one family; weights FAM_W jittered by Dirichlet(20) per call), by expected value:
    structural  daily period P in {24,48,96,144,288} x weekly 7P: weekday/weekend profiles (Fourier
                or commute double-peak), per-day level AR(1) x slow amplitude modulation, holidays,
                level shifts, slow trend, AR(1) noise (homo- or Poisson-like heteroscedastic,
                heavy-tailed for some rows), outages (drop to ~0, stuck sensor, or NaN gap).
                Non-negative, traffic / load / sensor-like.
    regime      Markov-switching AR(2) (2-3 sticky regimes, each with its own mean, AR roots incl.
                oscillatory pairs, volatility and seasonal amplitude) or endogenous SETAR switching.
    chaos       Lorenz-63 (RK4; observe x, z or x+y), damped driven pendulum (forcing period = P
                samples; chaotic for part of the (gamma, A) range), coupled van der Pol pair,
                noisy Kuramoto phase oscillators; plus observation noise.
    count       Croston-type intermittent demand (seasonal occurrence probability, NB / lognormal
                sizes) or latent-state Poisson / NB (log-intensity = level + season + weekday +
                AR(1) latent state + slow trend).
    hetero      GARCH(1,1) with heavy-tailed innovations: returns, geometric prices, or noise around
                a seasonal AR mean.

Speed: numpy only; vectorised over rows; recursions are a Python loop over TIME with row-vector ops;
daily profiles are evaluated on a 64-point grid and linearly interpolated; rows are served from
per-family pools (each row used once), refilled with the caller's rng (so seeded and reproducible).

Long series: every row is a crop of a process with a random calendar origin (phase of day, weekday,
holidays), and `length` can be anything, so TSAR_PREFIX + TSAR_PREFIX_SYN runs get >= 2048-point
histories holding whole weekly cycles of 5-minute data.

Leakage: NO parameter here is fitted to any data and the module reads no file. Periods are calendar
constants (samples per day at 1h/30/15/10/5 min, 7 days, 12 months, 4 quarters); every other range
is a hand-set prior from textbook values (Lorenz rho near 28, chaotic pendulum gamma~0.5 A~1.15,
GARCH alpha+beta<1, Croston p in [0.02, 0.6]). The only data-informed choice is the frequency mix
(heavy on sub-hourly), motivated by TRAIN-split frequency counts (no 5/10/15-min real rows; ~53% of
real rows monthly/quarterly; see tools/synth4.py), never by VAL/TEST series or any model's outputs.
"""
import numpy as np

FAMILIES = ("structural", "regime", "chaos", "count", "hetero")
FAM_W = np.array([0.35, 0.18, 0.15, 0.17, 0.15])

DAILY = np.array([24, 48, 96, 144, 288])                 # samples per day: 1h, 30m, 15m, 10m, 5m
DAILY_W = np.array([0.25, 0.10, 0.20, 0.15, 0.30])
GEN_P = np.array([1, 4, 7, 12, 24, 48, 96, 144, 288])    # periods for the other families (1 = none)
GEN_W = np.array([0.18, 0.04, 0.10, 0.08, 0.18, 0.06, 0.13, 0.09, 0.14])
G = 96                                                   # profile grid points per cycle
POOL_ELEMS = 1 << 18                                     # pool size per family refill, in points
_POOL = {}


# ----------------------------------------------------------------------------- helpers
def _u(rng, lo, hi, m):
    return rng.uniform(lo, hi, (m, 1))


def _logu(rng, lo, hi, m):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), (m, 1)))


def _std(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


def _ar1(e, phi):
    """x_t = phi x_{t-1} + e_t along axis 1, phi [m,1]; stationary start. Loop over time only."""
    eT = np.ascontiguousarray(e.T)
    ph = phi[:, 0]
    eT[0] /= np.sqrt(np.maximum(1 - ph ** 2, 1e-3))
    for i in range(1, len(eT)):
        eT[i] += ph * eT[i - 1]
    return eT.T


def _noise(rng, m, L, p_heavy=0.4):
    """Unit-scale noise; a p_heavy share of rows gets heavy tails (lognormal scale mixture of normals,
    excess kurtosis ~3..50), the rest Gaussian."""
    z = rng.standard_normal((m, L))
    h = np.nonzero(rng.random(m) < p_heavy)[0]
    if len(h):
        s = rng.choice([0.5, 0.8, 1.1], (len(h), 1))
        z[h] *= np.exp(s * rng.standard_normal((len(h), L)) - s ** 2)
        z[h] /= z[h].std(1, keepdims=True) + 1e-8
    return z


_KB = 16                                                 # profile harmonics (bumps need ~16)
_BASIS = np.concatenate([np.cos(2 * np.pi * np.arange(1, _KB + 1)[:, None] * np.arange(G)[None, :] / G),
                         np.sin(2 * np.pi * np.arange(1, _KB + 1)[:, None] * np.arange(G)[None, :] / G)])


def _grid_profile(rng, m, kmax=8):
    """Random cycle shape on G grid points [m,G], built from Fourier coefficients (one matmul): random
    harmonics with decaying amplitudes, or a commute double peak (two circular Gaussian bumps, whose
    Fourier coefficients are exp(-2 pi^2 k^2 w^2) e^{-2 pi i k c}), or a mix. Zero mean, unit std."""
    k = np.arange(1, _KB + 1)[None, :]
    amp = rng.standard_normal((m, _KB)) * k ** -_u(rng, 0.6, 2.0, m) * (k <= rng.integers(2, kmax + 1, (m, 1)))
    phs = rng.uniform(0, 2 * np.pi, (m, _KB))
    fc, fs = amp * np.cos(phs), -amp * np.sin(phs)
    bc = np.zeros((m, _KB)); bs = np.zeros((m, _KB))
    for c_lo, c_hi in ((0.25, 0.40), (0.62, 0.80)):
        c, w = _u(rng, c_lo, c_hi, m), _u(rng, 0.025, 0.09, m)
        g = _u(rng, 0.4, 1.0, m) * w * np.exp(-2 * np.pi ** 2 * k ** 2 * w ** 2)
        bc += g * np.cos(2 * np.pi * k * c); bs += g * np.sin(2 * np.pi * k * c)
    nf = np.sqrt((fc ** 2 + fs ** 2).sum(1, keepdims=True)) + 1e-12
    nb = np.sqrt((bc ** 2 + bs ** 2).sum(1, keepdims=True)) + 1e-12
    r = rng.random((m, 1))
    w = np.where(r < 0.4, 0.0, np.where(r < 0.7, 1.0, rng.uniform(0, 1, (m, 1))))
    coef = np.concatenate([w * bc / nb + (1 - w) * fc / nf, w * bs / nb + (1 - w) * fs / nf], 1)
    return _std(coef @ _BASIS)


def _lookup(grid, ph, which=None):
    """Linear interpolation of grid [m, k*G] at phase ph in [0,1) [m,L]; `which` [m,L] in 0..k-1
    selects the sub-grid (e.g. weekday / weekend profile). Flat takes (faster than take_along_axis)."""
    pos = ph * G
    i0 = pos.astype(np.int64)
    w = pos - i0
    i1 = i0 + 1
    i1[i1 == G] = 0
    base = (np.arange(len(grid)) * grid.shape[1])[:, None]
    if which is not None:
        base = base + which * G
    flat = grid.ravel()
    a = flat.take(i0 + base)
    return a + (flat.take(i1 + base) - a) * w


def _calendar(rng, m, L, P):
    """Random time origin for period P [m,1]: day index relative to the first day [m,L] (int64),
    absolute day [m,L], phase in [0,1) [m,L]. Float arithmetic (exact: q + 0.5 is never a multiple)."""
    o = rng.integers(0, 1 << 20, (m, 1)) % P
    q = np.arange(L, dtype=np.float64)[None, :] + o
    drel = np.floor((q + 0.5) / P)
    ph = (q - drel * P) / P
    drel = drel.astype(np.int64)
    return drel, drel + rng.integers(0, 7 * 400, (m, 1)), ph


def _pick_p(rng, m, vals, w):
    return rng.choice(vals, (m, 1), p=w / w.sum())


def _events(rng, m, L, rate):
    """Sparse event positions: per-row Poisson(rate[m] * L) count, uniform positions. -> rows, cols."""
    k = rng.poisson(rate * L)
    r = np.repeat(np.arange(m), k)
    return r, rng.integers(0, L, len(r))


def _season(rng, m, L, P, kmax=4):
    """Unit seasonal signal with period P [m,1] (0 where P == 1) plus the absolute day index."""
    drel, day, ph = _calendar(rng, m, L, np.maximum(P, 1))
    return np.where(P > 1, _lookup(_grid_profile(rng, m, kmax), ph), 0.0), day


# ----------------------------------------------------------------------------- families
def _structural(rng, m, L):
    P = _pick_p(rng, m, DAILY, DAILY_W)
    drel, day, ph = _calendar(rng, m, L, P)
    D = int(drel.max()) + 1
    a = _u(rng, 0.2, 1.0, m)
    wd, we = _grid_profile(rng, m), _grid_profile(rng, m)
    grid = np.concatenate([wd, a * we + (1 - a) * wd], 1)            # weekend = partial reshape
    dd = np.arange(D)[None, :] + day[:, :1]                          # absolute day per day slot
    weekend = (((dd + rng.integers(0, 7, (m, 1))) % 7) >= 5) & (rng.random((m, 1)) < 0.8)
    off = weekend | (rng.random((m, D)) < _u(rng, 0, 0.04, m))      # weekends + holidays
    # per-day factor: AR(1) log level x off-day level x slow amplitude modulation (3..60 day period)
    dl = _ar1(rng.standard_normal((m, D)) * _logu(rng, 0.01, 0.15, m), _u(rng, 0.3, 0.95, m))
    dfac = np.exp(dl) * np.where(off, _u(rng, 0.45, 1.1, m), 1.0)
    damp = 1 + _u(rng, 0, 0.5, m) * np.sin(2 * np.pi * dd / _u(rng, 3, 60, m) + _u(rng, 0, 6.3, m))
    di = drel + (np.arange(m) * D)[:, None]                           # flat per-day index
    A = _u(rng, 0.25, 0.95, m) / (np.abs(grid).max(1, keepdims=True) + 1e-8)
    lvl = dfac.ravel().take(di)
    prof = _lookup(grid, ph, off.ravel().take(di).astype(np.int64))
    pn = prof * (A * damp).ravel().take(di)
    # level shifts (sparse) + slow trend
    inc = np.zeros((m, L))
    r, c = _events(rng, m, L, _logu(rng, 0.05, 3, m)[:, 0] / L * (rng.random(m) < 0.6))
    np.add.at(inc, (r, c), rng.normal(0, 0.3, len(r)))
    inc[:, 1:] += rng.normal(0, 0.3, (m, 1)) / L
    lev = np.exp(np.cumsum(inc, 1)) * lvl
    mult = rng.random((m, 1)) < 0.75
    y = np.where(mult, lev * (1 + pn), lev + pn)
    # noise: AR(1), homoscedastic or Poisson-like, Gaussian or heavy-tailed
    phi = _u(rng, 0, 0.9, m)
    e = _ar1(_noise(rng, m, L) * np.sqrt(1 - phi ** 2), phi)
    sc = np.where(rng.random((m, 1)) < 0.5, np.sqrt(np.maximum(y, 0) + 0.05), 1.0) * _logu(rng, 0.01, 0.25, m)
    y = np.maximum(y + sc * e, 0.0) * _logu(rng, 0.1, 1e4, m)
    # outages (30% of rows): drop to ~0, stuck sensor, or missing (NaN); 1 h .. 12 h long
    r, c = _events(rng, m, L, _logu(rng, 0.2, 3, m)[:, 0] / L * (rng.random(m) < 0.3))
    if len(r):
        u, r = np.unique(r, return_inverse=True)                         # outage rows only
        Pr = P[u[r], 0]
        d = np.exp(rng.uniform(np.log(np.maximum(Pr / 24, 1)), np.log(Pr / 2))).astype(np.int64) + 1
        cnt = np.zeros((len(u), L + 1), np.int64)
        np.add.at(cnt, (r, c), 1); np.add.at(cnt, (r, np.minimum(c + d, L)), -1)
        o = np.cumsum(cnt[:, :L], 1) > 0
        kind = rng.integers(0, 4, (len(u), 1))                          # 0,1 -> ~0; 2 -> stuck; 3 -> NaN
        yu = y[u]
        idx = np.maximum.accumulate(np.where(o, 0, np.arange(L)[None, :]), 1)
        y[u] = np.where(o & (kind <= 1), yu * _u(rng, 0, 0.05, len(u)),
               np.where(o & (kind == 2), np.take_along_axis(yu, idx, 1),
               np.where(o & (kind == 3), np.nan, yu)))
    return y, P[:, 0]


def _markov_path(rng, m, L, K, stay):
    """Sticky Markov state path [m,L] without a time loop: segment states (K=2 alternate, K=3 jump to
    one of the other two), geometric sojourns with the state's stay probability, change points
    scattered with add.at and a cumsum. stay [m,Km]; K [m]."""
    S = max(8, int(4 * L * (1 - stay.min(1)).max()) + 8)               # segments: enough to cover L
    S = min(S, L + 1)
    seg = np.empty((m, S), np.int64)
    seg[:, 0] = rng.integers(0, K)
    step = 1 + (rng.random((m, S)) < 0.5) * (K[:, None] == 3)
    seg[:, 1:] = (seg[:, :1] + np.cumsum(step[:, 1:], 1)) % K[:, None]
    q = 1 - np.take_along_axis(stay, seg, 1)
    dur = rng.geometric(np.clip(q, 1e-6, 1))
    ends = np.cumsum(dur, 1)
    r, c = np.nonzero(ends[:, :-1] < L)
    cnt = np.zeros((m, L), np.int64)
    cnt[r, ends[r, c]] = 1                                              # ends strictly increase per row
    return np.take_along_axis(seg, np.minimum(np.cumsum(cnt, 1), S - 1), 1)


def _regime(rng, m, L):
    """Regime-switching AR(2): x_t = mu_s + p1_s (x_{t-1} - mu_s) + p2_s (x_{t-2} - mu_s) + sig_s e_t,
    with s a sticky Markov chain (precomputed path, then a 4-op time loop) or, for SETAR rows,
    s_t = [x_{t-1} > c] (endogenous; small separate loop). Regime-dependent seasonal amplitude."""
    P = _pick_p(rng, m, GEN_P, GEN_W)
    K = rng.integers(2, 4, m)
    Km = 3
    mu = rng.normal(0, 1.5, (m, Km))
    r = rng.uniform(0, 0.985, (m, Km))                                # AR(2) roots per regime:
    th = np.where(rng.random((m, Km)) < 0.5, 0.0, rng.uniform(0.05, np.pi, (m, Km)))   # real or complex
    p1 = np.where(th > 0, 2 * r * np.cos(th), r)
    p2 = np.where(th > 0, -r ** 2, 0.0)
    sig = np.exp(rng.uniform(np.log(0.05), np.log(1.0), (m, Km)))
    amp = rng.uniform(0, 1.5, (m, Km)) * (rng.random((m, 1)) < 0.6)
    stay = 1 - 1 / np.exp(rng.uniform(np.log(8), np.log(5 * P + 60), (m, Km)))   # mean duration 8..5P+60
    thr = (rng.random(m) < 0.3) & (K == 2)                                         # SETAR rows
    a0 = mu * (1 - p1 - p2)                                            # intercept form
    S = _markov_path(rng, m, L, K, stay)
    fi = (S + (np.arange(m) * Km)[:, None]).T                          # [L,m] flat parameter index
    E = rng.standard_normal((L, m))
    A_, P1_, P2_ = a0.ravel().take(fi), p1.ravel().take(fi), p2.ravel().take(fi)
    N_ = sig.ravel().take(fi) * E
    x1 = mu[np.arange(m), S[:, 0]].copy(); x2 = x1.copy()
    X = np.empty((L, m))
    for t in range(L):
        x = A_[t] + P1_[t] * x1 + P2_[t] * x2 + N_[t]
        x2 = x1; x1 = x
        X[t] = x
    X = X.T
    i = np.nonzero(thr)[0]
    if len(i):                                                         # SETAR: state from x_{t-1}
        c = rng.normal(0, 0.5, len(i)) + 0.5 * (mu[i, 0] + mu[i, 1])
        a, b1, b2, sg = a0[i, :2], p1[i, :2], p2[i, :2], sig[i, :2]
        x1 = mu[i, 0].copy(); x2 = x1.copy(); Ei = E[:, i]
        Xi = np.empty((L, len(i))); Si = np.empty((L, len(i)), np.int64)
        for t in range(L):
            hi = x1 > c
            x = np.where(hi, a[:, 1] + b1[:, 1] * x1 + b2[:, 1] * x2 + sg[:, 1] * Ei[t],
                         a[:, 0] + b1[:, 0] * x1 + b2[:, 0] * x2 + sg[:, 0] * Ei[t])
            x2 = x1; x1 = x
            Xi[t] = x; Si[t] = hi
        X[i] = Xi.T; S[i] = Si.T
    y = X
    j = np.nonzero(amp.max(1) > 0)[0]
    if len(j):
        seas, _ = _season(rng, len(j), L, P[j])
        y[j] += np.take_along_axis(amp[j], S[j], 1) * seas * (3 * sig[j].mean(1, keepdims=True))
    obs = E.T[rng.permutation(m)]                                      # other rows' innovations: independent
    y = y + obs * _logu(rng, 1e-3, 0.1, m) * (y.std(1, keepdims=True) + 1e-8)
    return y, np.where(amp.max(1) > 0, P[:, 0], 1)


def _lorenz(rng, n, L):
    rho = rng.uniform(24, 40, n); b = 8 / 3; sg = 10.0
    dt = rng.uniform(0.01, 0.04, n); h2 = dt / 2; h6 = dt / 6
    x, y, z = rng.normal(0, 8, n), rng.normal(0, 8, n), rng.uniform(15, 35, n)
    obs = rng.integers(0, 3, n)
    out = np.empty((L, n))
    for t in range(-100, L):                                          # 100 burn-in steps
        k1x = sg * (y - x); k1y = x * (rho - z) - y; k1z = x * y - b * z
        ax, ay, az = x + h2 * k1x, y + h2 * k1y, z + h2 * k1z
        k2x = sg * (ay - ax); k2y = ax * (rho - az) - ay; k2z = ax * ay - b * az
        ax, ay, az = x + h2 * k2x, y + h2 * k2y, z + h2 * k2z
        k3x = sg * (ay - ax); k3y = ax * (rho - az) - ay; k3z = ax * ay - b * az
        ax, ay, az = x + dt * k3x, y + dt * k3y, z + dt * k3z
        k4x = sg * (ay - ax); k4y = ax * (rho - az) - ay; k4z = ax * ay - b * az
        x = x + h6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + h6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        z = z + h6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        if t >= 0:
            out[t] = np.where(obs == 0, x, np.where(obs == 1, z, x + y))
    return out.T, np.ones(n, np.int64)


def _pendulum(rng, n, L):
    """theta'' = -gamma theta' - sin theta + A cos(w t); one forcing cycle = P samples (exact)."""
    P = rng.choice(DAILY, n, p=DAILY_W / DAILY_W.sum())
    P = np.where(rng.random(n) < 0.3, rng.choice([7, 12, 24], n), P)
    w = 2 / 3; dt = (2 * np.pi / w) / P; h2 = dt / 2; h6 = dt / 6
    gam = rng.uniform(0.2, 0.7, n); A = rng.uniform(0.4, 1.5, n)
    th, om = rng.uniform(-np.pi, np.pi, n), rng.normal(0, 1, n)
    ph = rng.integers(0, 1 << 20, n) % P * (2 * np.pi / P)          # drive phase, in whole samples
    obs = rng.integers(0, 3, n)
    out = np.empty((L, n))
    for t in range(L):
        f0 = A * np.cos(ph); fh = A * np.cos(ph + w * h2); f1 = A * np.cos(ph + w * dt)
        k1t = om; k1o = -gam * om - np.sin(th) + f0
        k2t = om + h2 * k1o; k2o = -gam * k2t - np.sin(th + h2 * k1t) + fh
        k3t = om + h2 * k2o; k3o = -gam * k3t - np.sin(th + h2 * k2t) + fh
        k4t = om + dt * k3o; k4o = -gam * k4t - np.sin(th + dt * k3t) + f1
        th = th + h6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om = om + h6 * (k1o + 2 * k2o + 2 * k3o + k4o)
        ph = ph + w * dt
        out[t] = np.where(obs == 0, om, np.where(obs == 1, np.sin(th), np.cos(th)))
    return out.T, P.astype(np.int64)


def _vdp(rng, n, L):
    """Two coupled van der Pol oscillators (RK4, 48 or 96 steps per unit-frequency cycle); observe x1 + c x2."""
    mu = rng.uniform(0.1, 2.5, n); w2 = rng.uniform(0.6, 1.6, n); kap = rng.uniform(0, 0.5, n)
    dt = 2 * np.pi / rng.choice([48.0, 96.0], n); h2 = dt / 2; h6 = dt / 6
    a, b, c, d = rng.normal(0, 1, (4, n))
    c2 = rng.uniform(0, 1, n)

    def f(a, b, c, d):
        return b, mu * (1 - a * a) * b - a + kap * (c - a), d, mu * (1 - c * c) * d - w2 * c + kap * (a - c)
    out = np.empty((L, n))
    for t in range(-100, L):
        k1 = f(a, b, c, d)
        k2 = f(a + h2 * k1[0], b + h2 * k1[1], c + h2 * k1[2], d + h2 * k1[3])
        k3 = f(a + h2 * k2[0], b + h2 * k2[1], c + h2 * k2[2], d + h2 * k2[3])
        k4 = f(a + dt * k3[0], b + dt * k3[1], c + dt * k3[2], d + dt * k3[3])
        a = a + h6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); b = b + h6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        c = c + h6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]); d = d + h6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        if t >= 0:
            out[t] = a + c2 * c
    return out.T, np.ones(n, np.int64)


def _kuramoto(rng, n, L, N=4):
    base = 2 * np.pi / rng.choice([7.0, 12.0, 24.0, 96.0, 288.0], n)
    w = base * np.exp(rng.normal(0, 0.15, (N, n)))
    Kc = rng.uniform(0, 2, n) * base; a = rng.uniform(0, 1, (N, n))
    nz = rng.uniform(0, 0.3, n) * np.sqrt(base)
    th = rng.uniform(0, 2 * np.pi, (N, n))
    Z = rng.standard_normal((L, N, n)) * nz
    out = np.empty((L, n))
    for t in range(L):
        cs, sn = np.cos(th), np.sin(th)
        C, S = cs.mean(0), sn.mean(0)                                  # order parameter R e^{i psi}
        th = th + w + Kc * (S * cs - C * sn) + Z[t]                    # K R sin(psi - theta)
        out[t] = (a * cs).sum(0)
    return out.T, np.ones(n, np.int64)


def _chaos(rng, m, L):
    kind = rng.integers(0, 4, m)
    y = np.zeros((m, L)); per = np.ones(m, np.int64)
    for k, g in enumerate((_lorenz, _pendulum, _vdp, _kuramoto)):
        i = np.nonzero(kind == k)[0]
        if len(i):
            y[i], per[i] = g(rng, len(i), L)
    y = _std(y) + rng.standard_normal((m, L)) * _logu(rng, 1e-3, 0.2, m)
    return y, per


def _count(rng, m, L):
    P = _pick_p(rng, m, GEN_P, GEN_W)
    seas, day = _season(rng, m, L, P)
    wf = rng.normal(0, 0.3, (m, 7))                                    # weekday factor (sub-daily rows)
    seas = seas + np.where(P >= 24, np.take_along_axis(wf, day % 7, 1), 0.0)
    y = np.zeros((m, L))
    cro = rng.random(m) < 0.45
    i = np.nonzero(cro)[0]
    if len(i):                                                         # Croston: occurrence x size
        n = len(i)
        p0 = _logu(rng, 0.02, 0.6, n)
        pt = 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + _u(rng, 0, 1.5, n) * seas[i])))
        r, c = np.nonzero(rng.random((n, L)) < pt)
        msz = _logu(rng, 1, 30, n)[r, 0]; k = _logu(rng, 0.5, 20, n)[r, 0]
        nb = rng.random(n)[r] < 0.5
        sz = np.where(nb, rng.poisson(rng.gamma(k, msz / k)), np.round(msz * np.exp(rng.normal(0, 0.7, len(r)))))
        yy = np.zeros((n, L)); yy[r, c] = 1 + sz
        y[i] = yy
    i = np.nonzero(~cro)[0]
    if len(i):                                                         # latent-state Poisson / NB
        n = len(i)
        phi = _u(rng, 0.8, 0.995, n)
        st = _ar1(rng.standard_normal((n, L)) * _logu(rng, 0.01, 0.2, n), phi)
        lam = np.exp(np.minimum(np.log(_logu(rng, 0.1, 200, n)) + _u(rng, 0, 1.2, n) * seas[i] + st
                                + rng.normal(0, 0.5, (n, 1)) * np.arange(L) / L, 13.0))
        j = np.nonzero(rng.random(n) < 0.5)[0]                           # NB = gamma-Poisson mixture
        if len(j):
            kk = _logu(rng, 1, 100, len(j))
            lam[j] = rng.gamma(np.broadcast_to(kk, lam[j].shape), lam[j] / kk)
        y[i] = rng.poisson(lam)
    return y, np.where(P[:, 0] > 1, P[:, 0], 1)


def _hetero(rng, m, L):
    P = _pick_p(rng, m, GEN_P, GEN_W)
    al = rng.uniform(0.02, 0.2, m); be = rng.uniform(0.6, 0.97, m) * (1 - al); om = 1 - al - be
    Z = np.ascontiguousarray(_noise(rng, m, L, p_heavy=0.7).T)
    h = np.ones(m); e = np.zeros(m)
    for t in range(L):
        h = om + al * e * e + be * h
        e = np.sqrt(h) * Z[t]
        Z[t] = e
    R = Z.T
    kind = rng.integers(0, 3, m)
    y = R.copy()                                                       # kind 0: returns
    i = np.nonzero(kind == 1)[0]
    if len(i):                                                         # kind 1: geometric price
        vol = _logu(rng, 0.002, 0.03, len(i))
        y[i] = np.exp(np.clip(np.cumsum(R[i] * vol + rng.normal(0, 0.3, (len(i), 1)) * vol, 1), -30, 30))
    i = np.nonzero(kind == 2)[0]
    if len(i):                                                         # kind 2: seasonal AR mean + GARCH noise
        n = len(i)
        seas, _ = _season(rng, n, L, P[i])
        y[i] = (_ar1(rng.normal(0, 0.3, (n, L)), _u(rng, 0, 0.95, n)) + _u(rng, 0, 3, n) * seas
                + R[i] * _u(rng, 0.2, 1.5, n))
    return y, np.where((kind == 2) & (P[:, 0] > 1), P[:, 0], 1)


_GEN = (_structural, _regime, _chaos, _count, _hetero)


def _make(rng, f, c, L):
    with np.errstate(all="ignore"):
        y, per = _GEN[f](rng, c, L)
    y = np.asarray(y, np.float64)
    bad = np.isinf(y).any(1) | np.isnan(y).all(1) | (np.abs(np.nan_to_num(y)) > 1e12).any(1)
    if bad.any():                                                      # never emit inf; fall back to noise
        y[bad] = rng.normal(0, 1, (int(bad.sum()), L)); per = np.where(bad, 1, per)
    if FAMILIES[f] in ("regime", "chaos", "hetero"):                   # signed families: random affine
        sc = rng.lognormal(0, 2, (c, 1))
        y = y * sc + rng.normal(0, 3, (c, 1)) * sc * (rng.random((c, 1)) < 0.5)
    per = np.where(np.asarray(per) >= L, 1, per)
    return y.astype(np.float32), np.asarray(per, np.int64)


def _take(rng, f, c, L):
    key = (f, L)
    buf = _POOL.get(key)
    if buf is None or buf[2] + c > len(buf[0]):
        if len(_POOL) > 4 * len(FAMILIES):
            _POOL.clear()
        rows = POOL_ELEMS // max(L, 1) * {"regime": 4, "chaos": 16, "hetero": 4}.get(FAMILIES[f], 1)
        y, per = _make(rng, f, max(c, 256, rows), L)       # time-loop families: bigger pools (per-step overhead)
        buf = _POOL[key] = [y, per, 0]
    i = buf[2]
    buf[2] = i + c
    return buf[0][i:i + c], buf[1][i:i + c]


def generate(rng, n, length, weights=None, short_frac=0.25, min_len=16, min_cycles=2,
             return_family=False, pool=True):
    """n rows of `length` points (right-aligned; NaN = missing or left pad), period int64 [n].
    Left-padded rows keep >= min_cycles cycles of their period when the window allows it."""
    L = int(length)
    y = np.zeros((n, L), np.float32); per = np.ones(n, np.int64); fam = np.zeros(n, np.int64)
    if n == 0:
        return (y, per, fam) if return_family else (y, per)
    w = rng.dirichlet(20 * FAM_W) if weights is None else np.asarray(weights, float) / np.sum(weights)
    counts = rng.multinomial(n, w)
    i = 0
    for f, c in enumerate(counts):
        if c:
            y[i:i + c], per[i:i + c] = _take(rng, f, c, L) if pool else _make(rng, f, c, L)
            fam[i:i + c] = f
            i += c
    lo = np.minimum(np.maximum(min_len, min_cycles * per + 1), L)
    lens = np.where(rng.random(n) < short_frac, rng.integers(lo, L + 1), L)
    y[np.arange(L)[None, :] < (L - lens)[:, None]] = np.nan
    o = rng.permutation(n)
    out = (y[o], per[o])
    return out + (fam[o],) if return_family else out


# ----------------------------------------------------------------------------- sanity checks
def _acf(x, lag):
    a, b = x[:, :-lag], x[:, lag:]
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).mean(1) / (a.std(1) * b.std(1) + 1e-12)


def _check(n=2000, L=2048, seed=0):
    import time
    rng = np.random.default_rng(seed)
    for LL in (320, 2048):
        cost = 0.0
        for f, nm in enumerate(FAMILIES):
            k = POOL_ELEMS // LL * {"regime": 4, "chaos": 16, "hetero": 4}.get(nm, 1)
            t0 = time.time(); _make(rng, f, k, LL); dt = time.time() - t0
            cost += FAM_W[f] * dt / k
            print(f"  L={LL} {nm:<10s} {k / dt:9,.0f} series/s (pool refill of {k} rows)")
        print(f"L={LL}: {1 / cost:,.0f} series/s expected steady state (FAM_W-weighted refills)")
    _POOL.clear(); generate(rng, 32, 320); t0 = time.time(); k = 0
    while k < 200_000:                                                  # >= 2 refills of every pool
        generate(rng, 32, 320); k += 32
    print(f"L=320: {k / (time.time() - t0):,.0f} series/s measured over {k} rows in 32-row calls")
    _POOL.clear()
    y, per, fam = generate(rng, n, L, short_frac=0.0, return_family=True, pool=False)
    print(f"\nn={n} L={L}: inf-free: {not np.isinf(y).any()}, NaN share {np.isnan(y).mean():.4f}")
    print("family share:", {nm: round(float(np.mean(fam == f)), 3) for f, nm in enumerate(FAMILIES)})
    vals, cnt = np.unique(per, return_counts=True)
    print("period coverage:", {int(v): round(float(c) / n, 3) for v, c in zip(vals, cnt)})
    print(f"fraction with period >= 96: {np.mean(per >= 96):.3f}; >= 24: {np.mean(per >= 24):.3f}; "
          f"none: {np.mean(per == 1):.3f}")
    z = _std(np.where(np.isnan(y), np.nanmean(y, 1, keepdims=True), y).astype(np.float64))
    F = np.abs(np.fft.rfft(z, axis=1)) ** 2; F[:, 0] = 0
    share = []
    for r in np.nonzero(per > 1)[0]:                                    # power near harmonics 1..4 of P
        b = L / per[r]
        idx = np.unique(np.concatenate([np.arange(round(k * b) - 1, round(k * b) + 2) for k in range(1, 5)]))
        share.append(F[r, idx[(idx > 0) & (idx < F.shape[1])]].sum() / F[r].sum())
    share = np.array(share)
    print(f"seasonal rows: median spectral power share at harmonics 1-4 of P = {np.median(share):.2f} "
          f"(rows > 0.2: {np.mean(share > 0.2):.2f})")
    for p in (24, 96, 144, 288):
        r = per == p
        if r.sum() > 5:
            a7 = np.median(_acf(z[r], 7 * p)) if 7 * p < L - 50 else float("nan")
            print(f"  P={p:3d}: n={r.sum():4d}  ACF(P)={np.median(_acf(z[r], p)):.2f}  "
                  f"ACF(P/2)={np.median(_acf(z[r], p // 2)):.2f}  ACF(7P)={a7:.2f}")
    fr = np.arange(1, F.shape[1])
    print(f"mean-spectrum log-log slope: {np.polyfit(np.log(fr), np.log(F[:, 1:].mean(0) + 1e-12), 1)[0]:.2f}")
    d = _std(np.diff(z, axis=1))
    kurt = (d ** 4).mean(1) - 3
    print(f"excess kurtosis of increments: median {np.median(kurt):.1f}, rows > 3: {np.mean(kurt > 3):.2f}")
    print(f"rows >30% zeros: {np.mean((y == 0).mean(1) > 0.3):.3f}; integer-valued rows: "
          f"{np.mean(np.all(np.isnan(y) | (y == np.round(y)), 1)):.3f}; rows with NaN gaps: "
          f"{np.mean(np.isnan(y).any(1)):.3f}")


if __name__ == "__main__":
    _check()
