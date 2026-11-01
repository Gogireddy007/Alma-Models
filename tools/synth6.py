"""
synth6: two more synthetic sources for train_syn5.py, both default-off there.

    gp(rng, n, length, min_len=16, short_frac=0.25) -> y float32 [n, length], period int64 [n]
        KernelSynth-style Gaussian-process prior, implemented here from the published method
        (Ansari et al. 2024, "Chronos", Sec. 4.2 / Alg. 2): draw J ~ U{1..5} kernels with replacement
        from a bank (periodic, RBF, rational-quadratic, linear, constant, white noise), fold them with
        random + or *, and sample y ~ GP(0, K) on a grid over [0, 1]. No corpus file is read (the
        TSAR_KS_FRAC path reads a cached parquet; this does not).
    tsmixup(rng, n, length, series_by_ds, season, min_len=16, kmax=3, alpha=1.5, same_p=0.5)
        TSMixup (same paper, Sec. 4.1 / Alg. 1): convex combination of k in {2..kmax} mean-scaled real
        windows with Dirichlet(alpha) weights. series_by_ds MUST be the trainer's TRAIN pool
        (train_syn5.by_dataset, i.e. bench.load_train minus any TSAR_ZS hold-outs); this module reads no
        file and never sees VAL/TEST. A same_p share of rows mixes series of ONE dataset (consistent
        frequency and season); the rest mixes across datasets, as in the paper.

Kernel bank (the paper's bank re-expressed for our window: periods are calendar constants in SAMPLES, not
fitted to anything; lengthscales are fractions of the window): periodic p in {4,7,12,24,48,52,96,144,168,
288,365} restricted to 2p <= L, lengthscale {0.5,1,2}; RBF l in {0.03,0.1,0.3,1}; RQ l in {0.1,0.3,1},
alpha {0.1,1,10}; linear sigma0 in {0,0.1,1} about a random centre; constant {0.1..1}; white {0.1,1}.
The returned period is the first periodic kernel's p (1 if the draw has none).

Speed: no Cholesky. A composition is kept in the closed form K = sum_k diag(u_k) Toeplitz(t_k) diag(u_k)
(stationary kernels are lag functions; linear = constant + rank-1 diag(x-c) 1 diag(x-c); products of terms
stay terms), and each term is sampled by circulant embedding (one FFT, size a multiple of its periods so
periodic factors embed exactly), so cost is O(terms * L log L) at any length. Rows are served from a pool refilled with the caller's rng (seeded, reproducible), like synth5.
"""
import numpy as np

PERIODS = np.array([4, 7, 12, 24, 48, 52, 96, 144, 168, 288, 365])
POOL_ROWS = 512
_POOL = {}
KINDS = ("periodic", "rbf", "rq", "linear", "const", "white")
KIND_W = np.array([0.35, 0.15, 0.15, 0.15, 0.05, 0.15])


def _terms(rng, L):
    """One random composition as a list of terms (u, factors): K = sum_k diag(u_k) Toeplitz(t_k) diag(u_k),
    u_k [L] or None (= ones), t_k = product of stationary lag factors (kind, param...). Sums concatenate
    term lists; products multiply terms pairwise, which keeps the form exact. Period = first periodic p."""
    okp = PERIODS[2 * PERIODS <= L]
    K, per = None, 1
    for j in range(int(rng.integers(1, 6))):
        kind = int(rng.choice(len(KINDS), p=KIND_W))
        if kind == 0 and not len(okp):
            kind = 1
        if kind == 0:
            pp = int(rng.choice(okp))
            t = [(None, [("per", pp, rng.choice([0.5, 1.0, 2.0]))])]
            per = pp if per == 1 else per
        elif kind == 1:
            t = [(None, [("rbf", rng.choice([0.03, 0.1, 0.3, 1.0]))])]
        elif kind == 2:
            t = [(None, [("rq", rng.choice([0.1, 0.3, 1.0]), rng.choice([0.1, 1.0, 10.0]))])]
        elif kind == 3:                                                    # s0^2 + (x-c)(x'-c)
            s0 = rng.choice([0.0, 0.1, 1.0])
            t = [(np.linspace(0, 1, L) - rng.uniform(0, 1), [])] + ([(None, [("const", s0 * s0)])] if s0 else [])
        elif kind == 4:
            t = [(None, [("const", rng.uniform(0.1, 1.0))])]
        else:
            t = [(None, [("white", rng.choice([0.01, 1.0]))])]
        if K is None:
            K = t
        elif rng.random() < 0.5:
            K = K + t
        else:
            K = [(u1 if u2 is None else u2 if u1 is None else u1 * u2, f1 + f2) for u1, f1 in K for u2, f2 in t]
    return K, per


def _lagfn(factors, k, L):
    """Product of the lag factors at integer lags k (tau = k / (L-1) on the unit interval)."""
    tau = k / max(L - 1, 1)
    t = np.ones(len(k))
    for f in factors:
        if f[0] == "per":
            t *= np.exp(-2 * np.sin(np.pi * k / f[1]) ** 2 / f[2] ** 2)
        elif f[0] == "rbf":
            t *= np.exp(-tau ** 2 / (2 * f[1] ** 2))
        elif f[0] == "rq":
            t *= (1 + tau ** 2 / (2 * f[2] * f[1] ** 2)) ** -f[2]
        elif f[0] == "const":
            t *= f[1]
        else:
            t *= f[1] * (k == 0)
    return t


def _draw(rng, m, L):
    """GP(0, K) draws via circulant embedding of each stationary term (one FFT per term, no Cholesky);
    independent terms add, so y = sum_k u_k * z_k. Each composition gives two rows (Re, Im parts). The embedding size M >= 4L-2 is a multiple of the lcm of
    the term's periods (when that lcm <= 8L), so periodic factors embed exactly; decaying factors are
    evaluated out to lag M/2 >= 2L. Remaining negative eigenvalues (only very long RBF/RQ lengthscales) are
    clipped to 0: the nearest PSD circulant, fine for a prior."""
    h = (m + 1) // 2                                                        # Re and Im of one FFT are two
    out = np.zeros((2 * h, L))                                              # independent draws of the same
    per = np.ones(2 * h, np.int64)                                          # GP: 2 rows per composition
    for i in range(h):
        K, per[2 * i] = _terms(rng, L)
        per[2 * i + 1] = per[2 * i]
        for u, fac in K:
            P = int(np.lcm.reduce([f[1] for f in fac if f[0] == "per"] or [1]))
            M = 4 * L - 2
            if 1 < P <= 8 * L:
                M = P * -(-M // P)
            k = np.arange(M)
            lam = np.clip(np.fft.rfft(_lagfn(fac, np.minimum(k, M - k), L)).real, 0, None)
            lam = np.concatenate([lam, lam[1:(M + 1) // 2][::-1]])            # full symmetric spectrum
            xi = rng.standard_normal(M) + 1j * rng.standard_normal(M)
            z = np.fft.fft(np.sqrt(lam / M) * xi)[:L]                      # Cov(Re) = Cov(Im) = Toeplitz(t)
            z = np.stack([z.real, z.imag])
            out[2 * i:2 * i + 2] += z if u is None else u * z
    o = rng.permutation(2 * h)[:m]                                          # pair-mates land in random places
    out, per = out[o], per[o]
    sc = rng.lognormal(0, 2, (m, 1))                                        # random affine, like synth5
    out = out * sc + rng.normal(0, 3, (m, 1)) * sc * (rng.random((m, 1)) < 0.5)
    bad = ~np.isfinite(out).all(1)
    if bad.any():
        out[bad] = rng.normal(0, 1, (int(bad.sum()), L)); per[bad] = 1
    return out.astype(np.float32), per


def _pad(rng, y, per, min_len, short_frac, min_cycles=2):
    n, L = y.shape
    lo = np.minimum(np.maximum(min_len, min_cycles * per + 1), L)
    lens = np.where(rng.random(n) < short_frac, rng.integers(lo, L + 1), L)
    y[np.arange(L)[None, :] < (L - lens)[:, None]] = np.nan
    return y


def gp(rng, n, length, min_len=16, short_frac=0.25, pool=True):
    L = int(length)
    if n == 0:
        return np.zeros((0, L), np.float32), np.ones(0, np.int64)
    if pool:
        buf = _POOL.get(L)
        if buf is None or buf[2] + n > len(buf[0]):
            if len(_POOL) > 4:
                _POOL.clear()
            buf = _POOL[L] = list(_draw(rng, max(n, POOL_ROWS * 320 // max(L, 320)), L)) + [0]
        i = buf[2]; buf[2] = i + n
        y, per = buf[0][i:i + n].copy(), buf[1][i:i + n].copy()
    else:
        y, per = _draw(rng, n, L)
    return _pad(rng, y, per, min_len, short_frac), per


def tsmixup(rng, n, length, series_by_ds, season, min_len=16, kmax=3, alpha=1.5, same_p=0.5):
    L = int(length)
    y = np.full((n, L), np.nan, np.float32); per = np.ones(n, np.int64)
    names = list(series_by_ds)
    for i in range(n):
        k = int(rng.integers(2, kmax + 1))
        if rng.random() < same_p:
            ds = [names[rng.integers(len(names))]] * k
        else:
            ds = [names[j] for j in rng.integers(len(names), size=k)]
        w = rng.dirichlet(alpha * np.ones(k))
        acc = np.zeros(L); n_obs = L; s0 = 1.0
        for j in range(k):
            s = series_by_ds[ds[j]][rng.integers(len(series_by_ds[ds[j]]))]
            end = rng.integers(min(min_len, len(s)), len(s) + 1)
            seg = np.asarray(s[max(0, end - L):end], np.float64)
            sc = np.nanmean(np.abs(seg))
            sc = sc if np.isfinite(sc) and sc > 0 else 1.0
            if j == 0:
                s0 = sc
            n_obs = min(n_obs, len(seg))
            acc[L - len(seg):] += w[j] * seg / sc
        acc[:L - n_obs] = np.nan                                            # right-aligned common support
        y[i] = acc * s0                                                     # back to a real magnitude
        per[i] = season[ds[int(np.argmax(w))]]
    per = np.where(per * 2 >= L, 1, per)
    return y, per


def _check(L=320, seed=0):
    import time
    rng = np.random.default_rng(seed)
    t0 = time.time(); y, per = gp(rng, 512, L, pool=False); dt = time.time() - t0
    print(f"gp L={L}: {512 / dt:,.0f} series/s, finite {np.isfinite(y[~np.isnan(y)]).all()}, "
          f"periodic share {np.mean(per > 1):.2f}")
    t0 = time.time(); y, per = gp(rng, 512, 2048, pool=False)
    print(f"gp L=2048: {512 / (time.time() - t0):,.0f} series/s")
    fake = {"a": [rng.normal(10, 1, 500)], "b": [np.abs(rng.normal(0, 5, 100))]}
    y, per = tsmixup(rng, 64, L, fake, {"a": 24, "b": 12})
    print(f"tsmixup: NaN share {np.isnan(y).mean():.2f}, periods {np.unique(per)}")


if __name__ == "__main__":
    _check()
