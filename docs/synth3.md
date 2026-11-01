# synth3: synthetic prior and real-window augmentation

Module: `tools/synth3.py`, numpy only, CPU. Status: implemented and checked for speed and sanity only.
It has **not been trained with** yet; `train.py` is unchanged.

```python
y, period = S3.generate(rng, n, length)          # float32 [n, length] (NaN = missing / left pad), int64 [n]
win2, per2 = S3.augment(win, rng, period=per)    # real windows; without period= returns only win2
```

`uv run --no-project --python 3.12 --with matplotlib --with numpy python tools/synth3.py` runs the speed benchmark
and diagnostics, and writes `docs/synth3_examples.png`, a grid of 30 examples with 2 to 3 per family plus 3
augmented rows. The project venv has no matplotlib. Under `.venv/bin/python` the script skips only the PNG.

## Generator families

Each row comes from one family. Family weights are drawn per call from Dirichlet(10·1), so each family's
share is about 9% ± 3% and no family dominates a batch. Nothing loops in Python over rows or time steps.
Linear recursions (AR, MA, seasonal AR, ETS slope, OU, stochastic volatility) are exact FFT convolutions
with closed-form impulse responses. Integrations are `cumsum`.

| family | what it draws | references |
|---|---|---|
| `gp` | KernelSynth-style composite GP. Bank: RBF and RQ at short, medium and slow lengthscales, a periodic kernel at a frequency-realistic period, a second periodic kernel, and a linear kernel. Random sums of 1 to 4 terms built from single kernels or products (locally periodic P·(1+E), P1·P2, A·Lin, C·E), plus a linear-trend term and a white-noise term. Stationary parts use spectral synthesis on a ≥2L circulant grid, costing O(L log L) with no Cholesky. The RQ kernel is a gamma scale mixture of RBFs. The periodic kernel is an exact random Fourier series with weights exp(-k²l²/4). Each product is computed as the product of independent draws, which has the product covariance exactly. | Chronos KernelSynth, Ansari et al. 2024 ([2403.07815](https://arxiv.org/abs/2403.07815)) App. A; Rahimi & Recht 2007 (random features); Dietrich & Newsam 1997 and Wood & Chan 1994 (circulant embedding); Rasmussen & Williams 2006, ch. 4 (RQ as a scale mixture, product kernels); MacKay 1998 (periodic kernel) |
| `season` | Multi-seasonal profiles. Period sets per frequency: 5min 288/2016, 10min 144/1008, 15min 96/672, 30min 48/336, H 24/168, D 7/365.25, W 52, M 12, Q 4. Profile shapes are smooth harmonics, step tables (p ≤ 12) or peaky shapes (exp-transformed, like rush hours). Sub-daily data gets a weekday factor with an optional weekend dip, and a second-period harmonic. Linear or wandering trend. Additive or multiplicative seasonality. Additive or multiplicative noise. | TBATS, De Livera, Hyndman & Snyder 2011 (JASA); ForecastPFN, Dooley et al. 2023 (frequency-specific seasonal priors); fev-bench / GIFT seasonality conventions |
| `arima` | SARIMA(2,d,2)(1,D,0)_s. AR roots are real or a complex pair (pseudo-cycles). MA(2). Seasonal AR at the frequency's period with Φ ∈ [0.3, 0.95], or Φ = 1 for a seasonal unit root. d ∈ {0, 1, 2}, with drift. 128-step burn-in. | Box, Jenkins, Reinsel & Ljung, *Time Series Analysis* (5th ed.); Hyndman & Athanasopoulos, *FPP3* ch. 9 |
| `ets` | Local-linear-trend ETS with a damped slope (φ ∈ [0.8, 0.98] or 1). Seasonality evolves through a stochastic amplitude. Additive or multiplicative error and seasonality. | Hyndman, Koehler, Ord & Snyder 2008, *Forecasting with Exponential Smoothing*; Gardner & McKenzie 1985 (damped trend) |
| `regime` | Markov switching among 2 or 3 regimes, with switch probability in [0.002, 0.05]. The switch changes the mean, the volatility, both, or the drift. OU mean reversion with a random speed. Optional seasonality. | Hamilton 1989 (Econometrica); TempoPFN OU/regime-switching SDE generator ([2510.25502](https://arxiv.org/abs/2510.25502)), its most important generator in ablation (+26% CRPS when removed) |
| `trend` | Piecewise-linear trend with 0 to 5 changepoints, logistic saturation (either sign), or exponential growth/decay. Optional additive or multiplicative seasonality and noise. | Prophet, Taylor & Letham 2018 (The American Statistician); ForecastPFN trend priors |
| `rw` | Random walk with drift (Gaussian or Student-t steps), integrated (smooth) random walk, geometric random walk. | standard; TempoPFN / Toto synthetic banks |
| `vol` | Stochastic volatility, where log-variance follows an AR(1) with φ ∈ [0.85, 0.995]. Gives GARCH-like volatility clustering with heavy tails. Output is returns, absolute returns, or prices. | Bollerslev 1986 (GARCH); Taylor 1986 and Harvey, Ruiz & Shephard 1994 (SV). SV is used because it vectorises exactly. A true GARCH recursion would need a per-step loop. |
| `count` | Intermittent and count demand. Zero-inflated Poisson or negative binomial (gamma mixing, dispersion 0.3 to 10). Rate 0.05 to 30 with optional seasonal and trending components. | Croston 1972; Syntetos & Boylan 2005; Lambert 1992 (zero inflation) |
| `pattern` | Sawtooth, square or duty cycle, triangle, decaying spike trains, random steps, random spikes. Sometimes a second pattern is overlaid, plus noise. | TempoPFN Sawtooth/Step/Spike generators (Step and Spike are worth 11-12% in its ablation) |
| `causal` | CauKer-like. 5 root signals (seasonal, periodic GP, AR, random walk, steps) feed a random DAG. Edges have random weights and lags of 0 to 11 steps. Each node applies a nonlinearity (identity, tanh, ReLU, sin, square, sigmoid, abs). The output is the last node or a random linear read-out. A period is reported only if the lag-p autocorrelation is > 0.5. | CauKer, Xie et al. 2025 (*CauKer: classification time series foundation models can be pretrained on synthetic data only*); TempoPFN ablation (+14% when removed) |

Post-processing applies to every row, each step with modest probability. Continuous rows are standardised first.
Steps: extra heteroscedastic or heavy-tailed noise (25%, Student-t df 2.5 to 10 from a cached 10 MB pool),
positivity shift (40%), floor censoring (5%), lognormal scale exp(N(0, 2)) and offset, quantisation (8%),
iid missing values (8%), one missing run (5%), and left padding to a random length (`short_frac=0.5`, min
`min_len=16`).

**Period** is the dominant seasonal period for the seasonal-naive reference: the frequency's primary period
(288, 144, 96, 48, 24, 7, 52, 12 or 4) when a seasonal component is present, and 1 otherwise. A period
≥ length is reset to 1. The spectral check below confirms that these periods are real.

## Augment (real windows)

Each transform fires per row with the probability in `AUG_P`, and only on rows with ≥ `min_obs` observed points:
amp 0.25, censor 0.10, spike 0.15, missing 0.12, reagg 0.10, shift 0.12. At these defaults about 60% of the rows
passed in are changed. `TSAR_AUG` decides which real rows are passed in, so the effective rate is about
0.6 × TSAR_AUG.

**No future information.** Every data-dependent quantity is causal:
- Spike and level-shift magnitudes use the expanding std of values strictly before the affected point.
- The censoring cap is a quantile of a prefix (20-50% of the observed span). It is applied only after that prefix.
- Amplitude curves, gap positions and shift times are random and independent of the data.

`reagg` sums or takes the mean over non-overlapping blocks of k ∈ {2, 3, 4, 6, 12, 24}, aligned to the last
point. Each block value uses only its own block. The result is right-aligned and left-padded, and the period
becomes p/k when k divides p (otherwise 1). The last point is never made missing by `missing`.

## Speed and diagnostics (M3, CPU)

`generate` serves rows from per-family pools. Each pool holds `POOL_ROWS = 512` fresh rows per (family, length)
and is refilled with the caller's `rng` when it runs out. Every row is used exactly once, and the family mixture
is still re-drawn on each call. Memory is about 1.3 MB per family at L = 576. Pooling amortises numpy's
per-call overhead, which dominated when a family produced only about 12 rows per call. `pool=False`
generates on the spot.

These numbers were measured while the machine was saturated (load average 90-140 on 8 cores, from the
training queue and other agents' multiprocessing jobs), so they are upper bounds. During these runs a reference
`rng.normal((128, 576))` took 0.95-1.3 ms of CPU, where about 0.4 ms is typical when the machine is idle.
The < 20 ms target has **not** been measured on an idle machine.

| call | measured under load |
|---|---|
| `generate(rng, 128, 576)`, amortised over 200 calls including refills | **23.6 ms CPU** per call (load avg 94). Median call 0.4 ms; refill calls spike to ~50-300 ms wall. |
| by family, CPU per 128 rows | rw 9, pattern 16, season 19, count 22, vol 24, trend 31, arima 37, ets 39, regime 44, gp 66, causal 68 ms |
| `augment(128 x 576)` at default probabilities | ~3.4 ms at load avg ~20; 20 ms at load avg ~90 |

Run `__main__` again on a quiet machine to get real numbers. If the refill spikes ever matter, call `generate`
from a prefetch thread. numpy releases the GIL in FFTs, RNG fills and most ufuncs.

Diagnostics from 8192 rows with seed 1 (output of `__main__`):
- Family shares: 7-13% each.
- 25% NaN overall, mostly left padding. 11% of rows have internal gaps.
- Period distribution: about 50% have period 1. Among the rest: 24 (9%), 7 (8%), 12 (6-7%), 52 (5%), 96 (4%),
  4 (3.5%), 288 (3.5%), 144 (3%), 48 (2%), and a small tail of random periods from `gp` and `pattern`.
- Spectral sanity on complete rows: rows with period > 1 put a median 0.49-0.51 of their detrended periodogram
  power on harmonics k/p, and the argmax lands on a harmonic in 62-64% of them. Non-seasonal rows checked at a
  random period give 0.02 and 3%.
- Lengths 24, 64, 320, 576 and 2048 all pass the checks: shapes and dtypes correct, finite values or NaN only,
  period < length.

## Integration snippet for `train.py` (for the lead; not applied)

Both flags default to 0. With both at 0, the batch and the RNG stream are bit-identical to the current
`sample_batch`, because `synth3` is not called and no random numbers are drawn.

**1. Config.** Next to `SYN2_SHARE` (around line 63):

```python
SYN3_SHARE = float(_env("TSAR_SYN3", 0))                   # share of synthetic rows from tools/synth3.generate
AUG_P = float(_env("TSAR_AUG", 0))                         # prob. that a real row goes through tools/synth3.augment
if SYN3_SHARE > 0 or AUG_P > 0:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import synth3 as S3
```

**2. `sample_batch`.** Replace the whole function with:

```python
def sample_batch(rng):
    """Right-aligned windows of WINDOW points, balanced across datasets. NaN = missing/pad."""
    win = np.full((BATCH, WINDOW), np.nan, dtype=np.float32)
    per = np.ones(BATCH, dtype=np.int64)
    n_syn = int(BATCH * SYN_FRAC)
    n_ks = int(BATCH * KS_FRAC) if KS is not None else 0
    n3 = int(round(n_syn * SYN3_SHARE))                                  # TSAR_SYN3: synth3 share of synthetic rows
    n2 = int((n_syn - n3) * SYN2_SHARE)
    if n3:
        win[:n3], per[:n3] = S3.generate(rng, n3, WINDOW, min_len=2 * PATCH)
    win[n3:n3 + n2], per[n3:n3 + n2] = synthetic2(rng, n2)
    win[n3 + n2:n_syn], per[n3 + n2:n_syn] = synthetic(rng, n_syn - n3 - n2)
    win[n_syn:n_syn + n_ks] = kernelsynth(rng, n_ks)
    for b in range(n_syn + n_ks, BATCH):
        d = datasets[rng.integers(len(datasets))]
        series = by_dataset[d]
        per[b] = SEASON[d]
        y = series[rng.integers(len(series))]
        end = rng.integers(min(2 * PATCH, len(y)), len(y) + 1)
        w = y[max(0, end - WINDOW):end]
        win[b, WINDOW - len(w):] = w
    if AUG_P > 0:                                                        # TSAR_AUG: augment real rows only
        rows = np.arange(n_syn + n_ks, BATCH)
        rows = rows[rng.random(len(rows)) < AUG_P]
        if len(rows):
            win[rows], per[rows] = S3.augment(win[rows], rng, period=per[rows], min_obs=4 * PATCH)
    return torch.from_numpy(win).to(DEVICE), torch.from_numpy(per).to(DEVICE)
```

Notes for the lead:
- **What TSAR_SYN3 means.** `TSAR_SYN3` is a share of the *synthetic* rows (`SYN_FRAC · BATCH`). To make a batch
  mostly synth3, set for example `TSAR_SYN_FRAC=0.5 TSAR_SYN3=1`. When `TSAR_SYN2` is also set, it applies to the
  remaining synthetic rows.
- **Cost.** At BATCH = 128, SYN_FRAC = 0.5 and WINDOW = 320, synth3 rows cost an amortised few ms per step on
  the CPU, plus about 1-2 ms for augment at TSAR_AUG = 0.5. A pool refill happens about once every 512/6 ≈ 85
  steps per family. If the refills show up in the step-time profile, move `sample_batch` into a prefetch thread.
- **When to judge it.** Following `docs/arch_plan.md` §0 fact 2 and item 1, judge synth3 at 30-60 min and on the
  `TSAR_ZS=1` / bench holdout, not in the 4-minute loop. A synthetic-first curriculum (high SYN_FRAC early) is the
  intended use.
- **Suggested first runs.** Two arms, each against the current best at the same `TSAR_STEPS`:
  `TSAR_SYN_FRAC=0.5 TSAR_SYN3=1` and `TSAR_AUG=0.5`. Then try them together.
