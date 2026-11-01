"""
Inference-time methods copy of train.py (TSAR_TTA_* / TSAR_TTT_* flags, all default off; see the
"Inference-time methods" section). TSAR_TTA_CKPT=<ckpt> evaluates a trained checkpoint on bench val only.

Latent world-model forecaster. THIS is the file the research agent edits.

Current design: multi-origin causal latent world model (tracks T1 + T8).
  encode   - a causal Transformer reads the series patch by patch; its output after patch t
             is the latent world state z_t ("what the system looks like now")
  dynamics - a GRU cell rolls z_t forward K patches with no new observations
  decode   - each imagined state is decoded into 9 quantiles for its patch
Training scores rollouts from every origin in a window, so one forward pass trains many
horizons. Values are standardized causally (stats up to the origin only) and passed
through asinh, as in Timestral. Usage: uv run train.py
"""

import os
import sys
import time
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare as P

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

PATCH = 8              # patch length (input and output)
CONTEXT = 256          # input window at inference
MAX_H = 64             # rollout length; eval crops to the task horizon
D_MODEL = 96
N_LAYERS = 2
N_HEADS = 4
BATCH = 128
LR = 3e-3
WEIGHT_DECAY = 0.01
SEED = 0
KS_FRAC = 0.0          # share of each batch from the KernelSynth GP corpus (data/pretrain)

# Scale-up runs (phase 2) may override sizes and the budget through the environment, e.g.
# TSAR_RUN=big TSAR_MINUTES=120 TSAR_D_MODEL=192 uv run train.py. The 4-minute loop never does.
_env = os.environ.get
CONTEXT = int(_env("TSAR_CONTEXT", CONTEXT)); MAX_H = int(_env("TSAR_MAX_H", MAX_H)); BATCH = int(_env("TSAR_BATCH", BATCH))
STEPS = int(_env("TSAR_STEPS", 0))       # if set, train for exactly this many steps (schedule on steps)
BENCH = int(_env("TSAR_BENCH", 0))        # 1 = leak-audited protocol (tools/bench.py): train / val / test families
D_MODEL = int(_env("TSAR_D_MODEL", D_MODEL))
N_LAYERS = int(_env("TSAR_N_LAYERS", N_LAYERS))
N_HEADS = int(_env("TSAR_N_HEADS", N_HEADS))
LR = float(_env("TSAR_LR", LR))
KS_FRAC = float(_env("TSAR_KS_FRAC", KS_FRAC))
EMA = float(_env("TSAR_EMA", 0))                            # weight EMA decay used for inference (0 = off)
TTA_STEPS = int(_env("TSAR_TTA_STEPS", 0))                 # T9: test-time adaptation steps on the context
TTA_LR = float(_env("TSAR_TTA_LR", 1e-3))
TTA_SERIES = int(_env("TSAR_TTA_SERIES", 0))               # 1 = also adapt a per-series latent offset
ENSEMBLE = [(int(v[:-1]), 1 if v[-1] == "+" else -1)          # e.g. "256+,256-,128+"
            for v in _env("TSAR_ENS", f"{CONTEXT}+").split(",")]
LOOPS = int(_env("TSAR_LOOPS", 0))                         # LoopWM dynamics: mean inner loops (0 = GRUCell)
LATENT = float(_env("TSAR_LATENT", 0))                     # weight of latent-consistency (world-model) loss
GATE = int(_env("TSAR_GATE", 0))                           # per-series gated seasonal copy
SYN2_SHARE = float(_env("TSAR_SYN2", 0))                   # share of synthetic rows from synthetic2
SYN_FRAC_ENV = _env("TSAR_SYN_FRAC")
SYN3_SHARE = float(_env("TSAR_SYN3", 0))                   # share of synthetic rows from tools/synth3.generate
AUG_P = float(_env("TSAR_AUG", 0))                         # prob. that a real row goes through tools/synth3.augment
SYN_CURR = float(_env("TSAR_SYN_CURR", 0))                 # synthetic share at step 0, decaying to SYN_FRAC by 60%
PROG = 0.0                                                 # training progress in [0, 1], updated by the loop
if SYN3_SHARE > 0 or AUG_P > 0:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import synth3 as S3
TRAIN_FLIP = float(_env("TSAR_TRAIN_FLIP", 0))               # training-side: prob. of negating a row (needed for sign-flip TTA)
SEAS_REF = int(_env("TSAR_SEAS_REF", 1))                   # T7 seasonal memory pathway in the decoder
NORM = _env("TSAR_NORM", "last")                          # "mean" | "last" (see causal_stats)
FRAME = int(_env("TSAR_FRAME", 1)); FRAMEFIX = int(_env("TSAR_FRAMEFIX", 0)); SREF_AVG = int(_env("TSAR_SREF_AVG", 1)); CTX_READ = int(_env("TSAR_CTX_READ", 0))
SCHED = _env("TSAR_SCHED", "cos"); WEIGHT_DECAY = float(_env("TSAR_WD", WEIGHT_DECAY))                         # frame-change token features (trend fix)
ZS = int(_env("TSAR_ZS", 0))                               # 1 = zero-shot protocol: hold ZS_SETS out of training
ZS_SETS = ["monash_tourism_monthly", "monash_traffic", "nn5", "m4_quarterly"]
SELFCAL = float(_env("TSAR_SELFCAL", 0))                   # T10: shrinkage of per-series self-calibration (0 = off)
SELFCAL_SHIFT = int(_env("TSAR_SELFCAL_SHIFT", 0))         # 1 = also fit a per-series level shift
SPREADS = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0])
SHIFTS = torch.tensor([-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3])
RUN = _env("TSAR_RUN")                                  # name for checkpoints; None = no checkpoints
BUDGET = float(_env("TSAR_MINUTES", P.TIME_BUDGET / 60)) * 60
EVAL_EVERY = float(_env("TSAR_EVAL_MIN", 0)) * 60       # periodic val scoring during long runs
# T1/T3 long context: history older than the fine CONTEXT is summarised into PREFIX // PREFIX_PATCH coarse
# tokens prepended to the encoder sequence (0 = off). Inputs are PREFIX + fine points, oldest first.
PREFIX = int(_env("TSAR_PREFIX", 0))
PREFIX_PATCH = int(_env("TSAR_PREFIX_PATCH", 32))
assert PREFIX % PREFIX_PATCH == 0, "TSAR_PREFIX must be a multiple of TSAR_PREFIX_PATCH"
N_PRE = PREFIX // PREFIX_PATCH
PREFIX_SYN = int(_env("TSAR_PREFIX_SYN", 0))   # 1 = synthetic rows also span the prefix (slower numpy generation)

K_STEPS = MAX_H // PATCH
WINDOW = CONTEXT + MAX_H
N_PATCH = WINDOW // PATCH
TWIN = PREFIX + WINDOW                    # training window incl. the coarse prefix
TCTX = PREFIX + CONTEXT                   # inference context incl. the coarse prefix
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = _env("TSAR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
Q = torch.tensor(P.QUANTILES, device=DEVICE)
NQ = len(P.QUANTILES)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TTA_CKPT = _env("TSAR_TTA_CKPT")                           # eval-only: load this checkpoint, skip training
if BENCH or TTA_CKPT:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import bench as BN
if TTA_CKPT:                               # inference-only: no training data is loaded at all
    train = []
elif BENCH:
    train = BN.load_train(extra=bool(int(_env("TSAR_BENCH_EXTRA", 0))), max_len=int(_env("TSAR_BENCH_MAXLEN", 20000)))
else:
    train = P.load_train_series()
by_dataset = {}
for s in train:
    by_dataset.setdefault(s["dataset"], []).append(s["target"])
ZS_FAMILY = {"nn5": ["monash_nn5_weekly"], "monash_tourism_monthly": ["monash_tourism_quarterly"]}
if ZS:                                     # hold out whole source families, not just the scored sets
    for d in ZS_SETS:
        for f in [d] + ZS_FAMILY.get(d, []):
            by_dataset.pop(f, None)
datasets = list(by_dataset)
SEASON = {s["dataset"]: s["season"] for s in train} if BENCH else {d[0]: d[2] for d in P.DATASETS}
EVALUATE = BN.evaluate if BENCH else P.evaluate


SYN_FRAC = float(SYN_FRAC_ENV) if SYN_FRAC_ENV else 0.3   # share of each batch that is synthetic
PERIODS = np.array([2, 3, 4, 5, 6, 7, 12, 14, 24, 28, 30, 48, 52, 168])


def synthetic(rng, n, W=WINDOW):
    """Random trend + seasonality + AR noise + shifts/spikes, with random lengths."""
    t = np.arange(W, dtype=np.float64)[None, :]
    u = t / W
    kind = rng.integers(0, 3, (n, 1))                                   # linear, damped, exponential
    slope = rng.normal(0, 1, (n, 1)) * (rng.random((n, 1)) < 0.8)
    trend = np.where(kind == 0, slope * u,
            np.where(kind == 1, slope * (1 - np.exp(-3 * u)),
                     np.expm1(rng.uniform(-2, 3, (n, 1)) * u)))
    season = np.zeros((n, W))
    main = np.ones(n, dtype=np.int64)                                     # dominant integer period, 1 = none
    for c in range(2):
        on = rng.random((n, 1)) < 0.6
        per = np.where(rng.random((n, 1)) < 0.7, rng.choice(PERIODS, (n, 1)), rng.uniform(2, 64, (n, 1)))
        phase = rng.uniform(0, 2 * np.pi, (n, 1))
        if c == 0:
            integer = on[:, 0] & (per[:, 0] == np.round(per[:, 0]))
            main = np.where(integer, np.round(per[:, 0]).astype(np.int64), 1)
        for k in range(1, 4):
            amp = rng.normal(0, 1, (n, 1)) / k
            season += on * amp * np.sin(2 * np.pi * k * t / per + k * phase)
    phi = rng.uniform(0, 0.95, (n, 1))
    eps = rng.normal(0, 1, (n, W)) * rng.uniform(0.02, 0.5, (n, 1))
    ar, epsT, ph = np.zeros((W, n)), np.ascontiguousarray(eps.T), phi[:, 0]   # time-major: same arithmetic,
    for i in range(1, W):                                                      # contiguous rows (fast for long W)
        ar[i] = ph * ar[i - 1] + epsT[i]
    ar = ar.T
    shifts = (rng.random((n, W)) < 0.003) * rng.normal(0, 1, (n, W))
    spikes = (rng.random((n, W)) < 0.005) * rng.normal(0, 3, (n, W))
    y = trend + season + ar + np.cumsum(shifts, 1) + spikes
    mult = rng.random((n, 1)) < 0.3                                     # multiplicative seasonality
    y = np.where(mult, (1 + 0.3 * trend) * (1 + 0.3 * season) + ar, y)
    y = y * rng.lognormal(0, 2, (n, 1)) + rng.normal(0, 5, (n, 1))
    count = rng.random((n, 1)) < 0.2                                     # count-valued series
    y = np.where(count, rng.poisson(np.clip(np.abs(y) / np.abs(y).mean(1, keepdims=True) * 3, 0, 1e6)), y)
    length = rng.integers(2 * PATCH, W + 1, n)
    y[np.arange(W)[None, :] < (W - length)[:, None]] = np.nan
    return y.astype(np.float32), main


def synthetic2(rng, n, W=WINDOW):
    """Richer generator for regimes the first one lacks: frequency-matched multi-seasonality,
    piecewise trends with changepoints, logistic growth, random walks / GBM, GARCH-like volatility,
    intermittent demand. Returns windows and a dominant period."""
    t = np.arange(W, dtype=np.float64)
    y = np.zeros((n, W))
    main = np.ones(n, dtype=np.int64)
    pairs = [(24, 168), (7, 365), (12, 0), (4, 0), (52, 0), (48, 336), (5, 0), (0, 0)]
    for i in range(n):
        kind = rng.integers(6)
        if kind == 0:                                              # piecewise-linear trend + seasons
            k = rng.integers(0, 4)
            cps = np.sort(rng.integers(0, W, k))
            slopes = rng.normal(0, 0.05, k + 1)
            sl = slopes[np.searchsorted(cps, t, side="right")]
            base = np.cumsum(sl)
        elif kind == 1:                                            # logistic / saturating growth
            mid, rate = rng.uniform(0.2, 1.5) * W, rng.uniform(0.01, 0.08)
            base = rng.uniform(1, 20) / (1 + np.exp(-rate * (t - mid)))
        elif kind == 2:                                            # random walk with drift
            base = np.cumsum(rng.normal(rng.normal(0, 0.05), rng.uniform(0.1, 1), W))
        elif kind == 3:                                            # GBM with GARCH-like volatility
            vol, r = np.empty(W), np.empty(W)
            v = rng.uniform(0.005, 0.03) ** 2
            a, b = rng.uniform(0.02, 0.15), rng.uniform(0.75, 0.95)
            b = min(b, 0.97 - a)
            w0 = v * (1 - a - b)
            for j in range(W):
                r[j] = rng.normal(0, np.sqrt(v))
                v = w0 + a * r[j] ** 2 + b * v
            base = np.log(rng.uniform(1, 100)) + np.cumsum(r)
            y[i] = np.exp(base)
            continue
        elif kind == 4:                                            # intermittent demand
            p_occ = rng.uniform(0.05, 0.6)
            size = rng.uniform(1, 20)
            y[i] = (rng.random(W) < p_occ) * rng.poisson(size, W)
            continue
        else:
            base = np.zeros(W)
        p1, p2 = pairs[rng.integers(len(pairs))]
        season = np.zeros(W)
        for per, amp in ((p1, rng.uniform(0.3, 2)), (p2, rng.uniform(0.2, 1))):
            if per:
                prof = rng.normal(0, 1, per)                       # arbitrary repeating profile, smoothed
                prof = np.convolve(np.r_[prof, prof[:3]], np.ones(3) / 3, "valid")[:per]
                season += amp * prof[(t.astype(int) + rng.integers(per)) % per]
        if p1:
            main[i] = p1
        noise = rng.normal(0, rng.uniform(0.05, 0.6), W)
        sc = rng.uniform(0.2, 3)
        y[i] = base * sc + season * (1 + 0.5 * (rng.random() < 0.3) * base / (np.abs(base).max() + 1e-9)) + noise
    y = y * rng.lognormal(0, 1.5, (n, 1)) + rng.normal(0, 3, (n, 1)) * (rng.random((n, 1)) < 0.5)
    length = rng.integers(2 * PATCH, W + 1, n)
    y[np.arange(W)[None, :] < (W - length)[:, None]] = np.nan
    return y.astype(np.float32), main


def load_kernelsynth():
    """Chronos KernelSynth corpus: Gaussian-process series, purely synthetic (no benchmark overlap)."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(P.CACHE_DIR, "pretrain", "kernel_synth_*.parquet")))
    if not files or KS_FRAC <= 0:
        return None
    rows = [np.asarray(t, dtype=np.float32) for f in files for t in pd.read_parquet(f, columns=["target"])["target"]]
    L = min(len(r) for r in rows)
    return np.stack([r[-L:] for r in rows])


KS = load_kernelsynth()


def kernelsynth(rng, n):
    win = np.full((n, WINDOW), np.nan, dtype=np.float32)
    if n == 0:
        return win
    rows = KS[rng.integers(len(KS), size=n)]
    start = rng.integers(0, KS.shape[1] - WINDOW + 1, size=n)
    length = rng.integers(2 * PATCH, WINDOW + 1, n)
    for i in range(n):
        w = rows[i, start[i]:start[i] + WINDOW]
        win[i, WINDOW - length[i]:] = w[WINDOW - length[i]:]
    return win * rng.lognormal(0, 2, (n, 1)).astype(np.float32)


def sample_batch(rng):
    """Right-aligned windows of TWIN = PREFIX + WINDOW points, balanced across datasets. NaN = missing/pad.
    Origins are fine patches only, so the prefix is strictly earlier history."""
    win = np.full((BATCH, TWIN), np.nan, dtype=np.float32)
    per = np.ones(BATCH, dtype=np.int64)
    frac = SYN_FRAC if not SYN_CURR else SYN_FRAC + (SYN_CURR - SYN_FRAC) * max(0.0, 1 - PROG / 0.6)
    n_syn = int(BATCH * frac)                                            # TSAR_SYN_CURR: synthetic-first curriculum
    n_ks = int(BATCH * KS_FRAC) if KS is not None else 0
    n3 = int(round(n_syn * SYN3_SHARE))                                  # TSAR_SYN3: synth3 share of synthetic rows
    n2 = int((n_syn - n3) * SYN2_SHARE)
    sw = slice(0, TWIN) if PREFIX_SYN else slice(PREFIX, TWIN)        # synthetic rows: prefix only if PREFIX_SYN
    if n3:
        win[:n3, sw], per[:n3] = S3.generate(rng, n3, sw.stop - sw.start, min_len=2 * PATCH)
    win[n3:n3 + n2, sw], per[n3:n3 + n2] = synthetic2(rng, n2, sw.stop - sw.start)
    win[n3 + n2:n_syn, sw], per[n3 + n2:n_syn] = synthetic(rng, n_syn - n3 - n2, sw.stop - sw.start)
    win[n_syn:n_syn + n_ks, PREFIX:] = kernelsynth(rng, n_ks)
    for b in range(n_syn + n_ks, BATCH):
        d = datasets[rng.integers(len(datasets))]
        series = by_dataset[d]
        per[b] = SEASON[d]
        y = series[rng.integers(len(series))]
        end = rng.integers(min(2 * PATCH, len(y)), len(y) + 1)
        w = y[max(0, end - TWIN):end]
        win[b, TWIN - len(w):] = w
    if AUG_P > 0:                                                        # TSAR_AUG: augment real rows only
        rows = np.arange(n_syn + n_ks, BATCH)
        rows = rows[rng.random(len(rows)) < AUG_P]
        if len(rows):
            win[rows], per[rows] = S3.augment(win[rows], rng, period=per[rows], min_obs=4 * PATCH)
    if TRAIN_FLIP > 0:                     # train on -x too, so the TSAR_ENS "-" member is in-distribution
        win[rng.random(BATCH) < TRAIN_FLIP] *= -1
    return torch.from_numpy(win).to(DEVICE), torch.from_numpy(per).to(DEVICE)


def causal_stats(x):
    """Per-patch loc/scale from observed values up to the end of each patch. x: [B, L].
    NORM="mean": running mean/std. NORM="last": loc = last observed value, scale = running mean
    |step| (the MASE scale), so persistence is the zero point and a trend is a constant slope."""
    obs = ~torch.isnan(x)
    v = torch.where(obs, x, torch.zeros_like(x))
    ends = torch.arange(PATCH - 1, x.shape[1], PATCH, device=x.device)
    cnt = obs.float().cumsum(1)[:, ends]
    s1 = v.cumsum(1)[:, ends]
    s2 = (v * v).cumsum(1)[:, ends]
    mean = s1 / cnt.clamp(min=1)
    std = (s2 / cnt.clamp(min=1) - mean * mean).clamp(min=0).sqrt()
    if NORM == "last":
        idx = torch.where(obs, torch.arange(x.shape[1], device=x.device).expand_as(x), torch.zeros_like(x, dtype=torch.long))
        last_idx = idx.cummax(1).values
        ffill = torch.gather(v, 1, last_idx)
        loc = ffill[:, ends]
        both = obs[:, 1:] & obs[:, :-1]
        d = torch.where(both, (x[:, 1:] - x[:, :-1]).abs(), torch.zeros_like(x[:, 1:]))
        d = torch.cat([torch.zeros_like(x[:, :1]), d], 1).cumsum(1)[:, ends]
        nd = torch.cat([torch.zeros_like(x[:, :1]), both.float()], 1).cumsum(1)[:, ends]
        scale = d / nd.clamp(min=1)
        scale = torch.where(scale > 1e-6, scale, std)
    else:
        loc, scale = mean, std
    scale = torch.where(scale > 1e-6, scale, torch.where(loc.abs() > 1e-6, loc.abs(), torch.ones_like(loc)))
    return loc, scale, cnt                                     # each [B, n_patch]


N_FRAME = 3 if FRAME else 0


def frame_features(loc, scale, cnt):
    """Each patch is normalised in its own running frame, which hides trend slope (a straight line looks
    the same in every frame). Give each token how its frame moved from the previous one, plus how much
    history it has, so the model can recover levels and slopes across frames. All causal."""
    if not FRAME:
        return []
    lp = torch.cat([loc[:, :1], loc[:, :-1]], 1)
    sp = torch.cat([scale[:, :1], scale[:, :-1]], 1)
    dloc = torch.asinh((loc - lp) / scale)
    dls = torch.log(scale / sp).clamp(-5, 5)
    if FRAMEFIX:                           # no previous frame yet (padding): don't leak absolute level/scale
        prev = torch.cat([torch.zeros_like(cnt[:, :1]), cnt[:, :-1]], 1) > 0
        dloc, dls = dloc * prev, dls * prev
    lc = torch.log1p(cnt) / 5
    return [dloc[..., None], dls[..., None], lc[..., None]]


def seasonal_ref(x, loc, scale, m):
    """Seasonal-naive reference for every origin and future step, in each origin's model space.
    x: [B, L] raw, m: [B] season (1 = naive). Returns [B, n, MAX_H]. Causal: only values <= origin.
    loc/scale cover the last n patches of x; anything before them (the coarse prefix's raw values) is
    used as older seasons, so seasons longer than the fine context (m=168, 288) still have a reference."""
    B, L = x.shape
    off = L - loc.shape[1] * PATCH                                             # prefix length (0 = none)
    obs = ~torch.isnan(x)
    ar = torch.arange(L, device=x.device)
    last = torch.where(obs, ar.expand_as(x), torch.full_like(x, -1, dtype=torch.long)).cummax(1).values
    ffill = torch.gather(torch.nan_to_num(x), 1, last.clamp(min=0))
    ends = torch.arange(off + PATCH - 1, L, PATCH, device=x.device)            # [n]
    k = torch.arange(1, MAX_H + 1, device=x.device)                            # [MAX_H]
    mm = m.clamp(min=1)[:, None, None]
    lag = mm * torch.div(k[None, None, :] + mm - 1, mm, rounding_mode="floor")  # m * ceil(k/m)
    idx = ends[None, :, None] + k[None, None, :] - lag                         # [B, n, MAX_H]
    tot, cnt = torch.zeros_like(idx, dtype=x.dtype), torch.zeros_like(idx, dtype=x.dtype)
    for s in range(SREF_AVG):                                                  # average the last SREF_AVG seasons
        i = idx - s * mm
        o = (i >= 0) & (torch.gather(last, 1, i.clamp(min=0).view(B, -1)).view_as(i) >= 0) & ((mm > 1) | (s == 0))
        tot = tot + torch.where(o, torch.gather(ffill, 1, i.clamp(min=0).view(B, -1)).view_as(i), 0.0)
        cnt = cnt + o
    ok = cnt > 0
    val = tot / cnt.clamp(min=1)
    ref = to_model_space(val, loc[..., None], scale[..., None])
    return torch.where(ok, ref, torch.zeros_like(ref))


def to_model_space(v, loc, scale):
    return torch.asinh((v - loc) / scale)


def from_model_space(q, loc, scale):
    return torch.sinh(q) * scale + loc

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LoopedDynamics(nn.Module):
    """LoopWM-style transition core (arXiv 2606.18208) for one latent step: a weight-shared block
    looped T times with a spectrally constrained residual, h <- A h + (1 - A) e + R(h, e), where
    A = exp(-softplus(dt) * exp(a)) is diagonal with eigenvalues in (0, 1), so the state stays
    bounded over any number of loops and rollout steps. The inner state is carried across rollout
    steps. T is sampled during training (Poisson) and can be raised at test time."""

    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(d))
        self.dt = nn.Parameter(torch.full((d,), -1.0))
        self.norm = nn.LayerNorm(d)
        self.block = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        nn.init.zeros_(self.block[-1].weight); nn.init.zeros_(self.block[-1].bias)

    def forward(self, e, h):
        A = torch.exp(-F.softplus(self.dt) * torch.exp(self.a))
        T = LOOPS if not self.training else 1 + int(np.random.poisson(LOOPS - 1))
        for _ in range(T):
            h = A * h + (1 - A) * e + self.block(torch.cat([self.norm(h), e], -1))
        return h


class CausalWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2 * PATCH + N_FRAME, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.pos = nn.Parameter(torch.zeros(1, N_PATCH, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 4 * D_MODEL, 0.0,
                                           batch_first=True, norm_first=True)
        self.lower = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.state_rnn = nn.GRU(D_MODEL, D_MODEL, batch_first=True)   # embedded world state
        self.upper = nn.TransformerEncoder(layer, N_LAYERS - 1, enable_nested_tensor=False)
        NT = N_PRE + N_PATCH                                          # prefix tokens precede fine tokens in time
        self.register_buffer("causal", torch.triu(torch.full((NT, NT), float("-inf")), 1))
        if PREFIX:                         # coarse long-context prefix: own embedding, positions and type
            self.pre_embed = nn.Sequential(nn.Linear(2 * PREFIX_PATCH + 3, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
            self.pre_pos = nn.Parameter(torch.zeros(1, N_PRE, D_MODEL))
            self.pre_type = nn.Parameter(0.02 * torch.randn(D_MODEL))
        self.to_state = nn.Linear(D_MODEL, D_MODEL)
        self.dynamics = LoopedDynamics(D_MODEL) if LOOPS else nn.GRUCell(D_MODEL, D_MODEL)
        self.cond = nn.Linear(D_MODEL, D_MODEL) if LOOPS else None      # observed world state -> per-step input
        self.step_emb = nn.Parameter(torch.zeros(K_STEPS, D_MODEL))
        self.decode = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, PATCH * NQ))
        self.ref_proj = nn.Linear(PATCH, D_MODEL)                     # seasonal memory read into the state
        self.ref_skip = nn.Parameter(torch.zeros(NQ))                 # learned copy weight per quantile
        self.skip_gate = nn.Linear(D_MODEL, NQ) if GATE else None     # per-series, per-step copy weight
        if GATE:
            nn.init.zeros_(self.skip_gate.weight); nn.init.zeros_(self.skip_gate.bias)
        if CTX_READ:                       # imagined states read the observed latent history (memory)
            self.read_q, self.read_k = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
            self.lag_emb = nn.Parameter(torch.zeros(N_PATCH, D_MODEL))
            self.read = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            nn.init.zeros_(self.read.out_proj.weight); nn.init.zeros_(self.read.out_proj.bias)

    def prefix_tokens(self, xpre, xfirst, loc, scale, cnt):
        """Coarse tokens for the history before the fine window. xpre: [B, PREFIX] raw, xfirst: [B, PATCH]
        the first fine patch. Every block is normalised in one frame: the robust ("last") stats at the end
        of the first fine patch computed over prefix + that patch, so all coarse tokens share a frame and
        nothing after the first fine patch is used (every origin has already seen it). Two features relate
        that frame to the first fine token's own frame. Returns [B, N_PRE, D]."""
        B = xpre.shape[0]
        ploc, pscale, _ = causal_stats(torch.cat([xpre, xfirst], 1))
        ploc, pscale = ploc[:, -1:], pscale[:, -1:]                   # [B, 1]
        xb = xpre.view(B, N_PRE, PREFIX_PATCH)
        obs = ~torch.isnan(xb)
        v = to_model_space(torch.nan_to_num(xb), ploc[..., None], pscale[..., None])
        has = (cnt[:, :1] > 0).float()                               # first fine patch observed
        dloc = torch.asinh((ploc - loc[:, :1]) / scale[:, :1]) * has
        dls = torch.log(pscale / scale[:, :1]).clamp(-5, 5) * has
        g = torch.cat([dloc, dls], -1)[:, None].expand(B, N_PRE, 2)
        frac = obs.float().mean(-1, keepdim=True)
        tok = torch.cat([torch.where(obs, v, torch.zeros_like(v)), obs.float(), g, frac], -1)
        return self.pre_embed(tok) + self.pre_pos + self.pre_type

    def states(self, x):
        """x: [B, PREFIX + L] raw values with NaN (oldest first). Returns latent states [B, n, D] for the
        n = L // PATCH fine patches and their per-patch loc/scale/count. Coarse prefix tokens (if any) are
        prepended in time: every fine token sees all of them under the causal mask."""
        if PREFIX:
            xpre, x = x[:, :PREFIX], x[:, PREFIX:]
        B, L = x.shape
        n = L // PATCH
        loc, scale, cnt = causal_stats(x)
        xp = x.view(B, n, PATCH)
        obs = ~torch.isnan(xp)
        z = to_model_space(torch.where(obs, xp, loc[..., None]), loc[..., None], scale[..., None])
        tok = torch.cat([torch.where(obs, z, torch.zeros_like(z)), obs.float()] + frame_features(loc, scale, cnt), -1)
        e = self.embed(tok) + self.pos[:, :n]
        if PREFIX:
            e = torch.cat([self.prefix_tokens(xpre, x[:, :PATCH], loc, scale, cnt), e], 1)
        T = e.shape[1]
        h = self.lower(e, mask=self.causal[:T, :T])
        state, _ = self.state_rnn(h)                                  # carried patch to patch
        h = self.upper(h + state, mask=self.causal[:T, :T])
        z = torch.tanh(self.to_state(h + state))
        return (z[:, T - n:] if PREFIX else z), loc, scale, cnt

    def memory(self, z, b, j):
        """Per-origin memory: states z[b, :j+1] with a learned lag embedding. Returns keys [N, n, D], ignore-mask."""
        n = z.shape[1]
        lag = j[:, None] - torch.arange(n, device=z.device)[None]                 # [N, n]
        mem = self.read_k(z[b] + self.lag_emb[lag.clamp(0, N_PATCH - 1)])
        return mem, lag < 0

    def rollout(self, z, ref=None, return_states=False, mem=None):
        """z: [N, D], ref: [N, MAX_H] seasonal reference -> quantiles in model space [N, K*PATCH, NQ].
        The transition cell imagines K future latent states without observations (prior);
        return_states also returns them [N, K, D] for the latent-consistency loss."""
        if ref is None or not SEAS_REF:
            ref = torch.zeros(z.shape[0], MAX_H, device=z.device)
        ref = ref.clamp(-20, 20)
        outs, zs = [], []
        c = self.cond(z) if LOOPS else 0
        for k in range(K_STEPS):
            z = self.dynamics(self.step_emb[k].expand(z.shape[0], -1) + c, z)
            zs.append(z)
        S = torch.stack(zs, 1)                                        # imagined states [N, K, D]
        if CTX_READ and mem is not None:
            keys, ignore = mem
            S = S + self.read(self.read_q(S), keys, keys, key_padding_mask=ignore, need_weights=False)[0]
        for k in range(K_STEPS):
            z = S[:, k]
            r = ref[:, k * PATCH:(k + 1) * PATCH]
            w = self.ref_skip + (self.skip_gate(z)[:, None] if GATE else 0)
            q = self.decode(z + self.ref_proj(r)).view(-1, PATCH, NQ) + r[..., None] * w
            outs.append(q)
        if return_states:
            return torch.cat(outs, 1), torch.stack(zs, 1)
        return torch.cat(outs, 1)


def pinball(pred, target, mask):
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(Q * err, (Q - 1) * err) * mask.unsqueeze(-1)
    return loss.sum() / mask.sum().clamp(min=1) / NQ


def latent_consistency(z, prior, keep):
    """World-model objective (JEPA / Dreamer style): the state imagined k patches ahead from origin j
    must match the encoder's state after actually observing patch j+k (stop-gradient target).
    This makes the transition cell a model of how the latent world state evolves, not just a
    forecast head. z: [B, n, D] posterior states, prior: [N_keep, K, D]."""
    B, n, D = z.shape
    k = torch.arange(1, K_STEPS + 1, device=z.device)
    j = torch.arange(n, device=z.device)
    tgt_idx = (j[:, None] + k[None, :])                                   # [n, K]
    ok = (tgt_idx < n)
    tgt = z.detach()[:, tgt_idx.clamp(max=n - 1)]                         # [B, n, K, D]
    tgt = tgt.reshape(B * n, K_STEPS, D)[keep]
    ok = ok[None].expand(B, n, K_STEPS).reshape(B * n, K_STEPS)[keep]
    err = (prior - tgt).pow(2).mean(-1)
    return (err * ok).sum() / ok.sum().clamp(min=1)


def train_loss(model, win, per):
    B = win.shape[0]
    z, loc, scale, cnt = model.states(win)                     # origins = every fine patch
    ref = seasonal_ref(win, loc, scale, per)                   # may read older seasons from the prefix
    pad = torch.full((B, MAX_H), float("nan"), device=win.device)
    fut = torch.cat([win[:, PREFIX:], pad], 1).unfold(1, MAX_H, PATCH)[:, 1:N_PATCH + 1]   # [B, n, MAX_H]
    valid = (cnt > 0)[..., None] & ~torch.isnan(fut)
    tgt = to_model_space(torch.nan_to_num(fut), loc[..., None], scale[..., None])
    keep = valid.any(-1).view(-1)                              # origins with at least one target
    mem = None
    if CTX_READ:
        bj = torch.arange(B * N_PATCH, device=win.device)[keep]
        mem = model.memory(z, bj // N_PATCH, bj % N_PATCH)
    if LATENT:
        pred, prior = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], True, mem)
        return pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float()) \
            + LATENT * latent_consistency(z, prior, keep)
    pred = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], mem=mem)
    return pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float())

# Training (fixed time budget)
# ---------------------------------------------------------------------------

model = CausalWorldModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
rng = np.random.default_rng(SEED)
ema = copy.deepcopy(model).requires_grad_(False) if EMA else model


# ---------------------------------------------------------------------------
# Inference-time methods (Dr. Tomas Rivera). Every method uses only the series' own context, is
# behind a flag, and the defaults reproduce train.py exactly.
#
#   TSAR_ENS="256+,128+,256-"   members: context crop x sign flip (as in train.py)
#   TSAR_TTA_XF="id,sqrt"       members: value transforms. Quantiles are equivariant under monotone maps,
#                               so q_tau(Y) = T^-1(q_tau(T(Y))). id | aff (x*1e3+7: the model is exactly
#                               affine-invariant, kept as a check) | sqrt | log (log1p); sqrt/log only on
#                               series whose context is >= 0, others fall back to id.
#   TSAR_TTA_ORIG="0,1,3,8"     members: origins. Drop the last j context points, forecast h+j, keep the
#                               last h. j < PATCH shifts the patch phase, j = 8, 16 reuses earlier latents.
#                               Steps a member cannot reach (h+j > MAX_H) get weight 0.
#   TSAR_TTA_OW=1.0             origin weight decay: member j gets weight OW**(j / PATCH)
#   TSAR_TTA_FUSE=mean          member fusion per quantile: mean (Vincentization) | median | mix (mixture)
#   TSAR_TTA_LONG=median        h > MAX_H: median feedback (train.py) | paths (S quantile-level sample paths,
#                               the next chunk is the equal-weight mixture over paths)
#   TSAR_TTA_PATHS=9            number of paths S; TSAR_TTA_PATH_MODE=keep | median | shuffle: what each
#                               path feeds back after the second chunk (its own level u_s, its median, or
#                               a fresh random permutation of the levels)
#   TSAR_TTT_STEPS=0            per-series test-time training: a FiLM (scale, shift) on the imagined states
#                               plus a log-spread of the quantiles, fit for this many Adam steps on rollouts
#                               from origins inside the series' own context. TSAR_TTT_LR, TSAR_TTT_L2,
#                               TSAR_TTT_ORIG (use only the last k origins, 0 = all)
#   TSAR_TTA_REF=0              1 = give adapt()/self_calibrate()/TTT the seasonal reference, as forecast()
#                               does (train.py's adapt and self_calibrate roll out with ref = 0)
#   TSAR_TTA_CLAMP=""          output clamp from the series' own context: nonneg (q >= 0 if the context is
#                               >= 0) | range (also q in [min - K*R, max + K*R], R = max - min, K =
#                               TSAR_TTA_CLAMP_K) . Applied per chunk, so long-horizon feedback is clamped.
#   Sign flip: TSAR_ENS="256+,256-". Needs a model trained with TSAR_TRAIN_FLIP=0.5 (negated rows) to help.
#   TSAR_SELFCAL, TSAR_SELFCAL_SHIFT, TSAR_TTA_STEPS: as in train.py (TTA_STEPS adapts shared weights on
#                               the whole batch, i.e. across series, so it is not per-series).
# ---------------------------------------------------------------------------

def tta_flags():
    """(Re)read the inference-time flags; the eval sweep calls this once per configuration."""
    global ENSEMBLE, TTA_STEPS, TTA_LR, TTA_SERIES, SELFCAL, SELFCAL_SHIFT, TTA_FUSE, TTA_XF, TTA_ORIG, TTA_OW
    global TTA_LONG, TTA_PATHS, TTA_PATH_MODE, TTA_REF, TTT_STEPS, TTT_LR, TTT_L2, TTT_ORIG, TTA_CLAMP, TTA_CLAMP_K
    e = os.environ.get
    ENSEMBLE = [(int(v[:-1]), 1 if v[-1] == "+" else -1) for v in e("TSAR_ENS", f"{CONTEXT}+").split(",")]
    TTA_STEPS = int(e("TSAR_TTA_STEPS", 0)); TTA_LR = float(e("TSAR_TTA_LR", 1e-3)); TTA_SERIES = int(e("TSAR_TTA_SERIES", 0))
    SELFCAL = float(e("TSAR_SELFCAL", 0)); SELFCAL_SHIFT = int(e("TSAR_SELFCAL_SHIFT", 0))
    TTA_FUSE = e("TSAR_TTA_FUSE", "mean")
    TTA_XF = e("TSAR_TTA_XF", "id").split(",")
    TTA_ORIG = [int(v) for v in e("TSAR_TTA_ORIG", "0").split(",")]
    TTA_OW = float(e("TSAR_TTA_OW", 1.0))
    TTA_LONG = e("TSAR_TTA_LONG", "median")
    TTA_PATHS = int(e("TSAR_TTA_PATHS", 9)); TTA_PATH_MODE = e("TSAR_TTA_PATH_MODE", "keep")
    TTA_REF = int(e("TSAR_TTA_REF", 0))
    TTT_STEPS = int(e("TSAR_TTT_STEPS", 0)); TTT_LR = float(e("TSAR_TTT_LR", 1e-2))
    TTT_L2 = float(e("TSAR_TTT_L2", 1e-2)); TTT_ORIG = int(e("TSAR_TTT_ORIG", 0))
    TTA_CLAMP = e("TSAR_TTA_CLAMP", ""); TTA_CLAMP_K = float(e("TSAR_TTA_CLAMP_K", 1.0))


tta_flags()


def qat(q, u):
    """Piecewise-linear quantile function through the NQ levels (linear extrapolation in the tails),
    evaluated at levels u [U]. q: [..., NQ] sorted. Returns [..., U]."""
    lv = Q.to(q.device)
    pos = (u.to(q.device) - lv[0]) / (lv[1] - lv[0])          # fractional index into the levels
    i0 = pos.floor().clamp(0, NQ - 2).long()
    fr = pos - i0
    return q[..., i0] + fr * (q[..., i0 + 1] - q[..., i0])


def fuse(qs, w, how):
    """qs: [M, B, h, NQ] member quantiles (raw space), w: [M, B, h] weights (0 = member does not cover).
    Returns [B, h, NQ]."""
    w = w / w.sum(0, keepdim=True).clamp(min=1e-12)
    if how == "mean":
        return (qs * w[..., None]).sum(0)
    if how == "median":
        return torch.nanquantile(torch.where(w[..., None] > 0, qs, torch.full_like(qs, float("nan"))), 0.5, dim=0)
    G = 16                                                  # mix: pool G levels per member, read off quantiles
    M, B, h, _ = qs.shape
    v = qat(qs.sort(-1).values, (torch.arange(G, device=qs.device) + 0.5) / G)          # [M, B, h, G]
    v = v.permute(1, 2, 0, 3).reshape(B, h, M * G)
    ww = (w / G)[..., None].expand(M, B, h, G).permute(1, 2, 0, 3).reshape(B, h, M * G)
    v, idx = v.sort(-1)
    ww = torch.gather(ww, -1, idx)
    c = ww.cumsum(-1) - ww / 2                              # mid-point CDF of the pooled sample
    tau = Q.to(qs.device).expand(B, h, NQ).contiguous()
    j = torch.searchsorted(c.contiguous(), tau).clamp(1, M * G - 1)
    c0, c1 = torch.gather(c, -1, j - 1), torch.gather(c, -1, j)
    v0, v1 = torch.gather(v, -1, j - 1), torch.gather(v, -1, j)
    f = ((tau - c0) / (c1 - c0).clamp(min=1e-12)).clamp(0, 1)
    return v0 + f * (v1 - v0)


def roll(m, z, ref=None, mem=None, film=None):
    """m.rollout, optionally with a per-row FiLM (g, b: [N, D]) on the imagined states and a per-row
    log-spread s [N, 1] of the quantiles around the median (model space). film=None is m.rollout."""
    if film is None:
        return m.rollout(z, ref, mem=mem)
    g, b, s = film
    if ref is None or not SEAS_REF:
        ref = torch.zeros(z.shape[0], MAX_H, device=z.device)
    ref = ref.clamp(-20, 20)
    zs, c = [], m.cond(z) if LOOPS else 0
    for k in range(K_STEPS):
        z = m.dynamics(m.step_emb[k].expand(z.shape[0], -1) + c, z)
        zs.append(z)
    S = torch.stack(zs, 1)
    if CTX_READ and mem is not None:
        keys, ignore = mem
        S = S + m.read(m.read_q(S), keys, keys, key_padding_mask=ignore, need_weights=False)[0]
    S = S * (1 + g[:, None]) + b[:, None]
    outs = []
    for k in range(K_STEPS):
        zk, r = S[:, k], ref[:, k * PATCH:(k + 1) * PATCH]
        w = m.ref_skip + (m.skip_gate(zk)[:, None] if GATE else 0)
        outs.append(m.decode(zk + m.ref_proj(r)).view(-1, PATCH, NQ) + r[..., None] * w)
    q = torch.cat(outs, 1)
    med = q[..., NQ // 2:NQ // 2 + 1]
    return med + torch.exp(s)[:, None] * (q - med)


def inside_targets(x, n, loc, scale, cnt):
    """Futures of origins 0..n-1 that lie inside the context x, in each origin's model space."""
    B = x.shape[0]
    fut = torch.cat([x[:, PREFIX:], torch.full((B, MAX_H), float("nan"), device=x.device)], 1).unfold(1, MAX_H, PATCH)[:, 1:n + 1]
    valid = (cnt[:, :n] > 0)[..., None] & ~torch.isnan(fut)
    return to_model_space(torch.nan_to_num(fut), loc[:, :n, None], scale[:, :n, None]), valid


def context_loss(m, x, z, loc, scale, cnt, offset=None, ref=None):
    """Multi-origin pinball loss using only futures inside the context window x: [B, L]."""
    B, n = z.shape[:2]
    if offset is not None:
        z = z + offset[:, None]
    tgt, valid = inside_targets(x, n, loc, scale, cnt)
    keep = valid.any(-1).view(-1)
    r = ref.reshape(B * n, MAX_H)[keep] if ref is not None else None
    pred = m.rollout(z.reshape(B * n, -1)[keep], r)
    return pinball(pred, tgt.reshape(B * n, MAX_H)[keep], valid.reshape(B * n, MAX_H)[keep].float())


def adapt(x, ref=None):
    """T9 test-time learning: freeze the encoder, fine-tune the dynamics and decoder (and optionally a
    per-series latent offset) on rollouts from origins inside the context. Returns the adapted model
    and offset. Only the given context is used, so there is no look-ahead."""
    m = copy.deepcopy(ema)
    with torch.no_grad():
        z, loc, scale, cnt = m.states(x)
    params = list(m.dynamics.parameters()) + list(m.decode.parameters()) + [m.step_emb]
    offset = torch.zeros(x.shape[0], D_MODEL, device=x.device, requires_grad=True) if TTA_SERIES else None
    for p_ in params:
        p_.requires_grad_(True)
    o = torch.optim.Adam(params + ([offset] if TTA_SERIES else []), lr=TTA_LR)
    with torch.enable_grad():
        for _ in range(TTA_STEPS):
            loss = context_loss(m, x, z[:, :-1], loc[:, :-1], scale[:, :-1], cnt[:, :-1], offset,
                                ref[:, :-1] if ref is not None else None)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    return m, offset


def ttt_fit(m, x, z, loc, scale, cnt, ref):
    """Per-series test-time training: fit FiLM (g, b) and a log-spread s for each series separately
    (parameters are per row, so no information is shared across series) on rollouts from origins
    inside that series' own context. Returns (g, b, s) for the B series."""
    B, n, D = z.shape
    n = n - 1                                          # the last origin has no future inside the context
    tgt, valid = inside_targets(x, n, loc, scale, cnt)
    lo = max(0, n - TTT_ORIG) if TTT_ORIG else 0
    zz, rr, tgt, valid = z[:, lo:n], ref[:, lo:n], tgt[:, lo:], valid[:, lo:]
    k = n - lo
    bi = torch.arange(B, device=x.device).repeat_interleave(k)
    keep = valid.reshape(B * k, MAX_H).any(-1)
    bi = bi[keep]
    zf, rf = zz.reshape(B * k, D)[keep], rr.reshape(B * k, MAX_H)[keep]
    tf, vf = tgt.reshape(B * k, MAX_H)[keep], valid.reshape(B * k, MAX_H)[keep].float()
    g = torch.zeros(B, D, device=x.device, requires_grad=True)
    b = torch.zeros(B, D, device=x.device, requires_grad=True)
    s = torch.zeros(B, 1, device=x.device, requires_grad=True)
    if len(bi) == 0:
        return g.detach(), b.detach(), s.detach()
    o = torch.optim.Adam([g, b, s], lr=TTT_LR)
    with torch.enable_grad():
        for _ in range(TTT_STEPS):
            pred = roll(m, zf, rf, film=(g[bi], b[bi], s[bi]))
            err = tf[..., None] - pred
            pl = (torch.maximum(Q * err, (Q - 1) * err) * vf[..., None]).sum((1, 2))
            per = torch.zeros(B, device=x.device).index_add(0, bi, pl)
            den = torch.zeros(B, device=x.device).index_add(0, bi, vf.sum(1)).clamp(min=1) * NQ
            loss = (per / den).sum() + TTT_L2 * (g.pow(2).mean(1) + b.pow(2).mean(1) + s.pow(2)[:, 0]).sum()
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    return g.detach(), b.detach(), s.detach()


def self_calibrate(m, x, z, loc, scale, h, q, ref=None, film=None):
    """T10 self-improvement at inference: replay the model's own forecasts from earlier origins inside
    the context, score them against what actually happened, and pick the per-series interval spread
    (and optionally level shift, both in model space) that would have minimised pinball loss.
    Shrunk toward the identity by SELFCAL. q: [B, h, NQ] model-space forecast from the last origin."""
    B, n = z.shape[:2]
    hh = min(h, MAX_H)
    fl = None if film is None else tuple(t.repeat_interleave(n, 0) for t in film)
    past = roll(m, z.reshape(B * n, -1), ref.reshape(B * n, MAX_H) if ref is not None else None, film=fl)
    past = past.view(B, n, MAX_H, NQ)[:, :, :hh]
    fut = torch.cat([x[:, PREFIX:], torch.full((B, MAX_H), float("nan"), device=x.device)], 1).unfold(1, MAX_H, PATCH)[:, 1:n + 1, :hh]
    tgt = to_model_space(torch.nan_to_num(fut), loc[..., None], scale[..., None])
    valid = ~torch.isnan(fut)
    r = n // 2                                                   # recent origins only
    past, tgt, valid = past[:, r:], tgt[:, r:], valid[:, r:]
    med = past[..., NQ // 2:NQ // 2 + 1]
    S = SPREADS.to(x.device)
    T = SHIFTS.to(x.device) if SELFCAL_SHIFT else torch.zeros(1, device=x.device)
    qq = Q.view(1, 1, 1, NQ, 1)
    w = valid[..., None, None].float()
    cnt = valid.sum((1, 2)).clamp(min=1)[:, None]
    scores = []
    for t in T:                                                  # [B, r, hh, NQ, nS] per shift
        cand = (med + t)[..., None] + S * (past - med)[..., None]
        err = tgt[..., None, None] - cand
        scores.append((torch.maximum(qq * err, (qq - 1) * err) * w).sum((1, 2, 3)) / cnt)
    score = torch.stack(scores, -1)                              # [B, nS, nT]
    best = score.view(B, -1).argmin(1)
    s_best, t_best = S[best // len(T)], T[best % len(T)]
    has = valid.sum((1, 2)) > 0
    s_fin = torch.where(has, 1 + SELFCAL * (s_best - 1), torch.ones_like(s_best))
    t_fin = torch.where(has, SELFCAL * t_best, torch.zeros_like(t_best))
    m0 = q[..., NQ // 2:NQ // 2 + 1]
    return m0 + t_fin[:, None, None] + s_fin[:, None, None] * (q - m0)


@torch.no_grad()
def predict(ctxs, h, m_season, freq):
    if h > MAX_H and TTA_LONG == "paths":
        return predict_paths(ctxs, h, m_season, freq)
    if h > MAX_H:                          # long horizons: roll the world model forward in MAX_H chunks,
        out, ctxs = [], [np.asarray(c, dtype=np.float32) for c in ctxs]   # feeding back the median
        for s in range(0, h, MAX_H):
            q = predict_chunk(ctxs, min(MAX_H, h - s), m_season, freq)
            out.append(q)
            ctxs = [np.concatenate([c, q[i, NQ // 2]])[-TCTX:] for i, c in enumerate(ctxs)]
        return np.concatenate(out, -1)
    return predict_chunk(ctxs, h, m_season, freq)


def predict_paths(ctxs, h, m_season, freq):
    """h > MAX_H with sample-path feedback: the first chunk is the usual forecast; S paths feed back its
    quantile function at stratified levels u_s (comonotone within a chunk), each path is rolled forward,
    and every later chunk is the equal-weight mixture of the S paths' forecasts."""
    S = TTA_PATHS
    B = len(ctxs)
    ctxs = [np.asarray(c, dtype=np.float32) for c in ctxs]
    u = (torch.arange(S) + 0.5) / S
    q = predict_chunk(ctxs, MAX_H, m_season, freq)                               # [B, NQ, MAX_H]
    out = [q]
    qt = torch.from_numpy(np.sort(q, 1)).permute(0, 2, 1)                          # [B, MAX_H, NQ]
    vals = qat(qt, u).numpy()                                                    # [B, MAX_H, S]
    paths = [np.concatenate([c, vals[i, :, s]])[-TCTX:] for i, c in enumerate(ctxs) for s in range(S)]
    lev = np.tile(np.arange(S), B)                                               # level index per path
    prng = np.random.default_rng(0)
    for s0 in range(MAX_H, h, MAX_H):
        hh = min(MAX_H, h - s0)
        qp = predict_chunk(paths, hh, m_season, freq)                            # [B*S, NQ, hh]
        qpt = torch.from_numpy(np.sort(qp, 1)).permute(0, 2, 1).reshape(B, S, hh, NQ)
        out.append(fuse(qpt.permute(1, 0, 2, 3), torch.ones(S, B, hh), "mix").permute(0, 2, 1).numpy())
        if s0 + hh < h:
            if TTA_PATH_MODE == "median":
                fb = qp[:, NQ // 2]
            else:
                if TTA_PATH_MODE == "shuffle":
                    lev = np.concatenate([prng.permutation(S) for _ in range(B)])
                allv = qat(qpt.reshape(B * S, hh, NQ), u).numpy()                # [B*S, hh, S]
                fb = allv[np.arange(B * S), :, lev]
            paths = [np.concatenate([c, fb[i]])[-TCTX:] for i, c in enumerate(paths)]
    return np.concatenate(out, -1)


XF = {"id": (lambda v: v, lambda q: q),
      "aff": (lambda v: v * 1e3 + 7.0, lambda q: (q - 7.0) / 1e3),
      "sqrt": (lambda v: np.sqrt(v), lambda q: q.clamp(min=0) ** 2),
      "log": (lambda v: np.log1p(v), lambda q: torch.expm1(q))}


@torch.no_grad()
def predict_chunk(ctxs, h, m_season, freq):
    ema.eval()
    out = []
    bs = 256 if TTT_STEPS or TTA_STEPS else 512
    for s in range(0, len(ctxs), bs):
        chunk = ctxs[s:s + bs]
        nonneg = np.array([np.nanmin(c) >= 0 if np.isfinite(c).any() else False for c in chunk])
        qs, ws = [], []
        for j in TTA_ORIG:                                # origins: drop the last j points
            for crop, sign in ENSEMBLE:                   # inference ensemble: context crops x sign flips
                for xf in TTA_XF:                         # monotone value transforms
                    fwd, inv = XF[xf]
                    use = nonneg if xf in ("sqrt", "log") else np.ones(len(chunk), bool)
                    ok = np.ones(len(chunk), bool)
                    ctx = np.full((len(chunk), TCTX), np.nan, dtype=np.float32)    # up to PREFIX + CONTEXT points
                    for i, c in enumerate(chunk):
                        if j:
                            ok[i] = np.isfinite(c[:-j]).sum() >= 2 if len(c) > j else False
                            c = c[:-j] if ok[i] else c
                        c = c[-TCTX:] if crop == CONTEXT else c[-crop:] if len(c) >= 2 * crop // 3 else c[-CONTEXT:]
                        ctx[i, TCTX - len(c):] = fwd(c) if use[i] else c
                    hj = min(h + j, MAX_H)
                    q = forecast(sign * torch.from_numpy(ctx).to(DEVICE), hj, m_season, sign)[:, j:]   # [B, <=h, NQ]
                    ut = torch.from_numpy(use).to(DEVICE)[:, None, None]
                    q = torch.where(ut, inv(q), q)
                    w = torch.zeros(len(chunk), h, device=DEVICE)
                    w[:, :q.shape[1]] = (TTA_OW ** (j / PATCH)) * torch.from_numpy(ok).float().to(DEVICE)[:, None]
                    if q.shape[1] < h:
                        q = torch.cat([q, torch.zeros(len(chunk), h - q.shape[1], NQ, device=DEVICE)], 1)
                    qs.append(q); ws.append(w)
        if len(qs) == 1:
            f = qs[0]
        else:
            f = fuse(torch.stack(qs).float().cpu(), torch.stack(ws).float().cpu(), TTA_FUSE)
        if TTA_CLAMP:
            f = f.float().cpu()
            lo = np.array([np.nanmin(c[-TCTX:]) if np.isfinite(c).any() else np.nan for c in chunk], np.float32)
            hi = np.array([np.nanmax(c[-TCTX:]) if np.isfinite(c).any() else np.nan for c in chunk], np.float32)
            lo_t, hi_t = torch.from_numpy(lo)[:, None, None], torch.from_numpy(hi)[:, None, None]
            if TTA_CLAMP == "range":
                R = hi_t - lo_t
                f = torch.where(torch.isfinite(R), torch.minimum(torch.maximum(f, lo_t - TTA_CLAMP_K * R), hi_t + TTA_CLAMP_K * R), f)
            f = torch.where(torch.from_numpy(lo >= 0)[:, None, None], f.clamp(min=0), f)
        out.append(f.permute(0, 2, 1).float().cpu().numpy())
    return np.concatenate(out)


def forecast(x, h, m_season, sign=1):
    """Quantile forecast [B, h, NQ] in raw space for contexts x (already multiplied by sign)."""
    need_ref = TTT_STEPS or (TTA_REF and (TTA_STEPS or SELFCAL))
    ms = torch.full((len(x),), int(m_season), device=DEVICE)
    if TTA_STEPS:
        ref0 = None
        if TTA_REF:
            _, l0, s0, _ = ema.states(x)
            ref0 = seasonal_ref(x, l0, s0, ms)
        m, offset = adapt(x, ref0)
    else:
        m, offset = ema, None
    z, loc, scale, cnt = m.states(x)
    zl = z[:, -1] if offset is None else z[:, -1] + offset
    ref_all = seasonal_ref(x, loc, scale, ms)
    ref = ref_all[:, -1]
    mem = None
    if CTX_READ:
        B, n = z.shape[:2]
        mem = m.memory(z, torch.arange(B, device=z.device), torch.full((B,), n - 1, device=z.device))
    film = ttt_fit(m, x, z, loc, scale, cnt, ref_all) if TTT_STEPS else None
    q = roll(m, zl, ref, mem=mem, film=film)[:, :h]      # origin = last context patch
    if SELFCAL:
        q = self_calibrate(m, x, z, loc, scale, h, q, ref_all if (TTA_REF or TTT_STEPS) else None, film)
    q = from_model_space(q, loc[:, -1, None, None], scale[:, -1, None, None])
    return q if sign > 0 else -q.flip(-1)                 # reflect: q_tau(x) = -q_{1-tau}(-x)


def tta_eval():
    """Evaluate a trained checkpoint on bench VAL under a sweep of inference configurations.
    TSAR_TTA_SWEEP: file with one config per line, "name KEY=VAL KEY=VAL ..." (first line = reference).
    TSAR_TTA_TASKS: ';'-separated val task names (default all). TSAR_TTA_MAXF: max forecasts per task
    (fixed-seed subsample of windows, 0 = all). TSAR_TTA_OUT: directory for per-task CSVs."""
    global ema
    ck = torch.load(TTA_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("ema", ck["model"]))
    model.eval().requires_grad_(False)
    ema = model
    out_dir = _env("TSAR_TTA_OUT", "runs/tta")
    os.makedirs(out_dir, exist_ok=True)
    names = _env("TSAR_TTA_TASKS")
    tasks = BN.load_tasks("val", names.split(";") if names else None)        # validation split only
    maxf = int(_env("TSAR_TTA_MAXF", 0))
    if maxf:
        for t in tasks:
            if len(t["contexts"]) > maxf:
                keep = np.sort(np.random.default_rng(0).permutation(len(t["contexts"]))[:maxf])
                t["contexts"] = [t["contexts"][i] for i in keep]
                t["targets"] = t["targets"][keep]
                t["series_id"] = [t["series_id"][i] for i in keep]
                t["window"] = t["window"][keep]
    print(f"{len(tasks)} val tasks, {sum(len(t['contexts']) for t in tasks)} forecasts, ckpt {TTA_CKPT}", flush=True)
    sweep = [l.split() for l in open(_env("TSAR_TTA_SWEEP")) if l.strip() and not l.startswith("#")] \
        if _env("TSAR_TTA_SWEEP") else [["env"]]
    predict([c.copy() for c in tasks[0]["contexts"][:64]], tasks[0]["horizon"], tasks[0]["season"], tasks[0]["freq"])  # warm-up
    ref_df = None
    for row in sweep:
        name, cfg = row[0], dict(kv.split("=", 1) for kv in row[1:])
        saved = {k: os.environ.get(k) for k in cfg}
        os.environ.update(cfg)
        tta_flags()
        torch.manual_seed(0)
        r = BN.evaluate(predict, split="val", verbose=False, tasks=tasks)
        df = r["per_dataset"]
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        line = (f"{name:14s} wql_rel {r['wql_rel']:.4f} mase_rel {r['mase_rel']:.4f} cov80_err {r['cov80_error']:.4f} "
                f"cov80 {df['cov80'].mean():.3f} fam_wql {r['wql_rel_family']:.4f} pred_s {r['predict_seconds']:.1f}")
        if ref_df is None:
            ref_df, ref_s = df, r["predict_seconds"]
        else:
            c = BN.compare(df, ref_df, "wql")
            cm = BN.compare(df, ref_df, "mase")
            line += (f" | vs ref: wql x{c['ratio']:.4f} famCI [{c['ci95_family'][0]:.3f},{c['ci95_family'][1]:.3f}]"
                     f" W/L {c['wins']}/{c['losses']} mase x{cm['ratio']:.4f} cost x{r['predict_seconds'] / max(ref_s, 1e-9):.1f}")
        print(line, flush=True)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    tta_flags()


if TTA_CKPT:
    tta_eval()
    sys.exit(0)


CKPT = os.path.join(P.CACHE_DIR, "ckpt", f"{RUN}.pt") if RUN else None
step, smooth, done = 0, None, 0.0          # done = training seconds completed in earlier sessions
if CKPT and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if ck.get("zs", 0) != ZS:
        raise SystemExit(f"{RUN} was trained with TSAR_ZS={ck.get('zs', 0)}; refusing to resume with TSAR_ZS={ZS}")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if EMA:
        ema.load_state_dict(ck.get("ema", ck["model"]))
    step, done, rng = ck["step"], ck["done"], ck["rng"]
    print(f"resumed {RUN} at step {step}, {done / 60:.1f} min trained", flush=True)


def save_ckpt(elapsed):
    if CKPT:
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "done": elapsed, "rng": rng, "zs": ZS}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)


t_start = time.time() - done
last_ckpt = last_eval = time.time()
while True:
    elapsed = time.time() - t_start
    prog = step / STEPS if STEPS else elapsed / BUDGET   # TSAR_STEPS: fixed-step budget, immune to machine load
    PROG = prog
    if prog >= 1:
        break
    if SCHED == "wsd":                     # warmup-stable-decay: constant, then cosine over the last 20%
        f = max(0.0, (prog - 0.8) / 0.2)
    else:
        f = prog
    lr = LR * min(1.0, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))
    for g in opt.param_groups:
        g["lr"] = lr
    loss = train_loss(model, *sample_batch(rng))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if EMA:
        with torch.no_grad():
            d = min(EMA, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
    smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
    if step % 100 == 0:
        print(f"step {step:5d}  loss {smooth:.4f}  lr {lr:.2e}  {elapsed:5.0f}s", flush=True)
    step += 1
    if CKPT and time.time() - last_ckpt > 600:
        save_ckpt(time.time() - t_start); last_ckpt = time.time()
    if EVAL_EVERY and time.time() - last_eval > EVAL_EVERY:
        t0 = time.time()
        r = EVALUATE(predict, split="val", verbose=False)
        model.train()
        print(f"EVAL step {step} minutes {(time.time() - t_start) / 60:.1f} val_wql_rel {r['wql_rel']:.4f} "
              f"val_mase_rel {r['mase_rel']:.4f} cov80_err {r['cov80_error']:.4f}", flush=True)
        t_start += time.time() - t0                       # evaluation time does not count as training
        last_eval = time.time()
training_seconds = time.time() - t_start
save_ckpt(training_seconds)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

split = "test" if os.environ.get("TSAR_SPLIT") == "test" else "val"
res = EVALUATE(predict, split=split)
peak_mb = torch.mps.driver_allocated_memory() / 2**20 if DEVICE == "mps" else 0.0
print("---")
print(f"{split}_wql_rel:      {res['wql_rel']:.6f}")
print(f"{split}_mase_rel:     {res['mase_rel']:.6f}")
print(f"{split}_cov80_error:  {res['cov80_error']:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"predict_seconds:  {res['predict_seconds']:.1f}")
print(f"total_seconds:    {time.time() - t_start:.1f}")
print(f"peak_mem_mb:      {peak_mb:.1f}")
print(f"num_params_M:     {n_params / 1e6:.3f}")
print(f"num_steps:        {step}")
pdx = res["per_dataset"].set_index("dataset")
zs = float(np.exp(np.log(pdx.loc[ZS_SETS, "wql_rel"]).mean())) if not BENCH else float("nan")
if BENCH:
    print(f"family_wql_rel:   {res['wql_rel_family']:.6f}")
print(f"zs4_wql_rel:      {zs:.6f}   ({'held out' if ZS else 'trained on'}: {', '.join(ZS_SETS)})")
if split == "test" and not BENCH:          # confirmatory run: also score the 5 never-trained holdout datasets
    ho = P.evaluate(predict, split="holdout")
    print(f"holdout_wql_rel:  {ho['wql_rel']:.6f}")
    print(f"holdout_mase_rel: {ho['mase_rel']:.6f}")
    print(f"holdout_cov80_error: {ho['cov80_error']:.6f}")
