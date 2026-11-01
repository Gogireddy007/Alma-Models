"""
Latent world-model forecaster: train_dyn.py = copy of train.py with transition-dynamics flags
TSAR_JUMP and TSAR_DYN_IN (default off = identical behaviour). Queues read train.py; this file is experimental.

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
SEED = int(_env("TSAR_SEED", SEED))
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
SEAS_REF = int(_env("TSAR_SEAS_REF", 1))                   # T7 seasonal memory pathway in the decoder
NORM = _env("TSAR_NORM", "last")                          # "mean" | "last" (see causal_stats)
FRAME = int(_env("TSAR_FRAME", 1)); FRAMEFIX = int(_env("TSAR_FRAMEFIX", 0)); SREF_AVG = int(_env("TSAR_SREF_AVG", 1)); CTX_READ = int(_env("TSAR_CTX_READ", 0))
# Transition-dynamics flags (train_dyn.py only; all default 0 = identical to train.py).
# TSAR_JUMP: horizon-conditioned direct jump j_k = J(z_0, k) fused into the rolled state by a zero-init gated
#   residual. 1 = fuse for decoding only (recurrence unchanged), 2 = anchored rollout: the fused state is carried
#   into the next transition, so every step is re-anchored to the origin and errors cannot compound freely.
# TSAR_DYN_IN: series-specific transition input (the GRU input is otherwise the shared step_emb[k] only).
#   bit 1 = origin conditioning cond(z_0), bit 2 = seasonal forcing: the step-k seasonal reference patch (causal,
#   values <= origin) enters the transition as a known exogenous "action". 3 = both. Zero-init projections.
JUMP = int(_env("TSAR_JUMP", 0))
DYN_IN = int(_env("TSAR_DYN_IN", 0))
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

if BENCH:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import bench as BN
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
        # New modules are created last so all pre-existing parameters get the same init as train.py.
        if JUMP:                           # direct jump: z_0 -> state at horizon k (one hidden layer, FiLM by k)
            self.jump_norm = nn.LayerNorm(D_MODEL)
            self.jump_in = nn.Linear(D_MODEL, D_MODEL)
            self.jump_h = nn.Parameter(0.02 * torch.randn(K_STEPS, D_MODEL))    # horizon embedding
            self.jump_out = nn.Linear(D_MODEL, D_MODEL)
            self.jump_gate = nn.Linear(2 * D_MODEL, D_MODEL)                    # per-dim trust in the jump
            nn.init.zeros_(self.jump_out.weight); nn.init.zeros_(self.jump_out.bias)
        if DYN_IN & 1 and not LOOPS:       # LoopedDynamics already has self.cond(z_0)
            self.dyn_cond = nn.Linear(D_MODEL, D_MODEL)
            nn.init.zeros_(self.dyn_cond.weight); nn.init.zeros_(self.dyn_cond.bias)
        if DYN_IN & 2:
            self.dyn_ref = nn.Linear(PATCH, D_MODEL)
            nn.init.zeros_(self.dyn_ref.weight); nn.init.zeros_(self.dyn_ref.bias)

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
        if DYN_IN & 1 and not LOOPS:
            c = self.dyn_cond(z)                                       # origin-conditioned transition input
        if JUMP:
            jb = self.jump_in(self.jump_norm(z))                       # [N, D], shared by all horizons
        dec = []                                                      # states that are decoded
        for k in range(K_STEPS):
            u = self.step_emb[k].expand(z.shape[0], -1) + c
            if DYN_IN & 2:
                u = u + self.dyn_ref(ref[:, k * PATCH:(k + 1) * PATCH])   # seasonal forcing ("action")
            z = self.dynamics(u, z)
            if JUMP:
                j = self.jump_out(F.gelu(jb + self.jump_h[k]))           # direct jump to horizon k
                g = torch.sigmoid(self.jump_gate(torch.cat([z, j], -1)))
                zf = z + g * j                                         # zero-init: equals z at init
                if JUMP == 2:
                    z = zf                                             # anchored rollout: carry the fused state
                dec.append(zf)
            zs.append(z)
        S = torch.stack(dec if JUMP else zs, 1)                       # imagined states [N, K, D]
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


def context_loss(m, x, z, loc, scale, cnt, offset=None):
    """Multi-origin pinball loss using only futures inside the context window x: [B, L]."""
    B, n = z.shape[:2]
    if offset is not None:
        z = z + offset[:, None]
    fut = torch.cat([x[:, PREFIX:], torch.full((B, MAX_H), float("nan"), device=x.device)], 1).unfold(1, MAX_H, PATCH)[:, 1:n + 1]
    valid = (cnt > 0)[..., None] & ~torch.isnan(fut)
    tgt = to_model_space(torch.nan_to_num(fut), loc[..., None], scale[..., None])
    keep = valid.any(-1).view(-1)
    pred = m.rollout(z.reshape(B * n, -1)[keep])
    return pinball(pred, tgt.reshape(B * n, MAX_H)[keep], valid.reshape(B * n, MAX_H)[keep].float())


def adapt(x):
    """T9 test-time learning: freeze the encoder, fine-tune the dynamics and decoder (and optionally a
    per-series latent offset) on rollouts from origins inside the context. Returns the adapted model
    and offset. Only the given context is used, so there is no look-ahead."""
    m = copy.deepcopy(ema)
    with torch.no_grad():
        z, loc, scale, cnt = m.states(x)
    params = list(m.dynamics.parameters()) + list(m.decode.parameters()) + [m.step_emb]
    params += [p_ for n_, p_ in m.named_parameters() if n_.startswith(("jump_", "dyn_"))]
    offset = torch.zeros(x.shape[0], D_MODEL, device=x.device, requires_grad=True) if TTA_SERIES else None
    for p_ in params:
        p_.requires_grad_(True)
    o = torch.optim.Adam(params + ([offset] if TTA_SERIES else []), lr=TTA_LR)
    with torch.enable_grad():
        for _ in range(TTA_STEPS):
            loss = context_loss(m, x, z[:, :-1], loc[:, :-1], scale[:, :-1], cnt[:, :-1], offset)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    return m, offset


def self_calibrate(m, x, z, loc, scale, h, q):
    """T10 self-improvement at inference: replay the model's own forecasts from earlier origins inside
    the context, score them against what actually happened, and pick the per-series interval spread
    (and optionally level shift, both in model space) that would have minimised pinball loss.
    Shrunk toward the identity by SELFCAL. q: [B, h, NQ] model-space forecast from the last origin."""
    B, n = z.shape[:2]
    hh = min(h, MAX_H)
    past = m.rollout(z.reshape(B * n, -1)).view(B, n, MAX_H, NQ)[:, :, :hh]
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
    if h > MAX_H:                          # long horizons: roll the world model forward in MAX_H chunks,
        out, ctxs = [], [np.asarray(c, dtype=np.float32) for c in ctxs]   # feeding back the median
        for s in range(0, h, MAX_H):
            q = predict_chunk(ctxs, min(MAX_H, h - s), m_season, freq)
            out.append(q)
            ctxs = [np.concatenate([c, q[i, NQ // 2]])[-TCTX:] for i, c in enumerate(ctxs)]
        return np.concatenate(out, -1)
    return predict_chunk(ctxs, h, m_season, freq)


@torch.no_grad()
def predict_chunk(ctxs, h, m_season, freq):
    ema.eval()
    out = []
    for s in range(0, len(ctxs), 512):
        chunk = ctxs[s:s + 512]
        qs = []
        for crop, sign in ENSEMBLE:                       # inference ensemble: context crops x sign flips
            ctx = np.full((len(chunk), TCTX), np.nan, dtype=np.float32)    # up to PREFIX + CONTEXT points
            for i, c in enumerate(chunk):
                c = c[-TCTX:] if crop == CONTEXT else c[-crop:] if len(c) >= 2 * crop // 3 else c[-CONTEXT:]
                ctx[i, TCTX - len(c):] = c
            qs.append(forecast(sign * torch.from_numpy(ctx).to(DEVICE), h, m_season, sign))
        out.append(torch.stack(qs).mean(0).permute(0, 2, 1).float().cpu().numpy())
    return np.concatenate(out)


def forecast(x, h, m_season, sign=1):
    """Quantile forecast [B, h, NQ] in raw space for contexts x (already multiplied by sign)."""
    m, offset = adapt(x) if TTA_STEPS else (ema, None)
    z, loc, scale, _ = m.states(x)
    zl = z[:, -1] if offset is None else z[:, -1] + offset
    ref = seasonal_ref(x, loc, scale, torch.full((len(x),), int(m_season), device=DEVICE))[:, -1]
    mem = None
    if CTX_READ:
        B, n = z.shape[:2]
        mem = m.memory(z, torch.arange(B, device=z.device), torch.full((B,), n - 1, device=z.device))
    q = m.rollout(zl, ref, mem=mem)[:, :h]                # origin = last context patch
    if SELFCAL:
        q = self_calibrate(m, x, z, loc, scale, h, q)
    q = from_model_space(q, loc[:, -1, None, None], scale[:, -1, None, None])
    return q if sign > 0 else -q.flip(-1)                 # reflect: q_tau(x) = -q_{1-tau}(-x)


    return np.concatenate(out)



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
