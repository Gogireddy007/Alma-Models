"""
train_v2.py (integration, Demir): train.py + the rung-1 winning flags from train_calib/dyn/latent/data/tta.py.
Every new flag defaults off; at defaults this file is bit-identical in behaviour to train.py.
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
import io
import json
import hashlib
import subprocess

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

# --- Frozen-checkpoint evaluation (docs/eval_rigor.md R5, audit_v2 S-2) ----------------------------
# TSAR_EVAL_ONLY=<ckpt path>: load that checkpoint, print its sha256, train nothing, score TSAR_SPLIT.
# The TSAR_* config stored in the checkpoint is restored into the environment before anything below
# reads it; an explicitly set value that differs is refused. TSAR_SPLIT=test is refused without it,
# without TSAR_CKPT_SHA256=<pre-registered sha256>, and without first reproducing the recorded VAL score.
CTRL_KEYS = {"TSAR_RUN", "TSAR_SPLIT", "TSAR_EVAL_MIN", "TSAR_DUMP", "TSAR_EVAL_ONLY", "TSAR_CKPT_SHA256",
             "TSAR_DEVICE", "TSAR_MINUTES"}                  # run control: may differ on resume
EVAL_FREE_KEYS = CTRL_KEYS | {"TSAR_SEED", "TSAR_STEPS"}     # training-only: irrelevant to frozen weights
INFER_KEYS = {"TSAR_ENS", "TSAR_TTA_STEPS", "TSAR_TTA_LR", "TSAR_TTA_SERIES", "TSAR_SELFCAL", "TSAR_SELFCAL_SHIFT"}
INFER_KEYS |= {"TSAR_CHUNK_MIX", "TSAR_CHUNK_SCALE", "TSAR_CHUNK_SCALE_MAX", "TSAR_CHUNK_MIX_M",   # train_v2: calib
               "TSAR_TTA_LONG", "TSAR_TTA_PATHS", "TSAR_TTA_PATH_MODE", "TSAR_TTA_CLAMP", "TSAR_TTA_CLAMP_K"}  # tta
if _env("TSAR_STATE_OFF") == "zero":     # A3 ablation zeroes the imagined states at eval only ("copy" changes
    INFER_KEYS |= {"TSAR_STATE_OFF"}       # the trained weights and stays refused on a mismatch)
EVAL_ONLY = _env("TSAR_EVAL_ONLY")
IS_TEST = _env("TSAR_SPLIT") == "test"
if IS_TEST and not EVAL_ONLY:
    raise SystemExit("TSAR_SPLIT=test scores only a frozen checkpoint: set TSAR_EVAL_ONLY=<data/ckpt/NAME.pt> "
                     "(TEST must score the exact VAL-selected weights, never a retrained model)")


def _tsar_env():
    return {k: v for k, v in os.environ.items() if k.startswith("TSAR_")}


CK_SHA = CK0 = None
if EVAL_ONLY:
    _ck_bytes = open(EVAL_ONLY, "rb").read()                 # hash and load the very same bytes
    CK_SHA = hashlib.sha256(_ck_bytes).hexdigest()
    print(f"eval-only: {EVAL_ONLY}  sha256={CK_SHA}", flush=True)
    if _env("TSAR_CKPT_SHA256") and _env("TSAR_CKPT_SHA256") != CK_SHA:
        raise SystemExit(f"eval-only: checkpoint sha256 {CK_SHA} != TSAR_CKPT_SHA256 {_env('TSAR_CKPT_SHA256')}")
    if IS_TEST and not _env("TSAR_CKPT_SHA256"):
        raise SystemExit("TSAR_SPLIT=test needs TSAR_CKPT_SHA256=<pre-registered sha256 of the frozen checkpoint>")
    CK0 = torch.load(io.BytesIO(_ck_bytes), map_location="cpu", weights_only=False)
    del _ck_bytes
    if "cfg" not in CK0:
        raise SystemExit(f"eval-only: {EVAL_ONLY} has no stored run config (saved before the resume guard); "
                         "it cannot be scored as a frozen model")
    _stored = {k: v for k, v in CK0["cfg"]["env"].items() if k not in EVAL_FREE_KEYS}
    _cur = {k: v for k, v in _tsar_env().items() if k not in EVAL_FREE_KEYS}
    for _k in sorted(set(_stored) | set(_cur)):
        if _k in _cur and _cur[_k] != _stored.get(_k):
            if _k in INFER_KEYS and not IS_TEST:
                print(f"eval-only WARNING: inference setting {_k}={_cur[_k]} differs from training "
                      f"({_stored.get(_k)}): exploratory, VAL only", flush=True)
            else:
                raise SystemExit(f"eval-only: {_k}={_cur[_k]!r} but the checkpoint was trained with "
                                 f"{_stored.get(_k)!r}; unset it (the stored config is restored automatically)")
        elif _k not in _cur:
            os.environ[_k] = _stored[_k]
    print("eval-only config: " + " ".join(f"{k}={v}" for k, v in sorted(_stored.items())), flush=True)
# audit_v3 H-2: every TSAR_* key must be one this script reads (or a run-control key). An unread key would let
# a recipe claim a component (TSAR_TRANS, TSAR_COV, ...) that the trained model does not contain.
# Runs after the eval-only restore, so a checkpoint trained by another script with foreign keys is refused too.
READ_KEYS = {"TSAR_BENCH_GEP", "TSAR_GIFT_CLEAN", "TSAR_AUG", "TSAR_BATCH", "TSAR_BENCH", "TSAR_BENCH_EXTRA", "TSAR_BENCH_MAXLEN", "TSAR_CHUNK_MIX",
             "TSAR_CHUNK_MIX_M", "TSAR_CHUNK_SCALE", "TSAR_CHUNK_SCALE_MAX", "TSAR_CONTEXT",
             "TSAR_CTX_READ", "TSAR_DUALDEC", "TSAR_DUALDEC_CONS", "TSAR_DUALDEC_SG", "TSAR_DYN_IN",
             "TSAR_D_MODEL", "TSAR_EMA", "TSAR_ENS", "TSAR_FAR_J0", "TSAR_FAR_ORIG", "TSAR_FAR_W",
             "TSAR_FRAME", "TSAR_FRAMEFIX", "TSAR_GATE", "TSAR_HDEC", "TSAR_JEPA", "TSAR_JEPA_COV",
             "TSAR_JEPA_DIST", "TSAR_JEPA_FORM", "TSAR_JEPA_GAMMA", "TSAR_JEPA_GND", "TSAR_JEPA_L1",
             "TSAR_JEPA_PROJ", "TSAR_JEPA_REG", "TSAR_JEPA_SIGREG", "TSAR_JEPA_SIG_M", "TSAR_JEPA_VAR",
             "TSAR_JEPA_WARM", "TSAR_JUMP", "TSAR_KS_FRAC", "TSAR_LATENT", "TSAR_LATENT_EMA",
             "TSAR_LONG_H", "TSAR_LOOPS", "TSAR_LR", "TSAR_MAX_H", "TSAR_MIX_ALPHA", "TSAR_MS_COARSE",
             "TSAR_MS_STRIDE", "TSAR_NORM", "TSAR_N_HEADS", "TSAR_N_LAYERS", "TSAR_PREFIX",
             "TSAR_PREFIX_PATCH", "TSAR_PREFIX_SYN", "TSAR_RFL", "TSAR_RFL_CKPT", "TSAR_RFL_WARM",
             "TSAR_RSSM_FREE", "TSAR_RSSM_KL", "TSAR_RSSM_POST", "TSAR_RSSM_S", "TSAR_SCHED", "TSAR_SDD",
             "TSAR_SDD_K", "TSAR_SEAS_REF", "TSAR_SEED", "TSAR_SELFCAL", "TSAR_SELFCAL_SHIFT",
             "TSAR_SREF_AVG", "TSAR_SSM_N", "TSAR_STATE_OFF", "TSAR_STEPS", "TSAR_STRETCH", "TSAR_SYN2",
             "TSAR_SYN3", "TSAR_SYN4", "TSAR_SYN4_GAP", "TSAR_SYN5", "TSAR_SYN_CURR", "TSAR_SYN_FRAC", "TSAR_TRANS",
             "TSAR_TTA_CLAMP", "TSAR_TTA_CLAMP_K", "TSAR_TTA_LONG", "TSAR_TTA_LR", "TSAR_TTA_PATHS",
             "TSAR_TTA_PATH_MODE", "TSAR_TTA_SERIES", "TSAR_TTA_STEPS", "TSAR_WD", "TSAR_ZS"}
_unknown = sorted(k for k in _tsar_env() if k not in READ_KEYS | CTRL_KEYS)
if _unknown:
    raise SystemExit(f"{os.path.basename(__file__)}: refusing unknown TSAR_* keys (not implemented here, they would "
                     f"be silently ignored): {' '.join(_unknown)}")
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
# JEPA / BYOL-style latent prediction (see jepa_loss). TSAR_JEPA = weight (0 = off, default behaviour unchanged).
JEPA = float(_env("TSAR_JEPA", 0))
LATENT_EMA = float(_env("TSAR_LATENT_EMA", 0.996))         # target-encoder EMA decay (0 = stop-grad online encoder)
JEPA_DIST = _env("TSAR_JEPA_DIST", "cos")                  # "cos" (BYOL) | "sl1" (smooth-L1 on layer-normed vectors)
JEPA_FORM = _env("TSAR_JEPA_FORM", "")                     # "l1cos" (WorldTS): JEPA_L1 * |p - t|_1 + (1 - cos); overrides JEPA_DIST
JEPA_L1 = float(_env("TSAR_JEPA_L1", 1.0))                 # L1 weight relative to the cosine term (lit_review_v2: 1 or 3)
assert JEPA_FORM in ("", "l1cos"), "TSAR_JEPA_FORM must be l1cos or unset"
# Dual decoding (WorldTS, docs/lit_review_v2.md #2). TSAR_DUALDEC = weight (0 = off). For each origin t and step k
# the quantile head also decodes the encoder's own state at origin t+k (already computed, it has observed the
# target patch) against the same target, plus a consistency term pulling the imagined decode toward it.
DUALDEC = float(_env("TSAR_DUALDEC", 0))
DUALDEC_SG = int(_env("TSAR_DUALDEC_SG", 1))               # 1 = detach the encoder state (only the decoder learns from it)
# Weight of |q_imagined - sg q_observed| inside the term. Default 0 (was 1.0): the target is decoded from the same
# online encoder, so it is self-distillation with no anti-collapse; it drove the encoder's residual stream to a
# constant, saturated tanh(to_state) (state std over series 0.44 -> 0.005 in 30 steps from the v2_base ckpt), and
# the model fell back to the seasonal-reference path with large constant quantile tails that sinh() blew up
# (dualdec_05 VAL wql_rel 16.6, + JEPA 2.3e6). CONS=0 keeps the state healthy. Set it > 0 only to reproduce that.
DUALDEC_CONS = float(_env("TSAR_DUALDEC_CONS", 0.0))
JEPA_PROJ = int(_env("TSAR_JEPA_PROJ", 64))                # projector / predictor width
JEPA_VAR = float(_env("TSAR_JEPA_VAR", 1.0))               # VICReg variance hinge weight (anti-collapse)
JEPA_COV = float(_env("TSAR_JEPA_COV", 0.0))               # VICReg covariance weight
JEPA_WARM = float(_env("TSAR_JEPA_WARM", 0.2))             # weight ramps linearly 0 -> JEPA over this share of training
JEPA_GAMMA = float(_env("TSAR_JEPA_GAMMA", 1.0))           # per-step discount gamma^(k-1) on the k-step term
JEPA_GND = float(_env("TSAR_JEPA_GND", 0.1))               # EIDOS grounding: embedding -> quantiles of its observed patch
JEPA_REG = _env("TSAR_JEPA_REG", "vicreg")                 # anti-collapse: "vicreg" (var/cov) | "sigreg" (LeWM) | "none"
JEPA_SIGREG = float(_env("TSAR_JEPA_SIGREG", 0.1))         # SIGReg weight (used when JEPA_REG=sigreg)
JEPA_SIG_M = int(_env("TSAR_JEPA_SIG_M", 256))             # SIGReg random 1-D directions per step
# Audit W3 ablations (docs/audit_v2.md 5.3). "copy" = A1: identity transition z_{t+k} = z_t + step_emb[k];
# "zero" = A3: imagined states zeroed before decode (seasonal-copy path only; meant for evaluating a trained ckpt).
STATE_OFF = _env("TSAR_STATE_OFF", "")
assert STATE_OFF in ("", "copy", "zero"), "TSAR_STATE_OFF must be copy or zero"
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
SYN5_SHARE = float(_env("TSAR_SYN5", 0))                   # share of synthetic rows from tools/synth5.generate
if SYN5_SHARE > 0:                                         # (world prior; taken first, synth3 etc. share the rest)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import synth5 as S5
# --- train_data.py (Dr. Yuki Tanaka): training-data mix flags, all default off ---
MIX_ALPHA = float(_env("TSAR_MIX_ALPHA", 0))   # temperature over real windows: p(dataset) ~ n_windows^alpha,
                                               # p(series | dataset) ~ n_windows^alpha (0 = current uniform)
STRETCH = float(_env("TSAR_STRETCH", 0))       # prob. a real H / 30min row is a causal k-times stretched copy
                                               # (period 96 / 144 / 288): real-shaped high-frequency rows
SYN4_SHARE = float(_env("TSAR_SYN4", 0))       # share of synthetic rows from tools/synth4.generate
SYN4_GAP = int(_env("TSAR_SYN4_GAP", 1))       # 1 = synth4 frequency weights complement the real rows
SDD_SHARE = float(_env("TSAR_SDD", 0))         # share of synthetic rows with an exact conditional law,
                                               # trained with the distilled pinball (lit_review #5)
SDD_K = int(_env("TSAR_SDD_K", 16))            # stratified quantile points of the exact law per target
RFL = float(_env("TSAR_RFL", 0))               # keep the central RFL band of real-row losses per dataset
                                               # (lit_review #6), e.g. 0.8 drops the easiest/hardest 10%
RFL_CKPT = _env("TSAR_RFL_CKPT")               # reference checkpoint (Alma's own, TSAR_BENCH=1-trained only);
                                               # unset = online variant (current model's per-row loss)
RFL_WARM = float(_env("TSAR_RFL_WARM", 0.1))   # no selection before this fraction of training
if SYN4_SHARE > 0 or SDD_SHARE > 0 or STRETCH > 0:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import synth4 as S4
SEAS_REF = int(_env("TSAR_SEAS_REF", 1))                   # T7 seasonal memory pathway in the decoder
NORM = _env("TSAR_NORM", "last")                          # "mean" | "last" (see causal_stats)
FRAME = int(_env("TSAR_FRAME", 1)); FRAMEFIX = int(_env("TSAR_FRAMEFIX", 0)); SREF_AVG = int(_env("TSAR_SREF_AVG", 1)); CTX_READ = int(_env("TSAR_CTX_READ", 0))
# Transition-dynamics flags (from train_dyn.py; all default 0).
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
if ZS and BENCH:                                           # audit_v2 L-4: would silently drop TRAIN sets
    raise SystemExit("TSAR_ZS is an old-suite protocol and meaningless under TSAR_BENCH=1; unset one of them")
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
# Long-horizon (h > MAX_H) chunked-rollout calibration, inference only (from train_calib.py; default off =
# median feedback). Each uses only the forecast's own context and the model's own forecasts.
CHUNK_MIX = int(_env("TSAR_CHUNK_MIX", 0))       # 1 = quantile-scenario mixture rollout instead of median feedback
CHUNK_SCALE = int(_env("TSAR_CHUNK_SCALE", 0))   # 1 = anchor later chunks' spread to the real context's scale
CHUNK_SCALE_MAX = float(_env("TSAR_CHUNK_SCALE_MAX", 5.0))
CHUNK_MIX_M = int(_env("TSAR_CHUNK_MIX_M", 20))  # equal-mass boundary levels the 9 scenario forecasts are interpolated to
# Inference-time options from train_tta.py (default off).
TTA_LONG = _env("TSAR_TTA_LONG", "median")       # h > MAX_H: median feedback | paths (S quantile-level sample paths)
TTA_PATHS = int(_env("TSAR_TTA_PATHS", 9))       # number of paths S
TTA_PATH_MODE = _env("TSAR_TTA_PATH_MODE", "keep")   # keep | median | shuffle: what each path feeds back
TTA_CLAMP = _env("TSAR_TTA_CLAMP", "")            # nonneg | range: clamp each chunk from the series' own context
TTA_CLAMP_K = float(_env("TSAR_TTA_CLAMP_K", 1.0))
assert TTA_LONG in ("median", "paths") and TTA_PATH_MODE in ("keep", "median", "shuffle") and TTA_CLAMP in ("", "nonneg", "range")
assert not (TTA_LONG == "paths" and (CHUNK_MIX or CHUNK_SCALE)), \
    "TSAR_TTA_LONG=paths and TSAR_CHUNK_MIX/SCALE are alternative long-horizon rollouts; pick one"
assert not (STATE_OFF == "copy" and (JUMP or DYN_IN)), \
    "TSAR_STATE_OFF=copy is the no-dynamics ablation; it cannot be combined with TSAR_JUMP/TSAR_DYN_IN"
# TSAR_TRANS (from train_arch.py, Mehta): latent transition architecture. "gru" (default) = nn.GRUCell, identical to
#   train.py. "koopman" = stable linear z <- A z + R(z, u), A block-diagonal 2x2 rotations with radius < 1
#   (complex eigenvalues inside the unit circle) and a zero-init input-dependent residual R. "ssm" = diagonal
#   selective SSM step (Mamba/S5): input-dependent dt, B, C; per-channel state of size TSAR_SSM_N carried
#   across rollout steps. "rssm" = DreamerV3-style stochastic latent on top of the GRU: Gaussian s_k with prior
#   p(s_k | h_k) and posterior q(s_k | h_k, z_{j+k}) whose target is the encoder state after the observed future
#   patch (stop-gradient); KL with free bits, posterior-decoded pinball as reconstruction; prior mean at
#   inference. "attn" = at each imagined step the state cross-attends (causally, lag-biased) to the encoder's
#   context states of its origin, then a small FFN. All new output projections are zero-init (state copy at init).
TRANS = _env("TSAR_TRANS", "gru")
assert TRANS in ("gru", "koopman", "ssm", "rssm", "attn"), TRANS
SSM_N = int(_env("TSAR_SSM_N", 8))                         # ssm: state size per channel
RSSM_S = int(_env("TSAR_RSSM_S", 32))                      # rssm: stochastic latent dims (Gaussian)
RSSM_KL = float(_env("TSAR_RSSM_KL", 1.0))                 # rssm: KL weight (split 5:1 dyn:rep as in DreamerV3)
RSSM_FREE = float(_env("TSAR_RSSM_FREE", 1.0))             # rssm: free nats per step
RSSM_POST = float(_env("TSAR_RSSM_POST", 0.5))             # rssm: weight of the posterior-decoded pinball
if TRANS != "gru":                         # train_v2 combination guards (each would run silently wrong otherwise)
    assert not LOOPS, "TSAR_TRANS replaces the transition; it cannot be combined with TSAR_LOOPS"
    assert STATE_OFF != "copy", "TSAR_STATE_OFF=copy bypasses the transition; do not combine with TSAR_TRANS"
    assert not TTA_STEPS, "TSAR_TTA_STEPS adapts model.dynamics, which TSAR_TRANS replaces"
    assert not (TRANS == "attn" and SELFCAL), "TSAR_SELFCAL's in-context rollouts carry no attn context"
# Multi-scale world-model flags (from train_ms.py, Okafor; default 0).
# TSAR_LONG_H: train and infer to this horizon (e.g. 256) instead of chunking MAX_H with median feedback. The data
#   window grows by LONG_H - MAX_H points of future-only targets; the encoder still sees PREFIX + CONTEXT + MAX_H
#   points, so the encoder and the near loss (every origin, leads <= MAX_H) are unchanged. Leads MAX_H+1..LONG_H
#   are scored on TSAR_FAR_ORIG random origins per row (patch index >= TSAR_FAR_J0), weight TSAR_FAR_W, so memory
#   stays roughly flat. With TSAR_MS_COARSE=0 the fine GRU simply rolls LONG_H/PATCH steps (horizon extension).
# TSAR_MS_COARSE: coarse world model stepping TSAR_MS_STRIDE patches per step from the same z_t; each coarse state
#   decodes its STRIDE patches through the shared decoder (+ a sub-patch embedding). Leads <= MAX_H fuse fine and
#   coarse with a learned per-lead gate (init 0.12 coarse); later leads come from the coarse rollout.
#   1 = own coarse GRUCell (+56k params), 2 = the fine cell is reused with coarse step embeddings (param-matched).
# TSAR_HDEC: horizon-conditioned decoding. The quantile spread around the median is scaled by
#   exp(a_k + r(z_0) * log1p(k)) for lead patch k: learned per-lead widening plus a per-series rate. Zero-init.
LONG_H = int(_env("TSAR_LONG_H", 0))
MS_COARSE = int(_env("TSAR_MS_COARSE", 0))
MS_STRIDE = int(_env("TSAR_MS_STRIDE", 4))
FAR_ORIG = int(_env("TSAR_FAR_ORIG", 4))
FAR_J0 = int(_env("TSAR_FAR_J0", 16))
FAR_W = float(_env("TSAR_FAR_W", 1.0))
HDEC = int(_env("TSAR_HDEC", 0))
if LONG_H:
    assert LONG_H > MAX_H and LONG_H % PATCH == 0, "TSAR_LONG_H must be a multiple of PATCH above MAX_H"
    assert MS_COARSE or not JUMP, "fine horizon extension does not support TSAR_JUMP"
    # train_loss_long replaces the whole training loss and the one-shot forecast replaces chunked inference:
    _bad = {"TSAR_LATENT": LATENT, "TSAR_JEPA": JEPA, "TSAR_DUALDEC": DUALDEC, "TSAR_SDD": SDD_SHARE,
            "TSAR_RFL": RFL, "TSAR_STATE_OFF": STATE_OFF, "TSAR_TRANS": TRANS != "gru", "TSAR_CHUNK_MIX": CHUNK_MIX,
            "TSAR_CHUNK_SCALE": CHUNK_SCALE, "TSAR_TTA_LONG=paths": TTA_LONG == "paths",
            "TSAR_TTA_STEPS": TTA_STEPS, "TSAR_SELFCAL": SELFCAL}
    _bad = [k for k, v in _bad.items() if v]
    assert not _bad, f"TSAR_LONG_H cannot be combined with {_bad}"
if MS_COARSE:
    assert MS_COARSE in (1, 2) and LONG_H and LONG_H % (MS_STRIDE * PATCH) == 0 and MAX_H % (MS_STRIDE * PATCH) == 0, \
        "TSAR_MS_COARSE (1|2) needs TSAR_LONG_H, and LONG_H and MAX_H must be multiples of MS_STRIDE * PATCH"
assert not (HDEC and DUALDEC), "TSAR_DUALDEC's observed-state decode is not HDEC-widened; do not combine"

K_STEPS = MAX_H // PATCH
WINDOW = CONTEXT + MAX_H
N_PATCH = WINDOW // PATCH
TWIN = PREFIX + WINDOW                    # training window incl. the coarse prefix
TCTX = PREFIX + CONTEXT                   # inference context incl. the coarse prefix
TDATA = TWIN + (LONG_H - MAX_H if LONG_H else 0)   # sampled window: TWIN plus far-horizon targets (TSAR_LONG_H)
H_OUT = LONG_H or MAX_H                   # one-shot forecast length
K_LONG = H_OUT // PATCH
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


def _git_head():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                        "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


_RUN_CFG = None


def run_cfg():
    """Data + model config stored in every checkpoint and verified on resume and in eval-only mode
    (audit_v2 L-1): bench flag, extra, the exact training corpus (content hash after TRAIN_DROP and
    TSAR_ZS), TRAIN_DROP, sizes, and the full TSAR_* env. Computed once, only when needed."""
    global _RUN_CFG
    if _RUN_CFG is None:
        h = hashlib.sha256()
        for d in datasets:
            for y in by_dataset[d]:
                a = np.ascontiguousarray(y, dtype=np.float32)
                h.update(f"{d}:{a.size};".encode())
                h.update(a.tobytes())
        drop = hashlib.sha256(json.dumps({k: sorted(v) for k, v in BN.TRAIN_DROP.items()},
                                         sort_keys=True).encode()).hexdigest() if BENCH else None
        _RUN_CFG = {"bench": BENCH, "bench_extra": int(_env("TSAR_BENCH_EXTRA", 0)),
                    "bench_maxlen": int(_env("TSAR_BENCH_MAXLEN", 20000)) if BENCH else None,
                    "zs": ZS, "context": CONTEXT, "prefix": PREFIX, "prefix_patch": PREFIX_PATCH,
                    "max_h": MAX_H, "patch": PATCH, "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                    "train_datasets": sorted(datasets), "train_series": sum(len(v) for v in by_dataset.values()),
                    "train_sha256": h.hexdigest(), "train_drop_sha256": drop,
                    "env": _tsar_env(), "git": _git_head()}
    return _RUN_CFG


def cfg_mismatch(stored, ignore):
    """Differences between a checkpoint's stored run config and this process ([] = identical).
    The git hash is provenance only: a different commit is printed, not refused."""
    cur, bad = run_cfg(), []
    for k in sorted(set(stored) | set(cur)):
        if k not in ("env", "git") and stored.get(k) != cur.get(k):
            bad.append(f"{k}: checkpoint={str(stored.get(k))[:100]}  now={str(cur.get(k))[:100]}")
    se = {k: v for k, v in stored.get("env", {}).items() if k not in ignore}
    ce = {k: v for k, v in cur["env"].items() if k not in ignore}
    for k in sorted(set(se) | set(ce)):
        if se.get(k) != ce.get(k):
            bad.append(f"{k}: checkpoint={se.get(k)!r}  now={ce.get(k)!r}")
    if stored.get("git") != cur["git"]:
        print(f"note: checkpoint written at git {stored.get('git')}, running at {cur['git']}", flush=True)
    return bad


SYN_FRAC = float(SYN_FRAC_ENV) if SYN_FRAC_ENV else 0.3   # share of each batch that is synthetic

# --- real-row sampling weights, stretch eligibility and synth4 gap weights (train_data.py) ---
if MIX_ALPHA or STRETCH or SYN4_SHARE or SDD_SHARE or RFL:
    assert BENCH, "the train_data.py mix flags are defined for TSAR_BENCH=1 (bench TRAIN split)"
FREQ = {s["dataset"]: s["freq"] for s in train} if BENCH else {}
_nwin = {d: np.array([max(1.0, len(y) / TWIN) for y in by_dataset[d]]) for d in datasets}
if MIX_ALPHA:
    _dw = np.array([_nwin[d].sum() ** MIX_ALPHA for d in datasets]); DS_P = _dw / _dw.sum()
    SER_P = {d: _nwin[d] ** MIX_ALPHA / (_nwin[d] ** MIX_ALPHA).sum() for d in datasets}
else:
    DS_P = np.full(len(datasets), 1.0 / max(len(datasets), 1)); SER_P = None
STRETCH_K = {"H": (4, 6, 12), "30min": (2, 3, 6)}         # -> periods 96 / 144 / 288 (15 / 10 / 5 min)
_KF = {96: "15min", 144: "10min", 288: "5min"}
REAL_FREQ_SHARE = {}
for d, w in zip(datasets, DS_P):
    f = FREQ.get(d, "?")
    ks = STRETCH_K.get(f) if STRETCH else None
    REAL_FREQ_SHARE[f] = REAL_FREQ_SHARE.get(f, 0) + w * (1 - STRETCH if ks else 1)
    for k in ks or ():
        g = _KF[SEASON[d] * k]
        REAL_FREQ_SHARE[g] = REAL_FREQ_SHARE.get(g, 0) + w * STRETCH / len(ks)
SYN4_FW = S4.gap_weights(REAL_FREQ_SHARE, SYN_FRAC) if (SYN4_SHARE or SDD_SHARE) and SYN4_GAP else None
if MIX_ALPHA or STRETCH or SYN4_SHARE or SDD_SHARE:
    print("real dataset weights:", {d: round(float(w), 3) for d, w in zip(datasets, DS_P)})
    print("real rows by frequency:", {f: round(float(v), 3) for f, v in sorted(REAL_FREQ_SHARE.items())})
    if SYN4_FW is not None:
        print("synth4 frequency weights (gap-filling):", dict(zip(S4.FNAMES, np.round(SYN4_FW, 3))))
_SDD = None      # (rows [B] bool, g [B, TWIN], phi [B], sig [B]) of the last batch, for train_loss
_SRC = None      # dataset index per row of the last batch (-1 = synthetic), for TSAR_RFL
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
    """Right-aligned windows of TDATA (= TWIN = PREFIX + WINDOW unless TSAR_LONG_H) points, balanced across datasets. NaN = missing/pad.
    Origins are fine patches only, so the prefix is strictly earlier history.
    With every TSAR_MIX_ALPHA/STRETCH/SYN4/SDD flag at 0 the batch and the RNG stream are
    identical to train.py."""
    global _SDD, _SRC
    win = np.full((BATCH, TDATA), np.nan, dtype=np.float32)
    per = np.ones(BATCH, dtype=np.int64)
    src = np.full(BATCH, -1, dtype=np.int64)
    frac = SYN_FRAC if not SYN_CURR else SYN_FRAC + (SYN_CURR - SYN_FRAC) * max(0.0, 1 - PROG / 0.6)
    n_syn = int(BATCH * frac)                                            # TSAR_SYN_CURR: synthetic-first curriculum
    n_ks = int(BATCH * KS_FRAC) if KS is not None else 0
    sw = slice(0, TDATA) if PREFIX_SYN else slice(PREFIX, TDATA)        # synthetic rows: prefix only if PREFIX_SYN
    n4 = int(round(n_syn * SYN4_SHARE))                                  # TSAR_SYN4: synth4 rows
    nd = int(round(n_syn * SDD_SHARE))                                   # TSAR_SDD: exact-law rows
    o = n4 + nd
    if n4:
        win[:n4, sw], per[:n4] = S4.generate(rng, n4, sw.stop - sw.start, freq_weights=SYN4_FW, min_len=2 * PATCH)
    _SDD = None
    if nd:
        g = np.zeros((BATCH, TDATA), np.float32); ph = np.zeros(BATCH, np.float32); sg = np.ones(BATCH, np.float32)
        win[n4:o, sw], per[n4:o], g[n4:o, sw], ph[n4:o], sg[n4:o] = S4.generate_sdd(
            rng, nd, sw.stop - sw.start, freq_weights=SYN4_FW, min_len=2 * PATCH)
        rows = np.zeros(BATCH, bool); rows[n4:o] = True
        _SDD = tuple(torch.from_numpy(a).to(DEVICE) for a in (rows, g, ph, sg))
    o5 = o                                                               # TSAR_SYN5 (train_syn5.py): synth5 rows next,
    o = o5 + int(round((n_syn - o5) * SYN5_SHARE))                       # before synth3/2/1 (0 = off, no rng draws)
    if o > o5:
        win[o5:o, sw], per[o5:o] = S5.generate(rng, o - o5, sw.stop - sw.start, min_len=2 * PATCH)
    n3 = int(round((n_syn - o) * SYN3_SHARE))                            # TSAR_SYN3: synth3 share of the rest
    n2 = int((n_syn - o - n3) * SYN2_SHARE)
    if n3:
        win[o:o + n3, sw], per[o:o + n3] = S3.generate(rng, n3, sw.stop - sw.start, min_len=2 * PATCH)
    win[o + n3:o + n3 + n2, sw], per[o + n3:o + n3 + n2] = synthetic2(rng, n2, sw.stop - sw.start)
    win[o + n3 + n2:n_syn, sw], per[o + n3 + n2:n_syn] = synthetic(rng, n_syn - o - n3 - n2, sw.stop - sw.start)
    win[n_syn:n_syn + n_ks, TDATA - WINDOW:] = kernelsynth(rng, n_ks)
    for b in range(n_syn + n_ks, BATCH):
        di = rng.choice(len(datasets), p=DS_P) if MIX_ALPHA else rng.integers(len(datasets))
        d = datasets[di]
        src[b] = di
        series = by_dataset[d]
        per[b] = SEASON[d]
        y = series[rng.choice(len(series), p=SER_P[d]) if MIX_ALPHA else rng.integers(len(series))]
        ks = STRETCH_K.get(FREQ.get(d)) if STRETCH else None
        if ks and rng.random() < STRETCH:                                # TSAR_STRETCH: causal k-x upsampling
            k = ks[rng.integers(len(ks))]
            n0 = -(-TDATA // k) + 3
            end = rng.integers(min(n0, len(y)), len(y) + 1)
            w = S4.stretch(y[max(0, end - n0):end], k, rng)[-TDATA:]
            per[b] = SEASON[d] * k
            src[b] = -1                                                  # not a real window of its dataset
        else:
            end = rng.integers(min(2 * PATCH, len(y)), len(y) + 1)
            w = y[max(0, end - TDATA):end]
        win[b, TDATA - len(w):] = w
    if AUG_P > 0:                                                        # TSAR_AUG: augment real rows only
        rows = np.arange(n_syn + n_ks, BATCH)
        rows = rows[rng.random(len(rows)) < AUG_P]
        if len(rows):
            win[rows], per[rows] = S3.augment(win[rows], rng, period=per[rows], min_obs=4 * PATCH)
    _SRC = src
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


def seasonal_ref(x, loc, scale, m, H=MAX_H, j=None, n=None):
    """Seasonal-naive reference for every origin and future step, in each origin's model space.
    x: [B, L] raw, m: [B] season (1 = naive). Returns [B, n, MAX_H]. Causal: only values <= origin.
    loc/scale cover the last n patches of x; anything before them (the coarse prefix's raw values) is
    used as older seasons, so seasons longer than the fine context (m=168, 288) still have a reference."""
    B, L = x.shape
    # j: optional [B, M] fine-patch origins (with n = number of fine patches in x): reference only for those
    # origins, loc/scale given as [B, M]. Long horizons need this; the full [B, n, H] table is large on MPS.
    off = L - (loc.shape[1] if j is None else n) * PATCH                       # prefix length (0 = none)
    obs = ~torch.isnan(x)
    ar = torch.arange(L, device=x.device)
    last = torch.where(obs, ar.expand_as(x), torch.full_like(x, -1, dtype=torch.long)).cummax(1).values
    ffill = torch.gather(torch.nan_to_num(x), 1, last.clamp(min=0))
    ends = torch.arange(off + PATCH - 1, L, PATCH, device=x.device)            # [n]
    if j is not None:
        ends = off + (j + 1) * PATCH - 1                                       # [B, M]
    k = torch.arange(1, H + 1, device=x.device)                                # [H]
    mm = m.clamp(min=1)[:, None, None]
    lag = mm * torch.div(k[None, None, :] + mm - 1, mm, rounding_mode="floor")  # m * ceil(k/m)
    idx = (ends[None, :, None] if j is None else ends[..., None]) + k[None, None, :] - lag   # [B, n|M, H]
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


class KoopmanTransition(nn.Module):
    """TSAR_TRANS=koopman: z <- A z + R([LN(z), u]). A is block-diagonal with 2x2 blocks r_i [[cos, -sin], [sin, cos]],
    i.e. complex-conjugate eigenvalues r_i exp(+-i theta_i) with r_i = exp(-exp(nu_i)) in (0, 1): the linear part is a
    contraction for any parameters, so long rollouts stay bounded. theta_i is a period of 2 pi / theta_i patches
    (initialised log-spaced 2..512 patches; inspect with .eig()). R is a zero-init MLP: at init the rollout is
    purely linear."""

    def __init__(self, d, hidden):
        super().__init__()
        m = d // 2
        assert d % 2 == 0
        self.theta = nn.Parameter(2 * math.pi / torch.exp(torch.linspace(math.log(2), math.log(512), m)))
        self.nu = nn.Parameter(torch.linspace(math.log(0.01), math.log(0.3), m))   # radius 0.99 .. 0.74
        self.norm = nn.LayerNorm(d)
        self.res = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)

    def eig(self):
        return torch.exp(-torch.exp(self.nu)), self.theta                  # radius, angle (rad per patch)

    def begin(self, z, ctx):
        pass

    def forward(self, u, z, k, cell=None):
        r, th = self.eig()
        c, s = r * torch.cos(th), r * torch.sin(th)
        a, b = z.view(*z.shape[:-1], -1, 2).unbind(-1)
        lin = torch.stack([c * a - s * b, s * a + c * b], -1).flatten(-2)
        return lin + self.res(torch.cat([self.norm(z), u], -1))


class SSMTransition(nn.Module):
    """TSAR_TRANS=ssm: one step of a diagonal selective SSM (Mamba / S5 style) per imagined patch. Input
    x = W[LN(z), u]; dt = softplus(W_dt x), B, C = W_BC x (input-dependent = selective); per-channel state
    h [N, D, n] <- exp(dt A) h + dt x B, A = -exp(A_log) < 0 (stable); y = C h + D x; z <- z + W_o(y * silu(W_g x)),
    W_o zero-init. h starts as the origin state broadcast (h0 = z0 * g) and is carried across rollout steps."""

    def __init__(self, d, n):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(2 * d, d)
        self.dt = nn.Linear(d, d)
        with torch.no_grad():                                                # dt in [0.01, 0.5] at init
            dt0 = torch.exp(torch.empty(d).uniform_(math.log(0.01), math.log(0.5)))
            self.dt.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))          # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float32)).repeat(d, 1))
        self.BC = nn.Linear(d, 2 * n)
        self.Dskip = nn.Parameter(torch.ones(d))
        self.h0 = nn.Parameter(torch.ones(n))
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def begin(self, z, ctx):
        self.h = z[..., None] * self.h0

    def forward(self, u, z, k, cell=None):
        x = self.inp(torch.cat([self.norm(z), u], -1))
        dt = F.softplus(self.dt(x))
        B, C = self.BC(x).chunk(2, -1)
        self.h = torch.exp(dt[..., None] * -torch.exp(self.A_log)) * self.h + (dt * x)[..., None] * B[:, None]
        y = (self.h * C[:, None]).sum(-1) + self.Dskip * x
        return z + self.out(y * F.silu(self.gate(x)))


class RSSMTransition(nn.Module):
    """TSAR_TRANS=rssm: DreamerV3-style stochastic latent on the model's GRUCell. Deterministic path
    h_k = GRU(u_k + W_in s_{k-1}, z_{k-1}); prior p(s_k | h_k) Gaussian; state z_k = h_k + W_out s_k (W_out zero-init).
    The open-loop path (decoded, carried, used at inference) takes the prior MEAN, so train and inference
    rollouts match and the quantile head carries the uncertainty. Training only: posterior q(s_k | h_k, e_{j+k})
    with e = encoder state after the observed future patch (stop-gradient, the "posterior target"); KL =
    0.5 max(free, KL(sg q || p)) + 0.1 max(free, KL(q || sg p)) (DreamerV3 balance + free bits), and the posterior
    state h_k + W_out s_q is decoded for a pinball "reconstruction" term (self.post_states)."""

    def __init__(self, d, s):
        super().__init__()
        self.in_s = nn.Linear(s, d)
        self.prior = nn.Linear(d, 2 * s)
        self.postnet = nn.Linear(2 * d, 2 * s)
        self.out_s = nn.Linear(s, d)
        nn.init.zeros_(self.out_s.weight); nn.init.zeros_(self.out_s.bias)
        self.S = s

    @staticmethod
    def _kl(mq, sq, mp, sp):
        return (torch.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5).sum(-1)

    def begin(self, z, ctx):
        self.s = z.new_zeros(z.shape[0], self.S)
        self.post = ctx.get("post") if (ctx and self.training) else None
        self.ok = ctx.get("ok") if self.post is not None else None
        self.kl, self.post_states = 0.0, []

    def forward(self, u, z, k, cell=None):
        h = cell(u + self.in_s(self.s), z)
        mp, sp = self.prior(h).chunk(2, -1)
        sp = F.softplus(sp) + 0.1
        if self.post is not None:
            mq, sq = self.postnet(torch.cat([h, self.post[:, k]], -1)).chunk(2, -1)
            sq = F.softplus(sq) + 0.1
            sample = mq + sq * torch.randn_like(sq)
            kl = 0.5 * self._kl(mq.detach(), sq.detach(), mp, sp).clamp(min=RSSM_FREE) \
                + 0.1 * self._kl(mq, sq, mp.detach(), sp.detach()).clamp(min=RSSM_FREE)
            ok = self.ok[:, k].float()
            self.kl = self.kl + (kl * ok).sum() / ok.sum().clamp(min=1) / K_STEPS
            self.post_states.append(h + self.out_s(torch.where(self.ok[:, k, None], sample, mp)))
        self.s = mp
        return h + self.out_s(mp)


class AttnTransition(nn.Module):
    """TSAR_TRANS=attn: each imagined step cross-attends from the rolled state (plus the step input u) to the
    encoder's context states of its own origin (keys = states at patches <= origin, learned per-head lag bias),
    then a small FFN; both residuals zero-init. Origins are scattered into a [B, n] grid so one batched
    attention with a causal mask serves every origin (no per-origin copies of the context)."""

    def __init__(self, d, heads, nlag):
        super().__init__()
        self.H, self.nlag = heads, nlag
        self.nq, self.nk, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, nlag))
        self.f1, self.f2 = nn.Linear(d, d), nn.Linear(d, d)
        for m in (self.o, self.f2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def begin(self, z, ctx):
        if ctx is None:                                                     # no context given: attend to z0 only
            ctx = {"z": z[:, None], "b": torch.arange(len(z), device=z.device),
                   "j": torch.zeros(len(z), dtype=torch.long, device=z.device)}
        zc = ctx["z"]
        B, n, D = zc.shape
        self.B, self.n, self.rows = B, n, ctx["b"] * n + ctx["j"]
        k, v = self.kv(self.nk(zc)).view(B, n, 2, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        self.K, self.V = k, v                                               # [B, H, n, dh]
        lag = torch.arange(n, device=z.device)[:, None] - torch.arange(n, device=z.device)[None]   # query j, key i
        bias = self.rel[:, lag.clamp(0, self.nlag - 1)]
        self.bias = bias.masked_fill(lag[None] < 0, float("-inf"))           # [H, n, n]

    def forward(self, u, z, k, cell=None):
        N, D = z.shape
        q = self.q(self.nq(z) + u)
        grid = z.new_zeros(self.B * self.n, D).index_copy(0, self.rows, q)
        grid = grid.view(self.B, self.n, self.H, D // self.H).transpose(1, 2)
        a = F.scaled_dot_product_attention(grid, self.K, self.V, attn_mask=self.bias)
        a = a.transpose(1, 2).reshape(self.B * self.n, D)[self.rows]
        z = z + self.o(a)
        return z + self.f2(F.gelu(self.f1(self.nf(z))))


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
        if JEPA:                           # created last so default-off runs keep identical init RNG draws
            self.jepa_proj = nn.Sequential(nn.Linear(D_MODEL, JEPA_PROJ), nn.LayerNorm(JEPA_PROJ), nn.GELU(),
                                           nn.Linear(JEPA_PROJ, JEPA_PROJ))           # state -> embedding (both branches)
            self.jepa_pred = nn.Sequential(nn.Linear(JEPA_PROJ, JEPA_PROJ), nn.LayerNorm(JEPA_PROJ), nn.GELU(),
                                           nn.Linear(JEPA_PROJ, JEPA_PROJ))           # online-only predictor
            self.jepa_gnd = nn.Linear(JEPA_PROJ, PATCH * NQ)                          # grounding head (EIDOS g_psi)
        if TRANS != "gru":                 # TSAR_TRANS: alternative transition, created last (GRUCell above keeps
            assert not LOOPS               # the init RNG stream identical; it is dropped unless rssm uses it)
            if TRANS == "koopman":
                self.trans = KoopmanTransition(D_MODEL, 2 * D_MODEL)
            elif TRANS == "ssm":
                self.trans = SSMTransition(D_MODEL, SSM_N)
            elif TRANS == "rssm":
                self.trans = RSSMTransition(D_MODEL, RSSM_S)
            else:
                self.trans = AttnTransition(D_MODEL, N_HEADS, N_PATCH)
            if TRANS != "rssm":
                self.dynamics = None
        if LONG_H and not MS_COARSE:       # horizon extension: step embeddings for the extra fine steps
            self.far_step_emb = nn.Parameter(torch.zeros(K_LONG - K_STEPS, D_MODEL))
        if MS_COARSE:                      # coarse world model: one step = MS_STRIDE patches
            self.ms_cell = nn.GRUCell(D_MODEL, D_MODEL) if MS_COARSE == 1 else None
            self.ms_emb = nn.Parameter(torch.zeros(K_LONG // MS_STRIDE, D_MODEL))
            if MS_COARSE == 2:             # shared cell: a learned offset marks the coarse time scale
                self.ms_scale = nn.Parameter(0.02 * torch.randn(D_MODEL))
            self.ms_sub = nn.Parameter(0.02 * torch.randn(MS_STRIDE, D_MODEL))   # position inside the stride
            self.ms_fuse = nn.Parameter(torch.full((K_STEPS,), -2.0))             # per-lead coarse weight (logit)
        if HDEC:                           # horizon-conditioned spread
            self.hd_lead = nn.Parameter(torch.zeros(K_LONG))
            self.hd_rate = nn.Linear(D_MODEL, 1)
            nn.init.zeros_(self.hd_rate.weight); nn.init.zeros_(self.hd_rate.bias)

    def hdec(self, q, z0, k0):
        """TSAR_HDEC: widen quantiles around the median with lead time. q: [N, K*PATCH, NQ] for lead patches
        k0..k0+K-1, z0: [N, D] origin state."""
        K = q.shape[1] // PATCH
        k = torch.arange(k0, k0 + K, device=q.device)
        s = self.hd_lead[k][None] + self.hd_rate(z0) * torch.log1p(k.float())[None]       # [N, K]
        s = s.clamp(-3, 3).repeat_interleave(PATCH, 1)[..., None]
        med = q[..., NQ // 2:NQ // 2 + 1]
        return med + torch.exp(s) * (q - med)

    def rollout_coarse(self, z0, ref, KC):
        """Coarse world model: KC steps of MS_STRIDE patches from z0. ref: [N, >= KC*STRIDE*PATCH] (model space).
        Returns [N, KC*STRIDE*PATCH, NQ] before HDEC."""
        N, S = z0.shape[0], MS_STRIDE
        cell = self.ms_cell if MS_COARSE == 1 else self.dynamics
        c = self.cond(z0) if LOOPS and MS_COARSE == 2 else 0
        z, zs = z0, []
        for j in range(KC):
            u = self.ms_emb[j].expand(N, -1) + (self.ms_scale if MS_COARSE == 2 else 0) + c
            z = cell(u, z)
            zs.append(z)
        st = torch.stack(zs, 1)[:, :, None] + self.ms_sub[None, None]           # [N, KC, S, D]
        st = st.reshape(N, KC * S, D_MODEL)
        if not SEAS_REF:
            ref = torch.zeros_like(ref)
        r = ref[:, :KC * S * PATCH].clamp(-20, 20).view(N, KC * S, PATCH)
        w = self.ref_skip + (self.skip_gate(st)[:, :, None] if GATE else 0)  # [NQ] or [N, K, 1, NQ]
        q = self.decode(st + self.ref_proj(r)).view(N, KC * S, PATCH, NQ) + r[..., None] * w
        return q.reshape(N, KC * S * PATCH, NQ)

    def fuse(self, qf, qc):
        """Leads <= MAX_H: fine + gate * (coarse - fine); later leads: coarse only."""
        g = torch.sigmoid(self.ms_fuse).repeat_interleave(PATCH)[None, :, None]
        return torch.cat([qf + g * (qc[:, :MAX_H] - qf), qc[:, MAX_H:]], 1)

    def rollout_near(self, z, ref, mem=None):
        """Leads 1..MAX_H used for the every-origin loss (fused with the coarse model when on)."""
        if not MS_COARSE:
            return self.rollout(z, ref, mem=mem)
        qf = self.rollout(z, ref, mem=mem, hd=False)
        q = self.fuse(qf, self.rollout_coarse(z, ref, MAX_H // (MS_STRIDE * PATCH)))
        return self.hdec(q, z, 0) if HDEC else q

    def rollout_long(self, z, ref, mem=None):
        """Leads 1..LONG_H in one shot. ref: [N, LONG_H]."""
        if not MS_COARSE:
            return self.rollout(z, ref, mem=mem, K=K_LONG)
        qf = self.rollout(z, ref[:, :MAX_H], mem=mem, hd=False)
        q = self.fuse(qf, self.rollout_coarse(z, ref, K_LONG // MS_STRIDE))
        return self.hdec(q, z, 0) if HDEC else q

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

    def rollout(self, z, ref=None, return_states=False, mem=None, ctx=None, K=K_STEPS, hd=True):
        """z: [N, D], ref: [N, MAX_H] seasonal reference -> quantiles in model space [N, K*PATCH, NQ].
        The transition cell imagines K future latent states without observations (prior);
        return_states also returns them [N, K, D] for the latent-consistency loss."""
        if ref is None or not SEAS_REF:
            ref = torch.zeros(z.shape[0], K * PATCH, device=z.device)
        ref = ref.clamp(-20, 20)
        zs = []
        c = self.cond(z) if LOOPS else 0
        z0 = z
        if DYN_IN & 1 and not LOOPS:
            c = self.dyn_cond(z)                                       # origin-conditioned transition input
        if JUMP:
            jb = self.jump_in(self.jump_norm(z))                       # [N, D], shared by all horizons
        dec = []                                                      # states that are decoded (TSAR_JUMP)
        if TRANS != "gru":
            self.trans.begin(z, ctx)                                   # ctx: origin context / posterior targets
        for k in range(K):
            if STATE_OFF == "copy":                                   # A1: no dynamics, decode from the last observed latent
                z = z0 + self.step_emb[k]
            else:
                u = (self.step_emb[k] if k < K_STEPS else self.far_step_emb[k - K_STEPS]).expand(z.shape[0], -1) + c
                if DYN_IN & 2:
                    u = u + self.dyn_ref(ref[:, k * PATCH:(k + 1) * PATCH])   # seasonal forcing ("action")
                z = self.dynamics(u, z) if TRANS == "gru" else self.trans(u, z, k, self.dynamics)
            if JUMP:
                j = self.jump_out(F.gelu(jb + self.jump_h[k]))           # direct jump to horizon k
                g = torch.sigmoid(self.jump_gate(torch.cat([z, j], -1)))
                zf = z + g * j                                         # zero-init: equals z at init
                if JUMP == 2:
                    z = zf                                             # anchored rollout: carry the fused state
                dec.append(zf)
            zs.append(z)
        S = torch.stack(dec if JUMP else zs, 1)                       # imagined states [N, K, D]
        if STATE_OFF == "zero":                                       # A3: only the seasonal-copy path remains
            S = torch.zeros_like(S)
        if CTX_READ and mem is not None:
            keys, ignore = mem
            S = S + self.read(self.read_q(S), keys, keys, key_padding_mask=ignore, need_weights=False)[0]
        out = self.decode_states(S, ref)
        if TRANS == "rssm" and self.trans.post_states:                 # decode posterior states (training only)
            self.trans.pred_post = self.decode_states(torch.stack(self.trans.post_states, 1), ref)
        if HDEC and hd:
            out = self.hdec(out, z0, 0)
        if return_states:
            return out, torch.stack(zs, 1)
        return out

    def decode_states(self, S, ref):
        """Quantile head on states S [N, K, D] with seasonal reference ref [N, >= K*PATCH] (clamped) -> [N, K*PATCH, NQ]."""
        N, K = S.shape[:2]
        if K != K_STEPS:                   # long rollout (TSAR_LONG_H): decode all steps in one batched call
            r = ref[:, :K * PATCH].view(N, K, PATCH)
            w = self.ref_skip + (self.skip_gate(S)[:, :, None] if GATE else 0)
            return (self.decode(S + self.ref_proj(r)).view(N, K, PATCH, NQ) + r[..., None] * w).reshape(N, K * PATCH, NQ)
        outs = []
        for k in range(K_STEPS):
            z = S[:, k]
            r = ref[:, k * PATCH:(k + 1) * PATCH]
            w = self.ref_skip + (self.skip_gate(z)[:, None] if GATE else 0)
            q = self.decode(z + self.ref_proj(r)).view(-1, PATCH, NQ) + r[..., None] * w
            outs.append(q)
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


JEPA_STATS = {}                            # last values of the JEPA terms, for logging


def jepa_str():
    v = {k_: float(x) for k_, x in JEPA_STATS.items()}
    return (f"jepa w {v['w']:.3f} pin {v['pinball']:.4f} pred {v['pred']:.4f} gnd {v['gnd']:.4f} reg {v['reg']:.4f}"
            f" w2 {v['w2_pred']:.4f}/copy {v['w2_copy']:.4f}")


def vicreg_terms(e):
    """Variance hinge (std >= 1 per dim) and off-diagonal covariance penalty on embeddings e: [N, P]."""
    e = e - e.mean(0)
    std = torch.sqrt(e.var(0) + 1e-4)
    var = F.relu(1 - std).mean()
    if not JEPA_COV:
        return var, e.new_zeros(())
    c = (e.T @ e) / max(e.shape[0] - 1, 1)
    off = c - torch.diag(torch.diag(c))
    return var, off.pow(2).sum() / e.shape[1]


def sigreg(e, M=JEPA_SIG_M, n_max=2048):
    """LeWM / LeJEPA SIGReg: Epps-Pulley distance between the empirical characteristic function of e
    projected on M random unit directions and that of N(0, 1), weighted by exp(-t^2/2). e: [N, P]."""
    if e.shape[0] > n_max:
        e = e[torch.randperm(e.shape[0], device=e.device)[:n_max]]
    A = F.normalize(torch.randn(e.shape[1], M, device=e.device), dim=0)
    t = torch.linspace(-5, 5, 17, device=e.device)
    x = (e @ A)[..., None] * t                                         # [N, M, T]
    phi = torch.exp(-0.5 * t * t)
    err = (x.cos().mean(0) - phi).pow(2) + x.sin().mean(0).pow(2)       # [M, T]
    return torch.trapezoid(err * phi, t, dim=-1).mean()


def jepa_loss(model, z, z_tgt, prior, keep, obs_val, obs_ok):
    """Grounded latent prediction objective (JEPA / BYOL / EIDOS). Three terms:
      pred   the state the dynamics imagines k patches ahead of origin j, after an online projector +
             predictor, must match (cosine or smooth-L1) the projected *target* state at origin j+k, from an
             EMA copy of the encoder (TSAR_LATENT_EMA) or a stop-grad of the online encoder. The target
             only reads data inside the training window and is never used at inference.
      gnd    EIDOS grounding: the online projected embedding of every encoder state must decode (pinball)
             to the patch it has just observed, so the space the targets live in keeps observation content
             and cannot drift to a constant. (The imagined states are grounded by the forecast pinball.)
      reg    anti-collapse on the online embeddings: VICReg variance (+ optional covariance) or SIGReg.
    z: [B, n, D] online states, z_tgt: [B, n, D] target states (no grad), prior: [N_keep, K, D],
    obs_val / obs_ok: [B, n, PATCH] observed patch values in each patch's own model-space frame / mask."""
    B, n, D = z.shape
    k = torch.arange(1, K_STEPS + 1, device=z.device)
    j = torch.arange(n, device=z.device)
    tgt_idx = j[:, None] + k[None, :]                                    # [n, K]
    ok = (tgt_idx < n)[None].expand(B, n, K_STEPS).reshape(B * n, K_STEPS)[keep].float()
    okw = ok * JEPA_GAMMA ** (k - 1).float()[None]
    tproj = (lat_tgt if LATENT_EMA else model).jepa_proj
    with torch.no_grad():
        t = tproj(z_tgt)[:, tgt_idx.clamp(max=n - 1)]                    # [B, n, K, P]
        t = t.reshape(B * n, K_STEPS, -1)[keep]
    p = model.jepa_pred(model.jepa_proj(prior))                          # [N_keep, K, P]
    if JEPA_FORM == "l1cos":               # WorldTS stage-1 form: L1 on the embeddings plus (1 - cos)
        dist = JEPA_L1 * (p - t).abs().mean(-1) + 1 - F.cosine_similarity(p, t, dim=-1)
    elif JEPA_DIST == "sl1":
        dist = F.smooth_l1_loss(F.layer_norm(p, p.shape[-1:]), F.layer_norm(t, t.shape[-1:]), reduction="none").mean(-1)
    else:
        dist = 2 - 2 * F.cosine_similarity(p, t, dim=-1)
    pred = (dist * okw).sum() / okw.sum().clamp(min=1)
    e = model.jepa_proj(z.reshape(B * n, D))                             # online embeddings, every origin
    loss = pred
    gnd = e.new_zeros(())
    if JEPA_GND:
        g = model.jepa_gnd(e).view(B * n, PATCH, NQ)
        gnd = pinball(g, obs_val.reshape(B * n, PATCH), obs_ok.reshape(B * n, PATCH).float())
        loss = loss + JEPA_GND * gnd
    ek = e[keep]
    if JEPA_REG == "vicreg":
        var, cov = vicreg_terms(ek)
        reg = JEPA_VAR * var + JEPA_COV * cov
    elif JEPA_REG == "sigreg":
        reg = JEPA_SIGREG * sigreg(ek)
    else:
        reg = e.new_zeros(())
    loss = loss + reg
    with torch.no_grad():                  # audit W2: k-step latent error of the dynamics vs the copy-state baseline
        zt = z.detach()[:, tgt_idx.clamp(max=n - 1)].reshape(B * n, K_STEPS, D)[keep]
        z0 = z.detach().reshape(B * n, D)[keep][:, None]
        w2p = ((prior.detach() - zt).pow(2).mean(-1) * ok).sum() / ok.sum().clamp(min=1)
        w2c = ((z0 - zt).pow(2).mean(-1) * ok).sum() / ok.sum().clamp(min=1)
    JEPA_STATS.update(pred=pred.detach(), gnd=gnd.detach(), reg=reg.detach(), w2_pred=w2p, w2_copy=w2c)
    return loss


def _dualdec_term(model, z, ref, tgt, valid, keep, pred):
    """TSAR_DUALDEC: decode the encoder state at origin t+k (z[:, t+k], which has observed target patch k of
    origin t) through the same quantile head, pinball against the same target, plus DUALDEC_CONS * the mean
    |pred - sg q_obs| over valid elements. z [B, n, D], ref [B, n, MAX_H], tgt/valid [B, n, MAX_H], pred [N_keep, MAX_H, NQ]."""
    B, n, D = z.shape
    k = torch.arange(1, K_STEPS + 1, device=z.device)
    idx = torch.arange(n, device=z.device)[:, None] + k[None, :]                       # [n, K]
    ok = (idx < n)[None].expand(B, n, K_STEPS).reshape(B * n, K_STEPS)[keep]          # [N_keep, K]
    zs = z.detach() if DUALDEC_SG else z
    zo = zs[:, idx.clamp(max=n - 1)].reshape(B * n, K_STEPS, D)[keep]                  # [N_keep, K, D]
    r = ref.reshape(B * n, MAX_H)[keep].clamp(-20, 20) if SEAS_REF else torch.zeros(len(zo), MAX_H, device=z.device)
    q_obs = model.decode_states(zo, r)                                              # same ref as rollout()
    m = valid.reshape(B * n, MAX_H)[keep].float() * ok.repeat_interleave(PATCH, 1).float()
    t = tgt.reshape(B * n, MAX_H)[keep]
    cons = ((pred - q_obs.detach()).abs().mean(-1) * m).sum() / m.sum().clamp(min=1)
    return DUALDEC * (pinball(q_obs, t, m) + DUALDEC_CONS * cons)


def _arch_ctx(z, keep):
    """TSAR_TRANS attn: each kept origin's context states; rssm: also the encoder states after the observed
    future patches j+1..j+K as posterior targets (stop-gradient). None for the other transitions."""
    if TRANS not in ("attn", "rssm"):
        return None
    B, n, D = z.shape
    rows = torch.arange(B * n, device=z.device)[keep]
    b, j = rows // n, rows % n
    ctx = {"z": z, "b": b, "j": j}
    if TRANS == "rssm":
        idx = j[:, None] + 1 + torch.arange(K_STEPS, device=z.device)[None]          # [N, K]
        ctx["post"], ctx["ok"] = z.detach()[b[:, None], idx.clamp(max=n - 1)], idx < n
    return ctx


def _rssm_term(model, t, v):
    return RSSM_POST * pinball(model.trans.pred_post, t, v) + RSSM_KL * model.trans.kl


def _train_loss_base(model, win, per):
    B = win.shape[0]
    z, loc, scale, cnt = model.states(win)                     # origins = every fine patch
    if JEPA:
        jw = JEPA * (min(1.0, PROG / JEPA_WARM) if JEPA_WARM > 0 else 1.0)
        if LATENT_EMA:
            with torch.no_grad():
                z_tgt = lat_tgt.states(win)[0]
        else:
            z_tgt = z.detach()
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
    ctx = _arch_ctx(z, keep)
    if LATENT or JEPA or DUALDEC or TRANS == "rssm":
        pred, prior = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], True, mem,
                                    ctx=ctx)
        loss = pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float())
        if JEPA:
            JEPA_STATS["pinball"], JEPA_STATS["w"] = loss.detach(), jw
            xp = win[:, PREFIX:].reshape(B, N_PATCH, PATCH)
            obs_ok = ~torch.isnan(xp) & (cnt > 0)[..., None]
            obs_val = to_model_space(torch.nan_to_num(xp), loc[..., None], scale[..., None])
            loss = loss + jw * jepa_loss(model, z, z_tgt, prior, keep, obs_val, obs_ok)
        if LATENT:
            loss = loss + LATENT * latent_consistency(z, prior, keep)
        if DUALDEC:
            loss = loss + _dualdec_term(model, z, ref, tgt, valid, keep, pred)
        if TRANS == "rssm":                # same association as train_arch.py's arch_loss
            t_, v_ = tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float()
            loss = loss + RSSM_POST * pinball(model.trans.pred_post, t_, v_) + RSSM_KL * model.trans.kl
        return loss
    pred = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], mem=mem, ctx=ctx)
    return pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float())


def _jepa_term(model, win, z, loc, scale, cnt, prior, keep):
    """Weighted JEPA loss for the TSAR_SDD / TSAR_RFL element path (same terms as _train_loss_base)."""
    B = win.shape[0]
    jw = JEPA * (min(1.0, PROG / JEPA_WARM) if JEPA_WARM > 0 else 1.0)
    if LATENT_EMA:
        with torch.no_grad():
            z_tgt = lat_tgt.states(win)[0]
    else:
        z_tgt = z.detach()
    JEPA_STATS["w"] = jw
    xp = win[:, PREFIX:].reshape(B, N_PATCH, PATCH)
    obs_ok = ~torch.isnan(xp) & (cnt > 0)[..., None]
    obs_val = to_model_space(torch.nan_to_num(xp), loc[..., None], scale[..., None])
    return jw * jepa_loss(model, z, z_tgt, prior, keep, obs_val, obs_ok)


_ZK = None
_RFL_BUF = {}                                                   # dataset index -> recent per-row losses
_RFL_REF = None                                                 # frozen reference model (TSAR_RFL_CKPT)


def _elements(model, win, per, aux=True):
    """Per-element pinball pieces: pred [N, H, NQ], tgt [N, H], mask [N, H], row index [N], extra loss,
    plus loc/scale per kept origin [N] and origin index j [N]."""
    B = win.shape[0]
    z, loc, scale, cnt = model.states(win)
    ref = seasonal_ref(win, loc, scale, per)
    pad = torch.full((B, MAX_H), float("nan"), device=win.device)
    fut = torch.cat([win[:, PREFIX:], pad], 1).unfold(1, MAX_H, PATCH)[:, 1:N_PATCH + 1]
    valid = (cnt > 0)[..., None] & ~torch.isnan(fut)
    tgt = to_model_space(torch.nan_to_num(fut), loc[..., None], scale[..., None])
    keep = valid.any(-1).view(-1)
    bj = torch.arange(B * N_PATCH, device=win.device)[keep]
    mem = model.memory(z, bj // N_PATCH, bj % N_PATCH) if CTX_READ else None
    extra = 0.0
    ctx = _arch_ctx(z, keep)
    if aux and (LATENT or JEPA or DUALDEC or TRANS == "rssm"):   # train_v2: world-model terms on the element path
        pred, prior = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], True, mem,
                                    ctx=ctx)
        if TRANS == "rssm":
            extra = extra + _rssm_term(model, tgt.reshape(B * N_PATCH, MAX_H)[keep],
                                       valid.reshape(B * N_PATCH, MAX_H)[keep].float())
        if DUALDEC:
            extra = extra + _dualdec_term(model, z, ref, tgt, valid, keep, pred)
        if JEPA:
            extra = extra + _jepa_term(model, win, z, loc, scale, cnt, prior, keep)
        if LATENT:
            extra = extra + LATENT * latent_consistency(z, prior, keep)
    else:
        pred = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], mem=mem, ctx=ctx)
    return (pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float(),
            bj // N_PATCH, extra, loc.reshape(-1)[keep], scale.reshape(-1)[keep], bj % N_PATCH)


def _pin_el(pred, tgt):
    err = tgt.unsqueeze(-1) - pred
    return torch.maximum(Q * err, (Q - 1) * err).sum(-1)                  # [N, H] summed over quantiles


def _sdd_el(pred, win, bi, j, loc, scale):
    """Distilled pinball for exact-law rows: the expected pinball under the generator's conditional law,
    evaluated at SDD_K stratified quantiles of that law and mapped through the same (monotone)
    asinh model-space transform as the targets. Its minimiser is the conditional quantile."""
    global _ZK
    rows, g, phi, sig = _SDD
    if _ZK is None:
        u = (torch.arange(SDD_K, dtype=torch.float32) + 0.5) / SDD_K
        _ZK = (math.sqrt(2) * torch.erfinv(2 * u - 1)).to(DEVICE)
    x = torch.nan_to_num(win[bi, PREFIX:]); gg = g[bi, PREFIX:]
    t0 = (j + 1) * PATCH - 1                                                # origin index in the fine window
    h = torch.arange(1, MAX_H + 1, device=win.device)
    idx = (t0[:, None] + h[None, :]).clamp(max=x.shape[1] - 1)
    ph, sg = phi[bi][:, None], sig[bi][:, None]
    e0 = (x.gather(1, t0[:, None]) - gg.gather(1, t0[:, None]))
    mu = gg.gather(1, idx) + ph ** h * e0
    sd = sg * torch.sqrt((1 - ph ** (2 * h)) / (1 - ph ** 2).clamp(min=1e-4))
    tk = to_model_space(mu[..., None] + sd[..., None] * _ZK, loc[:, None, None], scale[:, None, None])  # [N, H, K]
    err = tk[..., None] - pred[:, :, None, :]                               # [N, H, K, NQ]
    return torch.maximum(Q * err, (Q - 1) * err).sum(-1).mean(-1)          # [N, H]


def _row_loss(el, m, bi, B):
    num = torch.zeros(B, device=el.device).index_add_(0, bi, (el * m).sum(1))
    den = torch.zeros(B, device=el.device).index_add_(0, bi, m.sum(1))
    return num / den.clamp(min=1) / NQ, den


def _rfl_weights(rowl, den):
    """1/0 weight per row: real rows whose loss is outside the central RFL band of their dataset's
    recent losses (ring buffer of 256 per dataset) are dropped. Synthetic rows are always kept."""
    w = np.ones(len(rowl), np.float32)
    lo_q, hi_q = 50 * (1 - RFL), 100 - 50 * (1 - RFL)
    for b, (l, n, d) in enumerate(zip(rowl, den, _SRC)):
        if d < 0 or n == 0:
            continue
        buf = _RFL_BUF.setdefault(int(d), [])
        if PROG >= RFL_WARM and len(buf) >= 64:
            lo, hi = np.percentile(buf, [lo_q, hi_q])
            w[b] = float(lo <= l <= hi)
        buf.append(float(l))
        if len(buf) > 256:
            del buf[0]
    return torch.from_numpy(w).to(DEVICE)


def train_loss_long(model, win, per):
    """TSAR_LONG_H: near loss on every origin (leads <= MAX_H, as train_loss) plus a far loss on FAR_ORIG
    sampled origins per row for leads MAX_H+1..LONG_H. win: [B, TDATA]; the encoder sees win[:, :TWIN]."""
    B = win.shape[0]
    x = win[:, :TWIN]
    z, loc, scale, cnt = model.states(x)
    ref = seasonal_ref(x, loc, scale, per)                                          # [B, n, MAX_H], causal
    pad = torch.full((B, LONG_H), float("nan"), device=win.device)
    fut = torch.cat([win[:, PREFIX:], pad], 1).unfold(1, LONG_H, PATCH)[:, 1:N_PATCH + 1]   # [B, n, LONG_H]
    valid = (cnt > 0)[..., None] & ~torch.isnan(fut)
    tgt = to_model_space(torch.nan_to_num(fut), loc[..., None], scale[..., None])
    zf, BN_ = z.reshape(B * N_PATCH, -1), B * N_PATCH
    vn, tn = valid[..., :MAX_H].reshape(BN_, MAX_H), tgt[..., :MAX_H].reshape(BN_, MAX_H)
    keep = vn.any(-1)
    mem = None
    if CTX_READ:
        bj = torch.arange(BN_, device=win.device)[keep]
        mem = model.memory(z, bj // N_PATCH, bj % N_PATCH)
    pred = model.rollout_near(zf[keep], ref.reshape(BN_, MAX_H)[keep], mem=mem)
    near = pinball(pred, tn[keep], vn[keep].float())
    # far chunk: a few origins per row, late enough in the window to carry a long context
    j = torch.randint(min(FAR_J0, N_PATCH - 1), N_PATCH, (B, FAR_ORIG), device=win.device)
    b = torch.arange(B, device=win.device)[:, None].expand(B, FAR_ORIG)
    vf = valid[b, j][..., MAX_H:].reshape(B * FAR_ORIG, -1)       # fixed batch of B * FAR_ORIG origins (no
    memf = model.memory(z, b.reshape(-1), j.reshape(-1)) if CTX_READ else None   # filtering: stable MPS shapes)
    reff = seasonal_ref(x, loc[b, j], scale[b, j], per, H=LONG_H, j=j, n=N_PATCH)   # [B, FAR_ORIG, LONG_H]
    pf = model.rollout_long(z[b, j].reshape(B * FAR_ORIG, -1), reff.reshape(B * FAR_ORIG, LONG_H), mem=memf)
    far = pinball(pf[:, MAX_H:], tgt[b, j][..., MAX_H:].reshape(B * FAR_ORIG, -1), vf.float())
    return near + FAR_W * far


def train_loss(model, win, per):
    global _RFL_REF
    if LONG_H:
        return train_loss_long(model, win, per)
    if not SDD_SHARE and not RFL:
        return _train_loss_base(model, win, per)                       # bit-identical default path
    B = win.shape[0]
    pred, tgt, m, bi, extra, loc, scale, j = _elements(model, win, per)
    el = _pin_el(pred, tgt)
    if _SDD is not None:
        s_ = _SDD[0][bi]
        if s_.any():
            el = el.clone()
            el[s_] = _sdd_el(pred[s_], win, bi[s_], j[s_], loc[s_], scale[s_])
    w = torch.ones(B, device=win.device)
    if RFL:
        if RFL_CKPT and _RFL_REF is None:
            _RFL_REF = CausalWorldModel().to(DEVICE)
            ck = torch.load(RFL_CKPT, map_location=DEVICE, weights_only=False)
            mk, uk = _RFL_REF.load_state_dict(ck.get("ema", ck.get("model", ck)), strict=False)
            assert not uk and all(k_.startswith("jepa_") for k_ in mk), f"TSAR_RFL_CKPT does not match: {mk} {uk}"
            _RFL_REF.eval().requires_grad_(False)
        if RFL_CKPT:
            with torch.no_grad():
                p2, t2, m2, b2 = _elements(_RFL_REF, win, per, aux=False)[:4]
                rowl, den = _row_loss(_pin_el(p2, t2), m2, b2, B)
        else:
            rowl, den = _row_loss(el.detach(), m, bi, B)
        w = _rfl_weights(rowl.cpu().numpy(), den.cpu().numpy())
    mw = m * w[bi][:, None]
    main = (el * mw).sum() / mw.sum().clamp(min=1) / NQ
    if JEPA:
        JEPA_STATS["pinball"] = main.detach()
    return main + extra


# Training (fixed time budget)
# ---------------------------------------------------------------------------

model = CausalWorldModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
rng = np.random.default_rng(SEED)
ema = copy.deepcopy(model).requires_grad_(False) if EMA else model
lat_tgt = copy.deepcopy(model).requires_grad_(False) if JEPA and LATENT_EMA else None   # JEPA target encoder
n_tgt = sum(p.numel() for p in lat_tgt.parameters()) if lat_tgt is not None else 0
n_jepa = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("jepa_"))


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


def ctx_scale(ctxs):
    """Normaliser scale of the last patch the model would see for each context, as causal_stats computes it
    over the fine window (last CONTEXT points): mean |step| for NORM="last", else std. [N] float64."""
    out = np.ones(len(ctxs))
    for i, c in enumerate(ctxs):
        c = np.asarray(c, dtype=np.float64)[-CONTEXT:]
        o = c[~np.isnan(c)]
        d = np.abs(np.diff(c)); d = d[~np.isnan(d)]
        s = d.mean() if (NORM == "last" and len(d)) else 0.0
        if s <= 1e-6:
            s = o.std() if len(o) else 0.0
        if s <= 1e-6:
            s = abs(o[-1]) if len(o) and abs(o[-1]) > 1e-6 else 1.0
        out[i] = s
    return out


MIX_TAIL = 2.15


def densify_scenarios(qs, M):
    """qs: [N, S=NQ, NQ, T] forecasts from the NQ scenario contexts (scenario j = level quantile Q[j]).
    The 9 scenarios only span the 0.1..0.9 range of the boundary level, so an equal-weight mixture of them is
    under-dispersed (and lumpy when level uncertainty dominates). Treat each scenario's quantiles as a
    function of the level's normal score z_j = Phi^-1(Q[j]), linear between scenarios and linearly
    extrapolated beyond the ends, and evaluate it at M equal-mass levels u_k = (k + 0.5) / M.
    Exact for location-shift families. Returns [N, M, NQ, T], to be mixed with equal weights."""
    Qd = torch.from_numpy(P.QUANTILES).double()
    zS = math.sqrt(2) * torch.erfinv(2 * Qd - 1)                                              # [S]
    u = (torch.arange(M, dtype=torch.float64) + 0.5) / M
    zk = math.sqrt(2) * torch.erfinv(2 * u - 1)                                              # [M]
    j = (torch.searchsorted(zS, zk) - 1).clamp(0, NQ - 2)                                    # segment, ends extrapolate
    a = ((zk - zS[j]) / (zS[j + 1] - zS[j])).numpy()[None, :, None, None]
    j = j.numpy()
    return (qs[:, j] + a * (qs[:, j + 1] - qs[:, j])).astype(np.float32)


def mix_quantiles(qs, w):
    """Quantiles of an equal-shape mixture. qs: [N, S, NQ, T] per-scenario quantiles at P.QUANTILES, w: [S]
    scenario weights. Each component CDF is piecewise linear through its 9 knots with uniform tails to 0 and 1,
    MIX_TAIL end-segments wide (2.15 matches the mean of a Gaussian's 10% tail). The mixture CDF is then piecewise linear on the union of all knots, so it is
    inverted exactly. Returns [N, NQ, T]."""
    q = torch.from_numpy(np.sort(qs, axis=2)).double().permute(0, 3, 1, 2)                 # [N, T, S, NQ]
    lo = q[..., :1] - MIX_TAIL * (q[..., 1:2] - q[..., :1])
    hi = q[..., -1:] + MIX_TAIL * (q[..., -1:] - q[..., -2:-1])
    knots = torch.cat([lo, q, hi], -1)                                                      # [N, T, S, NQ+2]
    span = (knots[..., -1:] - knots[..., :1]).abs().clamp(min=1e-12)
    knots = knots + torch.arange(NQ + 2, dtype=knots.dtype) * 1e-9 * span                   # strictly increasing
    lev = torch.cat([torch.zeros(1, dtype=torch.float64), torch.from_numpy(P.QUANTILES).double(), torch.ones(1, dtype=torch.float64)])  # [NQ+2]
    N, T, S, K = knots.shape
    cand = knots.reshape(N, T, S * K).sort(-1).values                                       # [N, T, S*K]
    F = torch.zeros_like(cand)
    for s in range(S):
        ks = knots[:, :, s].contiguous()                                                     # [N, T, K]
        j = torch.searchsorted(ks, cand.contiguous()).clamp(1, K - 1)
        x0, x1 = ks.gather(-1, j - 1), ks.gather(-1, j)
        f = lev[j - 1] + (lev[j] - lev[j - 1]) * ((cand - x0) / (x1 - x0)).clamp(0, 1)
        f = torch.where(cand <= ks[..., :1], torch.zeros_like(f), torch.where(cand >= ks[..., -1:], torch.ones_like(f), f))
        F += float(w[s]) * f
    F = torch.cummax(F, -1).values + torch.arange(S * K, dtype=F.dtype) * 1e-12            # monotone for inversion
    tau = torch.from_numpy(P.QUANTILES).double().expand(N, T, NQ).contiguous()
    j = torch.searchsorted(F, tau).clamp(1, S * K - 1)
    f0, f1, c0, c1 = F.gather(-1, j - 1), F.gather(-1, j), cand.gather(-1, j - 1), cand.gather(-1, j)
    out = c0 + (c1 - c0) * ((tau - f0) / (f1 - f0).clamp(min=1e-15)).clamp(0, 1)
    return out.permute(0, 2, 1).float().numpy()                                              # [N, NQ, T]


def predict_long(ctxs, h, m_season, freq):
    """Calibrated chunked rollout for h > MAX_H. The default (median feedback) appends the median and restarts,
    so (a) every later chunk's spread restarts at a 1-step-ahead width, as if the boundary level were known,
    and (b) the smooth median path shrinks the context's mean |step|, so the normaliser and with it the raw
    interval width collapse chunk after chunk. Flags:
      CHUNK_MIX:   chunk c >= 1 is forecast from NQ scenario contexts, each extended with one quantile path of
                   the marginal forecast so far (comonotone scenarios); the 9 scenario forecasts are interpolated
                   to CHUNK_MIX_M equal-mass levels (densify_scenarios) and the next marginal is their exact
                   equal-weight mixture. The model itself decides
                   how much boundary uncertainty persists (mean reversion, seasonality) instead of a rule.
      CHUNK_SCALE: a chunk's deviations from its own median are multiplied by s0 / s_c, clipped to
                   [1, CHUNK_SCALE_MAX], where s0 is the real context's scale and s_c the extended context's.
    Uses only the contexts and the model's own forecasts."""
    ctxs = [np.asarray(c, dtype=np.float32) for c in ctxs]
    N, mid = len(ctxs), NQ // 2
    s0 = ctx_scale(ctxs) if CHUNK_SCALE else None

    def anchor(q, ext):                                    # q: [n, NQ, T] for contexts ext, original scale s0r
        if not CHUNK_SCALE:
            return q
        r = np.clip(s0r / ctx_scale(ext), 1.0, CHUNK_SCALE_MAX).astype(np.float32)[:, None, None]
        med = q[:, mid:mid + 1]
        return med + r * (q - med)

    q = predict_chunk(ctxs, min(MAX_H, h), m_season, freq)                                  # [N, NQ, h1]
    out = [q]
    if CHUNK_MIX:
        w = np.full(CHUNK_MIX_M, 1.0 / CHUNK_MIX_M)
        paths = np.sort(q, axis=1)                                                          # marginal so far
        for s in range(MAX_H, h, MAX_H):
            hc = min(MAX_H, h - s)
            ext = [np.concatenate([ctxs[i], paths[i, j]])[-TCTX:] for i in range(N) for j in range(NQ)]
            s0r = np.repeat(s0, NQ) if CHUNK_SCALE else None
            qs = anchor(predict_chunk(ext, hc, m_season, freq), ext)                        # [N*NQ, NQ, hc]
            qm = mix_quantiles(densify_scenarios(qs.reshape(N, NQ, NQ, hc), CHUNK_MIX_M), w)
            out.append(qm)
            paths = np.concatenate([paths, qm], -1)
        return np.concatenate(out, -1)
    cur = ctxs
    for s in range(MAX_H, h, MAX_H):
        hc = min(MAX_H, h - s)
        cur = [np.concatenate([c, out[-1][i, mid]])[-TCTX:] for i, c in enumerate(cur)]
        s0r = s0
        qc = anchor(predict_chunk(cur, hc, m_season, freq), cur)
        out.append(qc)
    return np.concatenate(out, -1)


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


@torch.no_grad()
def predict(ctxs, h, m_season, freq):
    if h > MAX_H and (CHUNK_MIX or CHUNK_SCALE):
        return predict_long(ctxs, h, m_season, freq)
    if h > MAX_H and TTA_LONG == "paths":
        return predict_paths(ctxs, h, m_season, freq)
    if h > H_OUT:                          # long horizons: roll the world model forward in H_OUT chunks,
        out, ctxs = [], [np.asarray(c, dtype=np.float32) for c in ctxs]   # feeding back the median
        for s in range(0, h, H_OUT):
            q = predict_chunk(ctxs, min(H_OUT, h - s), m_season, freq)
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
        f = torch.stack(qs).mean(0)
        if TTA_CLAMP:                     # TSAR_TTA_CLAMP (train_tta.py): clamp from the series' own context
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
    m, offset = adapt(x) if TTA_STEPS else (ema, None)
    z, loc, scale, _ = m.states(x)
    zl = z[:, -1] if offset is None else z[:, -1] + offset
    ms = torch.full((len(x),), int(m_season), device=DEVICE)
    if LONG_H:                             # last origin only: the full [B, n, LONG_H] table is large on MPS
        jl = torch.full((len(x), 1), z.shape[1] - 1, device=DEVICE)
        ref = seasonal_ref(x, loc[:, -1:], scale[:, -1:], ms, H=LONG_H, j=jl, n=z.shape[1])[:, 0]
    else:
        ref = seasonal_ref(x, loc, scale, ms)[:, -1]
    mem = None
    if CTX_READ:
        B, n = z.shape[:2]
        mem = m.memory(z, torch.arange(B, device=z.device), torch.full((B,), n - 1, device=z.device))
    ctx = None
    if TRANS == "attn":                                   # the transition reads this origin's context states
        B, n = z.shape[:2]
        ctx = {"z": z, "b": torch.arange(B, device=z.device), "j": torch.full((B,), n - 1, device=z.device)}
    if LONG_H:
        q = m.rollout_long(zl, ref, mem=mem)[:, :h]       # one shot to LONG_H, origin = last context patch
    else:
        q = m.rollout(zl, ref, mem=mem, ctx=ctx)[:, :h]   # origin = last context patch
    if SELFCAL:
        q = self_calibrate(m, x, z, loc, scale, h, q)
    q = from_model_space(q, loc[:, -1, None, None], scale[:, -1, None, None])
    return q if sign > 0 else -q.flip(-1)                 # reflect: q_tau(x) = -q_{1-tau}(-x)


    return np.concatenate(out)



CKPT = os.path.join(P.CACHE_DIR, "ckpt", f"{RUN}.pt") if RUN and not EVAL_ONLY else None
step, smooth, done = 0, None, 0.0          # done = training seconds completed in earlier sessions
CK_COMPLETE = False
if EVAL_ONLY:                              # frozen weights: verify the data/model config, load, never write
    bad = cfg_mismatch(CK0["cfg"], EVAL_FREE_KEYS | (set() if IS_TEST else INFER_KEYS))
    if bad:
        raise SystemExit("eval-only: the checkpoint does not match this data/model config:\n  " + "\n  ".join(bad))
    model.load_state_dict(CK0["model"])
    if EMA:
        ema.load_state_dict(CK0.get("ema", CK0["model"]))
    step, done, CK_COMPLETE = CK0["step"], CK0["done"], bool(CK0.get("complete"))
    print(f"eval-only: frozen weights at step {step}, {done / 60:.1f} min trained, complete={CK_COMPLETE}", flush=True)
    if IS_TEST and not CK_COMPLETE:
        raise SystemExit("TSAR_SPLIT=test needs a completed (end-of-budget) checkpoint")
    del CK0
elif CKPT and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if ck.get("zs", 0) != ZS:
        raise SystemExit(f"{RUN} was trained with TSAR_ZS={ck.get('zs', 0)}; refusing to resume with TSAR_ZS={ZS}")
    if "cfg" not in ck:                    # audit_v2 L-1: a legacy (old-suite) checkpoint must not warm-start BENCH
        if BENCH:
            raise SystemExit(f"{CKPT} has no stored run config (bench flag, train fingerprint); refusing to "
                             "resume it under TSAR_BENCH=1. Delete it to train from scratch.")
        print(f"WARNING: {CKPT} predates the resume guard; only TSAR_ZS was checked", flush=True)
    else:
        bad = cfg_mismatch(ck["cfg"], CTRL_KEYS)
        if bad:
            raise SystemExit(f"refusing to resume {RUN}: the checkpoint was trained with another config:\n  "
                             + "\n  ".join(bad))
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if EMA:
        ema.load_state_dict(ck.get("ema", ck["model"]))
    if lat_tgt is not None:
        lat_tgt.load_state_dict(ck.get("lat_tgt", ck["model"]))
    step, done, rng = ck["step"], ck["done"], ck["rng"]
    print(f"resumed {RUN} at step {step}, {done / 60:.1f} min trained", flush=True)


def save_ckpt(elapsed, final=False):
    if CKPT:
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "done": elapsed, "rng": rng, "zs": ZS, "cfg": run_cfg(), "complete": final,
                    "budget": {"steps": STEPS, "seconds": BUDGET},
                    **({"lat_tgt": lat_tgt.state_dict()} if lat_tgt is not None else {})}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)


t_start = time.time() - done
last_ckpt = last_eval = time.time()
while True:
    if EVAL_ONLY:                          # frozen checkpoint: no training step
        break
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
    if lat_tgt is not None:                # JEPA target encoder: slow EMA of the online model
        with torch.no_grad():
            d = min(LATENT_EMA, (1 + step) / (10 + step))
            for pt, pm in zip(lat_tgt.parameters(), model.parameters()):
                pt.lerp_(pm, 1 - d)
    smooth = loss.item() if smooth is None else 0.98 * smooth + 0.02 * loss.item()
    if step % 100 == 0:
        print(f"step {step:5d}  loss {smooth:.4f}  lr {lr:.2e}  {elapsed:5.0f}s"
              + ("  " + jepa_str() if JEPA and JEPA_STATS else ""), flush=True)
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
save_ckpt(training_seconds, final=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

split = "test" if IS_TEST else "val"


def _val_record_path():
    return (EVAL_ONLY or CKPT) + ".val.json"


if EVAL_ONLY and IS_TEST:                  # same session: reproduce the recorded VAL score of these weights first
    rp = _val_record_path()
    rec = json.load(open(rp)) if os.path.exists(rp) else None
    if rec is None or rec.get("sha256") != CK_SHA:
        raise SystemExit(f"no recorded VAL score for checkpoint sha256 {CK_SHA} ({rp}); not scoring TEST")
    rv = EVALUATE(predict, split="val")
    print(f"val_wql_rel:      {rv['wql_rel']:.6f}   (recorded {rec['val_wql_rel']:.6f})", flush=True)
    if abs(rv["wql_rel"] - rec["val_wql_rel"]) > 1e-3:
        raise SystemExit("VAL score not reproduced within 1e-3 by the frozen weights; not scoring TEST")
res = EVALUATE(predict, split=split)
if split == "val" and (CKPT or EVAL_ONLY):  # bind the VAL score to the exact checkpoint bytes (for the TEST check)
    _ck_path = EVAL_ONLY or CKPT
    _sha = CK_SHA or hashlib.sha256(open(_ck_path, "rb").read()).hexdigest()
    if not (EVAL_ONLY and os.path.exists(_val_record_path())):
        json.dump({"sha256": _sha, "val_wql_rel": res["wql_rel"], "val_mase_rel": res["mase_rel"],
                   "val_cov80_error": res["cov80_error"], "val_wql_rel_family": res.get("wql_rel_family"),
                   "source": "eval-only" if EVAL_ONLY else "training run", "git": _git_head(),
                   "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(_val_record_path(), "w"), indent=1)
if EVAL_ONLY:
    print(f"eval_only_ckpt_sha256: {CK_SHA}")
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
if JEPA:
    print(f"jepa_params_M:    {n_jepa / 1e6:.3f}  (target encoder copy: {n_tgt / 1e6:.3f}M, not trained)")
    print(f"jepa_last:        {jepa_str() if JEPA_STATS else 'n/a (no training steps this session)'}")
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
