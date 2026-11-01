"""
train_wm.py (Haddad, 2026-10-30): shared harness for screening world-model FAMILIES on equal terms.

The model is chosen by TSAR_WM=<family> from wm.REGISTRY; the family interface (encode / decode, ablation modes,
aux losses) is in wm/base.py, the fairness rules in docs/wm_families.md. Everything else is copied (not imported)
from train_arch2.py and is identical for every family: data (bench TRAIN, TSAR_GIFT_CLEAN, TSAR_DEV holdout),
synthetic mix (synth3 / synth5 / curriculum / augmentation), causal per-patch normalisation (NORM=last, frame
features, asinh), the seasonal-naive reference and its learned per-quantile skip, multi-origin pinball loss, AdamW
with warmup + cosine, fixed-step budget, checkpoint / resume, and bench evaluation (chunked median feedback for
horizons beyond MAX_H). train_arch2.py is untouched; family `gru` is its syn5 recipe + A2_DRIVE=1 design.

Keys read here (besides the train_arch2 recipe keys, same meaning and defaults as there):
  TSAR_WM          family name (default gru)
  TSAR_WM_ABL      none | direct | copy | zero   (default none; see wm/base.py). zero is an inference key: it may be
                   set with TSAR_EVAL_ONLY on a checkpoint trained with none (VAL only, warning printed).
  TSAR_WM_WIDTH    float width multiplier (default 1.0): d_model = 8 * round(TSAR_D_MODEL * width / 8). Depth is
                   TSAR_N_LAYERS. Parameter ladder for gru in docs/wm_families.md.
  TSAR_WM_INFO     1 = build the model, print params / MACs, run the causality check and exit (no data load).
  TSAR_WM_EVAL_TASKS  smoke tests only: comma-separated VAL task names to score instead of all 30. The metric lines
                   are then printed as smoke_val_* (never val_*), so jobq and the leaderboard can never take them
                   for a result. Allowed with TSAR_EVAL_ONLY (still smoke_val_*), refused with TSAR_DEV.

Long horizons (2026-10-31; both default-off, defaults bit-identical):
  TSAR_WM_LONG     median | paths (default median). How horizons beyond one rollout (h > MAX_H) are continued.
                   median: feed the median chunk back as if observed (the original behaviour).
                   paths:  Chronos-2-style quantile paths. Chunk 1 is one rollout; its sorted quantiles at the levels
                   in TSAR_WM_LONG_PATHS become |S| context paths, each rolled forward; the |S| predictive quantile
                   sets are combined as a mixture distribution (mixture_quantiles below), and path p then appends the
                   MIXTURE's level-tau_p quantile, so every path stays a comonotone level-tau_p trajectory and the
                   cost is |S| rollouts per extra chunk (not |S|^chunks). An inference key: allowed with
                   TSAR_EVAL_ONLY on a checkpoint trained with median (VAL only, warning printed).
  TSAR_WM_LONG_PATHS  comma-separated quantile levels to feed back under paths (default all 9: 0.1,...,0.9); each
                   must be one of prepare.QUANTILES. Inference key, like TSAR_WM_LONG.
  TSAR_MAX_H       native rollout length in points (default 64; a multiple of PATCH). K = MAX_H / PATCH lead
                   patches are imagined from the latent state with no observation feedback; training windows are
                   CONTEXT + MAX_H long, so every origin j <= CONTEXT/PATCH - 1 has the full MAX_H future. Refused for
                   a family whose max_k (wm/base.py) is below K. Changes params (step_emb, pos tables grow with K).
  TSAR_WM_ORIGINS  training only: at most this many origins (b, j) per batch, a uniform random subset of the valid
                   ones drawn from the data rng (default 0 = all, as before). Memory and decode compute scale with
                   origins x K, so MAX_H=320 at BATCH=128 needs it (all origins = ~9x the lead states of MAX_H=64).
  TSAR_WM_LEADW    training only: pinball weight of lead steps beyond 64 relative to steps 1..64 (default 1 =
                   flat; refused unless MAX_H > 64). The loss is the weighted mean sum(w*l)/sum(w*mask). With
                   MAX_H=320, flat gives steps > 64 80% of the loss; LEADW=0.25 makes the two halves equal.

Context: TSAR_CONTEXT (default 256; a multiple of PATCH) is the inference context and the context part of every
  training window (WINDOW = CONTEXT + MAX_H). The bench passes up to 2048 points, cropped here to the last CONTEXT;
  predict_chunk's eval batch shrinks for CONTEXT > 512 (as train_syn5). Smoke-checked at 512 and 1024 (2026-10-31).

Test-time augmentation (2026-10-31; default off, defaults bit-identical):
  TSAR_WM_TTA      0 | 1 | sym | clip (default 0). Timestral's "clampsym" (timestral/forecaster.py, forecast(): both
                   options on by default there, and our bench runs Timestral with them on), applied to the whole
                   forecast of any family, around the long-horizon machinery (as Timestral does around its
                   block-autoregressive decode):
                   sym:  forecast x and -x (each a complete forecast, chunked/paths continuation included), reflect
                         the second (-Q(1-p): negate, reverse the quantile axis; our 9 levels are symmetric about
                         0.5) and average: 0.5 * (q(x) + (-q(-x)).flip(levels)).
                   clip: after averaging, clamp at 0 every series whose observed context (the full context the
                         bench passes, up to 2048 points, not only the last CONTEXT; NaN = unobserved) is entirely
                         non-negative and has at least one observed value (nonnegative_context()).
                   1 = sym then clip (Timestral's default). Not reproduced: Timestral's quantile_real_cap_k cap at
                   [min - 1e4 s, max + 1e4 s] of the context, which is part of every Timestral pass, not of clampsym.
                   An inference key: allowed with TSAR_EVAL_ONLY on a checkpoint trained without it (VAL only,
                   warning printed). Costs 2x predict time with sym.

Recalibration (2026-10-31, calibration engineer; default off, defaults bit-identical; tools/wmrecal.py has the maths):
  TSAR_WM_RECAL    0 | lead | freqlead | file (default 0). After the full forecast (paths / TTA included), every
                   quantile is moved away from or toward the median by a scale per (quantile level, lead bucket[,
                   frequency group]): q' = m + c * (q - m); the median is unchanged. c is fitted by pinball
                   minimisation on calibration windows cut from bench TRAIN series ONLY (context y[:-H], target
                   y[-H:], H = 288 / 56 / 13 for sub-daily / daily / coarser); VAL/TEST are never read by the fit.
                   lead / freqlead without TSAR_DEV: fitted on the training pool itself (in-sample targets) and
                     applied to VAL. Smoke (ssm s0): TRAIN windows are already covered 0.79-0.93 in-sample, so this
                     narrows the intervals and hurts VAL; kept as the documented negative control.
                   lead / freqlead with TSAR_DEV=1: fitted on the DEV families that devsplit.holdout removed from
                     training (out-of-family for this model), written to TSAR_WM_RECAL_MAP, NOT applied to the DEV
                     score (which stays raw).
                   file: load the map from TSAR_WM_RECAL_MAP (written by a TSAR_DEV run) and apply it to VAL.
                   An inference key: allowed with TSAR_EVAL_ONLY on a checkpoint trained without it (VAL only).
  TSAR_WM_RECAL_N  calibration series per TRAIN dataset (default 150; inference key with TSAR_WM_RECAL).
  TSAR_WM_RECAL_MAP  json path of the map: written under TSAR_DEV, read by TSAR_WM_RECAL=file.

Period lever (2026-10-31, docs/period_lever.md; both default-off, defaults bit-identical):
  TSAR_PERIOD      1 = run the parameter-free period detector (wm/ssm.detect_periods: rFFT + Fisher's g-test with
                   Bonferroni on the AR(1)-whitened periodogram, >= 3 observed cycles, comb-autocorrelation coherence,
                   up to 4 integer periods) on the last 256 points ending at EVERY origin (causal; at inference the
                   window is exactly the context), and attach to the Context: ctx.periods [B, n, 4] (0 = none),
                   ctx.tok_phase [B, n, 16] (sin/cos phase at each context patch start under that patch's own
                   periods, active flag, log-period) and ctx.fut_phase [B, n, K, 16] (the same coordinates at every
                   lead patch start of origin j under origin j's periods: the rollout's clock). Families that do not
                   read them are unaffected; ssm adds one shared zero-init embedding of both to its residual inputs
                   (all ablations, so the direct twin gets the same clock).
  TSAR_PERIOD_BIN  1 (needs TSAR_PERIOD=1) = ssm phase binning: encoder outputs of the last 32 context patches are
                   averaged into 16 phase bins per detected period (a cycle template per origin); every lead patch
                   reads the template at its own phase (summed over periods), through zero-init projections into the
                   rollout input and the decoder head input (all ablations).

Dropped from train_arch2 (refused as unknown keys if set): TSAR_LOOPS, TSAR_LATENT (use a family aux loss), GATE,
CTX_READ, PREFIX*, TTA*, SELFCAL*, ENS, KS_FRAC, ZS, SEAS_REF (always on), every TSAR_A2_* axis (families own their
architecture) and the old-suite path (TSAR_BENCH=1 is required).

Hard rules: TEST is never scored here (TSAR_SPLIT=test is refused outright, also with TSAR_EVAL_ONLY); every
unknown TSAR_* key is refused (strict-key check: the keys this file reads through _env("TSAR_..."), the tool keys
read by tools/bench.py, and the selected family's declared env_keys; nothing else).

Startup prints `wm: ...` (family, ablation, width, d_model, params, trainable params, per-group params, MACs per
forecast) and `wm_causal: ok` (forecasts at origins j < c do not change when the series after patch c is altered).
"""

import os
import re
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

import prepare as P

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import wm as WM                                            # noqa: E402
from wm.base import WMConfig, Context, to_model_space, from_model_space   # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (train_arch2 defaults)
# ---------------------------------------------------------------------------

PATCH = 8
CONTEXT = 256
MAX_H = 64
D_MODEL = 96
N_LAYERS = 2
N_HEADS = 4
BATCH = 128
LR = 3e-3
WEIGHT_DECAY = 0.01
SEED = 0

_env = os.environ.get
CTRL_KEYS = {"TSAR_RUN", "TSAR_SPLIT", "TSAR_EVAL_MIN", "TSAR_EVAL_ONLY", "TSAR_DEVICE", "TSAR_MINUTES",
             "TSAR_WM_EVAL_TASKS", "TSAR_WM_INFO"}                 # run control: may differ on resume
EVAL_FREE_KEYS = CTRL_KEYS | {"TSAR_SEED", "TSAR_STEPS"}           # training-only: irrelevant to frozen weights
INFER_KEYS = {"TSAR_WM_ABL"} if _env("TSAR_WM_ABL") == "zero" else set()
LONG = _env("TSAR_WM_LONG", "median")
if LONG not in ("median", "paths"):
    raise SystemExit(f"TSAR_WM_LONG={LONG!r} not in (median, paths)")
if _env("TSAR_WM_LONG_PATHS") and LONG != "paths":
    raise SystemExit("TSAR_WM_LONG_PATHS needs TSAR_WM_LONG=paths")
if LONG != "median":
    INFER_KEYS |= {"TSAR_WM_LONG", "TSAR_WM_LONG_PATHS"}
TTA = _env("TSAR_WM_TTA", "0")
if TTA not in ("0", "1", "sym", "clip"):
    raise SystemExit(f"TSAR_WM_TTA={TTA!r} not in (0, 1, sym, clip)")
if TTA != "0":
    INFER_KEYS |= {"TSAR_WM_TTA"}
if _env("TSAR_WM_RECAL", "0") != "0":
    INFER_KEYS |= {"TSAR_WM_RECAL", "TSAR_WM_RECAL_N", "TSAR_WM_RECAL_MAP"}
if _env("TSAR_SPLIT", "val") != "val":
    raise SystemExit("train_wm never scores TEST or holdout (TSAR_SPLIT must be unset or val); the confirmatory "
                     "TEST run is a separate, pre-registered procedure")
EVAL_ONLY = _env("TSAR_EVAL_ONLY")


def _tsar_env():
    return {k: v for k, v in os.environ.items() if k.startswith("TSAR_")}


CK_SHA = CK0 = None
if EVAL_ONLY:                                  # frozen checkpoint, VAL (or DEV) only
    _ck_bytes = open(EVAL_ONLY, "rb").read()
    CK_SHA = hashlib.sha256(_ck_bytes).hexdigest()
    print(f"eval-only: {EVAL_ONLY}  sha256={CK_SHA}", flush=True)
    CK0 = torch.load(io.BytesIO(_ck_bytes), map_location="cpu", weights_only=False)
    del _ck_bytes
    if "cfg" not in CK0:
        raise SystemExit(f"eval-only: {EVAL_ONLY} has no stored run config")
    _stored = {k: v for k, v in CK0["cfg"]["env"].items() if k not in EVAL_FREE_KEYS}
    _cur = {k: v for k, v in _tsar_env().items() if k not in EVAL_FREE_KEYS}
    for _k in sorted(set(_stored) | set(_cur)):
        if _k in _cur and _cur[_k] != _stored.get(_k):
            if _k in INFER_KEYS:
                print(f"eval-only WARNING: inference setting {_k}={_cur[_k]} differs from training "
                      f"({_stored.get(_k)}): exploratory, VAL only", flush=True)
            else:
                raise SystemExit(f"eval-only: {_k}={_cur[_k]!r} but the checkpoint was trained with "
                                 f"{_stored.get(_k)!r}; unset it (the stored config is restored automatically)")
        elif _k not in _cur:
            os.environ[_k] = _stored[_k]
    print("eval-only config: " + " ".join(f"{k}={v}" for k, v in sorted(_stored.items())), flush=True)

# --- family and strict-key check ----------------------------------------------------------------------------
WM_NAME = _env("TSAR_WM", "gru")
FamilyCls, FAMILY_FILE = WM.get(WM_NAME)
_TOOL_KEYS = {"TSAR_GIFT_CLEAN", "TSAR_BENCH_EXTRA", "TSAR_BENCH_GEP"}  # read by tools/bench.py (GEP: stamp-guarded)
_FAM_KEYS = set(FamilyCls.env_keys)                             # the family's own knobs, declared explicitly
_bad = sorted(k for k in _FAM_KEYS if not k.startswith(f"TSAR_WM_{WM_NAME.upper()}_"))
if _bad:
    raise SystemExit(f"family {WM_NAME!r}: env_keys {_bad} must start with TSAR_WM_{WM_NAME.upper()}_")
# keys this file actually reads (every read goes through _env("TSAR_...")), not every TSAR_ string in a docstring
_KNOWN = set(re.findall(r'_env\("(TSAR_[A-Z0-9_]+)"', open(os.path.abspath(__file__)).read())) | _TOOL_KEYS | _FAM_KEYS
_unknown = sorted(set(_tsar_env()) - _KNOWN)
if _unknown:
    raise SystemExit(f"train_wm: unknown TSAR_* keys {_unknown} for family {WM_NAME!r} (typo, a train_arch2-only "
                     "flag, or another family's key); refusing to run")

CONTEXT = int(_env("TSAR_CONTEXT", CONTEXT)); MAX_H = int(_env("TSAR_MAX_H", MAX_H)); BATCH = int(_env("TSAR_BATCH", BATCH))
STEPS = int(_env("TSAR_STEPS", 0))
BENCH = int(_env("TSAR_BENCH", 0))
INFO = int(_env("TSAR_WM_INFO", 0))
if not BENCH and not INFO:
    raise SystemExit("train_wm needs TSAR_BENCH=1 (the old-suite path was not ported)")
D_MODEL = int(_env("TSAR_D_MODEL", D_MODEL))
N_LAYERS = int(_env("TSAR_N_LAYERS", N_LAYERS))
N_HEADS = int(_env("TSAR_N_HEADS", N_HEADS))
WIDTH = float(_env("TSAR_WM_WIDTH", 1.0))
ABL = _env("TSAR_WM_ABL", "none")
LR = float(_env("TSAR_LR", LR))
SEED = int(_env("TSAR_SEED", SEED))
EMA = float(_env("TSAR_EMA", 0))
SYN2_SHARE = float(_env("TSAR_SYN2", 0))
SYN_FRAC_ENV = _env("TSAR_SYN_FRAC")
SYN3_SHARE = float(_env("TSAR_SYN3", 0))
AUG_P = float(_env("TSAR_AUG", 0))
SYN_CURR = float(_env("TSAR_SYN_CURR", 0))
PROG = 0.0
if SYN3_SHARE > 0 or AUG_P > 0:
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import synth3 as S3
SYN5_SHARE = float(_env("TSAR_SYN5", 0))
if SYN5_SHARE > 0:
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import synth5 as S5
DEV = int(_env("TSAR_DEV", 0))
if DEV:
    if not BENCH:
        raise SystemExit("TSAR_DEV=1 needs TSAR_BENCH=1")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import devsplit as DV
NORM = _env("TSAR_NORM", "last")
FRAME = int(_env("TSAR_FRAME", 1)); FRAMEFIX = int(_env("TSAR_FRAMEFIX", 0)); SREF_AVG = int(_env("TSAR_SREF_AVG", 1))
SCHED = _env("TSAR_SCHED", "cos"); WEIGHT_DECAY = float(_env("TSAR_WD", WEIGHT_DECAY))
RUN = _env("TSAR_RUN")
BUDGET = float(_env("TSAR_MINUTES", P.TIME_BUDGET / 60)) * 60
EVAL_EVERY = float(_env("TSAR_EVAL_MIN", 0)) * 60
EVAL_TASKS = [t for t in _env("TSAR_WM_EVAL_TASKS", "").split(",") if t]
if EVAL_TASKS and DEV:
    raise SystemExit("TSAR_WM_EVAL_TASKS is a smoke-test subset of VAL; refused with TSAR_DEV")
LONG = _env("TSAR_WM_LONG", "median")                     # re-read: eval-only may have restored the stored value
if LONG not in ("median", "paths"):
    raise SystemExit(f"TSAR_WM_LONG={LONG!r} not in (median, paths)")
TTA = _env("TSAR_WM_TTA", "0")                            # re-read, as LONG
if TTA not in ("0", "1", "sym", "clip"):
    raise SystemExit(f"TSAR_WM_TTA={TTA!r} not in (0, 1, sym, clip)")
TTA_SYM, TTA_CLIP = TTA in ("1", "sym"), TTA in ("1", "clip")
if TTA_SYM and not np.allclose(np.asarray(P.QUANTILES) + np.asarray(P.QUANTILES)[::-1], 1.0):
    raise SystemExit("TSAR_WM_TTA: symmetric averaging needs quantile levels symmetric about 0.5")
RECAL = _env("TSAR_WM_RECAL", "0")                        # read after the eval-only restore, as LONG
if RECAL not in ("0", "lead", "freqlead", "file"):
    raise SystemExit(f"TSAR_WM_RECAL={RECAL!r} not in (0, lead, freqlead, file)")
RECAL_N = int(_env("TSAR_WM_RECAL_N", 150))
RECAL_FILE = _env("TSAR_WM_RECAL_MAP", "")
if (_env("TSAR_WM_RECAL_N") or RECAL_FILE) and RECAL == "0":
    raise SystemExit("TSAR_WM_RECAL_N / TSAR_WM_RECAL_MAP need TSAR_WM_RECAL")
if RECAL == "file" and (not RECAL_FILE or DEV):
    raise SystemExit("TSAR_WM_RECAL=file needs TSAR_WM_RECAL_MAP and is refused with TSAR_DEV")
if RECAL in ("lead", "freqlead") and DEV and not RECAL_FILE:
    raise SystemExit("TSAR_WM_RECAL under TSAR_DEV writes the map: set TSAR_WM_RECAL_MAP")
ORIGINS = int(_env("TSAR_WM_ORIGINS", 0))
PERIOD = int(_env("TSAR_PERIOD", 0))                       # period lever (docs/period_lever.md), default off
PERIOD_BIN = int(_env("TSAR_PERIOD_BIN", 0))
if PERIOD not in (0, 1) or PERIOD_BIN not in (0, 1) or (PERIOD_BIN and not PERIOD):
    raise SystemExit("TSAR_PERIOD and TSAR_PERIOD_BIN must be 0|1, and TSAR_PERIOD_BIN needs TSAR_PERIOD=1")
if PERIOD:
    from wm.ssm import detect_periods, phase_features, PER_LW, PER_K   # noqa: E402
LEADW = float(_env("TSAR_WM_LEADW", 1.0))
LEADW_FROM = 64                                            # steps 1..64 = the original rollout length
if CONTEXT % PATCH or CONTEXT < 2 * PATCH:
    raise SystemExit(f"TSAR_CONTEXT={CONTEXT} must be a multiple of PATCH={PATCH} and >= {2 * PATCH}")
if MAX_H % PATCH or MAX_H < PATCH:
    raise SystemExit(f"TSAR_MAX_H={MAX_H} must be a positive multiple of PATCH={PATCH}")
if LEADW != 1.0 and MAX_H <= LEADW_FROM:
    raise SystemExit(f"TSAR_WM_LEADW weights steps beyond {LEADW_FROM}; meaningless with TSAR_MAX_H={MAX_H}")
if ORIGINS < 0 or LEADW < 0:
    raise SystemExit("TSAR_WM_ORIGINS and TSAR_WM_LEADW must be >= 0")
PATH_LEVELS = [float(v) for v in _env("TSAR_WM_LONG_PATHS", "").split(",") if v.strip()] or list(P.QUANTILES)
_qs = [round(q, 6) for q in P.QUANTILES]
if any(round(v, 6) not in _qs for v in PATH_LEVELS) or len(set(PATH_LEVELS)) != len(PATH_LEVELS):
    raise SystemExit(f"TSAR_WM_LONG_PATHS={PATH_LEVELS} must be distinct levels from {[round(float(q), 2) for q in P.QUANTILES]}")
PATH_LEVELS = sorted(PATH_LEVELS)
PATH_IDX = [_qs.index(round(v, 6)) for v in PATH_LEVELS]
# mixture weight of each path: the share of the latent level u ~ U(0, 1) nearest to it (midpoint-rule quadrature
# of the integral over u of F(y | path at level u)); all 9 levels -> 0.15, 0.1 x 7, 0.15
_bnd = [0.0] + [(a + b) / 2 for a, b in zip(PATH_LEVELS, PATH_LEVELS[1:])] + [1.0]
PATH_W = [b - a for a, b in zip(_bnd, _bnd[1:])]

K_STEPS = MAX_H // PATCH
_maxk = getattr(FamilyCls, "max_k", None)
if _maxk is not None and K_STEPS > _maxk:
    raise SystemExit(f"family {WM_NAME!r} supports at most K={_maxk} lead patches (TSAR_MAX_H <= {_maxk * PATCH}); "
                     f"TSAR_MAX_H={MAX_H} gives K={K_STEPS}. Reason: {getattr(FamilyCls, 'max_k_why', '')}")
WINDOW = CONTEXT + MAX_H
N_PATCH = WINDOW // PATCH
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = _env("TSAR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
Q = torch.tensor(P.QUANTILES, device=DEVICE)
NQ = len(P.QUANTILES)
N_FRAME = 3 if FRAME else 0
D_WM = max(8, 8 * round(D_MODEL * WIDTH / 8))
if D_WM % N_HEADS:
    raise SystemExit(f"train_wm: TSAR_N_HEADS={N_HEADS} must divide d_model={D_WM} (TSAR_WM_WIDTH={WIDTH})")
CFG = WMConfig(patch=PATCH, context=CONTEXT, max_h=MAX_H, k_steps=K_STEPS, n_patch=N_PATCH, nq=NQ,
               tok_dim=2 * PATCH + N_FRAME, d_model=D_WM, n_layers=N_LAYERS, n_heads=N_HEADS, width=WIDTH, abl=ABL)

# ---------------------------------------------------------------------------
# Normalisation, tokens and the seasonal reference (train_arch2, unchanged maths)
# ---------------------------------------------------------------------------


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
    return loc, scale, cnt


def frame_features(loc, scale, cnt):
    """How each token's running frame moved from the previous one, plus how much history it has (causal)."""
    if not FRAME:
        return []
    lp = torch.cat([loc[:, :1], loc[:, :-1]], 1)
    sp = torch.cat([scale[:, :1], scale[:, :-1]], 1)
    dloc = torch.asinh((loc - lp) / scale)
    dls = torch.log(scale / sp).clamp(-5, 5)
    if FRAMEFIX:
        prev = torch.cat([torch.zeros_like(cnt[:, :1]), cnt[:, :-1]], 1) > 0
        dloc, dls = dloc * prev, dls * prev
    lc = torch.log1p(cnt) / 5
    return [dloc[..., None], dls[..., None], lc[..., None]]


def seasonal_ref(x, loc, scale, m):
    """Seasonal-naive reference for every origin and future step, in each origin's model space.
    x: [B, L] raw, m: [B] season (1 = naive). Returns [B, n, MAX_H]. Causal: only values <= origin."""
    B, L = x.shape
    off = L - loc.shape[1] * PATCH
    obs = ~torch.isnan(x)
    ar = torch.arange(L, device=x.device)
    last = torch.where(obs, ar.expand_as(x), torch.full_like(x, -1, dtype=torch.long)).cummax(1).values
    ffill = torch.gather(torch.nan_to_num(x), 1, last.clamp(min=0))
    ends = torch.arange(off + PATCH - 1, L, PATCH, device=x.device)
    k = torch.arange(1, MAX_H + 1, device=x.device)
    mm = m.clamp(min=1)[:, None, None]
    lag = mm * torch.div(k[None, None, :] + mm - 1, mm, rounding_mode="floor")  # m * ceil(k/m)
    idx = ends[None, :, None] + k[None, None, :] - lag
    tot, cnt = torch.zeros_like(idx, dtype=x.dtype), torch.zeros_like(idx, dtype=x.dtype)
    for s in range(SREF_AVG):
        i = idx - s * mm
        o = (i >= 0) & (torch.gather(last, 1, i.clamp(min=0).view(B, -1)).view_as(i) >= 0) & ((mm > 1) | (s == 0))
        tot = tot + torch.where(o, torch.gather(ffill, 1, i.clamp(min=0).view(B, -1)).view_as(i), 0.0)
        cnt = cnt + o
    ok = cnt > 0
    val = tot / cnt.clamp(min=1)
    ref = to_model_space(val, loc[..., None], scale[..., None])
    return torch.where(ok, ref, torch.zeros_like(ref))


def make_context(x, per):
    """x: [B, L] raw (NaN = missing / pad), per: [B] season -> Context (wm/base.py)."""
    B, L = x.shape
    n = L // PATCH
    loc, scale, cnt = causal_stats(x)
    xp = x.view(B, n, PATCH)
    obs = ~torch.isnan(xp)
    z = to_model_space(torch.where(obs, xp, loc[..., None]), loc[..., None], scale[..., None])
    tok = torch.cat([torch.where(obs, z, torch.zeros_like(z)), obs.float()] + frame_features(loc, scale, cnt), -1)
    ref = seasonal_ref(x, loc, scale, per).clamp(-20, 20)
    ctx = Context(tok=tok, x=x, loc=loc, scale=scale, cnt=cnt, ref=ref)
    if PERIOD:
        add_period_features(ctx, x)
    return ctx


def add_period_features(ctx, x):
    """TSAR_PERIOD: detected periods per origin (window = the PER_LW points ending at patch j, NaN-padded on the
    left, so only values <= the origin are read) and the phase coordinates of context and lead patch starts."""
    B, L = x.shape
    n = L // PATCH
    xp = torch.nn.functional.pad(x, (PER_LW - PATCH, 0), value=float("nan"))
    wins = xp.unfold(1, PER_LW, PATCH)                                          # [B, n, PER_LW], window j ends at patch j
    per = detect_periods(wins.reshape(B * n, PER_LW)).view(B, n, PER_K)
    s_tok = PATCH * torch.arange(n, device=x.device)
    s_fut = PATCH * (torch.arange(n, device=x.device)[:, None] + 1 + torch.arange(K_STEPS, device=x.device)[None])
    ctx.periods = per
    ctx.tok_phase = phase_features(per, s_tok[None].expand(B, n))
    ctx.fut_phase = phase_features(per[:, :, None].expand(B, n, K_STEPS, PER_K), s_fut[None].expand(B, n, K_STEPS))

# ---------------------------------------------------------------------------
# Model = family + the shared seasonal-ref skip
# ---------------------------------------------------------------------------


class WMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.family = FamilyCls(CFG)
        self.ref_skip = nn.Parameter(torch.zeros(NQ))          # learned copy weight per quantile (train_arch2)

    def forward(self, ctx, b, j):
        """Quantiles [N, K * PATCH, NQ] in the model space of each origin, and the family's aux losses."""
        q, aux = self.family.decode(self.family.encode(ctx), ctx, b, j)
        N = len(b)
        if q.shape != (N, K_STEPS, PATCH, NQ):
            raise RuntimeError(f"family {WM_NAME}: decode returned {tuple(q.shape)}, expected {(N, K_STEPS, PATCH, NQ)}")
        r = ctx.ref[b, j].view(N, K_STEPS, PATCH)
        return (q + r[..., None] * self.ref_skip).reshape(N, K_STEPS * PATCH, NQ), aux


def check_aux(aux):
    bad = set(aux) - set(FamilyCls.aux_weights)
    if bad:
        raise RuntimeError(f"family {WM_NAME}: aux losses {sorted(bad)} have no weight in aux_weights")


def causality_check(m):
    """Forecasts at origins j < c must not change when the series after patch c is altered (training uses every
    origin of a window at once, so a non-causal encode() would train on its own targets). CPU copy, eval mode,
    forked RNG (the training RNG stream is untouched), same torch seed for both passes."""
    m = copy.deepcopy(m).cpu().eval()
    npst = np.random.get_state()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(2, WINDOW, generator=g).cumsum(1) + 5
        x[1, :40] = float("nan")                                   # left padding, as short series have
        c = N_PATCH // 2
        x2 = x.clone()
        x2[:, c * PATCH:] += 10 * torch.randn(2, WINDOW - c * PATCH, generator=g)
        per = torch.tensor([24, 1])
        b = torch.arange(2).repeat_interleave(c)
        j = torch.arange(c).repeat(2)
        outs = []
        for xx in (x, x2):
            torch.manual_seed(0)
            q, _ = m(make_context(xx, per), b, j)
            outs.append(q)
    np.random.set_state(npst)
    if not torch.isfinite(outs[0]).all():
        raise SystemExit(f"wm_causal: family {WM_NAME} produced non-finite forecasts on the check series")
    diff = (outs[0] - outs[1]).abs().max().item()
    if diff > 1e-4:
        raise SystemExit(f"wm_causal: FAIL, forecasts at origins before patch {c} changed by {diff:.3g} when later "
                         f"values changed: family {WM_NAME} encode/decode is not causal")
    print(f"wm_causal: ok (max change {diff:.1e} at {len(b)} early origins)", flush=True)


def macs_per_forecast(m):
    """Approximate multiply-accumulates for one forecast (one series, CONTEXT points, one origin, MAX_H steps):
    matmul-type ops counted by torch's FlopCounterMode / 2, on a CPU copy in train mode (so the Transformer fast
    path, which the counter does not see, is not taken) under no_grad. Elementwise ops are not counted."""
    from torch.utils.flop_counter import FlopCounterMode
    m = copy.deepcopy(m).cpu().train()
    npst = np.random.get_state()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        x = torch.randn(1, CONTEXT).cumsum(1)
        ctx = make_context(x, torch.tensor([24]))
        n = CONTEXT // PATCH
        with FlopCounterMode(display=False) as fc:
            m(ctx, torch.zeros(1, dtype=torch.long), torch.full((1,), n - 1))
    np.random.set_state(npst)
    return fc.get_total_flops() // 2


model = WMModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
groups = " ".join(f"{k}={sum(p.numel() for p in v.parameters())}" for k, v in model.family.param_groups().items())
MACS = macs_per_forecast(model)
print(f"wm: family={WM_NAME} abl={ABL} width={WIDTH} d_model={D_WM} n_layers={N_LAYERS} params={n_params} "
      f"trainable={n_train} {groups} macs_per_forecast={MACS} aux_weights={FamilyCls.aux_weights}", flush=True)
causality_check(model)
if INFO:
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Data (bench TRAIN; DEV families held out under TSAR_DEV)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench as BN                                         # noqa: E402
train = BN.load_train(extra=bool(int(_env("TSAR_BENCH_EXTRA", 0))), max_len=int(_env("TSAR_BENCH_MAXLEN", 20000)))
by_dataset = {}
for s in train:
    by_dataset.setdefault(s["dataset"], []).append(s["target"])
CAL_SRC = None
if RECAL in ("lead", "freqlead"):                          # TSAR_WM_RECAL calibration series (TRAIN only)
    if DEV:                                                # the DEV families this run never trains on
        CAL_SRC = {d: list(by_dataset[d]) for d in DV.DEV_DATASETS if d in by_dataset}
if DEV:
    DV.holdout(by_dataset)
if RECAL in ("lead", "freqlead") and not DEV:              # the training pool itself (in-sample targets)
    CAL_SRC = {d: list(v) for d, v in by_dataset.items()}
if RECAL != "0":
    import wmrecal as RC                                   # noqa: E402  (tools/, on sys.path above)
    FREQ = {s["dataset"]: s["freq"] for s in train}
datasets = list(by_dataset)
SEASON = {s["dataset"]: s["season"] for s in train}
del train
EVALUATE = DV.evaluate if DEV else BN.evaluate


def _git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


_RUN_CFG = None


def run_cfg():
    """Data + model config stored in every checkpoint and verified on resume and in eval-only mode."""
    global _RUN_CFG
    if _RUN_CFG is None:
        h = hashlib.sha256()
        for d in datasets:
            for y in by_dataset[d]:
                a = np.ascontiguousarray(y, dtype=np.float32)
                h.update(f"{d}:{a.size};".encode())
                h.update(a.tobytes())
        drop = hashlib.sha256(json.dumps({k: sorted(v) for k, v in BN.TRAIN_DROP.items()},
                                         sort_keys=True).encode()).hexdigest()
        _RUN_CFG = {"bench": BENCH, "bench_extra": int(_env("TSAR_BENCH_EXTRA", 0)),
                    "bench_maxlen": int(_env("TSAR_BENCH_MAXLEN", 20000)), "family": WM_NAME,
                    "context": CONTEXT, "max_h": MAX_H, "patch": PATCH, "d_model": D_WM, "n_layers": N_LAYERS,
                    "n_heads": N_HEADS, "train_datasets": sorted(datasets),
                    "train_series": sum(len(v) for v in by_dataset.values()),
                    "train_sha256": h.hexdigest(), "train_drop_sha256": drop, "env": _tsar_env(), "git": _git_head()}
    return _RUN_CFG


def cfg_mismatch(stored, ignore):
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


def sample_batch(rng):
    """Right-aligned windows of WINDOW points, balanced across datasets. NaN = missing/pad (train_arch2)."""
    win = np.full((BATCH, WINDOW), np.nan, dtype=np.float32)
    per = np.ones(BATCH, dtype=np.int64)
    frac = SYN_FRAC if not SYN_CURR else SYN_FRAC + (SYN_CURR - SYN_FRAC) * max(0.0, 1 - PROG / 0.6)
    n_syn = int(BATCH * frac)
    o = int(round(n_syn * SYN5_SHARE))
    if o:
        win[:o], per[:o] = S5.generate(rng, o, WINDOW, min_len=2 * PATCH)
    n3 = int(round((n_syn - o) * SYN3_SHARE))
    n2 = int((n_syn - o - n3) * SYN2_SHARE)
    if n3:
        win[o:o + n3], per[o:o + n3] = S3.generate(rng, n3, WINDOW, min_len=2 * PATCH)
    win[o + n3:o + n3 + n2], per[o + n3:o + n3 + n2] = synthetic2(rng, n2, WINDOW)
    win[o + n3 + n2:n_syn], per[o + n3 + n2:n_syn] = synthetic(rng, n_syn - o - n3 - n2, WINDOW)
    for b in range(n_syn, BATCH):
        d = datasets[rng.integers(len(datasets))]
        series = by_dataset[d]
        per[b] = SEASON[d]
        y = series[rng.integers(len(series))]
        end = rng.integers(min(2 * PATCH, len(y)), len(y) + 1)
        w = y[max(0, end - WINDOW):end]
        win[b, WINDOW - len(w):] = w
    if AUG_P > 0:
        rows = np.arange(n_syn, BATCH)
        rows = rows[rng.random(len(rows)) < AUG_P]
        if len(rows):
            win[rows], per[rows] = S3.augment(win[rows], rng, period=per[rows], min_obs=4 * PATCH)
    return torch.from_numpy(win).to(DEVICE), torch.from_numpy(per).to(DEVICE)


def pinball(pred, target, mask):
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(Q * err, (Q - 1) * err) * mask.unsqueeze(-1)
    return loss.sum() / mask.sum().clamp(min=1) / NQ


def train_loss(model, win, per):
    """Multi-origin loss: every patch with at least one observed future value is an origin."""
    B = win.shape[0]
    ctx = make_context(win, per)
    n = ctx.tok.shape[1]
    pad = torch.full((B, MAX_H), float("nan"), device=win.device)
    fut = torch.cat([win, pad], 1).unfold(1, MAX_H, PATCH)[:, 1:n + 1]          # [B, n, MAX_H]
    valid = (ctx.cnt > 0)[..., None] & ~torch.isnan(fut)
    tgt = to_model_space(torch.nan_to_num(fut), ctx.loc[..., None], ctx.scale[..., None])
    keep = valid.any(-1).view(-1)
    bj = torch.nonzero(keep).squeeze(1)
    if ORIGINS and len(bj) > ORIGINS:                                           # TSAR_WM_ORIGINS: random subset
        bj = bj[torch.from_numpy(np.sort(rng.choice(len(bj), ORIGINS, replace=False))).to(bj.device)]
        keep = bj
    pred, aux = model(ctx, bj // n, bj % n)
    check_aux(aux)
    mask = valid.reshape(B * n, MAX_H)[keep].float()
    if LEADW != 1.0:                                                            # TSAR_WM_LEADW: steps > 64
        mask = mask * torch.where(torch.arange(MAX_H, device=mask.device) < LEADW_FROM, 1.0, LEADW)
    loss = pinball(pred, tgt.reshape(B * n, MAX_H)[keep], mask)
    total = loss + sum(FamilyCls.aux_weights[k] * v for k, v in aux.items())
    return total, loss, aux

# ---------------------------------------------------------------------------
# Training (fixed step budget)
# ---------------------------------------------------------------------------


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
rng = np.random.default_rng(SEED)
ema = copy.deepcopy(model).requires_grad_(False) if EMA else model


def mixture_quantiles(comp, w, levels=None, rows=4096):
    """Quantiles of a finite mixture of quantile-specified distributions (TSAR_WM_LONG=paths).

    comp: [n, C, NQ, T] component quantiles at prepare.QUANTILES (any order within a component; sorted here);
    w: C mixture weights (sum 1). Returns [n, NQ, T] mixture quantiles at the same levels (sorted).

    Each component's CDF is the piecewise-linear interpolation of its NQ knots (tau_i, q_i), i.e. uniform density
    between neighbouring quantiles, with linear tails that carry the remaining mass tau_0 below q_0 and 1 - tau_8
    above q_8 at the slope of the outermost segment (support [q_0 - (q_1-q_0) tau_0/(tau_1-tau_0), q_8 + ...]).
    The mixture density sum_c w_c f_c is then piecewise constant with breakpoints at every segment end, so the
    mixture CDF is exactly piecewise linear: sort the breakpoints, cumsum the density steps (+m/width at a segment
    start, -m/width at its end), integrate, and invert by linear interpolation. Exact (no sampling, no bisection),
    O(S log S) per step for S = C * (NQ + 1) segments. A component with a repeated value is a point mass (segment
    width floored at 1e-9 of the knot range). With C = 1 it returns the component's own (sorted) quantiles. Unlike
    the Chronos-Bolt shortcut (empirical quantiles of the C x NQ values pooled as if they were samples), it respects
    the mixture weights and the mass between and beyond the knots. float64 on CPU, in row chunks."""
    tau = torch.tensor(P.QUANTILES if levels is None else levels, dtype=torch.float64)
    nq = len(tau)
    mass = torch.cat([tau[:1], tau[1:] - tau[:-1], 1 - tau[-1:]])                  # [NQ + 1] per segment
    wt = torch.tensor(w, dtype=torch.float64)
    out = []
    for s in range(0, comp.shape[0], rows):
        q = torch.from_numpy(np.ascontiguousarray(comp[s:s + rows])).double().sort(2).values   # [r, C, NQ, T]
        q = q.permute(0, 3, 1, 2)                                                   # [r, T, C, NQ]
        lo_t = q[..., :1] - (q[..., 1:2] - q[..., :1]) * tau[0] / (tau[1] - tau[0])
        hi_t = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - tau[-1]) / (tau[-1] - tau[-2])
        kn = torch.cat([lo_t, q, hi_t], -1)                                         # [r, T, C, NQ + 2]
        eps = 1e-9 * (kn.amax((-1, -2)) - kn.amin((-1, -2)))[..., None, None] + 1e-300
        a = kn[..., :-1]
        wd = (kn[..., 1:] - a).clamp(min=0) + eps                                   # segment widths > 0
        dens = (wt[:, None] * mass[None, :]) / wd                                   # [r, T, C, NQ + 1]
        y = torch.cat([a.flatten(-2), (a + wd).flatten(-2)], -1)                    # breakpoints [r, T, 2S]
        dd = torch.cat([dens.flatten(-2), -dens.flatten(-2)], -1)                   # density steps
        y, order = y.sort(-1)
        dn = dd.gather(-1, order).cumsum(-1)[..., :-1].clamp(min=0)                 # density on [y_i, y_i+1]
        F = torch.cat([torch.zeros_like(y[..., :1]), (dn * y.diff(dim=-1)).cumsum(-1)], -1)
        F = (F / F[..., -1:].clamp(min=1e-300)).clamp(0, 1).cummax(-1).values       # exact up to rounding
        i = (F[..., None, :] < tau[:, None]).sum(-1).clamp(1, y.shape[-1] - 1)      # [r, T, NQ] first F >= tau
        y0, y1 = y.gather(-1, i - 1), y.gather(-1, i)
        F0, F1 = F.gather(-1, i - 1), F.gather(-1, i)
        r = ((tau - F0) / (F1 - F0).clamp(min=1e-300)).clamp(0, 1)
        out.append((y0 + r * (y1 - y0)).permute(0, 2, 1).float().numpy())            # [r, NQ, T]
    return np.concatenate(out) if out else np.zeros((0, nq, comp.shape[-1]), np.float32)


@torch.no_grad()
def predict_paths(ctxs, h, m_season, freq):
    """TSAR_WM_LONG=paths (h > MAX_H): chunk 1 is one rollout; later chunks roll |S| quantile paths and mix them."""
    ctxs = [np.asarray(c, dtype=np.float32) for c in ctxs]
    n, S = len(ctxs), len(PATH_IDX)
    t0 = time.time()
    q = np.sort(predict_chunk(ctxs, min(MAX_H, h), m_season, freq), axis=1)      # [n, NQ, MAX_H]
    out, paths = [q], [list(ctxs) for _ in PATH_IDX]
    for s in range(MAX_H, h, MAX_H):
        hc = min(MAX_H, h - s)
        for p, qi in enumerate(PATH_IDX):                                          # path p appends its level
            paths[p] = [np.concatenate([c, q[i, qi]])[-CONTEXT:] for i, c in enumerate(paths[p])]
        comp = predict_chunk([c for pp in paths for c in pp], hc, m_season, freq)  # [S * n, NQ, hc]
        t1 = time.time()
        q = mixture_quantiles(comp.reshape(S, n, NQ, hc).transpose(1, 0, 2, 3), PATH_W)
        PATH_COST["mixture_s"] += time.time() - t1
        PATH_COST["rollouts"] += S * n
        out.append(q)
    PATH_COST["rollouts"] += n
    PATH_COST["seconds"] += time.time() - t0
    return np.concatenate(out, -1)


PATH_COST = {"seconds": 0.0, "mixture_s": 0.0, "rollouts": 0}   # cost report of TSAR_WM_LONG=paths


def nonnegative_context(ctxs):
    """timestral/forecaster.py nonnegative_context: every observed (finite) value >= 0, and at least one observed."""
    out = np.zeros(len(ctxs), dtype=bool)
    for i, c in enumerate(ctxs):
        v = np.asarray(c, dtype=np.float32)
        v = v[np.isfinite(v)]
        out[i] = v.size > 0 and bool((v >= 0).all())
    return out


RECAL_MAP = None                                           # set by fit_recal() before scoring (TSAR_WM_RECAL)


@torch.no_grad()
def predict(ctxs, h, m_season, freq):
    """bench predict_fn. TSAR_WM_TTA wraps the whole forecast (Timestral's clampsym); off = predict_one as before."""
    if not (TTA_SYM or TTA_CLIP or RECAL_MAP is not None):
        return predict_one(ctxs, h, m_season, freq)
    q = predict_one(ctxs, h, m_season, freq)                                    # [n, NQ, h]
    if TTA_SYM:                                                                 # forecaster.py: symmetric averaging
        qm = predict_one([-np.asarray(c, dtype=np.float32) for c in ctxs], h, m_season, freq)
        q = 0.5 * (q + (-qm)[:, ::-1])
    if TTA_CLIP:                                                                # forecaster.py: clip_nonnegative
        elig = nonnegative_context(ctxs)
        q = np.where(elig[:, None, None], np.maximum(q, 0), q)
    if RECAL_MAP is not None:                                                   # TSAR_WM_RECAL (fitted on TRAIN)
        q = RC.apply(q, freq, RECAL_MAP, RECAL, nonnegative_context(ctxs) if TTA_CLIP else None)
    return np.ascontiguousarray(q, dtype=np.float32)


@torch.no_grad()
def predict_one(ctxs, h, m_season, freq):
    if h > MAX_H and LONG == "paths":
        return predict_paths(ctxs, h, m_season, freq)
    if h > MAX_H:                          # long horizons: roll forward in MAX_H chunks, feeding back the median
        out, ctxs = [], [np.asarray(c, dtype=np.float32) for c in ctxs]
        for s in range(0, h, MAX_H):
            q = predict_chunk(ctxs, min(MAX_H, h - s), m_season, freq)
            out.append(q)
            ctxs = [np.concatenate([c, q[i, NQ // 2]])[-CONTEXT:] for i, c in enumerate(ctxs)]
        return np.concatenate(out, -1)
    return predict_chunk(ctxs, h, m_season, freq)


@torch.no_grad()
def predict_chunk(ctxs, h, m_season, freq):
    ema.eval()
    out = []
    EB = 512 if CONTEXT <= 512 else max(64, 512 * 512 // CONTEXT)
    for s in range(0, len(ctxs), EB):
        chunk = ctxs[s:s + EB]
        ctx = np.full((len(chunk), CONTEXT), np.nan, dtype=np.float32)
        for i, c in enumerate(chunk):
            c = c[-CONTEXT:]
            ctx[i, CONTEXT - len(c):] = c
        out.append(forecast(torch.from_numpy(ctx).to(DEVICE), h, m_season).permute(0, 2, 1).float().cpu().numpy())
    return np.concatenate(out)


def forecast(x, h, m_season):
    """Quantile forecast [B, h, NQ] in raw space from the last context patch."""
    B = x.shape[0]
    ctx = make_context(x, torch.full((B,), int(m_season), device=x.device))
    n = ctx.tok.shape[1]
    q, _ = ema(ctx, torch.arange(B, device=x.device), torch.full((B,), n - 1, device=x.device))
    return from_model_space(q[:, :h], ctx.loc[:, -1, None, None], ctx.scale[:, -1, None, None])


CKPT = os.path.join(P.CACHE_DIR, "ckpt", f"{RUN}.pt") if RUN and not EVAL_ONLY else None
step, smooth, done = 0, None, 0.0
if EVAL_ONLY:
    bad = cfg_mismatch(CK0["cfg"], EVAL_FREE_KEYS | INFER_KEYS)
    if bad:
        raise SystemExit("eval-only: the checkpoint does not match this data/model config:\n  " + "\n  ".join(bad))
    model.load_state_dict(CK0["model"])
    if EMA:
        ema.load_state_dict(CK0.get("ema", CK0["model"]))
    step, done = CK0["step"], CK0["done"]
    print(f"eval-only: frozen weights at step {step}, {done / 60:.1f} min trained, complete={CK0.get('complete')}",
          flush=True)
    del CK0
elif CKPT and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    bad = cfg_mismatch(ck.get("cfg", {}), CTRL_KEYS) if "cfg" in ck else ["no stored config"]
    if bad:
        raise SystemExit(f"refusing to resume {RUN}: the checkpoint was trained with another config:\n  " + "\n  ".join(bad))
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if EMA:
        ema.load_state_dict(ck.get("ema", ck["model"]))
    step, done, rng = ck["step"], ck["done"], ck["rng"]
    print(f"resumed {RUN} at step {step}, {done / 60:.1f} min trained", flush=True)


def save_ckpt(elapsed, final=False):
    if CKPT:
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "done": elapsed, "rng": rng, "cfg": run_cfg(), "complete": final,
                    "budget": {"steps": STEPS, "seconds": BUDGET}}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)


def _eval(verbose=True):
    if EVAL_TASKS:
        return BN.evaluate(predict, split="val", verbose=verbose, names=set(EVAL_TASKS))
    return EVALUATE(predict, split="val", verbose=verbose)


t_start = time.time() - done
last_ckpt = last_eval = time.time()
aux_smooth = {}
while not EVAL_ONLY:
    elapsed = time.time() - t_start
    prog = step / STEPS if STEPS else elapsed / BUDGET
    PROG = prog
    if prog >= 1:
        break
    f = max(0.0, (prog - 0.8) / 0.2) if SCHED == "wsd" else prog
    lr = LR * min(1.0, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))
    for g in opt.param_groups:
        g["lr"] = lr
    loss, pin, aux = train_loss(model, *sample_batch(rng))
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
    for k, v in aux.items():
        aux_smooth[k] = v.item() if k not in aux_smooth else 0.98 * aux_smooth[k] + 0.02 * v.item()
    if step % 100 == 0:
        extra = "".join(f"  {k} {v:.4f}" for k, v in aux_smooth.items())
        print(f"step {step:5d}  loss {smooth:.4f}  lr {lr:.2e}  {elapsed:5.0f}s{extra}", flush=True)
    step += 1
    if CKPT and time.time() - last_ckpt > 600:
        save_ckpt(time.time() - t_start); last_ckpt = time.time()
    if EVAL_EVERY and time.time() - last_eval > EVAL_EVERY:
        t0 = time.time()
        r = _eval(verbose=False)
        model.train()
        print(f"EVAL step {step} minutes {(time.time() - t_start) / 60:.1f} wql_rel {r['wql_rel']:.4f} "
              f"mase_rel {r['mase_rel']:.4f} cov80_err {r['cov80_error']:.4f}", flush=True)
        t_start += time.time() - t0
        last_eval = time.time()
training_seconds = time.time() - t_start
if not EVAL_ONLY:
    save_ckpt(training_seconds, final=True)

# ---------------------------------------------------------------------------
# Evaluation (VAL, or DEV under TSAR_DEV; never TEST)
# ---------------------------------------------------------------------------

def fit_recal():
    """TSAR_WM_RECAL: fit the map on TRAIN calibration windows (CAL_SRC only; VAL/TEST are not read here)."""
    t0 = time.time()
    wins = RC.windows(CAL_SRC, FREQ, RECAL_N)
    preds = {d: predict(c, H, SEASON[d], FREQ[d]) for d, (c, _, H) in wins.items()}
    cmap, rep = RC.fit(preds, wins, FREQ, P.QUANTILES, RECAL, nonnegative_context if TTA_CLIP else None)
    for line in rep:
        print(line, flush=True)
    src = "DEV families held out of training (out-of-family)" if DEV else "the training pool (in-sample targets)"
    print(f"recal: mode={RECAL} {sum(len(w[0]) for w in wins.values())} TRAIN windows from {len(wins)} datasets "
          f"[{', '.join(sorted(wins))}] = {src}, {time.time() - t0:.1f}s", flush=True)
    PATH_COST.update(seconds=0.0, mixture_s=0.0, rollouts=0)    # the cost report covers VAL scoring only
    return cmap


if RECAL in ("lead", "freqlead"):
    _cmap = fit_recal()
    if DEV:                                                # write it for TSAR_WM_RECAL=file; DEV score stays raw
        RC.save(RECAL_FILE, _cmap, RECAL, {"run": RUN, "ckpt_sha256": CK_SHA, "git": _git_head(),
                                           "datasets": sorted(CAL_SRC), "n_per_dataset": RECAL_N, "env": _tsar_env()})
        print(f"recal: map written to {RECAL_FILE} (not applied to the DEV score)", flush=True)
    else:
        RECAL_MAP = _cmap
elif RECAL == "file":
    RECAL_MAP, _meta = RC.load(RECAL_FILE)
    RECAL = _meta["mode"]
    print(f"recal: map {RECAL_FILE} mode={RECAL} fitted on {_meta['datasets']} (run {_meta.get('run')})", flush=True)
split = "dev" if DEV else ("smoke_val" if EVAL_TASKS else "val")
res = _eval()
if EVAL_ONLY:
    print(f"eval_only_ckpt_sha256: {CK_SHA}")
peak_mb = torch.mps.driver_allocated_memory() / 2**20 if DEVICE == "mps" else 0.0
print("---")
print(f"wm_family:        {WM_NAME} abl={ABL} width={WIDTH} d_model={D_WM}")
print(f"{split}_wql_rel:      {res['wql_rel']:.6f}")
print(f"{split}_mase_rel:     {res['mase_rel']:.6f}")
print(f"{split}_cov80_error:  {res['cov80_error']:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"predict_seconds:  {res['predict_seconds']:.1f}")
print(f"total_seconds:    {time.time() - t_start:.1f}")
print(f"peak_mem_mb:      {peak_mb:.1f}")
print(f"num_params_M:     {n_params / 1e6:.3f}")
print(f"macs_per_forecast: {MACS}")
if TTA != "0":
    print(f"tta:              {TTA} (sym={int(TTA_SYM)} clip={int(TTA_CLIP)})")
if LONG == "paths":
    print(f"long_paths_cost:  {PATH_COST['seconds']:.1f}s in h>{MAX_H} tasks ({PATH_COST['mixture_s']:.1f}s mixture), "
          f"{PATH_COST['rollouts']} rollouts, {len(PATH_IDX)} paths")
print(f"num_steps:        {step}")
print(f"{'smoke_' if EVAL_TASKS else ''}family_wql_rel:   {res['wql_rel_family']:.6f}")
if DEV:
    print(f"dev_subday_wql_rel: {res['wql_rel_subday']:.6f}")
    print(f"dev_family_wql_rel: {res['wql_rel_family']:.6f}")
