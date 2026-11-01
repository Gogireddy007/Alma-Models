"""
Latent world-model forecaster. THIS is the file the research agent edits.

Current design: multi-origin causal latent world model (tracks T1 + T8).
  encode   - a causal Transformer reads the series patch by patch; its output after patch t
             is the latent world state z_t ("what the system looks like now")
  dynamics - a GRU cell rolls z_t forward K patches with no new observations
  decode   - each imagined state is decoded into 9 quantiles for its patch
Training scores rollouts from every origin in a window, so one forward pass trains many
horizons. Values are standardized causally (stats up to the origin only) and passed
through asinh, as in Timestral. Usage: uv run train.py

train_arch2.py (Mehta, 2026-10-30): train_syn5.py plus composable structural axes for tools/search.py.
Every axis defaults to off, and all defaults reproduce train_syn5.py bit for bit (no extra module is
built, no extra random number is drawn). The forecast at lead h is always decoded from the rolled latent
state at step k = h // PATCH (plus the existing seasonal-ref skip); every axis only changes how that state
is formed, rolled or read. Axes (env var: legal values, default first):

  TSAR_A2_ZDIM        int > 0, divisible by every head count in use   (default TSAR_D_MODEL)
                      latent state width; the encoder stays TSAR_D_MODEL wide and to_state projects.
  TSAR_A2_TRANS       gru | res                                         (default gru)
                      transition core: GRUCell, or a pure pre-LN residual MLP (z <- z + MLP(LN[z, u])).
  TSAR_A2_DEPTH       0 | 1 | 2 | 3                                     (default 0)
                      residual LN-MLP blocks stacked after the core (zero-initialised: starts as the core).
                      TRANS=res needs DEPTH >= 1 (it is the number of residual blocks).
  TSAR_A2_SLOTS       1 | 2 | 4 | 8                                     (default 1)
                      M latent slots per origin (z + learned slot offsets), each rolled by the shared
                      transition after a slot-mixing self-attention (TSAR_N_HEADS heads); the state
                      decoded at step k is mean(slots) + Linear(concat slots) (zero-initialised).
  TSAR_A2_HZ          0 | 1                                             (default 0)
                      learned per-lead-step horizon embedding added to the decoder input.
  TSAR_A2_DRIVE       0 | 1                                             (default 0)
                      the seasonal-naive reference patch r_k (past values only) drives the transition
                      input at step k, so the rolled state carries seasonal phase (an "action" signal).
  TSAR_A2_DREAD       0 | 1                                             (default 0)
                      decoder memory read: the rolled states cross-attend to the context tokens up to
                      the origin (encoder output + raw patch in its own causal frame, per-head learned
                      lag bias), so the state need not compress the whole context. Built per series:
                      memory cost is about one extra rolled-state tensor, not [origins, tokens, ZD].
  TSAR_A2_DREAD_HEADS 1 | 2 | 4 | 8                                     (default 4)
  TSAR_A2_MRES        8 | 8,16 | 8,32 | 8,16,32 | 8,16,32,64            (default 8)
                      multi-resolution patch embedding: each token also embeds the last s raw values
                      ending at its patch end (causal, in the token's own frame); fused by summation.

  TSAR_DEV            0 | 1   (default 0) screening split, mirrored from train_search.py: the DEV TRAIN families
                      (tools/devsplit.py) are held out of training and scored zero-shot instead of VAL; prints
                      dev_* lines and never reads VAL or TEST. Needs TSAR_BENCH=1; refused with TSAR_SPLIT=test.

World-model hypothesis tests (docs/wm_hypothesis_tests.md, Haddad; Phase 6). Not search axes:
  TSAR_A2_DIRECT      0 | 1   (default 0) W1 matched control: no state is carried between leads. Lead k is read from
                      z_k = z_0 + MLP(LN[z_0, u_k]), MLP 2*ZD -> 2*ZD -> ZD (zero-init last layer), u_k = step_emb[k]
                      (+ drive(r_k) with A2_DRIVE), replacing the GRU rollout (self.dynamics is not built).
                      6 ZD^2 + 7 ZD params vs the GRUCell's 6 ZD^2 + 6 ZD; 6 ZD^2 MACs per lead, as the GRU.
                      Decoder, seasonal skip, drive, DREAD, HZ, MRES unchanged. With SLOTS > 1 the MLP runs per
                      slot after slot mixing of the origin slots (never across leads).
  TSAR_A2_STATE_OFF   none | copy | freeze | zero   (default none)
                      copy   W2: z_k = z_0 + step_emb[k] (+ drive(r_k)); no transition is built (unmatched params)
                      freeze W3: the transition stays at its random init (requires_grad False; trainable count printed)
                      zero   W4: imagined states zeroed before CTX_READ/DREAD and the decoder (only ref paths remain).
                             As in train_v2, "zero" is an inference key: it may be set in eval-only on a checkpoint
                             trained without it (VAL only, with a warning); copy/freeze stay refused on a mismatch.
  Overlaps (lesson L5): STATE_OFF=copy and zero are train_v2's TSAR_STATE_OFF=copy/zero, except that copy here keeps
  the drive input; there is no DIRECT in any other script.

Refused at startup (SystemExit), never silently ignored:
  - any TSAR_* key this script does not read (typos);
  - any axis value outside its legal set; head counts that do not divide TSAR_A2_ZDIM;
  - TSAR_LOOPS with TRANS=res, DEPTH > 0 or SLOTS > 1 (LoopWM is its own transition);
  - TSAR_A2_DREAD with TSAR_PREFIX, TSAR_TTA_STEPS or TSAR_SELFCAL (those paths roll without memory);
  - TSAR_A2_MRES other than 8 with TSAR_PREFIX;
  - TSAR_A2_DRIVE with TSAR_SEAS_REF=0 (the drive would be identically zero);
  - TSAR_A2_ZDIM != TSAR_D_MODEL with TSAR_CTX_READ or TSAR_TTA_SERIES (both are D_MODEL wide).
  - TSAR_A2_DIRECT with TSAR_LOOPS, TSAR_LATENT > 0, TSAR_A2_TRANS=res, TSAR_A2_DEPTH > 0, TSAR_PREFIX, or any
    TSAR_A2_STATE_OFF other than none/zero;
  - TSAR_TTA_STEPS with TSAR_A2_DIRECT or TSAR_A2_STATE_OFF=copy (no transition to adapt);
  - TSAR_A2_STATE_OFF=copy with TSAR_LOOPS, TSAR_A2_TRANS=res, TSAR_A2_DEPTH > 0 or TSAR_A2_SLOTS > 1.
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
if _env("TSAR_A2_STATE_OFF") == "zero":  # W4 zeroes the imagined states (inference-only ablation, as train_v2's
    INFER_KEYS |= {"TSAR_A2_STATE_OFF"}  # A3); copy/freeze change the trained weights and stay refused on a mismatch
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
SYN5_SHARE = float(_env("TSAR_SYN5", 0))                   # share of synthetic rows from tools/synth5.generate
if SYN5_SHARE > 0:                                         # (world prior; taken first, synth3 etc. share the rest)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import synth5 as S5
DEV = int(_env("TSAR_DEV", 0))                 # train_search: 1 = DEV screening split (tools/devsplit.py)
if DEV:
    if IS_TEST or not int(_env("TSAR_BENCH", 0)):
        raise SystemExit("TSAR_DEV=1 needs TSAR_BENCH=1 and scores DEV only; it cannot be combined with TSAR_SPLIT=test")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
    import devsplit as DV
SEAS_REF = int(_env("TSAR_SEAS_REF", 1))                   # T7 seasonal memory pathway in the decoder
NORM = _env("TSAR_NORM", "last")                          # "mean" | "last" (see causal_stats)
FRAME = int(_env("TSAR_FRAME", 1)); FRAMEFIX = int(_env("TSAR_FRAMEFIX", 0)); SREF_AVG = int(_env("TSAR_SREF_AVG", 1)); CTX_READ = int(_env("TSAR_CTX_READ", 0))
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

# --- train_arch2 structural axes (see the module docstring) ---------------------------------------------
_TOOL_KEYS = {"TSAR_BENCH_GEP", "TSAR_GIFT_CLEAN", "TSAR_BENCH_EXTRA", "TSAR_BOLT_BATCH", "TSAR_CKPT_SHA256", "TSAR_EVAL_ONLY", "TSAR_SPLIT"}
_KNOWN = set(re.findall(r"TSAR_[A-Z0-9_]*[A-Z0-9]", open(os.path.abspath(__file__)).read())) | _TOOL_KEYS
_unknown = sorted(set(_tsar_env()) - _KNOWN)
if _unknown:
    raise SystemExit(f"train_arch2: unknown TSAR_* keys {_unknown} (typo?); refusing to run")


def _axis(key, default, legal, cast=int):
    v = cast(_env(key, default))
    if legal is not None and v not in legal:
        raise SystemExit(f"train_arch2: {key}={v!r} not in {sorted(legal)}")
    return v


A2_ZDIM = _axis("TSAR_A2_ZDIM", D_MODEL, None)
A2_TRANS = _axis("TSAR_A2_TRANS", "gru", {"gru", "res"}, str)
A2_DEPTH = _axis("TSAR_A2_DEPTH", 0, {0, 1, 2, 3})
A2_SLOTS = _axis("TSAR_A2_SLOTS", 1, {1, 2, 4, 8})
A2_HZ = _axis("TSAR_A2_HZ", 0, {0, 1})
A2_DRIVE = _axis("TSAR_A2_DRIVE", 0, {0, 1})
A2_DREAD = _axis("TSAR_A2_DREAD", 0, {0, 1})
A2_DREAD_HEADS = _axis("TSAR_A2_DREAD_HEADS", 4, {1, 2, 4, 8})
A2_MRES = _axis("TSAR_A2_MRES", "8", {"8", "8,16", "8,32", "8,16,32", "8,16,32,64"}, str)
MRES_EXTRA = [int(s) for s in A2_MRES.split(",")[1:]]          # extra scales beyond the base patch
A2_CUSTOM_TRANS = A2_TRANS != "gru" or A2_DEPTH > 0 or A2_SLOTS > 1
if A2_ZDIM <= 0:
    raise SystemExit(f"train_arch2: TSAR_A2_ZDIM={A2_ZDIM} must be positive")
if A2_TRANS == "res" and A2_DEPTH == 0:
    raise SystemExit("train_arch2: TSAR_A2_TRANS=res needs TSAR_A2_DEPTH >= 1 (the number of residual blocks)")
if LOOPS and A2_CUSTOM_TRANS:
    raise SystemExit("train_arch2: TSAR_LOOPS is its own transition; do not combine it with "
                     "TSAR_A2_TRANS=res, TSAR_A2_DEPTH>0 or TSAR_A2_SLOTS>1")
if A2_SLOTS > 1 and A2_ZDIM % N_HEADS:
    raise SystemExit(f"train_arch2: TSAR_N_HEADS={N_HEADS} (slot mixing) must divide TSAR_A2_ZDIM={A2_ZDIM}")
if A2_DREAD and A2_ZDIM % A2_DREAD_HEADS:
    raise SystemExit(f"train_arch2: TSAR_A2_DREAD_HEADS={A2_DREAD_HEADS} must divide TSAR_A2_ZDIM={A2_ZDIM}")
if A2_DREAD and (PREFIX or TTA_STEPS or SELFCAL):
    raise SystemExit("train_arch2: TSAR_A2_DREAD is refused with TSAR_PREFIX, TSAR_TTA_STEPS or TSAR_SELFCAL "
                     "(those paths roll the latent without the context memory)")
if MRES_EXTRA and PREFIX:
    raise SystemExit("train_arch2: TSAR_A2_MRES is refused with TSAR_PREFIX")
if A2_DRIVE and not SEAS_REF:
    raise SystemExit("train_arch2: TSAR_A2_DRIVE needs TSAR_SEAS_REF=1 (the drive signal is the seasonal ref)")
if A2_ZDIM != D_MODEL and (CTX_READ or TTA_SERIES):
    raise SystemExit("train_arch2: TSAR_A2_ZDIM != TSAR_D_MODEL is refused with TSAR_CTX_READ or TSAR_TTA_SERIES")
A2_DIRECT = _axis("TSAR_A2_DIRECT", 0, {0, 1})
A2_STATE_OFF = _axis("TSAR_A2_STATE_OFF", "none", {"none", "copy", "freeze", "zero"}, str)
if A2_DIRECT and (LOOPS or LATENT or A2_TRANS != "gru" or A2_DEPTH or PREFIX or A2_STATE_OFF in ("copy", "freeze")):
    raise SystemExit("train_arch2: TSAR_A2_DIRECT (no carried state) is refused with TSAR_LOOPS, TSAR_LATENT>0, "
                     "TSAR_A2_TRANS=res, TSAR_A2_DEPTH>0, TSAR_PREFIX and TSAR_A2_STATE_OFF=copy/freeze")
if A2_STATE_OFF == "copy" and (LOOPS or A2_TRANS != "gru" or A2_DEPTH or A2_SLOTS > 1):
    raise SystemExit("train_arch2: TSAR_A2_STATE_OFF=copy builds no transition; it is refused with TSAR_LOOPS, "
                     "TSAR_A2_TRANS=res, TSAR_A2_DEPTH>0 and TSAR_A2_SLOTS>1")
A2_NO_TRANS = A2_DIRECT or A2_STATE_OFF == "copy"            # self.dynamics is not built
if A2_NO_TRANS and TTA_STEPS:
    raise SystemExit("train_arch2: TSAR_TTA_STEPS adapts the transition; refused with TSAR_A2_DIRECT and STATE_OFF=copy")
ZD = A2_ZDIM

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
if DEV:                                    # train_search: DEV families are never trained on
    DV.holdout(by_dataset)
datasets = list(by_dataset)
SEASON = {s["dataset"]: s["season"] for s in train} if BENCH else {d[0]: d[2] for d in P.DATASETS}
EVALUATE = BN.evaluate if BENCH else P.evaluate
if DEV:
    EVALUATE = DV.evaluate                 # every call scores DEV (incl. TSAR_EVAL_MIN); VAL is never loaded


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
    sw = slice(0, TWIN) if PREFIX_SYN else slice(PREFIX, TWIN)        # synthetic rows: prefix only if PREFIX_SYN
    o = int(round(n_syn * SYN5_SHARE))                                   # TSAR_SYN5: synth5 rows first (0 = off)
    if o:
        win[:o, sw], per[:o] = S5.generate(rng, o, sw.stop - sw.start, min_len=2 * PATCH)
    n3 = int(round((n_syn - o) * SYN3_SHARE))                            # TSAR_SYN3: synth3 share of the rest
    n2 = int((n_syn - o - n3) * SYN2_SHARE)
    if n3:
        win[o:o + n3, sw], per[o:o + n3] = S3.generate(rng, n3, sw.stop - sw.start, min_len=2 * PATCH)
    win[o + n3:o + n3 + n2, sw], per[o + n3:o + n3 + n2] = synthetic2(rng, n2, sw.stop - sw.start)
    win[o + n3 + n2:n_syn, sw], per[o + n3 + n2:n_syn] = synthetic(rng, n_syn - o - n3 - n2, sw.stop - sw.start)
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


class ResBlock(nn.Module):
    """Pre-LN residual transition block z <- z + MLP(LN[z, u]); last layer zero-initialised (identity at init)."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(2 * d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u, z):
        return z + self.mlp(self.norm(torch.cat([z, u], -1)))


class A2Transition(nn.Module):
    """TSAR_A2_TRANS / DEPTH / SLOTS: one latent step (u, z) -> z' with the GRUCell(u, z) signature.
    z is [N, ZD] (or [N, M, ZD] slots when SLOTS > 1; the model keeps slots internally)."""

    def __init__(self, d):
        super().__init__()
        self.core = nn.GRUCell(d, d) if A2_TRANS == "gru" else None
        self.blocks = nn.ModuleList(ResBlock(d) for _ in range(A2_DEPTH))
        if A2_SLOTS > 1:
            self.mix_norm = nn.LayerNorm(d)
            self.mix = nn.MultiheadAttention(d, N_HEADS, batch_first=True)
            nn.init.zeros_(self.mix.out_proj.weight); nn.init.zeros_(self.mix.out_proj.bias)

    def forward(self, u, z):
        shp = z.shape
        if A2_SLOTS > 1:                                   # u [N, D] broadcast over slots; slots exchange info
            zn = self.mix_norm(z)
            u = u[:, None].expand_as(z) + self.mix(zn, zn, zn, need_weights=False)[0]
            u, z = u.reshape(-1, shp[-1]), z.reshape(-1, shp[-1])
        if self.core is not None:
            z = self.core(u, z)
        for blk in self.blocks:
            z = blk(u, z)
        return z.view(shp)


class CausalWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2 * PATCH + N_FRAME, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.pos = nn.Parameter(torch.zeros(1, N_PATCH, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 4 * D_MODEL, 0.0,
                                           batch_first=True, norm_first=True)
        self.lower = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.state_rnn = nn.GRU(D_MODEL, D_MODEL, batch_first=True)   # embedded world state
        self.upper = (nn.TransformerEncoder(layer, N_LAYERS - 1, enable_nested_tensor=False)
                      if N_LAYERS > 1 else None)       # N_LAYERS=1: no upper stack
        NT = N_PRE + N_PATCH                                          # prefix tokens precede fine tokens in time
        self.register_buffer("causal", torch.triu(torch.full((NT, NT), float("-inf")), 1))
        if PREFIX:                         # coarse long-context prefix: own embedding, positions and type
            self.pre_embed = nn.Sequential(nn.Linear(2 * PREFIX_PATCH + 3, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
            self.pre_pos = nn.Parameter(torch.zeros(1, N_PRE, D_MODEL))
            self.pre_type = nn.Parameter(0.02 * torch.randn(D_MODEL))
        self.to_state = nn.Linear(D_MODEL, ZD)                        # ZD = TSAR_A2_ZDIM (default D_MODEL)
        self.dynamics = (None if A2_NO_TRANS else LoopedDynamics(ZD) if LOOPS else A2Transition(ZD) if A2_CUSTOM_TRANS
                         else nn.GRUCell(ZD, ZD))
        self.cond = nn.Linear(ZD, ZD) if LOOPS else None              # observed world state -> per-step input
        self.step_emb = nn.Parameter(torch.zeros(K_STEPS, ZD))
        self.decode = nn.Sequential(nn.LayerNorm(ZD), nn.Linear(ZD, PATCH * NQ))
        self.ref_proj = nn.Linear(PATCH, ZD)                          # seasonal memory read into the state
        self.ref_skip = nn.Parameter(torch.zeros(NQ))                 # learned copy weight per quantile
        self.skip_gate = nn.Linear(ZD, NQ) if GATE else None          # per-series, per-step copy weight
        if GATE:
            nn.init.zeros_(self.skip_gate.weight); nn.init.zeros_(self.skip_gate.bias)
        if CTX_READ:                       # imagined states read the observed latent history (memory)
            self.read_q, self.read_k = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
            self.lag_emb = nn.Parameter(torch.zeros(N_PATCH, D_MODEL))
            self.read = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            nn.init.zeros_(self.read.out_proj.weight); nn.init.zeros_(self.read.out_proj.bias)
        # --- train_arch2 axes: built only when on, after every train_syn5 module (defaults draw no extra RNG)
        if MRES_EXTRA:                     # multi-resolution patch embedding, one projection per extra scale
            self.mres = nn.ModuleList(nn.Linear(2 * s, D_MODEL) for s in MRES_EXTRA)
        if A2_SLOTS > 1:                   # slot offsets from the origin state, and the slot readout
            self.slot_emb = nn.Parameter(0.02 * torch.randn(A2_SLOTS, ZD))
            self.slot_in = nn.Linear(ZD, A2_SLOTS * ZD)
            self.slot_out = nn.Linear(A2_SLOTS * ZD, ZD)
            for m_ in (self.slot_in, self.slot_out):
                nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)
        if A2_HZ:
            self.hz_emb = nn.Parameter(torch.zeros(K_STEPS, ZD))
        if A2_DRIVE:                       # seasonal-ref patch -> transition input (zero-init: starts as syn5)
            self.drive = nn.Linear(PATCH, ZD)
            nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        if A2_DREAD:                       # decoder memory read over the context tokens
            self.dread_v = nn.Linear(2 * PATCH, ZD)                   # raw patch (token frame) + mask
            self.dread_e = nn.Linear(D_MODEL, ZD)                     # encoder output at that token
            self.dread_q, self.dread_k = nn.LayerNorm(ZD), nn.LayerNorm(ZD)
            self.dread_wq, self.dread_wk, self.dread_wv = (nn.Linear(ZD, ZD) for _ in range(3))
            self.dread_bias = nn.Parameter(torch.zeros(A2_DREAD_HEADS, N_PATCH))   # per-head lag bias
            self.dread_out = nn.Linear(ZD, ZD)
            nn.init.zeros_(self.dread_out.weight); nn.init.zeros_(self.dread_out.bias)
        if A2_DIRECT:                      # W1: per-lead map from the origin state; no state carried between leads
            self.direct = nn.Sequential(nn.LayerNorm(2 * ZD), nn.Linear(2 * ZD, 2 * ZD), nn.GELU(), nn.Linear(2 * ZD, ZD))
            nn.init.zeros_(self.direct[-1].weight); nn.init.zeros_(self.direct[-1].bias)
            if A2_SLOTS > 1:               # slot mixing kept (origin slots, same for every lead)
                self.direct_mix_norm = nn.LayerNorm(ZD)
                self.direct_mix = nn.MultiheadAttention(ZD, N_HEADS, batch_first=True)
                nn.init.zeros_(self.direct_mix.out_proj.weight); nn.init.zeros_(self.direct_mix.out_proj.bias)
        if A2_STATE_OFF == "freeze":       # W3: the transition stays a random reservoir
            self.dynamics.requires_grad_(False)

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
        for s_, proj in zip(MRES_EXTRA, self.mres if MRES_EXTRA else []):   # last s_ values per token, causal
            xs = torch.cat([torch.full((B, s_ - PATCH), float("nan"), device=x.device), x], 1).unfold(1, s_, PATCH)
            os_ = ~torch.isnan(xs)                                    # [B, n, s_], token frame loc/scale
            vs = to_model_space(torch.nan_to_num(xs), loc[..., None], scale[..., None]).clamp(-20, 20)
            e = e + proj(torch.cat([torch.where(os_, vs, torch.zeros_like(vs)), os_.float()], -1))
        if PREFIX:
            e = torch.cat([self.prefix_tokens(xpre, x[:, :PATCH], loc, scale, cnt), e], 1)
        T = e.shape[1]
        h = self.lower(e, mask=self.causal[:T, :T])
        state, _ = self.state_rnn(h)                                  # carried patch to patch
        h = self.upper(h + state, mask=self.causal[:T, :T]) if self.upper is not None else h + state
        z = torch.tanh(self.to_state(h + state))
        if A2_DREAD:                       # kept for dmemory(); PREFIX is refused with DREAD, so T == n
            self._enc = h + state
        return (z[:, T - n:] if PREFIX else z), loc, scale, cnt

    def memory(self, z, b, j):
        """Per-origin memory: states z[b, :j+1] with a learned lag embedding. Returns keys [N, n, D], ignore-mask."""
        n = z.shape[1]
        lag = j[:, None] - torch.arange(n, device=z.device)[None]                 # [N, n]
        mem = self.read_k(z[b] + self.lag_emb[lag.clamp(0, N_PATCH - 1)])
        return mem, lag < 0

    def dmemory(self, x, loc, scale, b, j):
        """TSAR_A2_DREAD memory, built once per series (not per origin, which would cost [N, n, ZD]):
        token i = encoder output + its raw patch in its own causal frame. Origins (b, j) of the rows that
        will query it are carried along; the read masks tokens i > j. x: [B, L] raw fine context."""
        B, L = x.shape
        xp = x.view(B, L // PATCH, PATCH)
        obs = ~torch.isnan(xp)
        v = to_model_space(torch.nan_to_num(xp), loc[..., None], scale[..., None]).clamp(-20, 20)
        mem = self.dread_v(torch.cat([torch.where(obs, v, torch.zeros_like(v)), obs.float()], -1)) + self.dread_e(self._enc)
        return self.dread_k(mem), b, j

    def dread_read(self, S, dmem):
        """Rolled states S [N, K, ZD] of origins (b, j) cross-attend to their series' tokens i <= j, with a
        learned per-head lag bias. Rows are scattered to a dense [B, n*K] query grid so attention runs per
        series; the unused grid rows are zeros and are never gathered back."""
        mem, b, j = dmem
        B, n, _ = mem.shape
        H = A2_DREAD_HEADS
        qd = S.new_zeros(B, n, K_STEPS, ZD).index_put((b, j), self.dread_q(S))
        q = self.dread_wq(qd).view(B, n * K_STEPS, H, ZD // H).transpose(1, 2)
        k = self.dread_wk(mem).view(B, n, H, ZD // H).transpose(1, 2)
        v = self.dread_wv(mem).view(B, n, H, ZD // H).transpose(1, 2)
        ar = torch.arange(n, device=S.device)
        lag = ar[:, None] - ar[None]                                  # [origin j, token i] = j - i
        bias = self.dread_bias[:, lag.clamp(0, N_PATCH - 1)].masked_fill(lag < 0, float("-inf"))   # [H, n, n]
        bias = bias[:, :, None].expand(H, n, K_STEPS, n).reshape(1, H, n * K_STEPS, n)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)   # [B, H, n*K, dh]
        o = o.transpose(1, 2).reshape(B, n, K_STEPS, ZD)[b, j]
        return S + self.dread_out(o)

    def rollout(self, z, ref=None, return_states=False, mem=None, dmem=None):
        """z: [N, D], ref: [N, MAX_H] seasonal reference -> quantiles in model space [N, K*PATCH, NQ].
        The transition cell imagines K future latent states without observations (prior);
        return_states also returns them [N, K, D] for the latent-consistency loss."""
        if ref is None or not SEAS_REF:
            ref = torch.zeros(z.shape[0], MAX_H, device=z.device)
        ref = ref.clamp(-20, 20)
        outs, zs = [], []
        c = self.cond(z) if LOOPS else 0
        N = z.shape[0]
        if A2_SLOTS > 1:                   # [N, M, ZD] slots around the origin state
            z = z[:, None] + self.slot_emb + self.slot_in(z).view(N, A2_SLOTS, ZD)
        if A2_NO_TRANS:
            z0 = z
            if A2_DIRECT and A2_SLOTS > 1:
                zn = self.direct_mix_norm(z0)
                mixed = self.direct_mix(zn, zn, zn, need_weights=False)[0]
        for k in range(K_STEPS):
            u = self.step_emb[k].expand(N, -1) + c
            if A2_DRIVE:
                u = u + self.drive(ref[:, k * PATCH:(k + 1) * PATCH])
            if A2_DIRECT:                  # W1: lead k from the origin state only
                uk = u[:, None] + mixed if A2_SLOTS > 1 else u
                z = z0 + self.direct(torch.cat([z0, uk], -1))
            elif A2_STATE_OFF == "copy":   # W2: copy the origin state, keep step_emb and drive
                z = z0 + u
            else:
                z = self.dynamics(u, z)
            zs.append(z.mean(1) + self.slot_out(z.flatten(1)) if A2_SLOTS > 1 else z)   # decoded state
        S = torch.stack(zs, 1)                                        # imagined states [N, K, D]
        if A2_STATE_OFF == "zero":         # W4: only the seasonal-ref paths remain
            S = torch.zeros_like(S)
        if CTX_READ and mem is not None:
            keys, ignore = mem
            S = S + self.read(self.read_q(S), keys, keys, key_padding_mask=ignore, need_weights=False)[0]
        if A2_DREAD:
            if dmem is None:
                raise RuntimeError("TSAR_A2_DREAD: rollout called without its context memory")
            S = self.dread_read(S, dmem)
        for k in range(K_STEPS):
            z = S[:, k]
            r = ref[:, k * PATCH:(k + 1) * PATCH]
            w = self.ref_skip + (self.skip_gate(z)[:, None] if GATE else 0)
            q = self.decode(z + self.ref_proj(r) + (self.hz_emb[k] if A2_HZ else 0)).view(-1, PATCH, NQ) + r[..., None] * w
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
    dmem = None
    if A2_DREAD:
        bj = torch.arange(B * N_PATCH, device=win.device)[keep]
        dmem = model.dmemory(win, loc, scale, bj // N_PATCH, bj % N_PATCH)
    if LATENT:
        pred, prior = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], True, mem, dmem)
        return pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float()) \
            + LATENT * latent_consistency(z, prior, keep)
    pred = model.rollout(z.reshape(B * N_PATCH, -1)[keep], ref.reshape(B * N_PATCH, MAX_H)[keep], mem=mem, dmem=dmem)
    return pinball(pred, tgt.reshape(B * N_PATCH, MAX_H)[keep], valid.reshape(B * N_PATCH, MAX_H)[keep].float())

# Training (fixed time budget)
# ---------------------------------------------------------------------------

model = CausalWorldModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
if A2_DIRECT or A2_STATE_OFF != "none":
    _dyn = model.direct if A2_DIRECT else model.dynamics
    print(f"wm_test: params {n_params} trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
          f"transition_or_direct {sum(p.numel() for p in _dyn.parameters()) if _dyn is not None else 0}", flush=True)
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
    EB = 512 if TCTX <= 512 else max(64, 512 * 512 // TCTX)   # eval batch: smaller for long contexts (MPS memory)
    for s in range(0, len(ctxs), EB):
        chunk = ctxs[s:s + EB]
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
    dmem = None
    if A2_DREAD:                                          # PREFIX is refused with DREAD: x is the fine context
        B, n = z.shape[:2]
        dmem = m.dmemory(x, loc, scale, torch.arange(B, device=z.device), torch.full((B,), n - 1, device=z.device))
    q = m.rollout(zl, ref, mem=mem, dmem=dmem)[:, :h]                # origin = last context patch
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
    step, done, rng = ck["step"], ck["done"], ck["rng"]
    print(f"resumed {RUN} at step {step}, {done / 60:.1f} min trained", flush=True)


def save_ckpt(elapsed, final=False):
    if CKPT:
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "done": elapsed, "rng": rng, "zs": ZS, "cfg": run_cfg(), "complete": final,
                    "budget": {"steps": STEPS, "seconds": BUDGET}}, CKPT + ".tmp")
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
save_ckpt(training_seconds, final=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

split = "dev" if DEV else ("test" if IS_TEST else "val")


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
print(f"num_steps:        {step}")
pdx = res["per_dataset"].set_index("dataset")
zs = float(np.exp(np.log(pdx.loc[ZS_SETS, "wql_rel"]).mean())) if not BENCH else float("nan")
if BENCH:
    print(f"family_wql_rel:   {res['wql_rel_family']:.6f}")
if DEV:
    print(f"dev_subday_wql_rel: {res['wql_rel_subday']:.6f}")
    print(f"dev_family_wql_rel: {res['wql_rel_family']:.6f}")
print(f"zs4_wql_rel:      {zs:.6f}   ({'held out' if ZS else 'trained on'}: {', '.join(ZS_SETS)})")
if split == "test" and not BENCH:          # confirmatory run: also score the 5 never-trained holdout datasets
    ho = P.evaluate(predict, split="holdout")
    print(f"holdout_wql_rel:  {ho['wql_rel']:.6f}")
    print(f"holdout_mase_rel: {ho['mase_rel']:.6f}")
    print(f"holdout_cov80_error: {ho['cov80_error']:.6f}")
