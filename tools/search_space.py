"""Architecture search space for tools/search.py (Hana Kim). Plain data plus a few predicates.

Every axis names its env flag, its values (first = the incumbent's value), and the scripts that implement it.
A config runs under ONE script: the first entry of SCRIPTS that implements every flag the config sets (train_v2
lineage refuses unknown TSAR_* keys, audit_v3 H-2). Screens need TSAR_DEV, so a script without it is skipped
(see script_ok). Flags were taken from `grep '_env("TSAR_'` of train_search.py (= train_v2.py READ_KEYS +
TSAR_SYN5 + TSAR_DEV). train_syn5.py lacks TSAR_DEV and the architecture flags, so it is not a search script.
train_arch2.py (Mehta, 2026-10-30) = train_syn5.py + composable TSAR_A2_* structural axes (ARCH2_AXES, legal values
from its docstring and _axis() sets). It has no TSAR_* flag of train_v2's (TRANS, JUMP, HDEC, ...), so a config uses
either v2 axes or A2 axes, plus the shared ones. Its TSAR_DEV hook landed in d883207 (DEV metrics
identical to train_search.py at defaults). `python tools/search.py flags` lists each script's TSAR_* keys missing here.

Base recipe (fixed): the best data recipe (syn5_only, VAL 0.6853 at 2500 steps, seed 0).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["train_search.py", "train_arch2.py"]          # preference order

BASE = {"TSAR_BENCH": "1", "TSAR_SYN_FRAC": "0.3", "TSAR_SYN3": "1", "TSAR_SYN_CURR": "0.9", "TSAR_AUG": "0.5",
        "TSAR_SYN5": "1"}
# every search job (memory cap from docs/compute.md section 3 item 2, as wave3; raised for long contexts)
JOB_ENV = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.75", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}   # 0.55 OOMed 30/122 R0 configs (2026-10-30)

S = ["train_search.py"]
A = ["train_arch2.py"]
B = S + A                                                # both scripts implement it
# name: flag, values (values[0] = incumbent / script default), scripts, optional `when` (axis active only then),
# optional `tie` {other flag: {value: tied value}} for flags set together with this one.
AXES = {
    # context length (Park gap analysis: long-history hourly tasks carry ~70% of the gap). Batch tied to cap memory
    # (lead: ctx1024 at batch 128 peaks 2.6 GB on MPS; batch 64 at 2048).
    "context":   {"flag": "TSAR_CONTEXT", "values": [256, 512, 1024, 2048], "scripts": B,
                  "tie": {"TSAR_BATCH": {256: 128, 512: 128, 1024: 96, 2048: 64}}},
    "d_model":   {"flag": "TSAR_D_MODEL", "values": [96, 64, 128, 192], "scripts": B},
    "n_layers":  {"flag": "TSAR_N_LAYERS", "values": [2, 1, 3, 4], "scripts": B},
    "n_heads":   {"flag": "TSAR_N_HEADS", "values": [4, 2, 8], "scripts": B},
    # transition type and its latent size (z_t itself is d_model wide)
    "trans":     {"flag": "TSAR_TRANS", "values": ["gru", "koopman", "ssm", "rssm", "attn"], "scripts": S},
    "ssm_n":     {"flag": "TSAR_SSM_N", "values": [8, 4, 16], "scripts": S, "when": ("trans", "ssm")},
    "rssm_s":    {"flag": "TSAR_RSSM_S", "values": [32, 16, 64], "scripts": S, "when": ("trans", "rssm")},
    "loops":     {"flag": "TSAR_LOOPS", "values": [0, 2], "scripts": B},
    "jump":      {"flag": "TSAR_JUMP", "values": [0, 1, 2], "scripts": S},
    "dyn_in":    {"flag": "TSAR_DYN_IN", "values": [0, 1, 2, 3], "scripts": S},
    # K = MAX_H / PATCH rollout steps (4 / 8 / 16); longer horizons are chunked
    "max_h":     {"flag": "TSAR_MAX_H", "values": [64, 32, 128], "scripts": B},
    # one-shot long horizon and the coarse multi-scale world model
    "long_h":    {"flag": "TSAR_LONG_H", "values": [0, 256], "scripts": S},
    "ms_coarse": {"flag": "TSAR_MS_COARSE", "values": [0, 1, 2], "scripts": S, "when": ("long_h", 256)},
    # decoder variants
    "hdec":      {"flag": "TSAR_HDEC", "values": [0, 1], "scripts": S},
    "dualdec":   {"flag": "TSAR_DUALDEC", "values": [0, 0.5], "scripts": S},
    "seas_ref":  {"flag": "TSAR_SEAS_REF", "values": [1, 0], "scripts": B},
    "gate":      {"flag": "TSAR_GATE", "values": [0, 1], "scripts": B},
    "ctx_read":  {"flag": "TSAR_CTX_READ", "values": [0, 1], "scripts": B},
    "norm":      {"flag": "TSAR_NORM", "values": ["last", "mean"], "scripts": B},
    # world-model objective
    "jepa":      {"flag": "TSAR_JEPA", "values": [0, 0.1], "scripts": S},
    # optimisation
    "lr":        {"flag": "TSAR_LR", "values": [3e-3, 1.5e-3, 6e-3], "scripts": B},
    "sched":     {"flag": "TSAR_SCHED", "values": ["cos", "wsd"], "scripts": B},
}
# train_arch2.py structural axes. a2_zdim is the latent state width ("d" = TSAR_D_MODEL, flag unset).
ARCH2_AXES = {
    "a2_zdim":   {"flag": "TSAR_A2_ZDIM", "values": ["d", 48, 64, 128, 192], "scripts": A},
    "a2_trans":  {"flag": "TSAR_A2_TRANS", "values": ["gru", "res"], "scripts": A},
    "a2_depth":  {"flag": "TSAR_A2_DEPTH", "values": [0, 1, 2, 3], "scripts": A},
    "a2_slots":  {"flag": "TSAR_A2_SLOTS", "values": [1, 2, 4, 8], "scripts": A},
    "a2_hz":     {"flag": "TSAR_A2_HZ", "values": [0, 1], "scripts": A},
    "a2_drive":  {"flag": "TSAR_A2_DRIVE", "values": [0, 1], "scripts": A},
    "a2_dread":  {"flag": "TSAR_A2_DREAD", "values": [0, 1], "scripts": A},
    "a2_dread_heads": {"flag": "TSAR_A2_DREAD_HEADS", "values": [4, 1, 2, 8], "scripts": A, "when": ("a2_dread", 1)},
    "a2_mres":   {"flag": "TSAR_A2_MRES", "values": ["8", "8,16", "8,32", "8,16,32", "8,16,32,64"], "scripts": A},
}
AXES.update(ARCH2_AXES)


def active(cfg, name):
    w = AXES[name].get("when")
    return w is None or cfg.get(w[0]) == w[1]


# Refused combinations: (reason, predicate on the axis-value dict). The first eight mirror train_v2's own asserts
# (a refused job would crash and waste its slot); the rest are memory and budget caps for this machine.
REFUSED = [
    ("TRANS!=gru with LOOPS", lambda c: c["trans"] != "gru" and c["loops"]),
    ("LONG_H with TRANS!=gru/JEPA/DUALDEC", lambda c: c["long_h"] and (c["trans"] != "gru" or c["jepa"] or c["dualdec"])),
    ("LONG_H fine extension with JUMP", lambda c: c["long_h"] and not c.get("ms_coarse") and c["jump"]),
    ("LONG_H must exceed MAX_H", lambda c: c["long_h"] and c["long_h"] <= c["max_h"]),
    ("MS_COARSE needs MAX_H % 32 == 0", lambda c: c.get("ms_coarse") and c["max_h"] % 32),
    ("HDEC with DUALDEC", lambda c: c["hdec"] and c["dualdec"]),
    ("d_model % n_heads", lambda c: c["d_model"] % c["n_heads"]),
    ("DYN_IN bit 1 with LOOPS is a no-op", lambda c: c["loops"] and c["dyn_in"] & 1),
    # train_arch2.py's own refusals
    ("A2 TRANS=res needs DEPTH >= 1", lambda c: c["a2_trans"] == "res" and c["a2_depth"] == 0),
    ("LOOPS with A2 custom transition", lambda c: c["loops"] and (c["a2_trans"] != "gru" or c["a2_depth"] or c["a2_slots"] > 1)),
    ("A2 SLOTS: n_heads must divide ZDIM", lambda c: c["a2_slots"] > 1 and _zd(c) % c["n_heads"]),
    ("A2 DREAD heads must divide ZDIM", lambda c: c["a2_dread"] and _zd(c) % c.get("a2_dread_heads", 4)),
    ("A2 DRIVE needs SEAS_REF", lambda c: c["a2_drive"] and not c["seas_ref"]),
    # A2_DRIVE and train_search DYN_IN=2 are the same zero-init Linear(ref patch) added to the transition input; R0
    # c0020 vs c0046 gave identical DEV metrics. DRIVE is kept only where DYN_IN cannot go (with other A2 axes).
    ("A2 DRIVE alone duplicates DYN_IN=2", lambda c: c["a2_drive"] and not any(
        c[k] != AXES[k]["values"][0] for k in ("a2_zdim", "a2_trans", "a2_depth", "a2_slots", "a2_hz", "a2_dread", "a2_mres"))),
    ("A2 ZDIM != d_model with CTX_READ", lambda c: c["a2_zdim"] != "d" and c["ctx_read"]),
    # Mehta's peak MPS at batch 128: base 1422 MB, SLOTS=4 1230, DEPTH=2 1660, SLOTS=8 2252
    ("memory: A2 DREAD at ctx 2048 (preflight OOM in dread_read, 1.1 GiB alloc at the 0.75 cap)",
     lambda c: c["a2_dread"] and c["context"] >= 2048),
    ("memory: A2 SLOTS=8 only at ctx <= 512", lambda c: c["context"] > 512 and c["a2_slots"] == 8),
    ("memory: A2 SLOTS x DEPTH > 8", lambda c: c["a2_slots"] * max(c["a2_depth"], 1) > 8),
    ("memory: ctx >= 1024 with d_model 192 or 4 layers",
     lambda c: c["context"] >= 1024 and (c["d_model"] >= 192 or c["n_layers"] >= 4)),
    ("memory: ctx 2048 with MAX_H 128 or LONG_H", lambda c: c["context"] == 2048 and (c["max_h"] == 128 or c["long_h"])),
]


def _zd(c):
    return c["d_model"] if c["a2_zdim"] == "d" else c["a2_zdim"]


def refused(cfg):
    return [r for r, f in REFUSED if f(cfg)]


def incumbent():
    return normalize({})


def normalize(cfg):
    """Drop inactive conditional axes; fill missing active axes with their incumbent value."""
    out = {}
    for k, a in AXES.items():
        tmp = {**out, **cfg}
        if active(tmp, k):
            out[k] = cfg.get(k, a["values"][0])
    if out.get("a2_zdim") == out["d_model"]:          # an explicit ZDIM equal to D_MODEL is the default
        out["a2_zdim"] = "d"
    return out


def restrict(cfg, script):
    """Reset every axis the script does not implement to its default (used when sampling per script)."""
    return {k: (v if script in AXES[k]["scripts"] else AXES[k]["values"][0]) for k, v in cfg.items()}


def env_of(cfg):
    """TSAR_* env of a config: BASE + every non-default flag (+ tied flags). Defaults are left unset."""
    env = dict(BASE)
    for k, v in normalize(cfg).items():
        a = AXES[k]
        if v != a["values"][0]:
            env[a["flag"]] = str(v)
        for f, m in a.get("tie", {}).items():
            if m[v] != m[a["values"][0]]:
                env[f] = str(m[v])
    return env


def mem_mb(cfg):
    """jobq admission estimate: measured 2.9 GB footprint at ctx 256 on CPU, ~1.5-2 GB on MPS (wave3 mem= fields)."""
    c = normalize(cfg)
    m = {256: 2000, 512: 2500, 1024: 3000, 2048: 3500}[c["context"]]
    return m + (500 if c["d_model"] >= 192 or c["n_layers"] >= 4 or c["a2_slots"] == 8 or c["a2_depth"] >= 2 else 0)


def script_keys(script):
    p = ROOT / script
    return set(re.findall(r'"(TSAR_[A-Z0-9_]*[A-Z0-9])"', p.read_text())) if p.exists() else None


def script_ok(script):
    keys = script_keys(script)
    return keys is not None and "TSAR_DEV" in keys


def script_for(cfg, prefer=None):
    """`prefer` (the sampled script family) if it implements every axis the config moves off its default, BASE and
    TSAR_DEV; else the first such script in SCRIPTS."""
    c = normalize(cfg)
    need = [k for k, v in c.items() if v != AXES[k]["values"][0]]
    for s in ([prefer] if prefer else []) + SCRIPTS:
        if all(s in AXES[k]["scripts"] for k in need) and script_ok(s):
            keys = script_keys(s)
            if set(BASE) <= keys:
                return s
    return None
