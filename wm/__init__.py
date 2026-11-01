"""World-model families for train_wm.py. The interface is in wm/base.py; how to add one is in docs/wm_families.md.

REGISTRY maps TSAR_WM=<name> to "module:Class". Entries are imported lazily (get), so a broken or heavy family
never affects the others.
"""

import importlib

REGISTRY = {
    "gru": "wm.gru:GRUFamily",      # reference: train_arch2 syn5 recipe + A2_DRIVE=1 (causal Transformer + GRUCell rollout)
    # wave 5 families (pre-registered by the lead so agents never edit this file concurrently; each module is
    # written by its own agent and imported only when selected)
    "rssm": "wm.rssm:RSSMFamily",        # stochastic + deterministic latent state (Dreamer-style), prior/posterior KL
    "koopman": "wm.koopman:KoopmanFamily",  # linear latent dynamics with a learned complex spectrum (damped oscillators)
    "ssm": "wm.ssm:SSMFamily",           # selective state-space (Mamba/S6-style) encoder + state rollout
    "ctime": "wm.ctime:CTimeFamily",     # continuous-time latent dynamics (closed-form CfC / neural ODE)
    "jepa": "wm.jepa:JEPAFamily",        # latent-predictive world model: predict future encodings (EMA target encoder)
    "vqtok": "wm.vqtok:VQTokFamily",     # discrete latent tokens + autoregressive transition over codes
    "flow": "wm.flow:FlowFamily",        # generative latent rollout trained by flow matching; quantiles from samples
    "tswm": "wm.tswm:TSWMFamily",        # Timestral backbone (re-implemented, trained from scratch) with a world-model horizon block;
                                         # abl=direct is the faithful Timestral DHF replica
}


def get(name):
    """Return (family class, path of its source file) for TSAR_WM=name."""
    if name not in REGISTRY:
        raise SystemExit(f"TSAR_WM={name!r} is not a registered family; known: {sorted(REGISTRY)}")
    mod_name, cls_name = REGISTRY[name].split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name), mod.__file__
