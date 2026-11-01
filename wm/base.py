"""
The world-model family interface for train_wm.py (Haddad, 2026-10-30). Read docs/wm_families.md first.

The harness (train_wm.py) owns everything that must be identical across families: data and synthetic mix,
causal normalisation (per-patch running loc/scale + asinh), the seasonal-naive reference, the targets, the
quantile (pinball) loss, the learned per-quantile seasonal-ref skip, the optimiser and schedule, and the bench
evaluation. A family owns only the model between "normalised context" and "9 quantiles per lead patch".

A family is an nn.Module subclass of WMFamily with two methods:

    enc = family.encode(ctx)                      # once per batch of series
    q, aux = family.decode(enc, ctx, b, j)        # for N origins (b[i], j[i])

    ctx : Context (below). Training: B windows of WINDOW = CONTEXT + MAX_H points, n = WINDOW // PATCH tokens,
          and every patch j with at least one observed future value is an origin. Evaluation: B contexts of
          CONTEXT points, n = CONTEXT // PATCH tokens, one origin per series at j = n - 1.
    b, j: LongTensor [N]: series row and origin patch index; the forecast for origin (b, j) starts right after
          patch j and must depend ONLY on ctx values in patches 0..j of row b (checked at startup, see
          train_wm.causality_check). Training uses all origins of a window at once, so encode() must be causal
          (a causal mask, a recurrence, or per-origin recomputation).
    q   : FloatTensor [N, K, PATCH, NQ] quantiles in the model space of origin (b, j), i.e. asinh((y - loc) / scale)
          with loc/scale = ctx.loc[b, j], ctx.scale[b, j]. Lead patch k covers steps k*PATCH+1 .. (k+1)*PATCH
          after the origin. Quantile order = prepare.QUANTILES (0.1 .. 0.9). The harness then adds
          ref_skip[q] * ref[b, j, k*PATCH + t] (one learned weight per quantile, zero-initialised) and scores
          the pinball loss; it does not sort the quantiles in training (bench.evaluate sorts at scoring).
    aux : dict[str, scalar Tensor] of extra training losses (KL, latent prediction, ...). Each key must appear
          in the family's class attribute aux_weights {name: weight}; the objective is
          pinball + sum(aux_weights[k] * aux[k]). Return {} when there is none. aux is ignored at evaluation.

Ablation modes (TSAR_WM_ABL, given to the family as cfg.abl). Every family must implement the four, or refuse
one in __init__ with SystemExit(why) when it is meaningless for that family (base.refuse()):

    none    the family as designed: a latent state carried and rolled forward from lead k to lead k + 1.
    direct  no carried state: lead k is computed from the origin state and the lead's inputs only, with a
            parameter- and MAC-matched replacement for the transition (within 3% of 'none').
    copy    no transition: z_k = z_0 + (the lead's input embedding); the transition's parameters are removed
            (unmatched by construction; report the count).
    zero    the rolled states are zeroed before the decoder, so only the ref/input paths remain. It is an
            inference key: it must build exactly the same parameters as 'none' so that a checkpoint trained with
            'none' can be re-scored with TSAR_WM_EVAL_ONLY=<ckpt> TSAR_WM_ABL=zero.

Rules for families (enforced or checked by the harness where possible):
  - Read sizes from cfg (d_model, n_layers, n_heads, ...), never from the environment directly, so TSAR_WM_WIDTH
    scales every family the same way. A family-specific knob is an env var TSAR_WM_<FAMILY>_<NAME>,
    declared in the class attribute env_keys (the strict-key check accepts only the harness keys plus the selected
    family's env_keys) and defaulted so the default width lands within +-15% of the reference parameter count
    (docs/wm_families.md).
  - No foundation-model weights or outputs, no data access: a family sees only ctx.
  - Draw random numbers only from torch (not numpy's global state) so seeds reproduce.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

ABLATIONS = ("none", "direct", "copy", "zero")


@dataclass(frozen=True)
class WMConfig:
    patch: int          # points per patch (8); also the length of one lead patch
    context: int        # inference context length in points (CONTEXT, 256)
    max_h: int          # latent rollout length in points (TSAR_MAX_H, default 64); longer horizons are continued by
                        # the harness (TSAR_WM_LONG=median|paths)
    k_steps: int        # lead patches K = max_h // patch
    n_patch: int        # tokens in a TRAINING window ((context + max_h) // patch); eval uses context // patch
    nq: int             # number of quantiles (9)
    tok_dim: int        # features per token in ctx.tok (2 * patch + 3)
    d_model: int        # model width after TSAR_WM_WIDTH scaling (96 at width 1, a multiple of 8)
    n_layers: int       # TSAR_N_LAYERS (2)
    n_heads: int        # TSAR_N_HEADS (4); divides d_model
    width: float        # TSAR_WM_WIDTH multiplier (1.0)
    abl: str            # TSAR_WM_ABL, one of ABLATIONS


@dataclass
class Context:
    """Everything a family may read about a batch of series. Shapes: B series, L points, n = L // PATCH tokens.
    Every per-patch quantity at index j is computed from points <= the end of patch j only (causal)."""
    tok: torch.Tensor    # [B, n, tok_dim] token j: patch j in ITS OWN frame asinh((x - loc_j) / scale_j) (PATCH,
                         # missing = 0), observed mask (PATCH), frame features (dloc, dlog-scale, log1p(count) / 5)
    x: torch.Tensor      # [B, L] raw values, NaN = missing or left padding, oldest first
    loc: torch.Tensor    # [B, n] running frame at the end of patch j (TSAR_NORM: last value / running mean)
    scale: torch.Tensor  # [B, n] running scale (mean |step| for NORM=last), > 0
    cnt: torch.Tensor    # [B, n] observed points up to the end of patch j (0 = pure padding so far)
    ref: torch.Tensor    # [B, n, max_h] seasonal-naive forecast from origin j in origin j's model space, clamped
                         # to [-20, 20] (0 where no past season exists). The harness also adds it via ref_skip.


def to_model_space(v, loc, scale):
    return torch.asinh((v - loc) / scale)


def from_model_space(q, loc, scale):
    return torch.sinh(q) * scale + loc


def refuse(family, abl, why):
    raise SystemExit(f"wm family {family!r} refuses TSAR_WM_ABL={abl}: {why}")


class WMFamily(nn.Module):
    """Base class. Subclasses set name and (optionally) aux_weights and env_keys, call super().__init__(cfg), and implement
    encode / decode as documented in the module docstring. param_groups() is optional reporting: a dict
    {label: module} printed at startup (e.g. {"transition": self.dynamics}) so matched controls can be audited."""

    name = "base"
    aux_weights = {}          # {aux loss name: weight in the objective}
    env_keys = ()             # TSAR_WM_<NAME>_* keys this family reads (strict-key check)
    max_k = None              # largest number of lead patches K = MAX_H / PATCH it supports (None = any);
    max_k_why = ""            # the harness refuses TSAR_MAX_H above max_k * PATCH with this reason

    def __init__(self, cfg: WMConfig):
        super().__init__()
        if cfg.abl not in ABLATIONS:
            raise SystemExit(f"TSAR_WM_ABL={cfg.abl!r} not in {ABLATIONS}")
        self.cfg = cfg

    def encode(self, ctx: Context):
        raise NotImplementedError

    def decode(self, enc, ctx: Context, b: torch.Tensor, j: torch.Tensor):
        raise NotImplementedError

    def param_groups(self):
        return {}
