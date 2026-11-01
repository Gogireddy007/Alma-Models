"""
Family `jepa`: a latent-predictive world model (I-JEPA / V-JEPA style) with an EMA target encoder.

Design
  encode   online encoder f_theta: patch embed -> causal Transformer (N_LAYERS) -> z_j = tanh(to_state(h_j)), the
           latent world state after observing patch j. No recurrent state inside the encoder (unlike `gru`).
  rollout  a residual MLP predictor g (2 blocks, each x <- x + MLP(LN[x, u_k]), zero-initialised output) rolls the
           state one patch at a time: z_{k+1} = g(z_k, u_k), with action u_k = step_emb[k] + drive(ref_k) (the
           seasonal-naive patch as an exogenous input, as in `gru`).
  objective the rolled state z^_k (lead k = patch j+1+k) is trained to equal the EMA target encoder's state at that
           patch, sg(zbar_{j+1+k}) (aux `jepa`, L1). This is the world-model property the `gru` rollout lacks: the
           transition is supervised to reproduce the encoder's own future latent states, not only through the
           quantile decoder. Quantiles are decoded from the predicted latent: decode_head(z^_k + ref_proj(ref_k)).
  collapse EMA target (TSAR_WM_JEPA_TAU, updated at every training forward from the post-step online weights),
           stop-gradient, plus VICReg variance (aux `var`, hinge at TSAR_WM_JEPA_GAMMA) and covariance (aux `cov`)
           terms on the online encoder states. aux `zstd` (weight 0) logs the target-state std as a collapse monitor.

Future patches and leakage
  In training the harness passes the whole window (CONTEXT + MAX_H points), so the target states of patches
  j+1..j+K are computed from ctx.tok inside the family; they enter ONLY the aux loss, exactly like the pinball
  targets. The quantiles never read them (wm_causal checks the forecasts). At evaluation ctx holds only the context
  and j = n-1, so no target exists; the target path runs only when self.training and grad is enabled, so it costs
  nothing at eval and is not counted in macs_per_forecast. No harness change was needed.

EMA target weights are registered as buffers (not parameters): saved in checkpoints, not trainable and not in the
harness's params count. They add exactly the encoder's parameter count (248,064 at width 1) as non-trainable state.
Caveat: with TSAR_EMA (harness model averaging) the averaged copy's buffers are not averaged; that is harmless here
because the target encoder is never used at evaluation.

Prior work on this repo: train_latent's TSAR_JEPA (b_jepa_*) added a projected BYOL/JEPA aux at weight 0.1 on top
of the GRU rollout and was neutral. New hypothesis here: when latent prediction is the transition's own training
signal (no GRU, no projector, weight 1.0 in state space), the rolled state carries information that a direct
lead-indexed readout (`direct`) cannot, so `none` beats `direct`.

Ablations (TSAR_WM_ABL)
  none    z_{k+1} = g(z_k, u_k), rolled from z_0 = z_j.
  direct  z_k = g(z_0, u_k): the same predictor (identical params and MACs) applied to the origin state for each lead;
          no state carried from lead to lead. Same jepa aux (each lead still predicts zbar_{j+1+k}).
  copy    z_k = z_0 + u_k; predictor not built (111,936 fewer params at width 1). jepa aux still applied.
  zero    rolled states zeroed before the decoder (same params as none).

Sizes at width 1 (d=96): encoder 248,064, predictor 111,936, heads 9,672, ref_skip 9: total 369,681 (gru +0.05%),
copy 257,745 (111,936 fewer). Width 1.6667 (d=160): 1,005,201. Scales as d^2. MACs per forecast 9.07M (none,
  direct, zero) and 8.19M (copy) vs gru 10.4M.

Env keys
  TSAR_WM_JEPA_W      weight of the latent-prediction loss (default 1.0)
  TSAR_WM_JEPA_VAR    VICReg variance weight (default 0.1)
  TSAR_WM_JEPA_COV    VICReg covariance weight (default 0.01)
  TSAR_WM_JEPA_GAMMA  variance hinge target std (default 0.5; states are tanh-bounded)
  TSAR_WM_JEPA_TAU    EMA momentum of the target encoder (default 0.995)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from wm.base import WMFamily

_W = float(os.environ.get("TSAR_WM_JEPA_W", 1.0))
_VAR = float(os.environ.get("TSAR_WM_JEPA_VAR", 0.1))
_COV = float(os.environ.get("TSAR_WM_JEPA_COV", 0.01))
_GAMMA = float(os.environ.get("TSAR_WM_JEPA_GAMMA", 0.5))
_TAU = float(os.environ.get("TSAR_WM_JEPA_TAU", 0.995))


class _Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.to_state = nn.Linear(D, D)

    def forward(self, tok, mask):
        n = tok.shape[1]
        h = self.tf(self.embed(tok) + self.pos[:, :n], mask=mask)
        return torch.tanh(self.to_state(h))


class _Predictor(nn.Module):
    """x <- x + MLP(LN[x, u]) twice; zero-initialised outputs so it starts as the identity."""

    def __init__(self, D, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
            for _ in range(n_blocks))
        for blk in self.blocks:
            nn.init.zeros_(blk[-1].weight); nn.init.zeros_(blk[-1].bias)

    def forward(self, z, u):
        for blk in self.blocks:
            z = z + blk(torch.cat([z, u], -1))
        return z


def _vicreg(z):
    """Variance hinge and off-diagonal covariance of z [M, D] (VICReg)."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = F.relu(_GAMMA - std).mean()
    c = (z.T @ z) / max(z.shape[0] - 1, 1)
    off = c - torch.diag(torch.diag(c))
    return var, off.pow(2).sum() / z.shape[1]


class JEPAFamily(WMFamily):
    name = "jepa"
    aux_weights = {"jepa": _W, "var": _VAR, "cov": _COV, "zstd": 0.0}
    env_keys = ("TSAR_WM_JEPA_W", "TSAR_WM_JEPA_VAR", "TSAR_WM_JEPA_COV", "TSAR_WM_JEPA_GAMMA", "TSAR_WM_JEPA_TAU")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P = cfg.d_model, cfg.patch
        self.enc = _Encoder(cfg)
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        self.pred = _Predictor(D) if cfg.abl in ("none", "zero", "direct") else None
        self.step_emb = nn.Parameter(torch.zeros(cfg.k_steps, D))
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.ref_proj = nn.Linear(P, D)
        self.decode_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, P * cfg.nq))
        # EMA target encoder: buffers mirroring self.enc's parameters ('.' -> '__' in buffer names)
        self._tgt_names = []
        for n_, p in self.enc.named_parameters():
            bn = "tgt__" + n_.replace(".", "__")
            self.register_buffer(bn, p.detach().clone())
            self._tgt_names.append((n_, bn))

    def param_groups(self):
        return {"encoder": self.enc, "transition": self.pred if self.pred is not None else nn.Identity()}

    @torch.no_grad()
    def _ema_update(self):
        params = dict(self.enc.named_parameters())
        for n_, bn in self._tgt_names:
            getattr(self, bn).lerp_(params[n_].detach(), 1 - _TAU)

    def _train_aux(self):
        return self.training and torch.is_grad_enabled()

    def encode(self, ctx):
        n = ctx.tok.shape[1]
        mask = self.causal[:n, :n]
        z = self.enc(ctx.tok, mask)                                       # [B, n, D]
        tgt = None
        if self._train_aux():
            self._ema_update()
            with torch.no_grad():
                tgt = functional_call(self.enc, {n_: getattr(self, bn) for n_, bn in self._tgt_names},
                                      (ctx.tok, mask))
        return z, tgt

    def decode(self, enc, ctx, b, j):
        z_all, tgt = enc
        cfg, abl = self.cfg, self.cfg.abl
        P, N, K = cfg.patch, len(b), cfg.k_steps
        z0 = z = z_all[b, j]
        ref = ctx.ref[b, j]                                               # [N, MAX_H]
        zs = []
        for k in range(K):
            u = self.step_emb[k].expand(N, -1) + self.drive(ref[:, k * P:(k + 1) * P])
            if abl == "direct":
                z = self.pred(z0, u)
            elif abl == "copy":
                z = z0 + u
            else:
                z = self.pred(z, u)
            zs.append(z)
        S = torch.stack(zs, 1)                                            # [N, K, D] predicted latents
        aux = {}
        if tgt is not None:
            n = z_all.shape[1]
            pos = j[:, None] + 1 + torch.arange(K, device=j.device)[None]  # [N, K] target patch index
            ok = pos < n
            pc = pos.clamp(max=n - 1)
            ok = ok & (ctx.cnt[b[:, None], pc] > 0)
            t = tgt[b[:, None], pc]                                       # [N, K, D]
            okf = ok.float()[..., None]
            aux["jepa"] = ((S - t).abs() * okf).sum() / (okf.sum() * S.shape[-1]).clamp(min=1)
            valid = (ctx.cnt > 0).view(-1)
            zv = z_all.reshape(-1, z_all.shape[-1])[valid]
            if zv.shape[0] > 4096:
                zv = zv[torch.randperm(zv.shape[0], device=zv.device)[:4096]]
            aux["var"], aux["cov"] = _vicreg(zv)
            aux["zstd"] = tgt.reshape(-1, tgt.shape[-1])[valid].std(0).mean().detach()
        if abl == "zero":
            S = torch.zeros_like(S)
        R = ref.view(N, K, P)
        q = self.decode_head(S + self.ref_proj(R)).view(N, K, P, cfg.nq)
        return q, aux
