"""
Family `koopman`: a Koopman world model. The latent evolves LINEARLY under a learned operator with a stable complex
spectrum (damped oscillators), and the rollout is powers of that operator.

Why it is a world model and not a GRU tweak: the nonlinearity lives only in the lift (encoder: observations ->
Koopman observables) and the inverse lift (decoder). Between them the state is a set of M independent complex modes
z^(m) that each rotate and decay, z_{k}^(m) = lambda_m^(k+1) z_0^(m) + forced response. Every forecast is therefore a
sum of damped sinusoids in observable space whose frequencies can lock onto daily / weekly cycles, and an aux loss
asks the same operator to explain how the encoder's OWN observables move from patch j to patch j+k+1 on observed
data (latent linearity / consistency), so the transition is trained as a model of the world, not only as a readout.

  encode  embed patch tokens, + positional, N_LAYERS causal Transformer layers (no recurrent state RNN: the only
          dynamics in this family is the linear operator) -> h_j; observables z_j = to_obs(h_j) in R^{2M}, read as
          M complex numbers (re | im halves).
  spectrum lambda_m = r_m e^{i w_m}, r_m = exp(-softplus(rho_m)) < 1 always (stable), w_m learned. Init: M/8 pure
          decay modes (w = 0, level / trend), the rest with periods log-spaced over 2..64 patches (16..512 points,
          which covers the hourly day = 3 patches and week = 21 patches); radii spread over 0.6..0.995 per patch.
  input conditioning (TSAR_WM_KOOPMAN_MOD=1, default): a head on h_j shifts each mode per series and origin,
          rho += d_rho(h_j), w += 0.5 tanh(d_w(h_j)) (zero-initialised), so the operator can tune its frequencies and
          damping to the series it is looking at. |lambda| <= 1 still holds.
  decode  lead k (patch j+k+1): s_k = Lambda^(k+1) z_j  +  sum_{i<=k} Lambda^(k-i) B u_i, u_i = step_emb[i] +
          drive(ref_i). The free response is computed in closed form (r^(k+1), (k+1) w); the forced response by the
          equivalent linear recurrence f_k = Lambda f_{k-1} + B u_k. Decoder (inverse lift): LN, Linear(2M, D), GELU,
          Linear(D, PATCH*NQ) on s_k + ref_proj(ref_k). The harness adds the per-quantile ref skip.
  aux     lin = || s_k - sg(z_{j+k+1}) ||^2 / mean ||sg(z)||^2 over training origins whose patch j+k+1 is inside the
          window (the k-step Koopman identity g(x_{t+k}) = K^k g(x_t) on observed data; stop-gradient on the target
          to avoid collapse). Applied to the rolled states of every ablation mode, so all modes share one objective.
          Known caveat: z_{j+k+1} is computed from tokens in frame j+k+1 and s_k predicts in frame j (NORM=last
          shifts loc each patch), so the identity is only approximate; the weight is small (0.05).

Ablations (TSAR_WM_ABL):
  none    the Koopman rollout above.
  direct  no carried state, no operator: s_k = z_j + MLP(LN[z_j, u_k]) with MLP 4M -> H -> 2M (zero-init last), H
          chosen so its params match the transition (rho, w, B, modulation head) within 3% (65,007 vs 64,992 at width 1).
  copy    no transition: s_k = z_j + u_k (rho, w, B and the modulation head are not built).
  zero    rolled states zeroed before the decoder (same parameters as none).

Env keys: TSAR_WM_KOOPMAN_MODES (complex modes M, default d_model), TSAR_WM_KOOPMAN_LIN (aux weight, default 0.05;
0 switches the aux off), TSAR_WM_KOOPMAN_MOD (1 = per-series spectrum modulation, default 1).
Params at width 1 (d_model 96, M 96): none/zero 353,457, direct 353,472, copy 288,465 (including ref_skip).
Params scale ~d^2 like gru (the encoder dominates), so the gru width ladder applies.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily

_LIN = float(os.environ.get("TSAR_WM_KOOPMAN_LIN", "0.05"))


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


class KoopmanFamily(WMFamily):
    name = "koopman"
    aux_weights = {"lin": _LIN}
    env_keys = ("TSAR_WM_KOOPMAN_MODES", "TSAR_WM_KOOPMAN_LIN", "TSAR_WM_KOOPMAN_MOD")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P, NQ, K = cfg.d_model, cfg.patch, cfg.nq, cfg.k_steps
        M = int(os.environ.get("TSAR_WM_KOOPMAN_MODES", D))
        self.M, Z = M, 2 * M
        self.use_mod = int(os.environ.get("TSAR_WM_KOOPMAN_MOD", "1")) == 1
        # lift: causal Transformer -> observables
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        self.to_obs = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, Z))
        # lead inputs (exogenous "actions"), in observable space
        self.step_emb = nn.Parameter(torch.zeros(K, Z))
        self.drive = nn.Linear(P, Z)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        # transition: the Koopman operator (none / zero), or its matched replacement (direct)
        if cfg.abl in ("none", "zero"):
            n_dc = max(1, M // 8)
            radius = torch.linspace(0.6, 0.995, M)
            period = torch.exp(torch.linspace(math.log(2.0), math.log(64.0), M - n_dc))
            omega = torch.cat([torch.zeros(n_dc), 2 * math.pi / period])
            radius = radius[torch.randperm(M, generator=torch.Generator().manual_seed(0))]
            self.rho = nn.Parameter(torch.log(torch.expm1(-torch.log(radius))))
            self.omega = nn.Parameter(omega)
            self.B = nn.Linear(Z, Z, bias=False)
            with torch.no_grad():
                self.B.weight.copy_(torch.eye(Z))
            if self.use_mod:
                self.mod = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, Z))
                nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)
        elif cfg.abl == "direct":
            t = M + M + Z * Z + ((D * D + D + D * Z + Z) if self.use_mod else 0)
            H = max(8, round((t - 2 * Z - 2 * Z - Z) / (2 * Z + 1 + Z)))     # LN(2Z) + (2Z+1)H + (H+1)Z
            self.direct = nn.Sequential(nn.LayerNorm(2 * Z), nn.Linear(2 * Z, H), nn.GELU(), nn.Linear(H, Z))
            nn.init.zeros_(self.direct[-1].weight); nn.init.zeros_(self.direct[-1].bias)
        # inverse lift
        self.decode_head = nn.Sequential(nn.LayerNorm(Z), nn.Linear(Z, D), nn.GELU(), nn.Linear(D, P * NQ))
        self.ref_proj = nn.Linear(P, Z)

    def param_groups(self):
        a = self.cfg.abl
        if a in ("none", "zero"):
            mods = [self.B] + ([self.mod] if self.use_mod else [])
            g = nn.ModuleList(mods)
            g.spec = nn.ParameterList([self.rho, self.omega])
            return {"transition": g}
        return {"transition": self.direct if a == "direct" else nn.Identity()}

    def encode(self, ctx):
        n = ctx.tok.shape[1]
        h = self.encoder(self.embed(ctx.tok) + self.pos[:, :n], mask=self.causal[:n, :n])
        return h, self.to_obs(h)                                          # [B, n, D], [B, n, 2M]

    def spectrum(self, h0):
        """Per-origin log-radius (negative) and angle per mode, [N, M] each."""
        rho, om = self.rho, self.omega
        if self.use_mod:
            d = self.mod(h0)
            rho = rho + d[:, :self.M]
            om = om + 0.5 * torch.tanh(d[:, self.M:])
        else:
            rho, om = rho.expand(h0.shape[0], -1), om.expand(h0.shape[0], -1)
        return -F.softplus(rho), om

    def decode(self, enc, ctx, b, j):
        cfg, abl, M = self.cfg, self.cfg.abl, self.M
        P, N, K = cfg.patch, len(b), cfg.k_steps
        H, Zall = enc
        z0 = Zall[b, j]                                                   # [N, 2M]
        ref = ctx.ref[b, j].view(N, K, P)
        U = self.step_emb[None] + self.drive(ref)                         # [N, K, 2M]
        if abl in ("none", "zero"):
            logr, om = self.spectrum(H[b, j])
            kk = torch.arange(1, K + 1, device=z0.device, dtype=z0.dtype)[None, :, None]
            mag = torch.exp(logr[:, None] * kk)                           # closed-form powers |lambda|^(k+1)
            pr, pi = mag * torch.cos(om[:, None] * kk), mag * torch.sin(om[:, None] * kk)
            fr, fi = _cmul(pr, pi, z0[:, None, :M], z0[:, None, M:])     # free response Lambda^(k+1) z0
            BU = self.B(U)
            lr, li = pr[:, 0], pi[:, 0]                                   # lambda
            gr = torch.zeros_like(z0[:, :M]); gi = torch.zeros_like(gr)
            forced = []
            for k in range(K):                                            # forced response, linear recurrence
                gr, gi = _cmul(lr, li, gr, gi)
                gr, gi = gr + BU[:, k, :M], gi + BU[:, k, M:]
                forced.append(torch.cat([gr, gi], -1))
            S = torch.cat([fr, fi], -1) + torch.stack(forced, 1)          # [N, K, 2M]
        elif abl == "direct":
            S = z0[:, None] + self.direct(torch.cat([z0[:, None].expand_as(U), U], -1))
        else:                                                             # copy
            S = z0[:, None] + U
        aux = {}
        if self.training and _LIN > 0:
            n = Zall.shape[1]
            tj = j[:, None] + torch.arange(1, K + 1, device=j.device)[None]   # patch j+k+1
            ok = tj < n
            tgt = Zall.detach()[b[:, None].expand_as(tj), tj.clamp(max=n - 1)]
            if ok.any():
                err = ((S - tgt) ** 2).sum(-1)[ok].mean()
                aux["lin"] = err / ((tgt[ok] ** 2).sum(-1).mean() + 1e-6)
        if abl == "zero":
            S = torch.zeros_like(S)
        q = self.decode_head(S + self.ref_proj(ref)).view(N, K, P, cfg.nq)
        return q, aux
