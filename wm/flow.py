"""
Family `flow`: a generative world model that imagines the future patch by patch with a conditional rectified-flow
sampler, and absorbs each imagined patch into a carried latent state (Lindqvist's train_flow.py, redone).

Why it is a world model (and why it differs from train_flow.py)
  train_flow.py put the flow on the LATENT transition (MeanFlow over z_{k+1} | z_k): the target was the moving
  encoder output, the pinball head decoded one detached sampled path, and it scored 0.85-0.96 (discarded). Its
  DV-Flow head (dvf, 0.730-0.735) sampled the whole 64-step path in one shot, with no state in the loop. Here the
  flow generates OBSERVATIONS (one lead patch, PATCH values, in the origin's model space) conditioned on the
  current latent state, and the state is then updated by the observation it imagined:

      s_0 = z_j                                              (causal encoder, identical to family gru)
      c_k = s_k + u_k,  u_k = step_emb[k] + drive(ref_k)     (lead embedding + seasonal-naive "action")
      y_k ~ p(y_k | c_k):  x_0 = eps ~ N(0, I),  x <- x + v(x, tau, c_k) / E,  E Euler steps of a velocity MLP
      s_{k+1} = GRUCell(obs_emb([y_k, mask_k]), s_k)          (the state absorbs what it imagined)

  This is the TimeGrad / DIAMOND / GameNGen loop (generate the next observation with a conditional generative
  model, feed it back) in a patch latent space. Rolling S such paths from one origin gives S self-consistent
  imagined futures; their per-lead marginals are the forecast distribution.

  Training: the velocity net is trained by conditional flow matching (aux "fm"), with the state TEACHER-FORCED on
  the true future patches (s_{k+1} = GRUCell(obs_emb(y_true_k, mask), s_k)), so v learns p(y_k | y_<k, context),
  the autoregressive factorisation. The teacher-forced states are used only in the aux loss: the quantiles q that
  the harness scores are always computed from the model's own imagination (free-running), so q at origin j reads
  only patches 0..j (wm_causal) and there is no train/eval exposure mismatch in q. FM uses a random subset of at
  most TSAR_WM_FLOW_FMN origins per batch (memory).

Quantile output (TSAR_WM_FLOW_Q)
  samp  q = empirical quantiles (type-7 interpolation over sorted samples) of S imagined paths per origin, S =
        TSAR_WM_FLOW_S in training and evaluation alike (the same estimator in both, so no quantile-estimator bias
        shift between them). The harness pinball loss back-propagates through the sort and the E Euler steps into
        the sampler (reparameterised samples; sample-quantile pinball is a CRPS-like proper score as S grows), and
        FM keeps v a velocity field of the conditional law. No quantile head is built.
  head  one deterministic "central" imagination (x_0 = 0, the noise mean) is rolled and fed back, and a quantile
        head reads the state: q_k = head(c_k + ref_proj(ref_k)) (gru's decoder). The flow is then the transition's
        generative regulariser. Cheap (one path), but the uncertainty lives in a head, not in the samples.

Determinism: training noise comes from torch's global RNG (seeded by the harness). Evaluation noise is one fixed
  [S, K, PATCH] tensor drawn from a torch.Generator seeded by TSAR_SEED (antithetic pairs), broadcast over series:
  forecasts are deterministic and independent of batch composition (fixes audit v3.1 F-1 for this family).

Ablations (TSAR_WM_ABL)
  none    as above: carried state, imagined patch fed back through the GRUCell.
  direct  no carried state: s_k = z_0 + MLP(LN[z_0, u_k]) (2d -> 2d -> d, zero-init last layer, gru's W1). Each
          lead is still sampled by the flow, but independently of the imagined past. obs_emb is not built. Params
          6d^2 + 7d + 4d vs GRUCell + obs_emb 6d^2 + 6d + 17d (-0.4% total).
  copy    no transition: s_k = z_0 (c_k = z_0 + u_k); GRUCell and obs_emb not built (fewer params, see report).
  zero    the rolled states s_k are zeroed before the decoders (the velocity net's condition and, in head mode,
          the quantile head), so only u_k (step, ref drive) and the ref paths remain. Same parameters as none.

Env keys (all optional)
  TSAR_WM_FLOW_Q      samp | head (default samp)
  TSAR_WM_FLOW_S      sample paths per origin, train and eval (default 16; samp only)
  TSAR_WM_FLOW_EULER  Euler steps per lead patch, train and eval (default 2)
  TSAR_WM_FLOW_FMW    weight of the flow-matching aux loss (default 0.3)
  TSAR_WM_FLOW_FMN    max origins per batch in the flow-matching loss (default 1024)
  TSAR_WM_FLOW_QN     samp, training: max origins per batch whose pinball gradient flows through the sampler
                      (default 512). The other origins' quantiles are computed under no_grad (the logged loss stays
                      the full-batch pinball); the subset's gradient is scaled by N / QN (value unchanged), so the
                      pinball gradient is an unbiased estimate of the full-batch one. Memory: S paths x QN origins.

Width: the velocity MLP hidden size is d_model, so every part scales like d^2 with TSAR_WM_WIDTH (gru's ladder).
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily, refuse

_env = os.environ.get
Q_MODE = _env("TSAR_WM_FLOW_Q", "samp")
N_SAMP = int(_env("TSAR_WM_FLOW_S", 16))
EULER = int(_env("TSAR_WM_FLOW_EULER", 2))
FM_W = float(_env("TSAR_WM_FLOW_FMW", 0.3))
FM_N = int(_env("TSAR_WM_FLOW_FMN", 1024))
Q_N = int(_env("TSAR_WM_FLOW_QN", 512))
EVAL_SEED = int(_env("TSAR_SEED", 0))          # harness seed (read only; not a family key)


class VelocityNet(nn.Module):
    """v(x_tau, tau, c): PATCH -> PATCH velocity, conditioned additively on the state (computed once per lead)."""

    def __init__(self, P, D, H):
        super().__init__()
        self.inx = nn.Linear(P, H)
        self.int = nn.Linear(8, H, bias=False)
        self.inc = nn.Linear(D, H, bias=False)
        self.mid = nn.Linear(H, H)
        self.out = nn.Linear(H, P)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("freq", torch.tensor([1.0, 2.0, 4.0, 8.0]) * math.pi, persistent=False)

    def cond(self, c):
        return self.inc(c)

    def forward(self, x, tau, hc):
        a = tau[..., None] * self.freq                                   # tau: [R] -> [R, 4]
        h = F.gelu(self.inx(x) + self.int(torch.cat([a.sin(), a.cos()], -1)) + hc)
        return self.out(F.gelu(self.mid(h)))


class FlowFamily(WMFamily):
    name = "flow"
    aux_weights = {"fm": FM_W}
    env_keys = ("TSAR_WM_FLOW_Q", "TSAR_WM_FLOW_S", "TSAR_WM_FLOW_EULER", "TSAR_WM_FLOW_FMW", "TSAR_WM_FLOW_FMN",
                "TSAR_WM_FLOW_QN")

    def __init__(self, cfg):
        super().__init__(cfg)
        if Q_MODE not in ("samp", "head"):
            raise SystemExit(f"TSAR_WM_FLOW_Q={Q_MODE!r} not in (samp, head)")
        if N_SAMP < 2 or EULER < 1 or FM_N < 1 or Q_N < 1:
            raise SystemExit("TSAR_WM_FLOW_S >= 2 and TSAR_WM_FLOW_EULER, _FMN, _QN >= 1 required")
        D, P, NQ, abl = cfg.d_model, cfg.patch, cfg.nq, cfg.abl
        # encoder: identical to family gru (same module order, so the same init at the same seed)
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.lower = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.state_rnn = nn.GRU(D, D, batch_first=True)
        self.upper = nn.TransformerEncoder(layer, cfg.n_layers - 1, enable_nested_tensor=False) if cfg.n_layers > 1 else None
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        self.to_state = nn.Linear(D, D)
        # generative transition
        carried = abl in ("none", "zero")
        self.dynamics = nn.GRUCell(D, D) if carried else None
        self.obs_emb = nn.Linear(2 * P, D) if carried else None
        self.step_emb = nn.Parameter(torch.zeros(cfg.k_steps, D))
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.vel = VelocityNet(P, D, D)
        if Q_MODE == "head":
            self.decode_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, P * NQ))
            self.ref_proj = nn.Linear(P, D)
        if abl == "direct":
            self.direct = nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
            nn.init.zeros_(self.direct[-1].weight); nn.init.zeros_(self.direct[-1].bias)
        qs = torch.linspace(0.1, 0.9, NQ) * (N_SAMP - 1)                 # type-7 sample-quantile positions
        self.register_buffer("q_lo", qs.floor().long(), persistent=False)
        self.register_buffer("q_w", qs - qs.floor(), persistent=False)
        self._eps = None

    def param_groups(self):
        if self.dynamics is not None:
            trans = nn.ModuleList([self.dynamics, self.obs_emb])
        else:
            trans = getattr(self, "direct", nn.Identity())
        return {"transition": trans, "velocity": self.vel}

    # -- encoder (gru's) ----------------------------------------------------------------------------------------
    def encode(self, ctx):
        n = ctx.tok.shape[1]
        e = self.embed(ctx.tok) + self.pos[:, :n]
        h = self.lower(e, mask=self.causal[:n, :n])
        state, _ = self.state_rnn(h)
        h = self.upper(h + state, mask=self.causal[:n, :n]) if self.upper is not None else h + state
        return torch.tanh(self.to_state(h + state))

    # -- pieces -------------------------------------------------------------------------------------------------
    def _lead_inputs(self, ref):
        P, K = self.cfg.patch, self.cfg.k_steps
        r = ref.view(len(ref), K, P)
        return self.step_emb + self.drive(r), r                          # [N, K, D], [N, K, P]

    def _state(self, k, s, z0, u_k):
        """State s_k used at lead k (before the decoders); s carries the rolled state for none/zero."""
        abl = self.cfg.abl
        if abl == "direct":
            return z0 + self.direct(torch.cat([z0, u_k], -1))
        if abl == "copy":
            return z0
        return s

    def _advance(self, s, y, m):
        if self.dynamics is None:
            return s
        return self.dynamics(self.obs_emb(torch.cat([y, m], -1)), s)

    def _sample(self, x, hc):
        E = EULER
        for i in range(E):
            tau = x.new_full((len(x),), i / E)
            x = x + self.vel(x, tau, hc) / E
        return x

    def _eval_noise(self, device):
        K, P, S = self.cfg.k_steps, self.cfg.patch, N_SAMP
        if self._eps is None or self._eps.device != device:
            g = torch.Generator().manual_seed(1_000_003 * (EVAL_SEED + 1))
            half = torch.randn(S // 2 + S % 2, K, P, generator=g)
            self._eps = torch.cat([half, -half], 0)[:S].to(device)       # antithetic pairs
        return self._eps

    def _imagine(self, z0, u, n_paths, noise):
        """Free-running rollout of n_paths per origin. z0 [N, D], u [N, K, D], noise [N*n_paths, K, P] or None
        (central path, x_0 = 0). Returns imagined patches [N*n_paths, K, P] and conditions c_k [N*n_paths, K, D]."""
        K, P = self.cfg.k_steps, self.cfg.patch
        z0 = z0.repeat_interleave(n_paths, 0)
        u = u.repeat_interleave(n_paths, 0)
        s, ys, cs = z0, [], []
        ones = z0.new_ones(len(z0), P)
        for k in range(K):
            sk = self._state(k, s, z0, u[:, k])
            if self.cfg.abl == "zero":
                sk = torch.zeros_like(sk)
            c = sk + u[:, k]
            x0 = noise[:, k] if noise is not None else z0.new_zeros(len(z0), P)
            y = self._sample(x0, self.vel.cond(c))
            ys.append(y); cs.append(c)
            if k + 1 < K and self.dynamics is not None and self.cfg.abl == "none":
                s = self._advance(s, y.clamp(-20, 20), ones)
        return torch.stack(ys, 1), torch.stack(cs, 1)

    def _fm_loss(self, enc, ctx, b, j, u):
        """Conditional flow matching on teacher-forced states, a random subset of origins."""
        K, P = self.cfg.k_steps, self.cfg.patch
        n = ctx.tok.shape[1]
        if ctx.x.shape[1] < n * P:
            return None
        idx = torch.randperm(len(b), device=b.device)[:FM_N] if len(b) > FM_N else torch.arange(len(b), device=b.device)
        bb, jj = b[idx], j[idx]
        pad = torch.full((ctx.x.shape[0], K * P), float("nan"), device=ctx.x.device)
        fut = torch.cat([ctx.x, pad], 1).unfold(1, K * P, P)[:, 1:n + 1][bb, jj]          # [M, K*P], harness targets
        loc, scale = ctx.loc[bb, jj, None], ctx.scale[bb, jj, None]
        m = ~torch.isnan(fut)
        y = torch.asinh((torch.nan_to_num(fut) - loc) / scale).clamp(-20, 20) * m
        y, m = y.view(-1, K, P), m.float().view(-1, K, P)
        z0, uu = enc[bb, jj], u[idx]
        s, num, den = z0, 0.0, 0.0
        for k in range(K):
            sk = self._state(k, s, z0, uu[:, k])
            if self.cfg.abl == "zero":
                sk = torch.zeros_like(sk)
            c = sk + uu[:, k]
            eps = torch.randn_like(y[:, k])
            tau = torch.rand(len(y), device=y.device)
            xt = (1 - tau[:, None]) * eps + tau[:, None] * y[:, k]
            v = self.vel(xt, tau, self.vel.cond(c))
            num = num + (((v - (y[:, k] - eps)) ** 2) * m[:, k]).sum()
            den = den + m[:, k].sum()
            if k + 1 < K and self.dynamics is not None and self.cfg.abl == "none":
                s = self._advance(s, y[:, k], m[:, k])
        return num / den.clamp(min=1)

    def _samp_quantiles(self, z0, u, noise):
        """Type-7 empirical quantiles of S imagined paths per origin; noise [N, S, K, P] -> q [N, K, P, NQ]."""
        N, S, K, P = noise.shape
        y, _ = self._imagine(z0, u, S, noise.reshape(N * S, K, P))
        ys = y.view(N, S, K, P).sort(1).values
        lo = ys[:, self.q_lo]                                              # [N, NQ, K, P]
        hi = ys[:, (self.q_lo + 1).clamp(max=S - 1)]
        w = self.q_w.view(1, -1, 1, 1)
        return (lo * (1 - w) + hi * w).permute(0, 2, 3, 1)

    # -- decode -------------------------------------------------------------------------------------------------
    def decode(self, enc, ctx, b, j):
        cfg = self.cfg
        K, P, NQ, N = cfg.k_steps, cfg.patch, cfg.nq, len(b)
        z0 = enc[b, j]
        u, r = self._lead_inputs(ctx.ref[b, j])
        learning = self.training and torch.is_grad_enabled()
        if Q_MODE == "samp":
            S = N_SAMP
            if not learning:
                q = self._samp_quantiles(z0, u, self._eval_noise(z0.device).expand(N, -1, -1, -1))
            elif N <= Q_N:
                q = self._samp_quantiles(z0, u, torch.randn(N, S, K, P, device=z0.device))
            else:                       # pinball gradient through the sampler from a random subset of Q_N origins
                noise = torch.randn(N, S, K, P, device=z0.device)
                perm = torch.randperm(N, device=z0.device)
                sub, rest = perm[:Q_N], perm[Q_N:]
                qs = self._samp_quantiles(z0[sub], u[sub], noise[sub])
                g = N / Q_N                                               # unbiased full-batch gradient, same value
                qs = qs * g - qs.detach() * (g - 1)
                with torch.no_grad():
                    qr = self._samp_quantiles(z0[rest], u[rest], noise[rest])
                q = torch.cat([qs, qr], 0)[torch.argsort(perm)]
        else:
            _, c = self._imagine(z0, u, 1, None)
            ref_h = self.ref_proj(r)
            q = self.decode_head(c + ref_h).view(N, K, P, NQ)
        aux = {}
        if learning:
            fm = self._fm_loss(enc, ctx, b, j, u)
            if fm is not None:
                aux["fm"] = fm
        return q, aux
