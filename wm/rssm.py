"""
Family `rssm`: a Dreamer-style recurrent state-space model (DreamerV2/V3 categorical latents) as the world model.

State s_t = (h_t, z_t): a deterministic GRU state h_t [D] and a stochastic categorical state z_t [G x C one-hots].

  encode  e_t = causal Transformer(tokens)_t (N_LAYERS layers, the same layer as gru), then the RSSM *filters* the
          observed patches in order (a recurrence, hence causal):
              h_t   = GRUCell(img_in([z_{t-1}, a_{t-1}]), h_{t-1})      deterministic transition
              prior p(z_t | h_t)                                       what the model imagines patch t looks like
              post  q(z_t | h_t, e_t)                                  what it believes after observing patch t
              z_t ~ q  (straight-through one-hot sample, 1% unimix)
          The action a_{t-1} = step_emb[0] + drive(seasonal-naive patch t as seen from origin t-1), the same
          exogenous "action" the gru reference feeds its rollout.
          The aux loss `kl` (DreamerV3) trains the transition and prior to predict the next posterior:
              kl = BAL * max(FREE, KL(sg(q) || p)) + (1 - BAL) * max(FREE, KL(q || sg(p)))
          per observed transition (t >= 1, patch t-1 not pure padding), mean over (b, t). This is what makes it a
          world model and not a forecaster with a latent bottleneck: the transition is trained by a latent
          prediction objective on every observed patch, not only through the forecast head.
  decode  imagination from the filtered origin state (h_j, z_j): the SAME transition and prior roll K lead patches
              h_k = GRUCell(img_in([z_{k-1}, u_k]), h_{k-1}),  z_k = E_p[z | h_k]  (or a sample, IMAG=sample)
          with u_k = step_emb[k] + drive(ref_k) as in gru. Lead 0 from origin j is exactly the prior step the
          filter takes into patch j+1, so imagination is the filter with observations withheld.
          Quantiles: decode_head(LN([h_k, z_k]) + ref_proj(ref_k)) -> PATCH x NQ.

Why the prior MEAN for imagined leads (IMAG=mean, default): the decoder is trained by pinball loss, which estimates
the quantiles of the target given its input. With a sampled imagined latent it would learn quantiles conditional on
the sample, and averaging a few conditional quantile sets at eval (Vincentisation) under-disperses the predictive
distribution. Feeding the expected categorical state keeps the decoder input deterministic given the origin, so the
quantile head learns marginal quantiles directly, with no train/eval mismatch and no Monte-Carlo noise in the
gradient of every origin. img_in is linear in z, so img_in(E[z]) = E[img_in(z)]: the mean-field step is exact up
to the first nonlinearity. The stochastic latent still matters: the posterior is sampled during filtering (an
information bottleneck regularised by the KL), and the prior is the one-step predictive model the KL trains.
IMAG=sample gives the fully generative variant (prior samples during imagination, quantiles averaged over
EVAL_SAMPLES rollouts at eval).

Evaluation (module in eval mode): the posterior filter is run EVAL_SAMPLES times (posterior samples), imagination
from each, and the quantiles are averaged. Training uses one sample per series. MACs reported by the harness are for
one sample (train mode); eval costs about EVAL_SAMPLES x the RSSM part (the Transformer is shared).

Ablations (TSAR_WM_ABL). The transition is shared between filtering (the encoder) and imagination, so no mode can
delete its parameters without deleting the encoder recurrence; every mode builds the same modules as `none`.
  none    imagination as above: (h, z) carried and rolled from lead k to lead k+1.
  direct  no carried state: each lead is ONE transition step from the origin with its own action,
          h_k = GRUCell(img_in([z_j, u_k]), h_j), z_k = E_p[z | h_k]. Same parameters (0% difference) and the same
          MACs (K transition steps) as none; only the carry between leads is removed.
  copy    no imagination transition: h_k = h_j + u_k, z_k = z_j. The cell, img_in and prior still run in the filter
          and the KL, so the parameter count equals none (reported, not reduced).
  zero    imagined (h_k, z_k) zeroed before the decoder; identical parameters (eval-only on a none checkpoint).

Env keys (all optional):
  TSAR_WM_RSSM_GROUPS        categorical groups G (16)
  TSAR_WM_RSSM_CLASSES       classes per group C (8)          stochastic state = G*C dims, G*log2(C) bits max
  TSAR_WM_RSSM_KL            aux weight of `kl` (0.01; the pinball loss is a per-point mean, ~0.1-0.3)
  TSAR_WM_RSSM_FREE          free nats per transition (1.0)
  TSAR_WM_RSSM_BAL           KL balance: weight on the prior-training term (0.8)
  TSAR_WM_RSSM_UNIMIX        uniform mix in the categorical probabilities (0.01)
  TSAR_WM_RSSM_IMAG          mean | sample (mean)
  TSAR_WM_RSSM_EVAL_SAMPLES  posterior samples averaged at eval (2; 4 would put eval MACs above 2x gru)

Width ladder: every module scales like d^2 (the stochastic state is fixed at G*C = 128), so the d_model ladder of
docs/wm_families.md applies; see the TSAR_WM_INFO lines in the screen report.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.base import WMFamily

_e = os.environ.get
GROUPS = int(_e("TSAR_WM_RSSM_GROUPS", 16))
CLASSES = int(_e("TSAR_WM_RSSM_CLASSES", 8))
KL_W = float(_e("TSAR_WM_RSSM_KL", 0.01))
FREE = float(_e("TSAR_WM_RSSM_FREE", 1.0))
BAL = float(_e("TSAR_WM_RSSM_BAL", 0.8))
UNIMIX = float(_e("TSAR_WM_RSSM_UNIMIX", 0.01))
IMAG = _e("TSAR_WM_RSSM_IMAG", "mean")
EVAL_SAMPLES = int(_e("TSAR_WM_RSSM_EVAL_SAMPLES", 2))
if IMAG not in ("mean", "sample"):
    raise SystemExit(f"TSAR_WM_RSSM_IMAG={IMAG!r} must be mean or sample")


class RSSMFamily(WMFamily):
    name = "rssm"
    aux_weights = {"kl": KL_W}
    env_keys = ("TSAR_WM_RSSM_GROUPS", "TSAR_WM_RSSM_CLASSES", "TSAR_WM_RSSM_KL", "TSAR_WM_RSSM_FREE",
                "TSAR_WM_RSSM_BAL", "TSAR_WM_RSSM_UNIMIX", "TSAR_WM_RSSM_IMAG", "TSAR_WM_RSSM_EVAL_SAMPLES")

    def __init__(self, cfg):
        super().__init__(cfg)
        D, P, NQ = cfg.d_model, cfg.patch, cfg.nq
        S = GROUPS * CLASSES
        self.S = S
        self.embed = nn.Sequential(nn.Linear(cfg.tok_dim, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patch, D))
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.register_buffer("causal", torch.triu(torch.full((cfg.n_patch, cfg.n_patch), float("-inf")), 1))
        # transition (shared by filtering and imagination)
        self.img_in = nn.Sequential(nn.Linear(S + D, D), nn.GELU())
        self.cell = nn.GRUCell(D, D)
        self.prior = nn.Linear(D, S)
        self.post = nn.Linear(2 * D, S)
        self.step_emb = nn.Parameter(torch.zeros(cfg.k_steps, D))
        self.drive = nn.Linear(P, D)
        nn.init.zeros_(self.drive.weight); nn.init.zeros_(self.drive.bias)
        self.decode_head = nn.Sequential(nn.LayerNorm(D + S), nn.Linear(D + S, P * NQ))
        self.ref_proj = nn.Linear(P, D + S)

    def param_groups(self):
        return {"transition": nn.ModuleList([self.img_in, self.cell, self.prior]), "posterior": self.post}

    # --- categorical latent helpers -------------------------------------------------------------------------
    def _probs(self, logits):
        p = logits.view(*logits.shape[:-1], GROUPS, CLASSES).softmax(-1)
        return (1 - UNIMIX) * p + UNIMIX / CLASSES                     # [..., G, C]

    def _sample(self, probs):
        """Straight-through one-hot sample (Gumbel-max on the unimixed probabilities)."""
        g = -torch.log(-torch.log(torch.rand_like(probs).clamp(1e-9, 1 - 1e-9)))
        idx = (probs.log() + g).argmax(-1)
        hard = F.one_hot(idx, CLASSES).to(probs.dtype)
        return (hard + probs - probs.detach()).flatten(-2)

    def _step(self, h, z, a):
        return self.cell(self.img_in(torch.cat([z, a], -1)), h)

    # --- interface ------------------------------------------------------------------------------------------
    def encode(self, ctx):
        cfg, P = self.cfg, self.cfg.patch
        B, n = ctx.tok.shape[:2]
        e = self.encoder(self.embed(ctx.tok) + self.pos[:, :n], mask=self.causal[:n, :n])   # [B, n, D]
        M = 1 if self.training else EVAL_SAMPLES
        ref, cnt = ctx.ref, ctx.cnt
        if M > 1:
            e, ref, cnt = e.repeat(M, 1, 1), ref.repeat(M, 1, 1), cnt.repeat(M, 1)
        act = self.step_emb[0] + self.drive(ref[:, :, :P])           # [MB, n, D]: action for transition t -> t+1
        h = e.new_zeros(e.shape[0], cfg.d_model)
        z = e.new_zeros(e.shape[0], self.S)
        hs, zs, pri, pos = [], [], [], []
        for t in range(n):
            a = act[:, t - 1] if t > 0 else self.step_emb[0].expand_as(h)
            h = self._step(h, z, a)
            qp = self._probs(self.post(torch.cat([h, e[:, t]], -1)))
            z = self._sample(qp)
            hs.append(h); zs.append(z); pos.append(qp)
            if self.training:
                pri.append(self._probs(self.prior(h)))
        H = torch.stack(hs, 1).view(M, B, n, -1)
        Z = torch.stack(zs, 1).view(M, B, n, -1)
        kl = None
        if self.training and n > 1:
            q = torch.stack(pos[1:], 1)                                # [B, n-1, G, C]
            p = torch.stack(pri[1:], 1)
            lq, lp = q.log(), p.log()
            dyn = (q.detach() * (lq.detach() - lp)).sum((-1, -2))      # trains the prior / transition
            rep = (q * (lq - lp.detach())).sum((-1, -2))               # keeps the posterior predictable
            per = BAL * dyn.clamp(min=FREE) + (1 - BAL) * rep.clamp(min=FREE)
            m = (cnt[:, :-1] > 0).to(per.dtype)
            kl = (per * m).sum() / m.sum().clamp(min=1)
        return {"h": H, "z": Z, "kl": kl}

    def decode(self, enc, ctx, b, j):
        cfg, abl = self.cfg, self.cfg.abl
        P, K, NQ = cfg.patch, cfg.k_steps, cfg.nq
        M, N = enc["h"].shape[0], len(b)
        h0 = h = enc["h"][:, b, j].reshape(M * N, -1)
        z0 = z = enc["z"][:, b, j].reshape(M * N, -1)
        ref = ctx.ref[b, j].repeat(M, 1)                               # [MN, MAX_H]
        feats = []
        for k in range(K):
            rk = ref[:, k * P:(k + 1) * P]
            u = self.step_emb[k] + self.drive(rk)
            if abl == "copy":
                hk, zk = h0 + u, z0
            else:
                hk = self._step(h0, z0, u) if abl == "direct" else self._step(h, z, u)
                pk = self._probs(self.prior(hk))
                zk = self._sample(pk) if IMAG == "sample" else pk.flatten(-2)
                h, z = hk, zk
            f = torch.cat([hk, zk], -1)
            if abl == "zero":
                f = torch.zeros_like(f)
            feats.append(self.decode_head(f + self.ref_proj(rk)).view(M * N, P, NQ))
        q = torch.stack(feats, 1).view(M, N, K, P, NQ).mean(0)
        aux = {"kl": enc["kl"]} if enc["kl"] is not None else {}
        return q, aux
