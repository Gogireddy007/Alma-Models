"""Round-2 world models and matched direct twins (docs/commodity_r2_prereg.md).

svx  stochastic-volatility WM (round-1 SVWM: latent log-vol AR state with score-driven correction + GRU return
     state, Student-t returns) extended with per-step extra inputs: calendar (Wednesday = EIA weekly petroleum
     report day; sin/cos day of year = seasonal vol) and, optionally, lagged WTI / Brent / Henry Hub daily returns
     (lag >= 1 day). Imagination: sampled returns + latent vol shocks fed back through the same GRU for k = 1..H,
     with the (known ex ante) future calendar; exogenous inputs are unknown in imagination (mask 0).
     Direct twin (abl="direct"): identical modules and parameter count; lead k is one GRU step from the origin state
     with input step_emb[k] + embed(0, mask 0, scale, future calendar_k, exo masked) and a Student-t marginal of the
     cumulative change (scale exp(h_origin + m(d_k)) sqrt(k)); nothing is carried from lead k to k+1.
     Variants: cal=0/1, exo=0/1 (unused inputs are zeroed, the module list is unchanged).
rs, obs  round-1 regime-switching WM and observer-SSM (tools/commodity/wm_models.py), unchanged, univariate.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import wm_core as C          # noqa: E402
import wm_models as WM       # noqa: E402
import core_r2 as R          # noqa: E402

E = 9                         # cal 3 + exo 6
QLT = torch.tensor(R.QL, dtype=torch.float32)


def q_from_samples(sam):
    """[M, B, H] -> [B, 11, H]"""
    return torch.quantile(sam, QLT, dim=0).permute(1, 0, 2)


class SVX(WM.SVWM):
    def __init__(self, cal=1, exo=0, **kw):
        super().__init__(**kw)
        assert self.N == 1
        self.use_cal, self.use_exo = cal, exo
        d_in = self.step_emb.shape[1]
        self.embed = nn.Sequential(nn.Linear(3 + E, d_in), nn.GELU(), nn.Linear(d_in, d_in))
        self.register_buffer("emask", torch.tensor([cal] * 3 + [exo] * 6, dtype=torch.float32))

    def encode_x(self, lp_ctx, ex_ctx):
        x, s, ext = C.features(lp_ctx)
        xe = torch.cat([x, ex_ctx[:, 1:] * self.emask], -1)
        ds, dL = self.gru(self.embed(xe))
        return ds, dL[0], s, x

    def _step_feat(self, z, m, ls, calk):
        B = z.shape[0]
        return torch.cat([z, m, ls, calk * self.emask[:3], torch.zeros(B, 6)], -1)

    def rollout_x(self, dL, hL, Hh, M, ls, calf):
        B = dL.shape[0]
        d, h = dL.repeat(M, 1), hL.repeat(M, 1)
        ls, calf = ls.repeat(M, 1), calf.repeat(M, 1, 1)
        cum = torch.zeros(M * B, 1)
        out = []
        for k in range(Hh):
            mu, kappa, m, beta, gamma, sv, Fm = self.params(d)
            eta = self.sample_eta(mu, Fm)
            z = mu + torch.exp(h) * eta
            cum = cum + z
            out.append(cum[:, 0])
            h = self.h_update(h, eta.clamp(-20, 20), kappa, m, beta, gamma, sv, torch.randn_like(h))
            u = self.embed(self._step_feat(z.clamp(-10, 10), torch.ones_like(z), ls, calf[:, k])) + self.step_emb[k]
            d = self.gru_step(u, d)
        return torch.stack(out, -1).view(M, B, Hh)

    def direct_x(self, dL, hL, Hh, ls, calf):
        B = dL.shape[0]
        outs = []
        for k in range(Hh):
            u = self.embed(self._step_feat(torch.zeros(B, 1), torch.zeros(B, 1), ls, calf[:, k])) + self.step_emb[k]
            dk = self.gru_step(u, dL)
            mu, _, m, _, _, _, Fm = self.params(dk)
            outs.append((mu * math.sqrt(k + 1), torch.exp(hL + m) * math.sqrt(k + 1), Fm))
        return outs

    def loss(self, w, L, Hh, ex):
        ds, dL, s, x = self.encode_x(w[:, :L + 1], ex[:, :L + 1])
        nll, hL = self.filter(ds, x[:, :, :1])
        lo = nll.mean()
        info = {"nll": lo.item()}
        y = (w[:, L + 1:, 0] - w[:, L:L + 1, 0]) / s[:, :1]
        ls, calf = x[:, -1, 2:3], ex[:, L + 1:, :3]
        M = self.M_train
        if self.abl == "none":
            cr = WM.crps_samples(self.rollout_x(dL, hL, Hh, M, ls, calf), y)
        else:
            cr, nl = 0.0, 0.0
            for k, (mu, sc, Fm) in enumerate(self.direct_x(dL, hL, Hh, ls, calf)):
                eta = self.sample_eta(mu.expand(M, -1, -1), Fm)
                sam = mu[..., 0] + sc[..., 0] * eta[..., 0]
                cr = cr + WM.crps_samples(sam[..., None], y[:, k:k + 1]) / Hh
                et = (y[:, k:k + 1] - mu) / sc
                nl = nl - (self.logp(et, torch.log(sc), Fm)).mean() / Hh
            lo = lo + 0.1 * nl
        lo = lo + self.lam * cr
        info["crps"] = float(cr)
        return lo, info

    @torch.no_grad()
    def quantiles_x(self, lp_ctx, ex_ctx, calf, Hh):
        ds, dL, s, x = self.encode_x(lp_ctx, ex_ctx)
        _, hL = self.filter(ds, x[:, :, :1])
        ls = x[:, -1, 2:3]
        M = self.M_eval
        if self.abl == "none":
            sam = self.rollout_x(dL, hL, Hh, M, ls, calf)
        else:
            cols = []
            for mu, sc, Fm in self.direct_x(dL, hL, Hh, ls, calf):
                eta = self.sample_eta(mu.expand(M, -1, -1), Fm)
                cols.append(mu[..., 0] + sc[..., 0] * eta[..., 0])
            sam = torch.stack(cols, -1)
        return q_from_samples(sam * s[None, :, :1])


class Plain(nn.Module):
    """Adapter for the round-1 rs / obs models (no extra inputs; ex ignored)."""

    def __init__(self, arch, **kw):
        super().__init__()
        self.m = WM.ARCHS[arch](**kw)
        self.arch = arch

    def n_params(self):
        return self.m.n_params()

    def loss(self, w, L, Hh, ex):
        return self.m.loss(w, L, Hh)

    @torch.no_grad()
    def quantiles_x(self, lp_ctx, ex_ctx, calf, Hh):
        if self.arch == "obs":
            q9 = self.m.quantiles(lp_ctx, Hh)                    # [B, 9, H]
            nan = torch.full_like(q9[:, :1], float("nan"))
            return torch.cat([nan, q9, nan], 1)
        return q_from_samples(self.m.samples(lp_ctx, Hh))


def build(arch, abl, kw):
    kw = dict(kw)
    kw.setdefault("H", R.H)
    if arch == "svx":
        return SVX(abl=abl, **kw)
    if arch in ("rs", "obs"):
        return Plain(arch, abl=abl, **kw)
    raise SystemExit(arch)
