"""Alma's pretrained SSM world model (wm/ssm.py, ckpt data/ckpt/wm_ssm_none_s0.pt) fine-tuned on commodity data.

The pretrained checkpoint was trained on 14 non-commodity datasets (dominick, ercot, m5, monash_*, nn5; see its cfg)
and synthetic series, so it has never seen an oil price. The context maths of train_wm.py (NORM=last, FRAME=1,
FRAMEFIX=0, SREF_AVG=1, PATCH=8, CONTEXT=256, MAX_H=64) is re-implemented here verbatim (train_wm.py runs its whole
training script on import, so it cannot be imported). wm/ssm.py is imported read-only.

abl="none": the SSM state carried through the 8 lead patches (Alma's world model).
abl="direct": every lead patch is ONE step of the same SSM from the origin state (identical parameters; the
matched twin defined in wm/ssm.py), initialised from data/ckpt/wm_ssm_direct_s0.pt (fixed after the audit; the
r1-r3 alma_ft_direct / alma_zs_direct rows started from the rollout weights and are NOT valid twins).

Model input = log price (any positive series works: the frame is translation/scale normalised); quantiles() returns
the change of the input from its last value, so the caller treats it like the other wm_* models.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from wm.base import WMConfig, Context, to_model_space          # noqa: E402
from wm.ssm import SSMFamily                                    # noqa: E402

PATCH, CONTEXT, MAX_H, NQ = 8, 256, 64, 9
K_STEPS = MAX_H // PATCH
QT = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
CKPT0 = os.path.join(ROOT, "data", "ckpt", "wm_ssm_none_s0.pt")
CKPT0_DIRECT = os.path.join(ROOT, "data", "ckpt", "wm_ssm_direct_s0.pt")   # the direct twin's own pretrained weights


def causal_stats(x):
    obs = ~torch.isnan(x)
    v = torch.where(obs, x, torch.zeros_like(x))
    ends = torch.arange(PATCH - 1, x.shape[1], PATCH)
    cnt = obs.float().cumsum(1)[:, ends]
    s1 = v.cumsum(1)[:, ends]
    s2 = (v * v).cumsum(1)[:, ends]
    mean = s1 / cnt.clamp(min=1)
    std = (s2 / cnt.clamp(min=1) - mean * mean).clamp(min=0).sqrt()
    idx = torch.where(obs, torch.arange(x.shape[1]).expand_as(x), torch.zeros_like(x, dtype=torch.long))
    last_idx = idx.cummax(1).values
    ffill = torch.gather(v, 1, last_idx)
    loc = ffill[:, ends]
    both = obs[:, 1:] & obs[:, :-1]
    d = torch.where(both, (x[:, 1:] - x[:, :-1]).abs(), torch.zeros_like(x[:, 1:]))
    d = torch.cat([torch.zeros_like(x[:, :1]), d], 1).cumsum(1)[:, ends]
    nd = torch.cat([torch.zeros_like(x[:, :1]), both.float()], 1).cumsum(1)[:, ends]
    scale = d / nd.clamp(min=1)
    scale = torch.where(scale > 1e-6, scale, std)
    scale = torch.where(scale > 1e-6, scale, torch.where(loc.abs() > 1e-6, loc.abs(), torch.ones_like(loc)))
    return loc, scale, cnt


def frame_features(loc, scale, cnt):
    lp = torch.cat([loc[:, :1], loc[:, :-1]], 1)
    sp = torch.cat([scale[:, :1], scale[:, :-1]], 1)
    dloc = torch.asinh((loc - lp) / scale)
    dls = torch.log(scale / sp).clamp(-5, 5)
    lc = torch.log1p(cnt) / 5
    return [dloc[..., None], dls[..., None], lc[..., None]]


def seasonal_ref(x, loc, scale, m):
    B, L = x.shape
    off = L - loc.shape[1] * PATCH
    obs = ~torch.isnan(x)
    ar = torch.arange(L)
    last = torch.where(obs, ar.expand_as(x), torch.full_like(x, -1, dtype=torch.long)).cummax(1).values
    ffill = torch.gather(torch.nan_to_num(x), 1, last.clamp(min=0))
    ends = torch.arange(off + PATCH - 1, L, PATCH)
    k = torch.arange(1, MAX_H + 1)
    mm = m.clamp(min=1)[:, None, None]
    lag = mm * torch.div(k[None, None, :] + mm - 1, mm, rounding_mode="floor")
    idx = ends[None, :, None] + k[None, None, :] - lag
    i = idx
    o = (i >= 0) & (torch.gather(last, 1, i.clamp(min=0).view(B, -1)).view_as(i) >= 0)
    val = torch.where(o, torch.gather(ffill, 1, i.clamp(min=0).view(B, -1)).view_as(i), 0.0)
    ref = to_model_space(val, loc[..., None], scale[..., None])
    return torch.where(o, ref, torch.zeros_like(ref))


def make_context(x, per):
    B, L = x.shape
    n = L // PATCH
    loc, scale, cnt = causal_stats(x)
    xp = x.view(B, n, PATCH)
    obs = ~torch.isnan(xp)
    z = to_model_space(torch.where(obs, xp, loc[..., None]), loc[..., None], scale[..., None])
    tok = torch.cat([torch.where(obs, z, torch.zeros_like(z)), obs.float()] + frame_features(loc, scale, cnt), -1)
    ref = seasonal_ref(x, loc, scale, per).clamp(-20, 20)
    return Context(tok=tok, x=x, loc=loc, scale=scale, cnt=cnt, ref=ref)


class AlmaSSM(nn.Module):
    """Same forward as train_wm.WMModel (family + per-quantile ref skip). loss/quantiles follow wm_models' API:
    loss(w [B, L+1+H, 1]) uses the last CONTEXT + MAX_H points of every window and every origin (train_wm's
    multi-origin loss); quantiles(ctx [B, L+1, 1], H<=64) uses the last CONTEXT points."""

    def __init__(self, abl="none", init="default", season=1, n_orig=192, N=1, **_):
        super().__init__()
        for k in ("TSAR_WM_SSM_SPEC", "TSAR_WM_SSM_FB", "TSAR_WM_SSM_LOOP", "TSAR_PERIOD", "TSAR_PERIOD_BIN",
                  "TSAR_WM_SSM_EXPAND", "TSAR_WM_SSM_STATE", "TSAR_WM_SSM_CONV"):
            os.environ.pop(k, None)
        cfg = WMConfig(patch=PATCH, context=CONTEXT, max_h=MAX_H, k_steps=K_STEPS, n_patch=(CONTEXT + MAX_H) // PATCH,
                       nq=NQ, tok_dim=2 * PATCH + 3, d_model=96, n_layers=2, n_heads=4, width=1.0, abl=abl)
        self.family = SSMFamily(cfg)
        self.ref_skip = nn.Parameter(torch.zeros(NQ))
        self.season, self.abl, self.n_orig = season, abl, n_orig
        if init == "default":                                   # audit B6: each twin from its own pretraining
            init = CKPT0_DIRECT if abl == "direct" else CKPT0
        if init:
            sd = torch.load(init, map_location="cpu", weights_only=False)["model"]
            missing, unexpected = self.load_state_dict(sd, strict=True), None
        self.N = 1

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ctx, b, j):
        q, _ = self.family.decode(self.family.encode(ctx), ctx, b, j)
        N = len(b)
        r = ctx.ref[b, j].view(N, K_STEPS, PATCH)
        return (q + r[..., None] * self.ref_skip).reshape(N, MAX_H, NQ)

    def loss(self, w, L, H):
        win = w[:, -(CONTEXT + MAX_H):, 0].contiguous()
        B = win.shape[0]
        per = torch.full((B,), self.season)
        ctx = make_context(win, per)
        n = ctx.tok.shape[1]
        pad = torch.full((B, MAX_H), float("nan"))
        fut = torch.cat([win, pad], 1).unfold(1, MAX_H, PATCH)[:, 1:n + 1]
        valid = (ctx.cnt > 0)[..., None] & ~torch.isnan(fut)
        tgt = to_model_space(torch.nan_to_num(fut), ctx.loc[..., None], ctx.scale[..., None])
        keep = valid.any(-1).view(-1)
        bj = torch.nonzero(keep).squeeze(1)
        if self.n_orig and len(bj) > self.n_orig:              # random subset of origins (CPU budget)
            bj = bj[torch.randperm(len(bj))[:self.n_orig].sort().values]
        pred = self(ctx, bj // n, bj % n)
        mask = valid.reshape(B * n, MAX_H)[bj].float()
        y = tgt.reshape(B * n, MAX_H)[bj]
        err = y.unsqueeze(-1) - pred
        lo = (torch.maximum(QT * err, (QT - 1) * err) * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1) / NQ
        return lo, {"pin": lo.item()}

    @torch.no_grad()
    def quantiles(self, lp_ctx, H):
        assert H <= MAX_H
        x = lp_ctx[:, -CONTEXT:, 0].contiguous()
        if x.shape[1] < CONTEXT:
            x = torch.cat([torch.full((x.shape[0], CONTEXT - x.shape[1]), float("nan")), x], 1)
        B = x.shape[0]
        ctx = make_context(x, torch.full((B,), self.season))
        n = ctx.tok.shape[1]
        q = self(ctx, torch.arange(B), torch.full((B,), n - 1))
        lev = torch.sinh(q[:, :H]) * ctx.scale[:, -1, None, None] + ctx.loc[:, -1, None, None]
        last = x[:, -1]
        return (lev - last[:, None, None]).permute(0, 2, 1)
