"""Architect B model families (non-RSSM). One configurable module, BNet, operating on a DAY SET of assets:

  inputs per asset: X [W,16] daily window (wm_core features), hz [168,4] hourly window, har [8], aux [3]
  enc_d   : daily temporal encoder   'gru' | 'tf' (transformer) | 'lru' (diagonal linear SSM) | 'mlp' | 'none'
  enc_h   : hourly encoder           'none' | 'conv' | 'lru' | 'tf'
  cross   : cross-asset mixing        'none' | 'mean' (market pooling) | 'attn' (set self-attention, no asset id)
  head    : 'mlp' | 'moe' (K experts, gate on latent + market state = regime world model)
  har_skip: linear HAR term added to the median and the logit (HAR as inductive bias)
  wm      : latent world model: s(D-1) = enc(context of D); transition T(s, market) -> s_hat(D); heads decode day D
            from s_hat; training adds a latent-consistency loss ||s_hat(D) - sg(enc(context of D+1))||^2 (only on
            pairs whose D+1 context is inside the fit window) and intraday block decoding of day D.
  stoch   : 0 = deterministic transition; >0 = Gaussian transition with S samples (stochastic latent, mean of
            p over samples at inference; fixed noise -> deterministic)
Outputs: q [n, 13] quantiles of log RV(D) - A (sorted via cumulative softplus), logit [n] P(y_vol = 1).
No asset-ID embeddings anywhere (zero-shot portability).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
NQ = len(QS)
NF, NAUX, NH, NHC = 16, 3, 168, 4
NHAR = 15     # 8 HAR features + 7 weekday one-hot (feats_b.NHAR)


def mlp(i, o, h=64, n=1, act=nn.GELU, drop=0.0):
    L, d = [], i
    for _ in range(n):
        L += [nn.Linear(d, h), act()] + ([nn.Dropout(drop)] if drop > 0 else [])
        d = h
    L.append(nn.Linear(d, o))
    return nn.Sequential(*L)


class LRU(nn.Module):
    """Diagonal linear recurrence h_t = a*h_{t-1} + B x_t (a in (0,1) learned per channel), readout of last state
    and mean state. Parallel via cumulative products (short sequences)."""

    def __init__(self, i, d):
        super().__init__()
        self.inp = nn.Linear(i, d)
        self.nu = nn.Parameter(torch.linspace(-3.0, 2.0, d))     # a = sigmoid(nu): spread of time scales
        self.out = nn.Linear(2 * d, d)

    def forward(self, x):                     # x [n, T, i]
        u = self.inp(x)
        a = torch.sigmoid(self.nu)
        T = u.shape[1]
        h = torch.zeros_like(u[:, 0])
        hs = []
        for t in range(T):
            h = a * h + (1 - a) * u[:, t]
            hs.append(h)
        H = torch.stack(hs, 1)
        return self.out(torch.cat([H[:, -1], H.mean(1)], -1))


class TfEnc(nn.Module):
    def __init__(self, i, d, T, layers=1, heads=4):
        super().__init__()
        self.inp = nn.Linear(i, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)

    def forward(self, x):
        h = self.tf(self.inp(x) + self.pos[-x.shape[1]:])
        return h[:, -1]


class BNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = c = dict(cfg)
        d = c.get("d", 64)
        self.d = d
        W = c.get("W", 14)
        self.W = W
        ed, eh = c.get("enc_d", "gru"), c.get("enc_h", "none")
        if ed == "gru":
            self.encd = nn.GRU(NF, d, batch_first=True)
        elif ed == "tf":
            self.encd = TfEnc(NF, d, 14, layers=c.get("layers", 1))
        elif ed == "lru":
            self.encd = LRU(NF, d)
        elif ed == "mlp":
            self.encd = mlp(NF * W, d, d)
        else:
            self.encd = None
        if eh == "conv":
            self.ench = nn.Sequential(nn.Conv1d(NHC, 32, 6, stride=6), nn.GELU(), nn.Conv1d(32, 32, 4, stride=4),
                                      nn.GELU(), nn.Flatten(), nn.Linear(32 * 7, d))
        elif eh == "lru":
            self.ench = LRU(NHC, d)
        elif eh == "tf":
            self.ench = TfEnc(NHC * 4, d, 42, layers=1)
        else:
            self.ench = None
        nin = NHAR + NAUX + (d if self.encd is not None else 0) + (d if self.ench is not None else 0)
        self.fuse = mlp(nin, d, d)
        cross = c.get("cross", "none")
        if cross == "attn":
            self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.ln = nn.LayerNorm(d)
        if cross in ("mean", "attn"):
            self.mk = mlp(2 * d, d, d)
        self.wm = c.get("wm", 0.0)
        self.stoch = c.get("stoch", 0)
        if self.wm > 0 or self.stoch > 0:
            self.trans = mlp(2 * d, (2 if self.stoch else 1) * d, d, n=2)
            self.blk = mlp(d, 4, d)
        K = c.get("K", 1)
        hd = c.get("hd", 64)
        if c.get("head", "mlp") == "moe":
            self.gate = mlp(2 * d, K, 32)
            self.experts = nn.ModuleList([mlp(d, NQ + 1, hd) for _ in range(K)])
        else:
            self.headm = mlp(d, NQ + 1, hd, drop=c.get("drop", 0.0))
        self.har_skip = c.get("har_skip", 1)
        if self.har_skip:
            self.harq = nn.Linear(NHAR + NAUX, 1)
            self.harp = nn.Linear(NHAR + NAUX, 1)

    # --------------------------------------------------------------- encoders
    def encode(self, X, hz, har, aux, mask):
        """X [B,m,W,NF] hz [B,m,NH,NHC] har [B,m,NHAR] aux [B,m,NAUX] mask [B,m] -> s [B,m,d], market [B,d]."""
        B, m = X.shape[:2]
        parts = [har.reshape(B * m, -1), aux.reshape(B * m, -1)]
        if self.encd is not None:
            x = X.reshape(B * m, X.shape[2], NF)[:, -self.W:]
            if isinstance(self.encd, nn.GRU):
                parts.append(self.encd(x)[1][0])
            elif isinstance(self.encd, nn.Sequential):
                parts.append(self.encd(x.reshape(B * m, -1)))
            else:
                parts.append(self.encd(x))
        if self.ench is not None:
            h = hz.reshape(B * m, NH, NHC)
            if isinstance(self.ench, nn.Sequential):
                parts.append(self.ench(h.transpose(1, 2)))
            elif isinstance(self.ench, TfEnc):
                parts.append(self.ench(h.reshape(B * m, 42, 4 * NHC)))
            else:
                parts.append(self.ench(h[:, ::2]))
        s = self.fuse(torch.cat(parts, -1)).reshape(B, m, self.d)
        w = mask.float().unsqueeze(-1)
        mkt = (s * w).sum(1) / w.sum(1).clamp(min=1.0)
        cross = self.cfg.get("cross", "none")
        if cross == "attn":
            kpm = ~mask
            kpm = kpm & ~kpm.all(1, keepdim=True)          # a day with no valid asset: attend to all (avoid NaN)
            a, _ = self.attn(self.ln(s), self.ln(s), self.ln(s), key_padding_mask=kpm)
            a = torch.nan_to_num(a)
            s = s + a
            mkt = (s * w).sum(1) / w.sum(1).clamp(min=1.0)
        if cross in ("mean", "attn"):
            s = s + self.mk(torch.cat([s, mkt.unsqueeze(1).expand_as(s)], -1))
        return s, mkt

    def step(self, s, mkt, noise=None):
        """latent transition one day ahead."""
        if not (self.wm > 0 or self.stoch > 0):
            return s, None
        z = self.trans(torch.cat([s, mkt.unsqueeze(1).expand_as(s)], -1))
        if self.stoch:
            mu, ls = z.chunk(2, -1)
            sd = Fn.softplus(ls) * 0.5 + 1e-3
            if noise is None:
                return s + mu, (mu, sd)
            return s + mu + sd * noise, (mu, sd)
        return s + z, None

    def heads(self, s, mkt, har, aux):
        if self.cfg.get("head", "mlp") == "moe":
            g = torch.softmax(self.gate(torch.cat([s, mkt.unsqueeze(-2).expand_as(s)], -1)), -1)
            o = torch.stack([e(s) for e in self.experts], -1)
            o = (o * g.unsqueeze(-2)).sum(-1)
        else:
            o = self.headm(s)
        mid = o[..., NQ // 2:NQ // 2 + 1]
        lo = o[..., :NQ // 2]
        hi = o[..., NQ // 2 + 1:NQ]
        logit = o[..., NQ]
        if self.har_skip:
            ha = torch.cat([har, aux], -1)
            mid = mid + self.harq(ha)
            logit = logit + self.harp(ha).squeeze(-1)
        up = mid + torch.cumsum(Fn.softplus(hi) + 1e-3, -1)
        dn = mid - torch.flip(torch.cumsum(Fn.softplus(torch.flip(lo, [-1])) + 1e-3, -1), [-1])
        q = torch.cat([dn, mid, up], -1)
        return q, logit

    def forward(self, X, hz, har, aux, mask, S=None, gen=None):
        s, mkt = self.encode(X, hz, har, aux, mask)
        if self.stoch:
            S = S or self.stoch
            qs, ls, nexts = [], [], []
            for k in range(S):
                noise = torch.randn(s.shape, generator=gen) if gen is not None else torch.randn_like(s)
                sn, st = self.step(s, mkt, noise)
                q, lg = self.heads(sn, mkt, har, aux)
                qs.append(q); ls.append(lg); nexts.append(sn)
            q = torch.stack(qs).mean(0)
            q, _ = torch.sort(q, -1)
            prob = torch.sigmoid(torch.stack(ls)).mean(0)
            logit = torch.logit(prob.clamp(1e-5, 1 - 1e-5))
            return {"q": q, "logit": logit, "s": s, "mkt": mkt, "next": torch.stack(nexts).mean(0), "st": st,
                    "logits_all": torch.stack(ls)}
        sn, st = self.step(s, mkt)
        q, logit = self.heads(sn, mkt, har, aux)
        return {"q": q, "logit": logit, "s": s, "mkt": mkt, "next": sn, "st": st}


def pinball(q, y):
    t = torch.tensor(QS, dtype=q.dtype)
    e = y.unsqueeze(-1) - q
    return torch.maximum(t * e, (t - 1) * e).mean(-1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def save(path, members, cfg, meta):
    torch.save({"cfg": cfg, "meta": meta, "state": [m.state_dict() for m in members]}, path)


def load(path):
    z = torch.load(path, map_location="cpu", weights_only=False)
    ms = []
    for st in z["state"]:
        m = BNet(z["cfg"])
        m.load_state_dict(st)
        m.eval()
        ms.append(m)
    return ms, z["cfg"], z["meta"]
