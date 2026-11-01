"""
Alma-Fin world-model core (architect; docs/fin_search.md section 1). No register(): imported by the wrappers.

Features are computed ONLY from harness-context-shaped arrays (ctx["daily"], ctx["hourly"]); training samples are
built with harness.build_context on a training panel, so training and inference use the identical feature code and
information set (daily rows <= D-1, hourly rows ending <= D 00:00).
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
NQ = len(QS)
W = 14                      # window days (D-14..D-1); the last 336 of the 672 context hours = exactly these days
NF = 16                     # features per day
NAUX = 3                    # head aux: ref - A, slow - A, b = log(sigma*) - A/2
ZD = 16
SUB = 4                     # default sub-steps per day (6 h)
NB = 4                      # 6-hour input blocks per day (features 9..12)
W_MAX = 28
VOL_TARGET = {"crypto": 0.40 / np.sqrt(365), "etf": 0.10 / np.sqrt(252)}
F_LV, F_PK, F_R, F_ABSR, F_VZ, F_MLV, F_MR, F_TK, F_FU = range(9)
F_BLK = slice(9, 13)
F_MCRYPTO, F_MINTRA, F_VALID = 13, 14, 15


def _nm(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis)


def _tail(a, n):
    """last n rows of a [L, m], NaN-padded at the front if L < n."""
    if a.shape[0] >= n:
        return a[-n:]
    pad = np.full((n - a.shape[0],) + a.shape[1:], np.nan)
    return np.concatenate([pad, a], 0)


def make_features(daily, hourly, market, W=W, mf=True):
    """daily[f] [L, m] (rows <= D-1), hourly[f] [Lh, m] (hours ending <= D 00:00) or None.
    Returns X [m, W, NF] float32, aux [m, NAUX] float32, A [m] (anchor = mean log RV of D-30..D-1), ok [m] bool."""
    lr90 = _tail(np.asarray(daily["logrv"], float), 90)
    m = lr90.shape[1]
    lr = lr90[-(W + 30):]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = _nm(lr[-30:], 0)
        nfin = np.isfinite(lr[-30:]).sum(0)
        ok = np.isfinite(A) & (nfin >= 20) & np.isfinite(lr[-1])
        A = np.where(np.isfinite(A), A, 0.0)
        win = slice(30, 30 + W)
        X = np.zeros((m, W, NF))
        f0 = lr[win] - A
        hi = _tail(np.asarray(daily["high"], float), W + 30)[win]
        lo = _tail(np.asarray(daily["low"], float), W + 30)[win]
        pk = np.log(np.maximum(np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12)) ** 2 / (4 * math.log(2)), 1e-12))
        pk[~(np.isfinite(hi) & np.isfinite(lo) & (hi > lo))] = np.nan
        f1 = pk - A
        sA = np.exp(A / 2)
        r = _tail(np.asarray(daily["ret"], float), W + 30)[win] / sA
        vname = "quote_volume" if "quote_volume" in daily else "volume"
        lv = np.log(np.maximum(_tail(np.asarray(daily[vname], float), W + 30), 1.0))
        lv[~np.isfinite(_tail(np.asarray(daily[vname], float), W + 30))] = np.nan
        vz = np.stack([lv[30 + t] - _nm(lv[t:30 + t], 0) for t in range(W)], 0)
        X[:, :, F_LV] = np.clip(f0, -6, 6).T
        X[:, :, F_PK] = np.clip(f1, -8, 6).T
        X[:, :, F_R] = np.clip(r, -6, 6).T
        X[:, :, F_ABSR] = np.clip(np.abs(r), 0, 6).T
        X[:, :, F_VZ] = np.clip(vz, -4, 4).T
        if mf:
            X[:, :, F_MLV] = np.clip(_nm(f0, 1), -6, 6)[None, :]
            X[:, :, F_MR] = np.clip(_nm(np.clip(r, -6, 6), 1), -6, 6)[None, :]
        if market == "crypto" and "taker_ratio" in daily:
            tk = _tail(np.asarray(daily["taker_ratio"], float), W + 30)[win]
            fu = _tail(np.asarray(daily["funding"], float), W + 30)[win]
            X[:, :, F_TK] = np.clip((tk - 0.5) * 10, -3, 3).T
            X[:, :, F_FU] = np.clip(fu * 1e3, -5, 5).T
            X[:, :, F_MCRYPTO] = (np.isfinite(tk) & np.isfinite(fu)).T
        if hourly is not None:
            rv5 = _tail(np.asarray(hourly["rv5"], float), 24 * W)            # [672, m] = days D-28..D-1
            blk = rv5.reshape(W, NB, 24 // NB, m).sum(2)                     # NaN if any hour missing
            lb = np.log(np.maximum(NB * blk, 1e-12)) - A[None, None, :]
            good = np.isfinite(blk).all(1) & (blk > 0).all(1)                # [W, m]
            X[:, :, F_BLK] = np.clip(np.where(good[:, None, :], lb, 0.0), -8, 6).transpose(2, 0, 1)
            X[:, :, F_MINTRA] = good.T
        X[:, :, F_VALID] = np.isfinite(f0).T
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X[:, :, F_TK] *= X[:, :, F_MCRYPTO]
        X[:, :, F_FU] *= X[:, :, F_MCRYPTO]
        ref = _nm(lr[-7:], 0)
        slow = _nm(lr90, 0)
        b = np.log(VOL_TARGET[market]) - A / 2
        aux = np.stack([np.nan_to_num(np.clip(ref - A, -6, 6)), np.nan_to_num(np.clip(slow - A, -6, 6)),
                        np.clip(b, -4, 2)], 1)
    return X.astype(np.float32), aux.astype(np.float32), A, ok


DEF_WTS = {"q": 2.0, "p": 0.5, "r": 0.1, "bD": 0.5, "blk": 0.5, "rec": 0.5, "kl": 0.1, "pred": 0.05}


def with_dow(aux, days, on=True):
    """Append the weekday one-hot of the DECISION day D (calendar, known in advance; not an asset id).
    days: one day (all rows) or an array of days per row."""
    if not on:
        return aux
    import pandas as pd
    dd = pd.DatetimeIndex(np.atleast_1d(np.asarray(days, dtype="datetime64[ns]")))
    wd = np.asarray(dd.weekday)
    if len(wd) == 1:
        wd = np.repeat(wd, len(aux))
    oh = np.eye(7, dtype=np.float32)[wd]
    return np.concatenate([np.asarray(aux, np.float32), oh], 1)


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, o))


class RSSM(nn.Module):
    """DreamerV3-lite RSSM on an intraday clock (`sub` prior sub-steps per day, posterior at day end).
    abl='none': day heads read the latents rolled through day D. abl='direct': day heads read the filtered state at
    the end of D-1 (identical modules and losses)."""

    def __init__(self, d=128, abl="none", zd=ZD, sub=SUB, w=W, mf=True, wts=None, dow=False):
        super().__init__()
        assert sub in (1, 2, 4)
        self.d, self.abl, self.zd, self.sub, self.w, self.mf, self.dow = d, abl, zd, sub, w, mf, bool(dow)
        self.wts = dict(DEF_WTS, **(wts or {}))
        self.h0 = nn.Parameter(torch.zeros(d))
        self.enc = nn.Sequential(nn.Linear(NF, 64), nn.ELU())
        self.post = mlp(d + 64, 2 * zd)
        self.prior = mlp(d, 2 * zd)
        self.cell = nn.GRUCell(zd + sub, d)
        self.blockdec = mlp(d + zd, 1)                  # sub-block log RV (relative to A, daily units)
        self.dec = mlp(d + zd, 9)                       # reconstruct daily features 0..8
        self.pred = mlp(d + zd, 4)                      # day-t predictive: mu/logsd of return_n, mu/logsd of logRV-A
        self.heads = mlp(d + zd + NAUX + 7 * self.dow, NQ + 2, 128)    # quantiles (13), p_up, p_ret_up logits
        self.register_buffer("phase", torch.eye(sub))

    def cfg(self):
        return {"d": self.d, "abl": self.abl, "zd": self.zd, "sub": self.sub, "w": self.w, "mf": self.mf,
                "wts": self.wts, "dow": self.dow}

    def subblocks(self, lb4):
        """log(4*rv_6h) - A [..., 4] -> log(sub*rv_block) - A [..., sub] (blocks nest)."""
        g = NB // self.sub
        x = lb4.reshape(*lb4.shape[:-1], self.sub, g)
        return torch.logsumexp(x, -1) + math.log(self.sub / NB)

    @staticmethod
    def _stats(x):
        mu, ls = x.chunk(2, -1)
        return mu, Fn.softplus(ls) + 0.1

    def day_roll(self, h, z, noise=None):
        """Roll the 4 prior sub-steps of one day. noise: None (means) or list of 4 tensors like z.
        Returns h_end, z_end(prior), lists of h_k, z_k, prior stats."""
        hs, zs, ps = [], [], []
        for k in range(self.sub):
            ph = self.phase[k].expand(h.shape[0], self.sub)
            h = self.cell(torch.cat([z, ph], -1), h)
            mu, sd = self._stats(self.prior(h))
            z = mu if noise is None else mu + sd * noise[k]
            hs.append(h)
            zs.append(z)
            ps.append((mu, sd))
        return h, z, hs, zs, ps

    def filter(self, X, sample=True, gen=None, keep_states=False, losses=True):
        """X [B, W, NF]. Returns final (h, z), losses dict (window terms), optional per-day end states.
        The recurrent loop only runs cell/prior/posterior; decoders and KL are applied afterwards in one batch."""
        B, T = X.shape[0], X.shape[1]
        h = self.h0.expand(B, self.d)
        z = torch.zeros(B, self.zd, device=X.device)
        e = self.enc(X)
        HS, ZS, PM, PS, QM, QS_, HE, ZE = [], [], [], [], [], [], [], []
        for t in range(T):
            noise = [torch.randn(B, self.zd, generator=gen) for _ in range(self.sub)] if sample else None
            h, zp, hs, zs, ps = self.day_roll(h, z, noise)
            qmu, qsd = self._stats(self.post(torch.cat([h, e[:, t]], -1)))
            z = qmu + qsd * torch.randn(B, self.zd, generator=gen) if sample else qmu
            if losses:
                HS.append(torch.stack(hs, 1)); ZS.append(torch.stack(zs, 1))
                PM.append(ps[-1][0]); PS.append(ps[-1][1]); QM.append(qmu); QS_.append(qsd)
                HE.append(h); ZE.append(z)
            if keep_states:
                if not losses:
                    HE.append(h); ZE.append(z)
        L = {}
        if losses:
            hs = torch.stack(HS, 1)                         # [B, T, SUB, d]
            zs = torch.stack(ZS, 1)                         # [B, T, SUB, ZD]
            valid, intra = X[:, :, F_VALID], X[:, :, F_MINTRA]
            bp = self.blockdec(torch.cat([hs, zs], -1))[..., 0]          # [B, T, SUB]
            L["blk"] = (((bp - self.subblocks(X[:, :, F_BLK])) ** 2).mean(-1) * intra).sum() / (intra.sum() + 1)
            pr = self.pred(torch.cat([hs.mean(2), zs.mean(2)], -1))      # [B, T, 4]
            ls_r, ls_v = pr[..., 1].clamp(-5, 3), pr[..., 3].clamp(-5, 3)
            nll_r = ls_r + 0.5 * ((X[:, :, F_R] - pr[..., 0]) / ls_r.exp()) ** 2
            nll_v = ls_v + 0.5 * ((X[:, :, F_LV] - pr[..., 2]) / ls_v.exp()) ** 2
            L["pred"] = ((nll_r + nll_v) * valid).sum() / (valid.sum() + 1)
            pm, psd, qm, qsd = [torch.stack(v, 1) for v in (PM, PS, QM, QS_)]
            kl_dyn = _kl(qm.detach(), qsd.detach(), pm, psd)
            kl_rep = _kl(qm, qsd, pm.detach(), psd.detach())
            kl = 0.8 * torch.clamp(kl_dyn, min=1.0) + 0.2 * torch.clamp(kl_rep, min=1.0)
            L["kl"] = (kl * valid).sum() / (valid.sum() + 1)
            he, ze = torch.stack(HE, 1), torch.stack(ZE, 1)
            rec = self.dec(torch.cat([he, ze], -1))
            tgt = X[:, :, :9]
            msk = torch.ones_like(tgt)
            msk[..., F_TK] = X[:, :, F_MCRYPTO]
            msk[..., F_FU] = X[:, :, F_MCRYPTO]
            L["rec"] = (((rec - tgt) ** 2 * msk).mean(-1) * valid).sum() / (valid.sum() + 1)
        states = list(zip(HE, ZE)) if keep_states else []
        return h, z, L, states

    def head_input(self, h, z, aux, sample=True, gen=None, S=1, shared_noise=None):
        """Day-D head inputs. Returns list (len S) of [B, d+ZD+NAUX] tensors and block preds for day D."""
        outs, blks = [], []
        for s in range(S):
            if self.abl == "direct":
                noise = None if not sample else [torch.randn(h.shape[0], self.zd, generator=gen) for _ in range(self.sub)]
                if shared_noise is not None:
                    noise = [shared_noise[s, k, :self.zd].expand(h.shape[0], self.zd) for k in range(self.sub)]
                _, _, hs, zs, _ = self.day_roll(h, z, noise)
                outs.append(torch.cat([h, z, aux], -1))
            else:
                if shared_noise is not None:
                    noise = [shared_noise[s, k, :self.zd].expand(h.shape[0], self.zd) for k in range(self.sub)]
                else:
                    noise = None if not sample else [torch.randn(h.shape[0], self.zd, generator=gen) for _ in range(self.sub)]
                _, _, hs, zs, _ = self.day_roll(h, z, noise)
                outs.append(torch.cat([torch.stack(hs).mean(0), torch.stack(zs).mean(0), aux], -1))
            blks.append(torch.stack([self.blockdec(torch.cat([hs[k], zs[k]], -1))[:, 0] for k in range(self.sub)], 1))
        return outs, blks

    def head_out(self, hin):
        o = self.heads(hin)
        q0 = o[:, :1]
        q = torch.cat([q0, q0 + torch.cumsum(Fn.softplus(o[:, 1:NQ]) + 1e-3, 1)], 1)
        return q, o[:, NQ], o[:, NQ + 1]


def _kl(m1, s1, m2, s2):
    return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5).sum(-1)


QT = torch.tensor(QS)


def pinball(q, y):
    d = y[:, None] - q
    return torch.maximum(QT * d, (QT - 1) * d).mean(1)


def loss_fn(model, batch, gen=None):
    X, aux, tl, y, yr, blkD, mblkD = batch
    h, z, L, _ = model.filter(X, sample=True, gen=gen)
    hins, blks = model.head_input(h, z, aux, sample=True, gen=gen, S=1)
    q, lp, lr_ = model.head_out(hins[0])
    okv = torch.isfinite(tl)
    tl0 = torch.nan_to_num(tl)
    l_q = (pinball(q, tl0) * okv).sum() / (okv.sum() + 1)
    oky = torch.isfinite(y)
    l_p = (Fn.binary_cross_entropy_with_logits(lp, torch.nan_to_num(y), reduction="none") * oky).sum() / (oky.sum() + 1)
    okr = torch.isfinite(yr)
    l_r = (Fn.binary_cross_entropy_with_logits(lr_, torch.nan_to_num(yr), reduction="none") * okr).sum() / (okr.sum() + 1)
    l_b = (((blks[0] - model.subblocks(torch.nan_to_num(blkD))) ** 2).mean(1) * mblkD).sum() / (mblkD.sum() + 1)
    w = model.wts
    tot = (w["q"] * l_q + w["p"] * l_p + w["r"] * l_r + w["bD"] * l_b + w["blk"] * L["blk"] + w["rec"] * L["rec"]
           + w["kl"] * L["kl"] + w["pred"] * L["pred"])
    return tot, {k: float(v.detach()) for k, v in (("q", l_q), ("p", l_p), ("blk", L["blk"]), ("kl", L["kl"]),
                                                    ("rec", L["rec"]))}


def nparams(model):
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- inference helpers
S_INF = 16
INF_SEED = 1234


def shared_noise(S=S_INF):
    g = torch.Generator().manual_seed(INF_SEED)
    return torch.randn(S, 4, 32, generator=g)


@torch.no_grad()
def forecast_members(members, X, aux, A):
    """members: list of RSSM (eval). Deterministic filtering (means), S_INF shared-noise rollouts (abl none) or the
    filtered state (abl direct). Returns q [m, NQ] (absolute log RV), p_up [m], p_ret [m] averaged over members."""
    Xt, at = torch.from_numpy(X), torch.from_numpy(aux)
    qs, ps, prs = [], [], []
    sn = shared_noise()
    assert len({(m.w, m.mf) for m in members}) == 1, "ensemble members must share the feature recipe"
    for md in members:
        h, z, _, _ = md.filter(Xt, sample=False, losses=False)
        S = S_INF if md.abl == "none" else 1
        hins, _ = md.head_input(h, z, at, sample=False, S=S, shared_noise=sn if md.abl == "none" else None)
        for hin in hins:
            q, lp, lr_ = md.head_out(hin)
            qs.append(q.numpy())
            ps.append(torch.sigmoid(lp).numpy())
            prs.append(torch.sigmoid(lr_).numpy())
    q = np.sort(np.mean(qs, 0), 1) + A[:, None]
    return q.astype(np.float64), np.mean(ps, 0).astype(np.float64), np.mean(prs, 0).astype(np.float64)


def save_members(path, members, meta):
    torch.save({"meta": meta, "states": [m.state_dict() for m in members],
                "cfg": [m.cfg() for m in members]}, path)


def load_members(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for cfg, st in zip(ck["cfg"], ck["states"]):
        m = RSSM(**cfg)
        m.load_state_dict(st)
        m.eval()
        out.append(m)
    return out, ck["meta"]
