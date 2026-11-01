"""Alma-Power world models (DE-LU day-ahead prices). Owner: power world-model architect.

Every model maps one ORIGIN (target day d) to quantiles of the 24 hourly prices of day d, in a per-origin z-frame
(z = (price - mu) / sd, mu / sd over prices of days d-7..d-1). Inputs (built by wm_data.batch, which asserts the
information set): prices of days d-L..d-1, fundamentals (load, wind, solar, residual load) of days d-L-1..d-2
(i.e. lagged by one extra day), and the calendar (weekday, German public holiday, season) of every day incl. d.

Common observer: per-day token MLP([z prices 24, fundamentals 4x24, calendar]) -> GRU over the L context days ->
state s (the world state at the end of day d-1; fundamentals in it stop at d-2).

World models (abl="none") and their matched twins (abl="direct"): identical module list and parameter count, same
inputs; in the twin every hour is ONE step of the same transition from the origin state s (no carried state, no
closed-loop feedback), so the twin decodes all 24 hours at once from s. Differences are only in how the latent
state is advanced.

arch="cycle"   daily-cycle SSM WM: hourly rollout of a GRUCell over the 24 hours of day d driven by a phase clock
               (hour sin/cos, weekday, holiday), the same-hour prices of d-1 and d-7, the same-hour fundamentals of
               d-2, and closed-loop feedback of its own previous-hour median. Quantile head per hour.
arch="phys"    physical-driver WM: the latent "system state" is rolled 48 hours from the end of day d-2 (last known
               fundamentals) through d-1 (assimilating the observed d-1 prices) into d (imagination). At every hour the
               state emits a residual-load estimate rl_hat (aux loss vs the realised residual load: a TRAINING target
               only, never an input for that day) and a scarcity/surplus regime; price = monotone merit-order curve
               of rl_hat (sum of positive-weight sigmoids, curve parameters from s and the hour) + quantile spreads.
arch="regime"  regime-switching WM: the cycle rollout, but each hour's quantiles are a mixture (quantile averaging)
               of 3 regime heads (normal / spike / negative), regime probabilities from a learned transition
               pi_t = softmax(T(h_t, pi_{t-1})) carried through the rollout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QL = torch.tensor([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])
NQ = len(QL)
MED = 6
NF = 4            # load, wind, solar, residual load (normalised by fixed train scales)
NCAL = 12         # 7 weekday one-hot + holiday + bridge flag + season sin/cos + year trend


def quantile_from_increments(center, raw):
    """center [...], raw [..., NQ-1] -> monotone quantiles [..., NQ] with the median at `center`."""
    inc = F.softplus(raw) + 1e-3
    lo = inc[..., :MED].flip(-1).cumsum(-1).flip(-1)
    hi = inc[..., MED:].cumsum(-1)
    c = center[..., None]
    return torch.cat([c - lo, c, c + hi], -1)


class Observer(nn.Module):
    def __init__(self, d=64, L=14):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(24 + NF * 24 + NCAL, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, zp, fu, cal):
        # zp [B,L,24], fu [B,L,24,NF] (fundamentals of the day BEFORE each context day), cal [B,L,NCAL]
        B, L, _ = zp.shape
        x = torch.cat([zp, fu.reshape(B, L, -1), cal], -1)
        _, h = self.gru(self.tok(x))
        return h[0]


def hour_clock(device):
    t = torch.arange(24, device=device, dtype=torch.float32)
    return torch.stack([torch.sin(2 * math.pi * t / 24), torch.cos(2 * math.pi * t / 24),
                        torch.sin(4 * math.pi * t / 24), torch.cos(4 * math.pi * t / 24)], -1)   # [24,4]


class CycleWM(nn.Module):
    """arch cycle / regime."""

    def __init__(self, d=64, L=14, abl="none", regime=False, nreg=3):
        super().__init__()
        self.abl, self.regime, self.nreg = abl, regime, nreg
        self.obs = Observer(d, L)
        nin = 4 + NCAL + 2 + NF + 1 + (nreg if regime else 0)
        self.inp = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.GRUCell(d, d)
        nh = nreg if regime else 1
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, nh * NQ))
        self.ref_w = nn.Parameter(torch.tensor([0.7, 0.2]))
        if regime:
            self.trans = nn.Linear(d + nreg, nreg)

    def forward(self, b):
        s = self.obs(b["zp"], b["fu"], b["cal"])
        B, d = s.shape
        clk = hour_clock(s.device)
        y1, y7 = b["zp"][:, -1], b["zp"][:, -7]                    # same-hour prices of d-1, d-7 (z)
        fu2 = b["fu"][:, -1]                                       # fundamentals of d-2 [B,24,NF]
        cald = b["cal_d"]
        h = s
        fb = torch.zeros(B, device=s.device)
        pi = torch.full((B, self.nreg), 1.0 / self.nreg, device=s.device) if self.regime else None
        out = []
        for t in range(24):
            parts = [clk[t].expand(B, 4), cald, y1[:, t:t + 1], y7[:, t:t + 1], fu2[:, t],
                     (fb if self.abl == "none" else torch.zeros_like(fb))[:, None]]
            if self.regime:
                parts.append(pi if self.abl == "none" else torch.full_like(pi, 1.0 / self.nreg))
            u = self.inp(torch.cat(parts, -1))
            h = self.cell(u, h if self.abl == "none" else s)
            ref = self.ref_w[0] * y1[:, t] + self.ref_w[1] * y7[:, t]
            raw = self.head(torch.cat([h, s], -1))
            if self.regime:
                pin = pi if self.abl == "none" else torch.full_like(pi, 1.0 / self.nreg)
                pi_new = torch.softmax(self.trans(torch.cat([h, pin], -1)), -1)
                raw = raw.view(B, self.nreg, NQ)
                qk = quantile_from_increments(ref[:, None] + raw[..., MED], torch.cat([raw[..., :MED], raw[..., MED + 1:]], -1))
                q = (pi_new[..., None] * qk).sum(1)
                if self.abl == "none":
                    pi = pi_new
            else:
                q = quantile_from_increments(ref + raw[:, MED], torch.cat([raw[:, :MED], raw[:, MED + 1:]], -1))
            out.append(q)
            fb = q[:, MED]
        return {"q": torch.stack(out, 1)}                          # [B,24,NQ] in z


class MeritOrder(nn.Module):
    """price_z(rl) = a + sum_k w_k sigmoid((rl - c_k) / tau_k), w_k > 0, tau_k > 0 : monotone increasing in rl.
    Curve parameters (a, w, c, tau) per origin from the state s (fuel / CO2 / hydro context), per-hour offset."""

    def __init__(self, d, K=8):
        super().__init__()
        self.K = K
        self.par = nn.Linear(d, 1 + 3 * K)
        self.hour_off = nn.Parameter(torch.zeros(24))
        with torch.no_grad():
            self.par.bias[1 + 2 * K:1 + 3 * K] = torch.linspace(-2, 2, K)

    def forward(self, s, rl, hours):
        # s [B,d], rl [B,T] (normalised residual load), hours [T] long
        p = self.par(s)
        K = self.K
        a = p[:, :1]
        w = F.softplus(p[:, 1:1 + K]) * 0.5
        tau = F.softplus(p[:, 1 + K:1 + 2 * K]) + 0.05
        c = p[:, 1 + 2 * K:]
        sig = torch.sigmoid((rl[..., None] - c[:, None]) / tau[:, None])
        return a + (w[:, None] * sig).sum(-1) + self.hour_off[hours]


class PhysWM(nn.Module):
    """arch phys. 48-hour rollout from the end of day d-2: hours 0..23 = day d-1 (observed price assimilated),
    24..47 = day d (imagined). Outputs quantiles for day d, rl_hat for days d-1 and d (aux), and the merit-order
    median so that scenario interventions on rl_hat propagate through a monotone curve."""

    def __init__(self, d=64, L=14, abl="none", nreg=3):
        super().__init__()
        self.abl, self.nreg = abl, nreg
        self.obs = Observer(d, L)
        nin = 4 + NCAL + NF + 2 + 1          # clock, calendar of that day, fundamentals 2 days earlier, prices, phase flag
        self.inp = nn.Sequential(nn.Linear(nin, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.GRUCell(d, d)
        self.rl_head = nn.Linear(d, 1)
        self.reg_head = nn.Linear(d, nreg)
        self.reg_off = nn.Parameter(torch.tensor([0.0, 1.5, -1.5]))
        self.merit = MeritOrder(d)
        self.spread = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, NQ - 1))
        self.mix = nn.Parameter(torch.tensor([0.5]))

    def rollout(self, b):
        s = self.obs(b["zp"], b["fu"], b["cal"])
        B, d = s.shape
        clk = hour_clock(s.device)
        fu = b["fu"]               # fu[:, -1] = fundamentals of d-2, fu[:, -2] = d-3
        h = s
        rls, hs, regs = [], [], []
        for t in range(48):
            day1 = t >= 24
            hh = t % 24
            cal = b["cal_d"] if day1 else b["cal"][:, -1]
            # fundamentals two days before the simulated day: d-3 for day d-1, d-2 for day d
            f = fu[:, -1, hh] if day1 else fu[:, -2, hh]
            # prices: for day d-1 the observed price of d-1 (assimilation) and d-2; for day d: d-1 and d-7 (no peeking)
            if day1:
                pa, pb = b["zp"][:, -1, hh], b["zp"][:, -7, hh]
            else:
                pa, pb = b["zp"][:, -1, hh], b["zp"][:, -2, hh]
            u = self.inp(torch.cat([clk[hh].expand(B, 4), cal, f, pa[:, None], pb[:, None],
                                    torch.full((B, 1), float(day1), device=s.device)], -1))
            h = self.cell(u, h if self.abl == "none" else s)
            hs.append(h)
            rls.append(self.rl_head(h)[:, 0] + f[:, 3])          # residual load = lagged level + learned change
            regs.append(torch.softmax(self.reg_head(h), -1))
        return s, torch.stack(hs, 1), torch.stack(rls, 1), torch.stack(regs, 1)

    def readout(self, s, hs, rl, reg, b):
        hours = torch.arange(24, device=s.device)
        mo = self.merit(s, rl[:, 24:], hours)                                 # [B,24] monotone in rl_hat
        y1 = b["zp"][:, -1]
        center = mo + (reg[:, 24:] * self.reg_off).sum(-1) + self.mix * y1
        raw = self.spread(torch.cat([hs[:, 24:], s[:, None].expand(-1, 24, -1)], -1))
        return quantile_from_increments(center, raw)

    def forward(self, b, rl_shift=None):
        s, hs, rl, reg = self.rollout(b)
        if rl_shift is not None:                 # scenario intervention on the latent residual load of day d
            rl = rl.clone()
            rl[:, 24:] = rl[:, 24:] + rl_shift
        return {"q": self.readout(s, hs, rl, reg, b), "rl": rl}


def build(arch, abl, d=64, L=14):
    if arch == "cycle":
        return CycleWM(d, L, abl)
    if arch == "regime":
        return CycleWM(d, L, abl, regime=True)
    if arch == "phys":
        return PhysWM(d, L, abl)
    raise ValueError(arch)


def pinball(q, y, ql=QL):
    err = y[..., None] - q
    ql = ql.to(q.device)
    return torch.maximum(ql * err, (ql - 1) * err).mean()
