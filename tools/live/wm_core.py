"""
Alma-Live Gas world-model core (architect; docs/live_search.md section 0.4).

Model families (cfg["arch"]):
  struct  : encoder -> structured latent (named physical nodes + learned z); daily transition = physics balance
            (storage += supply - demand(forecast HDD/CDD) - net exports, learned coefficients) + learned residual
            + N-order shock-propagation graph over the nodes; readouts from the rolled-out state at each target's
            calendar step.
  rssm    : same transition, but the encoder is a stochastic RSSM filter over the window (prior/posterior, KL).
  direct  : same encoder; readouts straight from [h, flattened forecast inputs] (no rollout), params matched +-5%.
cfg["graph"] = 0 gives the no-graph twin (graph edges removed; transition otherwise identical).
cfg["enc"] in {gru, tfm} picks the window encoder.
"""
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

WD = __import__(os.environ.get("LIVE_SRC", "wm_store"))

QL = np.round(np.arange(0.05, 0.951, 0.05), 2)
NQ = len(QL)
NT = len(WD.TARGETS)
NODES = ["storage", "demand", "supply", "net_exports", "price", "vol", "positioning", "producer_eq"]
NN_ = len(NODES)
N_ST, N_DEM, N_SUP, N_LNG, N_PX, N_VOL, N_POS, N_EQ = range(NN_)
# structural sparsity of the shock graph (src -> dst); signs are learned
EDGES = [(N_DEM, N_ST), (N_SUP, N_ST), (N_LNG, N_ST), (N_ST, N_PX), (N_DEM, N_PX), (N_PX, N_EQ), (N_PX, N_VOL),
         (N_VOL, N_PX), (N_POS, N_PX), (N_PX, N_POS), (N_PX, N_SUP), (N_LNG, N_PX), (N_ST, N_VOL), (N_DEM, N_VOL)]
TQ = torch.tensor(QL, dtype=torch.float32)

# ===================================================================== THE EDITABLE RECIPE (autoresearch loop)
# tools/live/search/program.md: this file is the ONLY file the loop edits. `budget_s` is the fixed wall-clock
# training budget PER FOLD (4 DEV folds + eval + leak test ~= 4 min per experiment); the direct twin gets the same.
DEFAULT_CFG = {"arch": "struct", "enc": "gru", "d": 32, "hh": 32, "th": 64, "L": 20, "graph": 1, "sv": 1,
               "lr": 2e-3, "wd": 1e-4, "bs": 128, "budget_s": 45, "es": 0.15}


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class QHead(nn.Module):
    """monotone quantiles: median + cumulative softplus increments."""

    def __init__(self, i, h):
        super().__init__()
        self.net = mlp(i, NQ, h)

    def forward(self, x):
        o = self.net(x)
        mid = NQ // 2
        inc = Fn.softplus(o) * 0.25 + 1e-3
        up = torch.cumsum(inc[:, mid + 1:], 1)
        dn = torch.cumsum(inc[:, :mid].flip(1), 1).flip(1)
        m = o[:, mid:mid + 1]
        return torch.cat([m - dn, m, m + up], 1)


class Encoder(nn.Module):
    def __init__(self, kind, d, nf=WD.NF, L=20):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(nf, d)
        if kind == "gru":
            self.rnn = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.rnn = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, X):
        e = self.inp(X)
        if self.kind == "gru":
            o, _ = self.rnn(e)
            return o[:, -1]
        o = self.rnn(e + self.pos[:, -e.shape[1]:])
        return o[:, -1]


class RSSMEnc(nn.Module):
    """Dreamer-style filter over the window: h_t = GRU(h, z, x); prior p(z|h), posterior q(z|h, x)."""

    def __init__(self, d, zd=16, nf=WD.NF):
        super().__init__()
        self.zd = zd
        self.inp = nn.Linear(nf, d)
        self.cell = nn.GRUCell(d + zd, d)
        self.prior = nn.Linear(d, 2 * zd)
        self.post = nn.Linear(2 * d, 2 * zd)
        self.out = nn.Linear(d + zd, d)

    def forward(self, X, sample=True):
        B, L, _ = X.shape
        h = X.new_zeros(B, self.cell.hidden_size)
        z = X.new_zeros(B, self.zd)
        kl = 0.0
        e = self.inp(X)
        for t in range(L):
            h = self.cell(torch.cat([e[:, t], z], 1), h)
            pm, ps = self.prior(h).chunk(2, 1)
            qm, qs = self.post(torch.cat([h, e[:, t]], 1)).chunk(2, 1)
            ps, qs = Fn.softplus(ps) + 0.05, Fn.softplus(qs) + 0.05
            z = qm + qs * torch.randn_like(qs) if sample else qm
            kl = kl + (torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(1).mean()
        self.kl = kl / L
        return self.out(torch.cat([h, z], 1))


class WM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d, hh = cfg.get("d", 32), cfg.get("hh", 32)
        self.arch = cfg["arch"]
        L = cfg.get("L", 20)
        self.enc = RSSMEnc(d) if self.arch == "rssm" else Encoder(cfg.get("enc", "gru"), d, L=L)
        if self.arch == "direct":
            dh = cfg.get("dh", 64)
            self.mix = mlp(d + WD.H * WD.NU, d, dh)
            self.heads = nn.ModuleList([QHead(2 * d, hh) for _ in range(NT)])
        else:
            self.s0 = nn.Linear(d, d)
            self.dem = nn.Linear(WD.NU, 1)                      # demand node from forecast weather of the day
            self.bal = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))  # supply / demand / net-export weights
            self.res = mlp(d + WD.NU, d, cfg.get("th", 64))     # learned residual transition (whole state)
            self.graph = cfg.get("graph", 1)
            # stochastic-vol block: log-vol node mean-reverts (kappa, theta learned); noise only when sampling
            self.sv = cfg.get("sv", 1)
            self.kappa = nn.Parameter(torch.tensor(-2.0)); self.theta = nn.Parameter(torch.tensor(0.0))
            self.svs = nn.Parameter(torch.tensor(-2.0))
            if self.graph:
                self.ew = nn.Parameter(torch.zeros(len(EDGES)))
                src = torch.tensor([e[0] for e in EDGES]); dst = torch.tensor([e[1] for e in EDGES])
                self.register_buffer("src", src); self.register_buffer("dst", dst)
            self.heads = nn.ModuleList([QHead(2 * d, hh) for _ in range(NT)])
        self.d = d
        self.training_noise = False

    def adj(self):
        A = torch.zeros(NN_, NN_, device=self.ew.device)
        return A.index_put((self.dst, self.src), torch.tanh(self.ew) * 0.5)

    def step(self, s, u, shock=None):
        nodes, z = s[:, :NN_], s[:, NN_:]
        dem = 0.5 * nodes[:, N_DEM] + self.dem(u)[:, 0]
        # physics balance on the storage node (scaled units): storage += supply - demand - net exports
        stor = nodes[:, N_ST] + self.bal[0] * nodes[:, N_SUP] - self.bal[1] * dem - self.bal[2] * nodes[:, N_LNG]
        cols = [stor, dem] + [nodes[:, i] for i in range(2, NN_)]
        if self.sv:
            v = nodes[:, N_VOL]
            v = v + torch.sigmoid(self.kappa) * (self.theta - v)
            if self.training_noise:
                v = v + Fn.softplus(self.svs) * torch.randn_like(v)
            cols[N_VOL] = v
        nodes = torch.stack(cols, 1)
        if self.graph:
            nodes = nodes + nodes @ self.adj().T
        if shock is not None:
            nodes = nodes + shock
        s2 = torch.cat([nodes, z], 1)
        return s2 + 0.5 * torch.tanh(self.res(torch.cat([s2, u], 1)))

    def rollout(self, h, U, shocks=None):
        s = self.s0(h)
        S = [s]
        for k in range(WD.H):
            s = self.step(s, U[:, k], None if shocks is None else shocks[k])
            S.append(s)
        return torch.stack(S, 1)                                # [B, H+1, d]

    def forward(self, X, U, ST, sample=True, shocks=None, return_states=False):
        h = self.enc(X, sample) if self.arch == "rssm" else self.enc(X)
        if self.arch == "direct":
            g = self.mix(torch.cat([h, U.flatten(1)], 1))
            q = torch.stack([hd(torch.cat([h, g], 1)) for hd in self.heads], 1)
            return (q, None) if return_states else q
        S = self.rollout(h, U, shocks)
        B = X.shape[0]
        ar = torch.arange(B)
        outs = []
        for j, hd in enumerate(self.heads):
            sj = S[ar, ST[:, j].clamp(1, WD.H)]
            outs.append(hd(torch.cat([sj, sj - S[:, 0]], 1)))
        q = torch.stack(outs, 1)
        return (q, S) if return_states else q

    def latent(self, X, U):
        """decision-time latent for agents: [h-state s0 ; state after 2 steps (traded day)]."""
        with torch.no_grad():
            h = self.enc(X, False) if self.arch == "rssm" else self.enc(X)
            if self.arch == "direct":
                g = self.mix(torch.cat([h, U.flatten(1)], 1))
                return torch.cat([h, g], 1)
            S = self.rollout(h, U)
            return torch.cat([S[:, 0], S[:, 2]], 1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def matched_cfg(cfg):
    """direct twin with hidden width dh chosen so params are within +-5% of the WM recipe."""
    ref = nparams(WM(cfg))
    best = None
    for dh in range(8, 1025, 4):
        c = dict(cfg, arch="direct", dh=dh)
        n = nparams(WM(c))
        if best is None or abs(n - ref) < abs(best[1] - ref):
            best = (c, n)
    assert abs(best[1] - ref) / ref <= 0.05, (best[1], ref)
    return best[0]


def pinball(q, y):
    """q [B, T, NQ], y [B, T] (NaN masked) -> scalar mean pinball and per-sample-target [B, T]."""
    m = torch.isfinite(y)
    y0 = torch.where(m, y, torch.zeros_like(y))
    e = y0.unsqueeze(-1) - q
    l = torch.maximum(TQ * e, (TQ - 1) * e).mean(-1)
    l = torch.where(m, l, torch.zeros_like(l))
    return l.sum() / m.sum().clamp(min=1), l


from wm_harness import Norm, windows, fold_rows, predict, pinball_np  # noqa: E402,F401  (fixed harness)


def train_model(cfg, P, tr, seed, norm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    L = cfg.get("L", 20)
    m = WM(cfg)
    es = cfg.get("es", 0)
    if es:   # inner early stopping: the chronologically last es-fraction of the fold's train rows (with a 10-row gap)
        ntr = int(len(tr) * (1 - es))
        tr, va = tr[:ntr - 10], tr[ntr:]
        Xv, Uv, STv, Yv = windows(P, va, L, norm)
        best, best_state = float("inf"), None
    Xw, U, ST, Y = windows(P, tr, L, norm)
    budget = cfg.get("budget_s")
    steps = cfg.get("steps", 1500)
    lr0 = cfg.get("lr", 2e-3)
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=cfg.get("wd", 1e-4))
    g = torch.Generator().manual_seed(seed)
    bs = cfg.get("bs", 128)
    tw = torch.tensor(cfg.get("tw", [1.0] * NT), dtype=torch.float32)
    t0 = time.time()
    it = 0
    while True:
        frac = (time.time() - t0) / budget if budget else it / steps
        if frac >= 1:
            break
        # warmup 5% then cosine decay over the budget fraction
        lr = lr0 * min(1.0, frac / 0.05) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))) + 1e-5
        for gr in opt.param_groups:
            gr["lr"] = lr
        b = torch.randint(0, len(tr), (bs,), generator=g)
        q = m(Xw[b], U[b], ST[b])
        _, l = pinball(q, Y[b])
        mk = torch.isfinite(Y[b]).float()
        loss = ((l * tw).sum(0) / mk.sum(0).clamp(min=1)).sum() / tw.sum()
        if cfg["arch"] == "rssm":
            loss = loss + cfg.get("kl", 0.05) * m.enc.kl
        if cfg.get("l2graph", 0) and getattr(m, "graph", 0):
            loss = loss + cfg["l2graph"] * (m.ew ** 2).sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        it += 1
        if es and it % 50 == 0:
            m.eval()
            with torch.no_grad():
                vl = float(pinball(m(Xv, Uv, STv, sample=False), Yv)[0])
            m.train()
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    if es and best_state is not None:
        m.load_state_dict(best_state)
    m.n_steps = it
    m.eval()
    return m


