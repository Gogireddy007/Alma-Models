"""Commodity world-model search: shared windows, features, metric and training loop (docs/commodity_search.md).

Everything that must be identical across candidates lives here: the DEV backtest origins, the random-walk
reference, the metric and the pre-declared selection score, window sampling (targets strictly before the
training cut), the feature map, and the fixed-time-budget training loop.

Selection score (pre-declared, lower is better):
    score = mean over h in {1, 5, 20} of  0.5 * (CRPS_h / CRPS_h(RW)) + 0.5 * (1 - DA_h)
CRPS_h is approximated on log price by 2 * mean pinball over the 9 quantiles 0.1..0.9 (the fev/WQL grid).
RW = random walk on log price with Gaussian increments, sigma = RMS of the last 250 WTI log returns,
quantiles = last + z_q sigma sqrt(h). DA_h = share of origins where sign(median forecast - last) equals
sign(realised change); a zero median or zero change counts 0.5. RW therefore scores exactly 0.75.
Reported but not in the score: vol-direction accuracy (predicted next-20d vol > past-20d realised vol, the
prediction read off the h=20 quantile spread, sigma_hat = (q.9 - q.1) / (2 * 1.2816 * sqrt(20))).
"""
import math
import os
import time

import numpy as np
import pandas as pd
import torch

import wm_data as D

Q9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
Z9 = np.array([-1.2815516, -0.8416212, -0.5244005, -0.2533471, 0.0, 0.2533471, 0.5244005, 0.8416212, 1.2815516],
              dtype=np.float32)
HS = (1, 5, 20)
BT_START = pd.Timestamp("2023-01-01")      # DEV backtest start == training cut for the search models
L_CTX = 256
H_MAX = 20

# the cross-commodity subset (full coverage from 1997-01 after the causal 10-day ffill)
XC_SERIES = ["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "DGASNYH", "DHOILNYH", "DJFUELUSGULF", "DPROPANEMBTX"]


class Panel:
    """Log-price panel (DEV by default). lp[T, N], dates[T]; column 0 is WTI."""

    def __init__(self, split="dev", series=None):
        self.dates, self.names, self.lp = D.log_panel(split, series)
        self.split = split
        assert self.names[0] == D.TARGET
        self.T, self.N = self.lp.shape

    def idx(self, ts):
        return int(np.searchsorted(self.dates.values, np.datetime64(ts)))


def backtest_origins(P, start=BT_START, end=D.DEV_END, every=5, H=max(HS)):
    """Weekly origins (every 5th WTI trading day from the first day >= start); the h=H target must be <= end."""
    i0 = P.idx(start)
    last = P.idx(end + pd.Timedelta(days=1)) - 1           # last index with date <= end
    o = np.arange(i0, last - H + 1, every)
    assert P.dates[o[-1] + H] <= end
    return o


def train_origins(P, cut=BT_START, L=L_CTX, H=H_MAX, cols=None, start_full=True):
    """Origins t whose whole window [t-L, t+H] lies strictly before `cut` and is observed in `cols`."""
    c = P.idx(cut)                                         # first index with date >= cut
    cols = cols if cols is not None else [0]
    ok = ~np.isnan(P.lp[:, cols]).any(1)
    # a window is valid when every row in [t-L, t+H] is observed
    bad = np.concatenate([[0], np.cumsum(~ok)])
    t = np.arange(L, c - H)
    good = (bad[t + H + 1] - bad[t - L]) == 0
    o = t[good]
    assert len(o) == 0 or o.max() + H < c and P.dates[o.max() + H] < pd.Timestamp(cut)
    return o


def windows(P, origins, L=L_CTX, H=H_MAX, cols=(0,)):
    """[B, L+1+H, len(cols)] log prices for origins (row L is the origin)."""
    cols = list(cols)
    idx = origins[:, None] + np.arange(-L, H + 1)[None, :]
    return P.lp[idx][:, :, cols]


def ctx_windows(P, origins, L=L_CTX, cols=(0,)):
    idx = origins[:, None] + np.arange(-L, 1)[None, :]
    return P.lp[idx][:, :, list(cols)]


# ------------------------------------------------------------------------------------------------- features
def features(lp_ctx, spreads=False):
    """lp_ctx: [B, L+1, N] log prices (column 0 = target). Returns (x [B, L, F], scale [B, N], extras).
    Per-window, per-series scale = RMS of the window's log returns (causal: only context values)."""
    r = lp_ctx[:, 1:] - lp_ctx[:, :-1]
    m = ~torch.isnan(r)
    r0 = torch.where(m, r, torch.zeros_like(r))
    s = (r0.pow(2).sum(1) / m.sum(1).clamp(min=1)).sqrt().clamp(min=1e-4)       # [B, N]
    z = (r0 / s[:, None]).clamp(-10, 10)
    ls = (torch.log(s) - math.log(0.02))[:, None].expand_as(z)
    parts = [z, m.float(), ls]
    ext = {"s": s}
    if spreads and lp_ctx.shape[2] > 1:
        sp = lp_ctx[:, 1:, 1:] - lp_ctx[:, 1:, :1]                               # log ratio to WTI
        mu = torch.nanmean(sp, 1, keepdim=True)
        sd = torch.sqrt(torch.nanmean((sp - mu) ** 2, 1, keepdim=True)).clamp(min=1e-3)
        parts.append(torch.nan_to_num(((sp - mu) / sd).clamp(-6, 6)))
        ext["sp_mu"], ext["sp_sd"] = mu, sd
    return torch.cat(parts, -1), s, ext


def n_features(N, spreads=False):
    return 3 * N + (N - 1 if spreads and N > 1 else 0)


# ------------------------------------------------------------------------------------------------- metric
def pinball(q, y, levels=Q9):
    """q [B, 9, H], y [B, H] -> [B, H] mean pinball over levels."""
    lv = levels[None, :, None]
    e = y[:, None, :] - q
    return np.maximum(lv * e, (lv - 1) * e).mean(1)


def rw_quantiles(P, origins, H=H_MAX, n=250):
    r = np.diff(P.lp[:, 0])
    sig = np.array([np.sqrt(np.mean(r[t - n:t] ** 2)) for t in origins])     # r[t-1] = lp[t]-lp[t-1]
    return Z9[None, :, None] * sig[:, None, None] * np.sqrt(np.arange(1, H + 1))[None, None, :]


def evaluate(P, origins, qhat, H=H_MAX):
    """qhat [B, 9, H] quantiles of lp[t+h] - lp[t]. Returns dict of metrics and the selection score."""
    qhat = np.sort(qhat, axis=1)
    y = np.stack([P.lp[origins + h, 0] - P.lp[origins, 0] for h in range(1, H + 1)], 1)
    assert P.dates[origins.max() + H] <= (D.DEV_END if P.split == "dev" else D.VAL_END)
    qrw = rw_quantiles(P, origins, H)
    crps, crw = 2 * pinball(qhat, y), 2 * pinball(qrw, y)
    out = {}
    sc = []
    for h in HS:
        if h > H:
            continue
        med = qhat[:, 4, h - 1]
        yy = y[:, h - 1]
        hit = np.where((med == 0) | (yy == 0), 0.5, (np.sign(med) == np.sign(yy)).astype(float))
        ratio = crps[:, h - 1].mean() / crw[:, h - 1].mean()
        out[f"crps{h}"] = float(crps[:, h - 1].mean())
        out[f"ratio{h}"] = float(ratio)
        out[f"da{h}"] = float(hit.mean())
        sc.append(0.5 * ratio + 0.5 * (1 - hit.mean()))
    out["score"] = float(np.mean(sc))
    if H >= 20:
        r = np.diff(P.lp[:, 0])
        past = np.array([np.sqrt(np.mean(r[t - 20:t] ** 2)) for t in origins])
        fut = np.array([np.sqrt(np.mean(r[t:t + 20] ** 2)) for t in origins])
        pred = (qhat[:, 8, 19] - qhat[:, 0, 19]) / (2 * 1.2815516 * math.sqrt(20))
        out["volda"] = float(((pred > past) == (fut > past)).mean())
        cov = ((y[:, 19] >= qhat[:, 0, 19]) & (y[:, 19] <= qhat[:, 8, 19])).mean()
        out["cov80_20"] = float(cov)
    out["n"] = int(len(origins))
    return out


def score_ci(P, origins, qhat, B=500, seed=0, H=H_MAX):
    """Block-bootstrap (blocks of 4 weekly origins) 90% interval of the selection score."""
    rng = np.random.default_rng(seed)
    n = len(origins)
    nb = int(math.ceil(n / 4))
    vals = []
    for _ in range(B):
        st = rng.integers(0, n - 3, nb)
        ii = (st[:, None] + np.arange(4)[None]).reshape(-1)[:n]
        vals.append(evaluate(P, origins[ii], qhat[ii], H)["score"])
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


# ------------------------------------------------------------------------------------------------- training
def set_threads(n=2):
    torch.set_num_threads(n)


def train_loop(model, P, origins, cols, minutes, lr=2e-3, batch=64, seed=0, H=H_MAX, L=L_CTX, wd=1e-4,
               max_steps=None, log=print, pool=False):
    """Fixed wall-clock budget (or max_steps for a reproducible refit). Cosine LR over the budget.
    pool=True: each sample picks a random target column among `cols` rows that are fully observed
    (univariate models trained on every energy series)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    t0 = time.time()
    budget = minutes * 60
    step, ema = 0, None
    model.train()
    while minutes > 0 or max_steps:
        el = time.time() - t0
        frac = (step / max_steps) if max_steps else el / budget
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1))))
        if pool:
            c = rng.integers(len(origins))
            col, og = origins[c]
            o = og[rng.integers(0, len(og), batch)]
            w = torch.from_numpy(windows(P, o, L, H, (col,))).float()
        else:
            o = origins[rng.integers(0, len(origins), batch)]
            w = torch.from_numpy(windows(P, o, L, H, cols)).float()
        loss, info = model.loss(w, L, H)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
        if step % 200 == 0:
            log(f"  step {step} t={el:.0f}s loss {ema:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in info.items()))
    model.eval()
    return step, time.time() - t0, ema


@torch.no_grad()
def predict_origins(model, P, origins, cols, L=L_CTX, H=H_MAX, seed=123, chunk=256):
    torch.manual_seed(seed)
    out = []
    for s in range(0, len(origins), chunk):
        w = torch.from_numpy(ctx_windows(P, origins[s:s + chunk], L, cols)).float()
        out.append(model.quantiles(w, H).numpy())
    return np.concatenate(out)
