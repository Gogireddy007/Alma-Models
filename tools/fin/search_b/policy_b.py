"""Architect B agents (family 6) on the DEV-early inner folds, scored from the loop cache (nothing after 2024-06-30).

Policies (position in [-1, 1] per coin, decided at D 00:00 from the context only):
  voltgt  : sign(mom_k) * min(1, c * sigma* / sigma_B), sigma_B = sqrt(exp(median q_B)) of a B forecaster, with an
            optional no-trade band (keep yesterday's position if |target - prev| < band) to cut turnover.
  resid   : vol-target residual policy: pos = clip(base * exp(g) + h, -1, 1), (g, h) = small MLP on the B forecaster's
            latent-free summary (q quantiles - A, p_up, har block, mom, funding) trained on REAL past transitions of
            the training coins (label day <= fit_end) to maximise mean net return - lam * variance - kappa * CVaR_5%
            (risk-sensitive objective, differentiable through costs with the previous position = previous output).
Score: annualised net Sharpe on the 8 held-out coins (fixed 1/8 sleeves, costs + funding as harness.pnl).
"""
import os
import sys
import json
import time
import argparse
import traceback

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feats_b as FB     # noqa: E402
import loop_b as LB      # noqa: E402

TGT = 0.40 / np.sqrt(365)


def pnl(D, idx, pos, cols):
    """In-cache replica of harness.pnl (no delisting in the loop window). pos [n, m_all]."""
    r = np.nan_to_num(D["r"][idx][:, cols]); f = np.nan_to_num(D["fund"][idx][:, cols])
    alive = D["alive"][idx][:, cols] & np.isfinite(D["r"][idx][:, cols])
    p = np.where(alive, pos[:, cols], 0.0)
    prev = np.vstack([np.zeros((1, len(cols))), p[:-1]])
    net = p * r - p * f - np.abs(p - prev) * D["cost"][cols]
    n = net.mean(1)
    return n, np.abs(p - prev).mean(1)


def sharpe(n):
    return float(n.mean() / n.std(ddof=1) * np.sqrt(365)) if n.std() > 0 else 0.0


def mdd(n):
    w = np.cumprod(1 + n)
    return float(np.max(1 - w / np.maximum.accumulate(w)))


def voltgt(D, idx, q, k=4, c=1.0, band=0.0):
    sig = np.sqrt(np.exp(np.nan_to_num(q[:, :, 6], nan=np.log(TGT ** 2))))
    dirn = np.ones_like(sig) if k < 0 else np.sign(D["mom"][idx][:, :, k])      # k < 0: long-only vol target
    tgt = dirn * np.minimum(1.0, c * TGT / np.maximum(sig, 1e-6))
    if band <= 0:
        return tgt
    pos = np.zeros_like(tgt)
    prev = np.zeros(tgt.shape[1])
    for t in range(len(tgt)):
        p = np.where(np.abs(tgt[t] - prev) < band, prev, tgt[t])
        pos[t] = p
        prev = p
    return pos


def obs(D, idx, p, q):
    A = D["A"][idx][..., None]
    z = np.concatenate([np.nan_to_num(q - A)[..., ::3], np.nan_to_num(p)[..., None] - 0.5, D["har"][idx][..., :4], D["har"][idx][..., 8:],
                        np.tanh(D["mom"][idx][..., 1:] / 0.2), np.clip(D["fu_last"][idx][..., None] * 1e3, -5, 5)], -1)
    return z.astype(np.float32)


class Resid(nn.Module):
    def __init__(self, i, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i + 1, h), nn.Tanh(), nn.Linear(h, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, base):
        o = self.net(torch.cat([z, base.unsqueeze(-1)], -1))
        return torch.clamp(base * torch.exp(torch.clamp(o[..., 0], -2, 1)) + 0.3 * torch.tanh(o[..., 1]), -1, 1)


def train_resid(D, tr_idx, p, q, base, cols, cfg, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z = torch.from_numpy(obs(D, tr_idx, p, q)[:, cols])
    b = torch.from_numpy(base[:, cols].astype(np.float32))
    r = torch.from_numpy(np.nan_to_num(D["r"][tr_idx][:, cols]).astype(np.float32))
    f = torch.from_numpy(np.nan_to_num(D["fund"][tr_idx][:, cols]).astype(np.float32))
    al = torch.from_numpy((D["alive"][tr_idx][:, cols] & np.isfinite(D["r"][tr_idx][:, cols])).astype(np.float32))
    cost = torch.from_numpy(D["cost"][cols].astype(np.float32))
    pol = Resid(z.shape[-1], cfg.get("h", 32))
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.get("lr", 3e-3))
    H = cfg.get("H", 60)
    lam, kap = cfg.get("lam", 1.0), cfg.get("kappa", 0.0)
    for s in range(cfg.get("steps", 300)):
        st = rng.integers(0, len(tr_idx) - H, 16)
        sl = torch.from_numpy(np.stack([np.arange(a, a + H) for a in st]))
        pos = pol(z[sl], b[sl]) * al[sl]
        prev = torch.cat([torch.zeros_like(pos[:, :1]), pos[:, :-1]], 1)
        net = pos * r[sl] - pos * f[sl] - (pos - prev).abs() * cost
        port = net.mean(-1) / TGT                                   # [16, H] in vol-budget units
        loss = -port.mean() + lam * port.var()
        if kap > 0:
            k = max(1, int(0.05 * port.numel()))
            loss = loss + kap * (-torch.topk(-port.flatten(), k).values.mean())
        opt.zero_grad(); loss.backward(); opt.step()
    pol.eval()
    return pol


def fc_preds(D, fc_cfg, fe, seed, idx):
    m = LB.train_model(D, fc_cfg, fe, seed, log=lambda *a: None)
    return LB.predict([m], D, idx), m


def run(name, cfg, seed=0):
    D = LB.cache()
    ho = np.nonzero(~D["train_mask"])[0]
    tr = np.nonzero(D["train_mask"])[0]
    srs, rows = [], []
    try:
        for fk, (fe, a, b) in LB.FOLDS.items():
            idx = LB.fold_idx(D, a, b)
            idx = idx[np.isfinite(D["r"][idx]).any(1)]
            fcc = cfg["fc"]
            # forecaster on the fold (cached by name)
            pth = os.path.join(FB.OUTB, "preds", f"pol_fc_{cfg['fc_name']}_s{seed}_{fk}.npz")
            days = D["days"]
            trw = np.nonzero((days < np.datetime64(fe)) & (days >= np.datetime64("2021-01-01")))[0]  # RT-9 purge
            if os.path.exists(pth):
                z = np.load(pth); p, q, pt, qt = z["p"], z["q"], z["pt"], z["qt"]
            else:
                (p, q), m = fc_preds(D, fcc, fe, seed, idx)
                pt, qt = LB.predict([m], D, trw, train=True)        # in-sample on the fit window (policy training inputs)
                np.savez(pth, p=p, q=q, pt=pt, qt=qt)
            base = voltgt(D, idx, q, cfg.get("k", 4), cfg.get("c", 1.0), cfg.get("band", 0.0))
            if cfg.get("type") == "resid":
                baset = voltgt(D, trw, qt, cfg.get("k", 4), cfg.get("c", 1.0), 0.0)
                pol = train_resid(D, trw, pt, qt, baset, tr, cfg, seed)
                with torch.no_grad():
                    pos = pol(torch.from_numpy(obs(D, idx, p, q)), torch.from_numpy(base.astype(np.float32))).numpy()
                pos = pos * D["alive"][idx]
            else:
                pos = base
            n, turn = pnl(D, idx, pos, ho)
            na, _ = pnl(D, idx, pos, np.arange(len(D["symbols"])))
            sr = sharpe(n)
            srs.append(sr)
            sp = float(n.mean() / n.std(ddof=1))
            LB.log_row({"kind": "agent_b", "name": name, "config": cfg, "seed": seed, "fold": fk, "status": "ok",
                        "SR_heldout": sr, "SR_all": sharpe(na), "turn_heldout": float(turn.mean()),
                        "mdd_heldout": mdd(n), "note": f"sr_period {sp:.5f}"})
        LB.log_row({"kind": "agent_b", "name": name, "config": cfg, "seed": seed, "fold": "EM", "status": "ok",
                    "SR_heldout": float(np.mean(srs)), "note": " ".join(f"{x:.3f}" for x in srs)})
        print(f"{name}: EM SR_ho {np.mean(srs):.3f} " + " ".join(f"{x:.3f}" for x in srs), flush=True)
    except Exception as ex:
        traceback.print_exc()
        LB.log_row({"kind": "agent_b", "name": name, "config": cfg, "seed": seed, "fold": "EM", "status": "crash",
                    "note": f"{type(ex).__name__}: {str(ex)[:200]}"})


if __name__ == "__main__":
    torch.set_num_threads(3)
    for ln in open(sys.argv[1]):
        if ln.strip():
            j = json.loads(ln)
            run(j["name"], j["cfg"], j.get("seed", 0))
