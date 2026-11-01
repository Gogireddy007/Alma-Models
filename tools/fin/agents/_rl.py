"""
Shared RL code for the Alma-Fin agents (architect; docs/fin_search.md section 1). Not auto-registered (leading _).

Observation (all three agents): [h, z] = the frozen world model's filtered latent at the end of D-1, b = log sigma* -
A/2 (vol budget vs the asset's 30-day vol level), a_prev (previous action). Action a in [-1, 1] = fraction of the vol
budget; position = clip(a * sigma*/sigma_WM, -1, 1), sigma_WM = exp(median log RV(D) / 2) from the world model.
Reward of one day in units of sigma*: pos*r_n*e^-b - (cost*|pos - pos_prev| + pos*funding) * e^-(b + A/2)
                                      - lam * (pos*r_n*e^-b)^2
  wm_agent: Dreamer-style actor-critic, trajectories IMAGINED by the world model (prior rollouts, 4 six-hour sub-steps
            per day; r_n ~ the WM's predictive return distribution, sigma_WM from its predictive log RV), dynamics
            backprop of lambda-returns into a deterministic actor, critic on lambda-returns with an EMA target.
  ppo:      PPO (Gaussian, tanh mean), REAL DEV transitions only (historical next-day returns, funding, WM forecasts).
  dqn:      DQN over 5 sizes {-1, -0.5, 0, 0.5, 1}, REAL DEV transitions only (all-action targets, target network).
Same observation, horizon (10 days), batch (256 trajectories x 10 days), gradient steps, seeds, lam, gamma.
"""
import os
import sys
import copy
import math

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FIN))
sys.path.insert(0, FIN)
import wm_core as WC     # noqa: E402

HOR, BS, GAMMA, LAMBDA_GAE = 10, 256, 0.9, 0.95
SIZES = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])


def obs_dim(d, zd=WC.ZD):
    return d + zd + 2


class Actor(nn.Module):
    def __init__(self, d, zd=WC.ZD):
        super().__init__()
        self.net = WC.mlp(obs_dim(d, zd), 1, 64)
        self.log_std = nn.Parameter(torch.tensor(-0.7))        # used by PPO only

    def forward(self, o):
        return torch.tanh(self.net(o)[:, 0])


class Critic(nn.Module):
    def __init__(self, d, zd=WC.ZD):
        super().__init__()
        self.net = WC.mlp(obs_dim(d, zd), 1, 64)

    def forward(self, o):
        return self.net(o)[:, 0]


class QNet(nn.Module):
    def __init__(self, d, zd=WC.ZD):
        super().__init__()
        self.net = WC.mlp(obs_dim(d, zd), len(SIZES), 128)       # params within 5% of actor + critic

    def forward(self, o):
        return self.net(o)


def obs(h, z, b, a_prev):
    return torch.cat([h, z, b[:, None], a_prev[:, None]], -1)


def reward(a, pos_prev, ratio, rn, b, A, cost, fund, lam):
    pos = torch.clamp(a * ratio, -1, 1)
    eb = torch.exp(-b)
    pnl = pos * rn * eb
    r = pnl - (cost * (pos - pos_prev).abs() + pos * fund) * torch.exp(-(b + A / 2)) - lam * pnl ** 2
    return r, pos


# --------------------------------------------------------------------------- real states (frozen WM member)
@torch.no_grad()
def real_states(member, D, bs=2048):
    """Filtered states s_{D-1}, the WM's day-ahead log-RV median ratio and per-sample constants for every
    training sample of dataset D (all inputs <= D-1; rn/labels are the realised outcomes used as REAL rewards)."""
    hs, zs, q50 = [], [], []
    Xall = D["X"][:, -member.w:].copy()
    if not member.mf:
        Xall[:, :, WC.F_MLV] = 0.0
        Xall[:, :, WC.F_MR] = 0.0
    auxall = WC.with_dow(D["aux"], D["day_str"], member.dow) if "day_str" in D else D["aux"]
    for i in range(0, len(D["X"]), bs):
        X = torch.from_numpy(Xall[i:i + bs])
        aux = torch.from_numpy(auxall[i:i + bs])
        h, z, _, _ = member.filter(X, sample=False, losses=False)
        S = WC.S_INF if member.abl == "none" else 1
        hins, _ = member.head_input(h, z, aux, sample=False, S=S,
                                    shared_noise=WC.shared_noise() if member.abl == "none" else None)
        q = torch.stack([member.head_out(x)[0] for x in hins]).mean(0)
        hs.append(h); zs.append(z); q50.append(q[:, WC.QS.index(0.5)])
    b = torch.from_numpy(D["aux"][:, 2])
    lv = torch.cat(q50)                                     # median of log RV(D) - A
    return {"h": torch.cat(hs), "z": torch.cat(zs), "b": b, "ratio": torch.exp(b - lv / 2),
            "A": torch.from_numpy(D["A"]), "cost": torch.from_numpy(D["cost"]),
            "fund": torch.from_numpy(D["X"][:, -1, WC.F_FU] / 1e3),          # last KNOWN rate (Dreamer starts)
            # REAL rewards (PPO/DQN): harness-style fill-to-fill return and realised funding of day D (RT-8)
            "rn": torch.from_numpy(np.nan_to_num(D["rn_fill"] if "rn_fill" in D else D["rn"])),
            "fund_real": torch.from_numpy(D["fund_real"] if "fund_real" in D else D["X"][:, -1, WC.F_FU] / 1e3),
            "coin": D["coin"], "day": D["day"],
            "rvalid": np.isfinite(D["rn_fill"]) if "rn_fill" in D else np.ones(len(D["X"]), bool)}


def chains(S, H=HOR + 1):
    """indices of samples i0..i0+H-1 = H consecutive decision days of the same coin."""
    key = {(int(c), int(d)): k for k, (c, d) in enumerate(zip(S["coin"], S["day"]))}
    out = []
    for k, (c, d) in enumerate(zip(S["coin"], S["day"])):
        idx = [key.get((int(c), int(d) + t)) for t in range(H)]
        # a reward is earned at positions 0..H-2; its exit fill must lie on or before the fold's last label day
        if all(v is not None for v in idx) and all(S["rvalid"][v] for v in idx[:-1]):
            out.append(idx)
    return torch.tensor(out)


# --------------------------------------------------------------------------- Dreamer-style (imagination)
def train_dreamer(member, S, lam, seed, steps, lr=3e-4, log=print, hor=HOR, gamma=GAMMA, cm=1.0):
    HOR, GAMMA = hor, gamma
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = member.d
    actor, critic = Actor(d, member.zd), Critic(d, member.zd)
    tgt = copy.deepcopy(critic)
    for p in tgt.parameters():
        p.requires_grad_(False)
    oa, oc = torch.optim.Adam(actor.parameters(), lr=lr), torch.optim.Adam(critic.parameters(), lr=lr)
    N = len(S["h"])
    for p in member.parameters():
        p.requires_grad_(False)
    for s in range(steps):
        i = torch.from_numpy(rng.integers(0, N, BS))
        with torch.no_grad():                               # market dynamics do not depend on our actions
            h, z = S["h"][i], S["z"][i]
            b, A, cost, fund = S["b"][i], S["A"][i], S["cost"][i], S["fund"][i]
            traj = [(h, z, S["ratio"][i])]
            rns = []
            for t in range(HOR):
                noise = [torch.randn(BS, member.zd, generator=g) for _ in range(member.sub)]
                h, z, hs, zs, _ = member.day_roll(h, z, noise)
                pr = member.pred(torch.cat([torch.stack(hs).mean(0), torch.stack(zs).mean(0)], -1))
                rn = pr[:, 0] + pr[:, 1].clamp(-5, 3).exp() * torch.randn(BS, generator=g)
                rns.append(rn.clamp(-6, 6))
                traj.append((h, z, torch.exp(b - pr[:, 2] / 2)))
        a_prev = torch.from_numpy(rng.uniform(-1, 1, BS).astype(np.float32))
        pos_prev = torch.clamp(a_prev * traj[0][2], -1, 1)
        R, O = [], []
        for t in range(HOR):
            o = obs(traj[t][0], traj[t][1], b, a_prev)
            a = actor(o)
            r, pos_prev = reward(a, pos_prev, traj[t][2], rns[t], b, A, cost * cm, fund, lam)
            R.append(r); O.append(o)
            a_prev = a
        oH = obs(traj[HOR][0], traj[HOR][1], b, a_prev)
        O.append(oH)
        V = [tgt(o) for o in O]
        G = V[-1]
        rets = []
        for t in reversed(range(HOR)):
            G = R[t] + GAMMA * ((1 - LAMBDA_GAE) * V[t + 1] + LAMBDA_GAE * G)
            rets.append(G)
        rets = torch.stack(rets[::-1])
        la = -rets.mean()
        oa.zero_grad(); la.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); oa.step()
        vc = torch.stack([critic(o.detach()) for o in O[:-1]])
        lc = ((vc - rets.detach()) ** 2).mean()
        oc.zero_grad(); lc.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); oc.step()
        with torch.no_grad():
            for pt, pc in zip(tgt.parameters(), critic.parameters()):
                pt.mul_(0.98).add_(0.02 * pc)
        if s % 250 == 0 or s == steps - 1:
            log(f"dreamer step {s} ret {float(rets[0].mean()):.4f} critic {float(lc):.4f}")
    return {"actor": actor.state_dict(), "critic": critic.state_dict()}


# --------------------------------------------------------------------------- PPO (real transitions)
def train_ppo(d, S, lam, seed, steps, lr=3e-4, epochs=4, clip=0.2, log=print, zd=WC.ZD, hor=HOR, gamma=GAMMA,
              cm=1.0):
    HOR, GAMMA = hor, gamma
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor, critic = Actor(d, zd), Critic(d, zd)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    CH = chains(S, HOR + 1)
    it = 0
    for s in range(0, steps, epochs):
        c = CH[torch.from_numpy(rng.integers(0, len(CH), BS))]          # [BS, HOR+1]
        with torch.no_grad():
            a_prev = torch.from_numpy(rng.uniform(-1, 1, BS).astype(np.float32))
            pos_prev = torch.clamp(a_prev * S["ratio"][c[:, 0]], -1, 1)
            O, Aa, LP, R = [], [], [], []
            for t in range(HOR + 1):
                k = c[:, t]
                o = obs(S["h"][k], S["z"][k], S["b"][k], a_prev)
                O.append(o)
                if t == HOR:
                    break
                mu = actor(o)
                std = actor.log_std.exp()
                a = mu + std * torch.randn(BS, generator=g)
                LP.append(-0.5 * ((a - mu) / std) ** 2 - actor.log_std)
                ae = a.clamp(-1, 1)
                r, pos_prev = reward(ae, pos_prev, S["ratio"][k], S["rn"][k], S["b"][k], S["A"][k], S["cost"][k] * cm,
                                     S["fund_real"][k], lam)
                Aa.append(a); R.append(r)
                a_prev = ae
            V = [critic(o) for o in O]
            adv, gae = [], torch.zeros(BS)
            for t in reversed(range(HOR)):
                delta = R[t] + GAMMA * V[t + 1] - V[t]
                gae = delta + GAMMA * LAMBDA_GAE * gae
                adv.append(gae)
            adv = torch.stack(adv[::-1])
            ret = adv + torch.stack(V[:-1])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        Ot, At, LPt = torch.stack(O[:-1]), torch.stack(Aa), torch.stack(LP)
        for _ in range(epochs):
            mu = actor(Ot.reshape(-1, Ot.shape[-1])).reshape(HOR, BS)
            std = actor.log_std.exp()
            lp = -0.5 * ((At - mu) / std) ** 2 - actor.log_std
            ratio = torch.exp(lp - LPt)
            lpi = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            lv = ((critic(Ot.reshape(-1, Ot.shape[-1])).reshape(HOR, BS) - ret) ** 2).mean()
            loss = lpi + 0.5 * lv
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0); opt.step()
            it += 1
        if s % 248 == 0:
            log(f"ppo step {s} meanR {float(torch.stack(R).mean()):.4f} std {float(actor.log_std.exp()):.3f}")
    return {"actor": actor.state_dict(), "critic": critic.state_dict(), "grad_steps": it}


# --------------------------------------------------------------------------- DQN (real transitions)
def train_dqn(d, S, lam, seed, steps, lr=3e-4, log=print, zd=WC.ZD, tgt_every=50, hor=HOR, gamma=GAMMA, cm=1.0):
    HOR, GAMMA = hor, gamma
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = QNet(d, zd)
    tgt = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    CH = chains(S, 2)
    n = BS * HOR
    for s in range(steps):
        c = CH[torch.from_numpy(rng.integers(0, len(CH), n))]
        k0, k1 = c[:, 0], c[:, 1]
        ap_idx = torch.from_numpy(rng.integers(0, len(SIZES), n))
        a_prev = SIZES[ap_idx]
        pos_prev = torch.clamp(a_prev * S["ratio"][k0], -1, 1)
        o = obs(S["h"][k0], S["z"][k0], S["b"][k0], a_prev)
        with torch.no_grad():
            Rs, Ys = [], []
            for ai, a in enumerate(SIZES):
                aa = a.expand(n)
                r, _ = reward(aa, pos_prev, S["ratio"][k0], S["rn"][k0], S["b"][k0], S["A"][k0], S["cost"][k0] * cm,
                              S["fund_real"][k0], lam)
                o1 = obs(S["h"][k1], S["z"][k1], S["b"][k1], aa)
                Ys.append(r + GAMMA * tgt(o1).max(1).values)
            Y = torch.stack(Ys, 1)
        loss = ((q(o) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
        if s % tgt_every == tgt_every - 1:
            tgt.load_state_dict(q.state_dict())
        if s % 250 == 0 or s == steps - 1:
            log(f"dqn step {s} loss {float(loss):.5f}")
    return {"q": q.state_dict()}


# --------------------------------------------------------------------------- inference wrapper
class RLAgent:
    """act(ctx): WM features -> each member's filtered states over the window -> a_prev recomputed by running the
    policy over the last 10 window states from a = 0 (deterministic, context only) -> a_D -> position."""
    kind = "agent"

    def __init__(self, kind, members, policies, meta):
        self.algo, self.members, self.meta = kind, members, dict(meta)
        self.nets = []
        for m, pol in zip(members, policies):
            if kind == "dqn":
                net = QNet(m.d, m.zd); net.load_state_dict(pol["q"])
            else:
                net = Actor(m.d, m.zd); net.load_state_dict(pol["actor"])
            net.eval()
            self.nets.append(net)
        torch.set_num_threads(3)

    def _pi(self, net, o):
        if self.algo == "dqn":
            return SIZES[net(o).argmax(1)]
        return net(o)

    @torch.no_grad()
    def act(self, ctx):
        X, aux, A, ok = WC.make_features(ctx["daily"], ctx.get("hourly"), ctx["market"], self.members[0].w,
                                         self.members[0].mf)
        aux = WC.with_dow(aux, ctx["day"], self.members[0].dow)
        q, _, _ = WC.forecast_members(self.members, X, aux, A)
        sig_t = WC.VOL_TARGET[ctx["market"]]
        ratio = torch.from_numpy(np.exp(np.log(sig_t) - q[:, WC.QS.index(0.5)] / 2).astype(np.float32))
        Xt = torch.from_numpy(X)
        b = torch.from_numpy(aux[:, 2])
        pos = []
        for m, net in zip(self.members, self.nets):
            _, _, _, st = m.filter(Xt, sample=False, keep_states=True, losses=False)
            a = torch.zeros(len(b))
            for t in range(len(st) - 11, len(st) - 1):
                a = self._pi(net, obs(st[t][0], st[t][1], b, a))
            a = self._pi(net, obs(st[-1][0], st[-1][1], b, a))
            pos.append(torch.clamp(a * ratio, -1, 1).numpy())
        p = np.mean(pos, 0).astype(np.float64)
        p[~ok] = 0.0
        return np.nan_to_num(p)


def save_policies(path, policies, meta):
    torch.save({"meta": meta, "policies": policies}, path)


def load_policies(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["policies"], ck["meta"]
