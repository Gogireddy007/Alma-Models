"""
Alma-Live Gas policies (docs/live_search.md 0.6, 0.9 pillar 4). All share one observation: the frozen WM latent at
decision time (wm_core.WM.latent) + the previous position, and one execution model: target weight w_t decided at
decision time t, filled at the next open, P&L w_t * r2 - cost * |w_t - w_{t-1}| (r2 = open(d+1) -> open(d+2)).

  wm_agent : Dreamer-style actor trained purely in imagination. Imagined trajectories start from real training
             latents; the WM rollout states at calendar steps 2..H give per-day return distributions (the traded-
             day head applied along the rollout), returns are sampled from those quantiles (reparameterised through
             a random level), actions are sized by the WM-predicted vol, reward = net-of-cost P&L - risk penalty.
             Value bootstrap: a critic V(latent_H, a_H) trained on the imagined returns (lambda-return).
  ppo      : PPO on REAL transitions (realized r2), same observation, net, steps.
  dqn      : DQN on REAL transitions, 5 discrete actions {-1, -.5, 0, .5, 1}, same observation.
  voltgt   : non-RL rule on WM forecasts: w = sign(mu_hat) * min(tv / sigma_hat, CAP).
"""
import numpy as np
import torch
import torch.nn as nn

import wm_core as C

TV = 0.20 / np.sqrt(252)
CAP = 1.0          # gross <= 100% of equity in UNG (research 5.2)
COST = 1e-3        # 10 bp per side (A2); 2x / 3x reported
ACTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def net(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def wm_obs(m, P, rows, norm, L):
    """latent [n, dz], r2 quantiles in real units [n, NQ], sigma_hat, mu_hat, and rollout r2-head quantiles along
    calendar steps 2..H [n, H-1, NQ] (real units)."""
    out_z, out_q, out_roll = [], [], []
    j2 = C.WD.TARGETS.index("r1")   # traded day: UNG close(d) -> close(d+1)
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            Xw, U, ST, _ = C.windows(P, rows[i:i + 512], L, norm)
            q, S = m(Xw, U, ST, sample=False, return_states=True)
            out_z.append(m.latent(Xw, U).numpy())
            out_q.append(q[:, j2].numpy())
            if S is not None:
                hd = m.heads[j2]
                rr = [hd(torch.cat([S[:, k], S[:, k] - S[:, 0]], 1)).numpy() for k in range(2, C.WD.H + 1)]
                out_roll.append(np.stack(rr, 1))
            else:
                out_roll.append(np.repeat(q[:, j2].numpy()[:, None], C.WD.H - 1, 1))
    ym, ys = norm.ym[j2], norm.ys[j2]
    q = np.sort(np.concatenate(out_q), -1) * ys + ym
    roll = np.sort(np.concatenate(out_roll), -1) * ys + ym
    mu = q[:, C.NQ // 2]
    sig = np.maximum((q[:, -1] - q[:, 0]) / 3.29, 1e-3)
    return np.concatenate(out_z), q, roll, mu, sig


def pnl(w, r, cost=COST):
    r = np.where(np.isfinite(r), r, 0.0)
    dw = np.abs(np.diff(np.concatenate([[0.0], w])))
    return w * r - cost * dw


def sharpe(p):
    s = p.std()
    return float(p.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def size(a, sig):
    return np.clip(a * TV / sig, -CAP, CAP)


# ------------------------------------------------------------------ Dreamer-style imagination actor
def train_wm_agent(Z, roll, sig, steps=1500, lam=2.0, seed=0, H=5, gamma=0.95):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dz = Z.shape[1]
    actor = net(dz + 1, 1)
    critic = net(dz + 1, 1)
    oa = torch.optim.Adam(actor.parameters(), lr=1e-3)
    oc = torch.optim.Adam(critic.parameters(), lr=1e-3)
    Zt = torch.tensor(Z, dtype=torch.float32)
    Rq = torch.tensor(roll, dtype=torch.float32)          # [n, H-1, NQ] imagined per-day return quantiles
    St = torch.tensor(sig, dtype=torch.float32)
    zs = Zt.std(0) + 1e-6
    Zt = Zt / zs
    n = len(Z)
    Hh = min(H, Rq.shape[1])
    for it in range(steps):
        b = torch.randint(0, n, (256,), generator=g)
        a_prev = torch.rand(256, 1, generator=g) * 2 - 1
        ret, disc = 0.0, 1.0
        for k in range(Hh):
            a = torch.tanh(actor(torch.cat([Zt[b], a_prev], 1)))
            w = (a[:, 0] * TV / St[b]).clamp(-CAP, CAP)
            wp = (a_prev[:, 0] * TV / St[b]).clamp(-CAP, CAP)
            # reparameterised sample from the imagined quantile distribution of day k
            lv = torch.rand(256, generator=g) * (C.NQ - 1)
            lo = lv.floor().long(); fr = lv - lo
            q = Rq[b, k]
            r = q[torch.arange(256), lo] * (1 - fr) + q[torch.arange(256), (lo + 1).clamp(max=C.NQ - 1)] * fr
            var = ((q[:, -1] - q[:, 0]) / 3.29) ** 2
            rew = w * r - COST * (w - wp).abs() - lam * w ** 2 * var
            ret = ret + disc * rew * 100
            disc *= gamma
            a_prev = a
            # latent of later imagined days: the same decision latent (the WM state is summarised in Rq)
        v = critic(torch.cat([Zt[b], a_prev.detach()], 1))[:, 0]
        loss_a = -(ret + disc * v.detach()).mean()
        oa.zero_grad(); loss_a.backward(); oa.step()
        loss_c = ((v - ret.detach()) ** 2).mean()
        oc.zero_grad(); loss_c.backward(); oc.step()
    return lambda Z_, a_prev: torch.tanh(actor(torch.cat([torch.tensor(Z_ / zs.numpy(), dtype=torch.float32),
                                                          torch.tensor(a_prev, dtype=torch.float32)], 1))).detach().numpy()[:, 0]


def run_policy(pol, Z, sig):
    """sequential: a_t = pol(z_t, a_{t-1})."""
    a = np.zeros(len(Z))
    prev = 0.0
    for i in range(len(Z)):
        a[i] = pol(Z[i:i + 1], np.array([[prev]], np.float32))[0]
        prev = a[i]
    return size(a, sig)


# ------------------------------------------------------------------ PPO on real transitions
def train_ppo(Z, r, sig, steps=1500, seed=0, ep=64, lam_risk=2.0, imag_q=None, disagree=None, beta=1.0):
    """r: realized returns (raw PPO) or None with imag_q [n, NQ] (WM predictive quantiles of the traded day):
    rewards are then SAMPLED from the WM (imagination), minus beta * |w| * disagree (MOPO-style ensemble penalty)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dz = Z.shape[1]
    zs = Z.std(0) + 1e-6
    Zn = torch.tensor(Z / zs, dtype=torch.float32)
    pi = net(dz + 1, 1); vf = net(dz + 1, 1)
    logstd = nn.Parameter(torch.tensor(-0.5))
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()) + [logstd], lr=3e-4)
    rr = np.where(np.isfinite(r), r, 0.0) if r is not None else None
    n = len(Z)
    iters = max(1, steps // 10)
    for it in range(iters):
        obs, acts, logps, rews = [], [], [], []
        for _ in range(8):
            s0 = rng.integers(0, n - ep)
            prev = 0.0
            for i in range(s0, s0 + ep):
                o = torch.cat([Zn[i], torch.tensor([prev])])
                with torch.no_grad():
                    mu = pi(o)[0]
                    u = mu + torch.exp(logstd) * torch.randn(())
                    lp = -0.5 * ((u - mu) / torch.exp(logstd)) ** 2 - logstd
                a = float(np.tanh(u.item()))
                w = np.clip(a * TV / sig[i], -CAP, CAP)
                wp = np.clip(prev * TV / sig[i], -CAP, CAP)
                if rr is not None:
                    ri = rr[i]
                else:
                    lv = rng.uniform(0, C.NQ - 1); lo = int(lv)
                    ri = imag_q[i, lo] + (lv - lo) * (imag_q[i, min(lo + 1, C.NQ - 1)] - imag_q[i, lo])
                rew = (w * ri - COST * abs(w - wp) - lam_risk * w ** 2 * sig[i] ** 2) * 100
                if disagree is not None:
                    rew -= beta * abs(w) * disagree[i] * 100
                obs.append(o); acts.append(u); logps.append(lp); rews.append(rew)
                prev = a
        O = torch.stack(obs); A = torch.stack(acts); LP = torch.stack(logps)
        Rw = torch.tensor(rews, dtype=torch.float32).view(8, ep)
        with torch.no_grad():
            V = vf(O)[:, 0].view(8, ep)
        adv = torch.zeros_like(Rw); ret = torch.zeros_like(Rw); g = torch.zeros(8); last = torch.zeros(8)
        for k in reversed(range(ep)):
            nv = V[:, k + 1] if k + 1 < ep else torch.zeros(8)
            delta = Rw[:, k] + 0.9 * nv - V[:, k]
            g = delta + 0.9 * 0.95 * g
            adv[:, k] = g
            ret[:, k] = adv[:, k] + V[:, k]
        adv = ((adv - adv.mean()) / (adv.std() + 1e-6)).flatten(); ret = ret.flatten()
        for _ in range(10):
            mu = pi(O)[:, 0]
            lp = -0.5 * ((A - mu) / torch.exp(logstd)) ** 2 - logstd
            ratio = torch.exp(lp - LP)
            la = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            lv = ((vf(O)[:, 0] - ret) ** 2).mean()
            opt.zero_grad(); (la + 0.5 * lv).backward(); opt.step()
    return lambda Z_, a_prev: np.tanh(pi(torch.cat([torch.tensor(Z_ / zs, dtype=torch.float32),
                                                    torch.tensor(a_prev, dtype=torch.float32)], 1)).detach().numpy()[:, 0])


# ------------------------------------------------------------------ DQN on real transitions
def train_dqn(Z, r, sig, steps=1500, seed=0, gamma=0.9, lam_risk=2.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dz = Z.shape[1]
    zs = Z.std(0) + 1e-6
    Zn = torch.tensor(Z / zs, dtype=torch.float32)
    q = net(dz + 1, len(ACTS)); qt = net(dz + 1, len(ACTS)); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    rr = np.where(np.isfinite(r), r, 0.0)
    n = len(Z)
    At = torch.tensor(ACTS, dtype=torch.float32)
    St = torch.tensor(sig, dtype=torch.float32); Rt = torch.tensor(rr, dtype=torch.float32)
    for it in range(steps):
        i = torch.tensor(rng.integers(0, n - 1, 256))
        ap = torch.tensor(rng.integers(0, len(ACTS), 256))
        o = torch.cat([Zn[i], At[ap][:, None]], 1)
        ai = torch.tensor(rng.integers(0, len(ACTS), 256))          # off-policy uniform behaviour (offline)
        w = (At[ai] * TV / St[i]).clamp(-CAP, CAP); wp = (At[ap] * TV / St[i]).clamp(-CAP, CAP)
        rew = (w * Rt[i] - COST * (w - wp).abs() - lam_risk * w ** 2 * St[i] ** 2) * 100
        o2 = torch.cat([Zn[i + 1], At[ai][:, None]], 1)
        with torch.no_grad():
            tgt = rew + gamma * qt(o2).max(1).values
        pred = q(o).gather(1, ai[:, None])[:, 0]
        loss = nn.functional.smooth_l1_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            qt.load_state_dict(q.state_dict())
    return lambda Z_, a_prev: ACTS[q(torch.cat([torch.tensor(Z_ / zs, dtype=torch.float32),
                                                torch.tensor(a_prev, dtype=torch.float32)], 1)).argmax(1).numpy()]


def voltgt(mu, sig):
    return np.sign(mu) * np.minimum(TV / sig, CAP)


def kelly(mu, sig, k=0.25, band=0.25):
    """champion (research 5.4): w = clip(k * mu / var, -1, 1) * TV / sigma, |w| <= CAP, no-trade band
    band * TV / sigma (trade only if the target moves more than that from the current position)."""
    tgt = np.clip(np.clip(k * mu / sig ** 2, -1, 1) * TV / sig, -CAP, CAP)
    w = np.zeros_like(tgt)
    cur = 0.0
    for i in range(len(tgt)):
        if abs(tgt[i] - cur) > band * min(TV / sig[i], CAP):
            cur = tgt[i]
        w[i] = cur
    return w
