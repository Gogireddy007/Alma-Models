"""
Alma-Live Gas continual learning (docs/live_search.md 0.10). Day by day through a fold's test period:
  1. predict day i with the current online weights;
  2. add every row whose labels have matured by day i's decision time (LAST_j < D_i: the last close it needs is
     published at 18:00 ET of LAST_j, before 15:30 ET of D_i) to the replay buffer;
  3. k SGD steps on a batch from replay (newly matured rows + the last `replay` training rows), loss = pinball +
     l2 * ||theta - theta0||^2;
  4. guard: on the matured rows, rolling `gwin`-day mean loss of the online predictions vs the frozen predictions
     (both made at their own decision times); if online > frozen * (1 + margin), roll back to theta0;
  5. meta rule (optional): lr *= exp(+eta) if the online weights beat the frozen ones on the rows that just matured,
     else exp(-eta), clipped to [lr/10, lr*10].
"""
import copy

import numpy as np
import torch

import wm_core as C

DEFAULT = {"k": 1, "lr": 1e-4, "l2": 1.0, "replay": 60, "bs": 32, "gwin": 20, "margin": 0.02, "meta": 0, "eta": 0.1}


def run_fold(m0, P, tr, te, norm, L, rule=None, seed=0):
    rule = dict(DEFAULT, **(rule or {}))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(m0)
    m.train(False)
    theta0 = [p.detach().clone() for p in m0.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=rule["lr"])
    lr = rule["lr"]
    D = P["D"].astype("datetime64[D]"); LAST = P["LAST"].astype("datetime64[D]")
    Yn = norm.y(P["Y"])
    q_on, q_fr = [], []
    frozen = C.predict(m0, P, te, norm, L)
    pending = list(te)               # test rows not yet matured
    buf = list(tr[-rule["replay"]:])
    hist_on, hist_fr = {}, {}        # row -> loss of its decision-time prediction (online / frozen)
    matured_scores = []
    rollbacks = 0
    prev_gain = None
    for n, i in enumerate(te):
        q = C.predict(m, P, np.array([i]), norm, L)[0]
        q_on.append(q)
        hist_on[i] = q; hist_fr[i] = frozen[n]
        newly = [j for j in pending if LAST[j] < D[i]]
        if newly:
            pending = [j for j in pending if j not in newly]
            buf.extend(newly)
            for j in newly:
                y = Yn[j][None]
                lo = np.nanmean(C.pinball_np(hist_on[j][None], y)); lf = np.nanmean(C.pinball_np(hist_fr[j][None], y))
                matured_scores.append((lo, lf))
            # meta rule: did the online weights beat the frozen ones on the rows that just matured?
            k_new = len(newly)
            prev_gain = np.mean([a for a, _ in matured_scores[-k_new:]]) < np.mean([b for _, b in matured_scores[-k_new:]])
            if rule["meta"]:
                lr = float(np.clip(lr * np.exp(rule["eta"] if prev_gain else -rule["eta"]), rule["lr"] / 10, rule["lr"] * 10))
                for g in opt.param_groups:
                    g["lr"] = lr
            # guard
            if len(matured_scores) >= rule["gwin"]:
                w = np.array(matured_scores[-rule["gwin"]:])
                if w[:, 0].mean() > w[:, 1].mean() * (1 + rule["margin"]):
                    with torch.no_grad():
                        for p, p0 in zip(m.parameters(), theta0):
                            p.copy_(p0)
                    rollbacks += 1
                    matured_scores = []
            # update
            bx = np.array(buf[-max(rule["replay"], len(newly)):])
            for _ in range(rule["k"]):
                b = np.concatenate([np.array(newly), rng.choice(bx, min(rule["bs"], len(bx)), replace=True)])
                Xw, U, ST, Y = C.windows(P, b, L, norm)
                m.train(True)
                qq = m(Xw, U, ST, sample=False)
                loss, _ = C.pinball(qq, Y)
                reg = sum(((p - p0) ** 2).sum() for p, p0 in zip(m.parameters(), theta0))
                tot = loss + rule["l2"] * reg
                opt.zero_grad(); tot.backward(); opt.step()
                m.train(False)
    return np.array(q_on), frozen, {"rollbacks": rollbacks, "final_lr": lr}
