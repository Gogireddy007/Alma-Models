"""Family 7b: self-improving online learner for BNet, simulated on the DEV-early inner folds (from the cache).

Rule (frozen as a meta-rule before TEST; wrapper wm_b_fin.BReplay):
  - fold model = base weights theta0 (trained exactly as in loop_b);
  - shadow learners j = 1..J with learning rates lr_j (lr_1 = 0 = frozen), each a copy of theta0;
  - after day D is revealed (label of D, available D+1 00:00), every learner takes `steps` Adam-free SGD steps on a
    replay buffer = (features, labels) of the last R predicted days (all coins in the context), loss = BCE(p_up) +
    pinball, plus l2 * ||theta - theta0||^2;
  - meta-selection: the decision for D+1 uses the learner with the lowest trailing `win`-day BCE on days it predicted
    (ties -> lowest index, i.e. the frozen model first). With J = 1 it is a plain replay fine-tuner.
The buffer starts empty at the start of a run (no backfill). S_C = P1(online) - P1(frozen). Logged kind 'cl_b'.
"""
import os
import sys
import copy
import json
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats_b as FB     # noqa: E402
import loop_b as LB      # noqa: E402
import models_b as MB    # noqa: E402


class Replay:
    def __init__(self, base, rule):
        self.rule = rule
        self.th0 = [p.detach().clone() for p in base.parameters()]
        self.L = [copy.deepcopy(base) for _ in rule["lrs"]]
        self.opt = [torch.optim.SGD(m.parameters(), lr=lr) if lr > 0 else None for m, lr in zip(self.L, rule["lrs"])]
        self.buf = []
        self.hist = [[] for _ in rule["lrs"]]
        self.last = None

    def choose(self):
        w = self.rule.get("win", 30)
        sc = [np.mean(h[-w:]) if len(h) >= 5 else np.inf for h in self.hist]
        if not np.isfinite(sc).any():
            return 0
        return int(np.argmin(np.where(np.isfinite(sc), sc, np.inf)))

    def predict(self, inp):
        X, hz, har, aux, ok = inp
        outs = []
        with torch.no_grad():
            for m in self.L:
                g = torch.Generator().manual_seed(1234)
                o = m(X, hz, har, aux, ok, gen=g)
                outs.append((torch.sigmoid(o["logit"]).numpy()[0], o["q"].numpy()[0]))
        self.last = (inp, outs)
        return outs[self.choose()]

    def observe(self, y, tl):
        inp, outs = self.last
        ok = inp[4][0].numpy()
        k = ok & np.isfinite(y)
        for j, (p, _) in enumerate(outs):
            if k.any():
                pp = np.clip(p[k], 1e-6, 1 - 1e-6)
                self.hist[j].append(float(-np.mean(y[k] * np.log(pp) + (1 - y[k]) * np.log(1 - pp))))
        self.buf.append((inp, y.astype(np.float32), tl.astype(np.float32)))
        self.buf = self.buf[-self.rule["R"]:]
        X = torch.cat([b[0][0] for b in self.buf]); hz = torch.cat([b[0][1] for b in self.buf])
        har = torch.cat([b[0][2] for b in self.buf]); aux = torch.cat([b[0][3] for b in self.buf])
        okb = torch.cat([b[0][4] for b in self.buf])
        Y = torch.from_numpy(np.stack([b[1] for b in self.buf])); TL = torch.from_numpy(np.stack([b[2] for b in self.buf]))
        n = 0
        for m, opt in zip(self.L, self.opt):
            if opt is None:
                continue
            m.train()
            for _ in range(self.rule["steps"]):
                g = torch.Generator().manual_seed(77 + n)
                o = m(X, hz, har, aux, okb, gen=g)
                my = okb & torch.isfinite(Y)
                mq = okb & torch.isfinite(TL)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(o["logit"][my], Y[my]) if my.any() else 0.0
                if mq.any():
                    loss = loss + self.rule.get("wq", 1.0) * MB.pinball(o["q"], torch.nan_to_num(TL))[mq].mean()
                loss = loss + self.rule["l2"] * sum(((p - a) ** 2).sum() for p, a in zip(m.parameters(), self.th0))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); n += 1
            m.eval()
        return n


def run(name, cfg, rule, seed=0):
    D = LB.cache()
    dif, on = [], []
    try:
        for fk, (fe, a, b) in LB.FOLDS.items():
            idx = LB.fold_idx(D, a, b)
            base = LB.train_model(D, cfg, fe, seed, log=lambda *x: None)
            pf, qf = LB.predict([base], D, idx)
            R = Replay(base, rule)
            po = np.full_like(pf, np.nan); qo = np.full_like(qf, np.nan)
            for t, i in enumerate(idx):
                ii = np.array([i])
                X, hz, har, aux = LB.tens(D, ii)
                ok = torch.from_numpy(D["ok"][ii])
                p, q = R.predict((X, hz, har, aux, ok))
                po[t] = np.where(D["ok"][i], p, np.nan)
                qo[t] = np.where(D["ok"][i][:, None], np.sort(q, -1) + D["A"][i][:, None], np.nan)
                R.observe(D["y"][i], D["tl"][i])        # label of day i, revealed after the decision for day i
            s0, s1 = LB.score(D, idx, pf, qf), LB.score(D, idx, po, qo)
            dif.append(s1["all"] - s0["all"]); on.append(s1["all"])
            LB.log_row({"kind": "cl_b", "name": name, "config": {"base": cfg, "rule": rule}, "seed": seed, "fold": fk,
                        "status": "ok", "P1_all": s1["all"], "P1_train": s1["train"], "P1_heldout": s1["heldout"],
                        "P3_all": s1["qs"], "note": f"S_C {dif[-1]:+.5f} frozen {s0['all']:.5f} QS0 {s0['qs']:.5f}"})
        LB.log_row({"kind": "cl_b", "name": name, "config": {"base": cfg, "rule": rule}, "seed": seed, "fold": "EM",
                    "status": "ok", "P1_all": float(np.mean(on)),
                    "note": f"S_C {np.mean(dif):+.5f} " + " ".join(f"{d:+.4f}" for d in dif)})
        print(f"{name} s{seed}: S_C {np.mean(dif):+.5f} ({' '.join(f'{d:+.4f}' for d in dif)}) online {np.mean(on):.4f}",
              flush=True)
    except Exception as ex:
        traceback.print_exc()
        LB.log_row({"kind": "cl_b", "name": name, "config": {"base": cfg, "rule": rule}, "seed": seed, "fold": "EM",
                    "status": "crash", "note": f"{type(ex).__name__}: {str(ex)[:200]}"})


if __name__ == "__main__":
    torch.set_num_threads(3)
    for ln in open(sys.argv[1]):
        if ln.strip():
            j = json.loads(ln)
            run(j["name"], j["cfg"], j["rule"], j.get("seed", 0))
