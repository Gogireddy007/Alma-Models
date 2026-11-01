"""
Alma-Live Gas frozen-model adapter for the data lead's harness / paper broker (tools/live/README.md interface).

predict(ctx) builds today's inputs with wm_store.features_from_ctx(ctx) -- the identical function the training
panel used (LRT-13) -- and takes the previous L-1 input rows from the panel cache (loaded at construction, outside the
data guard) or, for days after the cache, from its own buffer of earlier predict() calls. When today's row exists
in the cache it is asserted equal to the ctx-built row (parity). Quantiles at models.QS levels; act() is the
pre-declared Kelly vol-target champion (wm_agent.kelly rule, stateful no-trade band).
"""
import os
import sys
import json
import glob
import hashlib

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import wm_store as WS  # noqa: E402
import wm_core as C  # noqa: E402
import wm_harness as HN  # noqa: E402
import wm_agent as A  # noqa: E402

FROZEN = os.path.join(ROOT, "runs", "live", "FROZEN.json")
HQS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class AlmaLive:
    def __init__(self, name, ckpts):
        self.name = name
        self.members = []
        for p in ckpts:
            ck = torch.load(p, weights_only=False)
            m = C.WM(ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
            nm = HN.Norm.__new__(HN.Norm)
            for k, v in ck["norm"].items():
                setattr(nm, k, np.array(v, dtype=np.float64))
            self.members.append((m, nm, ck["cfg"].get("L", 20)))
        self.meta = {"ckpt": [os.path.relpath(p, ROOT) for p in ckpts], "ckpt_sha256": [_sha(p) for p in ckpts]}
        P = WS.load_panel()
        self.cache = {pd.Timestamp(d): x for d, x in zip(P["D"], P["X"])}
        self.buf = {}
        self.cur = 0.0
        self.parity_fail = 0

    def _window(self, d, x, L):
        td = WS.trading_days()
        k = td.get_loc(d)
        rows = []
        for i in range(k - L + 1, k):
            dd = td[i]
            r = self.cache.get(dd, self.buf.get(dd))
            if r is None:
                return None
            rows.append(r)
        rows.append(x)
        return np.stack(rows)

    def _quant(self, ctx):
        d = pd.Timestamp(ctx["day"])
        fu = WS.features_from_ctx(ctx)
        if fu is None:
            return None
        x, u = fu
        if d in self.cache and not np.array_equal(self.cache[d], x):
            self.parity_fail += 1
        self.buf[d] = x
        qs = []
        for m, nm, L in self.members:
            w = self._window(d, x, L)
            if w is None:
                return None
            Xw = torch.tensor(nm.x(w)[None]); U = torch.tensor(u[None])
            ST = torch.tensor(WS.steps_for(d, ctx)[None])
            with torch.no_grad():
                q = m(Xw, U, ST, sample=False).numpy()[0]
            qs.append(q * nm.ys[:, None] + nm.ym[:, None])
        return np.sort(np.mean(qs, 0), -1)          # [T, NQ] real units

    def predict(self, ctx):
        q = self._quant(ctx)
        if q is None:
            return {}
        pick = lambda j: {lv: float(np.interp(lv, HN.QL, q[j])) for lv in HQS}  # noqa: E731
        r1 = q[WS.TARGETS.index("r1")]
        return {"ung_q": pick(WS.TARGETS.index("r1")), "rv5_q": pick(WS.TARGETS.index("lrv5")),
                "stor_dev_q": pick(WS.TARGETS.index("dsurp")), "mu": float(r1[len(r1) // 2]),
                "p_up": float(1 - np.interp(0.0, r1, HN.QL))}

    def act(self, ctx):
        q = self._quant(ctx)
        if q is None:
            self.cur = 0.0
            return {"UNG": 0.0}
        r1 = q[WS.TARGETS.index("r1")]
        mu = np.array([r1[len(r1) // 2]]); sig = np.array([max((r1[-1] - r1[0]) / 3.29, 1e-3)])
        tgt = float(np.clip(np.clip(0.25 * mu / sig ** 2, -1, 1) * A.TV / sig, -A.CAP, A.CAP)[0])
        if abs(tgt - self.cur) > 0.25 * min(A.TV / sig[0], A.CAP):
            self.cur = tgt
        return {"UNG": self.cur}

    def observe(self, outcome):
        return 0


def register(REGISTRY):
    if not os.path.exists(FROZEN):
        return
    fz = json.load(open(FROZEN))
    for name, spec in fz.get("models", {}).items():
        paths = [os.path.join(ROOT, p) for p in spec["ckpt"]]
        REGISTRY[name] = (lambda n=name, p=paths: AlmaLive(n, p))
