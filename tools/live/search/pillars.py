"""
Pre-declared pillar tests (docs/live_search.md 0.9): paired daily differences, stationary block bootstrap (mean
block 10, 2000 draws, seed 0), two-sided p.

    .venv/bin/python tools/live/search/pillars.py fc <predA> <predB>      # forecasting (A - B, rel. pinball; < 0 favours A)
    .venv/bin/python tools/live/search/pillars.py pol <dailyA> <dailyB>   # policies (A - B net P&L / Sharpe; > 0 favours A)
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as RN  # noqa: E402
C, WD, H = RN.C, RN.WD, RN.H


def sbb_idx(n, rng, block=10):
    idx = np.empty(n, int)
    i = 0
    p = 1.0 / block
    while i < n:
        s = rng.integers(n)
        L = rng.geometric(p)
        for k in range(L):
            if i >= n:
                break
            idx[i] = (s + k) % n
            i += 1
    return idx


def boot(d, stat=np.mean, draws=2000, seed=0, block=10):
    rng = np.random.default_rng(seed)
    obs = stat(d)
    bs = np.array([stat(d[sbb_idx(len(d), rng, block)]) for _ in range(draws)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    c = bs - bs.mean()
    p = float(np.mean(np.abs(c) >= abs(obs)))
    return {"diff": round(float(obs), 5), "ci": [round(float(lo), 5), round(float(hi), 5)], "p": round(p, 4)}


def daily_rel(Q, refs, prim=None):
    prim = prim or [WD.TARGETS.index(t) for t in WD.PRIMARY]
    out = []
    for Y in sorted(Q):
        y = refs[Y]["y"]
        pa = H.pinball_np(Q[Y], y)
        pr = np.nanmean(H.pinball_np(refs[Y]["linqr"], y), 0)
        rel = pa[:, prim] / pr[prim]
        out.append(np.nanmean(rel, 1))
    return np.concatenate(out)


def load_pred(rid):
    z = np.load(os.path.join(RN.PRED, f"{rid}.npz"))
    return {int(k): z[k] for k in z.files}


def fc(a, b, refs=None):
    refs = refs or RN.load_refs()
    QA = a if isinstance(a, dict) else load_pred(a)
    QB = b if isinstance(b, dict) else load_pred(b)
    return boot(daily_rel(QA, refs) - daily_rel(QB, refs))


def pol(a, b):
    za = np.load(os.path.join(RN.OUT, "daily", f"{a}.npz")); zb = np.load(os.path.join(RN.OUT, "daily", f"{b}.npz"))
    import wm_agent as A
    pa, pb = A.pnl(za["w"], za["r"]), A.pnl(zb["w"], zb["r"])
    res = boot(pa - pb)
    sh = lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252)  # noqa: E731
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = sbb_idx(len(pa), rng)
        d.append(sh(pa[i]) - sh(pb[i]))
    res["sharpe_diff"] = round(float(sh(pa) - sh(pb)), 3)
    res["sharpe_ci"] = [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]
    return res


if __name__ == "__main__":
    f = fc if sys.argv[1] == "fc" else pol
    print(json.dumps(f(sys.argv[2], sys.argv[3])))
