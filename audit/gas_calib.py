"""Gas interval recalibration experiments on the DEV backtest (no TEST data read).

Base forecast = the frozen-for-TEST candidate (14-member pool + 0.2 OLS, see GAS_FREEZE.md), out-of-fold on DEV.
Recalibration: q' = median + s_t * (q - median), with
  global   s_t = s                                     (one constant)
  adaptive s_t = clip((trailing mean |err| over the last N RELEASED weeks / ref) ** a, lo, hi)
ref = mean |err| of the weeks used to fit; errors of week j are known once week j is released, which is before the
origin of week j+1 (both origins of week k follow the release of week k-1), so s_t uses only past releases.
Parameters chosen leave-one-fold-out on DEV (QS of the 9 levels); the chosen setting is then fixed for TEST.
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
sys.path.insert(0, HERE)
import gas_exp as G  # noqa: E402

LV = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])
WM = "r2b_wm_reg2,f_reg2_s1,f_reg2_s2,x_reg2_s3,x_reg2_s4,x_reg2_s5,x_reg2_s6".split(",")
DR = "f_reg2dir_s0,f_reg2dir_s1,f_reg2dir_s2,x_reg2dir_s3,x_reg2dir_s4,x_reg2dir_s5,x_reg2dir_s6".split(",")


def qs9(ds, q):
    d = ds[:, None] - q[:, 2:11]
    lv = LV[2:11]
    return np.maximum(lv * d, (lv - 1) * d).mean(1)


def scales(err, n, a, ref, lo=0.4, hi=2.0):
    """err[t] = |median - actual| of week t (in time order); s[t] uses err[t-n..t-1] only."""
    s = np.ones(len(err))
    for t in range(len(err)):
        past = err[max(0, t - n):t]
        if len(past) >= max(4, n // 2):
            s[t] = np.clip((past.mean() / ref) ** a, lo, hi)
    return s


def apply(q, s):
    med = q[:, 6:7]
    return np.sort(med + s[:, None] * (q - med), 1)


def adaptive_s(q, ds, n, a, ref, base=1.0):
    err = np.abs(q[:, 6] - ds)
    return base * scales(err, n, a, ref)


GRID = [("global", s, None, None) for s in (0.8, 0.9, 1.0, 1.1)] + \
       [("adaptive", b, n, a) for b, n, a in itertools.product((0.9, 1.0, 1.1), (4, 6, 8, 13, 26), (0.25, 0.5, 0.75, 1.0))]


def run(q, ds, f, cfg, ref):
    kind, b, n, a = cfg
    s = np.full(len(ds), b) if kind == "global" else adaptive_s(q, ds, n, a, ref, b)
    return apply(q, s)


def main():
    p = G.mix(G.ens(WM + DR), G.load("ref_ols_wx"), 0.2)
    for kind in ("thu", "fri"):
        q, ks = p[kind]
        ds = G.W.ds_l48.to_numpy(float)[ks]
        f = G.fold_of(ks)
        out = np.empty_like(q)
        picks = []
        for y in sorted(set(f)):
            tr = f != y
            ref = np.mean(np.abs(q[tr, 6] - ds[tr]))
            best = min(GRID, key=lambda c: qs9(ds[tr], run(q, ds, f, c, ref)[tr]).mean())
            picks.append(best)
            out[f == y] = run(q, ds, f, best, ref)[f == y]
        ref_all = np.mean(np.abs(q[:, 6] - ds))
        full = min(GRID, key=lambda c: qs9(ds, run(q, ds, f, c, ref_all)).mean())
        cov = lambda Q: np.mean((ds >= Q[:, 2]) & (ds <= Q[:, 10]))
        print(f"[{kind}] base   QS {qs9(ds, q).mean():.3f} cov80 {cov(q):.3f}")
        print(f"[{kind}] LOFO   QS {qs9(ds, out).mean():.3f} cov80 {cov(out):.3f}  picks {picks}")
        print(f"[{kind}] full-DEV choice {full} (ref |err| {ref_all:.2f})")
        for fy in sorted(set(f)):
            m = f == fy
            print(f"   fold {fy}: cov80 base {cov(q[m]) if False else np.mean((ds[m] >= q[m, 2]) & (ds[m] <= q[m, 10])):.2f}"
                  f" -> {np.mean((ds[m] >= out[m, 2]) & (ds[m] <= out[m, 10])):.2f}")


if __name__ == "__main__":
    main()
