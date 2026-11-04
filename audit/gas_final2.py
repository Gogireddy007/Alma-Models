"""Gas candidate + amendment-1 interval recalibration (GAS_FREEZE.md), scored once on TEST. Trailing errors are
computed in release order over VAL then TEST weeks; week t's scale uses only weeks released before its origin."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "gas")); sys.path.insert(0, HERE)
import storage as S, models as M, wm_gas as WG
from gas_calib import scales, apply, qs9
CK = ["wm_gas", "wm_gas_direct", "x_wm_s3456", "x_dir_s3456"]; REF = {"thu": 15.39, "fri": 15.27}
w, o, f = S.load()
kv, kt = S.target_weeks("val", w), S.target_weeks("test", w)
ks = np.r_[kv, kt]; ds = w.ds_l48.to_numpy(float)[ks]; a5 = w.avg5_l48.to_numpy(float)[ks]; tm = np.r_[np.zeros(len(kv), bool), np.ones(len(kt), bool)]
assert np.all(np.diff(ks) == 1), "weeks must be consecutive for the trailing window"
ens = [WG.GasEnsemble(f"runs/gas/search/ckpt/{n}.pt", WG.sha256(os.path.join(REPO, f"runs/gas/search/ckpt/{n}.pt"))) for n in CK]
ols = M.get("st_ols_wx"); rows = []; store = {}
for kind in S.ORIGINS:
    ctx = [S.build_context(int(k), kind, w, o, f) for k in ks]
    nn = sum(e.quantiles(ctx)[:, 0] * len(e.members) for e in ens) / sum(len(e.members) for e in ens)
    q = np.sort(0.8 * nn + 0.2 * np.asarray(ols.storage(ctx, WG.QS_GRID, "l48"), float), -1)
    s = scales(np.abs(q[:, 6] - ds), 8, 0.5, REF[kind]); qc = apply(q, s)
    for name, Q in (("candidate (frozen intervals)", q), ("candidate + adaptive intervals", qc)):
        m = tm; dv = (ds - a5)[m]
        rows.append(dict(kind=kind, model=name, mae=np.mean(abs(Q[m, 6] - ds[m])), qs=qs9(ds[m], Q[m]).mean(),
                         cov80=np.mean((ds[m] >= Q[m, 2]) & (ds[m] <= Q[m, 10])), cov95=np.mean((ds[m] >= Q[m, 0]) & (ds[m] <= Q[m, 12])),
                         width80=np.mean(Q[m, 10] - Q[m, 2]), acc=np.mean(((Q[m, 6] > a5[m]) == (dv > 0))[dv != 0])))
    store[kind] = (qs9(ds[tm], q[tm]), qs9(ds[tm], qc[tm]), s[tm])
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
rng = np.random.default_rng(0)
for kind, (a, b, s) in store.items():
    d = b - a; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print(f"{kind}: dQS {d.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  scale range {s.min():.2f}-{s.max():.2f}")
