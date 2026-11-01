"""Prereg A5 baselines (auditor): har_dow, lgbm_dow, har_intra_dow. (1) inference: randomised future (NaN flips) ->
identical, D-1 / last hour randomised -> changed, crypto (+ETF where applicable); (2) F1 P1 with fit_end 2024-06-30
(A4 scoring rule) vs the auditor's own OLS reproduction (rt_gap.py). Training invariance: rt_train_invariance.py."""
import os, sys
import torch  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R
import rt_arch as RA
H = R.H
sys.path.insert(0, os.path.join(R.FIN, "search"))
import common as C
import models as M
rng = np.random.default_rng(3)
P, E = C.dev_panel("crypto"), C.dev_panel("etf")
y, syms, ho = RA.raw_labels()
for mn in ["har_dow", "lgbm_dow", "har_intra_dow"]:
    m = M.get(mn)
    for mk, PP in (("crypto", P), ("etf", E)):
        if mk == "etf" and mn == "har_intra_dow":
            continue
        for d in RA.DAYS:
            i = min(PP.index(d), len(PP.days) - 1)
            o0 = R.out_of(H.run(PP, m, [i], online=False))
            o1 = R.out_of(H.run(RA.rand_future(PP, i, rng), m, [i], online=False))
            o2 = R.out_of(H.run(RA.rand_future(PP, i, rng, from_prev=True), m, [i], online=False))
            ok = R.same(o0, o1) and not R.same(o0, o2)
            if mn == "har_intra_dow":
                Q = H.Panel.__new__(H.Panel); Q.__dict__.update(PP.__dict__)
                Q.hourly = {f: a.copy() for f, a in PP.hourly.items()}; Q.hourly["rv5"][24 * i - 1] *= 5
                ok &= not R.same(o0, R.out_of(H.run(Q, m, [i], online=False)))
            print(("OK   " if ok else "FAIL ") + f"{mn} {mk} {d}: future randomised identical; past perturbation changes", flush=True)
    with C.fit_end_patch(C.FOLD_FIT_END):
        mf = M.get(mn)
    idx = C.bt_days(P)
    r = H.run(P, mf, idx, online=False)
    days = pd.DatetimeIndex(P.days[idx]); Y = y.loc[days, r["symbols"]].to_numpy()
    if np.isfinite(r["p_up"]).any():
        p = r["p_up"]; dd = np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan)
    else:
        ref = np.stack([P.labels(int(i))["ref"] for i in idx]); med = r["q"][:, :, H.QS.index(0.5)]
        dd = np.where(np.isfinite(med), (med > ref).astype(float), np.nan)
    okk = np.isfinite(Y); dd = np.where(okk & ~np.isfinite(dd), 0.0, dd)
    hoj = [r["symbols"].index(s) for s in ho]
    print(f"F1 {mn}: P1 {((dd == Y) & okk).sum() / okk.sum():.4f} held-out {((dd == Y) & okk)[:, hoj].sum() / okk[:, hoj].sum():.4f}"
          f" coverage {np.isfinite(r['q'][:, :, 6])[okk].mean():.4f}", flush=True)
