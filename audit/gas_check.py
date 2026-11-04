"""Independent re-score of the frozen gas storage models on DEV-backtest-free splits (VAL, TEST).

Reuses: storage.build_context (read and checked by hand: every input is asserted <= origin), the frozen wrapper
(tools/gas/wm_gas.py) and the repo's st_ols_wx baseline. Own: labels from weekly.csv, metrics, an independent
ridge baseline, and a perturbation leak test (target-week dS, all later weeks and every CPC observation on/after the
origin day are scrambled -> wm_gas forecasts must not change).

    python gas_check.py [--split test]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "gas"))
import storage as S  # noqa: E402
import models as M   # noqa: E402

QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
LV9 = np.array([QS[i] for i in I9])


def pinball(y, q):
    d = y[:, None] - q
    return np.mean(np.maximum(LV9 * d, (LV9 - 1) * d), 1)


def score(name, Q, ds, a5):
    med = Q[:, 6]
    dv = ds - a5
    ok = np.isfinite(dv) & (dv != 0)
    acc = np.mean(((med > a5) == (dv > 0))[ok])
    return dict(model=name, n=int(ok.sum()), acc=acc, mae=np.mean(np.abs(med - ds)),
                qs=np.mean(pinball(ds, Q[:, I9])), cov80=np.mean((ds >= Q[:, 2]) & (ds <= Q[:, 10])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", default="wm_gas,wm_gas_direct,st_ols_wx,st_persistence,st_climatology")
    a = ap.parse_args()
    w, o, f = S.load()
    ks = S.target_weeks(a.split, w)
    print(f"{a.split}: {len(ks)} weeks {w.week.iloc[ks[0]].date()} .. {w.week.iloc[ks[-1]].date()}")
    rows = []
    preds = {}
    for kind in S.ORIGINS:
        ctxs = [S.build_context(int(k), kind, w, o, f) for k in ks]
        ds = w["ds_l48"].to_numpy(float)[ks]
        a5 = w["avg5_l48"].to_numpy(float)[ks]
        for mn in a.models.split(","):
            p = M.get(mn)
            Q = np.asarray(p.storage(ctxs, QS, "l48"), float)
            preds[(kind, mn)] = Q
            rows.append(dict(kind=kind, **score(mn, Q, ds, a5)))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gas_check_{a.split}.csv"), index=False)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gas_preds_{a.split}.npy"),
            {f"{k}|{m}": v for (k, m), v in preds.items()}, allow_pickle=True)

    # ---------------- leak test: scramble everything the model must not know and re-predict
    rng = np.random.default_rng(0)
    p = M.get("wm_gas")
    changed = 0
    test_ks = ks[::4]
    for k in test_ks:
        for kind in S.ORIGINS:
            base = p.quantiles([S.build_context(int(k), kind, w, o, f)])
            w2 = w.copy()
            for s in S.SERIES:                                   # target week and later: random storage numbers
                for col in (f"ds_{s}", f"dev_{s}", f"lvl_{s}"):
                    w2.loc[w2.index >= k, col] = rng.normal(0, 100, (w2.index >= k).sum())
            odt = S.origin_dt(w, int(k), kind)
            od = pd.Timestamp(odt.date())
            o2 = o.copy()
            m = o2.index >= od                                    # observed weather on/after the origin day
            o2.loc[m, ["hdd", "cdd"]] = rng.uniform(0, 60, (m.sum(), 2))
            f2 = f.copy()                                         # forecasts issued after the origin
            m2 = f2.issue > od
            f2.loc[m2, ["hdd", "cdd"]] = rng.uniform(0, 60, (m2.sum(), 2))
            S._C.pop("o7", None)
            pert = p.quantiles([S.build_context(int(k), kind, w2, o2, f2)])
            S._C.pop("o7", None)
            changed += not np.allclose(base, pert, atol=1e-6)
    print(f"leak test: {changed} of {2 * len(test_ks)} (week, origin) forecasts changed after scrambling the future")


if __name__ == "__main__":
    main()
