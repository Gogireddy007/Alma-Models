"""Scaled gas candidate, scored ONCE on TEST after every choice was fixed on DEV.

Candidate = quantile average of every refit seed member of the rollout (wm) and direct recipes, blended with the
repo's st_ols_wx baseline at weight W_OLS (fixed from the DEV backtest before this script was run).
Members are loaded from checkpoints written by tools/gas/search/refit.py (train weeks released <= 2025-06-30).

    python gas_final.py --ckpts wm_gas,wm_gas_direct,x_wm_s3456,x_dir_s3456 --w-ols 0.2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "gas"))
import storage as S  # noqa: E402
import models as M   # noqa: E402
import wm_gas as WG  # noqa: E402

QS = WG.QS_GRID
LV9 = np.array(QS[2:11])


def pinball(y, q):
    d = y[:, None] - q[:, 2:11]
    return np.mean(np.maximum(LV9 * d, (LV9 - 1) * d), 1)


def boot_diff(a, b, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, [2.5, 97.5]), np.mean(bs >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--w-ols", type=float, required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    w, o, f = S.load()
    ks = S.target_weeks(a.split, w)
    ds = w.ds_l48.to_numpy(float)[ks]
    a5 = w.avg5_l48.to_numpy(float)[ks]
    dv = ds - a5
    ols = M.get("st_ols_wx")
    rows, err = [], {}
    for kind in S.ORIGINS:
        ctxs = [S.build_context(int(k), kind, w, o, f) for k in ks]
        qs_m = {}
        for name in a.ckpts.split(","):
            rel = f"runs/gas/search/ckpt/{name}.pt"
            e = WG.GasEnsemble(rel, WG.sha256(os.path.join(REPO, rel)))
            qs_m[name] = (e.quantiles(ctxs)[:, 0], len(e.members))
        nn = sum(q * n for q, n in qs_m.values()) / sum(n for _, n in qs_m.values())   # member-weighted average
        q_ols = np.asarray(ols.storage(ctxs, QS, "l48"), float)
        cands = {"wm_gas (frozen)": qs_m["wm_gas"][0], "wm_gas_direct (frozen)": qs_m["wm_gas_direct"][0],
                 "st_ols_wx": q_ols, "NN pool": np.sort(nn, -1),
                 f"NN pool + {a.w_ols:.2f} OLS": np.sort((1 - a.w_ols) * nn + a.w_ols * q_ols, -1)}
        for n, q in cands.items():
            ok = dv != 0
            e_abs = np.abs(q[:, 6] - ds)
            pin = pinball(ds, q)
            err[(kind, n)] = (e_abs, pin, ((q[:, 6] > a5) == (dv > 0)).astype(float))
            rows.append(dict(kind=kind, model=n, mae=e_abs.mean(), qs=pin.mean(),
                             acc=np.mean(((q[:, 6] > a5) == (dv > 0))[ok]),
                             cov80=np.mean((ds >= q[:, 2]) & (ds <= q[:, 10]))))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    cand = f"NN pool + {a.w_ols:.2f} OLS"
    for kind in S.ORIGINS:
        for ref in ("wm_gas (frozen)", "st_ols_wx"):
            for j, nm in ((0, "MAE"), (1, "QS")):
                m, ci, p = boot_diff(err[(kind, cand)][j], err[(kind, ref)][j])
                print(f"{kind} {cand} - {ref}: d{nm} {m:+.2f} CI[{ci[0]:+.2f},{ci[1]:+.2f}] p1s={p:.3f}")


if __name__ == "__main__":
    main()
