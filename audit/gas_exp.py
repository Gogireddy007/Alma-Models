"""Gas improvement experiments on the DEV rolling-origin backtest ONLY (6 yearly folds, release 2019-07..2025-06,
each fold trained strictly before it; predictions from tools/gas/search/run.py bt in runs/gas/search/preds/).

Combination weights are chosen by leave-one-fold-out: the weight used on fold y is the one that minimises
DEV MAE on the other 5 folds. No VAL or TEST number is read here.

    python gas_exp.py                       # experiments on existing preds
    python gas_exp.py --extra tagA,tagB     # include more bt runs (e.g. new seeds)
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

PRED = os.path.join(REPO, "runs", "gas", "search", "preds")
LV9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def load(tag):
    z = np.load(os.path.join(PRED, tag + ".npz"))
    return {k: (z[f"q_{k}"][:, 0], z[f"k_{k}"]) for k in S.ORIGINS}       # L48 only


def ens(tags):
    ps = [load(t) for t in tags]
    out = {}
    for k in S.ORIGINS:
        assert all(np.array_equal(p[k][1], ps[0][k][1]) for p in ps)
        out[k] = (np.sort(np.mean([p[k][0] for p in ps], 0), -1), ps[0][k][1])
    return out


def mix(a, b, wb):
    return {k: (np.sort((1 - wb) * a[k][0] + wb * b[k][0], -1), a[k][1]) for k in S.ORIGINS}


W, _, _ = S.load()
FOLD = {}


def fold_of(ks):
    av = W.avail.iloc[ks]
    return np.array([(d.year if d.month >= 7 else d.year - 1) for d in av])


def metrics(p, kind, mask=None):
    q, ks = p[kind]
    ds = W.ds_l48.to_numpy(float)[ks]
    a5 = W.avg5_l48.to_numpy(float)[ks]
    m = np.ones(len(ks), bool) if mask is None else mask
    med = q[m, 6]
    d = ds[m, None] - q[m][:, 2:11]
    dv = (ds - a5)[m]
    ok = dv != 0
    return dict(mae=np.mean(np.abs(med - ds[m])), qs=np.mean(np.maximum(LV9 * d, (LV9 - 1) * d)),
                acc=np.mean(((med > a5[m]) == (dv > 0))[ok]),
                cov80=np.mean((ds[m] >= q[m, 2]) & (ds[m] <= q[m, 10])))


def report(name, p):
    r = {"exp": name}
    for kind in S.ORIGINS:
        for k, v in metrics(p, kind).items():
            r[f"{k}_{kind}"] = v
    # per-fold MAE (thu) to judge stability
    f = fold_of(p["thu"][1])
    r["fold_mae_thu"] = " ".join(f"{metrics(p, 'thu', f == y)['mae']:.1f}" for y in sorted(set(f)))
    return r


def lofo_blend(a, b, grid=np.round(np.arange(0, 1.01, 0.1), 2)):
    """Weight on b chosen per fold from the other folds' MAE (thu+fri)."""
    out = {}
    chosen = {}
    for kind in S.ORIGINS:
        qa, ks = a[kind]
        qb = b[kind][0]
        f = fold_of(ks)
        ds = W.ds_l48.to_numpy(float)[ks]
        q = np.empty_like(qa)
        for y in sorted(set(f)):
            tr = f != y
            maes = [np.mean(np.abs(((1 - w) * qa + w * qb)[tr, 6] - ds[tr])) for w in grid]
            w = grid[int(np.argmin(maes))]
            chosen.setdefault(kind, []).append(float(w))
            q[f == y] = (1 - w) * qa[f == y] + w * qb[f == y]
        out[kind] = (np.sort(q, -1), ks)
    return out, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="r2b_wm_reg2,f_reg2_s1,f_reg2_s2")
    ap.add_argument("--dir", default="f_reg2dir_s0,f_reg2dir_s1,f_reg2dir_s2")
    ap.add_argument("--out", default="gas_exp.csv")
    a = ap.parse_args()
    wm, dr = a.wm.split(","), a.dir.split(",")
    rows = [report("ref_ols_wx", load("ref_ols_wx")), report("ref_ols_seas", load("ref_ols_seas"))]
    for t in wm + dr:
        rows.append(report(f"single {t}", load(t)))
    E_wm, E_dir, E_all = ens(wm), ens(dr), ens(wm + dr)
    rows.append(report(f"E0 frozen recipe: wm x{len(wm)}", E_wm))
    rows.append(report(f"E1 direct x{len(dr)}", E_dir))
    rows.append(report(f"E2 wm+direct x{len(wm) + len(dr)}", E_all))
    for base_name, base in (("E2", E_all), ("E0", E_wm)):
        for ref in ("ref_ols_seas", "ref_ols_wx"):
            p, ch = lofo_blend(base, load(ref))
            r = report(f"{base_name}+{ref} LOFO blend", p)
            r["weights"] = str(ch)
            rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["exp", "mae_thu", "mae_fri", "qs_thu", "qs_fri", "acc_thu", "acc_fri", "cov80_thu", "fold_mae_thu"]
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "weights" in t:
        print(t.loc[t.weights.notna(), ["exp", "weights"]].to_string(index=False))
    t.to_csv(os.path.join(HERE, a.out), index=False)


if __name__ == "__main__":
    main()
