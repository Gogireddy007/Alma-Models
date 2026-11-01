"""Seed-ensemble DEV evaluation and the world-model-vs-twin comparison on the rolling DEV backtest.

    .venv/bin/python tools/gas/search/ens_eval.py --a r2_wm_x,f_wm_x_s1,f_wm_x_s2 --b r2_dir_x,f_dir_x_s1,f_dir_x_s2 \
        --name A --out runs/gas/search/ens_A.json

Ensemble = per-level average of the members' quantiles, re-sorted. Paired tests per origin kind on L48:
|error| and QS differences (a - b): mean, DM-HLN (h = 1) two-sided p, 95% bootstrap CI (weeks resampled, 2000
draws, seed 0); G1 exact McNemar. Also reported excluding the pre-2025 holiday weeks (prereg A2).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R        # noqa: E402
import storage as S    # noqa: E402

QL = np.array([R.QS[i] for i in R.I9])


def load_ens(tags):
    qs, ks = {}, {}
    for kind in S.ORIGINS:
        arr = []
        for t in tags:
            z = np.load(os.path.join(R.OUT, "preds", f"{t}.npz"))
            arr.append(z[f"q_{kind}"])
            if kind in ks:
                assert np.array_equal(ks[kind], z[f"k_{kind}"])
            ks[kind] = z[f"k_{kind}"]
        qs[kind] = np.sort(np.mean(arr, 0), -1)
    return qs, ks


def dm_hln(d):
    d = np.asarray(d, float)
    n = len(d)
    se = np.sqrt(np.var(d, ddof=0) / n)
    t = d.mean() / se * np.sqrt((n + 1 - 2 + 0) / n)
    return float(2 * stats.t.sf(abs(t), n - 1))


def boot(d, B=2000, seed=0):
    r = np.random.default_rng(seed)
    m = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def mcnemar(ca, cb):
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"a_only": b, "b_only": c, "p": p}


def losses(q, labs):
    ds = np.array([l["ds"][0] for l in labs])
    dv = np.array([l["dev"][0] for l in labs])
    med = q[:, 0, R.QS.index(0.5)]
    ae = np.abs(ds - med)
    d = ds[:, None] - q[:, 0, R.I9]
    qs = np.mean(np.maximum(QL * d, (QL - 1) * d), 1)
    corr = ((med - (ds - dv)) > 0) == (dv > 0)
    return ae, qs, corr, dv != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="")
    ap.add_argument("--name", default="ens")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    z = R.load_cache()
    w = S.load()[0]
    qa, ks = load_ens(a.a.split(","))
    res = {"a": a.a.split(","), "b": a.b.split(",") if a.b else None}
    pred = {k: (qa[k], [z["lab"][int(x)] for x in ks[k]]) for k in S.ORIGINS}
    ref = R.ref_metrics()
    res["a_metrics"] = R.score_row(pred, ref)
    hol = {k: np.array([w.avail_rule.iloc[int(x)] == "holiday_week_conservative" for x in ks[k]]) for k in S.ORIGINS}
    sub = {k: (qa[k][~hol[k]], [l for l, h in zip(pred[k][1], hol[k]) if not h]) for k in S.ORIGINS}
    res["a_metrics_no_holiday"] = R.score_row(sub)
    res["n_weeks"] = {k: int(len(ks[k])) for k in S.ORIGINS}
    res["n_weeks_no_holiday"] = {k: int((~hol[k]).sum()) for k in S.ORIGINS}
    if a.b:
        qb, kb = load_ens(a.b.split(","))
        predb = {k: (qb[k], pred[k][1]) for k in S.ORIGINS}
        res["b_metrics"] = R.score_row(predb, ref)
        res["b_metrics_no_holiday"] = R.score_row({k: (qb[k][~hol[k]], sub[k][1]) for k in S.ORIGINS})
        res["paired"] = {}
        for k in S.ORIGINS:
            assert np.array_equal(ks[k], kb[k])
            la, lb = losses(qa[k], pred[k][1]), losses(qb[k], pred[k][1])
            out = {}
            for nm, j in (("abs_err", 0), ("qs", 1)):
                d = la[j] - lb[j]
                out[nm] = {"mean_diff_a_minus_b": float(d.mean()), "dm_hln_p": dm_hln(d), "boot95": boot(d)}
                dn = d[~hol[k]]
                out[nm + "_no_holiday"] = {"mean_diff_a_minus_b": float(dn.mean()), "dm_hln_p": dm_hln(dn), "boot95": boot(dn)}
            m = la[3]
            out["g1_mcnemar"] = mcnemar(la[2][m], lb[2][m])
            res["paired"][k] = out
    s = json.dumps(res, indent=1, default=float)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
