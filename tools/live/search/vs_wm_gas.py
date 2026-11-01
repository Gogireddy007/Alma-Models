"""
FORWARD-TEST COMPARATOR ONLY (manager ruling on LRT-18: wm_gas was fit on data overlapping DEV/VAL; not a DEV/VAL baseline).
Storage target vs the prior frozen-recipe wm_gas (its out-of-sample rolling-fold DEV predictions, ensemble B =
runs/gas/search/ens_B.json members) on matched weeks. Live decision days on Thursday of week W (next release = W)
<-> gas origin `thu`; Friday of week W <-> `fri`. Both are scored on dS_W (first print) with QS over levels
0.1..0.9 (live dsurp quantiles + live avg5 = dS quantiles). Paired weekly difference, block bootstrap.

    .venv/bin/python tools/live/search/vs_wm_gas.py <live_pred_run_id>
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as RN  # noqa: E402
import pillars as PL  # noqa: E402
C, WD, H = RN.C, RN.WD, RN.H
ROOT = RN.ROOT
GQS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
L9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def qs9(q, levels, y):
    idx = [list(np.round(levels, 3)).index(round(a, 3)) for a in L9]
    e = y[:, None] - q[:, idx]
    a = np.array(L9)
    return np.maximum(a * e, (a - 1) * e).mean(1)


def main(rid):
    ens = json.load(open(os.path.join(ROOT, "runs", "gas", "search", "ens_B.json")))["a"]
    wk = pd.read_csv(os.path.join(ROOT, "data", "gas", "weekly.csv"), parse_dates=["week"])
    gas = {}
    for kind in ("thu", "fri"):
        qs = []
        for m in ens:
            z = np.load(os.path.join(ROOT, "runs", "gas", "search", "preds", f"{m}.npz"))
            qs.append(z[f"q_{kind}"][:, 0]); k = z[f"k_{kind}"]
        q = np.sort(np.mean(qs, 0), -1)
        gas[kind] = {str(wk.week.iloc[i].date()): q[j] for j, i in enumerate(k)}
    P = WD.load_panel()
    Q = PL.load_pred(rid)
    refs = RN.load_refs()
    jd = WD.TARGETS.index("dsurp")
    out = {}
    for kind, wd in (("thu", 3), ("fri", 4)):
        a_l, b_l, r_l = [], [], []
        for Y, (tr, te) in RN.folds(P, 20).items():
            norm = H.Norm(P["X"], P["Y"], tr)
            q = Q[Y][:, jd] * norm.ys[jd] + norm.ym[jd]
            y = P["Y"][te, jd]
            qr = refs[Y]["linqr"][:, jd] * norm.ys[jd] + norm.ym[jd]
            for n, r in enumerate(te):
                d = pd.Timestamp(P["D"][r])
                if d.weekday() != wd:
                    continue
                W = d + pd.Timedelta(days=4 - wd)            # Friday ending week W
                key = str(W.date())
                if key not in gas[kind] or not np.isfinite(y[n]):
                    continue
                row = wk[wk.week == W]
                a5 = float(row["avg5_l48"].iloc[0]); ds = float(row["ds_l48"].iloc[0])
                # live dS quantiles: dsurp quantiles + (dS - dsurp) offset (= live avg5)
                off = ds - y[n]
                a_l.append(qs9(q[n:n + 1] + off, H.QL, np.array([ds]))[0])
                r_l.append(qs9(qr[n:n + 1] + off, H.QL, np.array([ds]))[0])
                b_l.append(qs9(gas[kind][key][None], GQS, np.array([ds]))[0])
        a, b, r = map(np.array, (a_l, b_l, r_l))
        out[kind] = {"n_weeks": len(a), "qs_live": round(float(a.mean()), 3), "qs_wm_gas": round(float(b.mean()), 3),
                     "qs_live_ridge": round(float(r.mean()), 3), "live_minus_wm_gas": PL.boot(a - b, block=4)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
