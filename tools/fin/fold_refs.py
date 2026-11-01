"""
Registered P1 baselines on the architects' DEV inner folds (prereg A5), so candidates are compared against them on
identical windows. DEV only (every window ends <= 2025-06-30); the fitting cut is moved earlier by patching
data.TRAIN_END (models.train_view honours it), exactly as tools/fin/search/common.fit_end_patch does.

  E1 fit <= 2022-12-31, test 2023-01-01..2023-06-30     (architect B folds)
  E2 fit <= 2023-06-30, test 2023-07-01..2023-12-31
  E3 fit <= 2023-12-31, test 2024-01-01..2024-06-30
  F1 fit <= 2024-06-30, test 2024-07-01..2025-06-30     (architect A fold)

P1 scoring = run_eval rule (A4): p_up > 0.5 if given else median > ref; every labelled coin-day scored, missing ->
DEV-majority 'down'. P3 = mean pinball 0.1..0.9 on log RV over the model's finite forecasts.

    nice -n 10 .venv/bin/python tools/fin/fold_refs.py [--models har,har_dow,...]   -> runs/fin/fold_refs.json
"""
import os
import sys
import json
import argparse

import torch  # noqa: F401  (A3 import order)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H     # noqa: E402
import data as DT       # noqa: E402
import metrics as MT    # noqa: E402

FOLDS = {"E1": ("2022-12-31", "2023-01-01", "2023-06-30"), "E2": ("2023-06-30", "2023-07-01", "2023-12-31"),
         "E3": ("2023-12-31", "2024-01-01", "2024-06-30"), "F1": ("2024-06-30", "2024-07-01", "2025-06-30")}
OUT = os.path.join(H.ROOT, "runs", "fin", "fold_refs.json")
I9 = [H.QS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def score(P, idx, res):
    L = [P.labels(int(i)) for i in idx]
    Y, REF, LR = (np.stack([l[k] for l in L]) for k in ("y_vol", "ref", "logrv"))
    if np.isfinite(res["p_up"]).any():
        p = res["p_up"]
        d = np.where(np.isfinite(p), (p > 0.5).astype(float), np.nan)
    else:
        med = res["q"][:, :, H.QS.index(0.5)]
        d = np.where(np.isfinite(med) & np.isfinite(REF), (med > REF).astype(float), np.nan)
    out = {}
    for g in ("all", "train", "heldout"):
        js = [k for k, s in enumerate(P.symbols) if g == "all" or P.group[s] == g]
        y, dd = Y[:, js], d[:, js]
        ok = np.isfinite(y)
        dd = np.where(ok & ~np.isfinite(dd), 0.0, dd)
        q = res["q"][:, js][:, :, I9]
        okq = np.isfinite(LR[:, js]) & np.isfinite(q).all(-1)
        qs = float(MT.pinball(LR[:, js][okq], q[okq], [H.QS[i] for i in I9]).mean()) if okq.any() else None
        out[g] = {"P1": float(((dd == y) & ok).sum() / ok.sum()), "n": int(ok.sum()), "P3_qs": qs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="majority,rv_persist,har,har_roll,lgbm,lgbm_asset,har_dow,lgbm_dow,har_intra_dow")
    a = ap.parse_args()
    import models as M
    P = H.Panel("crypto")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    old = DT.TRAIN_END
    for f, (fit_end, t0, t1) in FOLDS.items():
        idx = np.nonzero((P.days >= np.datetime64(t0, "ns")) & (P.days <= np.datetime64(t1, "ns")))[0]
        assert P.days[idx].max() <= np.datetime64(DT.TRAIN_END, "ns")
        for mn in a.models.split(","):
            DT.TRAIN_END = pd.Timestamp(fit_end)
            try:
                m = M.get(mn)
            finally:
                DT.TRAIN_END = old
            r = H.run(P, m, idx, online=False)
            res.setdefault(mn, {})[f] = score(P, idx, r)
            s = res[mn][f]
            print(f"{f} {mn}: P1 all {s['all']['P1']:.4f} train {s['train']['P1']:.4f} heldout {s['heldout']['P1']:.4f}"
                  f" P3 {s['all']['P3_qs']}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    for mn, v in res.items():
        e = [v[k]["all"]["P1"] for k in ("E1", "E2", "E3") if k in v]
        if len(e) == 3:
            v["EM"] = {"all": {"P1": float(np.mean(e))}}
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
