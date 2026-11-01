"""R1/R2/R3 (auditor): training-data invariance of the lead's fitted factories. DEV contexts only.

R1  VAL invariance: corrupt every open row dated > 2025-06-30 (VAL period) BEFORE the factory fits -> the fitted
    model's outputs on fixed DEV contexts are bit-identical (no fitting on VAL).
R2  fold invariance: with data.TRAIN_END patched to 2024-06-30 (as the architects' refs do), corrupt rows dated
    > 2024-06-30 -> identical outputs (fold references are not fitted on the selection holdout).
R3  held-out invariance: corrupt every row of the 8 held-out coins -> pooled factories are bit-identical
    (lgbm_asset is per-asset by design: only its held-out columns may change).
Inference contexts always come from the clean panel, so only the FIT can differ.

  nice -n 10 .venv/bin/python tools/fin/redteam/rt_train_invariance.py [--models har,lgbm]
"""
import os
import sys
import argparse

import torch  # noqa: F401  (lead A3: torch first, LightGBM lazily with n_jobs=1)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_rt as R    # noqa: E402
H, DT = R.H, R.DT
sys.path.insert(0, os.path.join(R.FIN, "search"))

POOLED = ["majority", "rv_persist", "har", "harx", "lgbm", "voltgt_tsmom", "lgbm_policy"]
PERASSET = ["lgbm_asset"]
DAYS = ["2023-03-15", "2024-02-20", "2024-06-30", "2024-11-05", "2025-06-30"]


def outputs(M, name, P, idx):
    return R.out_of(H.run(P, M.get(name), idx, online=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(POOLED + PERASSET))
    a = ap.parse_args()
    import models as M
    import common as C
    P = C.dev_panel("crypto")
    idx = [P.index(d) for d in DAYS]
    _, ho = DT.universe()
    hoj = np.array([s in ho for s in P.symbols])
    lines, ok_all = [], True

    def rep(s, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(("OK   " if ok else "FAIL ") + s)
        print(lines[-1], flush=True)

    # positive control: the corruption must bite when the fit window DOES include the corrupted rows
    hb = outputs(M, "har", P, idx)
    with R.perturbed_load(cut=C.FOLD_FIT_END, mode="rand", seed=9):
        hc = outputs(M, "har", P, idx)
    rep("R0 control: har (fit <= 2025-06-30) CHANGES when rows > 2024-06-30 are randomised", not R.same(hb, hc))
    for mn in a.models.split(","):
        base = outputs(M, mn, P, idx)
        with R.perturbed_load(cut=DT.TRAIN_END, mode="rand", seed=1):
            o1 = outputs(M, mn, P, idx)
        rep(f"R1 {mn}: fit unchanged when every row > 2025-06-30 is randomised", R.same(base, o1))
        with C.fit_end_patch(C.FOLD_FIT_END):
            fb = outputs(M, mn, P, idx)
            with R.perturbed_load(cut=C.FOLD_FIT_END, mode="rand", seed=2):
                f1 = outputs(M, mn, P, idx)
        rep(f"R2 {mn}: fold fit (TRAIN_END=2024-06-30) unchanged when rows > 2024-06-30 are randomised",
            R.same(fb, f1))
        with R.perturbed_load(heldout=True, mode="rand", seed=3):
            o3 = outputs(M, mn, P, idx)
        if mn in PERASSET:
            # per-asset: training-coin columns must be identical
            if "pos" in o3:
                trc = base[:, ~hoj]
                rep(f"R3 {mn}: training-coin outputs unchanged under held-out corruption", R.same(trc, o3[:, ~hoj]))
            else:
                m = len(P.symbols)
                b = base[:, :m][:, ~hoj]
                rep(f"R3 {mn}: training-coin p_up unchanged under held-out corruption", R.same(b, o3[:, :m][:, ~hoj]))
        else:
            rep(f"R3 {mn}: pooled fit unchanged when the held-out coins' rows are randomised", R.same(base, o3))
    lines.append("ALL OK" if ok_all else "FAILURES")
    open(os.path.join(R.HERE, "rt_train_invariance.txt"), "w").write("\n".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    main()
