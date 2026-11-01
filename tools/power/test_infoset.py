"""
Information-set tests for Alma-Power (docs/power_prereg.md section 2). Run before any TEST:

    nice -n 10 .venv/bin/python tools/power/test_infoset.py [--models naive_epf,lear,arx]

1. Perturbation: for sampled target days D across DEV/VAL/TEST, overwrite every price of days >= D and every
   covariate of days >= D-1 with random values; build_context must return bit-identical inputs.
2. Sensitivity: perturbing the price of D-1 and the covariates of D-2 MUST change the inputs (the builder really
   uses the latest allowed data).
3. Training helper: train_windows / train_panel refuse anything after 2025-06-30.
4. Model determinism / no hidden data access: each listed model gives identical forecasts on the original and the
   perturbed panel (its only inputs are the build_context arrays) and on a repeated call.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D        # noqa: E402

CTX = 736


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="naive_d,naive_w,naive_epf,arx")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dates, P, C, flags = D.panel()
    rng = np.random.default_rng(1)
    pool = np.concatenate([D.split_days("dev"), D.split_days("val"), D.split_days("test")])
    days = rng.choice(pool, size=a.n, replace=False)
    days = np.concatenate([days, flags["dst_gap"][-2:], flags["dst_dup"][-2:]])
    days = [int(i) for i in days if i >= CTX]
    for i in days:
        w0 = D.build_context(i, CTX, dates, P, C, flags)
        P2, C2 = P.copy(), C.copy()
        P2[i:] = rng.normal(0, 500, P2[i:].shape)
        C2[i - 1:] = rng.normal(0, 1e5, C2[i - 1:].shape)
        w1 = D.build_context(i, CTX, dates, P2, C2, flags)
        assert np.array_equal(w0["prices"], w1["prices"], equal_nan=True), f"price leak at {dates[i]}"
        assert np.array_equal(w0["covs"], w1["covs"], equal_nan=True), f"covariate leak at {dates[i]}"
        P3, C3 = P.copy(), C.copy()
        P3[i - 1, 5] += 1.0
        C3[i - 2, 5, 0] += 1.0
        w3 = D.build_context(i, CTX, dates, P3, C3, flags)
        assert not np.array_equal(w0["prices"], w3["prices"], equal_nan=True), "D-1 price not used"
        assert not np.array_equal(w0["covs"], w3["covs"], equal_nan=True), "D-2 covariate not used"
        assert np.isnan(w0["covs"][-24:]).all(), "D-1 covariates must be NaN"
    print(f"perturbation/sensitivity: OK on {len(days)} days ({dates[min(days)]} .. {dates[max(days)]}), incl. DST days")
    try:
        D.train_panel("2025-07-01")
        raise AssertionError("train_panel accepted a date after TRAIN_END")
    except ValueError:
        pass
    X, XC, Y, T = D.train_windows(ctx_days=14, start="2025-06-01")
    assert T.max() <= np.datetime64("2025-06-30") and len(T) > 0
    print(f"train helpers: OK (last training target {T.max()}, {len(T)} windows from 2025-06-01)")
    import models as M
    sub = days[:4]
    for mn in [m for m in a.models.split(",") if m]:
        p = M.get(mn)
        res = []
        for variant in range(3):            # 0 original, 1 future perturbed, 2 repeat of 0
            ws = []
            for i in sub:
                Pv, Cv = P.copy(), C.copy()
                if variant == 1:
                    Pv[i:] = rng.normal(0, 500, Pv[i:].shape)
                    Cv[i - 1:] = rng.normal(0, 1e5, Cv[i - 1:].shape)
                ws.append(D.build_context(i, CTX, dates, Pv, Cv, flags))
            kw = {}
            if getattr(p, "wants_dates", False):
                kw["dates"] = [{"ctx_dates": w["ctx_dates"], "target_day": w["target_day"]} for w in ws]
            res.append(np.asarray(p.batch([w["prices"] for w in ws], [w["covs"] for w in ws], 24,
                                          [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975], **kw)))
        d01 = float(np.nanmax(np.abs(res[0] - res[1])))
        d02 = float(np.nanmax(np.abs(res[0] - res[2])))
        # bit-identical expected; <= 1e-8 EUR/MWh tolerated only as floating-point (BLAS) noise and printed
        assert d01 <= 1e-8 and np.array_equal(np.isnan(res[0]), np.isnan(res[1])), f"{mn}: changed when the future was perturbed ({d01})"
        assert d02 <= 1e-8, f"{mn}: not deterministic ({d02})"
        print(f"  {mn}: max |diff| perturbed {d01:.2e}, repeat {d02:.2e}")
        print(f"model {mn}: future-invariant and deterministic on {len(sub)} days")
    print("ALL INFORMATION-SET TESTS PASSED")


if __name__ == "__main__":
    main()
