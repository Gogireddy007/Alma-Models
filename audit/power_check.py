"""Independent re-score of the frozen power models (wm_power, wm_power_direct) on the TEST year.

Reuses data.build_context (read by hand: prices <= D-1, covariates <= D-2, asserted) and the frozen wrapper.
Own: P1/P2/P3 direction metrics, QS/MAE, an own ridge-ARX baseline fitted on <= 2025-06-30 only, and a leak test
(every price/covariate of day D and later scrambled -> forecasts must not change).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo")
sys.path.insert(0, os.path.join(REPO, "tools", "power"))
import data as PD   # noqa: E402
import wm_power as WP  # noqa: E402

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CTX = 30
PEAK = list(range(8, 20))


def metrics(name, med, Q, P, idx):
    prev, cur = P[idx - 1], P[idx]
    ok = np.isfinite(cur).all(1) & np.isfinite(prev).all(1)
    p1 = np.mean(((med.mean(1) > prev.mean(1)) == (cur.mean(1) > prev.mean(1)))[ok])
    m2 = np.isfinite(cur) & np.isfinite(prev) & (cur != prev)
    p2 = np.mean(((med > prev) == (cur > prev))[m2])
    spr = lambda x: x[:, PEAK].mean(1) - np.delete(x, PEAK, 1).mean(1)
    p3 = np.mean(((spr(med) > spr(prev)) == (spr(cur) > spr(prev)))[ok])
    out = dict(model=name, n=int(ok.sum()), P1=p1, P2=p2, P3=p3, MAE=np.nanmean(np.abs(med - cur)[ok]))
    if Q is not None:
        lv = np.array(QS)[None, :, None]
        d = cur[:, None, :] - Q
        out["QS"] = np.nanmean(np.maximum(lv * d, (lv - 1) * d)[ok])
        out["cov80"] = np.nanmean(((cur >= Q[:, 0]) & (cur <= Q[:, -1]))[ok])
    return out


def arx_fit(dates, P, C, end):
    """Per-hour ridge on: same-hour price D-1, D-2, D-7, D-1 daily mean/min/max, load/wind/solar of D-2, weekday dummies."""
    def feats(i):
        pd1, pd2, pd7 = P[i - 1], P[i - 2], P[i - 7]
        c2 = C[i - 2]
        dow = np.zeros(7)
        dow[pd.Timestamp(dates[i]).weekday()] = 1
        return [np.r_[pd1[h], pd2[h], pd7[h], np.nanmean(pd1), np.nanmin(pd1), np.nanmax(pd1),
                      c2[h] / 1e4, dow] for h in range(24)]
    tr = [i for i in range(400, len(dates)) if dates[i] <= np.datetime64(end) and dates[i] >= np.datetime64("2019-01-01")
          and np.isfinite(P[i]).all() and np.isfinite(P[i - 7:i]).all() and np.isfinite(C[i - 2]).all()]
    W = []
    for h in range(24):
        X = np.array([feats(i)[h] for i in tr])
        y = np.array([P[i, h] for i in tr])
        X1 = np.c_[X, np.ones(len(X))]
        W.append(np.linalg.solve(X1.T @ X1 + 1.0 * np.eye(X1.shape[1]), X1.T @ y))
    return lambda i: np.array([np.r_[feats(i)[h], 1] @ W[h] for h in range(24)])


def main():
    dates, P, C, flags = PD.panel()
    idx = np.array([i for i, d in enumerate(dates) if d >= np.datetime64("2025-11-01")])
    print(f"TEST days {len(idx)}: {dates[idx[0]]} .. {dates[idx[-1]]}")
    rows = []
    for name in ("wm_power", "wm_power_direct"):
        pred = WP._factory(name)()
        ctxs = [PD.build_context(int(i), CTX) for i in idx]
        Q = pred.batch([c["prices"] for c in ctxs], [c["covs"] for c in ctxs], 24, QS,
                       dates=[{"ctx_dates": c["ctx_dates"], "target_day": c["target_day"]} for c in ctxs])
        rows.append(metrics(name, Q[:, 4], Q, P, idx))
        if name == "wm_power":   # leak test
            rng = np.random.default_rng(0)
            bad = 0
            for i in idx[::15]:
                c0 = PD.build_context(int(i), CTX)
                P2, C2 = P.copy(), C.copy()
                P2[i:] = rng.uniform(-200, 600, P2[i:].shape)
                C2[i - 1:] = rng.uniform(0, 8e4, C2[i - 1:].shape)     # D-1 covariates are also unknown
                c1 = PD.build_context(int(i), CTX, dates, P2, C2, flags)
                a = pred.batch([c0["prices"]], [c0["covs"]], 24, QS, dates=[c0])
                b = pred.batch([c1["prices"]], [c1["covs"]], 24, QS, dates=[c1])
                bad += not np.allclose(a, b)
            print(f"leak test: {bad} of {len(idx[::15])} days changed after scrambling day D and later")
    # simple baselines
    rows.append(metrics("naive_d (P[D-1])", P[idx - 1], None, P, idx))
    rows.append(metrics("naive_w (P[D-7])", P[idx - 7], None, P, idx))
    arx = arx_fit(dates, P, C, "2025-06-30")
    rows.append(metrics("own_ridge_arx", np.stack([arx(i) for i in idx]), None, P, idx))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_check.csv"), index=False)


if __name__ == "__main__":
    main()
