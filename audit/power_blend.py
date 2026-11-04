"""Power small experiment: quantile blend of wm_power with LEAR (repo competitor). Weight chosen on VAL
(2025-07..10) by P2 hourly accuracy + QS; applied once to TEST. Forecasts via the repo harness into audit/power_*."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools", "power"))
import run_eval as R, data as D
dates, P, C, flags = D.panel()
for sp in ("val", "test"):
    R.RUN_DIR[sp] = os.path.join(HERE, f"power_{sp}")
    R.forecast(sp, ["wm_power", "lear"])
def load(sp, mn):
    z = np.load(os.path.join(HERE, f"power_{sp}", f"{mn}.npz"), allow_pickle=True)
    return z, z["days"] if "days" in z.files else z["idx"]
def score(q, idx):  # q [n, Q, 24]; median at QS index of 0.5
    qs = np.array(R.QS); im = int(np.argmin(np.abs(qs - 0.5))); i9 = [int(np.argmin(np.abs(qs - x))) for x in np.arange(0.1, 0.91, 0.1)]
    cur, prev = P[idx], P[idx - 1]; med = q[:, im]
    m2 = np.isfinite(cur) & np.isfinite(prev) & (cur != prev)
    p2 = np.mean(((med > prev) == (cur > prev))[m2])
    ok = np.isfinite(cur).all(1)
    p1 = np.mean(((med.mean(1) > prev.mean(1)) == (cur.mean(1) > prev.mean(1)))[ok])
    lv = qs[i9][None, :, None]; d = cur[:, None] - q[:, i9]
    return p1, p2, np.nanmean(np.maximum(lv * d, (lv - 1) * d)[ok]), np.nanmean(np.abs(med - cur)[ok])
res = {}
for sp in ("val", "test"):
    za, ia = load(sp, "wm_power"); zb, ib = load(sp, "lear"); assert np.array_equal(ia, ib)
    res[sp] = (za["q"], zb["q"], ia)
grid = np.round(np.arange(0, 1.01, 0.1), 1)
qa, qb, idx = res["val"]
vs = [(w,) + score(np.sort((1 - w) * qa + w * qb, 1), idx) for w in grid]
for v in vs: print("VAL w=%.1f P1 %.3f P2 %.3f QS %.2f MAE %.2f" % v)
w = max(vs, key=lambda r: r[2] - r[3] / 100)[0]; print("chosen", w)
qa, qb, idx = res["test"]
for name, q in (("wm_power", qa), ("lear", qb), (f"blend w={w}", np.sort((1 - w) * qa + w * qb, 1))):
    print("TEST %-14s P1 %.3f P2 %.3f QS %.2f MAE %.2f" % ((name,) + score(q, idx)))
