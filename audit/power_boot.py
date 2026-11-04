"""Day-block bootstrap (7-day blocks, 4000 reps) for the power blend vs wm_power and vs LEAR on TEST: per-day QS and P2."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "repo", "tools", "power"))
import run_eval as R, data as D
dates, P, C, flags = D.panel()
z = {m: np.load(os.path.join(HERE, "power_test", f"{m}.npz"), allow_pickle=True) for m in ("wm_power", "lear")}
idx = z["wm_power"]["days"] if "days" in z["wm_power"].files else z["wm_power"]["idx"]
qs = np.array(R.QS); im = int(np.argmin(np.abs(qs - 0.5))); i9 = [int(np.argmin(np.abs(qs - x))) for x in np.arange(0.1, 0.91, 0.1)]
qa, qb = z["wm_power"]["q"], z["lear"]["q"]
Q = {"wm_power": qa, "lear": qb, "blend": np.sort(0.5 * qa + 0.5 * qb, 1)}
cur, prev = P[idx], P[idx - 1]; ok = np.isfinite(cur).all(1)
def perday(q):
    lv = qs[i9][None, :, None]; d = cur[:, None] - q[:, i9]
    pin = np.maximum(lv * d, (lv - 1) * d).mean((1, 2))
    med = q[:, im]; m2 = np.isfinite(cur) & np.isfinite(prev) & (cur != prev)
    hit = np.where(m2, (med > prev) == (cur > prev), np.nan)
    return pin, np.nanmean(hit, 1)
S = {k: perday(v) for k, v in Q.items()}
n = int(ok.sum()); rng = np.random.default_rng(0); blk = 7
bidx = [np.concatenate([np.arange(s, s + blk) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(4000)]
for b in ("wm_power", "lear"):
    for j, nm in ((0, "QS"), (1, "P2")):
        d = (S["blend"][j] - S[b][j])[ok]; bs = np.array([np.nanmean(d[i]) for i in bidx])
        print(f"blend vs {b:9s} {nm}: diff {np.nanmean(d):+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  (n days {n})")
