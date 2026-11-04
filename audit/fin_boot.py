import numpy as np, sys
sys.argv = ["x"]; exec(open("fin_blend.py").read().split("grid =")[0])
def perday(q, L, RF):
    ok = np.isfinite(L) & np.isfinite(RF) & np.isfinite(q[..., I9]).all(-1)
    hit = np.where(ok, (q[..., 6] > RF) == (L > RF), np.nan)
    dd = L[..., None] - q[..., I9]; pin = np.where(ok, np.maximum(QS[I9] * dd, (QS[I9] - 1) * dd).mean(-1), np.nan)
    return hit, pin
qa, L, RF, H = get("test", "wm_fin"); qh = get("test", "har_intra_dow")[0]; qd = get("test", "wm_direct")[0]
cands = {"wm_fin": qa, "har_intra_dow": qh, "blend0.4": np.sort(0.6 * qa + 0.4 * qh, -1),
         "blend0.3": np.sort(0.7 * qa + 0.3 * qh, -1), "blend0.5": np.sort(0.5 * qa + 0.5 * qh, -1),
         "blend3(wm,dir,har)": np.sort((0.3 * qa + 0.3 * qd + 0.4 * qh), -1)}
S = {k: perday(v, L, RF) for k, v in cands.items()}
rng = np.random.default_rng(0); n = L.shape[0]; B = 4000; blk = 7
idx = [np.concatenate([np.arange(s, s + blk) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(B)]
def boot(a, b, j):
    da = np.nanmean(S[a][j], 1) - np.nanmean(S[b][j], 1)          # per-day difference (24 coins pooled)
    m = np.nanmean(da); bs = np.array([np.nanmean(da[i]) for i in idx])
    return m, np.percentile(bs, [2.5, 97.5]), np.mean(bs <= 0) if j == 0 else np.mean(bs >= 0)
for a in ["blend0.3", "blend0.4", "blend0.5", "blend3(wm,dir,har)"]:
    for b in ["wm_fin", "har_intra_dow"]:
        m, ci, p = boot(a, b, 0); mq, ciq, pq = boot(a, b, 1)
        print(f"{a:20s} vs {b:14s} acc {m*100:+.2f}pp CI[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] p1s={p:.3f} | QS {mq:+.4f} CI[{ciq[0]:+.4f},{ciq[1]:+.4f}] p1s={pq:.3f}")
