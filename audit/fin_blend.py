"""Fin small experiment: quantile blend of wm_fin with har_intra_dow (log-RV quantiles averaged); direction = blended
median > label reference. Weight chosen on VAL (2025-07..10) only, then applied once to TEST."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "repo")
QS = np.array([0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975]); I9 = slice(2, 11)
d = pd.concat([pd.read_parquet(f"{REPO}/data/fin/crypto_daily.parquet"), pd.read_parquet(f"{REPO}/data/fin/crypto_daily_SEALED.parquet")])
d["day"] = pd.to_datetime(d.t).dt.tz_localize(None).dt.normalize()
lrv = np.log(d.pivot_table(index="day", columns="symbol", values="rv").where(d.pivot_table(index="day", columns="symbol", values="n_lab") >= 274))
ref = lrv.shift(1).rolling(7).mean().where(lrv.shift(1).rolling(7).count() == 7)
held = set(json.load(open(f"{REPO}/data/fin/universe.json"))["heldout"])
def get(split, mn):
    z = np.load(f"{HERE}/fin_{split}/crypto/{mn}.npz", allow_pickle=True)
    days = pd.to_datetime(z["dates"]); syms = list(z["symbols"])
    return z["q"], lrv.reindex(index=days, columns=syms).to_numpy(), ref.reindex(index=days, columns=syms).to_numpy(), np.array([s in held for s in syms])
def score(q, L, RF, cols):
    q, L, RF = q[:, cols], L[:, cols], RF[:, cols]
    ok = np.isfinite(L) & np.isfinite(RF) & np.isfinite(q[..., I9]).all(-1)
    acc = np.mean((q[..., 6] > RF)[ok] == (L > RF)[ok])
    dd = L[..., None] - q[..., I9]; pin = np.maximum(QS[I9] * dd, (QS[I9] - 1) * dd).mean(-1)
    return acc, pin[ok].mean(), ((q[..., 6] > RF) == (L > RF))[ok], pin[ok]
grid = np.round(np.arange(0, 1.01, 0.1), 1)
for other in ("har_intra_dow", "lgbm_dow"):
    qa, L, RF, H = get("val", "wm_fin"); qb = get("val", other)[0]
    allc = np.arange(qa.shape[1])
    res = [(w,) + score(np.sort((1 - w) * qa + w * qb, -1), L, RF, allc)[:2] for w in grid]
    print(f"\nVAL wm_fin+{other}:", " ".join(f"w={w}:acc{a:.4f}/qs{q:.4f}" for w, a, q in res))
    best = max(res, key=lambda r: r[1] - 2 * r[2])     # one combined criterion, fixed before looking at TEST
    w = best[0]
    print(f"chosen on VAL: w={w}")
    qa, L, RF, H = get("test", "wm_fin"); qb = get("test", other)[0]
    for grp, cols in (("train", np.where(~H)[0]), ("heldout", np.where(H)[0]), ("all", allc)):
        a0, s0, h0, p0 = score(qa, L, RF, cols); ab, sb, hb, pb = score(qb, L, RF, cols)
        a1, s1, h1, p1 = score(np.sort((1 - w) * qa + w * qb, -1), L, RF, cols)
        print(f"TEST {grp:8s} wm_fin acc {a0:.4f} qs {s0:.4f} | {other} acc {ab:.4f} qs {sb:.4f} | blend acc {a1:.4f} qs {s1:.4f}")
