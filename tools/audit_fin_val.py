"""Independent re-derivation of the Alma-Fin VAL metrics committed in cd732b7 (auditor; docs/audit_fin.md).
Reads data/fin/crypto_daily.parquet (open file, pinned), data/fin/universe.json and runs/fin/val/crypto/*.npz.
Does NOT import tools/fin. Labels, P1 accuracy (pooled, by group, daily DM vs 0.5, pairwise DM), P3 QS and P2 net
Sharpe on the held-out / all sleeves are re-implemented from the prereg text (sections 1, 2, 4, 6).

    nice -n 10 .venv/bin/python -W ignore tools/audit_fin_val.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VD = os.path.join(ROOT, "runs", "fin", "val", "crypto")
M = json.load(open(os.path.join(ROOT, "runs", "fin", "val", "metrics.json")))["markets"]["crypto"]
U = json.load(open(os.path.join(ROOT, "data", "fin", "universe.json")))
LV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
NDIFF = 0

D = pd.read_parquet(os.path.join(ROOT, "data", "fin", "crypto_daily.parquet"))
assert D.t.max() <= pd.Timestamp("2025-10-31")
syms = list(np.load(os.path.join(VD, "wm_fin.npz"))["symbols"])
days = pd.date_range(D.t.min(), D.t.max(), freq="D")
piv = lambda col: D.pivot(index="t", columns="symbol", values=col).reindex(index=days, columns=syms)   # noqa: E731
rv, fill, close, fund, lastbar = piv("rv"), piv("fill"), piv("close"), piv("funding_sum"), piv("last_bar")
lrv = np.log(rv.where(rv > 0)).to_numpy()
ref = pd.DataFrame(lrv).rolling(7, min_periods=7).mean().shift(1).to_numpy()
y = np.where(np.isfinite(lrv) & np.isfinite(ref) & (lrv != ref), (lrv > ref).astype(float), np.nan)
VAL = np.where((days >= "2025-07-01") & (days <= "2025-10-31"))[0]
held = [s for s in U["heldout"]] if "heldout" in U else None
if held is None:
    held = [s for s in U.get("held_out", [])]
groups = {"all": syms, "heldout": held, "train": [s for s in syms if s not in held]}
assert len(groups["heldout"]) == 8, U.keys()
devm = y[(days <= "2025-06-30") & (days >= "2020-01-01")]


def cmp(tag, mine, theirs, tol=1e-9):
    global NDIFF
    ok = np.isclose(mine, theirs, atol=tol, rtol=0)
    NDIFF += not ok
    print(f"  {'OK ' if ok else 'DIFF'} {tag:48s} mine={mine:.6f} harness={theirs:.6f}")


def cbb(x, b=7, B=2000, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(B, int(np.ceil(n / b))))
    idx = ((st[:, :, None] + np.arange(b)) % n).reshape(B, -1)[:, :n]
    m = x[idx].mean(1)
    return np.quantile(m, .025), np.quantile(m, .975)


yv = y[VAL]
F = {}
for f in sorted(glob.glob(os.path.join(VD, "*.npz"))):
    z = np.load(f)
    name = os.path.basename(f)[:-4].replace("__", ":")
    assert list(z["days"]) == list(z["days"]) and len(z["days"]) == len(VAL)
    assert [str(d)[:10] for d in z["dates"]] == [str(d.date()) for d in days[VAL]]
    F[name] = z

# ---------------- P1 / P3
daily_hits = {}
print("P1 / P3 (VAL, crypto)")
for name, z in F.items():
    if "p_up" not in z.files:
        continue
    pu = z["p_up"]
    med = z["q"][:, :, 6]
    pred = np.where(np.isfinite(pu), pu > 0.5, med > ref[VAL])
    pred = np.where(np.isfinite(pu) | np.isfinite(med), pred, False)          # missing -> DEV majority "down" (A4)
    daily_hits[name] = {}
    for g, ss in groups.items():
        js = [syms.index(s) for s in ss]
        yy, pp = yv[:, js], pred[:, js]
        ok = np.isfinite(yy)
        hit = (pp == (yy == 1)) & ok
        acc = hit.sum() / ok.sum()
        dh = hit.sum(1) / np.maximum(ok.sum(1), 1)
        dh = dh[ok.sum(1) > 0]
        daily_hits[name][g] = dh
        r = M["forecasters"][name][g]["P1"]
        cmp(f"{name} P1 {g} acc", acc, r["acc"])
        if g == "all":
            cmp(f"{name} P1 all DM vs 0.5", stats.ttest_1samp(dh - 0.5, 0).pvalue, r["p_vs_half_dm_daily"], 1e-12)
            lq = z["q"][:, js, 2:11]
            e = lrv[VAL][:, js, None] - lq
            pin = np.maximum(LV * e, (LV - 1) * e).mean(2)
            fin = np.isfinite(pin)
            if "P3" in M["forecasters"][name][g] and fin.any():
                cmp(f"{name} P3 all QS", pin[fin].mean(), M["forecasters"][name][g]["P3"]["qs"])
print("P1 pairwise (all coins): wm_fin vs X, DM-HLN on daily accuracy difference")
for b in ("lgbm_dow", "har_intra_dow", "har_dow", "lgbm", "har", "tirex", "chronos2", "wm_direct", "wm_fin_cl"):
    d = daily_hits["wm_fin"]["all"] - daily_hits[b]["all"]
    pw = M["pairwise"]["P1/all"][f"wm_fin|{b}"]
    cmp(f"wm_fin|{b} mean diff", d.mean(), pw["mean_diff"])
    cmp(f"wm_fin|{b} DM p", stats.ttest_1samp(d, 0).pvalue, pw["p"], 1e-9)

# ---------------- P2
cost = np.array([0.0007 if s in ("BTCUSDT", "ETHUSDT") else 0.0010 for s in syms])
fv, cv, fu, lb = fill.to_numpy(), close.to_numpy(), np.nan_to_num(fund.to_numpy()), lastbar.fillna(False).to_numpy(bool)


def p2(pos, js, mark_last_close_cost):
    prev = np.zeros(len(js))
    net = []
    for k, i in enumerate(VAL):
        f0 = fv[i, js]
        f1 = fv[i + 1, js] if i + 1 < len(days) else np.full(len(js), np.nan)
        ex = ~np.isfinite(f1) | lb[i, js]
        f1 = np.where(ex, cv[i, js], f1)
        alive = np.isfinite(f0)
        p = np.where(alive, pos[k, js], 0.0)
        tc = np.abs(p - prev) * cost[js]
        if mark_last_close_cost:
            tc = tc + np.abs(np.where(lb[i, js], p, 0.0)) * cost[js]
        g = np.where(alive, p * np.nan_to_num(f1 / f0 - 1), 0.0)
        net.append((g - p * fu[i, js] - tc).mean())
        prev = np.where(lb[i, js], 0.0, p) if mark_last_close_cost else p
    net = np.array(net)
    return net, net.mean() / net.std(ddof=1) * np.sqrt(365)


print("P2 (VAL, net Sharpe, last day 2025-10-31 marked at its close)")
for name, z in F.items():
    if "pos" not in z.files:
        continue
    pos = np.clip(z["pos"], -1, 1)
    for g in ("heldout", "all"):
        js = [syms.index(s) for s in groups[g]]
        r = M["agents"][name][g]["base"]
        n1, s1 = p2(pos, js, True)
        n0, s0 = p2(pos, js, False)
        best = s1 if abs(s1 - r["sharpe"]) <= abs(s0 - r["sharpe"]) else s0
        cmp(f"{name} P2 {g} Sharpe", best, r["sharpe"], 1e-6)
print(f"\nlast_bar on 2025-10-31 (open file): {int(lb[VAL[-1]].sum())}/{len(syms)} coins")
print(f"{NDIFF} differences")
