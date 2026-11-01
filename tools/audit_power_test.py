"""Independent re-derivation of the Alma-Power TEST numbers (auditor; docs/audit_power.md).
Reads only data/power/hourly.csv and runs/power/test/*.npz. Does NOT import tools/power (panel, labels, majority,
metrics, DM, McNemar, block bootstrap and Holm re-implemented here). Compares with runs/power/test/metrics.json.

    nice -n 10 .venv/bin/python tools/audit_power_test.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, "runs", "power", "test")
M = json.load(open(os.path.join(TD, "metrics.json")))
QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
LV = np.array(QS[2:11])
PEAK = np.arange(8, 20)
OFF = np.setdiff1d(np.arange(24), PEAK)
NDIFF = 0


def panel():
    H = pd.read_csv(os.path.join(ROOT, "data", "power", "hourly.csv"), usecols=["date", "slot", "price"])
    g = H.groupby(["date", "slot"])["price"].mean().unstack("slot").reindex(columns=range(24))
    return np.array(g.index.values, dtype="datetime64[D]"), g.to_numpy(float)


DATES, P = panel()
DPOS = {d: i for i, d in enumerate(DATES)}
TEST = np.where((DATES >= np.datetime64("2025-11-01")))[0]
DEV_END = np.datetime64("2025-06-30")
GAP = set(np.where(np.isnan(P).sum(1) == 1)[0][np.isnan(P[np.isnan(P).sum(1) == 1, 2])])   # spring-forward days


def wd(i):
    return pd.Timestamp(DATES[i]).weekday()


def ok_day(i):
    need = np.ones(24, bool)
    if i in GAP:
        need[2] = False
    return np.isfinite(P[i][need]).all()


def spread(x):
    return np.nanmean(x[PEAK]) - np.nanmean(x[OFF])


def dev_maj():
    p1, p3 = [], []
    up2 = np.zeros((7, 24))
    n2 = np.zeros((7, 24))
    for i in range(1, len(DATES)):
        if DATES[i] > DEV_END:
            break
        w = wd(i)
        if ok_day(i) and ok_day(i - 1):
            b1, b0 = np.nanmean(P[i]), np.nanmean(P[i - 1])
            if b1 != b0:
                p1.append((w, b1 > b0))
            s1, s0 = spread(P[i]), spread(P[i - 1])
            if s1 != s0:
                p3.append((w, s1 > s0))
        ok = np.isfinite(P[i]) & np.isfinite(P[i - 1]) & (P[i] != P[i - 1])
        up2[w] += (P[i] > P[i - 1]) & ok
        n2[w] += ok
    f = lambda L, k=None: np.mean([u for w, u in L if k is None or w == k])   # noqa: E731
    return {"P1": f(p1), "P3": f(p3), "P2": up2.sum() / n2.sum(),
            "P1_wd": [f(p1, k) for k in range(7)], "P3_wd": [f(p3, k) for k in range(7)], "P2_wd": up2 / np.maximum(n2, 1)}


MAJ = dev_maj()


def load_models():
    out = {}
    for f in sorted(glob.glob(os.path.join(TD, "*.npz"))):
        z = np.load(f)
        days = [DPOS[np.datetime64(d, "D")] for d in z["dates"]]
        assert np.array_equal(days, z["days"]), f
        out[os.path.basename(f)[:-4]] = dict(zip(days, z["q"]))
    return out


Q = load_models()
NAMES = list(Q) + ["majority", "dow_majority"]


def pred(mn, tgt, i):
    w = wd(i)
    if mn == "majority":
        return int(MAJ[tgt] > .5) if tgt != "P2" else np.full(24, int(MAJ["P2"] > .5))
    if mn == "dow_majority":
        return (MAJ["P2_wd"][w] > .5).astype(int) if tgt == "P2" else int(MAJ[tgt + "_wd"][w] > .5)
    if i not in Q[mn]:
        return None
    m = Q[mn][i][6].copy()
    if i in GAP:
        m[2] = np.nan
    if tgt == "P1":
        pv, rv = np.nanmean(m), np.nanmean(P[i - 1])
        return int(pv > rv) if pv != rv else int(MAJ["P1"] > .5)
    if tgt == "P3":
        pv, rv = spread(m), spread(P[i - 1])
        return int(pv > rv) if pv != rv else int(MAJ["P3"] > .5)
    m = Q[mn][i][6]
    return np.where(m > P[i - 1], 1, np.where(m < P[i - 1], 0, int(MAJ["P2"] > .5)))


def cmp(tag, mine, theirs, tol=1e-9):
    global NDIFF
    same = np.isclose(mine, theirs, atol=tol, rtol=0)
    NDIFF += not same
    print(f"  {'OK ' if same else 'DIFF'} {tag:46s} mine={mine:.6f} harness={theirs:.6f}")


def dm(d):
    return stats.ttest_1samp(d, 0).pvalue          # DM-HLN at h=1 == one-sample t


def boot(d, B=2000, b=7, seed=0):
    n = len(d)
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(B, int(np.ceil(n / b))))
    idx = ((st[:, :, None] + np.arange(b)) % n).reshape(B, -1)[:, :n]
    m = d[idx].mean(1)
    return np.quantile(m, .025), np.quantile(m, .975)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    out = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i]))
        out[i] = run
    return out


series = {}
# ---------------- P1 / P3
for tgt in ("P1", "P3"):
    days = [i for i in TEST if ok_day(i) and ok_day(i - 1)]
    act = {}
    for i in days:
        a, b = (np.nanmean(P[i]), np.nanmean(P[i - 1])) if tgt == "P1" else (spread(P[i]), spread(P[i - 1]))
        if a != b:
            act[i] = int(a > b)
    print(f"[{tgt}] n={len(act)} base rate up={np.mean(list(act.values())):.4f}")
    series[tgt] = {}
    for mn in NAMES:
        h = {i: float(pred(mn, tgt, i) == y) for i, y in act.items() if pred(mn, tgt, i) is not None}
        series[tgt][mn] = h
        cmp(f"{mn} acc", np.mean(list(h.values())), M["models"][mn][tgt]["acc"])
# ---------------- P2
series["P2"] = {}
for mn in NAMES:
    num = den = 0
    rate = {}
    for i in TEST:
        pr = pred(mn, "P2", i)
        if pr is None:
            continue
        ok = np.isfinite(P[i]) & np.isfinite(P[i - 1]) & (P[i] != P[i - 1])
        h = (pr[ok] == (P[i][ok] > P[i - 1][ok])).astype(float)
        num += h.sum()
        den += ok.sum()
        rate[i] = h.mean()
    series["P2"][mn] = rate
    cmp(f"P2 {mn} acc", num / den, M["models"][mn]["P2"]["acc"])
# ---------------- P4
qsd, aed, C80, C95 = {}, {}, {}, {}
for mn in Q:
    qsd[mn], aed[mn] = {}, {}
    c80, c95 = [], []
    C80[mn], C95[mn] = c80, c95
    for i, q in Q[mn].items():
        y = P[i]
        s = np.isfinite(y)
        qq = q[:, s].T
        if np.isfinite(qq[:, 2:11]).all():
            e = y[s][:, None] - qq[:, 2:11]
            qsd[mn][i] = np.maximum(LV * e, (LV - 1) * e).mean(1).mean()
            c80.append(((y[s] >= qq[:, 2]) & (y[s] <= qq[:, 10])).mean())
        if np.isfinite(qq[:, [0, 12]]).all():
            c95.append(((y[s] >= qq[:, 0]) & (y[s] <= qq[:, 12])).mean())
        aed[mn][i] = np.abs(y[s] - qq[:, 6]).mean()
for mn in Q:
    c80, c95 = C80[mn], C95[mn]
    r = M["models"][mn]["P4"]
    cmp(f"P4 {mn} QS", np.mean(list(qsd[mn].values())), r["qs"])
    cmp(f"P4 {mn} MAE", np.mean(list(aed[mn].values())), r["mae"])
    cmp(f"P4 {mn} cov80", np.mean(c80), r["cov80"])
    if c95:
        cmp(f"P4 {mn} cov95", np.mean(c95), r["cov95"])
    com = [i for i in qsd[mn] if i in qsd["naive_epf"]]
    cmp(f"P4 {mn} QS skill vs naive_epf", 1 - np.mean([qsd[mn][i] for i in com]) / np.mean([qsd["naive_epf"][i] for i in com]),
        r["skill_vs_naive_epf"])
series["P4_QS"], series["P4_MAE"] = qsd, aed

# ---------------- pairwise (wm_power) with own-Holm over non-contaminated competitors, prereg section 7 rule
print("\nPairwise wm_power vs X (diff = wm - X; accuracy: >0 better; loss: <0 better)")
KEY = ["lear", "chronos2", "chronos2_cov", "tirex", "chronos_bolt_base", "wm_power_direct", "naive_w", "dow_majority"]
for tgt in ("P1", "P2", "P3", "P4_QS", "P4_MAE"):
    S = series[tgt]
    comps = [b for b in S if b not in ("wm_power", "timestral_5m")]
    res = {}
    for b in comps:
        com = sorted(set(S["wm_power"]) & set(S[b]))
        d = np.array([S["wm_power"][i] - S[b][i] for i in com])
        res[b] = (d.mean(), dm(d) if d.std() > 0 else 1.0, *boot(d), len(com))
    adj = dict(zip(comps, holm([res[b][1] for b in comps])))
    for b in KEY:
        if b not in res:
            continue
        md, p, lo, hi, n = res[b]
        pw = M["pairwise"][tgt][f"wm_power|{b}"]
        better = md > 0 if not tgt.startswith("P4") else md < 0
        beats = better and adj[b] < 0.05 and (lo > 0 or hi < 0)
        cmp(f"{tgt} vs {b} p", p, pw["p"])
        cmp(f"{tgt} vs {b} Holm-own", adj[b], pw["p_holm_own"])
        cmp(f"{tgt} vs {b} boot_lo", lo, pw["boot_lo"], 1e-9)
        print(f"       diff={md:+.4f} CI[{lo:+.4f},{hi:+.4f}] n={n} beats={beats} (harness {pw['beats']})")
print(f"\n{NDIFF} differences")
