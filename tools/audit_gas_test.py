"""Independent re-derivation of the Alma-Gas storage TEST numbers (auditor; docs/audit_gas.md).
Reads data/gas/weekly.csv (pinned), the raw EIA xls (L48 first-print cross-check) and runs/gas/test/storage/*/*.npz.
Does NOT import tools/gas; dS, avg5, labels, majority, metrics, McNemar, DM, bootstrap and Holm are re-implemented.
Also computes the A3 robustness table (weeks ending 2025-10-31 and 2025-11-07 dropped).

    nice -n 10 .venv/bin/python -W ignore tools/audit_gas_test.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, "runs", "gas", "test", "storage")
M = json.load(open(os.path.join(ROOT, "runs", "gas", "test", "metrics.json")))["storage"]
QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
LV = np.array(QS[2:11])
SER = ["l48", "east", "midwest", "mountain", "pacific", "south_central"]
CONTAM = {"timestral_5m"}
NDIFF = 0

W = pd.read_csv(os.path.join(ROOT, "data", "gas", "weekly.csv"), parse_dates=["week", "avail"])

# ---------------- labels from levels (own dS and avg5)
for s in SER:
    lvl = W[f"lvl_{s}"].to_numpy(float)
    ds = np.r_[np.nan, np.diff(lvl)]
    a5 = np.full(len(W), np.nan)
    wk = W.week.values
    for i in range(len(W)):
        vals = []
        for k in range(1, 6):
            tgt = wk[i] - np.timedelta64(364 * k, "D")
            j = np.where(np.abs((wk - tgt) / np.timedelta64(1, "D")) <= 3)[0]
            if len(j) == 1 and np.isfinite(ds[j[0]]):
                vals.append(ds[j[0]])
        if len(vals) == 5:
            a5[i] = np.mean(vals)
    W[f"my_ds_{s}"], W[f"my_a5_{s}"], W[f"my_dev_{s}"] = ds, a5, ds - a5
TEST = np.where((W.avail >= "2025-11-01") & np.isfinite(W["my_a5_l48"]))[0]
for s in SER:
    for c in ("ds", "avg5", "dev"):
        mine = W[f"my_{c if c != 'avg5' else 'a5'}_{s}"].to_numpy()[TEST]
        theirs = W[f"{c}_{s}"].to_numpy()[TEST]
        if not np.allclose(mine, theirs, equal_nan=True, atol=1e-6):
            NDIFF += 1
            print(f"DIFF label column {c}_{s}")
print(f"TEST weeks {len(TEST)}: {W.week.iloc[TEST[0]].date()} .. {W.week.iloc[TEST[-1]].date()}; own dS/avg5/dev "
      f"{'match' if NDIFF == 0 else 'DIFFER'} weekly.csv for all 6 series")

# L48 first-print cross-check against the raw EIA files
def sheet(f, sh):
    x = pd.read_excel(os.path.join(ROOT, "data", "gas", "raw", f), sheet_name=sh, header=None)
    h = [i for i in range(12) if str(x.iloc[i, 0]).startswith("Week ending")][0]
    df = x.iloc[h + 1:].copy()
    df.columns = [str(c) for c in x.iloc[h]]
    df = df[pd.to_datetime(df["Week ending"], errors="coerce").notna()]
    df.index = pd.to_datetime(df["Week ending"])
    return pd.to_numeric(df["Total Lower 48"], errors="coerce")
cur, org = sheet("ngshistory.xls", "html_report_history"), sheet("ngs_revisions.xls", "original_data")
fp = org.reindex(cur.index).combine_first(cur)
chk = W.iloc[np.r_[TEST[0] - 1, TEST]]
d = np.abs(fp.reindex(chk.week).to_numpy(float) - chk.lvl_l48.to_numpy(float))
print(f"L48 first-print levels vs raw xls over TEST weeks (+ prior week): max |diff| = {np.nanmax(d):.3g} Bcf")

MAJ = {}
dev_m = W.avail <= "2025-06-30"
for s in SER:
    dv = W.loc[dev_m, f"my_dev_{s}"].dropna()
    dv = dv[dv != 0]
    MAJ[s] = int((dv > 0).mean() > 0.5)


def cmp(tag, mine, theirs, tol=1e-9):
    global NDIFF
    ok = np.isclose(mine, theirs, atol=tol, rtol=0)
    NDIFF += not ok
    if not ok:
        print(f"  DIFF {tag}: mine={mine} harness={theirs}")


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out = np.empty_like(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i]))
        out[i] = run
    return out


def cbb(d, B=2000, b=1, seed=0):
    n = len(d)
    rng = np.random.default_rng(seed)
    xx = np.concatenate([d, d[:b]])
    st = rng.integers(0, n, size=(B, int(np.ceil(n / b))))
    idx = (st[:, :, None] + np.arange(b)[None, None]).reshape(B, -1)[:, :n]
    m = xx[idx].mean(1)
    return np.quantile(m, .025), np.quantile(m, .975)


def mcn(wa, wb):
    n01, n10 = int((wa & ~wb).sum()), int((~wa & wb).sum())
    return stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0


def evaluate(kind, keep, check):
    """keep: boolean mask over TEST weeks. Returns dict model -> metrics (L48) + pairwise."""
    ks = TEST[keep]
    files = sorted(glob.glob(os.path.join(TD, kind, "*.npz")))
    P = {}
    for f in files:
        z = np.load(f)
        pos = {int(k): j for j, k in enumerate(z["weeks"])}
        P[os.path.basename(f)[:-4]] = np.stack([z["q"][pos[int(k)]] for k in ks])    # [n, 6, 13]
    s = 0
    dv = W["my_dev_l48"].to_numpy()[ks]
    ds = W["my_ds_l48"].to_numpy()[ks]
    a5 = W["my_a5_l48"].to_numpy()[ks]
    ok = dv != 0
    act = dv > 0
    hit, ae, qs, out = {"majority": (np.full(len(ks), bool(MAJ["l48"])) == act)}, {}, {}, {}
    for mn, q in P.items():
        m = q[:, s, 6]
        pr = np.where(m > a5, True, np.where(m < a5, False, bool(MAJ["l48"])))
        hit[mn] = pr == act
        ae[mn] = np.abs(m - ds)
        e = ds[:, None] - q[:, s, 2:11]
        qs[mn] = np.maximum(LV * e, (LV - 1) * e).mean(1)
        cov = np.mean((ds >= q[:, s, 2]) & (ds <= q[:, s, 10]))
        out[mn] = {"acc": hit[mn][ok].mean(), "mae": ae[mn].mean(), "qs": qs[mn].mean(), "cov80": cov}
    out["majority"] = {"acc": hit["majority"][ok].mean()}
    # regions pooled
    for mn in list(P) + ["majority"]:
        num = den = 0
        for r in SER[1:]:
            dvr = W[f"my_dev_{r}"].to_numpy()[ks]
            okr = np.isfinite(dvr) & (dvr != 0)
            if mn == "majority":
                pr = np.full(len(ks), bool(MAJ[r]))
            else:
                mm = P[mn][:, SER.index(r), 6]
                a5r = W[f"my_a5_{r}"].to_numpy()[ks]
                pr = np.where(mm > a5r, True, np.where(mm < a5r, False, bool(MAJ[r])))
            num += (pr == (dvr > 0))[okr].sum()
            den += okr.sum()
        out[mn]["reg"] = num / den
    if check:
        L = M["kinds"][kind]["series"]["l48"]["models"]
        for mn, r in out.items():
            cmp(f"{kind} {mn} acc", r["acc"], L[mn]["acc"])
            cmp(f"{kind} {mn} regions pooled", r["reg"], M["kinds"][kind]["regions_pooled"][mn]["acc"])
            if mn != "majority":
                for a, b in (("mae", "mae_bcf"), ("qs", "qs"), ("cov80", "cov80")):
                    cmp(f"{kind} {mn} {a}", r[a], L[mn][b])
    # pairwise for the claim models, Holm over each one's non-contaminated competitors
    pw = {}
    for a in ("wm_gas", "wm_gas_direct", "alma_s0", "alma_direct_s0"):
        comp_dir = [c for c in hit if c != a and c not in CONTAM]
        comp_loss = [c for c in ae if c != a and c not in CONTAM]
        pd_ = {c: mcn(~hit[a][ok], ~hit[c][ok]) for c in comp_dir}
        hd = dict(zip(comp_dir, holm([pd_[c] for c in comp_dir])))
        res = {}
        for nm, L_ in (("mae", ae), ("qs", qs)):
            p = {c: stats.ttest_1samp(L_[a] - L_[c], 0).pvalue for c in comp_loss}
            h = dict(zip(comp_loss, holm([p[c] for c in comp_loss])))
            for c in comp_loss:
                lo, hi = cbb(L_[a] - L_[c])
                res[(nm, c)] = ((L_[a] - L_[c]).mean(), p[c], h[c], lo, hi)
        for c in comp_dir:
            res[("acc", c)] = (hit[a][ok].mean() - hit[c][ok].mean(), pd_[c], hd[c], None, None)
        pw[a] = res
        if check:
            T = M["kinds"][kind]["series"]["l48"]
            for (nm, c), v in res.items():
                key = {"acc": "pairwise", "mae": "pairwise_mae", "qs": "pairwise_qs"}[nm]
                e = T[key][f"{a}|{c}"]
                cmp(f"{kind} {a}|{c} {nm} p", v[1], e["p_mcnemar"] if nm == "acc" else e["p"])
                cmp(f"{kind} {a}|{c} {nm} holm", v[2], e["p_holm_own"])
                if nm != "acc":
                    cmp(f"{kind} {a}|{c} {nm} boot_hi", v[4], e["boot_hi"])
    return out, pw, int(ok.sum())


full, robust = {}, {}
drop = W.week.iloc[TEST].isin(pd.to_datetime(["2025-10-31", "2025-11-07"])).to_numpy()
assert drop.sum() == 2
for kind in ("thu", "fri"):
    full[kind] = evaluate(kind, np.ones(len(TEST), bool), True)
    robust[kind] = evaluate(kind, ~drop, False)
print(f"comparisons with the harness: {NDIFF} differences")

KEY = ["st_ols_wx", "chronos2_cov", "tirex", "chronos_bolt_base", "chronos2", "st_persistence", "wm_gas_direct", "wm_gas"]
for kind in ("thu", "fri"):
    out, pw, n = full[kind]
    rout, rpw, rn = robust[kind]
    print(f"\n=== {kind} (full n={n} | A3 robustness n={rn})")
    print(f"{'model':18s} acc    reg    MAE    QS     cov80 | A3: acc   MAE    QS")
    for mn in sorted(out, key=lambda m: out[m].get("mae", 99)):
        r, rr = out[mn], rout[mn]
        print(f"{mn:18s} {r['acc']:.3f}  {r['reg']:.3f}  {r.get('mae', np.nan):5.2f}  {r.get('qs', np.nan):5.2f}  "
              f"{r.get('cov80', np.nan):.3f} | {rr['acc']:.3f}  {rr.get('mae', np.nan):5.2f}  {rr.get('qs', np.nan):5.2f}")
    for a in ("wm_gas", "wm_gas_direct"):
        print(f"  -- {a} vs X: (diff, p, Holm-own, CI)   [A3 robustness: diff, p, Holm]")
        for c in KEY:
            if c == a:
                continue
            for nm in ("acc", "mae", "qs"):
                v, rv = pw[a][(nm, c)], rpw[a][(nm, c)]
                ci = "" if v[3] is None else f" CI[{v[3]:+.2f},{v[4]:+.2f}]"
                print(f"     {c:16s} {nm:3s} {v[0]:+7.3f} p={v[1]:.4f} holm={v[2]:.4f}{ci}   [{rv[0]:+7.3f} p={rv[1]:.4f} holm={rv[2]:.4f}]")
