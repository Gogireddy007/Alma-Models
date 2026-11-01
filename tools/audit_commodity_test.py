"""Independent re-derivation of the round-1 commodity TEST headline numbers (auditor; docs/audit_commodity.md).
Reads only data/commodity/<series>.csv and runs/commodity/test/<series>/<model>.npz; does NOT import run_eval.py or
metrics.py (every statistic is re-implemented here with numpy/scipy). Compares with runs/commodity/test/metrics.json.

    nice -n 10 .venv/bin/python tools/audit_commodity_test.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats, optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, "runs", "commodity", "test")
QS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
I9 = list(range(2, 11))
LV9 = np.array(QS[2:11])
M = json.load(open(os.path.join(TD, "metrics.json")))["series"]


def load(s):
    df = pd.read_csv(os.path.join(ROOT, "data", "commodity", f"{s}.csv"), parse_dates=["date"])
    return df.date.to_numpy(), df.value.to_numpy(float)


def npz(s, m):
    z = np.load(os.path.join(TD, s, f"{m}.npz"))
    return z["q"], z["origins"]


def train_major(d, v, h):
    vt = v[d <= np.datetime64("2025-06-30")]
    ch = vt[h:] - vt[:-h]
    ch = ch[ch != 0]
    return int((ch > 0).mean() > 0.5)


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - hw, c + hw


def mcnemar(wa, wb):
    n01, n10 = int((wa & ~wb).sum()), int((~wa & wb).sum())
    return n01, n10, (stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)


def dm0(d):
    """DM with HLN at horizon 1 = one-sample t-test."""
    return stats.ttest_1samp(d, 0).pvalue


def show(tag, mine, theirs):
    ok = "OK " if (np.isclose(mine, theirs, atol=1e-9, equal_nan=True)) else "DIFF"
    print(f"  {ok} {tag:48s} mine={mine:.6f} harness={theirs:.6f}")


def t1(s, models, hs):
    d, v = load(s)
    _, base = npz(s, "rw_gauss")                    # every-origin model: defines window positions
    for h in hs:
        blk = M[s]["T1"][str(h)]
        ok = base - 1 + h < len(v)
        oi, posn = base[ok], np.arange(len(base))[ok]
        last, fut = v[oi - 1], v[oi - 1 + h]
        nz = fut != last
        oi, posn, last, fut = oi[nz], posn[nz], last[nz], fut[nz]
        act = fut > last
        sub = posn % h == 0
        mj = train_major(d, v, h)
        print(f"[{s} T1 h={h}] n={len(oi)} non-overlap n={sub.sum()} base rate up={act.mean():.3f}")
        preds = {}
        for m in models:
            q, qo = npz(s, m)
            pos = {o: j for j, o in enumerate(qo)}
            sel = np.array([o in pos for o in oi])
            med = np.array([q[pos[o], 6, h - 1] for o in oi[sel]])
            p = np.where(med > last[sel], 1, np.where(med < last[sel], 0, mj)).astype(bool)
            hit = p == act[sel]
            preds[m] = (sel, p)
            r = blk["models"][m]
            show(f"{m} acc", hit.mean(), r["acc"])
            show(f"{m} acc non-overlap", hit[sub[sel]].mean(), r["acc_nonoverlap"])
            ks, ns = int(hit[sub[sel]].sum()), int(sub[sel].sum())
            show(f"{m} non-overlap binomial p", stats.binomtest(ks, ns).pvalue, r["p_binom"])
            show(f"{m} non-overlap Wilson lo", wilson(ks, ns)[0], r["ci_lo"])
        yield h, oi, act, sub, preds


def pair_dir(s, h, act, sub, preds, a, b):
    sa, pa = preds[a]
    sb, pb = preds[b]
    full_a = np.full(len(act), -1)
    full_a[sa] = pa
    full_b = np.full(len(act), -1)
    full_b[sb] = pb
    s_ = (full_a >= 0) & (full_b >= 0)
    wa, wb = full_a[s_] != act[s_], full_b[s_] != act[s_]
    pw = M[s]["T1"][str(h)]["pairwise"][f"{a}|{b}"]
    if h == 1:
        p = dm0(wa.astype(float) - wb.astype(float))
        show(f"{a} vs {b} DM-HLN p (h=1)", p, pw["p"])
    else:
        n01, n10, p = mcnemar(wa[sub[s_]], wb[sub[s_]])
        show(f"{a} vs {b} McNemar p (non-overlap; {n01}/{n10})", p, pw["p"])
    print(f"       acc diff {(wb.mean() - wa.mean()) * 100:+.1f}pp; harness Holm(family) "
          f"{pw.get('p_holm_family', float('nan'))}")


def garch_pred(d, v, oi, past):
    with np.errstate(invalid="ignore", divide="ignore"):
        lv = np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan)
    r = np.diff(lv)
    rd = d[1:]
    m = (rd >= np.datetime64("2000-01-01")) & (rd <= np.datetime64("2025-06-30")) & np.isfinite(r)
    x = r[m]
    v0 = x.var()

    def nll(th):
        w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        s2 = np.empty_like(x)
        s2[0] = v0
        for t in range(1, len(x)):
            s2[t] = w + a * x[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + x * x / s2)
    w, a, b = optimize.minimize(nll, [v0 * 0.05, 0.08, 0.9], method="Powell").x   # different optimiser on purpose
    rr = np.where(np.isfinite(r), r, 0.0)
    s2 = np.empty(len(rr) + 1)
    s2[0] = v0
    for t in range(len(rr)):
        s2[t + 1] = w + a * rr[t] ** 2 + b * s2[t]
    lr = w / (1 - a - b)
    ph = ((a + b) ** np.arange(20)).mean()
    return np.array([lr + (s2[i - 1] - lr) * ph > past[k] ** 2 for k, i in enumerate(oi)]), (w, a, b)


def t2(s, models):
    d, v = load(s)
    _, base = npz(s, "rw_gauss")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(v))
    ok = base + 19 < len(v)
    oi, posn = base[ok], np.arange(len(base))[ok]
    past = np.array([np.std(r[i - 21:i - 1], ddof=1) for i in oi])
    fut = np.array([np.std(r[i - 1:i + 19], ddof=1) for i in oi])
    act = fut > past
    sub = posn % 40 == 0
    blk = M[s]["T2"]["20"]
    print(f"[{s} T2] n={len(oi)} non-overlap n={sub.sum()} base rate higher={act.mean():.3f}")
    preds = {}
    for m in models:
        if m == "garch":
            p, th = garch_pred(d, v, oi, past)
            sel = np.ones(len(oi), bool)
        else:
            q, qo = npz(s, m)
            pos = {o: j for j, o in enumerate(qo)}
            sel = np.array([o in pos for o in oi])
            k = np.arange(1, 21)
            sm = np.array([np.mean((np.log(np.maximum(q[pos[o], 10, :20], .01)) - np.log(np.maximum(q[pos[o], 2, :20], .01)))
                                   / (2 * 1.2816 * np.sqrt(k))) for o in oi[sel]])
            p = sm > past[sel]
        hit = p == act[sel]
        preds[m] = (sel, p)
        show(f"{m} T2 acc", hit.mean(), blk["models"][m]["acc"])
        show(f"{m} T2 acc non-overlap (n={sub[sel].sum()})", hit[sub[sel]].mean(), blk["models"][m]["acc_nonoverlap"])
    return act, sub, preds


def t4(s, models, hs):
    d, v = load(s)
    for h in hs:
        blk = M[s]["T4"][str(h)]
        qr, base = npz(s, "rw_gauss")
        ok = base - 1 + h < len(v)
        oi, posn = base[ok], np.arange(len(base))[ok]
        y = np.log(v[oi - 1 + h])
        sub = posn % h == 0
        L = {}
        for m in ["rw_gauss"] + models:
            q, qo = npz(s, m)
            pos = {o: j for j, o in enumerate(qo)}
            lq = np.log(np.maximum(np.stack([q[pos[o], I9, h - 1] for o in oi]), 0.01))
            e = y[:, None] - lq
            L[m] = np.maximum(LV9 * e, (LV9 - 1) * e).mean(1)
        print(f"[{s} T4 h={h}] n={len(oi)} non-overlap n={sub.sum()}")
        for m in models:
            sk = 1 - L[m].mean() / L["rw_gauss"].mean()
            r = blk["models"][m]
            show(f"{m} skill vs rw", sk, r["skill_vs_rw"])
            dd = L[m] - L["rw_gauss"]
            p = dm0(dd) if h == 1 else dm0(dd[sub])
            show(f"{m} p vs rw ({'DM h=1' if h == 1 else 'DM lag0 non-overlap'})", p, r["p_vs_rw"])
        yield h, L, sub


if __name__ == "__main__":
    W = ["wm_sv", "wm_sv_direct", "alma_s0", "alma_direct_s0", "chronos2", "tirex", "rw_gauss"]
    for h, oi, act, sub, preds in t1("wti", W, (1, 5, 20)):
        for a, b in (("wm_sv", "wm_sv_direct"), ("alma_s0", "alma_direct_s0"), ("wm_sv", "rw_gauss"),
                     ("wm_sv", "chronos2"), ("alma_s0", "rw_gauss"), ("alma_s0", "chronos2")):
            pair_dir("wti", h, act, sub, preds, a, b)
    act, sub, preds = t2("wti", ["wm_sv", "wm_sv_direct", "chronos2", "alma_s0", "alma_direct_s0", "garch"])
    for a, b in (("wm_sv", "wm_sv_direct"), ("alma_s0", "alma_direct_s0"), ("wm_sv", "garch")):
        sa, pa = preds[a]
        sb, pb = preds[b]
        wa, wb = pa != act, pb != act
        n01, n10, p = mcnemar(wa[sub], wb[sub])
        show(f"T2 {a} vs {b} McNemar p ({n01}/{n10})", p, M["wti"]["T2"]["20"]["pairwise"][f"{a}|{b}"]["p"])
    for h, L, sub in t4("wti", ["wm_sv", "wm_sv_direct", "alma_s0", "alma_direct_s0", "chronos2", "tirex"], (1, 5, 20)):
        for a, b in (("wm_sv", "wm_sv_direct"), ("alma_s0", "alma_direct_s0")):
            dd = L[a] - L[b]
            p = dm0(dd) if h == 1 else dm0(dd[sub])
            show(f"T4 h={h} {a} vs {b} p (mean diff {dd.mean():+.2e})", p,
                 M["wti"]["T4"][str(h)]["pairwise"][f"{a}|{b}"]["p"])
