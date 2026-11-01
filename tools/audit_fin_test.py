"""Independent re-derivation of the Alma-Fin TEST metrics (e505cbd) from the sealed raw tables and the TEST forecast
npz files (auditor; docs/audit_fin.md). Imports nothing from tools/fin. Run only after the single TEST run.

    nice -n 10 .venv/bin/python -W ignore tools/audit_fin_test.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, "runs", "fin", "test")
M = json.load(open(os.path.join(TD, "metrics.json")))["markets"]
FZ = json.load(open(os.path.join(ROOT, "runs", "fin", "FROZEN.json")))
U = json.load(open(os.path.join(ROOT, "data", "fin", "universe.json")))
LV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
CONTAM = {"timestral_5m"}
EULER = 0.5772156649015329
NDIFF = 0
NCHK = 0
OUT = {}


def cmp(tag, mine, theirs, tol=1e-9):
    global NDIFF, NCHK
    NCHK += 1
    ok = np.isclose(mine, theirs, atol=tol, rtol=0, equal_nan=True)
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


def boot_idx(n, b=7, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(B, int(np.ceil(n / b))))
    return ((st[:, :, None] + np.arange(b)) % n).reshape(B, -1)[:, :n]


def psr(r, sr_star=0.0):
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    return stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr * sr))


def dsr(r, K, V):
    s = np.sqrt(V) * ((1 - EULER) * stats.norm.ppf(1 - 1 / K) + EULER * stats.norm.ppf(1 - 1 / (K * np.e)))
    return psr(r, s)


def load(market):
    a = pd.read_parquet(os.path.join(ROOT, "data", "fin", f"{market}_daily.parquet"))
    b = pd.read_parquet(os.path.join(ROOT, "data", "fin", f"{market}_daily_SEALED.parquet"))
    D = pd.concat([a, b], ignore_index=True)
    z0 = np.load(sorted(glob.glob(os.path.join(TD, market, "*.npz")))[0])
    syms = list(z0["symbols"])
    if market == "crypto":
        days = pd.date_range(D.t.min(), D.t.max(), freq="D")
    else:
        days = pd.DatetimeIndex(sorted(D.t.unique()))
    piv = lambda c: D.pivot(index="t", columns="symbol", values=c).reindex(index=days, columns=syms)   # noqa: E731
    out = {"days": days, "syms": syms, "rv": piv("rv").to_numpy(float), "close": piv("close").to_numpy(float),
           "last_bar": piv("last_bar").fillna(False).to_numpy(bool)}
    fill = piv("fill").to_numpy(float)
    if market == "etf":
        fill = fill * piv("adj_factor").to_numpy(float)
        out["fund"] = np.zeros_like(fill)
    else:
        out["fund"] = np.nan_to_num(piv("funding_sum").to_numpy(float))
    out["fill"] = fill
    return out


def run_market(market):
    X = load(market)
    days, syms = X["days"], X["syms"]
    lrv = np.log(np.where(X["rv"] > 0, X["rv"], np.nan))
    ref = pd.DataFrame(lrv).rolling(7, min_periods=7).mean().shift(1).to_numpy()
    y = np.where(np.isfinite(lrv) & np.isfinite(ref) & (lrv != ref), (lrv > ref).astype(float), np.nan)
    TEST = np.where((days >= "2025-11-01") & (days <= "2026-10-30"))[0]
    groups = {"all": syms} if market == "etf" else {"all": syms, "heldout": U["heldout"],
                                                     "train": [s for s in syms if s not in U["heldout"]]}
    F = {}
    for f in sorted(glob.glob(os.path.join(TD, market, "*.npz"))):
        z = np.load(f)
        name = os.path.basename(f)[:-4].replace("__", ":")
        pos = {str(d)[:10]: k for k, d in enumerate(z["dates"])}
        F[name] = (z, pos)
    tdates = [str(d.date()) for d in days[TEST]]
    res = {"P1": {}, "P3": {}, "P2": {}}
    # ---------------- forecasters
    dh, qd = {}, {}
    for name, (z, pos) in F.items():
        if "p_up" not in z.files:
            continue
        rows = np.array([pos.get(d, -1) for d in tdates])
        have = rows >= 0
        pu = np.full((len(TEST), len(syms)), np.nan)
        med = np.full_like(pu, np.nan)
        Q = np.full((len(TEST), len(syms), 9), np.nan)
        pu[have], med[have], Q[have] = z["p_up"][rows[have]], z["q"][rows[have], :, 6], z["q"][rows[have], :, 2:11]
        pred = np.where(np.isfinite(pu), pu > 0.5, med > ref[TEST])
        pred = np.where(np.isfinite(pu) | np.isfinite(med), pred, False)       # A4: missing -> majority "down"
        e = lrv[TEST][:, :, None] - Q
        pin = np.maximum(LV * e, (LV - 1) * e).mean(2)
        dh[name], qd[name] = {}, {}
        for g, ss in groups.items():
            js = [syms.index(s) for s in ss]
            yy, pp = y[TEST][:, js], pred[:, js]
            ok = np.isfinite(yy)
            hit = (pp == (yy == 1)) & ok
            acc = hit.sum() / ok.sum()
            dayok = ok.sum(1) > 0
            dh[name][g] = np.where(dayok, hit.sum(1) / np.maximum(ok.sum(1), 1), np.nan)
            pg = pin[:, js]
            qd[name][g] = pg
            r = M[market]["forecasters"][name][g]
            cmp(f"{market} {name} P1 {g}", acc, r["P1"]["acc"])
            d = dh[name][g][dayok]
            cmp(f"{market} {name} P1 {g} DM vs 0.5", stats.ttest_1samp(d - 0.5, 0).pvalue, r["P1"]["p_vs_half_dm_daily"], 1e-12)
            idx = boot_idx(len(d))
            lo, hi = np.quantile(d[idx].mean(1), [.025, .975])
            res["P1"][(name, g)] = (acc, lo, hi)
            if "P3" in r and np.isfinite(pg).any():
                cmp(f"{market} {name} P3 {g}", np.nanmean(pg), r["P3"]["qs"])
                res["P3"][(name, g)] = np.nanmean(pg)
    # pairwise P1 / P3 for claim models, own-Holm over the competitor set the harness lists
    pwres = {}
    for tgt in ("P1", "P3"):
        for g in groups:
            key = f"{tgt}/{g}"
            pw = M[market]["pairwise"].get(key, {})
            for a in ("wm_fin", "wm_fin_cl", "wm_direct"):
                comps = [k.split("|")[1] for k in pw if k.startswith(a + "|")]
                ps, rows = [], {}
                for b in comps:
                    if tgt == "P1":
                        ok = np.isfinite(dh[a][g]) & np.isfinite(dh[b][g])
                        diff = dh[a][g][ok] - dh[b][g][ok]
                        p = stats.ttest_1samp(diff, 0).pvalue
                    else:
                        both = np.isfinite(qd[a][g]) & np.isfinite(qd[b][g])
                        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                            __import__("warnings").simplefilter("ignore")
                            xa = np.nanmean(np.where(both, qd[a][g], np.nan), 1)
                            xb = np.nanmean(np.where(both, qd[b][g], np.nan), 1)
                        ok = np.isfinite(xa) & np.isfinite(xb)
                        diff = xa[ok] - xb[ok]
                        p = stats.ttest_1samp(diff, 0).pvalue
                    idx = boot_idx(len(diff))
                    lo, hi = np.quantile(diff[idx].mean(1), [.025, .975])
                    rows[b] = (diff.mean(), p, lo, hi, int(ok.sum()))
                    cmp(f"{market} {key} {a}|{b} mean", diff.mean(), pw[f"{a}|{b}"]["mean_diff"])
                    cmp(f"{market} {key} {a}|{b} p", p, pw[f"{a}|{b}"]["p"], 1e-9)
                    cmp(f"{market} {key} {a}|{b} boot_lo", lo, pw[f"{a}|{b}"]["boot_lo"], 1e-9)
                hb = [b for b in comps if b not in CONTAM]
                if "p_holm_own" in pw.get(f"{a}|{hb[0]}", {}) if hb else False:
                    for b, pa in zip(hb, holm([rows[b][1] for b in hb])):
                        cmp(f"{market} {key} {a}|{b} holm", pa, pw[f"{a}|{b}"]["p_holm_own"], 1e-9)
                        rows[b] = rows[b] + (pa,)
                pwres[(key, a)] = rows
    # ---------------- agents (P2)
    last = "2026-09-29"
    P2D = np.where((days >= "2025-11-01") & (days <= last))[0]
    cost = (np.array([0.0007 if s in ("BTCUSDT", "ETHUSDT") else 0.0010 for s in syms]) if market == "crypto"
            else np.full(len(syms), 0.0002))
    per = 365 if market == "crypto" else 252
    nets = {}
    for name, (z, pos) in F.items():
        if "pos" not in z.files:
            continue
        rows = np.array([pos.get(str(days[i].date()), -1) for i in P2D])
        P = np.where(rows[:, None] >= 0, np.clip(z["pos"][np.maximum(rows, 0)], -1, 1), 0.0)
        for g, ss in groups.items():
            js = np.array([syms.index(s) for s in ss])
            for cm, lab in ((1.0, "base"), (2.0, "cost_x2")):
                prev = np.zeros(len(js))
                dead = np.zeros(len(js), bool)
                net = []
                for k, i in enumerate(P2D):
                    f0 = X["fill"][i, js]
                    f1 = X["fill"][i + 1, js] if i + 1 < len(days) else np.full(len(js), np.nan)
                    ex = ~np.isfinite(f1) | X["last_bar"][i, js]
                    if market == "crypto":
                        f1 = np.where(ex, X["close"][i, js], f1)
                    alive = np.isfinite(f0) & ~dead
                    p = np.where(alive, P[k, js], 0.0)
                    tc = np.abs(p - prev) * cost[js] * cm
                    g_ = np.where(alive, p * np.nan_to_num(f1 / f0 - 1), 0.0)
                    fu = np.where(alive, -p * X["fund"][i, js], 0.0)
                    lb = X["last_bar"][i, js]
                    tc = tc + np.abs(np.where(lb, p, 0.0)) * cost[js] * cm
                    prev = np.where(lb, 0.0, p)
                    dead |= lb
                    net.append((g_ + fu - tc).mean())
                net = np.array(net)
                if lab == "base":
                    nets[(name, g)] = net
                r = M[market]["agents"][name][g][lab]
                sr = net.mean() / net.std(ddof=1) * np.sqrt(per)
                cmp(f"{market} {name} P2 {g} {lab} sharpe", sr, r["sharpe"], 1e-6)
                if lab == "base":
                    cmp(f"{market} {name} P2 {g} psr0", psr(net), r["psr0"], 1e-6)
                    cmp(f"{market} {name} P2 {g} dsr K=985", dsr(net, FZ["K"], FZ["var_sharpe_trials"]), r["dsr"], 1e-6)
                    res["P2"][(name, g)] = (sr, psr(net), dsr(net, 985, FZ["var_sharpe_trials"]),
                                            dsr(net, 1038, FZ["var_sharpe_trials"]))
    # paired Sharpe-difference bootstrap for wm_agent
    for g in groups:
        key = f"P2/{g}"
        pw = M[market]["pairwise"].get(key, {})
        comps = [k.split("|")[1] for k in pw if k.startswith("wm_agent|")]
        rows = {}
        for b in comps:
            ra, rb = nets[("wm_agent", g)], nets[(b, g)]
            idx = boot_idx(len(ra))
            sh = lambda x: x.mean(-1) / x.std(-1, ddof=1) * np.sqrt(per)   # noqa: E731
            d = sh(ra[idx]) - sh(rb[idx])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            obs = sh(ra) - sh(rb)
            rows[b] = (obs, p, *np.quantile(d, [.025, .975]))
            cmp(f"{market} {key} wm_agent|{b} sharpe_diff", obs, pw[f"wm_agent|{b}"]["sharpe_diff"], 1e-6)
            cmp(f"{market} {key} wm_agent|{b} p", p, pw[f"wm_agent|{b}"]["p"], 0.011)
        hb = [b for b in comps if b not in CONTAM]
        if hb:
            for b, pa in zip(hb, holm([rows[b][1] for b in hb])):
                rows[b] = rows[b] + (pa,)
        pwres[(key, "wm_agent")] = rows
    OUT[market] = (res, pwres)


for mk in ("crypto", "etf"):
    run_market(mk)
print(f"checked {NCHK} numbers against metrics.json: {NDIFF} differences\n")

res, pw = OUT["crypto"]
print("P1 crypto (acc [7-day block CI])   all / heldout / train")
for name in sorted({n for n, _ in res["P1"]}, key=lambda n: -res["P1"][(n, "all")][0]):
    print(f"  {name:18s} " + "  ".join(f"{res['P1'][(name, g)][0]:.4f}" for g in ("all", "heldout", "train"))
          + f"   CI all [{res['P1'][(name, 'all')][1]:.3f},{res['P1'][(name, 'all')][2]:.3f}]"
          + (f"   QS {res['P3'][(name, 'all')]:.4f}" if (name, "all") in res["P3"] else ""))
for key in ("P1/all", "P3/all", "P1/heldout"):
    print(f"\n{key}: wm_fin vs X  (diff, DM p, CI, own-Holm)")
    for b, v in sorted(pw[(key, "wm_fin")].items(), key=lambda kv: kv[1][1]):
        h = f"{v[5]:.4f}" if len(v) > 5 else "-"
        print(f"  {b:18s} {v[0]:+.5f}  p={v[1]:.4f}  CI[{v[2]:+.5f},{v[3]:+.5f}]  holm={h}")
for key in ("P1/all", "P3/all"):
    v = pw[(key, "wm_fin")]["wm_direct"]
    print(f"\nrollout vs direct {key}: {v[0]:+.5f} p={v[1]:.4f} CI[{v[2]:+.5f},{v[3]:+.5f}]")
    v = pw[(key, "wm_fin_cl")].get("wm_fin_cl:frozen")
    if v:
        print(f"continual vs frozen {key}: {v[0]:+.5f} p={v[1]:.4f} CI[{v[2]:+.5f},{v[3]:+.5f}]")
print("\nP2 crypto (Sharpe, PSR0, DSR K=985, DSR K=1038)")
for (n, g), v in sorted(res["P2"].items()):
    if g in ("heldout", "all"):
        print(f"  {n:14s} {g:8s} SR {v[0]:+.3f} PSR {v[1]:.3f} DSR985 {v[2]:.4f} DSR1038 {v[3]:.4f}")
for g in ("P2/heldout", "P2/all"):
    print(f"\n{g}: wm_agent vs X (Sharpe diff, boot p, CI, Holm over its comparisons)")
    for b, v in pw[(g, "wm_agent")].items():
        print(f"  {b:14s} {v[0]:+.3f} p={v[1]:.3f} CI[{v[2]:+.3f},{v[3]:+.3f}] holm={v[4]:.3f}" if len(v) > 4 else b)
res, pw = OUT["etf"]
print("\nETF P1 (all 13):")
for name in sorted({n for n, _ in res["P1"]}, key=lambda n: -res["P1"][(n, "all")][0]):
    print(f"  {name:18s} {res['P1'][(name, 'all')][0]:.4f}" + (f"  QS {res['P3'][(name, 'all')]:.4f}" if (name, "all") in res["P3"] else ""))
print("ETF P2 (Sharpe, PSR, DSR985):", {n: tuple(round(x, 3) for x in v[:3]) for (n, g), v in res["P2"].items() if n in ("wm_agent", "ppo", "dqn", "buy_hold", "voltgt_tsmom")})
