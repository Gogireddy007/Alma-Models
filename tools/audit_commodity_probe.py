"""Auditor probe for docs/audit_commodity.md (crude-oil forecasting, pre-TEST audit, 2026-10-31). Read-only on the
harness: nothing is written under runs/commodity/ (run_eval.OUT is redirected to a temp dir). CPU, small.

  stats   toy checks of tools/commodity/metrics.py (Wilson, Holm, DM-HLN at h=1 vs a t-test, PT under the null) and a
          Monte-Carlo of the overlapping-horizon problem: zero-skill predictors on a random walk, daily origins,
          n=230 (TEST-like), h in {1, 5, 20}: rejection rates of the binomial test, PT, Wilson-CI coverage of 0.5,
          DM-HLN (Bartlett lag h-1) between two zero-skill predictors, and a circular block bootstrap alternative.
  report  toy check of tools/commodity/report.py beats_all() (zero-valued metric handling).
  leak    harness perturbation: post-cutoff values multiplied by random factors / truncated; VAL (cut 2025-10-31) and
          DEV (cut 2025-06-30) forecasts and every score must be bit-identical. Models: rw_gauss, auto_ets and an
          architect search checkpoint through wm_commodity.make_predict.
  arch    architect search causality: (a) predict_origins output for origin t (one origin per call, same seed) unchanged when lp[t+1:] is replaced by
          noise (sv and fine-tuned Alma checkpoints); (b) train_origins last target < cut; (c) wm_data DEV cache max date.

    nice -n 10 .venv/bin/python tools/audit_commodity_probe.py stats|report|leak|arch
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "tools", "commodity")
sys.path.insert(0, CDIR)


# --------------------------------------------------------------------------------------------- stats
def stats():
    from scipy import stats as S
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.multitest import multipletests
    import metrics as MT
    lo, hi = MT.wilson(60, 100)
    ref = proportion_confint(60, 100, method="wilson")
    print(f"wilson(60,100) = ({lo:.5f}, {hi:.5f}) statsmodels = ({ref[0]:.5f}, {ref[1]:.5f})")
    p = [0.01, 0.04, 0.03, 0.005, 0.2, np.nan]
    ours = MT.holm(p)
    sm = multipletests([x for x in p if np.isfinite(x)], method="holm")[1]
    print(f"holm ours={np.round(ours, 5).tolist()} statsmodels={np.round(sm, 5).tolist()}")
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 1, 200)
    print(f"dm_hln(h=1) stat={MT.dm_hln(d, 1)[1]:.5f}  scipy one-sample t={S.ttest_1samp(d, 0).statistic:.5f}")
    rej = 0
    for _ in range(2000):
        a = rng.integers(0, 2, 200)
        b = rng.integers(0, 2, 200)
        rej += (MT.pesaran_timmermann(a, b)[1] or 1) < 0.05
    print(f"PT one-sided size, iid null, n=200: {rej / 2000:.3f} (nominal 0.05)")

    # overlapping horizons: zero-skill predictors on a driftless random walk
    n, R = 230, 600
    print(f"\nMonte-Carlo, random walk, daily origins, n={n}, {R} reps (zero-skill predictors; nominal 5%):")
    print("h  | binom rej | PT rej | Wilson cov(0.5) | DM-HLN(Bartlett h-1) rej | DM-HLN(Bartlett 2h) rej | "
          "CBB(block=2h) cov(0.5) | DM lag h-1 rect-kernel rej")
    for h in (1, 5, 20):
        rb = rp = cw = rd = rd2 = cb = rr = 0
        for _ in range(R):
            T = n + 300
            lp = np.cumsum(rng.normal(0, 0.02, T))
            o = np.arange(250, 250 + n)
            act = (lp[o + h] > lp[o]).astype(int)
            mom = (lp[o] > lp[o - h]).astype(int)                # zero skill on a RW, persistent
            mom2 = (lp[o] > lp[o - 60]).astype(int)              # another zero-skill persistent predictor
            k = int((mom == act).sum())
            rb += MT.binom_p(k, n) < 0.05
            pp = MT.pesaran_timmermann(mom, act)[1]
            rp += np.isfinite(pp) and pp < 0.05
            l, u = MT.wilson(k, n)
            cw += l <= 0.5 <= u
            dd = (mom != act).astype(float) - (mom2 != act).astype(float)
            rd += MT.dm_hln(dd, h)[2] < 0.05
            rd2 += _dm_lag(dd, 2 * h, kernel="bartlett") < 0.05
            rr += _dm_lag(dd, h, kernel="rect") < 0.05
            hit = (mom == act).astype(float)
            lo_b, hi_b = _cbb_ci(hit, max(2 * h, 1), rng)
            cb += lo_b <= 0.5 <= hi_b
        print(f"{h:2d} | {rb / R:.3f} | {rp / R:.3f} | {cw / R:.3f} | {rd / R:.3f} | {rd2 / R:.3f} | {cb / R:.3f} | "
              f"{rr / R:.3f}")


def _dm_lag(d, L, kernel="bartlett"):
    from scipy import stats as S
    d = np.asarray(d, float)
    n = len(d)
    e = d - d.mean()
    var = e @ e / n
    for k in range(1, L):
        w = (1 - k / L) if kernel == "bartlett" else 1.0
        var += 2 * w * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return np.nan
    s = d.mean() / np.sqrt(var / n)
    return 2 * S.t.sf(abs(s), n - 1)


def _cbb_ci(x, b, rng, B=300):
    n = len(x)
    nb = int(np.ceil(n / b))
    xx = np.concatenate([x, x])
    m = []
    for _ in range(B):
        st = rng.integers(0, n, nb)
        m.append(xx[(st[:, None] + np.arange(b)[None]).reshape(-1)[:n]].mean())
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# --------------------------------------------------------------------------------------------- report
def report():
    import report as RP
    blk = {"models": {"cand": {"skill_vs_rw": -0.05}, "rw_gauss": {"skill_vs_rw": 0.0},
                      "chronos2": {"skill_vs_rw": 0.01}},
           "pairwise": {"cand|rw_gauss": {"p": 0.001}, "cand|chronos2": {"p": 0.5}}}
    S = {"wti": {"T1": {}, "T2": {}, "T4": {"1": blk}}}
    print("Toy: candidate skill -0.05 (WORSE than RW, significantly, p=0.001); rw_gauss skill 0.0 exactly.")
    print(RP.beats_all(S, "cand"))


# --------------------------------------------------------------------------------------------- leak
def leak():
    import pandas as pd
    import data as D
    import run_eval as R
    import models as M
    import wm_commodity as W
    ck = os.path.join(ROOT, "runs", "commodity", "search", "ckpt", "r1_sv_none.pt")
    M.REGISTRY["audit_arch_sv"] = lambda: W.make_predict(ck)
    orig_load = D.load
    rng = np.random.default_rng(1)

    def variant(kind, cut):
        def load(name):
            d, v = orig_load(name)
            post = d > np.datetime64(cut)
            if kind == "perturb":
                v = v.copy()
                v[post] = v[post] * rng.uniform(0.3, 3.0, post.sum())
            elif kind == "truncate":
                d, v = d[~post], v[~post]
            return d, v
        return load

    out = {}
    for split, cut, series, models in [("val", "2025-10-31", ["wti", "inv"], ["rw_gauss", "auto_ets", "audit_arch_sv"]),
                                       ("dev", "2025-06-30", ["wti"], ["rw_gauss", "audit_arch_sv"])]:
        for kind in ("orig", "perturb", "truncate"):
            D.load = orig_load if kind == "orig" else variant(kind, cut)
            R._GARCH.clear()
            tmp = tempfile.mkdtemp(prefix=f"auditcom_{split}_{kind}_")
            R.OUT = tmp
            res = {}
            for s in series:
                ms = [m for m in models if not (s == "inv" and m == "audit_arch_sv")]
                R.forecast(split, s, ms)
                q = {m: np.load(os.path.join(tmp, split, s, f"{m}.npz"))["q"] for m in ms}
                res[s] = (q, json.dumps(R.add_holm(R.score_series(split, s)), sort_keys=True, default=float))
            out[(split, kind)] = res
        D.load = orig_load
        for kind in ("perturb", "truncate"):
            for s in series:
                q0, j0 = out[(split, "orig")][s]
                q1, j1 = out[(split, kind)][s]
                same_q = all(np.array_equal(q0[m], q1[m], equal_nan=True) for m in q0)
                print(f"[{split} {s}] {kind:8s} post-{cut}: forecasts identical={same_q}  scores identical={j0 == j1}")


# --------------------------------------------------------------------------------------------- arch
def arch():
    import torch
    import pandas as pd
    import wm_core as C
    import wm_data as WD
    import wm_commodity as W
    C.set_threads(2)
    P = C.Panel("dev")
    print(f"DEV panel max date {P.dates.max().date()} (cap {WD.DEV_END.date()}), columns {P.names}")
    for f in ("r1_sv_none.pt", "r1_alma_ft_none.pt", "r1_obs_direct.pt"):
        m, ck = W.load_model(os.path.join(ROOT, "runs", "commodity", "search", "ckpt", f))
        L = ck["L"]
        bo = C.backtest_origins(P)[::12]
        q0 = [C.predict_origins(m, P, np.array([t]), ck["cols"], L=L)[0] for t in bo]   # one origin per call: same RNG
        diff = 0.0
        lp0 = P.lp.copy()
        rng = np.random.default_rng(0)
        for j, t in enumerate(bo):
            P.lp = lp0.copy()
            P.lp[t + 1:] = rng.normal(0, 5, P.lp[t + 1:].shape)
            q1 = C.predict_origins(m, P, np.array([t]), ck["cols"], L=L)
            diff = max(diff, float(np.abs(q1[0] - q0[j]).max()))
        P.lp = lp0
        to = C.train_origins(P, cut=ck["cut"], L=L, H=64 if ck["arch"] == "alma" else C.H_MAX, cols=ck["cols"])
        H = 64 if ck["arch"] == "alma" else C.H_MAX
        print(f"{f}: arch={ck['arch']} abl={ck['abl']} cut={ck['cut']} steps={ck['steps']} | future-noise max |dq| = "
              f"{diff:.3g} over {len(bo)} origins | train last target {P.dates[to.max() + H].date()} < {ck['cut']}")


# --------------------------------------------------------------------------------------------- stats2 (A3)
def stats2():
    """Size/coverage of the amendment-A3 statistics, calling run_eval.dir_stats / pair_test themselves.
    Driftless RW, daily origins n=230; zero-skill persistent predictors (momentum-h vs momentum-60); T2-like vol
    labels with two persistent random predictors; T4-like QS from two equal-accuracy Gaussian forecasters (constant vol)."""
    import run_eval as R
    rng = np.random.default_rng(7)
    n, Rp = 230, 500
    print(f"A3 statistics, n={n}, {Rp} reps, nominal 5% / 95%")
    print("target | sub binom rej | sub PT rej | sub Wilson cov | boot CI cov | pair primary rej | pair DM-wide rej")
    for tgt, h in (("T1", 1), ("T1", 5), ("T1", 20), ("T2", 40), ("T4", 5), ("T4", 20)):
        rb = rp = cw = cb = rq = rw = 0
        nrp = 0
        for _ in range(Rp):
            T = n + 400
            r = rng.normal(0, 0.02, T)          # constant vol: every predictor below has zero skill / equal loss
            lp = np.cumsum(r)
            o = np.arange(300, 300 + n)
            posn = np.arange(n)
            if tgt == "T2":
                past = np.array([r[t - 20:t].std(ddof=1) for t in o])
                fut = np.array([r[t:t + 20].std(ddof=1) for t in o])
                prev = np.array([r[t - 40:t - 20].std(ddof=1) for t in o])
                act = (fut > past).astype(int)
                z1, z2 = np.zeros(n), np.zeros(n)
                for k in range(1, n):                    # persistent random predictors, independent of the data
                    z1[k] = 0.98 * z1[k - 1] + rng.normal()
                    z2[k] = 0.98 * z2[k - 1] + rng.normal()
                p1, p2 = (z1 > 0).astype(int), (z2 > 0).astype(int)
                step, block = 40, 40
                hh = 40
            else:
                hh = h
                act = (lp[o + h] > lp[o]).astype(int)
                p1 = (lp[o] > lp[o - h]).astype(int)
                p2 = (lp[o] > lp[o - 60]).astype(int)
                step, block = h, 2 * h
            sub = posn % step == 0
            if tgt != "T4":
                st = R.dir_stats(p1, act, sub, step, block)
                rb += st["p_binom"] < 0.05
                if np.isfinite(st["p_pt"]):
                    nrp += 1
                    rp += st["p_pt"] < 0.05
                cw += st["ci_lo"] <= 0.5 <= st["ci_hi"] if tgt == "T1" else 0
                cb += st["boot_lo"] <= 0.5 <= st["boot_hi"] if tgt == "T1" else 0
                pw = R.pair_test((p1 != act).astype(float), (p2 != act).astype(float), sub, hh, block)
            else:
                y = lp[o + h] - lp[o]
                s1 = np.array([r[t - 120:t - 60].std() for t in o]) * np.sqrt(h)   # equal-length disjoint windows:
                s2 = np.array([r[t - 60:t].std() for t in o]) * np.sqrt(h)         # equal expected loss
                from scipy.stats import norm
                lv = np.arange(1, 10) / 10
                def qs(s):
                    q = norm.ppf(lv)[None] * s[:, None]
                    e = y[:, None] - q
                    return np.maximum(lv * e, (lv - 1) * e).mean(1)
                pw = R.pair_test(qs(s1), qs(s2), sub, h, 2 * h)
            rq += pw["p"] < 0.05
            rw += pw["p_dm_wide"] < 0.05
        f = lambda x, m=Rp: f"{x / m:.3f}" if m else "-"   # noqa: E731
        print(f"{tgt} h={h:2d} | {f(rb) if tgt != 'T4' else '-'} | {f(rp, nrp) if tgt != 'T4' else '-'} | "
              f"{f(cw) if tgt == 'T1' else '-'} | {f(cb) if tgt == 'T1' else '-'} | {f(rq)} | {f(rw)}")


if __name__ == "__main__":
    {"stats2": stats2, "stats": stats, "report": report, "leak": leak, "arch": arch}[sys.argv[1]]()
