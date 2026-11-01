"""Round-2 DEV scoring and the pre-registered selection rule (docs/commodity_r2_prereg.md section 5).

    nice -n 10 .venv/bin/python tools/commodity/r2/score_r2.py            # writes runs/commodity/r2/dev_scores.json/.md

Candidates are declared in CANDIDATES (recipe -> member runs of the world model and of its direct twin; several
members = seed ensemble, quantile average). Cells: PRIMARY vol direction pooled over POOL5 at h = 5, 10;
SECONDARY CRPS on DHOILNYH and DJFUELUSGULF at h = 5, 10. K = number of (candidate, cell) rows scored.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
from scipy import stats       # noqa: E402

import core_r2 as R           # noqa: E402
import data_r2 as D           # noqa: E402
import metrics as MT          # noqa: E402

SEEDS = (0, 1, 2)
CANDIDATES = {
    "sv_pool": ([f"svx_plain_none_s{s}" for s in SEEDS], [f"svx_plain_direct_s{s}" for s in SEEDS]),
    "sv_cal": ([f"svx_cal_none_s{s}" for s in SEEDS], [f"svx_cal_direct_s{s}" for s in SEEDS]),
    "sv_cal_exo": ([f"svx_calexo_none_s{s}" for s in SEEDS], [f"svx_calexo_direct_s{s}" for s in SEEDS]),
    "rs_pool": (["rs_none_s0"], ["rs_direct_s0"]),
    "obs_pool": (["obs_none_s0"], ["obs_direct_s0"]),
    "alma_zs": (["alma_s0"], ["alma_direct_s0"]),
}
VOL_QBASE = ("rw_gauss", "ewma", "garch", "gjr_t")
VOL_CBASE = ("har", "loghar", "har_pooled", "loghar_pooled", "meanrev", "majority")
CRPS_BASE = ("rw_gauss", "ewma", "garch", "gjr_t")
CELLS = [("vol", h, "POOL5") for h in R.HS] + [("crps", h, s) for s in D.SECONDARY for h in R.HS]
MIN_INDEP = 60


def load_pred(name, split="dev"):
    z = np.load(os.path.join(R.OUT, "pred", split, f"{name}.npz"))
    return {s: (z[f"{s}_t"], z[f"{s}_q"]) for s in D.POOL5 if f"{s}_t" in z.files}, z["odates"]


def ensemble(names, split="dev"):
    preds = [load_pred(n, split) for n in names]
    od = preds[0][1]
    for p in preds[1:]:
        assert np.array_equal(p[1], od)
    out = {}
    for s in preds[0][0]:
        t = preds[0][0][s][0]
        for p in preds[1:]:
            assert np.array_equal(p[0][s][0], t)
        out[s] = (t, np.mean([p[0][s][1] for p in preds], 0))
    return out, od


def vol_losses(S, fc, h, idx=None):
    """[n_dates, 5] 0/1 losses and ok mask for a quantile forecast dict (or class dict when fc[s] = (t, cls_h))."""
    L, OK = [], []
    for s in D.POOL5:
        t, q = fc[s]
        lp = S[s].lp
        inr = t + h < len(lp)
        tt = np.where(inr, t, 0 * t + 30)
        y, ok = R.vol_actual(lp, tt, h)
        pred = R.vol_class_from_q(q, lp, tt, h) if q.ndim == 3 else q
        L.append((pred != y).astype(float))
        OK.append(ok & inr)
    return np.stack(L, 1), np.stack(OK, 1)


def baseline_fc(S, P, od):
    """{name: {sid: (t, q or class)}} for every baseline, and per h the class baselines."""
    qb, cb = {}, {h: {} for h in R.HS}
    for s in D.POOL5:
        t = R.t_index(S[s], od)
        q, v = R.baselines(S[s], t, P)
        for n, x in q.items():
            qb.setdefault(n, {})[s] = (t, x)
        for h in R.HS:
            for n, x in v[h].items():
                cb[h].setdefault(n, {})[s] = (t, x)
    return qb, cb


def crps_arr(S, fc, s, h):
    t, q = fc[s]
    lp = S[s].lp
    ok = t + h < len(lp)
    tt = np.where(ok, t, 30)
    return R.qs(q, lp, tt, h), ok


def score(split="dev", candidates=None, out_name="dev_scores"):
    S = R.load(split)
    P = R.fit_params(R.load("dev"))
    od = R.origin_dates(S, split)
    qb, cb = baseline_fc(S, P, od)
    rows, base_tab = [], {}
    # baselines per cell
    for kind, h, ser in CELLS:
        tab = {}
        if kind == "vol":
            for n in VOL_QBASE:
                Lb, OK = vol_losses(S, qb[n], h)
                tab[n] = (1 - Lb[OK].mean(), Lb, OK)
            for n in VOL_CBASE:
                Lb, OK = vol_losses(S, cb[h][n], h)
                tab[n] = (1 - Lb[OK].mean(), Lb, OK)
            best = max(tab, key=lambda n: tab[n][0])
        else:
            for n in CRPS_BASE:
                a, ok = crps_arr(S, qb[n], ser, h)
                tab[n] = (a[ok].mean(), a, ok)
            best = min(tab, key=lambda n: tab[n][0])
        base_tab[(kind, h, ser)] = (best, tab)
    for cand, (wm, tw) in (candidates or CANDIDATES).items():
        try:
            fw, odw = ensemble(wm, split)
            ft, odt = ensemble(tw, split)
        except FileNotFoundError as e:
            print(f"[score] {cand}: missing {e}")
            continue
        assert np.array_equal(odw.astype("datetime64[D]"), od.astype("datetime64[D]")), cand
        for kind, h, ser in CELLS:
            best, tab = base_tab[(kind, h, ser)]
            r = {"cand": cand, "target": kind, "h": h, "series": ser, "best_baseline": best, "members": wm,
                 "twin": tw}
            if kind == "vol":
                Lw, OKw = vol_losses(S, fw, h)
                Lt, OKt = vol_losses(S, ft, h)
                _, Lb, OKb = tab[best]
                ok = OKw & OKt & OKb
                tb = R.test_pooled_vol(Lw, Lb, ok, h)
                tt_ = R.test_pooled_vol(Lw, Lt, ok, h)
                r.update(metric=tb["acc_a"], base_metric=tb["acc_b"], twin_metric=tt_["acc_b"],
                         margin_base=tb["diff"], margin_twin=tt_["diff"], p_base=tb["p1"], p_twin=tt_["p1"],
                         n_indep=tb["n_indep"], n_pairs=tb["n_pairs"], ci_base=[tb["ci_lo"], tb["ci_hi"]],
                         ci_twin=[tt_["ci_lo"], tt_["ci_hi"]],
                         all_baselines={n: float(v[0]) for n, v in tab.items()})
            else:
                aw, okw = crps_arr(S, fw, ser, h)
                at, okt = crps_arr(S, ft, ser, h)
                _, ab, okb = tab[best]
                ok = okw & okt & okb
                tb = R.test_crps(aw, ab, ok, h)
                tt_ = R.test_crps(aw, at, ok, h)
                arw, _ = crps_arr(S, qb["rw_gauss"], ser, h)
                t, q = fw[ser]
                tt2 = np.where(ok, t, 30)
                c80, c95 = R.coverage(q[ok], S[ser].lp, tt2[ok], h)
                r.update(metric=float(aw[ok].mean()), base_metric=float(ab[ok].mean()), twin_metric=float(at[ok].mean()),
                         margin_base=tb["skill"], margin_twin=tt_["skill"], p_base=tb["p1"], p_twin=tt_["p1"],
                         n_indep=tb["n_indep"], n_pairs=tb["n"], ci_base=[tb["ci_lo"], tb["ci_hi"]],
                         ci_twin=[tt_["ci_lo"], tt_["ci_hi"]], skill_vs_rw=float(1 - aw[ok].mean() / arw[ok].mean()),
                         cov80=c80, cov95=c95, all_baselines={n: float(v[0]) for n, v in tab.items()})
            r["p_cand"] = max(r["p_base"], r["p_twin"])
            r["z_min"] = min(R.p_to_z(r["p_base"]), R.p_to_z(r["p_twin"]))
            r["eligible"] = bool(r["n_indep"] >= MIN_INDEP)
            r["positive"] = bool(r["margin_base"] > 0 and r["margin_twin"] > 0)
            rows.append(r)
    K = len(rows)
    hp = MT.holm([r["p_cand"] for r in rows])
    zK = stats.norm.isf(0.05 / max(K, 1))
    for r, p in zip(rows, hp):
        r["p_holm"] = float(p)
        z = R.p_to_z(r["p_base"])
        se = abs(r["margin_base"]) / z if abs(z) > 1e-6 else float("nan")
        r["margin_base_discounted"] = float(r["margin_base"] - zK * se) if np.isfinite(se) else float("nan")
    rank = sorted(rows, key=lambda r: (-r["z_min"], -r["margin_base"]))
    choice = select(rows)
    res = {"split": split, "K": K, "n_origin_dates": int(len(od)), "rows": rank, "choice": choice,
           "baselines": {f"{k[0]}_h{k[1]}_{k[2]}": {"best": v[0], "metrics": {n: float(x[0]) for n, x in v[1].items()}}
                         for k, v in base_tab.items()}}
    json.dump(res, open(os.path.join(R.OUT, f"{out_name}.json"), "w"), indent=1, default=str)
    write_md(res, os.path.join(R.OUT, f"{out_name}.md"))
    return res


def select(rows):
    """Pre-registered rule (prereg section 5)."""
    el = [r for r in rows if r["eligible"]]
    prim = [r for r in el if r["target"] == "vol"]
    sec = [r for r in el if r["target"] == "crps"]
    key = lambda r: (r["z_min"], r["margin_base"])          # noqa: E731
    for pool, why in ((prim, "primary, positive margins over strongest baseline and twin"),
                      (sec, "secondary, positive margins over strongest baseline and twin")):
        pos = [r for r in pool if r["positive"]]
        if pos:
            c = max(pos, key=key)
            return {"cand": c["cand"], "target": c["target"], "h": c["h"], "series": c["series"], "rule": why,
                    "p_holm": c["p_holm"], "z_min": c["z_min"]}
    c = max(prim, key=key)
    return {"cand": c["cand"], "target": c["target"], "h": c["h"], "series": c["series"],
            "rule": "fallback: no eligible row beat both comparators; best primary row by z_min (NEGATIVE result)",
            "p_holm": c["p_holm"], "z_min": c["z_min"]}


def write_md(res, path):
    L = [f"# Round-2 {res['split']} scores (K = {res['K']}, origin dates = {res['n_origin_dates']})", "",
         f"Choice: `{json.dumps(res['choice'])}`", "",
         "| rank | candidate | target | h | series | metric | best baseline (metric) | twin | margin vs base [95% CI] |"
         " margin vs twin | p_base | p_twin | z_min | Holm p | n_indep |", "|" + "---|" * 15]
    for i, r in enumerate(res["rows"]):
        L.append(f"| {i + 1} | {r['cand']} | {r['target']} | {r['h']} | {r['series']} | {r['metric']:.4f} | "
                 f"{r['best_baseline']} ({r['base_metric']:.4f}) | {r['twin_metric']:.4f} | {r['margin_base']:+.4f} "
                 f"[{r['ci_base'][0]:+.4f}, {r['ci_base'][1]:+.4f}] | {r['margin_twin']:+.4f} | {r['p_base']:.3f} | "
                 f"{r['p_twin']:.3f} | {r['z_min']:+.2f} | {r['p_holm']:.3f} | {r['n_indep']} |")
    L += ["", "## Baselines per cell", ""]
    for k, v in res["baselines"].items():
        L.append(f"- {k}: best `{v['best']}`; " + ", ".join(f"{n} {x:.4f}" for n, x in v["metrics"].items()))
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    r = score()
    print(open(os.path.join(R.OUT, "dev_scores.md")).read()[:6000])
