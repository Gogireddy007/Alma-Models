"""Build runs/deploy_2026_10_11/RESULTS.md from the results JSON files (no number is typed by hand).

    .venv/bin/python tools/deploy/make_report.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.environ.get("ALMA_DEPLOY_OUT") or os.path.join(ROOT, "runs", "deploy_2026_10_11")   # daily runs: runs/deploy_daily/<date>


def j(p):
    p = os.path.join(RUN, p)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, nd) for v in x) + "]"
    return f"{x:.{nd}f}"


def verdict(s, key):
    v = s.get(f"verdict_{key}")
    return {"insufficient_data": "n too small", "not_significant": "tie", "cand_better": "**cand. better**",
            "cand_worse": "**cand. worse**", None: "(candidate)"}[v]


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def oil(L):
    r = j("oil/results.json")
    L.append(f"## Oil — WTI spot (data to {r['data_cutoff']}; candidate `wm_sv`)\n")
    L.append("QS = pinball loss on log price (lower is better); skill = 1 − QS / QS(random walk). "
             "Diff = wm_sv − model (negative = wm_sv better), 95% bootstrap CI.\n")
    for wn, w in r["windows"].items():
        for hk, x in w.items():
            if not x.get("n"):
                continue
            L.append(f"**{wn}, horizon {hk[1:]} day(s)** — {x['n']} targets {x['first_target']} … {x['last_target']}\n")
            ms = sorted(x["models"].items(), key=lambda kv: kv[1]["qs"])
            L.append(table([(m, f(s["qs"], 5), f(s["skill_vs_rw_gauss"], 3), f(s["hit"], 2), f(s["cov80"], 2),
                             f(s.get("wm_sv_minus_this_qs"), 5), f(s.get("ci95_qs"), 5), verdict(s, "qs"))
                            for m, s in ms],
                           ["model", "QS", "skill vs RW", "dir. acc", "cov80", "diff", "95% CI", "verdict"]) + "\n")
    ex = r["ex_ante"]
    L.append(f"**Ex-ante (sealed) — origin after {ex['origin_after']} (last price {ex['last_price']}):** "
             "median forecasts for the next 1 / 5 / 20 published prints\n")
    L.append(table([(m, *[f(v, 2) for v in vs], f(ex["q10_q90_h5"][m], 2)) for m, vs in ex["median_h1_h5_h20"].items()],
                   ["model", "h=1", "h=5", "h=20", "80% interval h=5"]) + "\n")


def power(L):
    r = j("power/results.json")
    L.append(f"## Power — DE-LU day-ahead, 24 hours (delivery days to {r['data_cutoff_delivery_day']}; candidate `wm_power`)\n")
    L.append("QS = pinball loss over 9 quantiles, EUR/MWh; P2 = hourly direction vs the same hour of D-1; "
             "P1 = daily base direction. Diff/CI: wm_power − model on QS, 7-day block bootstrap.\n")
    for wn, w in r["windows"].items():
        if not w.get("n"):
            continue
        L.append(f"**{wn}** — {w['n']} delivery days {w['first']} … {w['last']}\n")
        q, h = w["by_candidate"]["wm_power"]["qs"], w["by_candidate"]["wm_power"]["p2_hit"]
        ms = sorted(q, key=lambda m: q[m]["qs"])
        L.append(table([(m, f(q[m]["qs"], 2), f(q[m]["abs_err"], 2), f(h[m]["p2_hit"], 3), f(q[m]["p1_hit"], 3),
                         f(q[m]["cov80"], 2), f(q[m].get("wm_power_minus_this_qs"), 2), f(q[m].get("ci95_qs"), 2),
                         verdict(q[m], "qs")) for m in ms],
                       ["model", "QS", "MAE", "P2 hourly dir.", "P1 daily dir.", "cov80", "diff QS", "95% CI", "verdict"]) + "\n")


def gas(L):
    r = j("gas/results.json")
    L.append(f"## Gas — EIA weekly storage change, Lower 48 (last release {r['last_release']}; candidates "
             "`wm_gas`, `wm_gas_ens`)\n")
    rep = r["reproduction_vs_audit"]
    L.append(f"Reproduction check: the 52 TEST weeks re-forecast with the rebuilt data match the audit's cached "
             f"forecasts to a max of {max(v['max_abs_diff_bcf'] for v in rep.values()):.1e} Bcf ({len(rep)} series).\n")
    for wn, w in r["windows"].items():
        if not w.get("n_weeks"):
            L.append(f"**{wn}** — no EIA release in this window yet (next: {r['ex_ante']['scheduled_release']}); "
                     "nothing to score.\n")
            continue
        L.append(f"**{wn}** — {w['n_weeks']} releases {', '.join(w['releases'])}; actual Δstorage "
                 f"{', '.join(f(x, 0) for x in w['actual_bcf'])} Bcf\n")
        for kind, kk in w["kinds"].items():
            a, q = kk["by_candidate"]["wm_gas_ens"]["abs_err"], kk["by_candidate"]["wm_gas_ens"]["qs"]
            ms = sorted(a, key=lambda m: a[m]["abs_err"])
            L.append(f"*{kind} origin*\n")
            L.append(table([(m, f(a[m]["abs_err"], 2), f(q[m]["qs"], 2), f(a[m]["hit"], 2), f(a[m]["cov80"], 2),
                             " / ".join(f(v, 1) for v in kk["median_forecast_bcf"][m]), verdict(a[m], "abs_err"))
                            for m in ms],
                           ["model", "MAE (Bcf)", "QS", "dir. vs 5-yr avg", "cov80", "weekly medians", "verdict"]) + "\n")
    ex = r["ex_ante"]
    L.append(f"**Ex-ante (sealed) — week ending {ex['target_week_ending']}, EIA release {ex['scheduled_release']}; "
             f"5-year average {f(ex['avg5_l48_bcf'], 1)} Bcf**\n")
    kinds = list(ex["forecasts"])
    mods = list(ex["forecasts"][kinds[0]]["models"])
    L.append(table([(m, *[f"{f(ex['forecasts'][k]['models'][m]['median'], 1)} [{f(ex['forecasts'][k]['models'][m]['q10'], 1)}, "
                          f"{f(ex['forecasts'][k]['models'][m]['q90'], 1)}]" for k in kinds]) for m in mods],
                   ["model"] + [f"{k} origin: median [80% interval]" for k in kinds]) + "\n")


def fin(L):
    for variant in ("pinned", "corrected"):
        r = j(f"fin/results_{variant}.json")
        if r is None:
            continue
        L.append(f"## Crypto volatility — 24 perps, data variant `{variant}` (to {r['data_cutoff']}; candidate `wm_fin`)\n")
        L.append(f"Share of coin-days with a non-zero funding input: "
                 f"{', '.join(f'{k} {f(v, 2)}' for k, v in r['funding_nonzero_share_by_window'].items())}. "
                 "QS = pinball on log RV (P3); acc = P1 direction vs the 7-day reference (official labels).\n")
        for wn, w in r["windows"].items():
            if not w.get("n_days"):
                continue
            L.append(f"**{wn}** — {w['n_days']} decision days {w['first']} … {w['last']}; "
                     f"{w['n_coin_days_logrv_p3']} coin-days with log RV, {w['n_coin_days_labelled_p1']} with a P1 label\n")
            q, h = w["by_candidate"]["wm_fin"]["qs"], w["by_candidate"]["wm_fin"]["hit"]
            ms = sorted(q, key=lambda m: (q[m]["qs"] is None, q[m]["qs"] or 0))
            L.append(table([(m, f(q[m]["qs"], 4), f(h[m]["hit"], 3), f(h[m].get("hit_heldout"), 3), f(q[m]["cov80"], 2),
                             f(q[m].get("wm_fin_minus_this_qs"), 4), f(q[m].get("ci95_qs"), 4), verdict(q[m], "qs"))
                            for m in ms],
                           ["model", "QS", "P1 acc", "P1 acc held-out", "cov80", "diff QS", "95% CI", "verdict"]) + "\n")


def main():
    L = ["# Deployment results, October and November 2026 (generated by tools/deploy/make_report.py)\n",
         "Frozen checkpoints, no retraining. Windows are by target date: oct_2026 = 2026-10-01 … 10-31; "
         "nov_2026 = 2026-11-01 … the data cutoff. \"n too small\" = fewer than 10 independent scoring units, so no "
         "significance verdict is issued (one month is not enough; see docs/deploy_2026_10_11.md).\n"]
    for fn in (oil, power, gas, fin):
        fn(L)
    a = j("AUDIT.json")
    if a:
        L.append(f"## Audit: {a['n_checks'] - a['n_fail']}/{a['n_checks']} checks passed (see AUDIT.md)\n")
    open(os.path.join(RUN, "RESULTS.md"), "w").write("\n".join(L))
    print("->", os.path.join(RUN, "RESULTS.md"))


if __name__ == "__main__":
    main()
