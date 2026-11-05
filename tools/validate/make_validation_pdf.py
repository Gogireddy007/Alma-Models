"""Build the overnight validation report (PDF) from runs/validation/** — every number is read from those files.

    .venv/bin/python tools/validate/make_validation_pdf.py   # -> reports/Alma_Models_Validation_Report_2026-11-05.pdf
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "deploy"))
import make_pdf_report as R  # noqa: E402  (styles, fonts, table helpers)
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate, Spacer  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

VAL = os.path.join(ROOT, "runs", "validation")
PT = os.path.join(VAL, "posttrain")
FIG = os.path.join(VAL, "figures")
OUT = os.path.join(ROOT, "reports", "Alma_Models_Validation_Report_2026-11-05.pdf")
os.makedirs(FIG, exist_ok=True)
P, tbl, box, B, SM, H1, H2 = R.P, R.tbl, R.box, R.B, R.SM, R.H1, R.H2
f = R.f


def J(p):
    p = os.path.join(VAL, p)
    return json.load(open(p)) if os.path.exists(p) else None


MUT, TAMP, LEAK, V5 = J("mutation/RESULTS.json"), J("tamper/RESULTS.json"), J("leak/RESULTS.json"), J("v5/RESULTS.json")
POST = {"oil": J("posttrain/oil/results.json"), "power": J("posttrain/power/results.json"),
        "gas": J("posttrain/gas/results.json"), "crypto": J("posttrain/fin/results_corrected.json")}
CAND = {"oil": "wm_sv", "power": "wm_power", "gas": "wm_gas_ens", "crypto": "wm_fin"}
TWIN = {"oil": ("wm_sv", "wm_sv_direct"), "power": ("wm_power", "wm_power_direct"), "gas": ("wm_gas", "wm_gas_direct"),
        "crypto": ("wm_fin", "wm_direct")}
ALMA = R.ALMA
MONTHS = [k for k in (POST["power"] or {}).get("windows", {}) if k.startswith("m_")]


def qs_block(proj, win, cand=None):
    """{model: {qs, diff, ci, verdict, n_eff}} for a window, from the cand's comparison block (QS)."""
    r = POST[proj]
    if r is None:
        return None
    w = r["windows"].get(win, {})
    cand = cand or CAND[proj]
    if proj == "oil":
        x = w.get("h1", {})
        return x.get("models") if x.get("n") else None
    if proj == "power":
        return w["by_candidate"]["wm_power"]["qs"] if w.get("n") else None
    if proj == "gas":
        return w["kinds"]["thu"]["by_candidate"]["wm_gas_ens"]["qs"] if w.get("n_weeks") else None
    return w["by_candidate"]["wm_fin"]["qs"] if w.get("n_days") else None


def n_of(proj, win):
    w = POST[proj]["windows"].get(win, {})
    return {"oil": w.get("h1", {}).get("n", 0), "power": w.get("n", 0), "gas": w.get("n_weeks", 0),
            "crypto": w.get("n_days", 0)}[proj]


def cand_key(proj):
    return {"oil": "wm_sv", "power": "wm_power", "gas": "wm_gas_ens", "crypto": "wm_fin"}[proj]


# =========================================================================== figures
def fig_monthly_rank():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), facecolor=R.SURF, sharex=True)
    for ax, proj in zip(axes.ravel(), ("gas", "power", "crypto", "oil")):
        xs, ranks, nmods = [], [], []
        for k, m in enumerate(MONTHS):
            blk = qs_block(proj, m)
            if not blk:
                continue
            sc = {a: b["qs"] for a, b in blk.items() if b.get("qs") is not None}
            order = sorted(sc, key=sc.get)
            c = CAND[proj]
            if c in order:
                xs.append(k)
                ranks.append(order.index(c) + 1)
                nmods.append(len(order))
        ax.plot(xs, ranks, color=R.BLUE, lw=2, marker="o", ms=5, markeredgecolor=R.SURF, markeredgewidth=1.2)
        if nmods:
            ax.set_ylim(max(nmods) + 0.5, 0.5)
        ax.axhline(1, color=R.GRID, lw=0.8)
        R.style_ax(ax)
        ax.yaxis.grid(True, color=R.GRID, lw=0.6)
        ax.xaxis.grid(False)
        ax.set_title(f"{proj}: rank of {CAND[proj]} (1 = best of {max(nmods) if nmods else '?'})", fontsize=8,
                     loc="left", color=R.INK, fontweight="bold")
        ax.set_xticks(range(0, len(MONTHS), 2))
        ax.set_xticklabels([MONTHS[i][2:].replace("_", "-") for i in range(0, len(MONTHS), 2)], fontsize=6.5, rotation=0)
    fig.tight_layout()
    p = os.path.join(FIG, "v4_monthly_rank.png")
    fig.savefig(p, dpi=200, facecolor=R.SURF)
    plt.close(fig)
    return p


def fig_forest():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.6), facecolor=R.SURF)
    for ax, proj in zip(axes, ("gas", "power", "crypto", "oil")):
        blk = qs_block(proj, "ALL_POST_TRAINING") or {}
        c = CAND[proj]
        key = f"{c}_minus_this_qs"
        rows = [(m, b[key], b.get(f"ci95_qs") or [np.nan, np.nan], b.get("verdict_qs")) for m, b in blk.items()
                if m != c and b.get(key) is not None]
        rows.sort(key=lambda r: r[1])
        scale = abs(blk.get(c, {}).get("qs") or 1.0)
        for y, (m, d, ci, v) in enumerate(rows):
            col = R.BLUE if v == "cand_better" else ("#a0361b" if v == "cand_worse" else R.NEUTRAL)
            ax.plot([100 * ci[0] / scale, 100 * ci[1] / scale], [y, y], color=col, lw=2)
            ax.plot([100 * d / scale], [y], "o", color=col, ms=4)
        ax.axvline(0, color=R.INK2, lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=6)
        R.style_ax(ax)
        ax.set_title(f"{proj} ({c})", fontsize=7.5, loc="left", color=R.INK, fontweight="bold")
        ax.set_xlabel("QS difference, % of cand. QS", fontsize=6.5, color=R.INK2)
        ax.tick_params(labelsize=6)
    fig.text(0.01, 0.005, "Left of 0 = the Alma model scores better. Blue = significantly better, red = significantly "
             "worse, grey = not significant (95% block-bootstrap CI). Jul 2025 - Nov 2026.", fontsize=6.5, color=R.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(FIG, "v4_forest.png")
    fig.savefig(p, dpi=200, facecolor=R.SURF)
    plt.close(fig)
    return p


def fig_reliability():
    if not V5:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4), facecolor=R.SURF)
    pal = [R.BLUE, R.ORANGE, R.AQUA]
    for ax, proj in zip(axes, ("gas", "power", "crypto", "oil")):
        ks = [k for k in V5["calibration"] if k.startswith(proj + "/")][:3]
        ax.plot([0, 1], [0, 1], color=R.GRID, lw=1)
        for col, k in zip(pal, ks):
            c = V5["calibration"][k]
            ax.plot(c["levels"], c["share_below"], color=col, lw=1.8, marker="o", ms=3, label=k.split("/")[1])
        R.style_ax(ax)
        ax.yaxis.grid(True, color=R.GRID, lw=0.6)
        ax.set_title(proj, fontsize=7.5, loc="left", color=R.INK, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.text(0.01, 0.01, "x = quantile level, y = share of outcomes below that quantile; on the diagonal = calibrated.",
             fontsize=6.5, color=R.MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = os.path.join(FIG, "v5_reliability.png")
    fig.savefig(p, dpi=200, facecolor=R.SURF)
    plt.close(fig)
    return p


# =========================================================================== document
def footer(c, d):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(colors.HexColor(R.MUTED))
    c.drawString(17 * mm, 10 * mm, "Alma Models · Overnight validation report, 5 Nov 2026 · every number is read from "
                 "runs/validation/**")
    c.drawRightString(193 * mm, 10 * mm, f"{d.page}")
    c.restoreState()


def count_ok(rows, key="ok"):
    return sum(bool(r[key]) for r in rows), len(rows)


def pooled_table(proj):
    blk = qs_block(proj, "ALL_POST_TRAINING")
    if not blk:
        return [P(f"{proj}: no pooled result.", SM)]
    c = CAND[proj]
    key = f"{c}_minus_this_qs"
    rows = [["Model", "QS", "diff vs " + c, "95% CI", "n_eff", "verdict"]]
    order = sorted(blk, key=lambda m: blk[m]["qs"] if blk[m].get("qs") is not None else 9e9)
    nd = 5 if proj == "oil" else (4 if proj == "crypto" else 2)
    vmap = {"cand_better": f"{c} better", "cand_worse": f"{c} worse", "not_significant": "tie",
            "insufficient_data": "n too small", None: "(candidate)"}
    for m in order:
        b = blk[m]
        rows.append([m, f(b.get("qs"), nd), f(b.get(key), nd), f(b.get("ci95_qs"), nd) if b.get("ci95_qs") else "–",
                     f(b.get("n_eff"), 0) if b.get("n_eff") else "–", vmap.get(b.get("verdict_qs"), "–")])
    t = tbl(rows, [36 * mm, 22 * mm, 26 * mm, 46 * mm, 14 * mm, 32 * mm],
            hl_rows=[i + 1 for i, m in enumerate(order) if m in ALMA])
    return [t]


def build():
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    s = [P("Alma Models — Overnight validation report", R.TITLE),
         P(f"Deeper tests of the frozen world models and of the integrity harness · 5 November 2026 · branch "
           f"claude/gracious-franklin-f8ml84 @ {git}", SM), Spacer(1, 6)]
    # ---------------- headline numbers
    mut = count_ok(MUT) if MUT else (0, 0)
    tam = count_ok(TAMP) if TAMP else (0, 0)
    lk = count_ok(LEAK) if LEAK else (0, 0)
    n_leak_fc = sum(r["n_forecasts"] for r in (LEAK or []) if not r["positive_control"])
    n_ctrl = sum(1 for r in (LEAK or []) if r["positive_control"])
    props = V5["properties"] if V5 else []
    pr = count_ok(props)
    verdicts = {}
    for proj in CAND:
        blk = qs_block(proj, "ALL_POST_TRAINING") or {}
        vs = [b.get("verdict_qs") for m, b in blk.items() if m != CAND[proj]]
        verdicts[proj] = (sum(v == "cand_better" for v in vs), sum(v == "not_significant" for v in vs),
                          sum(v == "cand_worse" for v in vs), len(vs))
    s.append(box([P("<b>Bottom line</b>"), P(
        f"<b>Integrity (the harness):</b> {mut[0]}/{mut[1]} planted fakes behaved as expected (caught, and the clean "
        f"control passed); {tam[0]}/{tam[1]} tamper and no-training tests refused as expected; the exhaustive leak test "
        f"checked {n_leak_fc:,} forecasts with corrupted futures and {lk[0]}/{lk[1]} rows behaved as expected, including "
        f"{n_ctrl} positive controls in which a deliberately leaky context WAS detected. "
        f"<b>Physics/economics properties:</b> {pr[0]}/{pr[1]} held on real inputs. "
        "<b>Performance over the 16 months after training (Jul 2025 – Nov 2026):</b> pooled comparisons of each frozen "
        "Alma model against every competitor — significantly better / tie / significantly worse: " +
        "; ".join(f"{p} {v[0]}/{v[1]}/{v[2]} of {v[3]}" for p, v in verdicts.items()) + ". "
        "Most of this period (VAL, TEST) had been scored before, so it is a re-look, not a fresh test; the fresh, "
        "sealed forward evidence is still accumulating (Part D).")]))
    s.append(Spacer(1, 6))
    s.append(P("How to read this report", R.H2))
    s.append(tbl([["Part", "Question", "Tests"],
                  ["A", "Can the audit be fooled? Would it notice a leak, a fake number, a tampered model?",
                   "V1 mutation tests, V2 tamper + determinism, V3 exhaustive leak tests with positive controls, "
                   "new re-run check (R)"],
                  ["B", "Is the performance real and stable after training?", "V4: every model, every month, Jul 2025 – "
                   "Nov 2026; pooled block-bootstrap verdicts; world-model vs direct twin over 16 months"],
                  ["C", "Are the forecasts sensible and honest about uncertainty?", "V5: calibration, physics/economics "
                   "property tests, placebo (shuffled outcomes)"],
                  ["D", "What is still open?", "sealed ex-ante forecasts, forward test, caveats"]],
                 [12 * R.mm, 70 * R.mm, 94 * R.mm]))
    s.append(PageBreak())
    # ================= Part A
    s.append(P("Part A — Can the integrity harness be fooled?", H1))
    s.append(P("A1. A gap found and closed: the re-run check (R)", H2))
    s.append(P("While designing the mutation tests, a real gap was found: the audit recomputed every metric from the "
               "saved forecasts, but for oil and power it never re-ran the model to confirm those saved forecasts are "
               "what the frozen model actually produces. A fabricated forecast file with consistent metrics would have "
               "passed. A new check (section R of tools/deploy/audit_deploy.py) now re-runs each frozen world model on "
               "a sample of the saved origins in every project and requires an exact match. The mutation tests below "
               "include two such 'perfect fakes' to prove the check works.", B))
    if MUT:
        s.append(P("A2. Mutation tests (V1): plant one fake, the audit must fail", H2))
        rows = [["Planted fake", "Must trip", "Tripped", "Result"]]
        for x in MUT:
            rows.append([x["what"], ", ".join(x["expected_fail_kinds"]) or "nothing (control)",
                         ", ".join(x["failed_kinds"]) or "nothing", x["verdict"]])
        s.append(tbl(rows, [104 * mm, 22 * mm, 20 * mm, 30 * mm]))
        s.append(P("Check kinds: B no-training, C truth re-derived from raw files, D metric recomputation, F sample-size "
                   "rule, G sealing (sha256), R fresh re-run of the frozen model.", SM))
    if TAMP:
        s.append(P("A3. Tamper and determinism tests (V2)", H2))
        rows = [["Test", "Project", "Expected", "Result", "Detail"]]
        for r in TAMP:
            rows.append([r["test"], r["project"], r["expected"], "as expected" if r["ok"] else "NOT as expected",
                         r["detail"][:150]])
        s.append(tbl(rows, [12 * mm, 16 * mm, 22 * mm, 24 * mm, 102 * mm]))
        s.append(P("T1 one-byte flip in a checkpoint → the wrapper refuses; T2 → the deployment refuses; T3 one-character "
                   "edit of frozen model code → refused; T4 a raw data file that no longer matches its pin → refused; "
                   "T5 optimizer step / backward() under the no-training lock → refused. D1 two runs → bit-identical "
                   "forecasts; D2 1 vs 4 CPU threads (reported only).", SM))
    if LEAK:
        s.append(P("A4. Exhaustive leak tests with positive controls (V3)", H2))
        s.append(P("Every October–November 2026 origin (not a sample), every world model and twin, four kinds of "
                   "corrupted future (random noise, all missing, ×1000, the past copied into the future). A forecast "
                   "must not change. The positive controls use a deliberately leaky context (an off-by-one that lets the "
                   "target in); the same test must flag those, proving it is not blind.", B))
        rows = [["Project", "Model", "Corruption", "Forecasts", "Changed", "Kind", "Result"]]
        for r in LEAK:
            rows.append([r["project"], r["model"], r["corruption"], r["n_forecasts"], r["changed"],
                         "positive control" if r["positive_control"] else "leak test",
                         "as expected" if r["ok"] else "NOT as expected"])
        s.append(tbl(rows, [16 * mm, 30 * mm, 24 * mm, 20 * mm, 18 * mm, 32 * mm, 36 * mm]))
    s.append(PageBreak())
    # ================= Part B
    s.append(P("Part B — Is the performance real and stable after training?", H1))
    s.append(P("Every frozen model and every pre-registered competitor was re-run on every forecast origin from "
               "2025-07-01 (the day after the training data end) to the November 2026 data cutoff — about 16 months — "
               "with exactly the deployment harness (no retraining). Windows: each calendar month, VAL (2025-07..10), "
               "TEST (2025-11..2026-10), FORWARD (2026-11) and ALL. Pooled verdicts use a 95% block bootstrap (7-day "
               "blocks for daily data, overlap-aware) and are issued only with ≥ 10 independent units. "
               "<b>Caveat:</b> VAL and TEST were scored by the original project, so this is a re-look at known data, "
               "not a fresh test; it answers 'is the advantage stable month to month?', not 'was it pre-registered?'.", B))
    s.append(Image(fig_forest(), width=176 * mm, height=176 * mm * 3.6 / 7.4))
    s.append(P("Figure B1. Pooled Jul 2025 – Nov 2026 quantile score: each Alma model minus each competitor.", SM))
    s.append(Image(fig_monthly_rank(), width=176 * mm, height=176 * mm * 4.8 / 7.4))
    s.append(P("Figure B2. Month-by-month rank of each frozen Alma model among all models (1 = best).", SM))
    for proj in ("gas", "power", "crypto", "oil"):
        s.append(KeepTogether([P(f"B — {proj}: pooled Jul 2025 – Nov 2026 (n = {n_of(proj, 'ALL_POST_TRAINING')} "
                                 f"{'weeks' if proj == 'gas' else 'days'})", H2)] + pooled_table(proj)))
    # splits + twin
    rows = [["Project", "Split", "n", f"Alma QS", "best competitor", "rank"]]
    for proj in ("gas", "power", "crypto", "oil"):
        for win in ("VAL_2025_07_10", "TEST_2025_11_2026_10", "FORWARD_2026_11", "ALL_POST_TRAINING"):
            blk = qs_block(proj, win)
            if not blk:
                continue
            sc = {m: b["qs"] for m, b in blk.items() if b.get("qs") is not None}
            order = sorted(sc, key=sc.get)
            c = CAND[proj]
            comp = [m for m in order if m not in ALMA]
            nd = 5 if proj == "oil" else (4 if proj == "crypto" else 2)
            rows.append([proj, win.split("_")[0], n_of(proj, win), f(sc.get(c), nd),
                         f"{comp[0]} {f(sc[comp[0]], nd)}" if comp else "–", f"{order.index(c) + 1}/{len(order)}"])
    s.append(KeepTogether([P("B5. By split", H2), tbl(rows, [20 * mm, 24 * mm, 14 * mm, 26 * mm, 62 * mm, 30 * mm])]))
    rows = [["Project", "World model QS", "Direct twin QS", "WM − twin", "95% CI", "verdict"]]
    for proj, (wm, tw) in TWIN.items():
        r = POST[proj]
        if r is None:
            continue
        w = r["windows"]["ALL_POST_TRAINING"]
        if proj == "oil":
            blk = w["h1"]["models"]
            a, b = blk[wm]["qs"], blk[tw]["qs"]
            d, ci, v = blk[tw].get("wm_sv_minus_this_qs"), blk[tw].get("ci95_qs"), blk[tw].get("verdict_qs")
        elif proj == "gas":
            blk = w["kinds"]["thu"]["by_candidate"].get("wm_gas", {}).get("qs") or {}
            a, b = blk.get(wm, {}).get("qs"), blk.get(tw, {}).get("qs")
            d, ci, v = blk.get(tw, {}).get("wm_gas_minus_this_qs"), blk.get(tw, {}).get("ci95_qs"), blk.get(tw, {}).get("verdict_qs")
        else:
            blk = qs_block(proj, "ALL_POST_TRAINING")
            a, b = blk[wm]["qs"], blk[tw]["qs"]
            d, ci, v = blk[tw].get(f"{wm}_minus_this_qs"), blk[tw].get("ci95_qs"), blk[tw].get("verdict_qs")
        nd = 5 if proj == "oil" else 4
        rows.append([proj, f(a, nd), f(b, nd), f(d, nd), f(ci, nd) if ci else "–",
                     {"cand_better": "rollout better", "cand_worse": "twin better", "not_significant": "tie"}.get(v, v or "–")])
    s.append(KeepTogether([P("B6. The world-model test over 16 months: rollout vs matched direct twin", H2),
                           tbl(rows, [20 * mm, 30 * mm, 30 * mm, 26 * mm, 44 * mm, 26 * mm])]))
    s.append(PageBreak())
    # ================= Part C
    s.append(P("Part C — Are the forecasts sensible and honest about uncertainty?", H1))
    if V5:
        s.append(P("C1. Physics / economics property tests (frozen models, real October 2026 inputs)", H2))
        rows = [["Project", "Property", "Cases", "Violations", "Measured effect"]]
        for p in V5["properties"]:
            rows.append([p["project"], p["test"], p["n_cases"], p["violations"], p["detail"][:120]])
        s.append(tbl(rows, [16 * mm, 56 * mm, 14 * mm, 18 * mm, 72 * mm]))
        rp = fig_reliability()
        if rp:
            s.append(P("C2. Calibration, Jul 2025 – Nov 2026", H2))
            s.append(Image(rp, width=176 * mm, height=176 * mm * 2.4 / 7.4))
            rows = [["Model", "n", "max |share − level|", "80% coverage"]]
            for k, c in V5["calibration"].items():
                rows.append([k, f"{c['n']:,}", f(c["max_abs_dev"], 3), f(c["cov80"], 3)])
            s.append(tbl(rows, [60 * mm, 30 * mm, 44 * mm, 42 * mm]))
        s.append(P("C3. Placebo: the true outcomes vs outcomes shuffled across dates (500 shuffles)", H2))
        s.append(P("If a forecast carried no information about the specific date, shuffling which date each outcome "
                   "belongs to would not hurt its score. All improvements below are on the true pairing.", B))
        rows = [["Model", "n", "QS true", "QS shuffled", "improvement", "p"]]
        for k, c in V5["placebo"].items():
            rows.append([k, f"{c['n']:,}", f"{c['qs_true']:.4g}", f"{c['qs_shuffled_mean']:.4g}",
                         f"{100 * c['improvement_vs_shuffled']:.1f}%", f"{c['p_value']:.4f}"])
        s.append(tbl(rows, [50 * mm, 22 * mm, 26 * mm, 26 * mm, 28 * mm, 24 * mm]))
    s.append(PageBreak())
    # ================= Part D
    s.append(P("Part D — What is still open", H1))
    led = os.path.join(ROOT, "runs", "deploy_daily", "EXANTE_LEDGER.json")
    ex = json.load(open(led)) if os.path.exists(led) else []
    n_pub = sum(r["published"] for r in ex)
    s += R.bullets([
        f"<b>Sealed ex-ante forecasts:</b> {len(ex)} model-forecasts are sealed (sha256, committed before the outcome); "
        f"{n_pub} graded so far. The first gas release (2026-11-05 15:30 UTC) and the 2026-11-06 power auction are "
        "graded automatically (tools/deploy/score_exante.py, runs/deploy_daily/EXANTE_LEDGER.md).",
        "<b>Fresh evidence needs time:</b> about 20 gas weeks or 60 power/crypto/oil days of forward data before a "
        "forward confidence interval is informative. The daily routine (09:13 UTC) accumulates it and audits every run.",
        "<b>Re-look caveat:</b> Part B's 16 months include VAL and TEST, both scored before; only FORWARD (Nov 2026 on) "
        "is untouched by any decision.",
        "<b>Not tested here:</b> retraining, new architectures, the HH-price task (no weather data in this environment), "
        "and the RL trading agents of Alma-Fin."])
    s.append(P("How to reproduce", H2))
    s.append(tbl([["Step", "Command"],
                  ["V1 mutation tests", ".venv/bin/python tools/validate/mutation_test.py"],
                  ["V2 tamper + determinism", ".venv/bin/python tools/validate/tamper_determinism.py"],
                  ["V3 exhaustive leak", ".venv/bin/python tools/validate/leak_exhaustive.py"],
                  ["V4 16-month evaluation", "ALMA_WINDOWS=posttrain ALMA_DEPLOY_OUT=runs/validation/posttrain "
                   ".venv/bin/python tools/deploy/deploy_{oil,power,gas,fin}.py"],
                  ["V5 calibration / properties / placebo", ".venv/bin/python tools/validate/calib_props_placebo.py"],
                  ["This PDF", ".venv/bin/python tools/validate/make_validation_pdf.py"]],
                 [46 * mm, 130 * mm]))
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=15 * mm,
                            bottomMargin=17 * mm, title="Alma Models — Overnight validation report",
                            author="Alma Models", subject="Integrity and performance validation of the frozen world models")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("->", OUT)


if __name__ == "__main__":
    build()
