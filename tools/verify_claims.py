"""Independent re-derivation of every bench-val number we report (evaluator: Rossi).

    uv run tools/verify_claims.py            # writes runs/verified.md, prints it

Deliberately does NOT import tools/compare.py, tools/leaderboard.py or tools/bench.py. Definitions
are re-implemented from docs/eval_rigor.md and the bench.evaluate docstring:
  headline wql_rel / mase_rel = geometric mean over tasks of per-task (model / seasonal naive);
  cov80_error = mean over tasks of |cov80 - 0.8|;
  family-weighted = geometric mean over families of the per-family geometric mean;
  paired ratio A/B = exp(mean_task(log A - log B));
  CI_fam  = 95% percentile family-cluster bootstrap of the task-weighted ratio (10,000 draws, seed 0);
  CI_famw = same, for the family-weighted ratio exp(mean_fam(mean log ratio in family)).
Reads only text files: runs/*.log (the per-task table printed by bench.evaluate) and
baselines/bench/per_task/<name>_val.csv. No training, no scoring, no model loading.
"""
import csv
import glob
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
BL = os.path.join(ROOT, "baselines", "bench", "per_task")
ID_KEYS = ("family", "horizon", "n_forecasts", "n_series")
LOG_HALF_ULP = 0.0005          # logs print per-task values to 3 decimals
N_BOOT, SEED = 10_000, 0
# Baselines the team's success criterion names (docs/team.md). chronos_bolt_base has no val CSV.
EXPECTED_BASELINES = ["chronos2", "timestral_5m", "timestral_28m", "timestral_209m", "tirex",
                      "chronos_bolt_base", "auto_ets", "auto_theta", "seasonal_naive_pi"]

# compare.py `pair` output, captured 2026-10-29 by running it under tools/smoke.py (not imported):
#   uv run tools/compare.py pair A B   ->  CI_fam, CI_famw
COMPARE_REF = {
    ("b_curr", "b_base256"): {"ratio": 0.9552, "ci_fam": (0.9067, 0.9963), "ci_famw": (0.9319, 0.9997)},
    ("b_syn3", "b_base256"): {"ratio": 0.9605, "ci_fam": (0.9296, 0.9906), "ci_famw": (0.9435, 0.9947)},
    ("b_base256", "chronos2"): {"ratio": 1.2993, "ci_fam": (1.2135, 1.4000), "ci_famw": (1.1883, 1.4057)},
}
# CIs written in research_log.md (19:50 entry)
LOG_CI = {("b_curr", "b_base256"): (0.904, 0.996), ("b_syn3", "b_base256"): (0.929, 0.989),
          ("c_mix", "c_base"): (0.988, 1.000), ("b_a1_copy", "b_base256"): (0.993, 1.071)}


# ---------------------------------------------------------------- parsing
def _num(x):
    try:
        return float(x)
    except ValueError:
        return x


def parse_log(path):
    """Last per-task table + summary in a train log. Returns (rows, summary) or (None, None)."""
    lines = open(path, errors="replace").read().splitlines()
    hdr = [i for i, l in enumerate(lines) if re.match(r"^\s*dataset\s+family\s", l)]
    if not hdr:
        return None, None
    i0 = hdr[-1]
    cols = lines[i0].split()
    rows = {}
    for l in lines[i0 + 1:]:
        if not l.strip() or l.startswith("val:") or l.startswith("test:") or l.startswith("---"):
            break
        parts = l.split()
        if len(parts) != len(cols):
            break
        r = dict(zip(cols, map(_num, parts)))
        rows[r["dataset"]] = r
    summ = {}
    for l in lines[i0:]:
        m = re.match(r"^val: wql_rel=(\S+) mase_rel=(\S+) cov80_error=(\S+) family-gmean wql_rel=(\S+)", l)
        if m:
            summ.update(line_wql=float(m[1]), line_mase=float(m[2]), line_cov=float(m[3]), line_fam=float(m[4]))
        m = re.match(r"^(val_wql_rel|val_mase_rel|val_cov80_error|family_wql_rel):\s+(\S+)", l)
        if m:
            summ[m[1]] = float(m[2])
    return rows, summ


def parse_csv(path):
    with open(path) as f:
        return {r["dataset"]: {k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)}


# ---------------------------------------------------------------- metrics
def gmean(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(np.clip(x, 1e-6, None)))))


def headline(rows):
    tasks = sorted(rows)
    w = [rows[t]["wql_rel"] for t in tasks]
    m = [rows[t]["mase_rel"] for t in tasks]
    cov = float(np.mean([abs(rows[t]["cov80"] - 0.8) for t in tasks]))
    fams = sorted({rows[t]["family"] for t in tasks})
    famw = gmean([gmean([rows[t]["wql_rel"] for t in tasks if rows[t]["family"] == f]) for f in fams])
    return {"wql": gmean(w), "mase": gmean(m), "cov": cov, "fam": famw, "n_tasks": len(tasks), "n_fam": len(fams)}


def paired(A, B, metric="wql_rel"):
    tasks = sorted(A)
    assert set(tasks) == set(B), "task sets differ"
    d = np.array([math.log(max(A[t][metric], 1e-12)) - math.log(max(B[t][metric], 1e-12)) for t in tasks])
    fam = np.array([A[t]["family"] for t in tasks])
    uf = sorted(set(fam))
    S = np.array([d[fam == f].sum() for f in uf])      # per-family sum of log ratios
    n = np.array([(fam == f).sum() for f in uf], float)
    mu = S / n                                         # per-family mean log ratio
    G = len(uf)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, G, size=(N_BOOT, G))        # resample families with replacement
    b_task = np.exp(S[draw].sum(1) / n[draw].sum(1))   # pooled tasks of the drawn families
    b_fam = np.exp(mu[draw].mean(1))
    pct = lambda b: tuple(float(v) for v in np.percentile(b, [2.5, 97.5]))
    return {"ratio": float(np.exp(d.mean())), "ratio_famw": float(np.exp(mu.mean())),
            "ci_fam": pct(b_task), "ci_famw": pct(b_fam), "G": G, "n": len(d),
            "fam_wins": int((mu < -1e-3).sum()), "fam_losses": int((mu > 1e-3).sum())}


# ---------------------------------------------------------------- main
def main():
    out, mism = [], []
    notes = []

    # 1. load
    alma = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.log"))):
        rows, summ = parse_log(p)
        if rows is not None:
            alma[os.path.basename(p)[:-4]] = (rows, summ)
    bls = {os.path.basename(p)[:-len("_val.csv")]: parse_csv(p) for p in sorted(glob.glob(os.path.join(BL, "*_val.csv")))}
    tsv = {}
    with open(os.path.join(ROOT, "baselines", "bench", "results.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == "val":
                tsv[r["model"]] = {k: _num(v) for k, v in r.items()}

    # 3. task identity, against chronos2 as the reference task set
    ref = bls["chronos2"]
    ref_id = {t: tuple(ref[t][k] for k in ID_KEYS) for t in ref}
    ident = []
    for name, rows in [(f"baseline:{k}", v) for k, v in bls.items()] + [(k, v[0]) for k, v in alma.items()]:
        miss, extra = sorted(set(ref) - set(rows)), sorted(set(rows) - set(ref))
        diff = [t for t in set(ref) & set(rows) if tuple(rows[t][k] for k in ID_KEYS) != ref_id[t]]
        nonfin = [t for t, r in rows.items()
                  if any(isinstance(r.get(k), float) and not math.isfinite(r[k]) for k in ("wql_rel", "mase_rel", "cov80"))
                  or (isinstance(r.get("n_nonfinite"), float) and r["n_nonfinite"] != 0)]
        no_col = "n_nonfinite" not in next(iter(rows.values()))
        # same windows: Alma's implied seasonal-naive WQL (wql / wql_rel) must match the baseline's sn_wql
        sn_bad = []
        for t in set(ref) & set(rows):
            r, s = rows[t], ref[t]["sn_wql"]
            if "sn_wql" in r:
                if abs(r["sn_wql"] - s) > 1e-9 * max(1, s):
                    sn_bad.append(t)
            elif r["wql_rel"] > 0:
                tol = LOG_HALF_ULP * (1 + s) + 1e-6 + LOG_HALF_ULP * s / r["wql_rel"]
                if abs(r["wql"] - r["wql_rel"] * s) > tol:
                    sn_bad.append(t)
        ok = not (miss or extra or diff or nonfin or sn_bad)
        ident.append((name, len(rows), ok, miss, extra, diff, nonfin, sn_bad, no_col))
    for b in EXPECTED_BASELINES:
        if b not in bls:
            notes.append(f"Baseline **{b}** has no `baselines/bench/per_task/{b}_val.csv` (not scored on val).")

    # 2. headline recompute vs summary lines
    head_rows = []
    for name, (rows, s) in sorted(alma.items()):
        h = headline(rows)
        checks = [("wql", "val_wql_rel", "line_wql"), ("mase", "val_mase_rel", "line_mase"),
                  ("cov", "val_cov80_error", "line_cov"), ("fam", "family_wql_rel", "line_fam")]
        worst = 0.0
        for k, full, line in checks:
            ref_v = s.get(full, s.get(line))
            if ref_v is not None:
                worst = max(worst, abs(h[k] - ref_v) / (ref_v if k != "cov" else 1))
        # 3-dp rounding of per-task values -> up to ~0.07% relative on a gmean, ~5e-4 abs on cov
        ok = worst < 2e-3 and h["n_tasks"] == 30
        head_rows.append((name, h, s, worst, ok))
    for name, rows in sorted(bls.items()):
        h = headline(rows)
        t = tsv.get(name, {})
        worst = max([abs(h["wql"] - t["wql_rel"]) / t["wql_rel"], abs(h["mase"] - t["mase_rel"]) / t["mase_rel"],
                     abs(h["cov"] - t["cov80_error"]), abs(h["fam"] - t["wql_rel_family"]) / t["wql_rel_family"]]) if t else float("nan")
        head_rows.append((f"baseline:{name}", h, {"val_wql_rel": t.get("wql_rel"), "val_mase_rel": t.get("mase_rel"),
                          "val_cov80_error": t.get("cov80_error"), "family_wql_rel": t.get("wql_rel_family")},
                          worst, worst < 1e-3))

    get = lambda n: alma[n][0] if n in alma else bls[n]
    H = {n: headline(get(n)) for n in ["b_base256", "b_curr", "b_curr_s1", "b_syn3", "c_base", "c_mix",
                                        "b_jepa_vic_A3", "b_jepa_vic", "b_a1_copy", "chronos2", "timestral_209m",
                                        "tirex", "auto_theta", "seasonal_naive_pi", "auto_ets"]}
    full = lambda n, k="val_wql_rel": alma[n][1][k]        # full-precision summary value in the log

    # 5. claims
    def claim(text, reported, value, dp, src):
        ok = abs(value - reported) <= 0.5 * 10 ** -dp + 1e-9
        out.append((text, f"{reported:.{dp}f}", f"{value:.{dp + 2}f}", "OK" if ok else "MISMATCH", src))
        if not ok:
            mism.append(text)

    claim("base (b_base256) wql_rel", 0.726, full("b_base256"), 3, "runs/b_base256.log")
    claim("b_curr wql_rel", 0.693, full("b_curr"), 3, "runs/b_curr.log")
    claim("b_curr seed 0 (\"0.6933\")", 0.6933, full("b_curr"), 4, "runs/b_curr.log")
    claim("b_curr seed 1 (\"0.6934\")", 0.6934, full("b_curr_s1"), 4, "runs/b_curr_s1.log")
    claim("b_syn3 wql_rel", 0.697, full("b_syn3"), 3, "runs/b_syn3.log")
    claim("c_mix cov80_error before (c_base)", 0.100, full("c_base", "val_cov80_error"), 3, "runs/c_base.log")
    claim("c_mix cov80_error after", 0.081, full("c_mix", "val_cov80_error"), 3, "runs/c_mix.log")
    claim("A3 state-zeroed wql_rel", 0.932, full("b_jepa_vic_A3"), 3, "runs/b_jepa_vic_A3.log")
    claim("A1 state-copy wql_rel", 0.745, full("b_a1_copy"), 3, "runs/b_a1_copy.log")
    for n, v, dp, lab in [("chronos2", 0.559, 3, "Chronos-2"), ("timestral_209m", 0.564, 3, "Timestral-209M"),
                          ("tirex", 0.582, 3, "TiRex"), ("auto_theta", 0.959, 3, "AutoTheta"),
                          ("seasonal_naive_pi", 0.988, 3, "Seasonal Naive (PI)"), ("auto_ets", 2.68, 2, "AutoETS")]:
        claim(f"{lab} wql_rel (recomputed from per-task CSV)", v, H[n]["wql"], dp, f"baselines/bench/per_task/{n}_val.csv")
    pc = paired(get("b_curr"), get("chronos2"))
    claim("\"about 24% behind Chronos-2\" (b_curr/Chronos-2 - 1)", 0.24, pc["ratio"] - 1, 2, "paired, 30 tasks")
    claim("A3 relative to its parent b_jepa_vic (\"+28%\")", 0.28, full("b_jepa_vic_A3") / full("b_jepa_vic") - 1, 2, "logs")
    claim("A3 relative to b_base256 (\"+28%\")", 0.28, full("b_jepa_vic_A3") / full("b_base256") - 1, 2, "logs")
    claim("A1 relative to b_base256 (\"+2.6%\")", 0.026, full("b_a1_copy") / full("b_base256") - 1, 3, "logs")
    pb = paired(get("b_base256"), get("chronos2"))
    claim("\"~30% behind Chronos-2\" at rung 1 (b_base256/Chronos-2 - 1)", 0.30, pb["ratio"] - 1, 2, "paired, 30 tasks")

    # 4. paired CIs vs compare.py
    ci_rows = []
    for (a, b), r in COMPARE_REF.items():
        m = paired(get(a), get(b))
        dev = max(abs(m["ratio"] - r["ratio"]), *(abs(x - y) for x, y in zip(m["ci_fam"], r["ci_fam"])),
                  *(abs(x - y) for x, y in zip(m["ci_famw"], r["ci_famw"])))
        ci_rows.append((a, b, m, r, dev, dev <= 0.005))
        if dev > 0.005:
            mism.append(f"CI {a} vs {b}")
    logci = []
    for (a, b), (lo, hi) in LOG_CI.items():
        m = paired(get(a), get(b))
        dev = max(abs(m["ci_fam"][0] - lo), abs(m["ci_fam"][1] - hi))
        logci.append((a, b, m, (lo, hi), dev, dev <= 0.005))
        if dev > 0.005:
            mism.append(f"research_log CI {a} vs {b}")

    # seasonal naive explanation data
    sn = bls["seasonal_naive_pi"]
    sn_mase1 = all(abs(r["mase_rel"] - 1) < 1e-9 for r in sn.values())
    sn_below = sum(r["wql_rel"] < 1 for r in sn.values())
    sn_min = min(sn.values(), key=lambda r: r["wql_rel"])
    sn_max = max(sn.values(), key=lambda r: r["wql_rel"])

    # ---------------------------------------------------------------- write
    L = ["# Verified claims (bench VAL)", "",
         "Generated by `tools/verify_claims.py` (independent; does not import compare.py, leaderboard.py or bench.py). "
         "Per-task inputs: `runs/<name>.log` tables (3 decimals) and `baselines/bench/per_task/<name>_val.csv`.", ""]
    L += [f"**Mismatches: {len(mism)}**" + (": " + "; ".join(mism) if mism else ""), ""]
    L += ["## Claims", "", "| Claim | Reported | Recomputed | Status | Source |", "|---|---|---|---|---|"]
    L += [f"| {c} | {r} | {v} | {s} | {src} |" for c, r, v, s, src in out]
    L += ["", "## Paired family-cluster bootstrap vs compare.py (10,000 draws, seed 0, own RNG stream)", "",
          "| A vs B | ratio (ours / compare) | CI_fam ours | CI_fam compare | CI_famw ours | CI_famw compare | max dev | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for a, b, m, r, dev, ok in ci_rows:
        L.append(f"| {a} vs {b} | {m['ratio']:.4f} / {r['ratio']:.4f} | [{m['ci_fam'][0]:.4f}, {m['ci_fam'][1]:.4f}] | "
                 f"[{r['ci_fam'][0]:.4f}, {r['ci_fam'][1]:.4f}] | [{m['ci_famw'][0]:.4f}, {m['ci_famw'][1]:.4f}] | "
                 f"[{r['ci_famw'][0]:.4f}, {r['ci_famw'][1]:.4f}] | {dev:.4f} | {'OK' if ok else 'MISMATCH'} |")
    L += ["", "CIs quoted in research_log.md (CI_fam):", "", "| A vs B | ratio | CI_fam ours | research_log | max dev | Status |", "|---|---|---|---|---|---|"]
    for a, b, m, (lo, hi), dev, ok in logci:
        L.append(f"| {a} vs {b} | {m['ratio']:.4f} | [{m['ci_fam'][0]:.4f}, {m['ci_fam'][1]:.4f}] | [{lo:.3f}, {hi:.3f}] | {dev:.4f} | {'OK' if ok else 'MISMATCH'} |")
    L += ["", f"Also: b_curr vs chronos2 ratio {pc['ratio']:.4f}, CI_fam [{pc['ci_fam'][0]:.4f}, {pc['ci_fam'][1]:.4f}], "
          f"family W/L {pc['fam_wins']}/{pc['fam_losses']} of {pc['G']}.", ""]
    L += ["## Headline recompute vs summary lines", "",
          "| Run | n tasks / fam | wql_rel (re / log) | mase_rel (re / log) | cov80_err (re / log) | family wql (re / log) | max rel dev | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for name, h, s, worst, ok in head_rows:
        f = lambda k: f"{s[k]:.4f}" if s.get(k) is not None else "-"
        L.append(f"| {name} | {h['n_tasks']}/{h['n_fam']} | {h['wql']:.4f} / {f('val_wql_rel')} | {h['mase']:.4f} / {f('val_mase_rel')} | "
                 f"{h['cov']:.4f} / {f('val_cov80_error')} | {h['fam']:.4f} / {f('family_wql_rel')} | {worst:.5f} | {'OK' if ok else 'MISMATCH'} |")
    L += ["", "## Task identity (reference: chronos2 task set; key = dataset; checked family, horizon, n_forecasts, n_series; "
          "same windows via implied seasonal-naive WQL)", ""]
    bad = [x for x in ident if not x[2]]
    L.append(f"{len(ident) - len(bad)}/{len(ident)} files match the 30-task reference exactly, with no nonfinite values.")
    for name, n, ok, miss, extra, diff, nonfin, sn_bad, no_col in ident:
        if not ok:
            L.append(f"- **{name}** ({n} tasks): missing {miss or '-'}, extra {extra or '-'}, id diff {diff or '-'}, "
                     f"nonfinite {nonfin or '-'}, seasonal-naive WQL mismatch {sn_bad or '-'}")
        if no_col:
            L.append(f"- {name}: no `n_nonfinite` column, so gate G5 cannot be checked from this file.")
    L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    L += [f"- Seasonal Naive (PI) wql_rel = {H['seasonal_naive_pi']['wql']:.4f}, not 1.0. The bench's denominator is the point "
          "seasonal-naive forecast copied into all 9 quantiles (`bench.evaluate`), whose WQL equals its scaled absolute error. "
          "The `seasonal_naive_pi` baseline is statsforecast `SeasonalNaive` with prediction intervals: the same median "
          f"(mase_rel = 1.0 on every task: {sn_mase1}) but spread quantiles, which lower the pinball loss whenever the intervals "
          f"are reasonable. It is below 1 on {sn_below}/30 tasks (min {sn_min['dataset']} {sn_min['wql_rel']:.3f}, "
          f"max {sn_max['dataset']} {sn_max['wql_rel']:.3f}). So 0.988 is correct, and 1.0 is the point-forecast reference, not this baseline.",
          "- \"24% behind\" is the ratio b_curr/Chronos-2 - 1. Stated as a gap relative to our score it is "
          f"{1 - 1 / pc['ratio']:.1%} (Chronos-2 has {1 - 1 / pc['ratio']:.1%} lower WQL). Both are fine if the definition is stated.",
          "- Alma per-task values come from 3-decimal log tables, so recomputed headlines agree with the logs to about 1e-4; "
          "pair CIs from logs inherit that quantization (eval_rigor: prefer dump dirs).",
          "- b_curr and b_curr_s1 are 2 seeds; R3 requires mean +- std over >= 3 seeds for any number that leaves the research log, "
          "so 0.693 is a single-config 2-seed figure, not yet reportable under R3.",
          "- c_base.log matches b_base256.log to 6 decimals in every summary value. MPS runs are not bit-reproducible "
          "(eval_rigor), so this is the same checkpoint re-scored or a copied log; either way the c_mix gain is measured on the "
          f"same base. Second seed: c_base_s1 cov80_err {full('c_base_s1', 'val_cov80_error'):.4f} -> c_mix_s1 "
          f"{full('c_mix_s1', 'val_cov80_error'):.4f}, so the calibration gain replicates (smaller: 0.114 -> 0.089).",
          "- A3 was evaluated on the b_jepa_vic checkpoint. \"+28%\" holds against b_base256 (+28.4%) but against its own "
          "parent b_jepa_vic it is +28.6% (rounds to 29%). State the reference.",
          "- \"0.6933/0.6934\" for the two b_curr seeds is off by 0.0001 each: the logs say 0.693242 / 0.693349, i.e. "
          "0.6932 / 0.6933 (mean 0.6933, sd 0.0001)."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(RUNS, "verified.md"), "w") as f:
        f.write(text)
    print(text)
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
