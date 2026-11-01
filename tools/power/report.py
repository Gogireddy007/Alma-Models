"""metrics.json -> report.md (tools/power). Usage: report.py runs/power/<split>/metrics.json"""
import os
import sys
import json


def f(x, d=3):
    if x is None:
        return "-"
    try:
        if x != x:
            return "-"
        return f"{x:.{d}f}"
    except TypeError:
        return str(x)


def write(path):
    r = json.load(open(path))
    M = r["models"]
    order = sorted(M, key=lambda m: (["baseline", "competitor", "fm", "alma", "candidate", "model"].index(M[m]["kind"]), m))
    L = [f"# Alma-Power {r['split'].upper()} report", "",
         f"Target days {r['first_day']} .. {r['last_day']} (n={r['n_days']}); commit {str(r.get('commit'))[:10]}; "
         f"data sha256 {str(r.get('data_sha256'))[:12]}; scored {r.get('scored')}.",
         f"DEV majority up-rates: P1 {f(r['dev_majority']['P1'])}, P2 {f(r['dev_majority']['P2'])}, "
         f"P3 {f(r['dev_majority']['P3'])}. Excluded: {r['excluded']}.", "",
         "Flags: C = contaminated (excluded from 'beats all'); S = subsample.", "",
         "## Direction (P1 daily base, P2 hourly, P3 peak/off-peak spread)", "",
         "| model | kind | P1 acc [Wilson 95%] | P1 p_binom | P1 p_PT | P2 acc [day-block 95%] | P2 p(DM vs .5) | P3 acc [Wilson] | P3 p_binom |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        x = M[m]
        fl = ("C" if x.get("contaminated") else "") + ("S" if x.get("subsample") else "")
        p1, p2, p3 = x.get("P1", {}), x.get("P2", {}), x.get("P3", {})
        L.append(f"| {m} {fl} | {x['kind']} | {f(p1.get('acc'))} [{f(p1.get('ci_lo'))}, {f(p1.get('ci_hi'))}] n={p1.get('n', '-')} "
                 f"| {f(p1.get('p_binom'), 4)} | {f(p1.get('p_pt'), 4)} "
                 f"| {f(p2.get('acc'))} [{f(p2.get('ci_lo'))}, {f(p2.get('ci_hi'))}] | {f(p2.get('p_vs_half_dm'), 4)} "
                 f"| {f(p3.get('acc'))} [{f(p3.get('ci_lo'))}, {f(p3.get('ci_hi'))}] | {f(p3.get('p_binom'), 4)} |")
    L += ["", "## Probabilistic / point accuracy (P4, EUR/MWh)", "",
          "| model | kind | n days | QS (0.1..0.9) | CRPS~ | skill vs naive_epf | MAE | rMAE | RMSE | cov80 | cov95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        x = M[m].get("P4")
        if not x:
            continue
        fl = ("C" if M[m].get("contaminated") else "") + ("S" if M[m].get("subsample") else "")
        L.append(f"| {m} {fl} | {M[m]['kind']} | {x.get('n', '-')} | {f(x.get('qs'), 2)} | {f(x.get('crps_approx'), 2)} "
                 f"| {f(x.get('skill_vs_naive_epf'))} | {f(x.get('mae'), 2)} | {f(x.get('rmae_vs_naive_epf'))} "
                 f"| {f(x.get('rmse'), 2)} | {f(x.get('cov80'))} | {f(x.get('cov95'))} |")
    blocks = None
    for m in order:
        if "P2" in M[m]:
            blocks = list(M[m]["P2"]["blocks"])
            break
    if blocks:
        L += ["", "## P2 accuracy by hour block (descriptive; day-block bootstrap CIs in metrics.json)", "",
              "| model | " + " | ".join(blocks) + " |", "|---|" + "---|" * len(blocks)]
        for m in order:
            if "P2" in M[m]:
                L.append(f"| {m} | " + " | ".join(f(M[m]["P2"]["blocks"].get(b, {}).get("acc")) for b in blocks) + " |")
    claims = []
    for tgt, pw in r["pairwise"].items():
        for k, v in pw.items():
            if "beats" in v:
                claims.append((tgt, k, v))
    if claims:
        L += ["", "## Claim-model comparisons (prereg section 7: Holm over own competitors + block-bootstrap CI)", "",
              "| target | a | b | mean diff (a-b) | boot 95% | p DM | p Holm (own) | beats |", "|---|---|---|---|---|---|---|---|"]
        for tgt, k, v in claims:
            a, b = k.split("|")
            L.append(f"| {tgt} | {a} | {b} | {f(v['mean_diff'], 4)} | [{f(v['boot_lo'], 4)}, {f(v['boot_hi'], 4)}] "
                     f"| {f(v['p'], 4)} | {f(v.get('p_holm_own'), 4)} | {'YES' if v['beats'] else 'no'} |")
    out = os.path.join(os.path.dirname(path), "report.md")
    open(out, "w").write("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    print(write(sys.argv[1]))
