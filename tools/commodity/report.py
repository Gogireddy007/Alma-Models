"""
Markdown tables from runs/commodity/<split>/metrics.json.

    .venv/bin/python tools/commodity/report.py --split val [--candidate alma_s0] > runs/commodity/val/report.md
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics as MT   # noqa: E402

WM_PAIRS = [("alma_s0", "alma_direct_s0"), ("alma_s0_tta", "alma_direct_s0_tta")]   # amendment A2: s0 pair only


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def fp(x):
    if x is None or not np.isfinite(x):
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def order(models, key):
    return sorted(models, key=lambda m: (models[m].get("kind", "model") == "baseline", -np.nan_to_num(models[m][key], nan=-9)))


def dir_table(blk, h, bases):
    out = [f"| model | n | acc (all origins) | block-bootstrap 95% CI | acc non-overlap (n) | Wilson 95% CI non-overlap | "
           "binom p non-overlap | PT p non-overlap | " + " | ".join(f"Δ vs {b} (Holm p)" for b in bases) + " | note |",
           "|---|---|---|---|---|---|---|---|" + "---|" * len(bases) + "---|"]
    for m in order(blk["models"], "acc"):
        r = blk["models"][m]
        cells = []
        for b in bases:
            if m == b or r["kind"] == "baseline":
                cells.append("")
                continue
            pw = blk["pairwise"].get(f"{m}|{b}", {})
            cells.append(f"{pw.get('acc_diff', np.nan) * 100:+.1f}pp ({fp(pw.get('p_holm_family', np.nan))})")
        note = ("baseline" if r["kind"] == "baseline" else "") + (" CONTAMINATED" if r.get("contaminated") else "") \
            + (" every-5th-origin subsample" if r.get("subsample") else "")
        out.append(f"| {m} | {r['n']} | {f(r['acc'])} | [{f(r['boot_lo'])}, {f(r['boot_hi'])}] | "
                   f"{f(r['acc_nonoverlap'])} ({r['n_nonoverlap']}) | [{f(r['ci_lo'])}, {f(r['ci_hi'])}] | "
                   f"{fp(r['p_binom'])} | {fp(r['p_pt'])} | " + " | ".join(cells) + f" | {note.strip()} |")
    return "\n".join(out)


def t4_table(blk):
    out = ["| model | n | mean QS | skill vs RW | p vs RW, primary (Holm) [n test] | DM p, Bartlett 2h (suppl.) | cov80 | cov90 | cov95 | note |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    ms = blk["models"]
    for m in sorted(ms, key=lambda m: -np.nan_to_num(ms[m].get("skill_vs_rw", 0.0), nan=-9)):
        r = ms[m]
        out.append(f"| {m} | {r['n']} | {f(r['qs'], 5)} | {f(r.get('skill_vs_rw'), 4)} | "
                   f"{fp(r.get('p_vs_rw', np.nan))} ({fp(r.get('p_vs_rw_holm_family', np.nan))}) [{r.get('n_sub', '–')}] | "
                   f"{fp(r.get('p_vs_rw_dm_wide', np.nan))} | "
                   f"{f(r.get('cov80'))} | {f(r.get('cov90'))} | {f(r.get('cov95'))} | "
                   f"{'CONTAMINATED' if r.get('contaminated') else ''}{' every-5th-origin subsample' if r.get('subsample') else ''} |")
    return "\n".join(out)


def wm_test(S, extra_pairs=()):
    pairs = WM_PAIRS + list(extra_pairs)
    out = ["| series | pair (rollout vs direct) | target | h | rollout | direct | diff | DM p | Holm p (h x pairs) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sname in ("wti",):
        s = S[sname]
        rows, ps = [], []
        for a, b in pairs:
            for h, blk in s["T4"].items():
                if a in blk["models"] and b in blk["models"]:
                    pw = blk["pairwise"][f"{a}|{b}"]
                    rows.append([sname, f"{a} vs {b}", "T4 QS (lower better)", h, f(blk["models"][a]["qs"], 5),
                                 f(blk["models"][b]["qs"], 5), f(pw["qs_diff"], 6), pw["p"]])
                    ps.append(pw["p"])
        adj = MT.holm(ps)
        for r, pa in zip(rows, adj):
            out.append("| " + " | ".join(map(str, r[:-1])) + f" | {fp(r[-1])} | {fp(pa)} |")
        rows, ps = [], []
        for a, b in pairs:
            for h, blk in s["T1"].items():
                if a in blk["models"] and b in blk["models"]:
                    pw = blk["pairwise"][f"{a}|{b}"]
                    rows.append([sname, f"{a} vs {b}", "T1 acc", h, f(blk["models"][a]["acc"]),
                                 f(blk["models"][b]["acc"]), f"{pw['acc_diff'] * 100:+.1f}pp", pw["p"]])
                    ps.append(pw["p"])
        for r, pa in zip(rows, MT.holm(ps)):
            out.append("| " + " | ".join(map(str, r[:-1])) + f" | {fp(r[-1])} | {fp(pa)} |")
    return "\n".join(out)


def beats_all(S, cand):
    out = ["| target | series | h | candidate | best competitor | candidate rank | competitors beaten (Holm p<0.05) | "
           "competitors significantly better |", "|---|---|---|---|---|---|---|---|"]
    specs = [("T1", "wti", "acc", True), ("T2", "wti", "acc", True), ("T1", "inv", "acc", True),
             ("T4", "wti", "skill_vs_rw", True)]
    for tgt, sname, key, hib in specs:
        if sname not in S:
            continue
        for h, blk in S[sname][tgt].items():
            ms = blk["models"]
            if cand not in ms:
                continue
            comp = [m for m in ms if m != cand and not ms[m].get("contaminated") and not m.startswith("alma")]
            ps = [blk["pairwise"][f"{cand}|{m}"]["p"] for m in comp]
            adj = MT.holm(ps)
            cv = ms[cand].get(key)
            nz = lambda x: -9.0 if x is None or not np.isfinite(x) else x   # noqa: E731
            better = lambda m: nz(ms[m].get(key)) > nz(cv)   # noqa: E731
            beaten = [m for m, pa in zip(comp, adj) if pa < 0.05 and not better(m)]
            worse_sig = [m for m, pa in zip(comp, adj) if pa < 0.05 and better(m)]
            rank = 1 + sum(better(m) for m in comp)
            best = max(comp, key=lambda m: np.nan_to_num(ms[m].get(key), nan=-9)) if comp else None
            out.append(f"| {tgt if sname != 'inv' else 'T3'} | {sname} | {h} | {f(cv, 4)} | {best} ({f(ms[best].get(key), 4)}) | "
                       f"{rank}/{len(comp) + 1} | {', '.join(beaten) or 'none'} | {', '.join(worse_sig) or 'none'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--candidate", default="alma_s0")
    ap.add_argument("--pair", action="append", default=[], help="extra rollout,direct pair for the WM test")
    a = ap.parse_args()
    M = json.load(open(os.path.join(ROOT, "runs", "commodity", a.split, "metrics.json")))
    S = M["series"]
    print(f"# Commodity metrics: split={a.split}, commit {M['commit']}, scored {M['scored']}\n")
    for sname, s in S.items():
        print(f"## {sname}: {s['n_origins']} origins {s['first_origin']} .. {s['last_origin']}; "
              f"TRAIN up-fractions {json.dumps({k: round(v, 3) for k, v in s['train_majority_upfrac'].items()})}\n")
        tname = "T3 inventory direction" if sname == "inv" else "T1 price direction"
        for h, blk in s["T1"].items():
            bases = [m for m, r in blk["models"].items() if r["kind"] == "baseline"]
            print(f"### {sname} {tname}, h={h} (base rate up {f(blk['base_rate_up'])}, ties excluded "
                  f"{blk['n_ties_excluded']})\n")
            print(dir_table(blk, h, bases) + "\n")
        for h, blk in s["T2"].items():
            bases = [m for m, r in blk["models"].items() if r["kind"] == "baseline"]
            print(f"### {sname} T2 volatility direction (next 20d vs last 20d; base rate higher {f(blk['base_rate_up'])})\n")
            print(dir_table(blk, 20, bases) + "\n")
        for h, blk in s["T4"].items():
            print(f"### {sname} T4 log-price quantile score, h={h}\n")
            print(t4_table(blk) + "\n")
    print("## World-model test (wti)\n")
    print(wm_test(S, [tuple(x.split(",")) for x in a.pair]) + "\n")
    print(f"## Does {a.candidate} beat every non-contaminated, non-Alma model? (primary metrics, prereg section 7)\n")
    print(beats_all(S, a.candidate) + "\n")


if __name__ == "__main__":
    main()
