"""Paired comparisons on the DEV fold from saved predictions (runs/fin/search/preds/<name>_<market>.npz).

  .venv/bin/python tools/fin/search/compare.py --pairs candA:twinA,candB:twinB [--out runs/fin/search/cmp.json]
Forecasters: P1 day-level accuracy difference (DM-HLN h=1 p, 7-day block-bootstrap 95% CI) and P3 daily QS
difference (same tests), groups all / train / heldout (crypto) and all (etf). Agents: net Sharpe difference
(paired 7-day block bootstrap, prereg 6.3), groups heldout / all (crypto) and all (etf).
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import harness as H      # noqa: E402
import metrics as MT     # noqa: E402

PANELS = {}


def panel(mk):
    if mk not in PANELS:
        PANELS[mk] = C.dev_panel(mk)
    return PANELS[mk]


def load(name, mk):
    p = os.path.join(C.OUT, "preds", f"{name}_{mk}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    r = {k: z[k] for k in z.files}
    r["symbols"] = r["symbols"].tolist()
    return r


def cmp_pair(a, b, mk):
    P = panel(mk)
    ra, rb = load(a, mk), load(b, mk)
    if ra is None or rb is None:
        return None
    out = {}
    if "pos" in ra:
        groups = ["heldout", "all"] if mk == "crypto" else ["all"]
        sa, sb = C.p2(P, ra), C.p2(P, rb)
        for g in groups:
            d, lo, hi, p = MT.sharpe_diff_test(sa[g]["net"], sb[g]["net"], P.periods)
            out[g] = {"SR_a": sa[g]["sharpe"], "SR_b": sb[g]["sharpe"], "diff": d, "ci": [lo, hi], "p": p,
                      "turn_a": sa[g]["turnover"], "turn_b": sb[g]["turnover"]}
        return out
    sa, sb = C.p1_p3(P, ra), C.p1_p3(P, rb)
    for g in sa:
        d = sa[g]["day_acc"] - sb[g]["day_acc"]
        d = d[np.isfinite(d)]
        _, _, p = MT.dm_hln(-d, 1)
        o = {"P1_a": sa[g]["acc"], "P1_b": sb[g]["acc"], "dP1": float(d.mean()), "dP1_ci": list(MT.boot_mean_ci(d)),
             "dP1_p_dm": float(p)}
        if sa[g]["day_qs"] is not None and sb[g]["day_qs"] is not None:
            dq = sa[g]["day_qs"] - sb[g]["day_qs"]
            dq = dq[np.isfinite(dq)]
            _, _, pq = MT.dm_hln(dq, 1)
            o.update(QS_a=sa[g]["qs"], QS_b=sb[g]["qs"], dQS=float(dq.mean()), dQS_ci=list(MT.boot_mean_ci(dq)),
                     dQS_p_dm=float(pq))
        out[g] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for pr in a.pairs.split(","):
        x, y = pr.split(":")
        for mk in ("crypto", "etf"):
            r = cmp_pair(x, y, mk)
            if r is not None:
                res[f"{x} vs {y} [{mk}]"] = r
                for g, o in r.items():
                    if "dP1" in o:
                        print(f"{x} vs {y} [{mk}/{g}]: P1 {o['P1_a']:.4f} vs {o['P1_b']:.4f} d {o['dP1']:+.4f} "
                              f"CI [{o['dP1_ci'][0]:+.4f},{o['dP1_ci'][1]:+.4f}] p {o['dP1_p_dm']:.3f}"
                              + (f" | QS {o['QS_a']:.4f} vs {o['QS_b']:.4f} d {o['dQS']:+.4f} CI "
                                 f"[{o['dQS_ci'][0]:+.4f},{o['dQS_ci'][1]:+.4f}] p {o['dQS_p_dm']:.3f}" if "dQS" in o else ""))
                    else:
                        print(f"{x} vs {y} [{mk}/{g}]: SR {o['SR_a']:.3f} vs {o['SR_b']:.3f} d {o['diff']:+.3f} "
                              f"CI [{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}] p {o['p']:.3f} turn {o['turn_a']:.3f}/{o['turn_b']:.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
