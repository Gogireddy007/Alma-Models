"""Perturbation leak test for architect B's families (extends tools/fin/search/test_leak.py logic). DEV days only, on
the crypto panel CUT at 2024-06-30 (B's loop rule) and the ETF panel cut at 2024-06-30.

For each checkpoint (one per family) and each test day D:
  deterministic   two runs -> bit-identical
  future          every daily value of days >= D (incl. fill(D)) and every hourly value of hours >= D 00:00 x U(0.3,3)
                  -> output at D bit-identical
  fill            fill(D) and the D 00:00 hourly bar -> bit-identical
  prev_day / prev_hour  perturbing day D-1 / hour D-1 23:00 -> output changes (sensitivity)
  cache parity    wrapper output == the loop's cached-feature prediction (the loop scores what the wrapper does)
Continual (wm_b_fin_cl): 12-day walk with a perturbation from day j on -> decisions <= j bit-identical, later differ.
Agents (wm_b_voltgt): same invariance checks on positions.

  nice -n 10 .venv/bin/python tools/fin/search_b/test_leak_b.py --ckpts a.pt,b.pt [--cl a.pt] [--agent a.pt]
"""
import os
import sys
import copy
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "search"))
sys.path.insert(0, os.path.dirname(HERE))
import common as C       # noqa: E402
import harness as H      # noqa: E402
import test_leak as TL   # noqa: E402  (A's perturbation logic, reused)
import feats_b as FB     # noqa: E402
import loop_b as LB      # noqa: E402
import wm_b_fin as WB    # noqa: E402

DAYS = ["2023-03-06", "2023-09-18", "2024-02-12", "2024-06-03", "2024-06-30"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--cl", default="")
    ap.add_argument("--agent", default="")
    ap.add_argument("--out", default=os.path.join(FB.OUTB, "test_leak_b.txt"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    panels = {"crypto": FB.loop_panel(), "etf": C.cut_panel(H.Panel("etf"), FB.LOOP_END)}
    lines, ok_all = [], True

    def rep(s, ok=True):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(("OK   " if ok else "FAIL ") + s)
        print(lines[-1], flush=True)

    makers = [(os.path.basename(c), lambda c=c: WB.BForecaster(*WB.load_ckpt(c))) for c in a.ckpts.split(",") if c]
    makers += [(os.path.basename(c) + ":voltgt", lambda c=c: WB.BVolTgt(WB.BForecaster(*WB.load_ckpt(c))))
               for c in a.agent.split(",") if c]
    D = LB.cache()
    for mk, P in panels.items():
        idx = [min(int(np.searchsorted(P.days, np.datetime64(d, "ns"))), len(P.days) - 1) for d in DAYS]
        for mn, make in makers:
            for i in idx:
                o0 = TL.out_of(H.run(P, make(), [i], online=False))
                o0b = TL.out_of(H.run(P, make(), [i], online=False))
                tag = f"{mk} {mn} {str(P.days[i])[:10]}"
                rep(f"{tag} deterministic", np.array_equal(o0, o0b, equal_nan=True))
                for mode in ("future", "fill") + (("prev_day", "prev_hour") if mk == "crypto" else ("prev_day",)):
                    Q = TL.perturb(P, i, mode, rng)
                    o1 = TL.out_of(H.run(Q, make(), [i], online=False))
                    same = np.array_equal(o0, o1, equal_nan=True)
                    if mode in ("future", "fill"):
                        rep(f"{tag} {mode}: unchanged", same)
                    else:
                        rep(f"{tag} {mode}: changed ({int((~np.isclose(o0, o1, equal_nan=True)).sum())} values)",
                            not same)
                    del Q
                if mk == "crypto" and ":" not in mn:
                    k = int(np.searchsorted(D["days"], P.days[i]))
                    if k < len(D["days"]) and D["days"][k] == P.days[i]:
                        fc = make()
                        pc, qc = LB.predict(fc.members, D, np.array([k]))
                        r = H.run(P, fc, [i], online=False)
                        rep(f"{tag} cache parity (p max diff {np.nanmax(np.abs(pc[0] - r['p_up'][0])):.2e})",
                            np.allclose(pc[0], r["p_up"][0], atol=1e-5, equal_nan=True))
        for c in [x for x in a.cl.split(",") if x]:
            i0 = idx[2]
            days = list(range(i0, i0 + 12))
            mkcl = lambda: WB.BContinual(*WB.load_ckpt(c))
            r0 = H.run(P, mkcl(), days)
            j = i0 + 6
            Q = TL.perturb(P, j, "future", rng)
            r1 = H.run(Q, mkcl(), days)
            o0, o1 = TL.out_of(r0), TL.out_of(r1)
            k = days.index(j)
            rep(f"{mk} {os.path.basename(c)}:cl online: decisions for days <= {str(P.days[j])[:10]} bit-identical",
                np.array_equal(o0[:k + 1], o1[:k + 1], equal_nan=True))
            rep(f"{mk} {os.path.basename(c)}:cl online: later decisions differ",
                not np.array_equal(o0[k + 1:], o1[k + 1:], equal_nan=True))
            rep(f"{mk} {os.path.basename(c)}:cl updates every day ({r0['updates'].tolist()})",
                bool((r0["updates"] > 0).all()))
    lines.append("ALL OK" if ok_all else "FAILURES")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(lines[-1])
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
