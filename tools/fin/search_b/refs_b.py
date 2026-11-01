"""Lead reference models on B's DEV-early inner folds (E1..E3), through harness.run on the crypto panel CUT at
2024-06-30 (nothing later in memory), fitting cut at each fold's fit_end via data.TRAIN_END patch.
Logged to runs/fin/search_b/results_b.tsv as kind 'ref'. Preds saved for paired comparisons.

  nice -n 10 .venv/bin/python tools/fin/search_b/refs_b.py [--models har,lgbm]
"""
import os, sys, time, argparse, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "search"))
import common as C     # noqa
import harness as H    # noqa
import feats_b as FB   # noqa
import loop_b as LB    # noqa

FC = ["majority", "rv_persist", "har", "harx", "har_roll", "lgbm", "lgbm_asset", "garch"]
AG = ["buy_hold", "tsmom_1", "tsmom_7", "tsmom_30", "meanrev_1", "voltgt_tsmom", "lgbm_policy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(FC + AG))
    ap.add_argument("--folds", default="E1,E2,E3")
    a = ap.parse_args()
    import models as M
    P = FB.loop_panel()
    import copy

    def tv(market="crypto", group="train"):
        """Same as models.train_view (cut at data.TRAIN_END, group filter) but built from the in-memory panel that
        is already cut at 2024-06-30 (avoids loading the full panel a second time: memory rule)."""
        assert market == "crypto"
        Q = copy.deepcopy(P)
        keep = [s for s in Q.symbols if group is None or Q.group[s] == group]
        return C.cut_panel(Q, C.DT.TRAIN_END, keep)
    M.train_view = tv
    for mn in a.models.split(","):
        accs = []
        for fk in a.folds.split(","):
            fe, d0, d1 = LB.FOLDS[fk]
            row = {"kind": "ref", "name": mn, "config": f"lead factory, fit_end {fe}", "seed": 0, "fold": fk}
            try:
                t0 = time.time()
                with C.fit_end_patch(fe):
                    model = M.get(mn)
                days = C.bt_days(P, d0, d1)
                if mn in AG:
                    days = days[days < len(P.days) - 1]          # exit fill must lie inside the cut panel
                res = H.run(P, model, days, online=False)
                np.savez(os.path.join(FB.OUTB, "preds", f"ref_{mn}_{fk}.npz"),
                         **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
                if mn in AG:
                    s = C.p2(P, res)
                    row.update(SR_heldout=s["heldout"]["sharpe"], SR_all=s["all"]["sharpe"],
                               turn_heldout=s["heldout"]["turnover"], mdd_heldout=s["heldout"]["max_drawdown"])
                else:
                    s = C.p1_p3(P, res)
                    row.update(P1_all=s["all"]["acc"], P1_train=s["train"]["acc"], P1_heldout=s["heldout"]["acc"],
                               P3_all=s["all"]["qs"])
                    accs.append(s["all"]["acc"])
                row.update(status="ok", note=f"{time.time() - t0:.0f}s")
            except Exception as ex:
                traceback.print_exc()
                row.update(status="crash", note=f"{type(ex).__name__}: {str(ex)[:200]}")
            LB.log_row(row)
            print({k: v for k, v in row.items() if k != "config"}, flush=True)
        if accs:
            LB.log_row({"kind": "ref", "name": mn, "config": "lead factory", "seed": 0, "fold": "EM", "status": "ok",
                        "P1_all": float(np.mean(accs))})


if __name__ == "__main__":
    main()
