"""Reference models on the walk-forward DEV fold (fit <= 2024-06-30 via data.TRAIN_END patch; backtest
2024-07-01..2025-06-30), through harness.run on DEV-cut panels. Logged to results.tsv as kind 'ref'.

  nice -n 10 .venv/bin/python tools/fin/search/refs.py [--models a,b]
"""
import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import harness as H      # noqa: E402

FC = ["majority", "rv_persist", "har", "harx", "har_roll", "lgbm", "lgbm_asset", "garch"]
AG = ["buy_hold", "tsmom_1", "tsmom_7", "tsmom_30", "meanrev_1", "voltgt_tsmom", "lgbm_policy"]
ETF_SKIP = {"harx"}
ETF_ALIAS = {"lgbm_asset": "lgbm_asset_etf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(FC + AG))
    a = ap.parse_args()
    import models as M
    panels = {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}
    for mn in a.models.split(","):
        row = {"kind": "ref", "name": mn, "config": "lead factory, fit_end 2024-06-30", "seed": 0, "fold": "F1"}
        try:
            t0 = time.time()
            for mk, P in panels.items():
                if mk == "etf" and mn in ETF_SKIP:
                    continue
                with C.fit_end_patch(C.FOLD_FIT_END):
                    model = M.get(ETF_ALIAS.get(mn, mn) if mk == "etf" else mn)
                res = H.run(P, model, C.bt_days(P), online=False)
                C.save_preds(f"ref_{mn}_{mk}", res)
                if mn in AG:
                    s = C.p2(P, res)
                    if mk == "crypto":
                        row.update(SR_heldout=s["heldout"]["sharpe"], SR_all=s["all"]["sharpe"],
                                   turn_heldout=s["heldout"]["turnover"], mdd_heldout=s["heldout"]["max_drawdown"])
                    else:
                        row["SR_etf"] = s["all"]["sharpe"]
                else:
                    s = C.p1_p3(P, res)
                    if mk == "crypto":
                        row.update(P1_all=s["all"]["acc"], P1_train=s["train"]["acc"], P1_heldout=s["heldout"]["acc"],
                                   P3_all=s["all"]["qs"])
                    else:
                        row.update(E1_etf=s["all"]["acc"], E3_etf=s["all"]["qs"])
            row.update(status="ok", note=f"{time.time() - t0:.0f}s")
        except Exception as ex:
            traceback.print_exc()
            row.update(status="crash", note=f"{type(ex).__name__}: {ex}")
        C.log_row(row)
        print({k: v for k, v in row.items() if k != "config"}, flush=True)


if __name__ == "__main__":
    main()
