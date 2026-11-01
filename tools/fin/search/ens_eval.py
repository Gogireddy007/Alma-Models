"""Seed-ensemble forecasters on the DEV fold (forecasts: mean p_up, mean quantiles re-sorted), logged as kind 'ens'.

  nice -n 10 .venv/bin/python tools/fin/search/ens_eval.py --name wm_fin_ens --wm a.pt,b.pt,c.pt
"""
import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wm", required=True)
    a = ap.parse_args()
    torch.set_num_threads(3)
    cks = a.wm.split(",")
    members = []
    for ck in cks:
        members += WC.load_members(ck)[0]
    row = {"kind": "ens", "name": a.name, "config": {"wm": [os.path.basename(c) for c in cks]}, "seed": "ens",
           "fold": "F1"}
    for mk in ("crypto", "etf"):
        P = C.dev_panel(mk)
        res = H.run(P, WF.WMForecaster(members, {}), C.bt_days(P), online=False)
        C.save_preds(f"{a.name}_{mk}", res)
        s = C.p1_p3(P, res)
        if mk == "crypto":
            row.update(P1_all=s["all"]["acc"], P1_train=s["train"]["acc"], P1_heldout=s["heldout"]["acc"],
                       P3_all=s["all"]["qs"])
        else:
            row.update(E1_etf=s["all"]["acc"], E3_etf=s["all"]["qs"])
    row["status"] = "ok"
    C.log_row(row)
    print(json.dumps({k: v for k, v in row.items() if k != "config"}, default=str))


if __name__ == "__main__":
    main()
