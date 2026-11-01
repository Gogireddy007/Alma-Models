"""Continual-rule rows on the DEV fold: WMContinual(members, rule) through harness.run with observe() (exactly the
VAL/TEST mechanics), compared with the same members frozen. Logged as kind 'cl'.

  nice -n 10 .venv/bin/python tools/fin/search/eval_cl.py --wm ck.pt[,..] --steps 1 --lr 1e-4 [--markets crypto]
"""
import os
import sys
import time
import json
import argparse
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--replay", type=int, default=60)
    ap.add_argument("--markets", default="crypto")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(3)
    members = []
    for ck in a.wm.split(","):
        members += WC.load_members(ck)[0]
    rule = {"steps": a.steps, "lr": a.lr, "l2_init": a.l2, "replay_days": a.replay}
    name = f"wm_cl_S{a.steps}_lr{a.lr:g}{a.tag}"
    row = {"kind": "cl", "name": name, "config": dict(rule, wm=[os.path.basename(c) for c in a.wm.split(",")]),
           "seed": "ens" if len(members) > 1 else 0, "fold": "F1"}
    try:
        t0 = time.time()
        notes = {}
        for mk in a.markets.split(","):
            P = C.dev_panel(mk)
            days = C.bt_days(P)
            r_c = H.run(P, WF.WMContinual(members, {}, rule), days)
            r_f = H.run(P, WF.WMForecaster(members, {}), days, online=False)
            C.save_preds(f"{name}_{mk}", r_c)
            sc, sf = C.p1_p3(P, r_c), C.p1_p3(P, r_f)
            d = sc["all"]["day_acc"] - sf["all"]["day_acc"]
            ci = C.MT.boot_mean_ci(d[np.isfinite(d)])
            if mk == "crypto":
                row.update(P1_all=sc["all"]["acc"], P1_train=sc["train"]["acc"], P1_heldout=sc["heldout"]["acc"],
                           P3_all=sc["all"]["qs"])
            else:
                row.update(E1_etf=sc["all"]["acc"], E3_etf=sc["all"]["qs"])
            notes[mk] = {"frozen_P1": sf["all"]["acc"], "S_C": sc["all"]["acc"] - sf["all"]["acc"],
                         "dP1_ci": [float(x) for x in np.atleast_1d(ci)][-2:], "frozen_QS": sf["all"]["qs"],
                         "dQS": sc["all"]["qs"] - sf["all"]["qs"], "updates": int(r_c["updates"].sum())}
        row.update(status="ok", note=json.dumps(dict(notes, sec=round(time.time() - t0))))
    except Exception as ex:
        traceback.print_exc()
        row.update(status="crash", note=f"{type(ex).__name__}: {ex}")
    C.log_row(row)
    print({k: v for k, v in row.items() if k != "config"}, flush=True)


if __name__ == "__main__":
    main()
