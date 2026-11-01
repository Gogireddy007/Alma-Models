"""Architect B finalist scoring on the DEV selection holdout F1 (Am1 in docs/fin_search.md): used ONCE per finalist.

  fit: training coins, label day <= 2024-06-30 (from the loop3 cache, which is cut at 2024-06-30), seeds 0..n-1
       -> one ensemble checkpoint runs/fin/search_b/ckpt/<name>_F1.pt (cfg + sha in meta)
  score: tools/fin/wm_b_fin.py wrapper through harness.run on the DEV panels (C.dev_panel: cut at 2025-06-30),
       backtest 2024-07-01..2025-06-30, A's metric code (common.p1_p3 / p2). Logged as fold F1.
  --refit: fit on label day <= 2025-06-30 (freeze refit; no scoring, no selection) -> <name>_refit.pt

  nice -n 10 .venv/bin/python tools/fin/search_b/holdout_b.py --name c3_gru --cfg '{...}' --seeds 3 [--cl lr] [--agent k]
"""
import os
import sys
import json
import time
import argparse
import hashlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "search"))
sys.path.insert(0, os.path.dirname(HERE))
import feats_b as FB     # noqa: E402
import loop_b as LB      # noqa: E402
import models_b as MB    # noqa: E402

CKB = os.path.join(FB.OUTB, "ckpt")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fit(name, cfg, seeds, D, fit_end, tag):
    t0 = time.time()
    ms = [LB.train_model(D, cfg, fit_end, 1000 * s, log=lambda *a: None) for s in range(seeds)]
    meta = {"name": name, "seeds": [1000 * s for s in range(seeds)], "fit_end": fit_end, "params": MB.nparams(ms[0]),
            "train_seconds": round(time.time() - t0, 1), "train_coins": 16, "cache": "loop4" if tag == "F1" else "refit4_2025-06-30"}
    os.makedirs(CKB, exist_ok=True)
    p = os.path.join(CKB, f"{name}_{tag}.pt")
    torch.manual_seed(0)
    MB.save(p, ms, cfg, meta)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cl", default="")              # online rule lr for the continual pillar
    ap.add_argument("--agent", default="")           # json voltgt rule for the agent pillar
    ap.add_argument("--ckpt", default="")            # reuse an F1 checkpoint (no refit)
    ap.add_argument("--refit", action="store_true")  # freeze refit on labels <= 2025-06-30 (no scoring)
    a = ap.parse_args()
    torch.set_num_threads(3)
    cfg = json.loads(a.cfg)
    if a.refit:
        D = FB.build_cache(end="2025-06-30")
        ck = fit(a.name, cfg, a.seeds, D, "2025-06-30", "refit")
        LB.log_row({"kind": "wm_b", "name": a.name, "config": cfg, "seed": "0..%d" % (a.seeds - 1), "fold": "refit",
                    "status": "trained", "note": f"refit, not scored; ckpt {os.path.relpath(ck, FB.ROOT)} sha {sha(ck)}"})
        print("refit", ck, sha(ck))
        return
    import common as C
    import harness as H
    import wm_b_fin as WB
    if a.ckpt:
        ck = a.ckpt
    else:
        D = LB.cache()
        ck = fit(a.name, cfg, a.seeds, D, "2024-06-30", "F1")
        del D
        LB._D = None
    rel = os.path.relpath(ck, FB.ROOT)
    panels = {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}
    row = {"kind": "wm_b", "name": a.name, "config": cfg, "seed": "0..%d" % (a.seeds - 1), "fold": "F1",
           "status": "ok"}
    note = f"ckpt {rel} sha {sha(ck)}"
    if a.agent:
        rule = json.loads(a.agent)
        row["kind"] = "agent_b"
        for mk, P in panels.items():
            ag = WB.BVolTgt(WB.BForecaster(*WB.load_ckpt(rel)), rule)
            days = C.bt_days(P)
            days = days[days < len(P.days) - 1]
            res = H.run(P, ag, days, online=False)
            s = C.p2(P, res)
            if mk == "crypto":
                row.update(SR_heldout=s["heldout"]["sharpe"], SR_all=s["all"]["sharpe"],
                           turn_heldout=s["heldout"]["turnover"], mdd_heldout=s["heldout"]["max_drawdown"])
                note += f" sr_period {s['heldout']['sr_period']:.5f} rule {json.dumps(rule)}"
            else:
                row["SR_etf"] = s["all"]["sharpe"]
    else:
        for mk, P in panels.items():
            if a.cl:
                fc = WB.BContinual(*WB.load_ckpt(rel), rule={"lr": float(a.cl), "lr_q": 0.0})
                row["kind"] = "cl_b"
            else:
                fc = WB.BForecaster(*WB.load_ckpt(rel))
            res = H.run(P, fc, C.bt_days(P), online=bool(a.cl) and mk == "crypto")
            C.save_preds(f"b_{a.name}{'_cl' + a.cl if a.cl else ''}_F1_{mk}", res)
            s = C.p1_p3(P, res)
            if mk == "crypto":
                row.update(P1_all=s["all"]["acc"], P1_train=s["train"]["acc"], P1_heldout=s["heldout"]["acc"],
                           P3_all=s["all"]["qs"])
            else:
                row.update(E1_etf=s["all"]["acc"], E3_etf=s["all"]["qs"])
        if a.cl:
            note += f" online lr {a.cl}"
    row["note"] = note
    LB.log_row(row)
    print({k: v for k, v in row.items() if k != "config"}, flush=True)


if __name__ == "__main__":
    main()
