"""Train/evaluate the agents on frozen WM members (architect search). DEV only.

  nice -n 10 .venv/bin/python tools/fin/search/train_agents.py --fit_end 2024-06-30 --wm ck1.pt[,ck2.pt,...] \
      --lam 0.5 --algos dreamer,ppo,dqn,voltgt --tag F1_lam0.5 [--no-eval]
One policy per WM member (seed i trains on member i with seed i); the agent = mean position over members.
"""
import os
import sys
import time
import json
import argparse
import traceback
import importlib.util

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402

_spec = importlib.util.spec_from_file_location("fin_agents_rl", os.path.join(C.FIN, "agents", "_rl.py"))
RL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RL)
NAMES = {"dreamer": "wm_agent", "ppo": "ppo", "dqn": "dqn", "voltgt": "wm_voltgt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_end", default="2024-06-30")
    ap.add_argument("--wm", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--algos", default="dreamer,ppo,dqn,voltgt")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 3)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    cks = a.wm.split(",")
    members = []
    for ck in cks:
        ms, meta = WC.load_members(ck)
        assert meta["fit_end"] == a.fit_end, (meta["fit_end"], a.fit_end)
        members += ms
    D = C.build_dataset(None if os.path.exists(os.path.join(C.CACHE, f"ds_{a.fit_end}.npz")) else C.train_panel(a.fit_end),
                        a.fit_end)
    fold = "F1" if a.fit_end == "2024-06-30" else "refit"
    panels = None
    for algo in a.algos.split(","):
        name = f"{NAMES[algo]}_{a.tag}"
        cfg = {"algo": algo, "lam": a.lam, "steps": a.steps, "hor": RL.HOR, "bs": RL.BS, "gamma": RL.GAMMA,
               "wm": [os.path.basename(c) for c in cks], "fit_end": a.fit_end}
        try:
            t0 = time.time()
            if algo == "voltgt":
                agent = WF.WMVolTgt(WF.WMForecaster(members, {"wm": cfg["wm"]}))
            else:
                pols = []
                for seed, m in enumerate(members):
                    S = RL.real_states(m, D)
                    if algo == "dreamer":
                        pols.append(RL.train_dreamer(m, S, a.lam, seed, a.steps))
                    elif algo == "ppo":
                        pols.append(RL.train_ppo(m.d, S, a.lam, seed, a.steps))
                    else:
                        pols.append(RL.train_dqn(m.d, S, a.lam, seed, a.steps))
                meta = dict(cfg, seeds=list(range(len(members))), train_seconds=round(time.time() - t0, 1),
                            wm_sha256=[C.sha(c) for c in cks])
                ck = os.path.join(C.CK, f"{name}.pt")
                RL.save_policies(ck, pols, meta)
                agent = RL.RLAgent(algo, members, pols, meta)
                nparam = sum(v.numel() for k, v in pols[0][("q" if algo == "dqn" else "actor")].items())
                if algo != "dqn":
                    nparam += sum(v.numel() for v in pols[0]["critic"].values())
                cfg["policy_params"] = int(nparam)
                print(f"saved {ck} sha {C.sha(ck)[:12]} params {nparam}", flush=True)
            if a.no_eval:
                C.log_row({"kind": "agent", "name": name, "config": cfg, "seed": "all", "fold": fold,
                           "status": "trained", "note": "refit, not scored"})
                continue
            panels = panels or {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}
            row = {"kind": "agent", "name": name, "config": cfg, "seed": "ens" if len(members) > 1 else 0, "fold": fold}
            for mk, P in panels.items():
                res = H.run(P, agent, C.bt_days(P), online=False)
                C.save_preds(f"{name}_{mk}", res)
                s = C.p2(P, res)
                if mk == "crypto":
                    row.update(SR_heldout=s["heldout"]["sharpe"], SR_all=s["all"]["sharpe"],
                               turn_heldout=s["heldout"]["turnover"], mdd_heldout=s["heldout"]["max_drawdown"],
                               note=json.dumps({"sr_period_heldout": s["heldout"]["sr_period"],
                                                "mean_abs_pos": float(np.abs(res["pos"]).mean()),
                                                "sr_x2_heldout": C.p2(P, res, 2.0)["heldout"]["sharpe"],
                                                "sec": round(time.time() - t0)}))
                else:
                    row["SR_etf"] = s["all"]["sharpe"]
            row["status"] = "ok"
            C.log_row(row)
            print({k: v for k, v in row.items() if k != "config"}, flush=True)
        except Exception as ex:
            traceback.print_exc()
            C.log_row({"kind": "agent", "name": name, "config": cfg, "seed": "?", "fold": fold, "status": "crash",
                       "note": f"{type(ex).__name__}: {ex}"})


if __name__ == "__main__":
    main()
