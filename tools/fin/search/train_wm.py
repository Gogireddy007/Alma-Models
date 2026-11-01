"""Train one RSSM member (architect search). DEV only.

  nice -n 10 .venv/bin/python tools/fin/search/train_wm.py --fit_end 2024-06-30 --d 128 --abl none --seed 0 --steps 1500
Fold runs (fit_end 2024-06-30) are scored on the walk-forward backtest 2024-07-01..2025-06-30 and logged to
runs/fin/search/results.tsv. Refit runs (fit_end 2025-06-30) are not scored (--no-eval).
"""
import os
import sys
import time
import json
import argparse
import traceback

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import wm_core as WC     # noqa: E402
import wm_fin as WF      # noqa: E402
import harness as H      # noqa: E402


def train(D, d, abl, seed, steps, bs=256, lr=1e-3, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = WC.RSSM(d=d, abl=abl)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    N = len(D["X"])
    rng = np.random.default_rng(seed)
    T = {k: torch.from_numpy(D[k]) for k in ("X", "aux", "tl", "y", "yr", "blk", "mblk")}
    t0 = time.time()
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, N, bs))
        X = T["X"][idx].clone()
        etf = torch.from_numpy(rng.random(bs) < 0.3)
        X[etf, :, 7:15] = 0.0                                  # daily-only regime (ETF adapter) seen in training
        intra = torch.from_numpy(rng.random(bs) < 0.1)
        X[intra, :, 9:13] = 0.0
        X[intra, :, WC.F_MINTRA] = 0.0
        batch = (X, T["aux"][idx], T["tl"][idx], T["y"][idx], T["yr"][idx], T["blk"][idx], T["mblk"][idx])
        loss, parts = WC.loss_fn(model, batch, gen=g)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 250 == 0 or s == steps - 1:
            log(f"step {s} loss {float(loss):.4f} {json.dumps({k: round(v, 4) for k, v in parts.items()})} {time.time() - t0:.0f}s")
    model.eval()
    return model


def eval_forecaster(fc, name, panels=None, log=print):
    """Walk-forward fold backtest through harness.run on DEV panels (crypto 24 coins + 13 ETFs)."""
    panels = panels or {"crypto": C.dev_panel("crypto"), "etf": C.dev_panel("etf")}
    out = {}
    for mk, P in panels.items():
        days = C.bt_days(P)
        res = H.run(P, fc, days, online=hasattr(fc, "observe"))
        C.save_preds(f"{name}_{mk}", res)
        out[mk] = (C.p1_p3(P, res), res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_end", default="2024-06-30")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--abl", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 3)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    name = f"wm_d{a.d}_{a.abl}_s{a.seed}_{a.fit_end}"
    cfg = {"d": a.d, "abl": a.abl, "steps": a.steps, "bs": 256, "lr": 1e-3, "W": WC.W, "fit_end": a.fit_end}
    fold = "F1" if a.fit_end == "2024-06-30" else "refit"
    try:
        P = C.train_panel(a.fit_end)
        D = C.build_dataset(P, a.fit_end)
        assert int(D["day"].max()) < len(P.days) and P.days[int(D["day"].max())] <= np.datetime64(pd.Timestamp(a.fit_end), "ns")
        del P
        t0 = time.time()
        model = train(D, a.d, a.abl, a.seed, a.steps)
        meta = dict(cfg, seed=a.seed, n_train=int(len(D["X"])), params=WC.nparams(model),
                    train_seconds=round(time.time() - t0, 1), last_label_day=str(a.fit_end))
        ck = os.path.join(C.CK, name + ".pt")
        WC.save_members(ck, [model], meta)
        print(f"saved {ck} params {meta['params']} sha {C.sha(ck)[:12]}", flush=True)
        if a.no_eval:
            C.log_row({"kind": "wm", "name": name, "config": cfg, "seed": a.seed, "fold": fold, "status": "trained",
                       "note": f"refit, not scored; sha {C.sha(ck)}"})
            return
        r = eval_forecaster(WF.WMForecaster([model], meta), name)
        c, e = r["crypto"][0], r["etf"][0]
        C.log_row({"kind": "wm", "name": name, "config": cfg, "seed": a.seed, "fold": fold, "status": "ok",
                   "P1_all": c["all"]["acc"], "P1_train": c["train"]["acc"], "P1_heldout": c["heldout"]["acc"],
                   "P3_all": c["all"]["qs"], "E1_etf": e["all"]["acc"], "E3_etf": e["all"]["qs"],
                   "note": f"params {meta['params']}"})
        print(f"{name}: P1 all {c['all']['acc']:.4f} train {c['train']['acc']:.4f} heldout {c['heldout']['acc']:.4f} "
              f"QS {c['all']['qs']:.4f} | ETF P1 {e['all']['acc']:.4f} QS {e['all']['qs']:.4f}", flush=True)
    except Exception as ex:
        C.log_row({"kind": "wm", "name": name, "config": cfg, "seed": a.seed, "fold": fold, "status": "crash",
                   "note": f"{type(ex).__name__}: {ex}"})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
