"""Alma's pretrained SSM world model (and its direct twin) on DE-LU hourly prices: zero-shot (--steps 0) or
fine-tuned on hourly windows whose last target hour is <= 2023-12-31 23:00 local. Scored with the same DEV
backtest / score as run.py (levels outside 0.1..0.9 are linearly extrapolated for the 13-level array; the score
uses 0.1..0.9 only)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(ROOT)
import importlib.util                # noqa: E402
import wm_data as W                 # noqa: E402
from run import score, append, BT, OUT   # noqa: E402
_spec = importlib.util.spec_from_file_location("commodity_wm_alma", os.path.join(ROOT, "tools", "commodity", "wm_alma.py"))
_wa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wa)
AlmaSSM, CONTEXT, MAX_H = _wa.AlmaSSM, _wa.CONTEXT, _wa.MAX_H

Q9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def to13(q9):
    """[n,9,24] -> [n,13,24]: 0.025 / 0.05 and 0.95 / 0.975 linearly extrapolated from the 0.1-0.2 / 0.8-0.9 gaps
    (not scored by the search; the harness wrapper returns NaN there)."""
    dl, dh = q9[:, 1] - q9[:, 0], q9[:, 8] - q9[:, 7]
    return np.concatenate([(q9[:, 0] - 1.5 * dl)[:, None], (q9[:, 0] - 1.0 * dl)[:, None], q9,
                           (q9[:, 8] + 1.0 * dh)[:, None], (q9[:, 8] + 1.5 * dh)[:, None]], 1)


def alma_predict(model, D, idx, chunk=128):
    flat = D.P.reshape(-1)
    out = []
    for k in range(0, len(idx), chunk):
        ii = idx[k:k + chunk]
        ctx = np.stack([flat[i * 24 - CONTEXT:i * 24] for i in ii])      # hours up to slot 23 of D-1
        x = torch.tensor(ctx, dtype=torch.float32)[..., None]
        with torch.no_grad():
            dq = model.quantiles(x, 24).numpy()                          # [n,9,24] change from last value
        out.append(np.sort(dq + ctx[:, -1][:, None, None], 1))
    return to13(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--abl", default="none")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_orig", type=int, default=96)
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    t0 = time.time()
    row = dict(name=a.name, arch="alma", abl=a.abl, seed=a.seed, d=96, L=CONTEXT // 24, steps=a.steps, lr=a.lr,
               train_end="2023-12-31", note="pretrained" + (" zero-shot" if a.steps == 0 else " fine-tuned"))
    try:
        D = W.PowerData("dev")
        model = AlmaSSM(abl=a.abl, season=24, n_orig=a.n_orig)
        row["n_params"] = model.n_params()
        if a.steps:
            flat = torch.tensor(D.P.reshape(-1))
            last = D.di[__import__("datetime").date(2023, 12, 31)] * 24 + 24      # exclusive hour index
            n = CONTEXT + MAX_H
            g = np.random.default_rng(a.seed)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            model.train()
            for s in range(a.steps):
                st = g.integers(0, last - n, a.batch)
                w = torch.stack([flat[t:t + n] for t in st])[..., None]
                assert st.max() + n <= last
                loss, _ = model.loss(w, CONTEXT, MAX_H)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(OUT, "ckpt", a.name + ".pt"))
        bt = W.valid_days(D, *BT, 14)
        q = alma_predict(model, D, bt)
        np.savez_compressed(os.path.join(OUT, "pred", a.name + ".npz"), q=q, idx=bt)
        row.update(score(D, bt, q))
        row["status"] = "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        row["status"] = "crash:" + type(e).__name__ + ":" + str(e)[:80]
    row["sec"] = round(time.time() - t0)
    for k in ("S", "P1", "P2", "QSskill", "QS", "MAE", "cov80"):
        if k in row:
            row[k] = round(row[k], 4)
    append(row)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
