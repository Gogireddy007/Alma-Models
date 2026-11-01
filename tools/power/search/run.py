"""One search run: train a power world model (or its twin) with a fixed step budget on target days <= 2023-12-31,
score it on the DEV rolling-origin backtest 2024-01-01..2025-06-30 (daily origins, no retraining), append a row to
runs/power/search/results.tsv. Protocol: docs/power_search.md section 0.

    nice -n 10 .venv/bin/python tools/power/search/run.py --name r1_cycle_none --arch cycle --abl none --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import wm_data as W            # noqa: E402
import wm_models as M          # noqa: E402

ROOT = W.ROOT
OUT = os.path.join(ROOT, "runs", "power", "search")
TSV = os.path.join(OUT, "results.tsv")
BT = ("2024-01-01", "2025-06-30")
TRAIN_END = "2023-12-31"
LV9 = slice(2, 11)                # 0.1..0.9 in the 13-level grid
COLS = ["name", "arch", "abl", "seed", "d", "L", "steps", "lr", "aux", "train_end", "n_params", "S", "P1", "P2",
        "QSskill", "QS", "MAE", "cov80", "sec", "status", "note"]


def naive_epf(D, idx):
    """Harness definition (docs/power_prereg.md s.5): naive_w on Mon/Sat/Sun else naive_d; quantiles = point +
    empirical quantiles of the same-slot 1-day (7-day) differences over the last 364 context days."""
    out = np.zeros((len(idx), M.NQ, 24), np.float32)
    ql = M.QL.numpy()
    for n, i in enumerate(idx):
        wd = D.days[i].weekday()
        lag = 7 if wd in (0, 5, 6) else 1
        lo = max(lag, i - 364)
        diffs = D.P[lo:i] - D.P[lo - lag:i - lag]                 # days lo..i-1 (all <= D-1)
        out[n] = D.P[i - lag][None] + np.nanquantile(diffs, ql, axis=0)
    return out


def score(D, idx, q):
    """q [n,13,24] EUR. Returns dict with S and the components (prereg section 4 definitions)."""
    y, y1 = D.P[idx], D.P[idx - 1]
    med = q[:, 6]
    # P1: daily direction; prediction tie -> DEV majority (down)
    lab = y.mean(1) > y1.mean(1)
    pm = med.mean(1)
    pred = pm > y1.mean(1)
    tie_l = y.mean(1) == y1.mean(1)
    p1 = (pred == lab)[~tie_l].mean()
    labh = y > y1
    predh = med > y1
    tie_h = y == y1
    p2 = (predh == labh)[~tie_h].mean()
    ql = M.QL.numpy()[LV9]
    err = y[:, None] - q[:, LV9]
    qs = np.maximum(ql[None, :, None] * err, (ql[None, :, None] - 1) * err).mean()
    ref = naive_epf(D, idx)
    erf = y[:, None] - ref[:, LV9]
    qs_ref = np.maximum(ql[None, :, None] * erf, (ql[None, :, None] - 1) * erf).mean()
    skill = 1 - qs / qs_ref
    S = 0.35 * p1 + 0.35 * p2 + 0.30 * skill
    cov = ((y >= q[:, 2]) & (y <= q[:, 10])).mean()
    return {"S": float(S), "P1": float(p1), "P2": float(p2), "QSskill": float(skill), "QS": float(qs),
            "MAE": float(np.abs(med - y).mean()), "cov80": float(cov)}


def predict(model, D, idx, L, chunk=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(idx), chunk):
            b, mu, sd, _ = W.batch(D, idx[k:k + chunk], L, with_target=False)
            qz = model(b)["q"].numpy()                                  # [n,24,13]
            q = qz * sd[:, None, None] + mu[:, None, None]
            outs.append(np.sort(q, -1).transpose(0, 2, 1))
    return np.concatenate(outs)


def train(model, D, tr, L, steps, lr, seed, aux=0.3, batch=64, wd=1e-4, dirw=0.0):
    g = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    last = []
    for s in range(steps):
        ii = g.choice(tr, batch)
        b, mu, sd, t = W.batch(D, ii, L)
        out = model(b)
        q = out["q"]
        loss = M.pinball(q, t["y"])
        if "rl" in out and aux > 0:
            loss = loss + aux * ((out["rl"] - t["rl"]) ** 2).mean()
        if dirw > 0:              # soft direction loss on the hourly median vs same hour of d-1
            y1 = b["zp"][:, -1]
            lab = (t["y"] > y1).float()
            loss = loss + dirw * torch.nn.functional.binary_cross_entropy_with_logits(4 * (q[..., M.MED] - y1), lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last.append(loss.item())
        if not np.isfinite(last[-1]):
            raise FloatingPointError("loss NaN")
    return float(np.mean(last[-100:]))


def append(row):
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in COLS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="cycle")
    ap.add_argument("--abl", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=14)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aux", type=float, default=0.3)
    ap.add_argument("--dirw", type=float, default=0.0)
    ap.add_argument("--train_end", default=TRAIN_END)
    ap.add_argument("--note", default="")
    ap.add_argument("--data", default="smard", help="smard (search) | lead (final refit on tools/power/data.train_panel)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ALMA_THREADS", 2)))  # ALMA_THREADS=1: bit-reproducible across machines
    if os.environ.get("ALMA_THREADS") == "1":
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    t0 = time.time()
    row = dict(name=a.name, arch=a.arch, abl=a.abl, seed=a.seed, d=a.d, L=a.L, steps=a.steps, lr=a.lr, aux=a.aux,
               train_end=a.train_end, note=a.note + (f" dirw={a.dirw}" if a.dirw else ""))
    try:
        D = W.PowerData("dev") if a.data == "smard" else W.LeadData()
        assert a.train_end <= "2025-06-30"
        tr = W.valid_days(D, "2018-10-01", a.train_end, a.L)
        model = M.build(a.arch, a.abl, a.d, a.L)
        row["n_params"] = sum(p.numel() for p in model.parameters())
        train(model, D, tr, a.L, a.steps, a.lr, a.seed, a.aux, dirw=a.dirw)
        os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(OUT, "ckpt", a.name + ".pt"))
        if a.train_end <= "2023-12-31":
            bt = W.valid_days(D, *BT, a.L)
            q = predict(model, D, bt, a.L)
            os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
            np.savez_compressed(os.path.join(OUT, "pred", a.name + ".npz"), q=q, idx=bt)
            row.update(score(D, bt, q))
            row["status"] = "ok"
        else:
            row["status"] = "refit(no DEV score: in-sample)"
    except Exception as e:                       # failures are logged too
        row["status"] = "crash:" + type(e).__name__ + ":" + str(e)[:80].replace("\t", " ")
        import traceback
        traceback.print_exc()
    row["sec"] = round(time.time() - t0)
    for k in ("S", "P1", "P2", "QSskill", "QS", "MAE", "cov80"):
        if k in row:
            row[k] = round(row[k], 4)
    append(row)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
