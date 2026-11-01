"""One round-2 autoresearch run: train a pooled world model (or its twin) for a FIXED number of steps on the POOL5
products with every training target dated before 2015-01-01, forecast the DEV backtest origins (2015-01-01 ..
2025-06-30, intersection calendar, every trading day), save checkpoint + forecasts, append one row to
runs/commodity/r2/results.tsv (failures included).

    nice -n 10 .venv/bin/python tools/commodity/r2/train_r2.py --name svx_cal_none_s0 --arch svx --abl none \
        --kw '{"cal":1,"exo":0}' --seed 0 --max_steps 800

The run-level scalar (keep/discard guide only; selection uses score_r2.py): pooled vol-direction accuracy, mean of
h = 5 and h = 10.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import torch                  # noqa: E402

import core_r2 as R           # noqa: E402
import data_r2 as D           # noqa: E402
import models_r2 as MR        # noqa: E402

TSV = os.path.join(R.OUT, "results.tsv")
COLS = ["ts", "name", "arch", "abl", "kw", "seed", "params", "steps", "train_s", "vol5", "vol10", "scalar",
        "crps_skill_ho5", "crps_skill_jet5", "status", "sha12", "note"]


def train(model, S, L, steps, lr=2e-3, batch=64, seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sids = list(S)
    orig = {s: S[s].train_origins(D.TRAIN_CUT, L) for s in sids}
    log("train origins: " + ", ".join(f"{s}:{len(o)} last target {str(S[s].dates[o[-1] + R.H])[:10]}"
                                      for s, o in orig.items()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    ema, t0 = None, time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        s = sids[rng.integers(len(sids))]
        o = orig[s][rng.integers(0, len(orig[s]), batch)]
        idx = o[:, None] + np.arange(-L, R.H + 1)[None]
        w = torch.from_numpy(S[s].lp[idx][:, :, None]).float()
        ex = torch.from_numpy(S[s].ex[idx]).float()
        loss, info = model.loss(w, L, R.H, ex)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % 200 == 0:
            log(f"  step {step + 1} t={time.time() - t0:.0f}s loss {ema:.4f} " +
                " ".join(f"{k}={v:.3f}" for k, v in info.items()))
    model.eval()
    return time.time() - t0


@torch.no_grad()
def forecast(model, S, odates, L, chunk=256, seed=1234):
    """{sid: (t [n], q [n, 11, H])}; deterministic for a given origin list (seed per series and chunk)."""
    out = {}
    for si, s in enumerate(D.POOL5 if all(x in S for x in D.POOL5) else list(S)):
        ser = S[s]
        t = R.t_index(ser, odates)
        assert (t >= L).all()
        qs = []
        for c0 in range(0, len(t), chunk):
            tt = t[c0:c0 + chunk]
            torch.manual_seed(seed + 7919 * si + c0)
            idx = tt[:, None] + np.arange(-L, 1)[None]
            lp = torch.from_numpy(ser.lp[idx][:, :, None]).float()
            ex = torch.from_numpy(ser.ex[idx]).float()
            calf = torch.from_numpy(np.stack([R.future_cal(ser.dates[i]) for i in tt])).float()
            qs.append(model.quantiles_x(lp, ex, calf, R.H).numpy().astype(np.float64))
        out[s] = (t, np.concatenate(qs))
    return out


def quick_scalar(S, fc):
    """pooled vol accuracy at h = 5, 10 and CRPS skill vs rw_gauss at h = 5 (HO, jet) on DEV."""
    res = {}
    for h in R.HS:
        hits = []
        for s, (t, q) in fc.items():
            lp = S[s].lp
            ok_t = t + h < len(lp)
            tt, qq = t[ok_t], q[ok_t]
            y, ok = R.vol_actual(lp, tt, h)
            p = R.vol_class_from_q(qq, lp, tt, h)
            hits.append((p == y)[ok])
        res[f"vol{h}"] = float(np.concatenate(hits).mean())
    P = R.fit_params(S)
    for s, key in (("DHOILNYH", "ho5"), ("DJFUELUSGULF", "jet5")):
        t, q = fc[s]
        lp = S[s].lp
        ok = t + 5 < len(lp)
        qb, _ = R.baselines(S[s], t[ok], P)
        res[f"crps_skill_{key}"] = float(1 - R.qs(q[ok], lp, t[ok], 5).mean() / R.qs(qb["rw_gauss"], lp, t[ok], 5).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--abl", default="none")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    kw = json.loads(a.kw)
    row = {k: "" for k in COLS}
    row.update(ts=time.strftime("%Y-%m-%d %H:%M"), name=a.name, arch=a.arch, abl=a.abl, kw=json.dumps(kw),
               seed=a.seed, note=a.note)
    try:
        S = R.load("dev")
        torch.manual_seed(a.seed)
        model = MR.build(a.arch, a.abl, kw)
        row["params"] = model.n_params()
        secs = train(model, S, a.L, a.max_steps, lr=a.lr, seed=a.seed)
        row.update(steps=a.max_steps, train_s=round(secs))
        os.makedirs(os.path.join(R.OUT, "ckpt"), exist_ok=True)
        ck = os.path.join(R.OUT, "ckpt", f"{a.name}.pt")
        torch.save({"state": model.state_dict(), "arch": a.arch, "abl": a.abl, "kw": kw, "L": a.L,
                    "steps": a.max_steps, "seed": a.seed, "train_cut": str(D.TRAIN_CUT.date()),
                    "series": D.POOL5, "complete": True}, ck)
        row["sha12"] = hashlib.sha256(open(ck, "rb").read()).hexdigest()[:12]
        od = R.origin_dates(S, "dev")
        fc = forecast(model, S, od, a.L)
        os.makedirs(os.path.join(R.OUT, "pred", "dev"), exist_ok=True)
        np.savez_compressed(os.path.join(R.OUT, "pred", "dev", f"{a.name}.npz"), odates=od.astype("datetime64[D]"),
                            **{f"{s}_t": v[0] for s, v in fc.items()}, **{f"{s}_q": v[1] for s, v in fc.items()})
        r = quick_scalar(S, fc)
        row.update({k: round(v, 4) for k, v in r.items()}, scalar=round((r["vol5"] + r["vol10"]) / 2, 4),
                   status="ok")
        print(json.dumps(r))
    except Exception as e:
        traceback.print_exc()
        row.update(status="crash", note=(a.note + " " + repr(e)[:150]).strip())
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(row[k]).replace("\t", " ") for k in COLS) + "\n")
    print("RESULT", row["name"], row["status"], row["scalar"])


if __name__ == "__main__":
    main()
