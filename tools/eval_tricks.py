"""
tools/eval_tricks.py: score inference-time tricks (wm/infer_tricks.py) on a frozen Alma checkpoint, VAL only.

    # full VAL (prints val_wql_rel / family_wql_rel, so tools/jobq.py marks the job done)
    ALMA_TRICK='sel:c256+c256_d2' nice -n 10 uv run python tools/eval_tricks.py --run wm_ssm_none_s0
    # smoke on a few tasks (prints smoke_val_* only)
    ALMA_TRICK=c256 uv run python tools/eval_tricks.py --run wm_ssm_none_s0 --tasks ETT_1H,M_DENSE_1H
    # as a queue job (runs/jobs/*.txt; jobq sets TSAR_RUN from run=):
    #   tricks_sel_s0 | tools/eval_tricks.py | ALMA_TRICK=sel:c256+c256_d2 | run=wm_ssm_none_s0 | mem=1200

Trick grammar (ALMA_TRICK or --trick; several separated by ';' are scored in one process, sharing the cache):
    <expert>                       one forecaster, e.g. c256 (= current best: ctx 256 + paths + TTA=1), c512,
                                   c256_d2, c256_m168, c256_s4, c256_raw (wm/infer_tricks.Expert)
    avg:<e1>+<e2>+...              quantile average of the experts
    sel[<r>]:<e1>+<e2>+...         per window, the expert with the lowest in-context backtest pinball (summed
                                   over the backtest origins); e1 is the default and is kept unless another
                                   expert's loss is below r * loss(e1) (r default 1.0); no valid backtest -> e1
    soft[<T>]:<e1>+...             weights exp(-(L_e / min L - 1) / T) (T default 0.1), quantile-averaged
    sub:<t1>+<t2>                  t1 on sub-daily tasks (freq H/30min/15min/10min/5min, task metadata), t2 else
    cal[<lam>]:<trick>             per-window interval scale c on a grid, chosen on the in-context backtests of
                                   <trick>'s output, shrunk toward 1 in log space: log c = n_eff/(n_eff+lam) log c*
    Backtests: origins at ctx[:-b], b = j * h for j = 1..NBT (env ALMA_NBT, default 2), each scored on the h
    observed points that follow; valid only if at least ALMA_BT_MIN (default 64) points remain before it.

Only the window's own context is used for every decision (no target, no other window, no pooled statistic).
Expert forecasts and backtests are cached in runs/infer_tricks/cache/<run>/<expert>/<task>.npz (forecasts are
deterministic given checkpoint + spec + task). Per-task tables go to runs/infer_tricks/<run>__<tag>.csv.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import wm.infer_tricks as IT                                   # noqa: E402

OUT = os.path.join(ROOT, "runs", "infer_tricks")
NBT = int(os.environ.get("ALMA_NBT", 2))
BT_MIN = int(os.environ.get("ALMA_BT_MIN", 64))
CAL_GRID = np.exp(np.linspace(np.log(0.5), np.log(2.5), 25)).astype(np.float32)
HOURLYISH = ("H", "15min", "10min", "5min", "30min")


class Cache:
    def __init__(self, run):
        self.dir = os.path.join(OUT, "cache", run)
        self.mem = {}

    def get(self, spec, task, kind, fn):
        key = (spec, task["name"], kind)
        if key in self.mem:
            return self.mem[key]
        p = os.path.join(self.dir, spec, task["name"].replace("/", "__") + f".{kind}.npz")
        if os.path.exists(p):
            z = np.load(p)
            v = {k: z[k] for k in z.files}
        else:
            t0 = time.time()
            v = fn()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p + ".tmp.npz", **v)
            os.replace(p + ".tmp.npz", p)
            print(f"  computed {spec:<16} {kind:<4} {task['name']:<30} {time.time() - t0:6.1f}s", flush=True)
        self.mem[key] = v
        return v


def expert_fc(cache, spec, task):
    e = IT.make_expert(spec)
    return cache.get(spec, task, "fc", lambda: {"q": e.predict(task["contexts"], task["horizon"], task["season"])})["q"]


def expert_bt(cache, spec, task):
    """Backtests: list over j=1..NBT of (q [n, NQ, h] (zeros where invalid), y [n, h], valid [n])."""
    e, h = IT.make_expert(spec), task["horizon"]

    def make():
        out = {}
        for j in range(1, NBT + 1):
            b = j * h
            ctxs = task["contexts"]
            valid = np.array([len(c) - b >= BT_MIN for c in ctxs])
            y = np.full((len(ctxs), h), np.nan, np.float32)
            q = np.zeros((len(ctxs), IT.H["NQ"], h), np.float32)
            idx = np.flatnonzero(valid)
            if len(idx):
                sub = [ctxs[i][:len(ctxs[i]) - b] for i in idx]
                q[idx] = e.predict(sub, h, task["season"])
                for i in idx:
                    c = ctxs[i]
                    y[i] = c[len(c) - b:len(c) - b + h]
            out[f"q{j}"], out[f"y{j}"], out[f"v{j}"] = q, y, valid
        return out

    v = cache.get(spec, task, f"bt{NBT}", make)
    return [(v[f"q{j}"], v[f"y{j}"], v[f"v{j}"]) for j in range(1, NBT + 1)]


def bt_loss(bts):
    """Per-window backtest loss summed over valid origins, and the number of valid origins."""
    L = np.zeros(len(bts[0][0]))
    nv = np.zeros(len(bts[0][0]))
    for q, y, v in bts:
        L += np.where(v, IT.window_pinball(q, y), 0.0)
        nv += v
    return L, nv


def split_args(s):
    m = re.match(r"^([a-z]+)([0-9.]*):(.*)$", s)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, s)


def run_trick(trick, cache, task):
    """-> (forecast [n, NQ, h], backtests list as in expert_bt) for the composed trick."""
    op, par, rest = split_args(trick)
    if op is None:
        return expert_fc(cache, rest, task), (lambda: expert_bt(cache, rest, task))
    if op == "cal":
        lam = float(par) if par else 2.0
        fc, btf = run_trick(rest, cache, task)
        bts = btf()
        n = len(fc)
        loss = np.zeros((len(CAL_GRID), n))
        nv = np.zeros(n)
        for q, y, v in bts:
            for gi, c in enumerate(CAL_GRID):
                loss[gi] += np.where(v, IT.window_pinball(IT.scale_about_median(q, c), y), 0.0)
            nv += v
        best = CAL_GRID[loss.argmin(0)]
        w = nv / (nv + lam) if lam > 0 else (nv > 0).astype(float)
        c = np.exp(w * np.log(best))
        c = np.where(nv > 0, c, 1.0).astype(np.float32)
        task.setdefault("_cal", []).append(c)
        new_bts = [(IT.scale_about_median(q, c), y, v) for q, y, v in bts]
        return IT.scale_about_median(fc, c), (lambda: new_bts)
    specs = rest.split("+")
    if op == "sub":                                    # sub:<sub-daily trick>+<other trick> (gate on task freq)
        return run_trick(specs[0] if task["freq"] in HOURLYISH else "+".join(specs[1:]), cache, task)
    parts = [run_trick(s, cache, task) for s in specs]
    fcs = np.stack([p[0] for p in parts])                                  # [E, n, NQ, h]
    if op == "avg":
        def btf():
            bl = [p[1]() for p in parts]
            return [(np.mean([b[j][0] for b in bl], 0), bl[0][j][1], bl[0][j][2]) for j in range(NBT)]
        return fcs.mean(0), btf
    bl = [p[1]() for p in parts]
    L = np.stack([bt_loss(b)[0] for b in bl])                              # [E, n]
    nv = bt_loss(bl[0])[1]
    if op == "sel":
        r = float(par) if par else 1.0
        alt = L[1:].argmin(0) + 1 if len(specs) > 1 else np.zeros(L.shape[1], int)
        choose = np.where((nv > 0) & (L[alt, np.arange(L.shape[1])] < r * L[0]), alt, 0)
        task.setdefault("_choice", []).append(np.bincount(choose, minlength=len(specs)))
        W = np.eye(len(specs))[choose].T                                    # [E, n]
    elif op == "soft":
        T = float(par) if par else 0.1
        rel = L / np.maximum(L.min(0, keepdims=True), 1e-12) - 1
        W = np.exp(-rel / T)
        W = W / W.sum(0, keepdims=True)
        W = np.where(nv[None] > 0, W, np.eye(len(specs))[:, :1])
    else:
        raise ValueError(f"unknown op {op!r} in {trick!r}")
    fc = np.einsum("en,enqh->nqh", W, fcs).astype(np.float32)
    new_bts = [(np.einsum("en,enqh->nqh", W, np.stack([b[j][0] for b in bl])).astype(np.float32),
                bl[0][j][1], bl[0][j][2]) for j in range(NBT)]
    return np.sort(fc, axis=1), (lambda: new_bts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.environ.get("TSAR_RUN"))
    ap.add_argument("--trick", default=os.environ.get("ALMA_TRICK", "c256"))
    ap.add_argument("--tasks", default=os.environ.get("ALMA_TASKS", ""))
    ap.add_argument("--device", default=os.environ.get("ALMA_DEVICE", "cpu"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ALMA_THREADS", 4)))
    a = ap.parse_args()
    if not a.run:
        raise SystemExit("--run (or TSAR_RUN) is required")
    if os.environ.get("TSAR_SPLIT", "val") != "val":
        raise SystemExit("eval_tricks scores VAL only")
    import torch
    torch.set_num_threads(a.threads)
    ckpt = os.path.join(ROOT, "data", "ckpt", f"{a.run}.pt")
    print(f"eval_tricks: run={a.run} ckpt_sha256={hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()}", flush=True)
    IT.load_harness(ckpt, device=a.device)
    import bench as BN
    names = [t for t in a.tasks.split(",") if t]
    tasks = BN.load_tasks("val", set(names) if names else None)
    cache = Cache(a.run)
    tricks = [t.strip() for t in a.trick.split(";") if t.strip()]
    prefix = "smoke_" if names else ""
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for trick in tricks:
        t0 = time.time()
        final = {}
        for t in tasks:
            t.pop("_choice", None); t.pop("_cal", None)
            final[t["name"]] = run_trick(trick, cache, t)[0]
            if "_choice" in t:
                print(f"  {t['name']:<30} sel counts {t['_choice'][-1].tolist()}", flush=True)
            if "_cal" in t:
                c = t["_cal"][-1]
                print(f"  {t['name']:<30} cal c median {np.median(c):.3f} [{np.quantile(c, .1):.2f}, "
                      f"{np.quantile(c, .9):.2f}]", flush=True)
        order = iter([t["name"] for t in tasks])
        res = BN.evaluate(lambda c, h, m, f: final[next(order)], split="val", verbose=True, tasks=tasks)
        df = res["per_dataset"]
        tag = re.sub(r"[^A-Za-z0-9.+_-]", "_", trick)[:120]
        df.assign(trick=trick, run=a.run).to_csv(os.path.join(OUT, f"{a.run}__{prefix}{tag}.csv"), index=False)
        sub = df[df["freq"].isin(HOURLYISH)]
        print(f"trick: {trick}")
        print(f"{prefix}val_wql_rel:      {res['wql_rel']:.6f}")
        print(f"{prefix}val_mase_rel:     {res['mase_rel']:.6f}")
        print(f"{prefix}val_cov80_error:  {res['cov80_error']:.6f}")
        print(f"{prefix}family_wql_rel:   {res['wql_rel_family']:.6f}")
        if len(sub):
            print(f"{prefix}subdaily_wql_rel: {IT_gmean(sub['wql_rel']):.6f}  ({len(sub)} tasks)")
        print(f"trick_seconds:    {time.time() - t0:.1f}", flush=True)
        summary.append((trick, res["wql_rel_family"], res["wql_rel"], res["cov80_error"]))
    print("=== summary (family_wql_rel, wql_rel, cov80_error) ===")
    for s in summary:
        print(f"{s[1]:.4f}  {s[2]:.4f}  {s[3]:.4f}  {s[0]}")


def IT_gmean(s):
    s = np.asarray(s, dtype=np.float64)
    return float(np.exp(np.log(np.clip(s, 1e-6, None)).mean()))


if __name__ == "__main__":
    main()
