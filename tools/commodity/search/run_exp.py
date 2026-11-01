"""One autoresearch experiment: train a candidate for a fixed wall-clock budget on data before the cut, score it on
the DEV rolling-origin backtest (WTI, weekly origins 2023-01-01..2025-06-30), append one row to
runs/commodity/search/results.tsv, save the checkpoint and the backtest quantiles.

    nice -n 10 python3 tools/commodity/search/run_exp.py --name sv_base --arch sv --abl none --minutes 5

Never touches VALIDATION or TEST: the panel is loaded with split="dev" (asserts max date <= 2025-06-30).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np        # noqa: E402
import torch              # noqa: E402

import wm_core as C       # noqa: E402
import wm_data as D       # noqa: E402

OUT = os.path.join(D.ROOT, "runs", "commodity", "search")
TSV = os.path.join(OUT, "results.tsv")
COLS = ["ts", "name", "arch", "abl", "params", "steps", "train_s", "score", "ci5", "ci95", "ratio1", "ratio5",
        "ratio20", "da1", "da5", "da20", "volda", "cov80_20", "score_d", "ratio_d", "da_d", "status", "cfg", "note"]


def build(arch, abl, kw):
    if arch == "alma":
        import wm_alma
        return wm_alma.AlmaSSM(abl=abl, **kw)
    import wm_models
    return wm_models.ARCHS[arch](abl=abl, **kw)


def resolve_cols(P, xc):
    return [P.names.index(n) for n in C.XC_SERIES] if xc else [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--abl", default="none")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--xc", type=int, default=0)
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--cut", default="2023-01-01")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    C.set_threads(a.threads)
    kw = json.loads(a.kw)
    P = C.Panel("dev")
    cols = resolve_cols(P, a.xc)
    L = 256 if a.arch == "alma" else a.L
    Htr = 64 if a.arch == "alma" else C.H_MAX
    if a.xc:
        kw.setdefault("N", len(cols))
        kw.setdefault("spreads", True)
    cut = np.datetime64(a.cut)
    assert cut <= np.datetime64("2023-01-01") or a.name.startswith("refit"), "search models train before the backtest"
    row = {k: "" for k in COLS}
    row.update(ts=time.strftime("%Y-%m-%d %H:%M"), name=a.name, arch=a.arch + ("+xc" if a.xc else "") +
               ("+pool" if a.pool else ""), abl=a.abl, note=a.note,
               cfg=json.dumps(dict(vars(a), kw=kw), sort_keys=True))
    torch.manual_seed(a.seed)
    try:
        model = build(a.arch, a.abl, kw)
        row["params"] = model.n_params()
        if a.pool:
            og = []
            for c in range(P.N):
                o = C.train_origins(P, cut=a.cut, L=L, H=Htr, cols=[c])
                if len(o):
                    og.append((c, o))
            origins = og
            print("pool:", [(P.names[c], len(o)) for c, o in og])
        else:
            origins = C.train_origins(P, cut=a.cut, L=L, H=Htr, cols=cols)
            print(f"train origins {len(origins)} {P.dates[origins[0]].date()}..{P.dates[origins[-1]].date()}"
                  f" (last target {P.dates[origins[-1] + Htr].date()})")
        print(f"{a.name}: {a.arch}/{a.abl} params {row['params']}")
        steps, secs, ema = C.train_loop(model, P, origins, cols, a.minutes, lr=a.lr, batch=a.batch, seed=a.seed,
                                        H=Htr, L=L, max_steps=a.max_steps or None, pool=bool(a.pool))
        row.update(steps=steps, train_s=round(secs))
        os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
        ck = os.path.join(OUT, "ckpt", f"{a.name}.pt")
        torch.save({"state": model.state_dict(), "arch": a.arch, "abl": a.abl, "kw": kw, "L": L, "cols": cols,
                    "names": [P.names[c] for c in cols], "cut": a.cut, "steps": steps, "cfg": vars(a)}, ck)
        bo = C.backtest_origins(P)
        q = C.predict_origins(model, P, bo, cols, L=L, H=C.H_MAX)
        np.savez_compressed(os.path.join(OUT, "ckpt", f"{a.name}.bt.npz"), q=q, origins=bo)
        r = C.evaluate(P, bo, q)
        lo, hi = C.score_ci(P, bo, q)
        row.update({k: round(r[k], 4) for k in r if k in COLS}, ci5=round(lo, 4), ci95=round(hi, 4), status="ok")
        row["note"] = (a.note + f" sha={hashlib.sha256(open(ck, 'rb').read()).hexdigest()[:12]}").strip()
        bd = C.backtest_origins(P, every=1)
        qd = C.predict_origins(model, P, bd, cols, L=L, H=C.H_MAX)
        rd = C.evaluate(P, bd, qd)
        row.update(score_d=round(rd["score"], 4), ratio_d=round(np.mean([rd[f"ratio{h}"] for h in C.HS]), 4),
                   da_d=round(np.mean([rd[f"da{h}"] for h in C.HS]), 4))
        print(json.dumps(r), "daily", json.dumps(rd))
    except Exception as e:
        traceback.print_exc()
        row.update(status="crash", note=(a.note + " " + repr(e)[:150]).strip())
    new = not os.path.exists(TSV)
    with open(TSV, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(row[k]).replace("\t", " ") for k in COLS) + "\n")
    print("RESULT", row["name"], row["status"], row["score"])


if __name__ == "__main__":
    main()
