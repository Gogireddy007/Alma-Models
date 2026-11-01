"""
Stage 2 of tools/fev_full.py: forecast the exported fev-bench contexts with a frozen Alma checkpoint.

Runs train_wm.py itself in eval-only mode (TSAR_EVAL_ONLY), so the model, the checkpoint/config verification
(cfg_mismatch against the stored data config), the causality check and the inference path (predict(): TTA, long-horizon
paths) are exactly the ones every VAL number came from. Only the VAL scorer is swapped: bench.evaluate is replaced by
a function that forecasts runs/fev_full/inputs/<task>.npz and writes runs/fev_full/<model>/preds/<task>.npz.
Nothing here reads a target; the contexts were cut by fev.Task before export.

    nice -n 10 uv run tools/fev_full_alma.py --ckpt data/ckpt/wm_ssm_none_s0.pt --model alma_ssm_s0 \\
        [--tasks t1,t2] [--threads 4] [--tta 1] [--long paths]

CPU only (TSAR_DEVICE=cpu). Rows are forecast in slices of --rows contexts (predict() is row-wise: audit_r1 s1).
"""
import os
import sys
import json
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "runs", "fev_full", "inputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: every exported task)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--tta", default="1")
    ap.add_argument("--long", default="paths")
    ap.add_argument("--full-cfg-check", action="store_true", help="stock train_wm path (loads TRAIN, ~2.6 GB)")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    for k in [k for k in os.environ if k.startswith("TSAR_")]:
        del os.environ[k]
    os.environ.update({"TSAR_EVAL_ONLY": a.ckpt, "TSAR_DEVICE": "cpu", "TSAR_WM_TTA": a.tta,
                       "TSAR_WM_LONG": a.long})
    out = os.path.join(ROOT, "runs", "fev_full", a.model, "preds")
    os.makedirs(out, exist_ok=True)
    want = [t for t in a.tasks.split(",") if t]
    files = sorted(os.listdir(IN))
    if want:
        files = [w.replace("/", "__") + ".npz" for w in want]

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import bench as BN

    def fev_eval(predict_fn, split="val", verbose=True, names=None, tasks=None):
        g = predict_fn.__globals__
        import prepare as P
        env = {k: v for k, v in os.environ.items() if k.startswith("TSAR_")}
        tot = 0.0
        for fn in files:
            dst = os.path.join(out, fn)
            if os.path.exists(dst):
                continue
            z = np.load(os.path.join(IN, fn))
            meta = json.loads(str(z["meta"]))
            lens = z["lens"]
            offs = np.concatenate([[0], np.cumsum(lens)])
            ctxs = [z["flat"][offs[i]:offs[i + 1]] for i in range(len(lens))]
            H, m = int(meta["horizon"]), int(meta["seasonality"])
            t0 = time.time()
            qs = [predict_fn(ctxs[s:s + a.rows], H, m, meta["freq"]) for s in range(0, len(ctxs), a.rows)]
            q = np.concatenate(qs).astype(np.float32)
            dt = time.time() - t0
            tot += dt
            if q.shape != (len(ctxs), 9, H):
                raise SystemExit(f"{fn}: forecast shape {q.shape} != {(len(ctxs), 9, H)}")
            pm = dict(meta, predict_s=round(dt, 3), ckpt=a.ckpt, ckpt_sha256=g.get("CK_SHA"), env=env,
                      quantile_levels=[round(float(v), 6) for v in P.QUANTILES])
            np.savez(dst + ".tmp.npz", q=q, window=z["window"], meta=np.array(json.dumps(pm)))
            os.replace(dst + ".tmp.npz", dst)
            print(f"pred {meta['task_name']:32s} n={len(ctxs):7d} H={H:4d} m={m:5d} {dt:7.1f}s "
                  f"nonfinite={int((~np.isfinite(q)).sum())}", flush=True)
            del ctxs, q, qs, z
        nan = float("nan")
        return {"wql_rel": nan, "mase_rel": nan, "cov80_error": nan, "predict_seconds": tot,
                "wql_rel_family": nan, "wql_rel_subday": nan}

    BN.evaluate = fev_eval
    path = os.path.join(ROOT, "train_wm.py")
    src = open(path).read()
    if a.full_cfg_check:                       # the stock path: loads TRAIN (~2.6 GB peak) to re-hash the data config
        code = src
    else:                                      # light path: no TRAIN load; the stored config is checked below instead
        code = _light(src, a.ckpt)
    sys.argv = [path]
    exec(compile(code, path, "exec"), {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__})


LOAD = 'train = BN.load_train(extra=bool(int(_env("TSAR_BENCH_EXTRA", 0))), max_len=int(_env("TSAR_BENCH_MAXLEN", 20000)))'
CHECK = 'bad = cfg_mismatch(CK0["cfg"], EVAL_FREE_KEYS | INFER_KEYS)'


def _light(src, ckpt):
    """train_wm.py with two lines swapped (each must occur exactly once): TRAIN is not loaded (eval never reads it),
    and cfg_mismatch's data-hash comparison is replaced by _stored_cfg_check(): the checkpoint's stored data config
    must equal the reference one (train_sha256, datasets, bench flags; tools/fev_full_alma.py REF_CKPT, whose hash
    was re-derived from the data by the stock path, --full-cfg-check), and its stored env must equal the run env
    except the eval-free/inference keys, exactly as cfg_mismatch's env part does."""
    for s in (LOAD, CHECK):
        if src.count(s) != 1:
            raise SystemExit(f"train_wm.py changed: expected exactly one line {s!r}")
    src = src.replace(LOAD, "train = []  # fev_full_alma light path: TRAIN not loaded for eval")
    return src.replace(CHECK, "bad = __import__('fev_full_alma')._stored_cfg_check(CK0['cfg'], EVAL_FREE_KEYS | INFER_KEYS)")


REF_CKPT = "data/ckpt/wm_ssm_none_s0.pt"
DATA_KEYS = ("bench", "bench_extra", "bench_maxlen", "family", "context", "max_h", "patch", "d_model", "n_layers",
             "n_heads", "train_datasets", "train_series", "train_sha256", "train_drop_sha256")


def _stored_cfg_check(stored, ignore):
    import torch
    ref = torch.load(os.path.join(ROOT, REF_CKPT), map_location="cpu", weights_only=False)["cfg"]
    bad = [f"{k}: checkpoint={str(stored.get(k))[:80]} reference={str(ref.get(k))[:80]}"
           for k in DATA_KEYS if stored.get(k) != ref.get(k)]
    cur = {k: v for k, v in os.environ.items() if k.startswith("TSAR_") and k not in ignore}
    se = {k: v for k, v in stored.get("env", {}).items() if k not in ignore}
    bad += [f"{k}: checkpoint={se.get(k)!r} now={cur.get(k)!r}" for k in sorted(set(se) | set(cur)) if se.get(k) != cur.get(k)]
    print(f"fev_full_alma: stored-config check vs {REF_CKPT}: {'ok' if not bad else 'FAIL'} "
          f"(train_sha256={stored.get('train_sha256', '')[:12]}, {len(stored.get('train_datasets', []))} TRAIN datasets)",
          flush=True)
    return bad


if __name__ == "__main__":
    main()
