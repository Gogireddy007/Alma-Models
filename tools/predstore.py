"""
Prediction store: every model's raw bench forecasts, saved once, so scores can be regenerated
(tools/rescore.py) and models compared without re-running inference.

Layout
    runs/preds/<model_id>/meta.json                 provenance + per split: task list, shapes,
                                                    target sha256, the producing harness's scores
    runs/preds/<model_id>/<split>/<task>.npz        ('/' in a task name -> '__') q [windows, Q, h] float32 (raw, as returned by
                                                    the model), series_id [windows], window [windows]
    runs/preds/<model_id>/<split>/harness_scores.csv  bench.evaluate's per-task table at write time
    runs/preds/_targets/<sha256>.npy                content-addressed target arrays (one copy shared
                                                    by every model; model folders only cite the hash)

target sha256 = sha256 over b"float32" + repr(shape) + C-order float32 bytes (NaN kept).

Commands (VAL only; the test split is refused here)
    # any train*.py script, eval-only on a frozen checkpoint (wraps bench.evaluate like tools/run.py)
    TSAR_BENCH=1 TSAR_SPLIT=val TSAR_EVAL_ONLY=data/ckpt/b_curr.pt \\
        uv run tools/predstore.py run --model-id b_curr --script train.py
    # import an existing TSAR_DUMP directory (tools/run.py or tools/dump_baseline.py --bench)
    uv run tools/predstore.py import baselines/bench/dumps/chronos2 --model-id chronos2
    # bench's own reference forecasters (cheap; a self-test for rescore: snaive has wql_rel = 1)
    uv run tools/predstore.py builtin snaive
    uv run tools/predstore.py ls
"""
import os
import sys
import json
import time
import hashlib
import argparse
import platform
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)
STORE = os.path.join(ROOT, "runs", "preds")
ALLOWED_SPLITS = ("val",)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_sha256(y):
    y = np.ascontiguousarray(y, dtype=np.float32)
    h = hashlib.sha256(b"float32" + repr(tuple(y.shape)).encode())
    h.update(y.tobytes())
    return h.hexdigest()


def save_target(y):
    sha = target_sha256(y)
    path = os.path.join(STORE, "_targets", f"{sha}.npy")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + ".tmp.npy", np.ascontiguousarray(y, dtype=np.float32))
        os.replace(path + ".tmp.npy", path)
    return sha


def load_target(sha):
    y = np.load(os.path.join(STORE, "_targets", f"{sha}.npy"))
    if target_sha256(y) != sha:
        raise ValueError(f"target cache {sha} is corrupt")
    return y


def _git():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) != 0
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": "unknown", "dirty": None}


def _env_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__}
    for mod in ("torch", "pandas"):
        m = sys.modules.get(mod)
        if m is not None:
            info[mod] = getattr(m, "__version__", "?")
    return info


def _bench():
    import run as R          # tools/run.py: imports bench under the name train scripts use
    return R.bench_module()


def task_file(name):
    """Task names may contain '/' (e.g. uk_covid_nation_1D/cumulative): stored as '__'."""
    assert "__" not in name, name
    return name.replace("/", "__") + ".npz"


def write_split(model_id, split, tasks, preds, harness_table=None, extra_meta=None):
    """tasks: bench task dicts (same order as preds). preds: list of [n, Q, h] arrays."""
    if split not in ALLOWED_SPLITS:
        raise SystemExit(f"predstore: split {split!r} refused (VAL only)")
    import prepare as P
    out = os.path.join(STORE, model_id, split)
    os.makedirs(out, exist_ok=True)
    rows = {}
    for t, q in zip(tasks, preds):
        q = np.asarray(q, dtype=np.float32)
        n, h = len(t["contexts"]), t["horizon"]
        assert q.shape == (n, len(P.QUANTILES), h), f"{t['name']}: {q.shape} != {(n, len(P.QUANTILES), h)}"
        np.savez_compressed(os.path.join(out, task_file(t["name"])), q=q,
                            series_id=np.asarray(t["series_id"]), window=np.asarray(t["window"]))
        rows[t["name"]] = {"family": t["family"], "windows": n, "horizon": h, "season": t["season"],
                           "freq": t["freq"], "target_sha256": save_target(t["targets"]),
                           "n_nonfinite": int((~np.isfinite(q)).any(axis=(1, 2)).sum())}
    if harness_table is not None:
        harness_table.to_csv(os.path.join(out, "harness_scores.csv"), index=False)
    meta_path = os.path.join(STORE, model_id, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"model_id": model_id, "quantile_levels": [float(x) for x in P.QUANTILES],
                 "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": _git(), "env_info": _env_info(),
                 "store_format": 1})
    meta.update(extra_meta or {})
    meta.setdefault("splits", {})[split] = {"tasks": rows}
    if harness_table is not None:
        meta["splits"][split]["harness"] = "tools/bench.py evaluate"
    tmp = meta_path + ".tmp"
    json.dump(meta, open(tmp, "w"), indent=1, sort_keys=True, default=str)
    os.replace(tmp, meta_path)
    return out


def load_split(model_id, split):
    """-> (meta, {task: dict(q, series_id, window)}). Reads only the store."""
    meta = json.load(open(os.path.join(STORE, model_id, "meta.json")))
    out = {}
    for name in meta["splits"][split]["tasks"]:
        z = np.load(os.path.join(STORE, model_id, split, task_file(name)))
        out[name] = {k: z[k] for k in z.files}
    return meta, out


# ---------------------------------------------------------------------------
# run: wrap bench.evaluate and exec a train*.py script (eval-only)
# ---------------------------------------------------------------------------

def cmd_run(a):
    if os.environ.get("TSAR_SPLIT", "val") not in ALLOWED_SPLITS:
        sys.exit("predstore run: VAL only")
    if not int(os.environ.get("TSAR_BENCH", 0) or 0):
        sys.exit("predstore run: needs TSAR_BENCH=1")
    os.chdir(ROOT)
    script = os.path.join(ROOT, a.script)
    BN = _bench()
    orig = BN.evaluate
    g = {"__name__": "__main__", "__file__": script}

    def evaluate(predict_fn, split="val", verbose=True, names=None, tasks=None):
        tasks = tasks if tasks is not None else BN.load_tasks(split, names)
        captured = []

        def capture(ctxs, h, m, freq):
            q = predict_fn(ctxs, h, m, freq)
            captured.append(np.asarray(q, dtype=np.float32))
            return q
        capture.__name__ = getattr(predict_fn, "__name__", "predict")
        res = orig(capture, split=split, verbose=verbose, names=names, tasks=tasks)
        scored = [t for t in tasks if t["contexts"]]
        assert [t["name"] for t in scored] == list(res["per_dataset"]["dataset"]) and len(scored) == len(captured)
        ck = os.environ.get("TSAR_EVAL_ONLY")
        extra = {"source": "predstore run", "script": a.script, "script_sha256": sha256_file(script),
                 "ckpt": {"path": ck, "sha256": sha256_file(ck)} if ck else None,
                 "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("TSAR_", "PYTORCH_"))},
                 "context_length": g.get("CONTEXT"), "context_total": g.get("TCTX", g.get("CONTEXT")),
                 "bench_max_context": BN.MAX_CONTEXT if hasattr(BN, "MAX_CONTEXT") else None,
                 "harness_headline": {split: {k: float(res[k]) for k in
                                              ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family")}}}
        out = write_split(a.model_id, split, scored, captured, res["per_dataset"], extra)
        print(f"predstore: {len(scored)} tasks -> {out}", flush=True)
        return res

    BN.evaluate = evaluate
    sys.argv = [script]
    exec(compile(open(script).read(), script, "exec"), g)


# ---------------------------------------------------------------------------
# import: an existing TSAR_DUMP directory (preds_<split>.npz + meta.json + per_task_<split>.csv)
# ---------------------------------------------------------------------------

def cmd_import(a):
    import pandas as pd
    BN = _bench()
    npz = os.path.join(a.dump, f"preds_{a.split}.npz")
    if not os.path.exists(npz):
        sys.exit(f"no {npz}")
    z = np.load(npz)
    tasks = [t for t in BN.load_tasks(a.split) if t["name"] in z.files]
    missing = set(z.files) - {t["name"] for t in tasks}
    if missing:
        sys.exit(f"dump has tasks the bench does not: {sorted(missing)}")
    dmeta_path = os.path.join(a.dump, "meta.json")
    dmeta = json.load(open(dmeta_path)) if os.path.exists(dmeta_path) else {}
    tab = os.path.join(a.dump, f"per_task_{a.split}.csv")
    extra = {"source": f"import {os.path.relpath(a.dump, ROOT)}", "dump_meta": dmeta,
             "ckpt": dmeta.get("ckpt"), "env": dmeta.get("env"), "script": a.script,
             "script_sha256": sha256_file(os.path.join(ROOT, a.script)) if a.script else None,
             "context_length": a.context}
    if dmeta.get("splits", {}).get(a.split):
        extra["harness_headline"] = {a.split: dmeta["splits"][a.split]}
    out = write_split(a.model_id, a.split, tasks, [z[t["name"]] for t in tasks],
                      pd.read_csv(tab) if os.path.exists(tab) else None, extra)
    print(f"predstore: imported {len(tasks)} tasks -> {out}")


def cmd_builtin(a):
    BN = _bench()
    fn = {"snaive": BN._snaive, "naive": BN._naive}[a.name]
    tasks = [t for t in BN.load_tasks(a.split) if t["contexts"]]
    preds = [fn([c.copy() for c in t["contexts"]], t["horizon"], t["season"], t["freq"]) for t in tasks]
    out = write_split(a.name, a.split, tasks, preds, None,
                      {"source": f"builtin tools/bench.py _{a.name}", "script": "tools/bench.py",
                       "script_sha256": sha256_file(os.path.join(TOOLS, "bench.py")), "ckpt": None,
                       "context_length": None})
    print(f"predstore: {len(tasks)} tasks -> {out}")


def cmd_ls(a):
    if not os.path.isdir(STORE):
        return
    for m in sorted(os.listdir(STORE)):
        p = os.path.join(STORE, m, "meta.json")
        if m.startswith("_") or not os.path.exists(p):
            continue
        meta = json.load(open(p))
        for s, d in meta.get("splits", {}).items():
            hh = (meta.get("harness_headline") or {}).get(s, {})
            print(f"{m:28s} {s:4s} {len(d['tasks']):3d} tasks  harness wql_rel={hh.get('wql_rel', float('nan')):.4f}"
                  f"  ckpt={((meta.get('ckpt') or {}).get('sha256') or '-')[:12]}  {meta['date']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model-id", required=True); r.add_argument("--script", default="train.py")
    i = sub.add_parser("import"); i.add_argument("dump"); i.add_argument("--model-id", required=True)
    i.add_argument("--split", default="val", choices=ALLOWED_SPLITS); i.add_argument("--script")
    i.add_argument("--context", type=int)
    b = sub.add_parser("builtin"); b.add_argument("name", choices=["snaive", "naive"])
    b.add_argument("--split", default="val", choices=ALLOWED_SPLITS)
    sub.add_parser("ls")
    a = ap.parse_args()
    {"run": cmd_run, "import": cmd_import, "builtin": cmd_builtin, "ls": cmd_ls}[a.cmd](a)


if __name__ == "__main__":
    main()
