"""
Drop-in launcher for train.py that adds two things WITHOUT editing train.py:

  TSAR_SEED=<int>   overrides train.py's hard-coded `SEED = 0` (torch, numpy and data rng)
  TSAR_DUMP=<dir>   after every evaluate call, writes
                      old suite (prepare.evaluate):
                        <dir>/per_dataset_<split>.csv   (exactly res["per_dataset"])
                        <dir>/preds_<split>.npz         (raw [n, 9, H] quantile forecasts per dataset)
                      bench (TSAR_BENCH=1, tools/bench.evaluate):
                        <dir>/per_task_<split>.csv      (exactly res["per_dataset"], one row per task)
                        <dir>/preds_<split>.npz         (raw [n, 9, H] forecasts per task, key = task name)
                      both:
                        <dir>/meta.json                 (seed, git commit, TSAR_* env, headline numbers,
                                                         checkpoint sha256 under TSAR_EVAL_ONLY)
                    tools/compare.py reads these directories directly.

Everything else is train.py, unchanged:
    TSAR_SEED=1 TSAR_DUMP=runs/dumps/base_s1 TSAR_MINUTES=4 uv run tools/run.py > runs/base_s1.log 2>&1
With neither variable set it behaves exactly like `uv run train.py`.
"""
import os
import re
import sys
import json
import hashlib
import functools
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import prepare as P  # noqa: E402  (train.py's `import prepare as P` reuses this patched module)

DUMP = os.environ.get("TSAR_DUMP")
_orig_evaluate = P.evaluate
_BN = None                     # tools/bench.py module, imported as "bench" exactly like train.py does
_orig_bench_evaluate = None


def bench_module():
    """Import tools/bench.py under the module name train.py uses ("bench"), so a patch made here is
    the function train.py's `EVALUATE = BN.evaluate` picks up (and bench.run_baselines too)."""
    global _BN, _orig_bench_evaluate
    if _BN is None:
        tools = os.path.join(ROOT, "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import bench as BN
        _BN, _orig_bench_evaluate = BN, BN.evaluate
    return _BN


def _git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "train.py", "tools"], cwd=ROOT) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _capturing(predict_fn, captured):
    @functools.wraps(predict_fn)            # keeps predict_fn's name for bench's test_access.log entry
    def capture(ctxs, h, m, freq):
        q = predict_fn(ctxs, h, m, freq)
        captured.append(np.asarray(q, dtype=np.float32))
        return q
    return capture


def _write_dump(res, split, captured, table):
    names = list(res["per_dataset"]["dataset"])
    assert len(names) == len(captured), "dataset / forecast count mismatch"
    dump = os.environ["TSAR_DUMP"]
    os.makedirs(dump, exist_ok=True)
    res["per_dataset"].to_csv(os.path.join(dump, f"{table}_{split}.csv"), index=False)
    np.savez_compressed(os.path.join(dump, f"preds_{split}.npz"), **dict(zip(names, captured)))
    meta_path = os.path.join(dump, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update({"seed": int(os.environ.get("TSAR_SEED", 0)), "commit": _git_commit(),
                 "bench": table == "per_task",
                 "env": {k: v for k, v in os.environ.items() if k.startswith("TSAR_")}})
    ck = os.environ.get("TSAR_EVAL_ONLY")
    if ck and os.path.exists(ck):
        meta["ckpt"] = {"path": ck, "sha256": _sha256(ck)}
    keys = ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family", "mase_rel_family")
    meta.setdefault("splits", {})[split] = {k: res[k] for k in keys if k in res}
    if "n_nonfinite" in res["per_dataset"]:
        meta["splits"][split]["n_nonfinite"] = int(res["per_dataset"]["n_nonfinite"].sum())
    json.dump(meta, open(meta_path, "w"), indent=1)


def _evaluate_and_dump(predict_fn, split="val", verbose=True):
    """prepare.evaluate (old suite) + dump."""
    captured = []
    res = _orig_evaluate(_capturing(predict_fn, captured), split=split, verbose=verbose)
    _write_dump(res, split, captured, "per_dataset")
    return res


def _bench_evaluate_and_dump(predict_fn, split="val", verbose=True, names=None, tasks=None):
    """tools/bench.evaluate + dump. Same signature and return value as bench.evaluate; the metric
    code is bench's own (this only records the forecasts it was given)."""
    bench_module()
    captured = []
    res = _orig_bench_evaluate(_capturing(predict_fn, captured), split=split, verbose=verbose,
                               names=names, tasks=tasks)
    _write_dump(res, split, captured, "per_task")
    return res


def main():
    if DUMP:
        P.evaluate = _evaluate_and_dump
        if int(os.environ.get("TSAR_BENCH", 0) or 0):
            bench_module().evaluate = _bench_evaluate_and_dump
    src = open(os.path.join(ROOT, "train.py")).read()
    if "TSAR_SEED" not in src:            # train.py has no native seed override: patch the literal
        pat = re.compile(r"^SEED = \d+", re.M)
        if len(pat.findall(src)) != 1:
            sys.exit("tools/run.py: could not find a unique `SEED = <int>` line in train.py")
        src = pat.sub("SEED = int(os.environ.get('TSAR_SEED', 0))", src, count=1)
    sys.argv = [os.path.join(ROOT, "train.py")] + sys.argv[1:]
    g = {"__name__": "__main__", "__file__": os.path.join(ROOT, "train.py")}
    exec(compile(src, os.path.join(ROOT, "train.py"), "exec"), g)


if __name__ == "__main__":
    main()
