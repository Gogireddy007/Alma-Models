"""
Full fev-bench (100 tasks) for Alma with the official fev 0.10.0 Task API. Runs in the eval venv (.venv-eval, which
has fev; the project venv has torch). Three stages, one task at a time, so memory stays small:

  1. export  (eval venv)    fev.Task windows -> univariate contexts, runs/fev_full/inputs/<task>.npz
  2. predict (project venv)  tools/fev_full_alma.py: frozen Alma checkpoint -> runs/fev_full/<model>/preds/<task>.npz
  3. score   (eval venv)    predictions -> fev Task.evaluation_summary -> runs/fev_full/<model>/summaries/<task>.json

Protocol (identical to how fev scores Chronos-Bolt and other univariate models):
  - every task of benchmarks/fev_bench/tasks.yaml, its horizon / windows / seasonality / quantile levels;
  - multivariate targets are split with fev.convert_input_data(as_univariate=True) and recombined with
    fev.combine_univariate_predictions_to_multivariate; covariates are ignored;
  - contexts = the last 2048 observations (NaN kept as missing), as in tools/bench.py (Alma reads its last 256);
  - non-finite forecast values are replaced by 0 (bench/rescore sanitise rule) and counted in the summary;
  - point forecast = the 0.5 quantile.

Data: the parquet files of autogluon/fev_datasets at the revision pinned in tools/bench.py (FEV_REV). Files already in
data/bench/fev are reused; the others are downloaded to data/fev_full/raw. The dataset fingerprint fev computes is
stored next to the leaderboard's fingerprint for the same task.

TEST guard: 48 fev-bench tasks are this project's one-shot TEST split (tools/bench.py TEST_TASKS, docs/protocol.md).
export refuses them unless --allow-test is given, and logs every such access to data/bench/test_access.log.

    HF_DATASETS_OFFLINE=1 .venv-eval/bin/python -I tools/fev_full.py export --tasks nontest
    HF_DATASETS_OFFLINE=1 .venv-eval/bin/python -I tools/fev_full.py score --model alma_ssm_s0 --tasks nontest
"""
import os
import re
import sys
import json
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "fev_full")
RAW = os.path.join(ROOT, "data", "fev_full", "raw")
BENCH_FEV = os.path.join(ROOT, "data", "bench", "fev")
TEST_LOG = os.path.join(ROOT, "data", "bench", "test_access.log")
FEV_CLONE = os.environ.get("FEV_CLONE", "")
MAX_CONTEXT = 2048
FEV_REPO = "autogluon/fev_datasets"

# leakage: fev-bench tasks whose dataset is in Alma's real training data (TSAR_GIFT_CLEAN=1 corpus, see
# docs/fev_full_results.md section "Leakage"). Filled by the overlap review; read by score.
LEAK_FILE = os.path.join(OUT, "leakage.json")


def _bench_src():
    return open(os.path.join(ROOT, "tools", "bench.py")).read()


def bench_split_configs():
    src = _bench_src()
    val = src.split("VAL_TASKS = [")[1].split("TEST_TASKS = [")[0]
    test = src.split("TEST_TASKS = [")[1].split("\n]\n")[0]
    pat = r"^\s+\('([^']+)', '[^']+', '([^']+)'"
    return ({c for _, c in re.findall(pat, val, re.M)}, {c for _, c in re.findall(pat, test, re.M)})


def fev_rev():
    return re.search(r'FEV_REPO, FEV_REV = "[^"]+", "([0-9a-f]+)"', _bench_src()).group(1)


def subdir_families():
    src = _bench_src()
    body = src.split("_SUBDIR_FAMILIES = {")[1].split("}")[0]
    return set(re.findall(r'"([^"]+)"', body))


def hf_path(config):
    for fam in sorted(subdir_families() | {"m5", "ercot", "favorita_transactions", "kdd_cup_2022",
                                            "LOOP_SEATTLE"}, key=len, reverse=True):
        if config.startswith(fam + "_"):
            return f"{fam}/{config[len(fam) + 1:]}/train-00000-of-00001.parquet"
    return f"{config}/train-00000-of-00001.parquet"


def local_file(config, download=True):
    p = os.path.join(BENCH_FEV, config + ".parquet")
    if os.path.exists(p):
        return p
    p = os.path.join(RAW, config + ".parquet")
    if os.path.exists(p) or not download:
        return p
    import hashlib
    import requests
    os.makedirs(RAW, exist_ok=True)
    url = f"https://huggingface.co/datasets/{FEV_REPO}/resolve/{fev_rev()}/{hf_path(config)}"
    h = hashlib.sha256()
    with requests.get(url, timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(p + ".tmp", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
    os.replace(p + ".tmp", p)
    man = os.path.join(RAW, "manifest.json")
    m = json.load(open(man)) if os.path.exists(man) else {}
    m[config] = {"repo": FEV_REPO, "rev": fev_rev(), "path": hf_path(config), "sha256": h.hexdigest(),
                 "bytes": os.path.getsize(p), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(m, open(man, "w"), indent=1, sort_keys=True)
    return p


def bench_tasks():
    import fev
    return fev.Benchmark.from_yaml(os.path.join(FEV_CLONE, "benchmarks", "fev_bench", "tasks.yaml")).tasks


def safe(name):
    return name.replace("/", "__")


def select(tasks, which):
    val, test = bench_split_configs()
    tag = {t.task_name: ("VAL" if t.dataset_config in val else "TEST" if t.dataset_config in test else "other")
           for t in tasks}
    if which == "all":
        keep = [t for t in tasks]
    elif which == "nontest":
        keep = [t for t in tasks if tag[t.task_name] != "TEST"]
    elif which == "test":
        keep = [t for t in tasks if tag[t.task_name] == "TEST"]
    else:
        names = set(which.split(","))
        keep = [t for t in tasks if t.task_name in names]
        if len(keep) != len(names):
            sys.exit(f"unknown tasks {sorted(names - {t.task_name for t in tasks})}")
    return keep, tag


def local_task(t, download=True):
    import fev
    d = t.to_dict()
    d.pop("dataset_config", None)
    d["dataset_path"] = local_file(t.dataset_config, download)
    d["task_name"] = t.task_name
    keep = set(fev.Task.__dataclass_fields__) if hasattr(fev.Task, "__dataclass_fields__") else set(d)
    return fev.Task(**{k: v for k, v in d.items() if k in keep})


def export(args):
    import fev
    tasks, tag = select(bench_tasks(), args.tasks)
    os.makedirs(os.path.join(OUT, "inputs"), exist_ok=True)
    for t in tasks:
        if tag[t.task_name] == "TEST":
            if not args.allow_test:
                print(f"skip {t.task_name}: project TEST split (use --allow-test)", flush=True)
                continue
            with open(TEST_LOG, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tfev_full.export({t.task_name})\t"
                        f"argv={' '.join(sys.argv)}\n")
        dst = os.path.join(OUT, "inputs", safe(t.task_name) + ".npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        lt = local_task(t)
        ctx, win, lens = [], [], []
        for k, w in enumerate(lt.iter_windows()):
            past, _ = fev.convert_input_data(w, adapter="datasets", as_univariate=True)
            ys = past.with_format("numpy")["target"]
            for y in ys:
                y = np.asarray(y, dtype=np.float32)[-MAX_CONTEXT:]
                ctx.append(y)
                lens.append(len(y))
                win.append(k)
        lens = np.asarray(lens, np.int32)
        flat = np.concatenate(ctx) if ctx else np.zeros(0, np.float32)
        meta = {"task_name": t.task_name, "dataset_config": t.dataset_config, "horizon": t.horizon,
                "num_windows": t.num_windows, "seasonality": t.seasonality, "quantile_levels": t.quantile_levels,
                "target_columns": lt.target_columns, "n_forecasts": int(len(lens)), "split_tag": tag[t.task_name],
                "fingerprint": lt._dataset_fingerprint, "file": os.path.relpath(lt.dataset_path, ROOT),
                "freq": str(getattr(lt, "freq", "")), "export_s": round(time.time() - t0, 1)}
        np.savez(dst + ".tmp.npz", flat=flat, lens=lens, window=np.asarray(win, np.int16),
                 meta=np.array(json.dumps(meta)))
        os.replace(dst + ".tmp.npz", dst)
        print(f"export {t.task_name:32s} n={len(lens):7d} H={t.horizon:4d} m={t.seasonality:5d} "
              f"cols={len(lt.target_columns):3d} {time.time() - t0:6.1f}s", flush=True)
        del lt, ctx, flat


def leakage():
    return json.load(open(LEAK_FILE))["trained_on_this_dataset"] if os.path.exists(LEAK_FILE) else {}


def score(args):
    import fev
    import datasets
    import pandas as pd
    tasks, tag = select(bench_tasks(), args.tasks)
    leak = leakage()
    if not os.path.exists(LEAK_FILE):
        sys.exit(f"{LEAK_FILE} missing: write the leakage review first")
    sn = pd.read_csv(os.path.join(FEV_CLONE, "benchmarks", "fev_bench", "results", "seasonal_naive.csv"))
    sn = sn.drop_duplicates("task_name").set_index("task_name")
    mdir = os.path.join(OUT, args.model)
    os.makedirs(os.path.join(mdir, "summaries"), exist_ok=True)
    for t in tasks:
        pf = os.path.join(mdir, "preds", safe(t.task_name) + ".npz")
        dst = os.path.join(mdir, "summaries", safe(t.task_name) + ".json")
        if not os.path.exists(pf) or (os.path.exists(dst) and not args.force):
            continue
        z = np.load(pf)
        pm = json.loads(str(z["meta"]))
        q = z["q"].astype(np.float64)                                   # [N, Q, H] in export order
        nonfinite = int((~np.isfinite(q)).sum())
        q = np.sort(np.where(np.isfinite(q), q, 0.0), axis=1)
        levels = [float(v) for v in pm["quantile_levels"]]
        if levels != [float(v) for v in t.quantile_levels]:
            sys.exit(f"{t.task_name}: quantile levels {levels} != task {t.quantile_levels}")
        lt = local_task(t, download=False)
        win = z["window"]
        cols = lt.target_columns
        per_window = []
        for k in range(t.num_windows):
            qk = q[win == k]
            rec = {"predictions": list(qk[:, levels.index(0.5)])}
            for li, lv in enumerate(levels):
                rec[str(lv)] = list(qk[:, li])
            ds = datasets.Dataset.from_dict(rec)
            per_window.append(fev.combine_univariate_predictions_to_multivariate(ds, cols) if len(cols) > 1
                              else datasets.DatasetDict({cols[0]: ds}))
        summ = lt.evaluation_summary(per_window, model_name=args.model_name, training_time_s=0.0,
                                     inference_time_s=float(pm["predict_s"]),
                                     trained_on_this_dataset=bool(leak.get(t.task_name, False)),
                                     extra_info={"n_nonfinite": nonfinite, "split_tag": tag[t.task_name],
                                                 "ckpt_sha256": pm.get("ckpt_sha256"),
                                                 "alma_env": json.dumps(pm.get("env", {}), sort_keys=True)})
        # task-definition columns as the leaderboard writes them (we loaded a local copy of the same file)
        ref = sn.loc[t.task_name]
        summ["local_dataset_path"] = summ["dataset_path"]
        summ["local_fingerprint"] = summ["dataset_fingerprint"]
        summ["leaderboard_fingerprint"] = ref["dataset_fingerprint"]
        for c in ("dataset_path", "dataset_config", "initial_cutoff", "dataset_fingerprint"):
            summ[c] = None if pd.isna(ref[c]) else ref[c]
        json.dump(summ, open(dst, "w"), indent=1, default=str)
        print(f"score {t.task_name:32s} SQL={summ['SQL']:.4f} MASE={summ['MASE']:.4f} WQL={summ['WQL']:.4f} "
              f"SN_SQL={ref['SQL']:.4f} leak={summ['trained_on_this_dataset']} nonfinite={nonfinite}", flush=True)


def main():
    global FEV_CLONE
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "score", "list"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", default="nontest", help="all | nontest | test | comma-separated task names")
        p.add_argument("--fev-clone", default=os.environ.get("FEV_CLONE", ""))
        if name == "export":
            p.add_argument("--allow-test", action="store_true")
        if name == "score":
            p.add_argument("--model", required=True, help="runs/fev_full/<model>")
            p.add_argument("--model-name", default="Alma")
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    FEV_CLONE = a.fev_clone
    if not FEV_CLONE:
        sys.exit("--fev-clone (or FEV_CLONE) must point at a clone of github.com/autogluon/fev")
    if a.cmd == "list":
        tasks, tag = select(bench_tasks(), a.tasks)
        for i, t in enumerate(tasks):
            print(i, t.task_name, tag[t.task_name], t.horizon, t.num_windows, t.seasonality)
        return
    {"export": export, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
