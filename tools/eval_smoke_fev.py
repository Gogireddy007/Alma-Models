"""
Official-fev smoke test (VAL only): score a stored prediction (tools/predstore.py) for the bench task
ETT_1D with fev 0.10.0's own Task/metric code and compare with tools/rescore.py. No inference.

Runs in the separate eval venv (fev is not in the project venv); reads only local files:
    HF_DATASETS_OFFLINE=1 .venv-eval/bin/python -I tools/eval_smoke_fev.py \\
        --store runs/preds/b_curr --data data/bench/fev/ETT_1D.parquet

Two numbers per metric:
  fev_official  fev.Task.evaluation_summary: metric per window (each target column scored separately,
                then averaged over the 7 columns), then averaged over the 20 windows.
  fev_pooled    fev.metrics.WQL / MASE on all windows and series pooled as one univariate [N, H, 1]
                problem, which is bench's protocol; it must equal rescore's wql / mase.
"""
import os
import sys
import json
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="ETT_1D")
    a = ap.parse_args()
    import fev
    from fev import metrics as FM

    meta = json.load(open(os.path.join(a.store, "meta.json")))
    rec = meta["splits"]["val"]["tasks"][a.task]
    levels = [float(x) for x in meta["quantile_levels"]]
    z = np.load(os.path.join(a.store, "val", a.task.replace("/", "__") + ".npz"))
    q, sids, wins = z["q"].astype(np.float64), list(z["series_id"]), z["window"]
    q = np.sort(np.where(np.isfinite(q), q, 0.0), axis=1)          # bench/rescore sanitise rule
    H, m, W = rec["horizon"], rec["season"], int(wins.max()) + 1
    cols = sorted({s.split("/")[1] for s in sids})
    lookup = {(s, int(w)): i for i, (s, w) in enumerate(zip(sids, wins))}

    task = fev.Task(dataset_path=a.data, horizon=H, num_windows=W, initial_cutoff=-W * H,
                    window_step_size=H, target=cols, seasonality=m, eval_metric="SQL",   # fev_bench/tasks.yaml config
                    extra_metrics=["MASE", {"name": "WAPE", "epsilon": 1.0}, {"name": "WQL", "epsilon": 1.0}],
                    quantile_levels=levels)
    preds_per_window, ys, ps, pasts, plens = [], [], [], [], []
    for k, win in enumerate(task.iter_windows()):
        past, future = win.get_input_data()
        _, _, test = win._get_past_future_test_data()
        items = list(test[task.id_column])
        per_col = {}
        for c in cols:
            rows = []
            for j, it in enumerate(items):
                i = lookup[(f"{it}/{c}", k)]
                r = {"predictions": q[i, levels.index(0.5)].tolist()}
                r.update({str(lv): q[i, li].tolist() for li, lv in enumerate(levels)})
                rows.append(r)
                ys.append(np.asarray(test[j][c], np.float64)); ps.append(q[i].T)
                hist = np.asarray(past[j][c], np.float64)
                pasts.append(hist); plens.append(len(hist))
            per_col[c] = rows
        preds_per_window.append(per_col)
    summ = task.evaluation_summary(preds_per_window, model_name=meta["model_id"])

    y = np.stack(ys)[:, :, None]                                    # [N, H, 1]
    qp = np.stack(ps)[:, :, None, :]                                # [N, H, 1, Q]
    kw = dict(y_true=y, y_pred=qp[..., levels.index(0.5)], y_past=np.concatenate(pasts)[:, None],
              y_past_lengths=np.asarray(plens), q_pred=qp, seasonality=m, quantile_levels=levels)
    pooled = {"WQL": FM.WQL().compute(**kw), "MASE": FM.MASE().compute(**kw)}

    out = {"task": a.task, "model_id": meta["model_id"], "fev_version": fev.__version__,
           "fev_official": {k: summ[k] for k in ("SQL", "MASE", "WQL", "WAPE")}, "fev_pooled": pooled}
    rs = os.path.join(a.store, "val", "rescore.csv")
    if os.path.exists(rs):
        import pandas as pd
        r = pd.read_csv(rs).set_index("dataset").loc[a.task]
        out["rescore"] = {"WQL": float(r["wql"]), "MASE": float(r["mase"])}
        out["abs_diff_pooled_vs_rescore"] = {k: abs(pooled[k] - out["rescore"][k]) for k in pooled}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
