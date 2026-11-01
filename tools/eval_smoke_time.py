"""
TIME official-scorer smoke test: seasonal naive and naive on the 3 local smoke tasks. Windows come
from TIME's timebench Dataset and datasets.yaml; per-window metrics and files from its
save_window_predictions; the overall table from the functions in scripts/compute_local_leaderboard.py
(get_all_datasets_results, compute_ranks, get_overall_leaderboard).

The script imports those leaderboard functions instead of running its main(), which would fetch
seasonal-naive results from the HF Hub when none exist locally. No network calls are made.

    HF_HUB_OFFLINE=1 .venv-eval/bin/python -I tools/eval_smoke_time.py \\
        --repo vendor/eval/TIME --data data/eval/time --out runs/eval_smoke/time

Deviation from the official run: TIME's SeasonalNaivePredictor needs statsforecast, which is not
installed. It also samples 100 draws from statsforecast intervals with an unseeded RNG. This script
uses the identical point forecast (TIME's own NaN handling: forward-fill, then first valid value;
then tile the last season) with all 9 quantiles equal to it. MASE is therefore the official
seasonal-naive MASE. CRPS is the point forecast's (= ND), not the official interval CRPS.
"""
import os
import sys
import json
import types
import argparse
import importlib.util

import numpy as np

DATASETS = ["Job_Claims/W", "Housing_Inventory/M", "JOLTS/M"]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def fill(h):
    h = np.asarray(h, np.float64).copy()
    if np.isnan(h).any():
        mask = np.isnan(h)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        h = h[idx]
        if np.isnan(h).any():
            ok = ~np.isnan(h)
            h = np.nan_to_num(h, nan=h[ok][0] if ok.any() else 0.0)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *x, **k: None)
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, os.path.join(repo, "src"))
    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
    from timebench.evaluation.saver import save_window_predictions
    spec = importlib.util.spec_from_file_location("time_lb", os.path.join(repo, "scripts",
                                                                         "compute_local_leaderboard.py"))
    LB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LB)          # defines functions only; main() is not called

    out_root = os.path.abspath(a.out)
    cfg = load_dataset_config()
    rows = []
    for name in DATASETS:
        st = get_dataset_settings(name, "short", cfg)
        kw = dict(name=name, term="short", prediction_length=st["prediction_length"],
                  test_length=st["test_length"], val_length=st["val_length"], storage_path=os.path.abspath(a.data))
        uni = Dataset(to_univariate=False, **kw).target_dim != 1
        ds = Dataset(to_univariate=uni, **kw)
        m, H = get_seasonality(ds.freq), ds.prediction_length
        for model, season in (("seasonal_naive", m), ("naive", 1)):
            fc = []
            for entry in ds.test_data.input:
                y = fill(entry["target"])
                s = season if len(y) >= season else 1
                fc.append(np.repeat(np.resize(y[-s:], H)[None, :], len(LEVELS), axis=0))
            fc = np.stack(fc)                                      # (instances, Q, H)
            meta = save_window_predictions(dataset=ds, fc_quantiles=fc, ds_config=f"{name}/short",
                                           output_base_dir=os.path.join(out_root, model), seasonality=m,
                                           model_hyperparams={"season_length": season, "note": "point SN, no intervals"})
            met = np.load(os.path.join(out_root, model, name, "short", "metrics.npz"))
            rows.append({"task": f"{name}/short", "model": model, "windows": ds.windows, "h": H, "season": m,
                         "series": meta.get("num_series"), "MASE": float(np.nanmean(met["MASE"])),
                         "CRPS": float(np.nanmean(met["CRPS"]))})
    df = LB.get_all_datasets_results(__import__("pathlib").Path(out_root))
    df = LB.compute_ranks(df, groupby_cols=["dataset_id", "horizon"])
    board = LB.get_overall_leaderboard(df, metric="MASE")
    print(json.dumps({"per_task": rows, "leaderboard": board.to_dict(orient="records")}, indent=1, default=float))


if __name__ == "__main__":
    main()
