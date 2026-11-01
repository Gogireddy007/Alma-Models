"""
GIFT-Eval official-scorer smoke test: seasonal naive on the local smoke configs, scored with GIFT's
own pipeline (gift_eval.data.Dataset windows + gluonts evaluate_model with the metric list and flags
from notebooks/naive.ipynb at 82f89c3), compared with the published results/Seasonal_Naive rows.

Runs in the eval venv, reads only local files, and passes paths as arguments:
    .venv-eval/bin/python -I tools/eval_smoke_gift.py --repo vendor/eval/gift-eval --data data/eval/gift

The eval venv lacks statsforecast and python-dotenv (no further installs). So:
  - dotenv: gift_eval.data calls load_dotenv() only to read GIFT_EVAL; a no-op stub is used, and
    GIFT_EVAL is set to --data.
  - forecaster: the published row used statsforecast SeasonalNaive with prediction intervals. This
    script uses the same point forecast (last-value imputation, then tile the last season) with
    every quantile equal to it. Point metrics (MASE, MAE, MSE, ND, sMAPE, MAPE) must therefore match
    the published row; WQL and MSIS cannot, because there are no intervals.
"""
import os
import sys
import csv
import json
import types
import argparse

import numpy as np

CONFIGS = ["us_births/D", "us_births/M", "us_births/W", "covid_deaths", "hierarchical_sales/D",
           "hierarchical_sales/W"]
POINT = ["MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "ND[0.5]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *x, **k: None)
    os.environ["GIFT_EVAL"] = os.path.abspath(a.data)
    sys.path.insert(0, os.path.join(os.path.abspath(a.repo), "src"))
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.model.forecast import QuantileForecast
    from gluonts.dataset.util import forecast_start
    from gluonts.time_feature import get_seasonality
    from gluonts.transform.feature import LastValueImputation
    from gluonts.ev.metrics import (MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
                                    MeanWeightedSumQuantileLoss)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = [MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
               RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=levels)]

    class SNPredictor:
        def __init__(self, h, m):
            self.prediction_length, self.m, self.imp = h, m, LastValueImputation()

        def predict(self, dataset, **kw):
            for entry in dataset:
                y = np.asarray(entry["target"], np.float32)
                if np.isnan(y).any():
                    y = self.imp(y.copy())
                m = self.m if len(y) >= self.m else 1
                p = np.resize(y[-m:], self.prediction_length)
                yield QuantileForecast(forecast_arrays=np.stack([p] * (1 + len(levels))),
                                       forecast_keys=["mean"] + [str(q) for q in levels],
                                       start_date=forecast_start(entry), item_id=entry.get("item_id"))

    pub = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(a.repo, "results", "Seasonal_Naive",
                                                                      "all_results.csv")))}
    props = json.load(open(os.path.join(a.repo, "notebooks", "dataset_properties.json")))
    out = []
    for name in CONFIGS:
        key = name.split("/")[0].lower()
        freq = name.split("/")[1] if "/" in name else props[key]["frequency"]
        cfg = f"{key}/{freq}/short"
        uni = Dataset(name=name, term="short", to_univariate=False).target_dim != 1
        ds = Dataset(name=name, term="short", to_univariate=uni)
        m = get_seasonality(ds.freq)
        res = evaluate_model(SNPredictor(ds.prediction_length, m), test_data=ds.test_data, metrics=metrics,
                             batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False,
                             seasonality=m)
        mine = {k: float(res[k][0]) for k in POINT + ["mean_weighted_sum_quantile_loss"]}
        p = pub[cfg]
        rel = {k: abs(mine[k] / float(p[f"eval_metrics/{k}"]) - 1) for k in POINT}
        out.append({"config": cfg, "windows": ds.windows, "h": ds.prediction_length, "season": m,
                    "MASE": mine["MASE[0.5]"], "MASE_pub": float(p["eval_metrics/MASE[0.5]"]),
                    "WQL": mine["mean_weighted_sum_quantile_loss"],
                    "WQL_pub": float(p["eval_metrics/mean_weighted_sum_quantile_loss"]),
                    "max_rel_diff_point_metrics": max(rel.values())})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
