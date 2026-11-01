# Evaluation pipeline: predict once, score many times (plan v2)

Owner: Dr. Tomás Ferreira. Suite contracts (windows, metrics, licences, pins) are in docs/benchmark_contract.md (Iyer); this file does not repeat them.

## 1. Prediction store (`tools/predstore.py`, bench VAL only)

```
runs/preds/<model_id>/meta.json                    model id, script + sha256, ckpt path + sha256, git sha/dirty,
                                                   TSAR_*/PYTORCH_* env, env_info (python/torch/numpy/pandas),
                                                   context length, quantile levels, date, harness headline,
                                                   per task: family, windows, horizon, season, freq, target_sha256
runs/preds/<model_id>/<split>/<task>.npz           q [windows, Q, h] float32 raw; series_id, window ('/' -> '__')
runs/preds/<model_id>/<split>/harness_scores.csv   bench.evaluate's per-task table at write time
runs/preds/<model_id>/<split>/rescore.{csv,json}   written by tools/rescore.py
runs/preds/_targets/<sha256>.npy                   targets, content-addressed, one copy for all models
```
`target_sha256 = sha256(b"float32" + repr(shape) + float32 C-order bytes)`. Model folders never hold targets.

Producers:
- `predstore.py run --model-id X --script train*.py` wraps `bench.evaluate` and execs any train script (use TSAR_EVAL_ONLY + TSAR_BENCH=1 + TSAR_SPLIT=val).
- `predstore.py import <TSAR_DUMP dir> --model-id X` converts tools/run.py and dump_baseline.py dumps.
- `predstore.py builtin snaive|naive`.

The test split is refused by both tools.

## 2. Independent re-scorer (`tools/rescore.py`)

The metric code is written fresh in float64; the definitions are in the docstring. It uses bench only for `load_tasks`, and checks each target against its stored sha256 and window/series alignment. `--check-bench` replays the stored forecasts through `bench.evaluate` and fails if any per-task or aggregate difference exceeds 1e-6.

Results (2026-10-30, VAL, 30 tasks):

| model | rescore wql_rel | mase_rel | cov80_err | vs logged headline | vs bench replay (task / agg) |
|---|---|---|---|---|---|
| b_curr (ckpt e64672ff) | 0.693243 | 0.843669 | 0.094323 | 9.3e-7 | 3.8e-7 / 9.8e-9 |
| syn5_10k (ckpt a0b6b248) | 0.680627 | 0.838080 | 0.110203 | 7.3e-7 | 3.2e-7 / 3.4e-8 |
| snaive (bench) | 1.000000 | 1.000000 | 0.759395 | n/a | 0 / 0 |

The eval-only runs reproduce the `.val.json` records of both checkpoints (b_curr 0.693242, syn5_10k 0.680626). The residual differences of about 1e-7 come from bench summing in float32 while rescore sums in float64.

Baselines: `baselines/bench/dumps/` does not exist. Only per-task CSVs exist, with no forecasts, so no baseline is in the store yet. Filling it needs `tools/dump_baseline.py --bench <name>`, which is a heavy queue job, followed by `predstore.py import`.

## 3. Adapter interface (design only)

Every model emits forecasts once, through one function, into the store. Official scorers are adapters that read the store; they never call the model.

```python
class Forecaster(Protocol):
    model_id: str                      # store key; one id per (weights, inference config)
    quantile_levels: tuple[float, ...] # superset of 0.1..0.9 (fev/GIFT/TIME all accept these)
    def provenance(self) -> dict: ...  # ckpt/weights sha256 or HF repo@revision, script sha, env, context cap
    def predict(self, contexts: list[np.ndarray], horizon: int, season: int, freq: str,
                ) -> np.ndarray: ...   # [n, Q, horizon] float32; NaN in contexts allowed; no future covariates
```

- **Alma checkpoint:** `predstore.py run` (exists): the train script's `predict` under TSAR_EVAL_ONLY.
- **HF baseline:** the `baselines.py` registry functions already have this signature; `dump_baseline.py` → `import`.
- **Statistical model:** the same signature (statsforecast in baselines.py, or bench `_snaive`).

Producing for an external suite (later, after the downloads are approved): `predstore.py emit --suite {fev,gift,time} --config <task>`. The suite's own loader yields windows and the store's `windows`, `series_id` and `window` keep its order. The output goes to `runs/preds/<model_id>/<suite>/<task>.npz` with `[windows, Q, h]` plus `item_id` and `cutoff`, and `meta.json` records the harness pin and data revision.

Per-suite consumer adapters convert the stored arrays into each scorer's input. None of them re-runs inference.
- fev: `fev.Task` predictions as a `datasets.Dataset` with `predictions` and `"0.1"`..`"0.9"` columns, then `task.evaluation_summary`.
- GIFT-Eval: a GluonTS `QuantileForecast(forecast_arrays=q[i], forecast_keys=levels)` per window, then `evaluate_model`. CRPS is approximated by the mean weighted quantile loss over the 9 levels, as GIFT does for quantile models.
- TIME: `save_window_predictions` with `(instances, quantiles, H)`, then `compute_local_leaderboard.py`.

Rules:
- Targets for external suites are content-hashed the same way.
- Each adapter has a self-test: seasonal naive scores 1.0 relative.
- An adapter's score is cross-checked against `rescore.py` on any task the two share, such as fev ETT_1D, within 1e-6 on WQL.
- Leakage flags (fev self-flag, GIFT `testdata_leakage`) live in meta.json per suite.

## 4. Official-scorer smoke tests (2026-10-30; VAL or suite smoke data only, no TEST)

The lead did the downloads under the user's approval, about 26 MB in total; see `data/eval/manifest.json`. Everything here runs with `.venv-eval/bin/python -I`, gets paths as arguments and works offline (`HF_*_OFFLINE=1`). I read the harness code before running it.

The eval venv has no statsforecast and no python-dotenv. The scripts stub `dotenv` with a no-op, and use a deterministic seasonal-naive point forecast with all quantiles equal to it. Point metrics must therefore match the published rows, while interval metrics (WQL, CRPS, MSIS) cannot.

**fev 0.10.0** (`tools/eval_smoke_fev.py`)

Stored forecasts for our VAL task ETT_1D are scored with `fev.Task.evaluation_summary`, using the `fev_bench/tasks.yaml` metric configuration.
- **ETT_1D is the fev-bench task itself:** h 28, 20 windows, cutoff -560, step 28, season 7, 280 forecasts. Our VAL numbers on fev-bench tasks are therefore val-selected for any later fev-bench claim.
- **The bench pooled convention reproduces exactly.** `fev.metrics.WQL/MASE` on all windows pooled as one univariate problem equal rescore's per-task wql and mase within 2e-16 for b_curr, syn5_10k and snaive.
- **The official fev number is a different protocol:** a metric per window and per target column, then a mean. b_curr WQL is 0.2687 official versus 0.1890 pooled; MASE is the same, 1.4006.
- **Check against the published Seasonal Naive** (statsforecast, fev 0.8.0): our snaive gives MASE 1.491547 and WAPE 0.356686, both equal to the published values. SQL is 1.4915 versus 1.3728 published, because there are no intervals.
- **Official SQL on ETT_1D:** b_curr 1.1806, syn5_10k 1.1455. Published: Chronos-2 1.1315, Timestral-209M 1.1215, Seasonal Naive 1.3728. These are single-task numbers, val-selected, and not a claim.

**GIFT-Eval @ 82f89c3** (`tools/eval_smoke_gift.py`)

Windows come from `gift_eval.data.Dataset`. Scoring is gluonts `evaluate_model`, with the metric list and flags from `notebooks/naive.ipynb`. The run covers 6 short configs: us_births D/M/W, covid_deaths, hierarchical_sales D/W.
- MASE equals the published `results/Seasonal_Naive` value exactly for all 6.
- All point metrics (MSE, MAE, MAPE, sMAPE, ND) agree within 6e-8 relative.
- WQL differs as expected: for example, us_births/M is 0.0208 versus 0.0168 published.

**TIME @ c633e23** (`tools/eval_smoke_time.py`)

The run covers Job_Claims/W, Housing_Inventory/M and JOLTS/M at the short horizon. It uses TIME's `Dataset` with `datasets.yaml`, then `save_window_predictions`, then the functions of `compute_local_leaderboard.py`. Its `main()` is not run, because it fetches seasonal-naive results from the HF Hub when none exist locally.
- Seasonal-naive MASE: 2.7214, 0.6840 and 1.6856.
- The overall table runs end to end: naive / seasonal naive is 1.219 on MASE and 1.185 on CRPS.
- **Not checked against a published number:** TIME's published seasonal-naive results live in HF `Real-TSF/TIME-Output`, which was not downloaded.
- Official TIME seasonal naive samples from statsforecast intervals with an unseeded RNG, so its CRPS is not bit-reproducible in any case.

**Open items**
- Before any official GIFT or TIME claim: install statsforecast in `.venv-eval`, which needs approval, or adopt the stored-forecast seasonal-naive baselines.
- Write `predstore emit` per suite (section 3), so Alma forecasts reach these scorers without re-running inference.
