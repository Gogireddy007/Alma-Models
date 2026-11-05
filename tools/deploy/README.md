# tools/deploy: inference-only deployment and its audit

Write-up: [`docs/deploy_2026_10_11.md`](../../docs/deploy_2026_10_11.md). Outputs: `runs/deploy_2026_10_11/`.

| File | Role |
|---|---|
| `common.py` | Guards used by every deploy run: `no_training()` (optimizer step / backward raise), parameter hashing at load and at the end, checkpoint pins (`SHA256SUMS` + `tools/forward/FREEZE.json`), no verdict below 10 independent units |
| `deploy_oil.py`, `deploy_power.py`, `deploy_gas.py`, `deploy_fin.py` | Run the frozen model and every competitor on Oct 2026 + Nov 2026; save sealed forecasts (`preds/*.npz`, quantiles only) and `results.json` |
| `gas_daily_noweather.py` | Builds `data/gas/daily.csv` when Open-Meteo is unreachable (the storage models do not read weather from it) |
| `funding_effect.py` | October crypto with and without the funding data that was missing from the pinned TEST data |
| `audit_deploy.py` | **Independent** audit (imports nothing from the deploy scripts): pins, no-fit, values re-derived from RAW files, every metric recomputed, future-scramble leak tests, sample-size rule, sealing. Exit code 1 on any failure |
| `make_report.py` | `RESULTS.md` built from the JSON (no hand-typed numbers) |
| `check_doc_numbers.py` | Every decimal number in the write-up must trace to the outputs |
| `score_exante.py` | Grades the sealed ex-ante forecasts after publication; refuses if a sealed file changed |
| `run_all.sh` | The whole chain |

Realised market values are stored only in local, git-ignored `*_LABELS_local*.npz` files, because raw market data
is not redistributed. The audit re-derives them from the raw downloads.
