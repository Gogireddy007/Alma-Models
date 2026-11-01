# tools/commodity: crude-oil forecasting harness

Pre-registration: `docs/commodity_prereg.md` (the binding protocol). Leakage: `docs/commodity_leakage.md`.
Results: `docs/commodity_results.md`.

## Splits (hard)

| split | dates | use |
|---|---|---|
| TRAIN | everything <= 2025-06-30 | fitting / fine-tuning (only via `data.train_windows`) |
| DEV backtest | origins 2019-01-01 .. 2025-06-30 (targets also <= 2025-06-30) | architecture search, model selection |
| VALIDATION | origins 2025-07-01 .. 2025-10-31 | confirm ONE frozen candidate (+ its direct twin); debugging |
| TEST | origins >= 2025-11-01 | run exactly ONCE, all models together, after the candidate is frozen |

An origin is a date; the model sees only observations strictly before it. Constants are in `tools/commodity/data.py`
(`TRAIN_END`, `DEV_START`, `VAL_START`, `VAL_END`, `TEST_START`).

## Data

`data/commodity/{wti,brent,hh,inv}.csv` (date,value), built by `data.py build` from `data/commodity/raw/` (sha256 and
download date in `data/commodity/manifest.json`). wti = FRED DCOILWTICO (primary), brent = DCOILBRENTEU, hh = DHHNGSP,
inv = EIA WCESTUS1 weekly commercial crude stocks (thousand bbl; not on FRED, taken from eia.gov).
WTI has one non-positive value (2020-04-20, -37.63), inside TRAIN/DEV.

Training windows: `data.train_windows(name, context, horizon, stride=1, start=None, log=False)` returns
`(X[n, context], Y[n, horizon], origin_dates)` from data <= 2025-06-30 only and asserts that no target date is after
2025-06-30 (passing `end` later than that raises).

## Model interface

A model is a callable registered by name in `tools/commodity/models.py`:

```python
def predict(context: np.ndarray, horizon: int, quantiles: list[float]) -> np.ndarray:  # shape [len(quantiles), horizon]
```

- `context`: 1-D float array of raw observed levels (USD/bbl, USD/MMBtu, or thousand bbl), oldest first, last value =
  the last observation before the origin. The harness passes the last 1024 observations (crop further yourself).
- Return quantiles of the raw level at steps 1..horizon. Levels you cannot produce: fill with NaN (scoring uses
  0.1..0.9 for every model; 0.05/0.95 and 0.025/0.975 only where provided).
- Must be deterministic (fix seeds). At inference it must use nothing but `context` (no file reads of data/commodity/,
  no data after the origin). Training data only through `data.train_windows` (<= 2025-06-30).
- Optional attributes on the callable: `predict.batch(contexts: list[np.ndarray], horizon, quantiles) -> [n, nq, h]`
  (faster path), `predict.wants_dates = True` (then `dates=` with the context's dates is passed as a keyword),
  `predict.wants_freq = True` (then `freq="B"` for daily business-day series or `"W"` for weekly is passed),
  `predict.meta = {...}` (checkpoint path and sha256, settings), recorded in the run manifest.

Register: in `tools/commodity/models.py` add `REGISTRY["name"] = factory` where `factory()` returns the callable
(lazy, so unused models are never loaded). Architect modules `tools/commodity/wm_*.py` that define
`register(REGISTRY)` are picked up automatically.

## Running

```
nice -n 10 .venv/bin/python tools/commodity/run_eval.py --split val --models alma_s0,chronos2,...   # debug / confirm
nice -n 10 .venv/bin/python tools/commodity/run_eval.py --split dev --models <candidate>            # selection
```

Predictions are cached in `runs/commodity/<split>/<series>/<model>.npz`; metrics in `runs/commodity/<split>/metrics.json`.
`--split test` refuses to run unless `runs/commodity/FROZEN.json` exists (written when the candidate is frozen), and
writes `runs/commodity/test/RUN.json` (commit sha, timestamp, model shas); a second TEST run is refused.
