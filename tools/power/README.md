# tools/power: Alma-Power, German day-ahead electricity price harness

Binding protocol: `docs/power_prereg.md`. Lead-owned: `tools/power/**` except `tools/power/wm_*.py` and
`tools/power/search/**` (architect). Data: `data/power/` (raw + manifest). Runs: `runs/power/`.

## Data

Source: Fraunhofer ISE Energy-Charts public API (`https://api.energy-charts.info`), fetched by `fetch.py` in yearly
raw JSON files (`data/power/raw/<series>/<year>.json`) with url, UTC download time and sha256 in
`data/power/manifest.json`. `data.py build` turns them into `data/power/hourly.csv` (one row per UTC hour:
`utc, price, load, wind_onshore, wind_offshore, solar, date, slot, zone, price_res`) and `data/power/gaps.json`
(every missing / incomplete hour, by series; nothing is forward-filled).

- price: day-ahead auction price, EUR/MWh. Zone DE-AT-LU for delivery days <= 2018-09-30, DE-LU from 2018-10-01.
  Hourly products until 2025-09-30; 15-min products from 2025-10-01 (hourly value = mean of the 4 quarter-hours).
  Negative prices are kept as they are.
- covariates (actuals, MW, public net): `load`, `wind_onshore`, `wind_offshore`, `solar` (15-min -> hourly mean).
- The API's TSO day-ahead forecasts (`fc_*`) are downloaded for documentation only: they carry no publication-time
  (vintage) field, so no pre-registered model may use them.

## Panel and time convention

`data.panel()` -> `dates [n] (datetime64[D])`, `P [n, 24]` price, `C [n, 24, 4]` covariates (order `data.COVS`), flags.

- Day = local Europe/Berlin **delivery day** of the auction; slot s = local clock hour 0..23. All arithmetic is in
  UTC; the slot is the local start hour of the UTC hour.
- Spring-forward day: slot 2 does not exist -> NaN (`flags["dst_gap"]`). Fall-back day: slot 2 = mean of the two
  02:00 hours (`flags["dst_dup"]`). Targets are never imputed; a NaN target slot is excluded from scoring.

## Information set (enforced by `data.build_context`, assertions with explicit UTC timestamps)

The forecast for delivery day D is made at the gate, 12:00 local on D-1.

- Prices: days <= D-1 only (day D-1's prices cleared at noon D-2).
- Fundamentals (load, wind, solar actuals): days <= D-2 only (conservative: D-1 morning actuals exist before noon
  D-1 but publication lags are not recorded by the API). In the covariate array the rows of day D-1 are NaN.
- Calendar facts about D (date, weekday, public holidays) are allowed: they are known in advance.
- Nothing about day D or later, ever. TSO forecasts: not used (no vintages).

## Model interface

Register a factory in `tools/power/models.py` (`REGISTRY["name"] = factory`, lazy), or define
`register(REGISTRY)` in an architect module `tools/power/wm_*.py` (picked up automatically; an import error is
recorded in `models.LOAD_ERRORS` and aborts strict TEST mode).

```python
def predict(context_prices: np.ndarray,          # [T] EUR/MWh, T = 24 * ctx_days, oldest first; last value = slot 23 of D-1
            context_covariates: np.ndarray|None, # [T, 4] (load, wind_onshore, wind_offshore, solar); rows of D-1 NaN
            horizon: int = 24,                   # the 24 slots of day D
            quantiles: list[float] = QS) -> np.ndarray:   # [len(quantiles), 24]
```

- The harness passes `ctx_days = 736` days (17,664 hours: enough for a 728-day LEAR calibration + 7 lag days).
  Crop as you like. Contexts may contain NaN where raw data are missing (rare; see gaps.json); the spring-forward
  slot is linearly interpolated in model inputs only.
- Quantile grid `QS = [0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975]` (13 levels). Fill levels you cannot produce
  with NaN; scoring uses 0.1..0.9 for everyone (0.025/0.975 coverage only where provided).
- Deterministic (fix seeds). At inference use nothing but the arguments (no reading `data/power/`).
- Optional attributes: `predict.batch(list_of_prices, list_of_covs, horizon, quantiles, **kw) -> [n, nq, 24]`;
  `predict.wants_dates = True` -> keyword `dates=` (list of dicts `{"ctx_dates": [ctx_days], "target_day": D}`
  per origin; calendar only); `predict.uses_covariates = True|False` (recorded); `predict.meta = {...}` (checkpoint
  path + sha256, settings) recorded in the run manifest.
- Point models return the same value in every quantile row or NaN outside the median.

Direction is derived by the harness only (never by the model): median vs reference (prereg section 4).

## Training data (DEV only)

- `data.train_panel()` -> `(dates, P, C)` for delivery days <= 2025-06-30 (asking later raises).
- `data.train_windows(ctx_days=28, start=None, stride=1)` -> `X [n, 24*ctx_days], XC [n, 24*ctx_days, 4], Y [n, 24],
  target_dates`; built with `build_context`, so the information set is identical to evaluation; asserts no target
  after 2025-06-30. For long contexts call `build_context(i, ctx_days, *train_panel(), flags)` lazily.

## Running

```
nice -n 10 .venv/bin/python tools/power/run_eval.py --split val --models naive_epf,lear,chronos2,...   # debugging
nice -n 10 .venv/bin/python tools/power/run_eval.py --split dev --models <candidate>                   # selection
nice -n 10 .venv/bin/python tools/power/run_eval.py --split val --rehearse      # TEST procedure, fresh dir, on VAL
nice -n 10 .venv/bin/python tools/power/run_eval.py --split test                # ONCE, needs runs/power/FROZEN.json
```

Splits (target delivery days): DEV backtest 2024-07-01..2025-06-30 (training: anything <= 2025-06-30);
VAL 2025-07-01..2025-10-31; TEST 2025-11-01..last complete day of the pinned data snapshot.
Forecasts: `runs/power/<split>/<model>.npz` (`q [n_days, 13, 24]`, `days`); metrics: `runs/power/<split>/metrics.json`
and `report.md`. Strict TEST mode (as in tools/commodity): refuses without `runs/power/FROZEN.json`, with a dirty
`tools/power` / prereg, with any existing forecast file, with `--models/--force`, on a second run, or if the data
snapshot sha256 differs from the pin; preflight resolves every model before `RUN.json` is written; a crash still
consumes the TEST.
