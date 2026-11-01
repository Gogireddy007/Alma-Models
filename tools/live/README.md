# tools/live: Alma-Live Gas (paper-only)

This is a live US natural-gas world model. It runs **paper trading only**: no broker API exists anywhere in this
tree, and the user decides separately whether real money is ever used.

- Binding docs: `docs/live_prereg.md` (forward test), `docs/live_leakage.md` (vintage risks), `docs/live_ops.md`
  (schedule, operations).
- Red-team log: `docs/live_redteam.md`.
- Ownership:
  - The data lead owns `tools/live/**`, except `wm_*` and `search/**`.
  - The architect owns `tools/live/wm_*.py`, `tools/live/search/**` and `runs/live/search/**`.
  - The red team owns `tools/live/redteam/**`.

## Clock (single source of truth: `common.py`)

- `common.decision_time(day)` is **15:30 ET** on trading day d. The ctx holds only records with
  `published_at <= t`. In live/paper mode it also requires `fetched_at <= t`.
- `common.fill_time(day)` is the **16:00 ET close** of d, plus 10 bp per side.
  - The position is held from close(d) to close(d+1).
  - Every P&L is reported at 1x, 2x and 3x costs.
- `common.trading_days(a, b)` and `common.decision_times(a, b)` give the calendar.
- Primary instrument: UNG. Secondary: BOIL. NG futures are used for research and features only, with the calendar
  roll rule `common.ng_active_contract`.

## Data

```
nice -n 10 .venv/bin/python tools/live/update.py            # storage, px, cpc (+ paused: cot, edgar, om); idempotent
nice -n 10 .venv/bin/python tools/live/test_infoset.py      # must print ALL PASSED
```

- Raw responses go to `data/live/raw/<source>/<stamp>_<name>`. They are append-only, and an identical response is
  not stored twice.
- Point-in-time records go to `data/live/pit/<source>.csv`, using the schema in `common.py`:
  `source, series, obs, value, published_at, fetched_at, vintage, extra`.
  - These files are append-only.
  - Each file's sha256, byte length and row count are pinned in `data/live/manifest.json`.
  - Before every append, the existing prefix is checked against the pin.
- Arrivals are logged to `data/live/update_log.jsonl`.

| source | content | published_at |
|---|---|---|
| storage | EIA WNGSR: L48 and 5 regions, levels and `<s>.chg_printed`. Vintages: `first_release` (from revisions.xls, 2015-06-19+), `revised` (current value), `current_backfill` (pre-2015-06-19) | first print at Thu 10:30 ET, with EIA alternates at 12:00 ET and pre-2025 holiday weeks at F+10 12:00 ET; revisions at the next report's release; the 33-week resubmission at 2026-04-23 14:30Z |
| px | Yahoo daily OHLCV for UNG, BOIL, NG=F and the live NG contracts. Bars are **unadjusted**; `<sym>.split` records hold the splits | bar of session d at d 18:00 ET; split at the ex-date 00:00 ET |
| cpc_fcst | CPC 7-day CONUS gas-weighted HDD / population-weighted CDD forecasts, per issue (`vintage = issue:YYYY-MM-DD`), daily 2014-01..now | HTTP Last-Modified if it is within 3 days of the issue (median issue + 15.0 h), otherwise issue + 1 day 17:00Z |
| cpc_obs | CPC observed CONUS degree days (REALIZED weather, used only for normals and scoring) | day + 1 18:00Z |
| cot, edgar_filing, edgar_fact, om_fcst, curve | **paused** by the manager's re-scope; the code and files are kept but not loaded by default | see `sources.py` |

## Store: `store.py`

```python
from store import Store, storage_labels
S = Store()                                   # core sources; Store(sources=store.ALL_SOURCES) for paused ones
ctx = S.asof(common.decision_time(day))       # history / research
ctx = S.asof(common.decision_time(day), live=True)   # paper/live: also fetched_at <= t
S.visible(source, t, live)                    # fast path: latest visible vintage per (series, obs)
S.labels(days)                                # outcomes, NEVER in ctx: ung_ret, boil_ret, ng_cc, ng_oo, ung_rv5, available_at
storage_labels(S)                             # storage target per report week: dev, expectation, released_at, origin_day
```

ctx keys:
- `t`, `day`, `decision_time`, `fill_time`, `split` (dev | val | exposed), `live`.
- `px[sym]`: DataFrame of daily bars through d-1, split-adjusted as of t, with a `published_at` column.
- `ng_roll`: the active contract and the Yahoo switch days.
- `storage`: levels, `<s>.chg_printed` and `released_at`.
- `next_storage`: `{week, expectation_l48}`, where expectation_l48 is the naive 5-year mean of the printed change for
  the same week.
- `cpc_fcst`: the last 30 issues, as a long frame. Any row whose valid day had already ended at publication is
  dropped.
- `cpc_obs`: about 11 years.
- `features`: a dict over `store.FEATURES`.
- `meta`: `n_records`, `max_published_at`, `max_fetched_at`, and `input_sha256` (a canonical hash of everything in
  ctx).

## Plugging in a model (architect)

Create `tools/live/wm_<name>.py` with `register(REGISTRY)` that adds a factory: `REGISTRY["wm_x"] = lambda: X()`.
`models.py` auto-imports it, and any import error goes to `models.LOAD_ERRORS`.

```python
class X:
    name = "wm_x"
    meta = {"ckpt": "data/ckpt/x.pt", "ckpt_sha256": "...", "frozen_commit": "..."}
    def predict(self, ctx) -> dict:
        return {"ung_q": {0.05: ..., ..., 0.95: ...},    # quantiles of UNG log return close(d)->close(d+1)
                "rv5_q": {...},                          # quantiles of log RV5 (sum sq. UNG returns d+1..d+5)
                "stor_dev_q": {...},                     # quantiles of dev(W), W = ctx["next_storage"]["week"]
                "p_up": float, "mu": float}              # optional
    def act(self, ctx) -> dict:                          # target weights in [-1, 1]; gross <= 1; BOIL |w| <= 0.5
        return {"UNG": w}
    def observe(self, outcome) -> int:                   # optional continual update, called after the outcome is
        ...                                              # available and before the next decision:
                                                         # {"day", "labels": {...}, "available_at"}
```

- Quantile levels are `models.QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)`.
- `predict`, `act` and `observe` run under `common.no_data_access()`. An audit hook raises `PermissionError` on any
  open() under `data/live`, `data/gas`, `data/fin` or `runs/live`; `*/ckpt/` is allowed.
- Models get data **only** through ctx. Training (DEV only) may use `S.asof` / `S.labels` /
  `storage_labels` on DEV days. Never fit on days > 2024-06-30, or on labels completed after that.
- Be deterministic: fix seeds.

## Harness: `run_eval.py`

```
nice -n 10 .venv/bin/python tools/live/run_eval.py --split dev  [--models fc_hist,fc_ewma,wm_x] [--trading rules|none]
nice -n 10 .venv/bin/python tools/live/run_eval.py --split val
```

- Splits:
  - DEV is ≤ 2024-06-30, scored from 2017-01-03.
  - VAL is 2024-07-01..2025-10-31.
  - Days from 2025-11-01 on are **EXPOSED**. They are refused unless `--allow-exposed` is given, and then labelled
    as such.
- Primary metrics: quantile score (mean pinball loss) for return, log RV5 and storage dev. Each comes with a block
  bootstrap CI, skill vs `fc_hist` with a paired bootstrap p-value, and 90% coverage. Storage also gets median-sign
  accuracy with a Wilson CI.
- Secondary metrics (trading): direction accuracy with Wilson CI; Sharpe at 1x/2x/3x costs with a bootstrap CI;
  PSR; DSR (k = number of strategies); max drawdown; turnover.
- Baselines:
  - Rules: buy_hold, short_ung_carry, tsmom_5/20/60, seasonal, weather_rev, storage_surprise.
  - Forecasters: fc_hist, fc_ewma.
  - Optional via `--trading` (code kept): ridge, lgbm (single thread), dqn, ppo (torch), chronos2, tirex (offline
    HF cache).
- Outputs go to `runs/live/eval/<split>/`: `report.md`, `forecast_metrics.csv`, `trading_metrics.csv` and
  `manifest.json` (the data pins).

## Paper broker: `paper.py` (no real orders)

```
.venv/bin/python tools/live/paper.py decide --book <b> --model <m>   # 15:30-15:50 ET only (wall clock)
.venv/bin/python tools/live/paper.py settle --book <b>               # after 18:00 ET (needs update.py --only px)
.venv/bin/python tools/live/paper.py verify|status --book <b>
.venv/bin/python tools/live/paper.py resume --book <b> --note "reviewed by ..."
```

- The ledger is `runs/live/paper/<book>/ledger.jsonl`: an append-only hash chain (sha256 over prev + canonical
  JSON).
  - The genesis line fixes the mode, costs, limits and clock.
  - Each decision line carries the model meta, the ctx input sha256, the data manifest sha256 and git HEAD.
  - Live decide and settle auto-commit the ledger file. This is the external anchor: `verify` checks the committed
    prefix.
- Risk:
  - A 2% daily loss sets halt_day, which forces the next decision flat.
  - A 10% drawdown sets halt_review: the book stays flat until `resume` is run with a human note.
  - Stale inputs force the decision flat. Stale means: UNG bar older than d-1, storage older than 9 days, or CPC
    older than 3 days.
- Splits are applied to held shares on their ex-date.
