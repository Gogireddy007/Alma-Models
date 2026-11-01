# tools/fin: Alma-Fin, crypto-perp volatility + trading harness

Binding protocol: `docs/fin_prereg.md`. Research basis: `docs/fin_research.md` (f285496). Lead-owned:
`tools/fin/**` except `tools/fin/wm_*.py`, `tools/fin/agents/**`, `tools/fin/search/**` (architect). Data:
`data/fin/` (raw + manifest + processed). Runs: `runs/fin/` (architect: `runs/fin/search/`, `runs/fin/FROZEN.json`).

## Data

- `fetch.py` downloads Binance public data (https://data.binance.vision, no key): USD-M perpetual 5-min
  klines (with taker-buy volume) and funding rates, monthly zips (+ daily zips for 2026-10). Every file: url, UTC
  download time, sha256, and Binance's own `.CHECKSUM` verified (mismatch -> rejected) in `data/fin/manifest.json`.
  The S3 listing (`data/fin/listing_um.json`) keeps delisted symbols (e.g. LUNAUSDT, SRMUSDT, MATICUSDT).
- `fetch_etf.py` downloads daily OHLCV of SPY, QQQ and the 11 SPDR sector ETFs from the public Yahoo chart endpoint
  (Stooq now serves a bot check, which we do not bypass). Raw JSON + sha256 in the same manifest.
- `universe.py` fixes the universe once from data <= 2025-10-31 (rule in its docstring and prereg section 1):
  24 perps by 30-day quote volume, listed by 2024-11-01; held-out = ranks 3, 6, ..., 24 -> `data/fin/universe.json`.
- `data.py build` -> `data/fin/crypto_hourly.parquet`, `crypto_daily.parquet`, `etf_daily.parquet` (pre-TEST rows)
  and `*_SEALED.parquet` (rows dated >= 2025-11-01; loaded only by strict TEST mode). `build_report.json`: gaps,
  outliers, delistings, row counts, sha256 (pinned in `run_eval.DATA_PINS`).

## Clock and information set (enforced by `harness.build_context`; tested by `test_infoset.py`)

Crypto, decision day D:
- decision at **D 00:00 UTC**; the context holds daily rows of days <= D-1 and hourly rows of hours ending <= D 00:00;
- fill at the **00:05-00:10 UTC bar of D**, price (O+H+L+C)/4 (the 00:00-00:05 bar is skipped: no same-bar fill);
- position held until the fill of D+1; funding settled in (D 00:00, D+1 00:00] is charged to it (long pays a
  positive rate); costs: taker fee 5 bp + slippage 2 bp (BTC, ETH) / 5 bp (others) per side, on |delta position|;
- labels of D: RV(D) = sum of squared 5-min log returns of the bars opening 00:05..23:55 (needs >= 274 bars), and
  `y_vol(D) = 1[log RV(D) > mean(log RV(D-7..D-1))]`; complete at D+1 00:00.
ETF, session D: decision at the close of session D-1, fill at the open of D (dividend-adjusted), held to the open of
D+1; RV = Garman-Klass variance of the session; cost 2 bp per side.

## Plug-in interface (forecasters, agents, continual learners)

Register a factory: `REGISTRY["name"] = factory` in `models.py`, or (architect) define `register(REGISTRY)` in
`tools/fin/wm_*.py` or `tools/fin/agents/*.py` (auto-imported; an import error lands in `models.LOAD_ERRORS` and
aborts strict TEST mode). The factory returns an object:

```python
class MyForecaster:
    kind = "forecaster"
    meta = {"ckpt": "data/ckpt/x.pt", "ckpt_sha256": "...", ...}      # recorded in the run manifest
    def predict(self, ctx) -> dict:
        return {"p_up": np.ndarray[m],            # P(y_vol(D) = 1), optional
                "q": np.ndarray[m, 13],           # quantiles QS of log RV(D), optional (NaN where not produced)
                "p_ret_up": np.ndarray[m]}        # P(return of D > 0), optional (P4)

class MyAgent:
    kind = "agent"
    def act(self, ctx) -> np.ndarray[m]:          # target positions in [-1, 1] (clipped), one per ctx["symbols"]

# CONTINUAL (either kind): add
    def observe(self, rec) -> int:                # called after the decision for D, before the decision for D+1
        ...                                       # rec = day D's completed row; return the number of parameter
        return n_updates                          # updates made (logged per day in the forecast file)
```

`ctx` (fresh read-only copies; nothing dated at or after the decision time):
- `market` ('crypto' | 'etf'), `day` (D), `decision_time`, `symbols` [m], `groups` [m] ('train' | 'heldout' | 'etf'),
  `quantiles`, `lookback` (7);
- `days` [L] and `daily[field]` [L, m] for the last L = 400 days <= D-1. Crypto fields: open, high, low, close,
  quote_volume, taker_ratio, rv, logrv, ret (close-to-close log), funding (sum settled that day), fill, n5. ETF:
  open, high, low, close, volume, rv, logrv, ret (adjusted), fill. NaN before listing / on gaps.
- crypto only: `hours` [672] and `hourly[field]` [672, m] (last 28 days of hours ending <= D 00:00): open, high,
  low, close, volume, quote_volume, count, taker_ratio, rv5 (sum of squared 5-min returns in the hour), n5,
  funding (settled at the hour's close), ret.

`rec` (online only): `day`, `symbols`, `row[field]` [m] (day D's daily fields), `hourly[field]` [24, m] (crypto),
`available_from` (D+1 00:00).

Rules:
- Deterministic (fix seeds); identical outputs for identical inputs, whatever the batch / call order.
- Use nothing but the arguments at inference: while `predict`/`act`/`observe` run, an audit hook raises
  `PermissionError` on any open() under `data/fin/` (checkpoints in `data/ckpt/` are fine).
- Direction for P1 is derived by the harness: `p_up > 0.5` if given, else median of `q` > mean(log RV(D-7..D-1));
  ties -> DEV majority.
- Do not adapt to the evaluation symbols' identity: no asset-ID embeddings for zero-shot models (prereg section 7).
  A zero-shot (portability) run is scored with `observe` disabled.

## Training data (DEV only)

`models.train_view(market="crypto", group="train")` -> a `harness.Panel` cut (by assertion) at decision day
2025-06-30 and to the 16 training coins. `group=None` gives all 24 coins (only for per-asset twins and baselines that
the prereg allows; never for a zero-shot model). For contexts inside training use
`harness.build_context(P, i)` on that panel: the information set is then identical to evaluation. ETFs are never
used for training (transfer target only). Anything fitted on data after 2025-06-30 (VAL) is a prereg violation,
except online updates through `observe` during a VAL/TEST run.

## Running

```
nice -n 10 .venv/bin/python tools/fin/test_infoset.py [--models har,lgbm]
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val --models har,lgbm
nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val --metrics-only
nice -n 10 .venv/bin/python tools/fin/run_eval.py --split val --rehearse     # TEST procedure, fresh dir, on VAL
nice -n 10 .venv/bin/python tools/fin/run_eval.py --split test --i-have-auditor-go   # ONCE; needs FROZEN.json
```

Splits (decision days): DEV backtest 2022-07-01..2025-06-30 (training: labels <= 2025-06-30); VAL
2025-07-01..2025-10-31; TEST 2025-11-01..2026-10-30 (sealed). Outputs: `runs/fin/<split>/<market>/<model>.npz`
(`p_up`, `q`, `p_ret_up` or `pos`, `days`, `updates`, `meta`), `metrics.json`, `report.md`.
Strict TEST mode refuses without `--i-have-auditor-go`, without a committed `runs/fin/FROZEN.json`, with a dirty
`tools/fin` / prereg (incl. untracked files), with existing forecast files or RUN.json, with `--models/--force`, or
if any pinned data file's sha256 differs; preflight resolves every model before RUN.json is written; a crash after
that consumes the TEST.
