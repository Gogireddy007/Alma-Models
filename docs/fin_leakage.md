# Alma-Fin leakage and contamination record

Author: Alma-Fin lead (agent), 2026-11-01. Binding rules: `docs/fin_prereg.md` sections 2-3. Code checks:
`tools/fin/test_infoset.py` (output `runs/fin/test_infoset.txt`).

## 1. Look-ahead (information set)

| risk | control |
|---|---|
| same-bar execution | decision at D 00:00 UTC; fill at the 00:05-00:10 bar's (O+H+L+C)/4 (one full 5-min bar skipped). Re-derived from the raw zips in test 3 |
| label inside the context | label RV(D) uses bars opening 00:05..23:55 of D; the context ends with day D-1 / the hour ending D 00:00 (asserted in `build_context`; perturbation test 1) |
| funding timing | settlement at D+1 00:00 is charged to the position opened at fill(D) and appears in features only from decision D+1 on (hourly row ending at D+1 00:00) |
| online learners peeking | `observe(rec)` is called only after the decision for D is recorded; `rec.available_from = D+1 00:00 <= next decision` asserted; toy continual model test (decisions up to a perturbation day bit-identical) |
| models reading files | audit hook refuses open() under `data/fin/` and `runs/fin/` (except `*/ckpt/`) during predict/act/observe (test 6 with a snooping model; prereg A1.3) |
| fitting on VAL/TEST | `models.train_view()` asserts the 2025-06-30 cut and the 16 training coins; strict TEST mode loads sealed rows only with the auditor-GO flag |
| TEST inspection by the lead | TEST rows live only in `*_SEALED.parquet`; build prints counts and hashes only; `data.load()` asserts the open files end before 2025-11-01 |
| boundary values | the 2025-11-01 00:00 funding settlement is kept out of the open 2025-10-31 daily row; it does appear in the open hourly row 2025-10-31 23:00 (known at the first TEST decision time, so legitimate for contexts; disclosed) |

## 2. Universe and survivorship

Fixed once from data <= 2025-10-31 (30-day quote volume, listed by 2024-11-01). TEST is survivorship-free: no TEST
information selects coins; delisted coins stay (closed at their last close). DEV and VAL are not survivorship/selection-free
(the rank uses 2025-10 volume, i.e. VAL days, and requires coins alive in 2025-10 (prereg A1.2); coins that died
before 2025-10 are absent; the S3 listing shows 27 symbols that end before 2025-06, e.g. LUNAUSDT,
SRMUSDT, MATICUSDT). Held-out coins (every third by rank) were fixed by rule, not by looking at any model.

## 3. Competitor contamination

| model | release / pretraining range | status |
|---|---|---|
| Kronos-small (+ tokenizer) | pretrained on K-lines from 45 exchanges up to June 2024 (crypto incl.) [KR] | has seen DEV, not VAL/TEST; DEV ranks flattering |
| Chronos-Bolt-base | 2024-11 | clean for VAL/TEST |
| TiRex | 2025-05 | clean for VAL/TEST |
| Chronos-2 | released 2025-10-17; corpus cut-off not public | weights predate TEST; may overlap VAL |
| Timestral-5M | released 2026-10-15 | CONTAMINATED for TEST; reported, never in a "beats" claim |
| LightGBM / HAR / GARCH / rules | fitted on DEV only (or rolling on context history) | clean |

## 4. Prior exposure in this repo

No crypto or US-ETF series was used by any earlier Alma round (grep of tools/, docs/, baselines/ for btc, bitcoin,
binance, SPY, crypto: only a mention of the GIFT `bitcoin` set as "not chosen" in `docs/gift_clean_corpus.md`). No
Alma checkpoint of earlier rounds is used. The architect's models are trained from scratch on DEV.

## 5. Data-source caveats

Binance archive: per-file sha256 checked against Binance's CHECKSUM; the 2026-10-28 daily files were missing at
download time. Yahoo (ETFs): unofficial endpoint, values can be revised by the provider; our raw JSON snapshot is
pinned by sha256 and is what every run uses.
