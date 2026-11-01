# Alma-Power (German day-ahead electricity prices): pre-registration

Author: Alma-Power lead (agent), 2026-10-31. Branch `alma/investor-sprint`. Harness: `tools/power/`
(interface and information-set rules: `tools/power/README.md`).

Status: written and committed BEFORE any forecast or score of any kind was computed on any window (DEV, VAL or
TEST). At the time of writing the raw data were being downloaded; the only looks at the data were API probes
(field names, resolution, the zone split, the 15-min switch) and missing-value checks. Every change after this
commit is a dated amendment at the bottom, made before the TEST run, with its reason.

## 1. Data

Source: Fraunhofer ISE Energy-Charts public API (`api.energy-charts.info`, no key), downloaded 2026-10-31 by
`tools/power/fetch.py`; raw yearly JSON with url, UTC download time and sha256 in `data/power/manifest.json`;
processed by `tools/power/data.py build` into `data/power/hourly.csv` (sha256 pinned in `run_eval.DATA_SHA256` when
the harness is committed; strict TEST mode refuses a different file) and `data/power/gaps.json`.

| series | endpoint | use |
|---|---|---|
| day-ahead price, EUR/MWh | `/price?bzn=DE-AT-LU` (delivery days <= 2018-09-30), `/price?bzn=DE-LU` (from 2018-10-01) | target and price inputs |
| load, wind onshore, wind offshore, solar (actual public net, MW) | `/public_power?country=de` | fundamentals (lagged, section 2) |
| TSO day-ahead forecasts of load / wind / solar | `/public_power_forecast?forecast_type=day-ahead` | downloaded for documentation only; NOT used (no publication-time / vintage field) |

- Time: all arithmetic in UTC. A day is a local (Europe/Berlin) delivery day of the auction; slot s = local clock
  hour 0..23. Spring-forward day: slot 2 does not exist (NaN, never filled in targets). Fall-back day: slot 2 = mean
  of the two 02:00 hours.
- Resolution: 15-min values are averaged to the hour; an hour needs all its sub-hourly values, otherwise NaN, and
  every such hour is listed in `gaps.json`. No forward fill. Day-ahead prices are hourly products until
  2025-09-30 and 15-min products from 2025-10-01 (hourly value = mean of the four quarter-hour prices); so DEV is
  entirely in the hourly-product regime, VAL contains one month (October 2025) and TEST is entirely in the 15-min
  regime. This regime change is disclosed with every result.
- Zone: DE-AT-LU until 2018-09-30, DE-LU from 2018-10-01 (structural break inside DEV, disclosed).
- Negative prices are kept. No log transform anywhere in scoring (levels, EUR/MWh). Models may transform internally
  (LEAR/ARX use asinh, section 5).
- Vintage: values as published on the download date; the generation/load actuals can be revised by the TSOs (current
  vintage used, disclosed). Day-ahead prices are final auction results.

## 2. Information set (the leakage rule; enforced in code)

The forecast for delivery day D is made at the auction gate, 12:00 local time on D-1.

1. Prices: only delivery days <= D-1. (Day D-1's prices were published after the auction at noon on D-2.)
2. Fundamentals (actual load, wind, solar): only delivery days <= D-2. Actuals for the morning of D-1 exist before
   noon D-1 in principle, but the API records no publication times, so we use the conservative rule.
   Actual load, wind and solar for day D are never used. In the model interface the covariate rows of day D-1 are
   NaN.
3. TSO day-ahead forecasts: not used by any pre-registered model (the API gives no vintage / publication time, so
   we cannot prove they were available before noon D-1).
4. Calendar facts about D (date, weekday, holidays) are allowed.
5. Enforcement: `tools/power/data.build_context` builds every model input (evaluation and training) and asserts,
   with explicit UTC timestamps, that (a) every context price belongs to a day <= D-1 whose auction time (noon of
   the previous day) is before the forecast time (noon D-1, local) and that no context hour overlaps day D; (b) the
   last non-NaN covariate row belongs to a day <= D-2 and ends at or before the start of D-1. Models receive only
   the arrays built there (no file access at inference; registry code reviewed by the auditor).
6. Training: any fitting uses only delivery days <= 2025-06-30, through `data.train_panel` / `data.train_windows`
   (which raise or assert if anything later is requested). Per-origin recalibrated models (LEAR, ARX, empirical
   naive intervals) fit only on their context, which ends at D-1 (D-2 for fundamentals).

## 3. Splits (target delivery days)

- DEV: every delivery day <= 2025-06-30: training and architecture search. Harness DEV backtest: target days
  2024-07-01 .. 2025-06-30 (selection only).
- VAL: target days 2025-07-01 .. 2025-10-31. Debugging, and confirmation of ONE frozen candidate (+ its direct
  twin, if it has one). No further selection after VAL except the documented go/no-go.
- TEST: target days 2025-11-01 .. the last delivery day with a complete price vector in the pinned data snapshot
  (downloaded 2026-10-31; expected end 2026-10-31, ~365 days). Run exactly ONCE, all models of section 5 in one
  run, after the candidate is frozen (`runs/power/FROZEN.json`) and the auditor gives GO (section 8).
- Context: the harness passes the last 736 days (17,664 hourly prices) before D; models crop.

Model release dates vs windows (contamination): Chronos-Bolt Nov 2024, TiRex May 2025, Kronos Aug 2025, Chronos-2
Oct 2025 (its training data may overlap VAL, not TEST), Timestral-5M 2026-10-15 (training data MAY COVER TEST:
reported, marked CONTAMINATED, excluded from any "beats all" claim). Public EPF benchmark datasets (e.g. German
prices up to 2017) may be in FM training corpora; they end long before VAL/TEST. Alma's SSM checkpoints were
trained on 14 public datasets (retail, M5, Monash, NN5, ERCOT electricity LOAD; no electricity prices; latest real
date 2021-09) plus synthetic series (`docs/commodity_leakage.md`).

## 4. Targets and decision rules (fixed; no threshold fitted on any window)

Notation: P(D,h) = price of slot h on day D; base(D) = mean over the available slots of D (24; 23 on the
spring-forward day); peak = slots 8..19 (08:00-20:00 local), off-peak = the other 12 slots;
spread(D) = mean(peak) - mean(off-peak). m(D,h) = the model's median for slot h.

| id | target | actual label | model's prediction |
|---|---|---|---|
| P1 (primary) | daily direction | base(D) > base(D-1) | mean_h m(D,h) > base(D-1) (mean over the slots that exist on D) |
| P2 (primary) | hourly direction | P(D,h) > P(D-1,h), every slot | m(D,h) > P(D-1,h) |
| P3 (secondary) | peak/off-peak spread direction | spread(D) > spread(D-1) | spread of the medians > spread(D-1) |
| P4 (primary) | probabilistic accuracy of the 24 prices | - | quantiles |

- Direction from any probabilistic or point model = sign(median - reference) as above. If the prediction equals the
  reference exactly, the DEV majority class of that target is predicted (so `naive_d`, whose median IS the
  reference, reduces to the majority class; stated with its results).
- Ties in the actual label (exact equality) are excluded and counted. Days with a missing non-DST target slot or a
  missing reference are excluded from P1/P3 and counted; P2 drops only the affected (D,h) pairs.
- P4: per day, QS(D) = mean over the available slots of the mean pinball loss over the 9 levels 0.1..0.9 (EUR/MWh);
  CRPS is approximated by 2 x QS on that grid. Primary P4 metric: QS skill vs `naive_epf` = 1 - mean QS(model) /
  mean QS(naive_epf). Also: MAE of the median (per day mean absolute error, EUR/MWh) and rMAE = MAE / MAE(naive_epf)
  (the EPF-standard relative MAE, Lago et al. 2021), RMSE of the median (descriptive), coverage of the central 80%
  [q0.1, q0.9] for every model and 95% [q0.025, q0.975] for models that output those levels.

## 5. Models (fixed inference settings)

CPU only, `nice -n 10`. Quantile levels requested: 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975.

| name | kind | settings |
|---|---|---|
| majority | baseline (direction only) | DEV (<= 2025-06-30) up-fraction per target; predicts the majority class |
| dow_majority | baseline (direction only) | DEV up-fraction by weekday of D (P1, P3) and by weekday x slot (P2) |
| naive_d | baseline | median = P(D-1,h); quantiles = P(D-1,h) + empirical quantiles of P(d,h)-P(d-1,h) over the last 364 context days, same slot |
| naive_w | baseline | median = P(D-7,h); quantiles from the 7-day differences, same construction |
| naive_epf | baseline, P4 reference | EPF-standard naive (Nowotarski & Weron): naive_w if D is Mon/Sat/Sun, else naive_d |
| lear | competitor (EPF standard) | LEAR, compact implementation of Lago et al. (2021): one model per slot, daily recalibration on the last 728 days; inputs: prices of D-1, D-2, D-3, D-7 (96), load / wind onshore / wind offshore / solar actuals of D-2 and D-7 (192), weekday dummies (7); variance-stabilising transform asinh((x - median)/MAD) on the calibration window (inputs and target); alpha chosen by LassoLarsIC (AIC) then Lasso refit (as epftoolbox); quantiles = point + empirical quantiles of the in-sample calibration residuals of that slot (transformed space; in-sample, so intervals are expected to be too narrow: disclosed) |
| lear_nocov | competitor | lear without the fundamentals (prices + weekday only) |
| arx | competitor | per-slot OLS on the asinh-transformed scale, 728-day daily recalibration; inputs: P(D-1,h), P(D-2,h), P(D-7,h), min and max of D-1, P(D-1,23), load(D-2,h), wind onshore+offshore (D-2,h), solar(D-2,h), 7 weekday dummies; Gaussian quantiles with the residual sd |
| chronos2 | FM | amazon/chronos-2, univariate, last 2048 hours |
| chronos2_cov | FM | amazon/chronos-2, last 2048 hours, past covariates = the 4 fundamentals shifted by 24 hours (row t carries the value at t-24h, so every covariate value is from <= D-2; no NaN at the end) |
| chronos_bolt_base | FM | amazon/chronos-bolt-base, last 2048 hours, levels 0.1..0.9 |
| tirex | FM | NX-AI/TiRex, last 2048 hours, levels 0.1..0.9 |
| timestral_5m | FM, CONTAMINATED | mistralai/Timestral-5M, last 2048 hours, native levels |
| kronos_small | FM | NeoQuasar/Kronos-small + Kronos-Tokenizer-base (official code @67b630e), open=high=low=close=price, volume 0, last 256 hours, hourly time stamps, 10 samples, T=1, top_p=0.9, seed 0 per origin, empirical quantiles; every 5th VAL/TEST target day (speed; its own accuracy rests on n/5, pairwise tests on common days) |
| alma_s0 | Alma zero-shot | `data/ckpt/wm_ssm_none_s0.pt` (sha256 pinned), train_wm.py eval-only light path (as tools/commodity), internal context 256, season 24, freq "H", levels 0.1..0.9 |
| alma_direct_s0 | Alma zero-shot (direct twin) | `data/ckpt/wm_ssm_direct_s0.pt` (sha256 pinned), same settings |
| architect candidate (+ direct twin) | candidate | frozen per section 8, listed in `runs/power/FROZEN.json` |

## 6. Statistics

- P1 (one independent label per day): accuracy, Wilson 95% CI, exact two-sided binomial test vs 0.5,
  Pesaran-Timmermann (one-sided). Model vs competitor: DM-HLN (h = 1, no HAC lag) on the daily 0/1 loss difference,
  two-sided; supplementary: exact McNemar and a circular block bootstrap (blocks of 7 days, 2000 draws, seed 0) 95%
  CI of the accuracy difference.
- P2 (24 dependent labels per day): accuracy over all (D,h) pairs; 95% CI by circular block bootstrap over days
  (each draw resamples whole days, blocks of 7 consecutive days, 2000 draws, seed 0). Test vs 0.5: DM-HLN on the
  day-level series of (daily hit rate - 0.5). Model vs competitor: DM-HLN on the day-level series of daily
  hit-rate differences, plus the block-bootstrap CI of the difference. The iid Wilson/PT for P2 are printed as
  "naive" only. Accuracy by hour block (slots 0-3, 4-7, 8-11, 12-15, 16-19, 20-23), with the day-block CI, is
  descriptive.
- P3: as P1.
- P4: DM-HLN on the daily QS difference (and daily absolute-error difference for MAE), h = 1, two-sided, plus the
  block-bootstrap CI of the mean difference.
- Holm within each target family (P1, P2, P3, P4-QS, P4-MAE) over all (model, competitor) pairs in which the model
  is the frozen candidate, alma_s0 or alma_direct_s0. FM-vs-baseline comparisons are reported with Holm within
  their own family and are descriptive.
- Base rates and majority accuracy are printed next to every accuracy.

## 7. Claims

"C beats X on target T" only if C's metric is better AND the DM-HLN p-value, Holm-adjusted over C's comparisons
with every non-contaminated competitor on T, is < 0.05 AND the 95% block-bootstrap CI of the difference excludes 0.
"Beats all models" on T requires this against every non-contaminated model and baseline. Otherwise: "numerically
higher (or lower), not significant". Every target is reported, wins and losses alike, with the number of
configurations K tried in the search and a winner's-curse caveat. Timestral results are reported but marked
contaminated.

## 8. Selection, freeze, TEST

1. Search and selection use DEV only (training <= 2025-06-30; selection on the DEV backtest or architect-internal
   DEV splits).
2. VAL (this harness, `run_eval.py --split val`, is authoritative) confirms the single frozen candidate (+ twin).
3. Freeze: `runs/power/FROZEN.json` (architect) lists model names, checkpoint paths and sha256, code commit, and K;
   it is committed (force-added) before TEST.
4. TEST: `run_eval.py --split test`, strict mode, exactly once, all models of section 5 + FROZEN models, only after
   FROZEN.json exists and the manager relays the auditor's GO. Strict mode refuses: missing FROZEN.json, existing
   RUN.json or forecast files, `--models`/`--force`, a dirty `tools/power` or this file, a data file whose sha256
   differs from the pin, an unpinned or mismatching Alma checkpoint, any architect-module import error (preflight
   before RUN.json). A crash after RUN.json is written still consumes the TEST. Before TEST, the identical procedure
   is rehearsed on VAL into a fresh directory (`--split val --rehearse`).

## 9. World-model test

The frozen candidate vs its direct twin (if the architect provides one) and alma_s0 vs alma_direct_s0: P4 QS (DM-HLN)
and P1/P2 accuracy (DM-HLN on day-level losses), Holm over the endpoints. Stated as rollout better / worse / no
significant difference.

## Amendments

- A1 (2026-10-31 ~20:30, before any forecast or score on any window). Data snapshot pinned:
  `data/power/hourly.csv` sha256 `79a2c96e90b51ea7f6db1879830e63615b1f82b3a97392953e1b08e651f76dea` (built from the
  raw files in `data/power/manifest.json`, downloaded 2026-10-31 20:05-20:20 UTC-4); 4,322 delivery days
  2015-01-01 .. 2026-10-31. TEST = 2025-11-01 .. 2026-10-31 (365 days), VAL 123 days, DEV backtest 365 days.
  Missing price slots: only the 12 spring-forward slots (structural) and 2015-01-01 slot 0 (before the first API
  value); covariates: the same plus the last UTC hour of 2026-10-31 (incomplete, never used: it could only enter
  forecasts for days >= 2026-11-02). Disclosure: for the sanity check of the raw data (units, the zone split, the
  15-min switch) the lead printed yearly and monthly mean prices for the whole sample, including the TEST months
  (prices rise from ~85 EUR/MWh in Oct 2025 to ~170 in Oct 2026); no forecast or score was computed. Information-set
  tests (`tools/power/test_infoset.py`) pass: perturbing all prices of days >= D and covariates of days >= D-1 leaves
  every model input bit-identical; LEAR outputs then differ by <= 1e-11 EUR/MWh (floating-point noise of the Lasso
  solver, also present between two identical calls; tolerance 1e-8).
