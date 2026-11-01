# Alma-Gas (US natural gas): pre-registration

Author: Alma-Gas lead (agent), 2026-10-31. Branch `alma/investor-sprint`. Harness: `tools/gas/`
(`fetch.py`, `data.py`, `storage.py`, `models.py`, `run_eval.py`, `test_infoset.py`). Research basis:
`docs/gas_research.md` (commit 6667375). Leakage record: `docs/gas_leakage.md`.

Status: written and committed BEFORE any forecast or score of any kind was computed on any window (DEV, VAL or
TEST) by the lead. Looks at the data before this commit: download sizes, date ranges, row counts per year (to find the
FRED gaps and the end of the EIA futures series), missing-value counts, and an equality check of FRED vs EIA spot on
dates <= 2025-10-31; no TEST-period value was printed or summarised (counts of TEST rows only). The research agent's
DEV-only feasibility numbers (`docs/gas_research.md` section 4; observed-weather nowcast 0.76 on 2020-01..2025-06) were
read before this was written. Every change after this commit is a dated amendment at the bottom, made before the TEST
run, with its reason.

## 1. Data (free, no key; raw files + url + UTC download time + sha256 in `data/gas/manifest.json`, `manifest_cpc.json`)

| series | source | use |
|---|---|---|
| Henry Hub spot, USD/MMBtu, daily | EIA RNGWHHD table (`rngwhhdd.htm`); FRED DHHNGSP (same series; ~1,900 blanks before 2007, identical on all 5,474 common days <= 2025-10-31) as a cross-check only | price task (secondary / negative control) |
| Weekly working gas in storage, Lower 48 + 5 regions (East, Midwest, Mountain, Pacific, South Central), Bcf, week ending Friday, from 2010-01-01 | EIA `ngshistory.xls` (current vintage) and `revisions.xls` sheet `original_data` (FIRST PRINT of every week from 2015-06-19) | PRIMARY target; first print used where available |
| WNGSR release schedule (alternate holiday dates 2025-2026) | EIA `ir.eia.gov/ngs/schedule.html` | release timestamps |
| CPC observed daily degree days, CONUS: gas-utility-weighted HDD (`UtilityGas.Heating`), population-weighted CDD (`Population.Cooling`), 1997+ | NOAA CPC `degree_days/weighted/daily_data/{YYYY}/` | weather inputs (observed, lagged) |
| CPC 7-day degree-day FORECASTS, same two products, archived issues 2014+ (genuine ex-ante; header "Issued 0000UTC YYYYMMDD"; HTTP Last-Modified recorded per file) | NOAA CPC `degree_days/weighted/daily_forecasts_7day/{YYYY}/{MM}/{DD}/` | weather inputs for days after the origin. Downloaded for the issue days the storage origins can use (each origin day and the day before it); full daily archive is an amendment candidate |
| Population-weighted HDD/CDD from 10 fixed cities (`tools/gas/cities.py`, weights = 2020 MSA population) | Open-Meteo archive (ERA5 reanalysis) daily mean 2 m temperature | price-task covariates only (observed, lagged one day) |
| NYMEX futures contracts 1-2 | EIA RNGC1/RNGC2 | NOT used: EIA stopped publishing after 2024-04-05 |

Processing: `tools/gas/data.py build` -> `data/gas/daily.csv` (panel on EIA spot dates; missing days dropped, never
filled), `storage.csv`; `tools/gas/storage.py build` -> `data/gas/weekly.csv`, `cpc_obs.csv`, `cpc_fcst.csv`. The
snapshot sha256s are pinned in `run_eval.DATA_PINS` by amendment A1 (before any score); strict mode refuses other
files. Prices: none non-positive through 2025-10-31; any log floors at 0.01. Not used: analyst storage consensus (not
free), Yahoo NG=F/TTF=F (unofficial, roll artefacts).

Vintages: spot prices are not revised. Storage: first print from 2015-06-19, current vintage before (DEV only;
disclosed). CPC observed degree days in the yearly files are the current (final) values of what was preliminary on the
next day (small revisions; disclosed). ERA5 is a reanalysis (revised; price covariates only).

## 2. Information set (the leakage rule; enforced in code, `tools/gas/test_infoset.py`)

Price task. An origin at panel index i is the close of trade date t = d[i-1]; step k targets the k-th later panel row.
1. Spot prices dated <= t.
2. A storage report (week ending F) is usable iff its release date A(F) <= t. A(F) = the alternate date in EIA's
   2025-2026 holiday schedule if listed; otherwise in 2025-2026 the standard Thursday F+6; before 2025, F+10 (the
   Monday after) if any US federal holiday falls on F+3..F+7 (conservative; the longest observed delay is Monday
   2025-12-29), else F+6. Release time 10:30 ET (alternate releases up to 12:00 ET); the close of the release day is
   after it.
3. Weather: observed degree days of calendar days <= t-1.
4. Calendar facts are allowed. Enforcement: `data.build_context` asserts (with dates) that every context row is <= t,
   every storage value used was released on or before its row date and that the release is >= week end + 5 days (the earliest alternate release is a Wednesday), and
   every weather date is <= t-1. `test_infoset.py` rebuilds the panel from PERTURBED RAW SOURCES (spot >= first target
   date, reports released after t, weather >= t) and requires bit-identical model inputs, plus sensitivity checks.

Storage task. Target week W (ends Friday F_W; released A_W, used as 12:00 ET).
1. Origins: `thu` = A_{W-1} 17:00 ET (close of the day the previous report came out, normally Thursday F_W - 1; the
   report of the week ending the next day is the target, ~6.8 days before its release: "1 week ahead"); `fri` =
   max(F_W, A_{W-1}) 17:00 ET (close of the report week, the nowcast, ~6 days before release). Both < A_W (asserted).
2. Storage history: exactly the weeks released at or before the origin (weeks <= W-1; asserted).
3. Weather for the 7 days of week W (Sat..Fri): a day x <= origin date - 1 uses the CPC OBSERVED value; any later day
   uses the CPC FORECAST of the latest issue usable at the origin: issue date < origin date, or issue date == origin
   date and its Last-Modified <= origin datetime. Normals: causal (same month-day, 10 previous years, observed).
4. avg5_W (the mean of dS of the same report week, F_W - 364k +- 3 days, k = 1..5) is computed by us from past weeks
   only (it is part of the label definition and of every model's input).
5. Enforcement: `storage.build_context` asserts all of the above with explicit timestamps; `test_infoset.py`
   rebuilds the weekly table from perturbed levels (week W and later), perturbs observed weather from the origin day
   and every non-usable forecast issue, and requires identical inputs (plus sensitivity checks).

Training (any fitting by any model, including the architect's): only targets <= 2025-06-30 (price: target dates;
storage: weeks RELEASED <= 2025-06-30) through `data.train_panel` / `data.train_windows` / `storage.train_weeks`
(which raise or assert otherwise). Models receive only context dicts; no file access at inference.

## 3. Splits

- DEV: everything with target (release) <= 2025-06-30. Harness DEV backtests (selection only): price origins whose
  first target is 2022-07-01 .. 2025-06-30; storage weeks released 2019-07-01 .. 2025-06-30. Storage models that need
  CPC forecasts can train on weeks released 2014-01-15 .. 2025-06-30.
- VAL: price origins with first target 2025-07-01 .. 2025-10-31, a pair scored only if its target date is
  <= 2025-10-31; storage weeks released 2025-07-01 .. 2025-10-31 (weeks ending 2025-06-27 .. 2025-10-24, ~18).
  Debugging and confirmation of ONE frozen candidate (+ its direct twin); no selection after VAL except the go/no-go.
- TEST: price first target >= 2025-11-01 to the last spot date of the snapshot; storage weeks released >= 2025-11-01
  (~52 weeks to the week ending 2026-10-23). SEALED: no metric and no target value is computed or inspected before the
  single TEST run (section 8).

Contamination: Chronos-Bolt Nov 2024, TiRex May 2025, Kronos Aug 2025, Chronos-2 Oct 2025 (may overlap VAL, not TEST),
Timestral-5M 2026-10-15 (its training data MAY COVER TEST: reported, marked CONTAMINATED, excluded from every "beats"
claim). Alma's s0 checkpoints: 14 public datasets + synthetic (`docs/commodity_leakage.md`; gas-specific scan in
`docs/gas_leakage.md`). Known prior TEST exposure: commodity round 1 ran a TEST of HH spot (`hh`, 2025-11 .. 2026-10)
with several of the same FMs/baselines (`runs/commodity/test/hh`); those numbers exist in the repo. The lead has not
read them; the architect must not use them; this is one more reason the price task is only a negative control. The
storage target was never evaluated in this repo.

## 4. Targets and decision rules (fixed; no threshold fitted on any window)

| id | target | label | model's prediction |
|---|---|---|---|
| G1 (PRIMARY) | storage deviation direction, Lower 48, origin `thu` (main) and `fri` (co-primary) | y_W = 1 if dev_W = dS_W - avg5_W > 0 (0 if < 0; exact 0 excluded) | sign(median dS_W - avg5_W); exact tie -> DEV majority |
| G2 (PRIMARY, continuous; the WM-vs-twin endpoint) | storage change dS_W (Bcf), L48, both origins | dS_W (first print) | MAE of the median (Bcf); QS = mean pinball over levels 0.1..0.9 (CRPS ~ 2 QS); cov80 |
| G3 (secondary) | regional deviation direction, 5 regions pooled (~5 x 52 labels) | as G1 per region | as G1 per region |
| P1 (negative control) | HH spot direction, h = 1, 5, 20 trading days | sign(P[t+h] - P[t]) (ties excluded) | sign(median_h - P[t]); tie -> DEV majority |
| P2 (secondary) | 5-day realised-vol direction of HH | std of the 5 log returns t+1..t+5 > std of the 5 ending at t | sigma = mean_k (log q0.9_k - log q0.1_k)/(2 * 1.2816 * sqrt(k)), k = 1..5, > past std |
| P3 (secondary) | log-price QS, h = 1, 5, 20 | log P[t+h] | pinball of log quantiles (floored 0.01), skill vs rw_gauss; cov80 |

dS_W = first-print level(W) - first-print level(W-1). This equals EIA's reported net change except in weeks with a
revision/reclassification of week W-1 (disclosed). avg5_W is ours (EIA's published 5-year average uses calendar-year
windows; ours is the causal same-week mean of the 5 previous years).

## 5. Models (fixed settings; `tools/gas/models.py`)

CPU only, `nice -n 10`. Quantile levels requested: 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975 (NaN where not native).

Storage task (each model forecasts dS_W of each of the 6 series; direction from the median vs avg5):

| name | kind | settings |
|---|---|---|
| majority | baseline | DEV up-fraction of y (weeks released <= 2025-06-30) |
| st_climatology | baseline | median dev of the same report week (+-7 days of year) in all previous years of the context (up to 10 years); quantiles from +-14 days |
| st_persistence | baseline | dev_W = dev_{W-1}; quantiles from past weekly changes of dev (260 weeks) |
| st_seasonal_naive | baseline | dS of the same week a year earlier; quantiles from 52-week differences |
| st_ols_wx | baseline (the bar) | OLS dev ~ 1 + HDD anomaly + CDD anomaly of week W (observed + CPC forecast days, as known at the origin; anomaly vs causal normal) + dev_{W-1}; fitted once per origin kind and series on DEV weeks released 2014-01-15 .. 2025-06-30; empirical in-sample residual quantiles |
| st_ols_wx_nolag | baseline | same without dev_{W-1} |
| st_arima_dev | baseline | statsforecast AutoARIMA (season 1) on the deviation series, last 520 weeks, h = 1 |
| chronos2, chronos_bolt_base, tirex | FM | univariate on the weekly dS series, last 520 weeks, h = 1 |
| chronos2_cov | FM | Chronos-2 with past covariates (weekly observed HDD, CDD of the history weeks) and future covariates (week W's HDD, CDD sums as known at the origin; missing days -> normal) |
| timestral_5m | FM, CONTAMINATED | univariate, last 520 weeks |
| alma_s0 / alma_direct_s0 | Alma zero-shot (+ direct twin) | `data/ckpt/wm_ssm_{none,direct}_s0.pt` (sha256 pinned), train_wm eval-only light path, season 52, freq W |
| architect candidate + matched direct twin | candidate | frozen per section 8 (`runs/gas/FROZEN.json`) |

Price task (context: the last 1024 panel rows; H = 20):

| name | kind | settings |
|---|---|---|
| majority, momentum, seasonal | baselines (direction) | DEV majority; sign(P[t] - P[t-h]); sign of the mean same-horizon log change at t - 365k days, k = 1..5 |
| rw_gauss | baseline (P3 reference) | driftless Gaussian log random walk, sigma from the last 250 log returns |
| auto_ets, auto_arima | baselines | statsforecast, last 512 obs, season 1 |
| px_fund_ols | baseline | per-horizon OLS of the h-day log return on [1, stor_dev5/100, stor_chg/100, hdd7/100, cdd7/100] (origin row; all known at its close), DEV origins 2015-01-01 .. with targets <= 2025-06-30; residual quantiles |
| vol_meanrev, vol_persist, garch | baselines (P2) | as oil round 1 with 5-day windows; GARCH(1,1) Gaussian fitted once on DEV log returns 2000-01-01 .. 2025-06-30 |
| chronos2, chronos_bolt_base, tirex, timestral_5m (CONTAMINATED) | FM | univariate, last 1024 obs |
| chronos2_cov | FM | past covariates stor_dev5, hdd7, cdd7 (NaN -> 0) |
| kronos_small | FM | Kronos-small + Kronos-Tokenizer-base @67b630e, o=h=l=c=price, volume 0, context 256, 10 samples, T = 1, top_p 0.9, seed 0 per origin, every 5th VAL/TEST origin |
| alma_s0 / alma_direct_s0 | Alma zero-shot | season 5, freq B |

## 6. Statistics

- G1 (weekly labels; treated as independent): accuracy, Wilson 95% CI, exact two-sided binomial vs 0.5,
  Pesaran-Timmermann (one-sided). Model vs competitor: **exact McNemar** on the discordant weeks (primary);
  DM-HLN on the 0/1 loss as supplementary. Base rate and DEV majority printed with every accuracy.
- G2: DM-HLN (h = 1, no HAC lag) on the weekly differences of absolute error (MAE) and of QS, two-sided, plus a
  bootstrap 95% CI of the mean difference (weeks resampled, 2000 draws, seed 0).
- G3: pooled accuracy with a week-clustered bootstrap 95% CI (whole weeks resampled, 2000 draws, seed 0).
- P1-P3: oil round-1 amendment A3 rules (`docs/commodity_prereg.md`): h = 1 DM-HLN on 0/1 loss; h > 1 and P2: exact
  McNemar on the non-overlapping subsample (every h-th origin; every 10th for P2), Wilson/binomial/PT on that
  subsample, block-bootstrap CI (block 2h; 10 for P2) for the full-sample accuracy; DM-HLN lag 2h supplementary. P3:
  DM-HLN on QS differences (h > 1: on the non-overlapping subsample).
- Holm: storage, per (origin kind, endpoint in {G1 McNemar, G2-MAE, G2-QS}) over all (claim model, non-contaminated
  competitor) pairs, claim models = the frozen candidate, its twin, alma_s0, alma_direct_s0; FM-vs-baseline pairs in a
  separate descriptive family. Price: per target family over (non-baseline model x baseline x horizon) pairs.
- Power on TEST (~52 weeks, research section 7): accuracy must reach ~0.635 for p < 0.05 vs 0.5 (0.654 for a Wilson
  lower bound > 0.5); vs st_persistence/st_ols_wx only the McNemar discordant count matters (e.g. 11 of 13 in favour:
  p ~ 0.02). VAL (~18 weeks) can only detect very large effects: VAL is for debugging, not claims.

## 7. Claims

"C beats X on T" only if C's metric is better AND the primary p, Holm-adjusted over C's comparisons with every
non-contaminated competitor on T (same origin kind), is < 0.05 (AND, for losses, the bootstrap CI of the difference
excludes 0). "Beats all" requires it against every non-contaminated model and baseline. Otherwise: "numerically
better/worse, not significant". All targets reported, wins and losses alike, with K (the number of configurations
the architect tried, from `FROZEN.json`) and the winner's-curse caveat: the DEV-selected maximum of K configurations
is biased upward, so DEV/VAL numbers of the candidate are optimistic; only TEST is a confirmation. P1 is a negative
control: an apparent P1 win is reported as such, with the expectation (research section 4) that it is ~0.5.

## 8. Selection, freeze, TEST

1. Search and selection on DEV only (training targets <= 2025-06-30; selection on the DEV backtests or
   architect-internal DEV splits).
2. VAL (`run_eval.py --split val`, this harness, authoritative) is run ONCE for the frozen candidate + twin; the
   go/no-go rule must be written in FROZEN.json before reading VAL.
3. Freeze: `runs/gas/FROZEN.json` (architect) lists model names, checkpoint paths + sha256, code commit, K, the
   twin-matching evidence (identical inputs, data, steps, seeds, initialisation scheme; oil audit B6); committed
   (`git add -f`) before TEST.
4. TEST: `run_eval.py --task all --split test --i-have-auditor-go`, exactly once, all section-5 models + FROZEN models,
   only after the manager relays the auditor's GO. Strict mode refuses: no flag; missing or uncommitted FROZEN.json;
   existing RUN.json or forecast files; `--models`/`--force`/single task; dirty `tools/gas` or this file (incl.
   untracked files); data files whose sha256 differ from `DATA_PINS`; unpinned/mismatching Alma or FROZEN checkpoints;
   any architect-module import error (preflight, before RUN.json). A crash after RUN.json is written consumes the TEST
   (status failed recorded). Rehearsed first on VAL into a fresh directory (`--split val --rehearse`). After the run:
   RUN.json, hashes.json and FROZEN.json are force-added and committed.

## 9. World-model test

The frozen candidate vs its matched direct twin, and alma_s0 vs alma_direct_s0: G2 MAE and QS (DM-HLN + bootstrap CI)
and G1 accuracy (exact McNemar), each origin kind; Holm over these 3 endpoints x 2 kinds per pair. Stated as rollout
better / worse / no significant difference. A world-model claim requires a twin that sees exactly the same inputs and
training data with the same budget (steps, seeds), differing only in the rollout computation.

## Amendments

- A1 (2026-10-31 ~21:45 ET, before any forecast or score on any window). Data snapshot pinned in
  `run_eval.DATA_PINS`: `data/gas/daily.csv` 569dd1e1…265c, `weekly.csv` 929d210b…2309, `cpc_obs.csv` 083991d2…2fc9,
  `cpc_fcst.csv` fd7352d1…5a3e (full hashes in `run_eval.py`; build reports `data/gas/build_report*.json`). Raw files
  downloaded 2026-11-01 01:16-01:42 UTC (manifests). Panel: 7,485 spot days 1997-01-07 .. 2026-10-27 (DEV 7,155,
  VAL 87, TEST 243 rows). Storage: 878 weeks 2010-01-01 .. 2026-10-23, first print for 565 weeks; avg5 defined from
  2015-01 (617 weeks); 18 VAL and 52 TEST target weeks; one exact-zero deviation through VAL (excluded as a tie).
  Release dates: 629 standard Thursdays, 89 Thursdays in the schedule years, 7 schedule alternates (incl. 2025-12-29
  Monday for the week ending 2025-12-19 and 2025-12-31 for 2025-12-26), 153 pre-2025 holiday weeks with the
  conservative F+10. CPC: observed 1997-01-01 .. 2026-10-29 complete through VAL; forecasts downloaded for the
  2,333 issue days the storage origins can use (origin day and the day before), 44 files 404 (22 issue days) and one
  malformed issue (2023-08-09) -> the latest earlier issue is used (lead up to +6 days); 95% of issues have a
  same-day Last-Modified. `tools/gas/test_infoset.py` passes (`runs/gas/test_infoset.txt`): raw-source perturbation
  on 16 price origins and 11 storage weeks x 2 origins (18 contexts use CPC forecasts), sensitivity checks,
  training-helper refusals, determinism of rw_gauss, px_fund_ols, st_persistence, st_climatology,
  st_seasonal_naive, st_ols_wx. Code changes since the prereg commit (no change of any rule): the storage-release
  assertion uses week end + 5 days (Wednesday alternates are genuine); malformed CPC files are skipped and counted;
  the weekly table is a pure function of the levels (`storage.weekly_from`) so it can be perturbation-tested.
  The harness interface (`tools/gas/README.md`) is frozen from this amendment on; any later change is an amendment.
- A2 (2026-10-31 ~21:55 ET, disclosure only; no rule change; raised by the gas architect). Of the 313 storage weeks in
  the harness DEV backtest, 60 are pre-2025 holiday weeks with the conservative release rule F+10. For those weeks
  both origins (thu and fri) fall on the Monday after the target week, so all 7 target days are observed, the origins
  are identical, and DEV accuracy/MAE of weather-driven models is somewhat flattering relative to VAL (no such week)
  and TEST (actual EIA schedule; holiday shifts of about one day). It errs on the side of no leakage. DEV numbers are
  reported with this caveat. The architect may exclude these weeks from selection, and must say whether it did.
  VAL baseline status at this time: the storage and price baselines had been run on VAL
  (`runs/gas/val/`, debugging; prereg section 3); the storage baselines were also run on DEV.
- A3 (2026-11-01, before the TEST run; required by the auditor's GO condition C1 in docs/audit_gas.md, efdb0da).
  No rule change to any primary analysis.
  (a) Pre-declared robustness table: G1 and G2 for both origins are also reported with the two TEST weeks affected by
  the Oct–Nov 2025 US government shutdown dropped (weeks ending 2025-10-31 and 2025-11-07). EIA release dates for
  those weeks are assumed to follow its published schedule and could not be cross-checked offline. The primary result
  stays the full-TEST table; the robustness table is reported next to it and any difference is disclosed.
  (b) The Henry Hub price task (negative control P1, secondary P2/P3) is labelled "HH TEST previously exposed":
  commodity round 1 scored HH spot over 2025-11..2026-10 (runs/commodity/test/hh) before this project, and its
  committed reports contain HH tables. No gas agent read them as far as can be verified, but the price-task results
  are not blind and carry this label wherever reported.
