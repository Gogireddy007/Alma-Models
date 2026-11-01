# Alma-Gas leakage record

Alma-Gas lead, 2026-10-31. Covers (1) time leakage in the data pipeline and harness, (2) training-data
contamination of the pre-trained models, (3) prior exposure of the TEST window inside this repo. Binding rules:
`docs/gas_prereg.md` section 2. Tests: `tools/gas/test_infoset.py` (output: `runs/gas/test_infoset.txt`).

## 1. Time leakage: pitfalls and how each is closed

| pitfall | rule / fix | enforced by |
|---|---|---|
| Storage value used before its release (Thursday 10:30 ET, holiday shifts) | per-week release date A(F): EIA 2025-26 schedule; pre-2025 holiday weeks F+10 (conservative); else F+6. Price task: report usable iff A(F) <= origin date. Storage task: history = weeks with release <= origin datetime, i.e. weeks <= W-1 | `data.build_context` / `storage.build_context` assertions; perturbation of every report released after the origin (test 1, 2) |
| Using week W's own value or later weeks in avg5 / features | avg5_W from the 5 previous years only (`storage.weekly_from`, rows < k); weekly levels of week W and later perturbed in test 2 -> inputs identical | test 2 |
| Revisions (current vintage embeds later revisions) | first print (`revisions.xls` original_data) from 2015-06-19 for levels and labels; current vintage only before (DEV) | data build; disclosed |
| Observed weather of the forecast day or later | observed degree days only for days <= origin date - 1 (price: Open-Meteo ERA5; storage: CPC observed) | assertions; weather >= origin day perturbed in tests 1-2 |
| Weather "forecasts" that are really analyses (Open-Meteo historical-forecast API) | not used. Only CPC archived 7-day issues, usable iff issued before the origin day, or same day with Last-Modified <= origin time | `storage.week_weather`; non-usable issues perturbed in test 2 |
| CPC observed values are preliminary in real time | yearly files hold the final values (small revisions); disclosed | - |
| Spot trade vs flow date undocumented | the spot of date t is used only from the close of t (conservative either way) | `data.build_context` |
| Futures roll artefacts | futures not used (EIA stopped RNGC1-4 after 2024-04-05; Yahoo not used) | - |
| Fitting on VAL/TEST | all fitting via `data.train_panel/train_windows`, `storage.train_weeks` (raise/assert beyond 2025-06-30; storage: weeks RELEASED <= 2025-06-30); GARCH and px_fund_ols/st_ols_wx fitted on DEV only | test 3 |
| Scoring VAL with TEST values | VAL price pairs scored only if the target date <= 2025-10-31; VAL storage weeks = released <= 2025-10-31 | `run_eval.score_*` |
| Inspecting TEST | the lead printed only TEST row counts; data checks (FRED vs EIA, missing values, non-positive prices) restricted to <= 2025-10-31; info-set tests sample DEV/VAL origins only | this record |

## 2. Training-data contamination

| model | release | risk for VAL (2025-07..10) | risk for TEST (2025-11..) |
|---|---|---|---|
| Chronos-Bolt base | Nov 2024 | none | none |
| TiRex | May 2025 | none | none |
| Kronos-small | Aug 2025 (exchange K-lines; may include NG futures to an undisclosed cutoff) | possible overlap with Jul 2025 | none |
| Chronos-2 | Oct 2025 | possible | none |
| Timestral-5M | 2026-10-15 | possible | **possible: CONTAMINATED, excluded from claims** |
| Alma s0 checkpoints | trained on 14 public datasets (latest real date 2021-09) + synthetic | none | none |

Alma content scan for gas series (`tools/gas/leakage_scan.py` -> `runs/gas/leakage_scan.txt`): see section 4.

## 3. Prior exposure of the TEST window in this repo

Commodity round 1 (`docs/commodity_prereg.md`) ran its single TEST including the secondary series `hh` (FRED DHHNGSP,
2025-11 .. 2026-10) for Chronos-2, Chronos-Bolt, TiRex, Timestral, auto_ets/arima, rw_gauss and Alma; the results
are in `runs/commodity/test/` and `docs/audit_commodity.md`. Anyone who read them has seen TEST-window HH direction
and QS results. The Alma-Gas lead did not open those files. Mitigation: HH price direction is a pre-registered
negative control, not a headline target; the storage target was never evaluated in this repo. The architect must not
open `runs/commodity/test/hh` or the hh rows of the commodity audit.

## 4. Results of the tests and scan

Information-set tests (`runs/gas/test_infoset.txt`, amendment A1): ALL PASSED. Price task: raw-source perturbation on
16 origins (2022-08 .. 2025-08, including release-day origins around the 2025 schedule shifts), with bit-identical
inputs and sensitivity confirmed. Storage task: 11 weeks x 2 origins. Of those 22 contexts, 18 use CPC forecasts;
the other 4 are pre-2025 holiday weeks whose origin is the Monday after the week (A2). All gave identical inputs and
passed sensitivity. Training helpers refuse dates after 2025-06-30. Six models are deterministic on repeat.

Alma TRAIN content scan (`runs/gas/leakage_scan.txt` / `.json`, 6-grams, longest consecutive matching run):
- **Lower 48 storage (the primary target), levels and weekly changes: zero gram hits in all 14 datasets.**
  Henry Hub daily: only scale-free ratio grams, with longest run 4 (dominick; the coarse price-lattice coincidence
  documented in `docs/commodity_leakage.md`).
- Region weekly changes, as integer-Bcf 6-grams: hits only for the small regions. Mountain: longest run 15 in
  monash_weather, 6 in m5. Pacific: longest run 6. East, Midwest and South Central: runs of 1 to 4. These are
  small-integer coincidences. 68% of Mountain and 56% of Pacific weekly changes have |dS| <= 5 Bcf, and the
  matching datasets are rounded integer series (weather readings, unit sales). Region levels give at most single
  ratio grams. Verdict: no storage or gas price series is in Alma's training data. Region results are secondary in
  any case, and the Mountain/Pacific regional direction carries this caveat.
