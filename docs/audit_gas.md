# Audit: Alma-Gas (US natural gas storage and Henry Hub), pre-TEST (2026-10-31, ~23:59 ET)

Auditor: independent auditor with veto, held to the same standard as `docs/audit_power.md` and
`docs/audit_commodity.md`.

Scope:
- Prereg: `docs/gas_prereg.md` (f9e01bd), with A1 (82afa32) and A2 (652621f).
- Leakage record: `docs/gas_leakage.md`.
- Harness: `tools/gas/**`.
- Lead VAL: `docs/gas_results.md` (bb67b83).
- Architect: `docs/gas_search.md`, with protocol 56a856f, Am1 4be7b66, freeze 7678c65, VAL 46fbd51; plus
  `runs/gas/FROZEN.json`, `tools/gas/wm_gas.py`, `tools/gas/wm_scenario.py` and `tools/gas/search/**`.

Checked at HEAD 69a3bd5. `git status` is clean for `tools/gas`, `docs/gas_prereg.md` and `runs/gas/FROZEN.json`.

Checks I ran myself:
- **Info-set tests:** `test_infoset.py` (peak 663 MB) and `tools/gas/search/test_leak.py`.
- **Strict-mode preflight:** `check_data_pins()` and `preflight()` on the exact TEST model lists. They write no
  RUN.json.
- **Guard break-tests:** a sandbox git repo holding a copy of `tools/gas`, `data/gas`, the prereg and `FROZEN.json`,
  with no checkpoints, so it cannot forecast anything.
- **Checkpoints:** sha256 and configs.
- **Timestamps:** file and commit times.

## Verdict: **GO**, once one pre-run condition is met

**Condition C1.** Commit a dated amendment A3 to `docs/gas_prereg.md` before the run. It must:
- (a) pre-declare a robustness table for G1/G2 that excludes the TEST weeks whose release, or whose thu origin's
  previous release, fell in the federal shutdown (2025-10-01 .. 2025-11-12): the weeks ending 2025-10-31 (released
  2025-11-06) and 2025-11-07 (released 2025-11-14);
- (b) state that the price task (P1-P3) is reported as "HH TEST period previously exposed in commodity round 1".

The prereg is under the strict clean-tree check, so commit it before running. No code change is needed. Run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Checklist (coordinator flags 1-11 plus the standard items)

| # | item | status | evidence |
|---|---|---|---|
| 1 | Henry Hub TEST exposure (commodity round 1) | CONCERN, cannot be verified | No file in `tools/gas` (or in the gas docs other than the disclosure) references `runs/commodity/test`. File access times cannot prove anything: every `runs/commodity/test/hh/*.npz` shows atime 20:00:57, the harness hashing at the end of that run, and my own later hash reads (about 20:10) did not change it, so atime is not updated on re-reads. The committed `runs/commodity/test/report_*.md` contain HH tables that any agent could have read. Mitigation: the price task is a pre-declared negative control, every model setting was fixed in the prereg, the storage target was never evaluated in this repo, and P1-P3 carry the exposure label (C1b). |
| 2 | Protocol changes mid-search (Am1/Am2); VAL never influenced selection | PASS (Am2 not independently timestamped) | **Am1** was committed at 22:18:44 (4be7b66), after R1 (last R1 row 22:13:50) and before the first R2 row (22:21:47). It is motivated by DEV findings (cov80 0.02-0.05 from collapsed quantiles). **Am2** is dated "~22:30, after r2_wm_reg, before R2b", but was first committed in the freeze commit (23:41). Its content matches the `results.tsv` order: r2_wm_reg at 22:28:31, then R2b rows from 22:34:56. The architect's code never loads VAL: everything goes through `storage.train_weeks()` (asserts release ≤ 2025-06-30), the cache is `ctx_train.pkl`, and `test_leak` samples DEV only. Freeze commit 7678c65 at 23:41:27 contains FROZEN.json, with `val_go_no_go_rule` present in the committed version, plus the checkpoints and wrappers. The VAL npz for `wm_gas*` were written at 23:41:40-41, after the commit, and every selection row is from 23:24 or earlier. The go/no-go rule (finite output; MAE ≤ 1.5 × st_ols_wx; cov80 in [0.5, 1]) is a breakage check, not a selection. It was written when the lead's VAL baseline numbers (bb67b83, 22:16) were already in the repo, and that is acceptable for such a lenient rule. |
| 3 | Holiday weeks (A2) | PASS (disclosed) | 60 of the 313 DEV weeks use the conservative F+10 rule, which makes them pure nowcasts. This errs towards no leakage and inflates DEV. FROZEN.json reports candidate and twin metrics without these weeks. TEST weeks use the 2025-2026 EIA schedule (`avail_rule` is `schedule` or `thursday_schedule_period` for all 53 TEST rows), so none of them is an F+10 week. |
| 4 | Release dates during the Oct-Nov 2025 shutdown | CONCERN → C1a | The raw `ngs_schedule.html` (EIA, current) lists the alternates actually used and marks the two changes made after the fact as "(Updated)": Jan 8, 2025 and Dec 29, 2025. It lists no shift during Oct-Nov 2025 other than the planned Veterans Day (Fri Nov 14) and Thanksgiving (Wed Nov 26) dates. The harness uses exactly these (TEST rows: 2025-11-06 Thursday, 2025-11-14 schedule, 2025-11-26 schedule). Neither raw XLS contains release dates, so they cannot be cross-checked without online sources. If a release had been late, the leak would be limited to the previous week's report being visible one or more days early at the thu origin; the target week itself is never visible. A pre-declared exclusion of the 2 affected TEST weeks (C1a) bounds this. |
| 5 | First print vs revised | PASS | Targets use first-print levels from `revisions.xls` `original_data` for weeks ending 2015-06-19 .. about 2026-04-03 (565 contiguous weeks); after that, current vintage (28 TEST weeks). EIA updates that file on every revision, so weeks after its last update have not been revised and their current vintage **is** the first print. History inputs also use first print, which is conservative. The 25 + 28 split over TEST is shown in `weekly.csv` `vintage`. |
| 6 | CPC forecast availability gate | PASS | `storage.week_weather` (`storage.py:247`) uses an issue only if issue date < origin date, or issue date == origin date with HTTP Last-Modified ≤ the origin datetime, and asserts it again in `build_context` (`storage.py:303`). Over all 155 TEST-period issues, Last-Modified falls 14.3-15.4 h after the 00 UTC issue time (about 10:30 ET the same day). That is before the 17:00 ET origins, and no TEST issue was posted later than the next day. Observed degree days are used only for days ≤ origin date - 1. |
| 7 | Strict-mode refusals | PASS (tested in a sandbox that cannot forecast) | Each case below was refused with no RUN.json written and no forecast made (the planted files in the cached-npz and RUN.json cases were mine): no `--i-have-auditor-go`; FROZEN.json missing; FROZEN.json untracked; `--task storage`; `--models`; a cached npz in `test/`; RUN.json already present; data pin mismatch; modified `tools/gas/storage.py`; untracked `tools/gas/wm_evil.py`; modified prereg. In the clean case, preflight aborted (no checkpoints) before RUN.json. Gap: a FROZEN.json that is tracked but has uncommitted edits is not refused (only `ls-files` is checked). RUN.json records its sha256, and the file is clean now. |
| 8 | Every pre-registered competitor in the single run | PASS | Storage: `STORAGE_MODELS` (13) plus `wm_gas` and `wm_gas_direct`, with `majority` computed in scoring. That matches prereg §5. Price: `PRICE_MODELS` (12), with majority, momentum, seasonal, vol_meanrev, vol_persist and garch computed in scoring. The frozen WMs have no price `batch`, so `forecast_price` skips them by design. Preflight on the exact lists passed: no import errors, data pins OK, Alma pins OK, `wm_gas*` `meta.ckpt_sha256` = FROZEN.json = `shasum` (52e26f50…, 7ebc0fbf…). |
| 9 | Matched twins | PASS | The two checkpoints' cfgs differ only in `arch` (wm vs direct): d 32, assim 1, regime 1, 1200 steps, lr 3e-3, batch 64, wd 0.05, drop 0.3, aux 0.3, wl48 3, cal 1. Each has 3 members (seeds 0, 1, 2) of 23,534 params, the same training scales, train_end 2025-06-30, 547 training weeks, and release range 2015-01-08 .. 2025-06-26. |
| 10 | Leakage tests | PASS | `test_infoset.py` (my run): raw-source perturbation and sensitivity on 16 price origins and 11 storage weeks × 2 origins (18 with CPC forecasts), training-helper refusals, and determinism of the baselines. ALL PASSED. `test_leak.py` (architect): forbidden perturbations leave all candidate and twin forecasts unchanged; the latest allowed observed weather and the used CPC issue change them (18/18, 10/10). ALL OK. |
| 11 | Clean tree, WM files committed | PASS | `wm_gas.py`, `wm_scenario.py` and `search/**` are tracked, and `git status` is clean for the strict paths. Not covered by the check: `runs/gas/val/metrics.json` and `dev/metrics.json` are modified relative to bb67b83 (the architect's VAL run, and the in-sample DEV plumbing check). Force-add them for the record; this is not blocking. |
| 12 | K and winner's curse | PASS | FROZEN.json: 23 scored network runs, 17 configurations (10 WM, 7 twin). The best single row 0.7915 sits against a member mean of 0.826, and the seed sd of about 0.03 is far larger than the 0.01 band. "The WM does NOT beat its twin on DEV": the twin is numerically better (S 0.7765 vs 0.7966; MAE difference CIs straddle 0). |
| 13 | TEST untouched | PASS | `runs/gas/test/` does not exist. Every forecast file I saw is for VAL/DEV/rehearsal, and both info-set tests sample DEV/VAL only. |
| 14 | Rehearsal | CONCERN | `rehearsal_val` (22:09-22:15, commit 60b70d7) ran with a dirty tree, before the freeze and without `wm_gas*`. Mitigation: the code is unchanged since the freeze, the exact-list preflight passed, and `wm_gas*` ran through `forecast_storage` on VAL. |

## Concerns to disclose with the TEST results

1. Henry Hub TEST numbers were already in the repo (flag 1). The price task is only a negative control.
2. Am2 was committed with the freeze, so its "~22:30" time is self-reported. It is consistent with `results.tsv`.
3. Shutdown-period release dates are assumed from EIA's schedule (C1a sensitivity).
4. DEV is flattered by 60 nowcast-only holiday weeks.
5. On DEV the twin was numerically better than the world model; with K = 17 configurations, DEV and VAL numbers of
   the candidate are optimistic.
6. The rehearsal predates the freeze. A FROZEN.json that is tracked but edited would not be refused (it is clean now).

---

# TEST results audit (2026-11-01, ~00:50 ET)

Inputs: `runs/gas/test/{RUN.json, hashes.json, metrics.json, report.md, test_run.log}` and 42 forecast npz files,
committed in 01fb0f7. The independent re-derivation is `tools/audit_gas_test.py`. It reads the pinned
`data/gas/weekly.csv` and the raw EIA xls and recomputes dS, the causal avg5, labels, DEV majorities, metrics, exact
McNemar, DM-HLN, the bootstrap and Holm; it imports nothing from `tools/gas`.

## Integrity: PASS

- **One complete run.** RUN.json: status complete, 23:54:25 to 00:30:47, commit a114cbe (A3 on top of my GO
  efdb0da), `tools_gas_dirty == ""`, argv `--task all --split test --i-have-auditor-go`. Model lists: storage = 13
  pre-registered + `wm_gas` + `wm_gas_direct`; price = 12 (the WMs have no price path).
- **Pins match.** `frozen_sha256` = FROZEN.json; all 4 data pins match; the npz metas carry
  `wm_gas` 52e26f50…, `wm_gas_direct` 7ebc0fbf… and the Alma pins.
- **Hashes match.** `hashes.json` (43 entries) lists exactly the 42 npz plus metrics.json, and every sha256 matches.
- **No file predates the run.** Every npz has `created >= started` and `split == "test"`.
- **The log is clean.** It contains no error, failure or skip.
- **A3 was in force.** A3 is in a114cbe, before the run.

## Re-derivation: 0 differences

- **Labels.** My dS, avg5 (same report week ±3 days, k = 1..5) and dev equal `weekly.csv` for all 6 series over the
  52 TEST weeks (weeks ending 2025-10-31 .. 2026-10-23). The L48 first-print levels equal the raw EIA xls
  (first print where available, else current) to 0 Bcf.
- **Metrics.** For every model and both origins, these match the harness exactly: L48 G1 accuracy, MAE, QS, cov80,
  and the pooled regional accuracy.
- **Pairwise.** For all claim models (wm_gas, wm_gas_direct, alma_s0, alma_direct_s0), these match exactly: the McNemar
  and DM p-values, own-Holm and the bootstrap CI bounds.

## Headline (L48, 52 weeks; MAE/QS in Bcf, lower is better)

| model | thu acc | fri acc | regions pooled thu / fri | MAE thu / fri | QS thu / fri | cov80 thu / fri |
|---|---|---|---|---|---|---|
| wm_gas | 0.788 | 0.808 | 0.805 / 0.812 | 11.53 / 11.56 | 4.59 / 4.56 | 0.865 / 0.885 |
| wm_gas_direct (twin) | **0.865** | 0.808 | 0.809 / 0.828 | **11.40 / 11.08** | **4.57 / 4.47** | 0.865 / 0.885 |
| st_ols_wx (the bar) | 0.808 | 0.808 | 0.746 / 0.738 | 14.35 / 14.46 | 6.14 / 6.08 | 0.827 / 0.846 |
| st_ols_wx_nolag | 0.808 | 0.827 | 0.652 / 0.660 | 15.54 / 15.56 | 6.35 / 6.34 | |
| chronos2_cov | 0.712 | 0.712 | 0.758 | 19.80 / 19.94 | 7.91 / 7.96 | 0.673 |
| tirex | 0.692 | 0.692 | 0.734 | 29.08 | 11.20 | 0.885 |
| chronos2 / chronos_bolt_base | 0.654 / 0.673 | same | 0.719 / 0.688 | 30.50 / 30.52 | 11.63 / 12.02 | |
| alma_s0 / alma_direct_s0 | 0.635 / 0.654 | same | 0.703 | 34.06 / 34.31 | 13.46 / 13.30 | |
| st_persistence / majority | 0.673 / 0.519 | same | 0.652 / 0.488 | 38.91 / - | 16.98 / - | |

The univariate foundation models give the same numbers at both origins, because they see only released history.

## A3 robustness table (pre-declared; weeks ending 2025-10-31 and 2025-11-07 dropped, n = 50)

- **Levels.** wm_gas: thu acc 0.800, MAE 11.69, QS 4.65; fri acc 0.820, MAE 11.73, QS 4.62. Twin: thu 0.880 /
  11.44 / 4.60; fri 0.820 / 11.12 / 4.50. st_ols_wx: 0.840 / 14.66 / 6.24 and 0.840 / 14.74 / 6.18.
- **Significance status.** Every significance status below is unchanged:
  - wm_gas vs st_ols_wx QS Holm 0.032 (thu) and 0.043 (fri); MAE p 0.126 for both (not significant).
  - Every foundation model: Holm ≤ 0.003.
  - WM vs twin: p ≥ 0.29.
- **Conclusion.** The shutdown-week assumption does not drive any result.

## What can be claimed (prereg §7: better, AND own-Holm p < 0.05 over all non-contaminated competitors, AND for losses a bootstrap CI that excludes 0; §6: G1 uses exact McNemar)

1. **G2 probabilistic accuracy (QS), both origins.** wm_gas, and equally its direct twin, beats **every external
   competitor**:
   - st_ols_wx: -1.55 Bcf thu, Holm 0.030; -1.53 fri, Holm 0.039.
   - st_ols_wx_nolag, ARIMA, climatology, seasonal naive and persistence.
   - Every non-contaminated foundation model: chronos2, chronos2_cov, chronos_bolt_base, tirex, all Holm ≤ 0.0012.
   - Both zero-shot Alma models.

   This holds under A3 too. The gain over OLS on QS reflects both a better median and better calibration (cov80 0.87
   vs 0.83).
2. **G2 MAE.** wm_gas (and the twin) beats every non-contaminated foundation model, by about 8.3 Bcf vs chronos2_cov
   (Holm 0.0016) and 17.5-19.0 Bcf vs tirex, chronos2 and chronos_bolt_base. It also beats persistence, climatology,
   seasonal naive, ARIMA and Alma.
3. **G1 direction.** wm_gas accuracy of 0.788 (thu) and 0.808 (fri) is far above 0.5: exact binomial p = 3.6e-5 at
   thu, Wilson CI [0.66, 0.88].
4. **Disclosure.** Each claim is stated with K = 17 configurations (winner's curse) and the regime and exposure
   notes. It is the model family (the daily storage-balance architecture with weather inputs and holdout
   calibration) that carries these results, not the rollout.

## What cannot be claimed

1. **"Beats all models".** §7 counts every non-contaminated model, and wm_gas never beats its own twin, so the
   statement fails for both wm_gas and the twin. Use "beats every external baseline and foundation model on QS"
   instead.
2. **MAE over the weather-regression bar.** wm_gas vs st_ols_wx gives -2.8 Bcf (thu) and -2.9 Bcf (fri), with p 0.13
   and CI [-6.5, +0.8]. It is numerically better but not significant, and neither is the comparison with
   st_ols_wx_nolag.
3. **Any G1 accuracy win over a serious competitor.**
   - wm_gas's only significant direction win is over st_climatology at fri.
   - It does not beat the majority baseline after Holm (McNemar 21 vs 7, raw p 0.013, Holm 0.15).
   - It does not beat OLS (0.788 vs 0.808 thu; equal at fri) or any foundation model.
   - The twin beats only majority, climatology and seasonal naive.
4. **The world model.** Under the §9 test (Holm over 3 endpoints × 2 origins), wm_gas vs wm_gas_direct shows no
   significant difference on any endpoint (all raw p ≥ 0.29). The twin is numerically better on thu accuracy
   (0.865 vs 0.788), on MAE and QS at both origins, and on pooled regional accuracy. Together with DEV (twin
   numerically better), the **world-model hypothesis is not supported**: the daily state rollout adds nothing over
   one-step decoding from the same state. alma_s0 vs alma_direct_s0 shows no difference either, and zero-shot Alma
   beats no competitor.
5. **Calibration as a strength.** cov80 is 0.87-0.89 for the candidates against a nominal 0.80, so the intervals are
   slightly wide. Say "conservative intervals", not "well calibrated".
6. **The price task (P1-P3).** It must be labelled **"HH TEST previously exposed"** (A3b). It is a negative control,
   not a result. I did not re-derive it.
7. **Timestral.** It is contaminated and excluded from every claim (on TEST it is no better than tirex anyway).

## Context to show with every number

- **Sample.** 52 weekly labels, so G1 can only detect large effects.
- **Selection.** K = 17 configurations; the frozen recipe was adapted on DEV folds (Am1/Am2).
- **DEV.** DEV is flattered by 60 holiday nowcast weeks.
- **Release dates.** Shutdown-period release dates were assumed; A3 shows they do not matter.
- **Rehearsal.** The rehearsal predated the freeze.
