# Alma on fev-bench: PARTIAL (paused 2026-10-31, memory emergency)

**Status: PARTIAL. There is no Alma skill score yet. Do not quote any Alma fev-bench number from this file.**
The coordinator paused all inference: the Mac was swapping about 7.5 GB, and the training job had slowed fivefold. The
training process (`train_wm.py`, pid 2558) was using about 9.7 GB, not the 2.6 GB in the brief, and that is what
drove the swap. My fev processes peaked at 1.44 GB (export) and 0.29 GB (prediction). One earlier attempt to export
`m5_1D` pushed the machine into heavy swap, and I stopped it.

## Scope (auditor veto, docs/audit_sprint.md 1c/R1/R2)

- 48 of the 100 fev-bench tasks are our sealed TEST split (`tools/bench.py` TEST_TASKS). **None was exported,
  predicted or scored.** `tools/fev_full.py export` refuses TEST tasks unless `--allow-test` is passed. I checked
  the `split_tag` of all 47 exported inputs: none is TEST. `data/bench/test_access.log` has no new entries.
- Alma is scored only on the **52 non-TEST tasks**: 30 of our VAL tasks, which are selection-exposed, and 22
  others. Every leaderboard model will be recomputed on the same subsets: nontest52, nonval22 and val30. Subset
  numbers are never compared with full-100 leaderboard numbers.

## Progress

| stage | done | remaining |
|---|---|---|
| export (fev Task windows to contexts) | 47/52 | m5_1D/1W/1M (leaked, imputed with Chronos-Bolt anyway; m5_1D export caused heavy swap), LOOP_SEATTLE_5T (209 MB), kdd_cup_2022_10T (79 MB) |
| predict s0 (paths+TTA, CPU) | 7/52: ETT_1W, ETT_1D, ETT_1H, ETT_15T, LOOP_SEATTLE_1D, australian_tourism, uk_covid_nation_1W/new | 45; seeds s3 and s4 not started |
| score (fev evaluation_summary) | 3 smoke tasks only (ETT_1W, australian_tourism, uk_covid_nation_1W/new), scored before the leakage file existed | all; australian_tourism **must be re-scored with `--force`** to carry its leak flag |
| aggregation | official leaderboard reproduced (below); `tools/fev_full_agg.py` written but not yet run on Alma | everything |

Prediction speed under the swap was about 16 ms per forecast (h=28, TTA) and about 0.2 s per forecast at h=96–168
with paths. The remaining s0 work is roughly 40–60 min on an unloaded machine, and each extra seed costs about the
same again.

## Aggregation check (official code, full 100 tasks)

`fev.analysis.leaderboard` with the settings of the fev repo's `scripts/generate_fev_bench_figures.py`:
baseline Seasonal Naive, leakage imputation Chronos-Bolt, `missing_strategy="impute"`, clip [0.01, 100]. Source:
`autogluon/fev` clone at commit `193e4fcdbafe37bd5f676a00db65cf02ca1f737e`, `benchmarks/fev_bench/results/*.csv`
(44 models), fev 0.10.0.

| model | SQL skill | MASE skill | WQL skill | published |
|---|---|---|---|---|
| Chronos-2 | 47.28 | 35.50 | 51.52 | SQL 47.3 / MASE 35.5 (Chronos-2 paper) |
| TiRex | 42.58 | 30.01 | 46.67 | 42.6 / 30.0 |
| Timestral-209M | 50.39 | 39.49 | 54.63 | 50.39 / 39.49 (Timestral Table 3) |
| Timestral-5M | 46.19 | 34.54 | 50.49 | 46.19 / 34.54 |
| Chronos-Bolt | 38.89 | 26.52 | 43.19 | |
| AutoTheta | 5.46 | 10.99 | 7.85 | |
| AutoETS | -26.82 | 2.26 | -27.03 | (3 failures, imputed) |

The reproduction matches every published value to the rounding shown.

## Leakage (Alma's real data: TSAR_GIFT_CLEAN=1 corpus)

The checkpoint's stored config lists 14 TRAIN datasets: dominick, ercot, m5, monash_australian_electricity,
cif_2016, m1_monthly, m3_monthly, m3_quarterly, nn5_weekly, tourism_monthly, tourism_quarterly, traffic, weather and
nn5. `monash_hospital` is **not** among them, because GIFT_CLEAN drops it. Conservative self-flags are in
`runs/fev_full/leakage.json`; flagged tasks get Chronos-Bolt's score under the fev rule.

| task(s) | flag | basis |
|---|---|---|
| m5_1D, m5_1W, m5_1M | **trained_on** | exact: m5 is a TRAIN dataset |
| ercot_1H/1D/1W/1M | **trained_on** | exact: ercot is a TRAIN dataset |
| fred_md_2025/cee, /macro | **trained_on** (content) | lag scan: 33 hits vs monash_m1_monthly / m3_monthly, including r = 1.000 (HOUST, S&P 500), n = 119–143 |
| fred_qd_2025/cee, /macro | **trained_on** (content) | 98 hits vs monash_m3_quarterly and m3_monthly aggregated to quarters (EXUSUKx r = 0.9998, TB3MS 0.995) |
| australian_tourism | **trained_on** (possible, conservative) | hits r 0.95–0.96, n = 35, vs monash_tourism_quarterly. One probe matches several train series, which suggests the matches are driven by seasonality. Same source family (Tourism Australia). Report the result with and without this flag |
| hospital | clean | 0 hits vs m1/m3/tourism monthly and cif_2016; monash_hospital is not in the GIFT_CLEAN TRAIN |
| proenfo_gfc12 | clean | 0 hits vs ercot and traffic |
| proenfo_gfc14/17, favorita_transactions_*, kdd_cup_2022_30T | **not yet scanned** | the scan was stopped by the pause |
| the 30 VAL tasks | not trained on | bench integrity scans, but **selection-exposed** (Alma was tuned on them) |

FRED is a content flag, not fev's dataset-name rule: M1/M3 contain anonymised copies of the same US macro series.
That is the same finding behind `bench.TRAIN_DROP` for the TEST macro tasks. The raw scan log is in
`runs/fev_full/overlap_scan_partial.log`.

## Pipeline and exact commands to resume

Files: `tools/fev_full.py` (export/score, eval venv), `tools/fev_full_alma.py` (predict, project venv),
`tools/fev_full_overlap.py` (leakage scan) and `tools/fev_full_agg.py` (official aggregation, subsets, pairwise
CIs). All are resumable: finished tasks are skipped.

```bash
cd /Users/srimanarayana/Timeseries
export FEV_CLONE=<fev clone @193e4fc>
# 1. export (non-TEST only; m5 / LOOP_SEATTLE_5T / kdd_cup_2022_10T one at a time, watching memory)
HF_DATASETS_OFFLINE=0 nice -n 10 .venv-eval/bin/python -I tools/fev_full.py export --tasks "$(cat runs/fev_full/light_tasks.txt)"
nice -n 10 .venv-eval/bin/python -I tools/fev_full.py export --tasks LOOP_SEATTLE_5T
nice -n 10 .venv-eval/bin/python -I tools/fev_full.py export --tasks kdd_cup_2022_10T
# 2. predict (CPU, 4 threads, paths+TTA = the audit_r1 / ref_ssm_* recipe); repeat for s3, s4
nice -n 10 .venv/bin/python tools/fev_full_alma.py --ckpt data/ckpt/wm_ssm_none_s0.pt --model alma_ssm_s0
# 3. finish the leakage scan (it rewrites runs/fev_full/overlap_scan.json), update runs/fev_full/leakage.json
nice -n 15 .venv/bin/python tools/fev_full_overlap.py --only proenfo_gfc14,proenfo_gfc17,favorita_transactions_1D,favorita_transactions_1W,kdd_cup_2022_30T
# 4. score (re-score the 3 smoke tasks: they predate leakage.json)
HF_DATASETS_OFFLINE=1 nice -n 10 .venv-eval/bin/python -I tools/fev_full.py score --model alma_ssm_s0 --tasks nontest --force
# 5. aggregate: nontest52 / nonval22 / val30, every leaderboard model recomputed on the same subset
HF_DATASETS_OFFLINE=1 .venv-eval/bin/python -I tools/fev_full_agg.py --check --models alma_ssm_s0
```

The prediction wrapper runs `train_wm.py` itself in eval-only mode, so the model, the causality check and
`predict()` with TTA and paths are unchanged, and only the VAL scorer is swapped. It skips the TRAIN load (2.6 GB),
so the stock `cfg_mismatch` data-hash check is not run there. Instead, `_stored_cfg_check` requires the checkpoint's
stored data config (`train_sha256`, the dataset list, the flags) and its env to equal those of
`wm_ssm_none_s0.pt`. The stock full check (`--full-cfg-check`) was run once for s0 and passed.

## Caveats known so far

- 30 of the 52 tasks are our VAL tasks, on which about 60 configurations were screened, so they are
  selection-exposed. Always report nonval22 next to nontest52.
- 12 of the 52 tasks are self-flagged (m5, ercot, fred, australian_tourism) and replaced by Chronos-Bolt. That
  pulls Alma toward Chronos-Bolt on those tasks. Also report the result with only the 7 exact flags.
- Univariate only: covariates are ignored and multivariate targets are split (as Chronos-Bolt does). Alma reads
  256 points of context.
- `inference_time_s` was measured under heavy swap and CPU contention. Do not use it for efficiency claims.
- fev's `dataset_fingerprint` differs between a local load and a hub load of the same pinned file. Task-definition
  columns are copied from the leaderboard's Seasonal Naive row, and `num_forecasts` matches the leaderboard (checked
  on the smoke tasks). The local fingerprint is kept in `local_fingerprint`.
