# Evaluation contract: are Alma's bench scores comparable to the baselines? (Phase 0)

Author: L. Andrade, 2026-10-30. Read-only audit. No training, no downloads, no TEST data.
Inputs: tools/bench.py, prepare.py, baselines.py, baselines/bench/{results.tsv,per_task/*_val.csv},
research_log.md, docs/protocol.md, docs/bench_integrity.md, the HF cache, and the fev-bench / Chronos-2 / TiRex papers.

**Verdict.** Alma's 0.681/0.695 and Chronos-2's 0.559 come from the same harness, so the comparison is *internally* fair:
the tasks, windows, targets and Seasonal-Naive denominators are byte-identical. The comparison is **not** a fev-bench
number, it is **not** zero-shot for the baselines on every family, and the Alma side is val-selected. Report it as
"VAL (tuned on), our bench protocol", never as a fev-bench skill score.

## 1. Metric definitions (tools/bench.py `evaluate`, :535-591; prepare.py `_score`, :194-212)

- The forecast is `q [n, 9, H]`. NaN or inf values become 0, are counted in `n_nonfinite`, and each row is then sorted
  across quantiles (bench.py:559-562).
- **WQL per task** (prepare.py:196-201): `mean_q [ 2·Σ_{n,t} pinball_q / Σ_{n,t} |y| ]`, with Q = {0.1..0.9}
  (prepare.py:27). NaN targets are masked. One pooled ratio covers **all windows and series** of the task.
- **MASE per task** (prepare.py:202-205): `nanmean_n( mean_t |y − q0.5| / scale_n )`. Here `scale_n` is the mean
  |c[t] − c[t−m]| over the *cropped* context. NaNs are removed **before** differencing (prepare.py:183-191), so the lags
  are misaligned across gaps. If the scale is 0 or undefined it becomes NaN and that forecast is dropped by `nanmean`.
- **cov80**: the fraction of observed targets inside [q0.1, q0.9] (prepare.py:206-207).
- **Seasonal naive** (prepare.py:175-180): fill_nans (ffill, then bfill, then 0), then repeat the last m points.
  If len < m, then m = 1. As the denominator it is **degenerate**: all 9 quantiles equal the point forecast
  (bench.py:564-565).
- **Per task:** `wql_rel = wql / sn_wql` and `mase_rel = mase / sn_mase` (bench.py:569).
- **Headline:** the geometric mean over 30 tasks, clipped below at 1e-6 with no upper clip (bench.py:530-532, :575-576).
  Also reported: `cov80_error = mean_task |cov80 − 0.8|` (:577), a family geometric mean (:572, :578), and
  `win_rate_vs_sn` = the fraction of tasks with wql_rel < 1 (:580).

**Comparison with official fev-bench** (Shchur et al., arXiv 2509.26468, §3.2 and §4.1):

| Item | fev-bench | Our bench | Effect |
|---|---|---|---|
| Probabilistic headline | **SQL** (scaled quantile loss, per-series MASE-style scale). WQL appears only in the supplement. | WQL | A different metric. Our numbers cannot be mapped to fev's SQL skill. |
| Window aggregation | Error computed per window, then averaged over windows | WQL pooled over all windows (one denominator) | High-volume windows dominate. Neutral across models. |
| MASE scale | In-sample seasonal error over the **full history** up to the cutoff | Context cropped to ≤2048, NaN-removed lags | Same scale for all models. Neutral within the bench. |
| SN reference | statsforecast SeasonalNaive (fev baseline) | Degenerate SN (no intervals) | wql_rel ≈1.2% lower for everyone (SN-PI scores 0.9876, research_log:198) |
| Aggregation | 1 − gmean(clip(rel, 0.01, 100)) | gmean, clip [1e-6, ∞) | No upper clip: AutoETS blow-ups (2.68) would be capped at 100 in fev |
| Win rate | Pairwise over all models, ties = ½ | Only vs SN | Not comparable |
| Failures | Imputed with SN error | NaN → 0 and flagged (all CSVs: 0 non-finite) | None observed |
| Quantiles | 0.1..0.9 | 0.1..0.9 | Identical |

**Task definitions.** Horizon, num_windows, initial_cutoff, step, seasonality, min_context and the target columns
match fev `tasks.yaml` at f323f6c, with 0 mismatches over 78 specs (docs/bench_integrity.md §3). The data is pinned to
fev_datasets@d3e7e72 (bench.py:71). Each window is the context before cutoff_k and the next H points; window k+1's
context contains window k's target, the same as in fev.

**Deviations from the official tasks** (with who they favour):

1. **Series subsample** (bench.py:66, :482-487). At most 1000 forecasts per task. I checked the parquet row counts:
   13/30 tasks are subsampled, for example favorita_stores_1D (100 of 1579 series), LOOP_SEATTLE (100 of 323) and
   restaurant (125 of 817). Applied equally to all models, so neutral in expectation. It adds sampling noise relative
   to fev.
2. **Context cap of 2048** (bench.py:65, :498). This **hurts Chronos-2** (native 8192, HF config; fev ran it with the
   full history) and Timestral (native 32768). It is neutral for TiRex and Bolt (2048 native). It favours Alma
   *relative to published* baseline numbers.
3. **Multivariate targets scored as independent univariate series** (bench.py:476-480; the ETT, jena, bizitobs and
   uk_covid tasks). Covariates are dropped. fev ran Chronos-2 in full cross-learning mode with covariates (Chronos-2
   paper §5.1-5.2). Our run uses `cross_learning=False` (chronos2/pipeline.py:833 default). This **hurts Chronos-2**.
   For univariate models it matches fev practice.
4. **Leading-NaN crop and min-context ≥ 2** (bench.py:67, :499-502). This drops windows with fewer than 2 finite
   points (restaurant: 999 vs 1000). Neutral.
5. **NaN handling per model.** Chronos, TiRex and Timestral receive raw NaNs (native masking). Stats models get
   `fill_nans` (baselines.py:42). Alma uses its own observation masks (train_v2.py:627, :681). This follows each
   model's own convention, as fev does. Neutral.
6. **Degenerate SN denominator** and **no upper clip**. Applied the same to everyone. They change the absolute level,
   not the ranking among the foundation models.

Net effect: the protocol deviations **disfavour the baselines (mainly Chronos-2) relative to their published setup**.
Inside the bench they are applied equally.

## 2. Baseline provenance

All nine CSVs were produced by **local inference** through `bench.py baselines --split val <model>`
(`run_baselines`, bench.py:871-893). It calls `baselines.registry()` unmodified (baselines.py:170), on the same
`load_tasks` objects with copied contexts (bench.py:556), and appends rows to results.tsv. Nothing was imported.
`baselines/bench/` is **untracked** (git status `??`), so `git log -p` has no history for it. Provenance rests on
results.tsv (date and git sha) and research_log.md:5-10, :147, :210.

| Model | HF id @ cached snapshot (refs/main) | Package | Device | Context actually used | Bench sha |
|---|---|---|---|---|---|
| chronos2 | amazon/chronos-2 @29ec376 | chronos-forecasting 2.3.2 | MPS fp32 (baselines.py:71-84) | min(len, 2048) (baselines.py:90) | 6c3c6c2 |
| chronos_bolt_base | amazon/chronos-bolt-base @5d9f166 | same | MPS. `_mem_guard` batch-halving, single-series CPU fallback on OOM (bench.py:823-868) | ≤2048 | cce8070 |
| tirex | NX-AI/TiRex @63c7409 | tirex-ts 1.4.2 | MPS (baselines.py:110) | ≤2048 (:113) | 6c3c6c2 |
| timestral_5m/28m/209m | mistralai/Timestral-{5M,28M,209M} @bdd20cf/49b2640/9874d25 | timestral 1.0.0 | CPU (baselines.py:147) | full given context ≤2048, NaN left-padded | f983682 / f983682 / 6c3c6c2 |
| auto_ets, auto_theta, seasonal_naive_pi | statsforecast 2.0.1 | — | CPU, n_jobs=-1 | last 512 after fill_nans (baselines.py:27, :42) | c8d003c / c8d003c / f983682 |

- **Gap: weights are not pinned.** `from_pretrained(model_id)` takes no revision. The snapshot ids above are what the
  cache holds *now*. Pin them in baselines.py, or record them in results.tsv, before the TEST run.
- **Parity check (this audit).** All 9 CSVs have identical task order, n_forecasts (16,885), sn_wql and sn_mase on
  all 30 tasks. All non-finite counts are 0 (SN-PI predates the column). The VAL scoring path (`_build_task`,
  `evaluate`, prepare.py) has no metric change between f983682 and HEAD. The only diffs are input copies, NaN
  flagging, TRAIN_DROP and guards.
- **Information given.** Alma reads 256 points by default (train_v2.py:38, :128) and up to 2048 with TSAR_PREFIX or
  TSAR_CONTEXT. Every foundation baseline reads up to 2048 (8x more) and the stats models read 512. **The baselines
  get more context than Alma.** That is a disadvantage Alma chose (ctx512/1024/2048 screens gave no gain,
  research_log:241-252), not a leak. No model sees targets, covariates or other series (Chronos-2 cross-learning off).
- research_log.md:24-30 (Chronos-2 0.621, Timestral 0.569, …) are **old prepare.py-suite numbers**, not bench
  numbers. Do not mix them.

## 3. Cross-check with published fev-bench numbers

- **Not possible from local files.** No fev result CSVs, no `tasks.yaml` and no `fev` package exist in the repo,
  `.venv` or HF cache. To do it you would need `benchmarks/fev_bench/results/*.csv` from github.com/autogluon/fev at
  f323f6c (per-task WQL, MASE and SQL for chronos-2, tirex, seasonal_naive). Even then the per-task numbers would differ
  because of window averaging, the context cap, subsampling and Chronos-2's multivariate mode. The useful check is
  the per-task *ratio to SN* and the *rank* of Chronos-2 vs TiRex.
- **Plausibility from the aggregates** (Chronos-2 paper, Tables 3 and 7, 100 tasks):

| | Published SQL skill → gmean rel | Ours wql_rel (skill) | Published MASE skill → rel | Ours mase_rel |
|---|---|---|---|---|
| Chronos-2 | 47.3% → 0.527 | 0.559 (44.1%) | 35.5% → 0.645 | 0.720 |
| TiRex | 42.6% → 0.574 | 0.582 (41.8%) | 30.0% → 0.700 | 0.727 |

  Our levels are close despite a different metric and task mix. The ranking (Chronos-2 > TiRex) is preserved. The
  Chronos-2/TiRex gap is smaller here (0.960 vs 0.918). That is expected, because we removed its multivariate and
  covariate mode and the 8192 context. Chronos-2 wins 21 of 30 tasks on wql_rel against TiRex, and its largest edges
  are uk_covid cumulative (0.70x, 0.85x), solar_1W (0.82x) and bizitobs_1H (0.90x). **Plausible, with no sign of a broken run.** Caveat: our Chronos-2 is a
  *weakened* Chronos-2 relative to fev.

## 4. Contamination of VAL families (baselines pretraining)

| VAL family | Chronos-2 | Chronos-Bolt | TiRex (std ckpt) | Timestral |
|---|---|---|---|---|
| solar (NREL) | **Likely**: "Solar" in training Table 6 (2510.15821) | **Likely**: Chronos corpus Solar 5min/H (2403.07815 App. B) | **Likely**: Chronos corpus (2505.23719 Table 5) | Unknown |
| kdd_cup_2022 | Claims fev-clean (§5.1). Only KDD-2018 listed | No | **Likely**: GIFT pretrain "kdd2022" (Table 6). fev flags TiRex on 10T | Unknown |
| favorita_stores | Claims fev-clean | No | **Possible**: "favorita sales" (same Kaggle source, item level) | Unknown |
| ETT, LOOP_SEATTLE, M_DENSE, SZ_TAXI, jena, bizitobs, hier. sales, restaurant | No (§5.1, GIFT-test clean) | No known | No (marked GiftEval-ZS, Table 7) | Unknown |
| uk_covid_* | No | No | No | Unknown |

- **Timestral's real post-training data is undisclosed** (protocol.md §3). Every VAL family is "unknown" for it.
- **Selection exposure.** Every VAL family except favorita, kdd and covid is a GIFT-Eval *test* dataset. All four
  model families were developed and published against GIFT-Eval, so their design was selected on these series. This
  is weaker than training on them, but they are not pristine. Alma is likewise selected on VAL (about 60 screens),
  so neither side is zero-shot on VAL in the selection sense.
- Note: bench.py:185-196 `VAL_EXPOSURE` lists only Moirai-2 flags plus solar/favorita. **Add TiRex→kdd_cup_2022 and
  favorita, and Chronos/Bolt/TiRex→solar.**

## 5. Comparability verdict and caveats for "Alma 0.681/0.695 vs Chronos-2 0.559"

Arithmetically comparable: same tasks, windows, targets, denominators and code path (parity verified). Caveats:

1. **Selection.** 0.681 is the best single seed (syn5_10k s0, research_log:249) out of about 60 VAL screens. Its
   3-seed analogue (syn5_only) is 0.6945 ± 0.008. 0.695 is a 3-seed mean (b_d_a6p). Both are val-selected and
   optimistic. The baselines are untuned single runs.
2. **Not fev-bench.** The metric is WQL not SQL, windows are pooled, the SN has no intervals, 13/30 tasks are
   subsampled, and only VAL families (30 of 100 tasks) are used. Do not quote it as a fev skill score.
3. **Chronos-2 is handicapped** compared with its published setup: 2048 vs 8192 context, univariate only, no
   covariates. Its real fev-protocol strength is higher.
4. **Contamination** cuts the other way for solar (all Chronos-lineage), kdd_cup_2022 and favorita (TiRex), and is
   unknown for Timestral. Report family-level results with and without these families.
5. **Single-run baselines** with unpinned checkpoints, MPS nondeterminism, and Bolt's CPU fallback path. Seed-level
   CIs exist only for Alma.
6. **Context asymmetry.** The baselines read 2048 points and Alma reads 256. That is fair (Alma's choice), but the
   table must show each model's effective context (protocol.md §5).
7. **Gap statement.** Alma's error is about 22-24% higher than Chronos-2 on VAL (compare.py ratio 1.227 for syn5_only,
   research_log:220). This is a VAL gap, with a family-bootstrap CI, not a zero-shot result.

**Before Gate G:** pin the HF revisions; add the exposure rows above; optionally add SQL with window-averaged
aggregation and fev clipping as a secondary column, so that one fev-convention number exists.
