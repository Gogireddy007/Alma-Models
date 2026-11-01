# Leak-free evaluation protocol (zero-shot TRAIN / VAL / TEST)

Written 2026-10-29, before any score on the TEST split below was computed. Implementation:
`tools/bench.py`. `prepare.py` and `baselines.py` are unchanged. The old suite stays in use as a
fast in-domain iteration benchmark.

## 1. Why a second protocol

The `prepare.py` suite has three fairness problems, found by audit:

1. **Unfair in our favour.** Our model trains in-domain on earlier windows of the very series it
   is scored on. Timestral, Chronos and TiRex are scored zero-shot.
2. **Unfair in their favour.** Foundation models may have pretrained on these public datasets.
   M4 and Monash are in the Chronos, TimesFM and Moirai corpora.
3. **Only five datasets are like-for-like.** Those are the 5 holdout datasets.

The field's answer is to split by **dataset (source) family**:

- the model never trains on any series from an evaluation family;
- each compared model's pretraining exposure is published per task.

This protocol applies that rule to every compared model, and to us.

## 2. What the field does (sources)

| Benchmark | Leakage handling | Source |
|---|---|---|
| Chronos Benchmark I/II | Benchmark I (15 sets) is in-domain: it is in Chronos's training data. Benchmark II (27 sets) is zero-shot and excluded from training. | Ansari et al. 2024, [arXiv:2403.07815](https://arxiv.org/abs/2403.07815), App. B Table 3 |
| GIFT-Eval | 23 datasets / 97 configs. A separate pretraining corpus, `Salesforce/GiftEvalPretrain` (71 uni + 17 multivariate sets, ~4.5M series, 230B points), is published as having "no leakage issue" with the test set. The leaderboard has a per-model `testdata_leakage` flag. | Aksu et al. 2024, [arXiv:2410.10393](https://arxiv.org/abs/2410.10393); [HF card](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain); [results/*/config.json](https://github.com/SalesforceAIResearch/gift-eval/tree/main/results) |
| fev-bench | 100 tasks. Each model self-reports per task whether it trained on that dataset at the same frequency. Leaked results are replaced by Chronos-Bolt scores. The paper reports a Leakage % per model. | Shchur et al. 2025, [arXiv:2509.26468](https://arxiv.org/abs/2509.26468) §7; [per-task flags](https://github.com/autogluon/fev/tree/main/benchmarks/fev_bench/results) (`trained_on_this_dataset`) |
| BOOM | Datadog observability metrics from a staging environment. The paper says they are disjoint from Toto's production training data. fev-bench includes 15 "boomlet" subsets. | Cohen et al. 2025, [arXiv:2505.14766](https://arxiv.org/abs/2505.14766) |

## 3. Who trained on what

The table below draws on three kinds of source:

- the model papers and model cards;
- the GIFT-Eval leaderboard flags;
- the fev-bench per-task flags (fev commit `f323f6c`, read from the result CSVs).

| Model | Real pretraining data | GIFT-Eval test leak flag | fev-bench tasks flagged |
|---|---|---|---|
| Chronos (T5) | 13 pretrain-only sets plus Benchmark I: Electricity, KDD Cup 2018, London smart meters, M4 D/H/M/W, Pedestrian, Rideshare, Taxi, Temp-Rain, Uber, and others. Also TSMixup and KernelSynth. [2403.07815](https://arxiv.org/abs/2403.07815) App. B | Yes | n/a |
| Chronos-Bolt | ~100B observations. The card names no datasets (UNVERIFIED; presumably the Chronos corpus). | Yes | 0 |
| Chronos-2 | Chronos corpus subset plus GiftEvalPretrain subset (Alibaba, Azure, Borg, LargeST, Q-Traffic, Buildings900K) plus synthetic. [2510.15821](https://arxiv.org/abs/2510.15821) §4.1 | No | 0 |
| TiRex | Chronos data with TSMixup, plus KernelSynth, plus ~2.5M GiftEvalPretrain series. The standard checkpoint overlaps 1616 of 9797 GIFT settings, so the leaderboard entry uses a separate `-gifteval` checkpoint. [2505.23719](https://arxiv.org/abs/2505.23719) §4, App. C | No (only for the `-gifteval` checkpoint) | 1 (kdd_cup_2022_10T) |
| TimesFM 1.0 / 2.0 | M4 (all frequencies), Electricity, Traffic, Weather, Favorita, LibCity, Wiki, Trends, synthetic. 2.0 adds a LOTSA subset. [2310.10688](https://arxiv.org/abs/2310.10688) Table 1; HF card | Yes / Yes | n/a |
| TimesFM-2.5 | Card: GiftEvalPretrain, Wiki, Trends, synthetic. The card is incomplete (see the fev flags). | No | 10 (favorita_transactions, fred_md, kdd_cup_2022_10T, m5, proenfo_gfc12/14/17) |
| Moirai 1.x / 2.0 | LOTSA, which includes Monash. 2.0 adds GIFT-Eval **train** splits, Chronos mixup and internal data. [2402.02592](https://arxiv.org/abs/2402.02592), [2511.11698](https://arxiv.org/abs/2511.11698) | No | 28 (ETT, LOOP_SEATTLE, M_DENSE, SZ_TAXI, bizitobs, jena, hierarchical_sales, restaurant, hospital, favorita, fred_md, m5, kdd2022, proenfo_gfc*) |
| Toto 1.0 | Datadog production metrics plus GiftEvalPretrain plus Chronos data (GIFT-overlapping subset removed) plus synthetic. | No | 8 |
| Timestral 5M/28M/123M/209M | Synthetic pretraining, then a "mix of real and synthetic" post-training. The real datasets are **not disclosed** ([blog](https://mistral.ai/news/timestral/); the tech report is "coming soon"). | No (self-report) | 0 (self-report) |
| **Ours** | `tools/bench.py: load_train()` (section 4) plus synthetic data | — | 0 by construction |

Two facts drive the design:

- **Chronos Benchmark II is not clean for every model.** TimesFM and Moirai overlap it (TiRex
  paper §4).
- **The GIFT-Eval test set is not clean at series level.** Moirai-2 trains on GIFT train splits,
  and standard TiRex overlaps part of it.

The cleanest public option is the fev-bench tasks that **no model flags at any frequency**. We
drop every family that is flagged at even one frequency.

## 4. The three splits

Splitting is by source family. `audit()` asserts that VAL ∩ TEST = ∅ and that no excluded family
appears in either.

### TRAIN: what our model may learn from (plus synthetic data generated in code)

**Local data (0 MB to download, ~520 MB already in `data/raw/`).** 17 of the 20 `prepare.py`
datasets, used as whole series:

- tourism M/Q, M1 M, M3 M/Q, M4 Q, CIF-2016, hospital, car parts;
- NN5 D/W, weather, M5, Dominick;
- ERCOT, traffic, Australian electricity.

**Optional `--extra` (~246 MB, `autogluon/chronos_datasets` @ `eeecad0`).** The public part of the
Chronos/Chronos-2/TiRex corpus, so that we train on what they trained on:

- M4 D/H/M/W/Y, M1 Q/Y, M3 Y, tourism Y;
- Electricity H/W, KDD Cup 2018, pedestrian counts, rideshare, Saugeen, temperature-rain;
- taxi 30min, Uber D/H, wind farms D/H.

**Deliberately excluded from TRAIN:**

| Dataset | Reason |
|---|---|
| `monash_fred_md` | Shares the FRED/BEA macro source with TEST `us_consumption` and `gvar` |
| `exchange_rate` | FX rates; overlaps the `ep` columns in TEST `gvar` |
| `monash_covid_deaths` | 2020 panel that includes the UK; overlaps VAL `uk_covid` and the TEST `ecdc_ili` period |

**Optional later.** Subsets of `Salesforce/GiftEvalPretrain`, which Chronos-2, TiRex, TimesFM-2.5
and Toto all used. Run `bench.py overlap` after adding any source.

### VAL: model selection and all tuning (30 tasks, 13 families, ~42 MB)

These are fev-bench tasks from families that share no source with TRAIN or TEST. Some
foundation models have seen some of them (`VAL_EXPOSURE`). That is acceptable, because VAL
numbers are only used to choose our configuration and are never reported as a zero-shot
comparison.

| Family (tasks) | Freq / H | MB | Known exposure |
|---|---|---|---|
| ETT (15T, 1H, 1D, 1W) | 96 / 168 / 28 / 13 | 2.6 | Moirai-2 |
| LOOP_SEATTLE (1D, 1H) | 28 / 168 | 17.1 | Moirai-2 |
| M_DENSE (1D, 1H) | 28 / 168 | 2.2 | Moirai-2 |
| SZ_TAXI (15T, 1H) | 96 / 168 | 3.4 | Moirai-2 |
| bizitobs_l2c (5T, 1H) | 288 / 24 | 0.4 | Moirai-2 |
| jena_weather (10T, 1H, 1D) | 144 / 24 / 28 | 3.2 | Moirai-2 |
| hierarchical_sales (1D, 1W) | 28 / 13 | 0.5 | Moirai-2 |
| restaurant (1D) | 28 | 0.4 | Moirai-2 |
| favorita_stores (1D, 1W, 1M) | 28 / 13 / 12 | 9.1 | Same source as favorita_transactions (TimesFM-2.5, Moirai-2, Toto) |
| kdd_cup_2022 (1D) | 14 | 1.9 | 10T variant flagged for 4 models |
| solar (1D, 1W) | 28 / 13 | 0.3 | Same NREL source as Chronos "solar" |
| uk_covid_nation (1D, 1W; new and cumulative) | 28 / 8 | 0.2 | none |
| uk_covid_utla (1D new, 1W cumulative) | 28 / 13 | 0.9 | none |

### TEST: entirely unseen, used once (48 tasks, 18 families, ~167 MB)

Every family below meets all three conditions:

- no compared model flags it **at any frequency** in fev-bench;
- it is not in any disclosed pretraining corpus;
- it shares no source with TRAIN or VAL.

| Family (tasks) | Domain | Freq | H | MB |
|---|---|---|---|---|
| boomlet (15) | Cloud observability (BOOM) | 1min / 5min / 30min / H | 60 / 288 / 96 / 24 | 42.5 |
| entsoe (15T, 30T, 1H) | European grid load (OPSD 2020) | 15min / 30min / H | 96 / 96 / 168 | 35.9 |
| epf (be, de, fr, np, pjm) | Electricity prices | H | 24 | 5.0 |
| solar_with_weather (15T, 1H) | PV generation | 15min / H | 96 / 24 | 3.1 |
| redset (5T, 15T, 1H) | Amazon Redshift query load | 5min / 15min / H | 288 / 96 / 24 | 13.1 |
| rossmann (1D, 1W) | Retail sales | D / W | 48 / 13 | 4.8 |
| rohlik_orders (1D, 1W) | Grocery orders, datetime cutoff 2023-05-01 | D / W | 61 / 8 | 0.2 |
| rohlik_sales (1D, 1W) | Grocery sales, cutoff 2023-12-15 | D / W | 14 / 8 | 37.9 |
| walmart | Retail sales | W | 39 | 3.1 |
| hermes | Fashion trends | W | 52 | 20.2 |
| us_consumption (1M, 1Q, 1Y) | BEA PCE | M / Q / Y | 12 / 8 / 5 | 0.2 |
| gvar | Macro, 33 countries | Q | 8 | 0.2 |
| world_co2_emissions | Emissions | Y | 5 | 0.1 |
| world_life_expectancy | Demography | Y | 5 | 0.1 |
| world_tourism | Tourism arrivals | Y | 5 | 0.02 |
| ecdc_ili | Influenza-like illness | W | 13 | 0.02 |
| hospital_admissions (1D, 1W) | Riyadh hospitals 2020-24 | D / W | 28 / 13 | 0.05 |
| uci_air_quality (1D, 1H) | Air pollutants | D / H | 28 / 168 | 0.3 |

**Excluded everywhere (`EXCLUDED_FAMILIES`):**

| Family | Reason |
|---|---|
| ercot, m5, hospital | In TRAIN |
| fred_md_2025, fred_qd_2025 | Same FRED source; the monthly version is flagged |
| favorita_transactions, proenfo_gfc12/14/17 | Flagged |
| australian_tourism | Possible overlap with Monash tourism |
| LOOP_SEATTLE_5T | Size |

Total new download is about 210 MB (VAL plus TEST), plus 246 MB if `--extra` is used. That is
well under the 2 GB target.

## 5. Windows and scoring (identical for every model)

**Task definitions.** Task definitions are taken verbatim from fev-bench: horizon, num_windows,
initial_cutoff (negative index, or a datetime with past ≤ cutoff), window_step_size, seasonality
and target columns. Data is pinned to `autogluon/fev_datasets@d3e7e72`, and every file's sha256
is stored in `data/bench/manifest.json`. Multivariate targets are scored as independent
univariate series, and covariates are not used. That is how univariate models are run in fev.

**Deviations from fev.** There are two, both made to keep the benchmark laptop-sized:

- **Series subsample.** At most 1000 forecasts per task. If series × windows exceeds 1000, a
  fixed-seed subset of series is drawn (seed 20261029 plus a CRC of the task name). All windows
  of the kept series are scored.
- **Context cap.** Every model is given the same contexts: the last 2048 points at most, with
  leading all-NaN padding dropped. 2048 is a shared cap, not the context each model uses. Each model
  crops further as it chooses (corrected 2026-10-29, audit_v2 E-1): statistical models 512, TimesFM
  1024, Chronos-2 and TiRex 2048, Timestral the full 2048, and Alma `TSAR_CONTEXT` (256 by default)
  plus an optional `TSAR_PREFIX`, at most 2048 in total. Report each model's effective context in
  the results table.

VAL comes to about 16.9k forecasts. TEST comes to about 27k, estimated from fev's counts. Counts match fev's `num_forecasts`
wherever no subsample was needed; this was checked on ETT, M_DENSE, SZ_TAXI_1H, jena,
uk_covid and rohlik_orders.

**Metric.** The metric is `prepare._score`, imported rather than copied:

- WQL over 9 quantiles;
- MASE of the median, scaled by the in-context seasonal difference;
- coverage of the 10–90% interval.

Each is divided by Seasonal Naive on the same windows.

**Headline numbers:**

- the geometric mean over tasks of `wql_rel` (fev convention);
- `mase_rel`;
- `cov80_error = mean |cov80 − 0.8|`;
- `wql_rel_family`, a geometric mean over families, so that 15 boomlet tasks cannot dominate.

**How baselines are scored.** Run `bench.py baselines --split val <models>`. It calls
`baselines.registry()` unmodified, scores on the same `load_tasks` objects, and writes to
`baselines/bench/`.

## 6. Statistics and the decision rule

`compare(a, b)` gives the paired ratio, which is the geometric mean over tasks of WQL_A / WQL_B.
Passing a list of per-seed DataFrames for A averages log-scores over seeds. It reports:

- a 95% percentile bootstrap over tasks;
- a **95% cluster bootstrap over families**, which is the primary interval because tasks from
  one source are correlated;
- wins and losses with an exact sign test;
- per-seed ratios and their SD, which measures training noise;
- per-family ratios.

**Rules:**

1. **Seeds.** Our model is trained with ≥ 3 seeds (5 if affordable) using the frozen config.
   Report mean ± SD of `wql_rel`. Baselines are deterministic and run once.
2. **Claiming a win.** Use **Gate G** in `docs/eval_rigor.md`, computed by `tools/compare.py
   claim`. It is the single gate for the bench, and it replaces the earlier three-part rule
   (family CI, family majority, Holm-corrected sign test) and eval_rigor R4. In short:
   - CI_fam and CI_famw (family-cluster bootstrap, task- and family-weighted) must have upper
     bounds below 1;
   - A must win a strict majority of families;
   - every seed must win, and the seed t-interval must be below 1;
   - there must be no non-finite forecasts;
   - all of this on `wql_rel` and on `mase_rel`.

   "Beats every baseline" is an intersection-union test with no correction. An individual
   "beats B" statement also needs a Holm-adjusted family-bootstrap p below 0.05. Holm is applied
   across all (baseline, metric) statements. The family sign test is reported with its Holm
   adjustment but is not gated. Otherwise write "comparable".
3. **Breakdowns.** Report results by frequency and domain, and the worst-family regression
   (`per_family_ratio`). Averages must not hide failures.
4. **Pre-registered gates.** The gates in `RESEARCH_PLAN.md` apply with TEST in place of
   "holdout".

## 7. Procedure

1. **Train and tune.** Train only on `load_train()` plus synthetic data. Tune only on
   `evaluate(fn, "val")`. Baselines may be scored on VAL for reference.
2. **Freeze.** Freeze at most 3 candidates (commit hashes) and the val-best baseline B in
   `RESEARCH_PLAN.md`.
3. **Score TEST once.** Score every baseline with `TSAR_SPLIT=test`. Score each candidate
   **from its frozen checkpoint only**, with `TSAR_EVAL_ONLY=<ckpt> TSAR_CKPT_SHA256=<sha>
   TSAR_SPLIT=test`. `train.py` refuses `TSAR_SPLIT=test` in any other form, and first
   re-scores VAL to within 1e-3 of the recorded value (`docs/eval_rigor.md` R5).
   - Without `TSAR_SPLIT=test`, `load_test()` and `evaluate(..., "test")` raise
     `PermissionError`.
   - Every access is appended to `data/bench/test_access.log`, with timestamp, git hash and
     argv.
4. **Check for duplicates.** `bench.py overlap --split {val,test}` hashes 32-point runs, exactly
   and z-normalised (near-duplicates), against every TRAIN series. For TEST it hashes contexts
   only, never a final target. The full check ran on 2026-10-29:
   - VAL: 0 hits.
   - TEST: two single-run z-norm hits. A follow-up lag-searching growth-correlation scan of the
     macro tasks found 18 TRAIN series (in M3 and M4) that duplicate BEA `us_consumption` and
     `gvar` series. They are removed from TRAIN by `bench.TRAIN_DROP`.

   See `docs/bench_integrity.md`.

   **TEST access note:** the TEST split was accessed (with `TSAR_SPLIT=test`) for this
   target-free overlap hashing and scan only. No forecast was made and nothing was scored. All
   6 accesses are logged in `data/bench/test_access.log`.

## 8. Open risks

- **Self-reported flags.** Exposure flags are self-reported. Timestral's real post-training data
  is undisclosed, and the TimesFM-2.5 card is known to be incomplete. A model could have trained
  on a TEST family without flagging it. Mitigations:
  - TEST contains many families released or curated recently (fev-bench, 2025);
  - report the result per family;
  - rerun the claims after excluding any family later found to be leaked.
- **Shared drivers in calendar time.** No series is shared, but TRAIN ERCOT/traffic and TEST
  epf_pjm/entsoe cover overlapping years and may share weather or calendar drivers. This is
  weak, cross-domain correlation, not leakage.
- **Old holdout numbers become in-domain.** Training on `load_train()` puts the old
  `prepare.py` holdout sets (car parts, Dominick, Australian electricity) in TRAIN. After that
  switch, the old "holdout" split is no longer a zero-shot number and must not be reported as
  one.
- **Statistical baselines on long seasonality (checked 2026-10-29).** Neither model fails or
  produces NaNs. However, for m ≥ 144 (jena_weather_10T, bizitobs_l2c_5T, boomlet 288/1440,
  redset_5T) AutoETS and AutoTheta silently fit non-seasonal models on the 512-point context.
  This is fair, since every model gets the same context, but they are weak references on those
  tasks. See `docs/bench_integrity.md` §2.
- **Deviations from fev.** The subsample and context crop mean our numbers are not directly
  comparable to the public fev leaderboard. Rankings should be consistent, but we have not
  checked this.
