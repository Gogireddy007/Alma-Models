# Audit v2: leakage, evaluation parity, claim gate, pre-registered final claim

Auditor: Dr. Sofia Andrade (independent evaluation, red team). Date: 2026-10-29.
Branch `autoresearch/oct29`, HEAD `6c3c6c2`.

**Scope and method.** This audit was read-only. I read these files:

- `docs/protocol.md`, `docs/eval_rigor.md`, `docs/bench_integrity.md`;
- `tools/bench.py`, `tools/compare.py`, `tools/run.py`, `tools/seeds.sh`, `tools/dump_baseline.py`;
- `train.py`, `baselines.py`, `prepare.py` (scoring only);
- the `runs/queue*.sh` scripts and `baselines/bench/`.

The only computation was pandas on the existing `baselines/bench/per_task/*_val.csv` files, run under `nice -n 19`. No model was trained or run, and no TEST data was opened.

**Bottom line.**

- **The data split itself is sound.** I found no path by which VAL or TEST values reach Alma's training, normalisation or season map in the default `TSAR_BENCH=1` configuration.
- **The claim machinery is not ready.** The documented claim gate (`compare.py claim`, eval_rigor R4) does not work on the bench. It either crashes or silently compares against old-suite baseline files.
- **The confirmatory run would test different weights.** As written, the TEST run retrains the model rather than scoring the weights that were selected on VAL.
- **Two leakage paths are open.** Neither is in the default configuration, but both are reachable through flags: checkpoint resume and `TSAR_BENCH_EXTRA`.
- **None of these needs compute to fix.** All must be fixed before TEST is touched (section 5, gate L).

Severity scale:

- **CRITICAL**: a claim made today would be invalid.
- **HIGH**: a leak or invalid claim is reachable through a plausible action.
- **MEDIUM**: it biases or weakens the inference.
- **LOW**: it should be disclosed or tidied up.

---

## 1. Can VAL or TEST information reach training or model selection?

### Verified clean (no action needed)

| Item | Where | Finding |
|---|---|---|
| Data loading | `train.py:108-111`, `bench.py:350-372` | With `TSAR_BENCH=1`, real data comes only from `load_train()`, which reads TRAIN_LOCAL files and applies `TRAIN_DROP`. VAL and TEST parquet files are read only by `_build_task`, which is reachable only through `load_tasks`/`evaluate`/`overlap`. `train.py` calls `evaluate` only at line 734 (optional periodic val) and line 748 (final). |
| Normalisation stats | `train.py:289-315` (`causal_stats`), `:460`, `:662` | Normalisation is per window and causal: up to the end of each patch, within the given context only. No global or dataset-level statistics exist, so nothing is fitted on VAL or TEST. |
| SEASON map | `train.py:123` | Under BENCH it is built from the TRAIN specs (`s["season"]` from `TRAIN_LOCAL`). At inference the season comes from the task spec `m` (`train.py:664`), the same value every baseline receives. |
| Early stopping / checkpoint selection | `train.py:703-741` | None. The final model is the end-of-budget model (`TSAR_STEPS` or minutes). The periodic val eval (`:732-739`) only prints. |
| Long horizons | `train.py:632-638` | Chunked rollout feeds back the model's own median. No targets are used. |
| VAL/TEST disjointness | `bench.py:588-608` | `audit()` asserts that VAL ∩ TEST families = ∅ and that no excluded family is present. The overlap scan gives VAL 0 hits and TEST 0 hits after `TRAIN_DROP` (`docs/bench_integrity.md`). |

### Issues

**L-1 (HIGH): checkpoint resume checks only `TSAR_ZS`, so an old-suite or mismatched checkpoint can be resumed under BENCH.**

- **Where:** `train.py:682-690` (resume) and `:696-697` (saved fields: model, ema, opt, step, done, rng, zs).
- **Risk:** `TSAR_RUN=<name> TSAR_BENCH=1` resumes any existing `data/ckpt/<name>.pt`. On disk today there are `big1.pt`, `big2.pt`, `exp2x.pt` and `q_*.pt`. Those were trained on `prepare.load_train_series()`, which includes `monash_fred_md`, `exchange_rate` and `monash_covid_deaths`. `bench.TRAIN_EXCLUDED` (`bench.py:211-215`) bars exactly these because they overlap TEST `us_consumption`/`gvar` and VAL `uk_covid`.
- **Current mitigation:** the queues `rm -f` the checkpoint first (`runs/queue5-7.sh`), but nothing in the code enforces this. A warm start from `big2` would be a silent TEST leak.
- **Fix:**
  - Save `bench=BENCH`, the full `TSAR_*` env dict, a fingerprint of the train set (sorted `(dataset, key)` list hash, including the `TRAIN_DROP` state), `BENCH_EXTRA` and the git hash in the checkpoint.
  - On resume, refuse on any mismatch.
  - Refuse any checkpoint that lacks the `bench` field when `BENCH=1`.

**L-2 (HIGH): `TSAR_BENCH_EXTRA=1` adds 21 corpora that were never overlap-scanned.**

- **Where:** `train.py:111`, `bench.py:218-228`, `docs/bench_integrity.md` §1 "Limits".
- **Risk:** M4 Monthly and M4 Yearly (and M1/M3 Yearly) carry anonymised macro and demographic series. That is exactly how 18 BEA/GVAR duplicates entered M3/M4 Quarterly and Monthly. The TEST families `us_consumption_1M/1Q/1Y`, `gvar`, `world_co2_emissions`, `world_life_expectancy` and `world_tourism` are at risk.
- **Fix:**
  - `train.py` must refuse `TSAR_BENCH_EXTRA=1` unless a committed scan record exists. For example, `data/bench/overlap_extra.json` with the hash of the TRAIN_EXTRA file list.
  - Run `bench.py overlap --split test --extra`, plus the lag scan, extended to the yearly and monthly extras against all macro and `world_*` TEST tasks.
  - Extend `TRAIN_DROP` accordingly.

**L-3 (MEDIUM): the TEST guard runs on an honour system.**

- **Where:** `bench.py:247-256`.
- **Risk:** all 48 TEST parquet files are on disk in `data/bench/fev/`. The guard only covers `load_tasks`/`evaluate`/`overlap`, so a direct `pd.read_parquet` bypasses it. Three of the six entries in `data/bench/test_access.log` were written by hand. The growth-correlation scan script lives in a session scratchpad, not in the repo, so the evidence behind `TRAIN_DROP` cannot be reproduced.
- **Fix:**
  - Move `data/bench/fev/<test configs>.parquet` out of the working tree, for example into a sealed archive, until the confirmatory session. At that point verify each file against the sha256 in `manifest.json`.
  - Commit the lag-scan script as `tools/overlap_lag.py` and let the guard write its log entries.
  - Make `_build_task` itself call `_guard` for TEST configs.

**L-4 (LOW): `TSAR_ZS` / `ZS_FAMILY` are meaningless under BENCH.**

- **Where:** `train.py:76-77`, `:117-121`, `:761`.
- **Risk:** `TSAR_ZS=1 TSAR_BENCH=1` silently removes six TRAIN datasets from the bench corpus: the old ZS4 sets plus `monash_nn5_weekly` and `monash_tourism_quarterly`. It then prints `zs4_wql_rel: nan`. This does not leak, but it is a confusing configuration. It is also the only config field that resume checks.
- **Fix:** have `train.py` raise when `ZS and BENCH`.

**L-5 (LOW, disclose): one TRAIN decision was made using TEST data.**

- **Where:** `bench.py:338-343`.
- **What happened:** `TRAIN_DROP` was derived by scanning TEST contexts. It only removes data, and it removes data in the direction that reduces leakage, so this is acceptable. It is still a TEST-informed decision and must be listed in the final report. No targets were read (`bench_integrity.md` §4).

**L-6 (LOW, disclose): the synthetic prior uses benchmark seasonality conventions.**

- **Where:** `tools/synth3.py:44-46`.
- **What happened:** `FREQS` uses the fev/GIFT period conventions (288, 144, 96, 48, 24, 7, 52, 12, 4) with hand-set frequency weights. These are calendar facts, not data. If the weights were chosen by looking at the VAL or TEST task mix, that is a weak design prior.
- **Fix:** state in the report how the weights were chosen.

**L-7 (LOW): periodic VAL evaluation invites manual early stopping.**

- **Where:** `train.py:732-739`.
- **Rule:** whenever `TSAR_EVAL_MIN` is set, the final model is the end-of-budget checkpoint. Choosing a step from the `EVAL` lines counts as model selection on VAL. That is allowed, but it must be counted in the multiplicity budget (S-3).

---

## 2. Are Alma and the baselines evaluated identically?

### Verified

- **Same scorer.** `train.py:124` sets `EVALUATE = BN.evaluate`. Baselines go through `bench.run_baselines` and then the same `evaluate` (`bench.py:759`). Both share the WQL/MASE/coverage code (`prepare._score`), the seasonal-naive denominator and the geometric mean (`bench.py:510-518`).
- **Same tasks.** My CSV check over all six val result files found:
  - identical task lists, `n_forecasts` (16,885 in total) and horizons;
  - identical `sn_wql` per task;
  - `n_nonfinite = 0` for every FM.
- **Same quantiles.** All models are scored on `P.QUANTILES` (0.1 to 0.9).
- **Same horizons and seasons.** `evaluate` passes the same `(contexts copy, h, m, freq)` to every model. Input hashes were verified identical in `bench_integrity.md` §3.
- **Same target masking.** NaN targets are masked in the same way (`prepare._score`).

### Issues

**E-1 (MEDIUM): `protocol.md` misstates the context length.**

- **Where:** `protocol.md:174` ("The last 2048 points, for every model").
- **What actually happens:** 2048 is a shared cap (`bench.py:65`, `:436`), but the context each model actually uses differs:

| Model | Context used | Source |
|---|---|---|
| Statistical models | 512 | `baselines.py:27`, `:42` |
| TimesFM | 1024 | `baselines.py:132` |
| Chronos, TiRex | 2048 | `baselines.py:90`, `:113` |
| Timestral | Full 2048 (no own crop) | `baselines.py:150-155` |
| Alma | `CONTEXT=256` by default, 512, or `PREFIX+CONTEXT` ≤ 2048 | `train.py:46`, `:96-97`, `:652` |

- **Assessment:** this is fair, because each model chooses how much context to use, but the protocol text is wrong.
- **Fix:** correct the protocol text and report each model's effective context in the results table.

**E-2 (MEDIUM): NaN handling differs by design.**

- **Input side:**
  - Alma, Chronos and TiRex take NaN natively.
  - Timestral gets NaN left-padding (`baselines.py:151-154`).
  - The statistical models get ffill/bfill/0 (`prepare.fill_nans`).
  - Every model receives the same arrays, so this is fair.
- **Output side:** NaN/inf forecasts are scored as 0 for everyone (`bench.py:497-500`).
- **Fix:**
  - The final claim must require `n_nonfinite == 0` for Alma on every task.
  - `seasonal_naive_pi_val.csv` predates the `n_nonfinite` column. Rerun it at the frozen commit.

**E-3 (MEDIUM): baselines were scored at two commits.**

- **Where:** `results.tsv`. SN-PI, 5M and 28M were scored at `f983682`; 209M, TiRex and Chronos-2 at `6c3c6c2`.
- **Assessment:** I verified that tasks, counts and SN denominators are identical, so the val numbers are comparable.
- **Fix:** TEST baselines must all run at the single frozen commit.

**E-4 (MEDIUM): `predict_seconds` cannot support an efficiency claim.**

- Timestral runs on CPU (`baselines.py:147`); the others run on MPS (`baselines.py:73`).
- The machine was loaded by queued jobs.
- Wall-clock times include first-call model loading.
- See gate E in section 5 for the controlled protocol.

**E-5 (LOW, disclose): Alma uses the season `m`; the foundation models ignore it.**

- **Where:** `train.py:664` (`seasonal_ref`), `train.py:502` (seasonal skip).
- Every model receives the same `m`, so this is fair. It is still an informational input that Alma uses and Chronos, TiRex and Timestral do not.
- fev sets `m = 1` for 11 of the 30 VAL tasks and 11 of the 48 TEST tasks. On those tasks the pathway is a plain naive copy.

**E-6 (LOW): headline weighting is uneven.**

- The headline `wql_rel` is a geometric mean over tasks. On TEST, boomlet contributes 15 of 48 tasks (31%).
- `wql_rel_family` (`bench.py:510`, `:516`) corrects for this.
- Both must be gated (section 5).

---

## 3. Is the claim gate statistically sound?

**S-1 (CRITICAL): the documented claim tooling does not work on the bench.**

`docs/eval_rigor.md` R4 says a claim requires `compare.py claim … PASS`. On the bench:

| What fails | Where | What happens |
|---|---|---|
| `run.py` dumps nothing | `tools/run.py:31`, `:69` | It patches only `prepare.evaluate`. Under BENCH, `train.py` calls `bench.evaluate`, so `TSAR_DUMP` is a **silent no-op**: no `per_dataset_*.csv`, no `preds_*.npz`, no `meta.json`. |
| `seeds.sh` cannot summarise | `tools/seeds.sh:35` | It ends with `compare.py seeds "runs/dumps/NAME_s*"`, which finds no dumps and exits. |
| `compare.py` cannot read bench logs | `tools/compare.py:51` | The header regex `^\s*dataset\s+wql\s+mase\s+wql_rel` does not match the bench table header (`dataset family n_forecasts …`, printed at `bench.py:525`). The string columns `family` and `freq` would also fail the float parse at `:66`. |
| Baseline names resolve to the old suite | `tools/compare.py:100-108` | A name such as `chronos2` loads `baselines/per_dataset/chronos2_val.csv`, the **old 15-dataset suite** (`monash_tourism_monthly`, …), not `baselines/bench/per_task/`. The dataset intersection is empty, so it crashes; or, with an old-suite log, it produces an old-suite verdict. |
| Series bootstrap uses old windows | `tools/compare.py:155-161` | It rebuilds windows from `prepare.load_suite()`, so it cannot run on bench tasks. `tools/dump_baseline.py:30` has the same problem. |
| Gate lacks the family interval | `tools/compare.py:312-319` (verdict) | Even when fed bench CSVs, the verdict uses a task-level bootstrap and a t-interval. It has no family-cluster interval, which `protocol.md` §6 makes primary. |

**Fix:**

- Write a single bench claim entry point, for example `tools/claim_bench.py`, on top of `bench.compare` (section 5 gives the exact spec).
- Make `run.py` also patch `bench.evaluate` so it dumps `per_task_<split>.csv` and `preds_<split>.npz` keyed by task.
- Make `dump_baseline.py` bench-aware, writing to `baselines/bench/dumps/<name>/`.
- Make the regex in `compare.py` refuse bench logs loudly rather than mis-parse them.
- Commit all of this, with a unit test on two baseline CSVs, before freezing.

**S-2 (CRITICAL): the TEST run would score a different model from the one selected on VAL.**

- **Where:** `train.py:747-748`, `tools/seeds.sh:13-14`, `docs/eval_rigor.md` R5.
- **What happens:** with `TSAR_SPLIT=test`, `train.py` trains again from scratch and scores the new model. MPS is not bit-reproducible (`eval_rigor.md:20-21`), so the TEST model is a new draw, not the frozen candidate.
- **Partial mitigation:** the queues use `TSAR_RUN`, and a completed checkpoint resumes straight into evaluation (`prog >= 1` at `:707`). However, `seeds.sh` forbids `TSAR_RUN`, and resume does not verify the configuration (L-1).
- **Fix:**
  - Add an eval-only path, for example `TSAR_EVAL_ONLY=1`. It loads a frozen checkpoint, asserts its stored config and its sha256 against the pre-registration record, and scores the requested split.
  - Refuse `TSAR_SPLIT=test` unless `TSAR_EVAL_ONLY=1` is set and the checkpoint's `done` equals its budget.
  - Score VAL from the same checkpoint in the same session, and require it to reproduce the recorded val score within 1e-3.

**S-3 (HIGH): multiplicity and winner's curse.**

- **Where:** `runs/queue5-7.sh`, `train.py:40`, `protocol.md:218`, `:230`.
- **Selection pressure:** about 7 bench configurations are queued (b_base256/512, syn3, aug, curr, prefix, prefixsyn) on top of about 15 old-suite ablations. All run `train.py` directly, so every one uses `SEED = 0` with the same data and synthetic RNG streams. The only seed override is in `run.py`, and the queues do not use it.
- **Consequences:**
  - Differences between configurations confound the config change with a single draw of training noise.
  - Picking the best of k on VAL biases its VAL score upward.
  - `protocol.md:218` requires Holm correction, but nothing implements it.
  - `protocol.md:230` allows 3 candidates on TEST with no correction across candidates.
- **Fix:**
  1. Pre-register **one** primary candidate. Others may be reported as exploratory, with no claim.
  2. Run the primary candidate on **fresh** seeds (5–9), never seed 0, before freezing (R7).
  3. If more than one candidate carries a claim, use Bonferroni across candidates: 3 candidates means 98.33% intervals.
  4. The headline "beats **every** baseline" is an intersection–union test. Each component must pass at α, and no correction is needed across baselines or metrics *for the conjunction*. Holm is needed only if individual "beats B" statements are reported separately as discoveries. State this explicitly in place of the current Holm sentence.

**S-4 (HIGH): the two gates are inconsistent.**

- `protocol.md` §6 requires a family-cluster CI, a majority of families and a Holm-corrected sign test.
- `eval_rigor.md` R4 requires a dataset bootstrap, a hierarchical bootstrap and a seed t-interval, and does not use the family CI.
- The family CI matters. On VAL the task interval is too narrow. For example, Timestral-209M vs Chronos-2 gives task CI [0.976, 1.048] but family CI [0.972, 1.063].
- **Fix:** one gate, defined in section 5, which supersedes R4 for the bench.

**S-5 (MEDIUM): the bootstrap ignores training noise.**

- **Where:** `bench.py:553-554`.
- `bench.compare` averages log-scores over seeds before bootstrapping, so the CI reflects task and family sampling only. It computes `seed_ratios` but no gate uses them.
- **Fix:** also require every seed's ratio to be below 1, plus a seed t-interval on the per-seed log ratios with upper bound below 1 when there are at least 5 seeds.

**S-6 (MEDIUM): few, unbalanced clusters.**

- **Where:** `bench.py:562-567`.
- A percentile cluster bootstrap with G = 13 (VAL) or 18 (TEST) families under-covers. Resamples that draw boomlet several times (15 tasks each) dominate the mean.
- **Fix:**
  - Gate on both the task-weighted ratio (existing) and a family-weighted ratio: the mean over families of the per-family mean log ratio, bootstrapped over families.
  - As a cross-check, add a family-level t-interval with G − 1 degrees of freedom.
  - Report a sensitivity analysis that excludes boomlet.
  - Use a fixed bootstrap seed (0) and B = 10,000.

**S-7 (LOW): sign-test ties.**

- **Where:** `bench.py:568`.
- `d < 0` / `d > 0` treats a tie only at exact equality, whereas `compare.py` uses a 0.1% band.
- The sign test is secondary, so harmonise on the 0.1% band and report it, but do not gate on it.

**Power, from val baselines (CSV only).**

| Pair (val) | Ratio | Task CI | Family CI | Wins/losses (tasks) |
|---|---|---|---|---|
| Timestral-209M / Chronos-2 | 1.009 | [0.976, 1.048] | [0.972, 1.063] | 20/10 |
| Timestral-28M / Chronos-2 | 1.024 | [0.991, 1.059] | [0.985, 1.075] | 18/12 |
| TiRex / Chronos-2 | 1.042 | [1.016, 1.074] | [1.013, 1.075] | 9/21 |
| Timestral-5M / Timestral-28M | 1.022 | [1.010, 1.038] | [1.013, 1.032] | 6/24 |

The family-CI half-width between strong models is about 3–5%. For the gate to pass reliably against Chronos-2, Alma probably needs a point ratio around ≤ 0.95. An improvement of about 1% would most likely read as "comparable". Plan for this.

---

## 4. Fairness caveats to disclose in any report

1. **Alma was tuned on VAL; the baselines are zero-shot everywhere.** Alma's VAL numbers are therefore optimistic and must never appear as zero-shot comparisons. Only TEST supports a zero-shot claim.
2. **Foundation models may have trained on TRAIN families, by design.** TRAIN is the public Chronos corpus subset. Some VAL families are exposed to some models (`VAL_EXPOSURE`). Timestral's real post-training data is undisclosed, so its "no TEST exposure" is self-reported. Undisclosed exposure would favour the baselines, not Alma.
3. **AutoETS and AutoTheta fall back to non-seasonal models at m ≥ 144** (`bench_integrity.md` §2). Affected tasks: VAL jena_weather_10T and bizitobs_l2c_5T; TEST boomlet 288/1440 and redset_5T. They are weak references on those tasks. They are also not yet scored on the bench (`results.tsv`). If they enter the baseline set, say so.
4. **Effective contexts differ** (E-1). Alma by default uses only 256 points, against 2048 for Chronos-2, TiRex and Timestral.
5. **Alma uses the task season m** (E-5).
6. **Two documented deviations from fev:** at most 1000 forecasts per task, and a 2048-point context cap. Numbers are therefore not comparable with the public fev leaderboard.
7. **TEST-informed TRAIN filtering** (L-5) and **benchmark-convention synthetic periods** (L-6).
8. **Baseline set.** Moirai-2, TimesFM-2.5 and Toto are not evaluated. "Every baseline" means the pre-registered set below, and nothing more.
9. **Hardware.** Everything ran on an M3 with 8 GB under load. Efficiency numbers come only from the controlled protocol (gate E).

---

## 5. Pre-registered final-claim protocol

This section is binding once committed. Any deviation must be logged with a reason in `research_log.md` **before** TEST is run. Anything that changes after TEST is seen is reported as a deviation, never silently.

### 5.0 Preconditions: all must be done and committed before freezing

| ID | Item | Fixes |
|---|---|---|
| P1 | `tools/claim_bench.py` implementing the gates below on `bench` per-task CSVs and dumps, with a unit test that compares two existing val baselines | S-1, S-4 to S-6 |
| P2 | `run.py` dumps under `bench.evaluate`; `dump_baseline.py` is bench-aware | S-1 |
| P3 | `TSAR_EVAL_ONLY` path with config and sha256 assertion; TEST refused without it | S-2 |
| P4 | The checkpoint stores and verifies the full config, the `bench` flag and the train fingerprint | L-1 |
| P5 | `BENCH_EXTRA` is refused without a scan record (or is not used) | L-2 |
| P6 | TEST parquets sealed out of the tree; lag-scan script committed | L-3 |
| P7 | `ZS` with `BENCH` raises an error | L-4 |
| P8 | `protocol.md` context sentence corrected; R4 in `eval_rigor.md` points to this gate | E-1, S-4 |

### 5.1 What is frozen (record in `RESEARCH_PLAN.md`, committed, before TEST)

- **One primary candidate C\*:** the git hash and the full `TSAR_*` env. It must include `TSAR_BENCH=1` and `TSAR_STEPS=<N>`. It must not include `TSAR_SPLIT`, `TSAR_ZS` or `TSAR_EVAL_MIN`. `TSAR_BENCH_EXTRA` may be set only if P5 is satisfied.
  - Up to 2 secondary candidates may be listed as **exploratory**: reported, no claim.
- **Seeds:** 5 fresh seeds, {101, 102, 103, 104, 105}, none of which was used during development. Each is trained once with `TSAR_RUN=final_s<seed>` via `tools/run.py`.
  - Record each checkpoint's sha256, its `done`/steps and its val result.
  - If a seed crashes, rerun the same seed and log the crash. No seed may be dropped for being bad.
- **Baseline set B (fixed):**
  - Chronos-2, Timestral-209M, Timestral-28M, Timestral-5M, TiRex, Seasonal-Naive-PI.
  - Any further baselines (auto_ets, auto_theta, timesfm_2p5, chronos_bolt_*) are listed now or never.
  - The efficiency gate excludes SN-PI (a trivial reference).
- **Ablations A1 to A3** (gate W) are trained with the same seeds and steps.
- **Analysis:** `tools/claim_bench.py` at its frozen hash, bootstrap seed 0, B = 10,000.

### 5.2 TEST procedure (one session)

1. Restore the TEST parquets and verify each sha256 against `manifest.json`. Run `bench.py audit` and `bench.py overlap --split test` with `TRAIN_DROP` applied. Expect 0 hits.
2. Score each baseline: `TSAR_SPLIT=test uv run tools/bench.py baselines --split test <B…>`, at the frozen commit, on an idle machine, with dumps.
3. For each seed checkpoint of C\* and each ablation, run `TSAR_EVAL_ONLY=1 TSAR_SPLIT=test`. Each run must first reproduce its recorded val score within 1e-3.
4. Run `tools/claim_bench.py --split test` once. Commit its output, `test_access.log` and every dump in the same commit.
5. Make no changes after this point. A bug that invalidates the run gets a logged rerun of the **same** frozen artefacts. A model change means a new pre-registration and a fresh test set, which this benchmark does not have. So: report as is.

### 5.3 Success gates

Notation:

- **r(B):** the geometric-mean ratio Alma/B of the per-task metric, with Alma's per-task logs averaged over 5 seeds.
- **CI_fam:** the 95% family-cluster bootstrap interval of the task-weighted ratio.
- **CI_famw:** the same for the family-weighted ratio.
- **seed condition:** every seed's ratio is below 1, and the 5-seed t-interval upper bound is below 1.
- **IUT:** each gate below must pass for **every** B in the set, at two-sided 95% (one-sided α = 0.025). No correction is needed across baselines or across A, C, E, Z for the conjunctive claim. Use Bonferroni only across claim-bearing candidates.

**A. Accuracy (wql_rel and mase_rel, separately).** For every B in the set:

- **A1:** the upper bound of CI_fam is below 1, the upper bound of CI_famw is below 1, and the seed condition holds.
- **A2:** Alma wins (log ratio < −0.001) on a strict majority of families: at least 10 of 18.
- **A3 (report, no gate):** per-family ratios, the worst family, results by frequency and domain, and the sensitivity analysis excluding boomlet.
- **A4:** `n_nonfinite = 0` on every task.
- **"Large margin" (secondary claim):** r(best B) ≤ 0.95 *and* the upper bound of CI_fam ≤ 0.98. Below that, say "beats", not "beats by a large margin".

**C. Interval calibration.**

- Per task, let e = |cov80 − 0.80|. For every B, the paired difference e_Alma − e_B (Alma averaged over seeds), mean over tasks, must have its family-bootstrap 95% CI upper bound below 0. The point value of `cov80_error` must be at or below B's.
- **Multi-level (needs P2 dumps on both sides):**
  - Compute the empirical coverage of the central 20/40/60/80% intervals, i.e. the quantile pairs (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9).
  - Compute the one-sided hit rate P(y ≤ q_τ) for all 9 levels τ.
  - Gate: the mean absolute calibration error over the four central levels is lower than every B's, with the same CI rule.
  - Report the 9-level reliability table.
- Calibration alone can be gamed by wide intervals. Sharpness is guarded by gate A (WQL), so C is claimed only if A passes.

**E. Efficiency (against the foundation models in B).**

- **E1, parameters:** Alma's total trainable parameters (`num_params_M`) must be fewer than every foundation model's. Count each baseline in code (`state["params"]`), not from the model card.
- **E2, inference:**
  - Setup: idle machine, a single device type for all models (MPS where supported; also report CPU-only for every model), fixed batch 64, and an untimed warm-up on one task.
  - Measure seconds per 1,000 forecasts over all TEST tasks, median of 3 repeats.
  - Gate: Alma is at or below every foundation model on the same device.
- **E3, training compute:**
  - Report Alma's training FLOPs (estimate: 6 × params × tokens processed) and wall-clock hours for the final run, **plus the total over all tuning runs** on this branch.
  - Claim "less training compute than B" only where B's pretraining compute is published. Otherwise write "not comparable (undisclosed)".

**Z. Zero-shot generalisation.**

- **Z1, construction:** every TEST family is disjoint from Alma's TRAIN by family (`audit()`), by content (exact, znorm and lag scans with 0 remaining hits), and from VAL. TEST was never used for tuning (`test_access.log` holds only the logged content checks and this session).
- **Z2:** gates A and C pass on TEST. That **is** the zero-shot claim. VAL results are never presented as zero-shot.
- **Z3, report:**
  - The far-OOD subset: families with no TRAIN domain or frequency analogue (boomlet, redset, rossmann, rohlik, walmart, hermes, world_*, ecdc_ili, hospital_admissions, uci_air_quality, solar_with_weather), as against the near-domain families (entsoe and epf vs ercot and Australian electricity; us_consumption and gvar vs M3/M4 macro).
  - Gate A1 is recomputed on the far-OOD subset and reported. Its failure must be disclosed, but it does not by itself void the claim.
- **Z4, report:** the generalisation gap, i.e. Alma's ratio vs Chronos-2 on VAL compared with TEST. A large deterioration signals overfitting to VAL.

**L. No leakage: all must pass before step 2 of section 5.2.**

| Check | Pass condition |
|---|---|
| L1 | `bench.py audit` OK |
| L2 | `overlap --split val` gives 0 exact and 0 znorm hits. `overlap --split test` gives 0 after `TRAIN_DROP`, apart from the documented boomlet_772 false positive. Lag scan gives 0 remaining hits. If `BENCH_EXTRA` is used, all three are rerun with `--extra`. |
| L3 | The frozen env has `TSAR_BENCH=1`, has no `TSAR_ZS` or `TSAR_SPLIT`, and checkpoint metadata shows training from scratch under BENCH with the expected train fingerprint (P4) |
| L4 | `git grep -n "data/bench/fev\|load_test\|load_tasks(\"test\"" train.py` is empty, and `train.py` reaches TEST only through `EVALUATE` under `TSAR_EVAL_ONLY` |
| L5 | `test_access.log` has no `evaluate(`, `load_tasks(test)` or forecast entries before this session |
| L6 | Identical-input hash check: Alma's and Chronos-2's `(contexts, h, m, freq)` SHA-1 match on TEST, computed in the same session |
| L7 | VAL-informed decisions listed: config choice, TSAR_STEPS, synthetic mixture, any `TSAR_EVAL_MIN` readings |

**W. "Genuine world model".**

- **W1, architecture.** The frozen model must have each of these, and the report names the code for each:
  - an explicit latent state z_t from an encoder over observations (`states()`, `train.py:452-473`);
  - a transition z_{t+1} = f(z_t) applied **without new observations** (`rollout()`, `:482-494`, GRUCell or LoopedDynamics);
  - a decoder from latents to quantiles (`decode`, `:420`, `:498-503`).
- **W2, objective.** The transition must be trained to predict future latents. This means `TSAR_LATENT > 0` in the frozen config (`latent_consistency`, `:515-529`).
  - **Today the default is `LATENT = 0` (`train.py:61`).** In that configuration the transition is trained only through the forecast loss, and the model does **not** meet this criterion.
  - Measurable condition on VAL, then TEST: for k = 1…K, the k-step latent prediction error ‖ẑ_{t+k} − sg(z_{t+k})‖² must beat the copy-state baseline ‖z_t − sg(z_{t+k})‖² for every k, with the family-bootstrap CI excluding 0.
- **W3, the dynamics must matter.** Each ablation is trained with the same seeds and steps and scored on VAL and once on TEST. The gate is that the full model beats the ablation on wql_rel with the upper bound of CI_fam below 1 and the seed condition holding, on TEST:
  - **A1, no dynamics:** replace the transition with identity, z_{t+k} = z_t, keeping `step_emb`, so the decoder must forecast every step from the last observed latent.
  - **A2, no world-model objective:** `TSAR_LATENT=0`, otherwise identical.
  - **A3, seasonal-copy bypass:** the decoder adds `ref * ref_skip` directly to the output (`train.py:501-502`), so forecasts can bypass the latent. Evaluate the trained model with the imagined states zeroed before `decode`, which leaves only the seasonal-copy path. This shows how much of the forecast flows through the dynamics. Report the learned `ref_skip` weights.
- If W2 or W3 fails, the "world model" wording is dropped and Alma is described as a "latent-state forecaster". Accuracy claims can still stand.

### 5.4 Reporting format (whatever the outcome)

1. **Header:** frozen commit, `claim_bench.py` hash, checkpoint sha256 values, seeds, date, the `test_access.log` excerpt, and the deviations log.
2. **Main table (TEST):** one row per model. Columns:
   - `wql_rel`, `mase_rel`, `wql_rel_family`, `cov80_error` and the 4-level calibration error (Alma as mean ± sd over 5 seeds, n stated; baselines as single values);
   - params, effective context, s/1k forecasts (controlled), training compute.
3. **Paired table:** Alma/B for each B and each metric. Columns: ratio, CI_fam, CI_famw, task CI, seed range, family wins/losses, task wins/losses, and PASS/FAIL for each gate.
4. **Breakdowns:** per family, per frequency, far-OOD vs near-domain, excluding boomlet, worst family.
5. **World-model section:** latent prediction error vs copy-state; ablations A1 to A3 with their CIs.
6. **Disclosures:** section 4, verbatim.
7. **Wording rules:**
   - "beats B" only if A1 and A2 pass for B;
   - "beats every baseline" only if they pass for all B;
   - "large margin" only per gate A;
   - otherwise "comparable to B";
   - failed gates are printed, not omitted.
