# Audit: investor sprint (harness agent, 2026-10-31)

Scope: splits and TEST access, leakage (GIFT-clean TRAIN, synthetic generators, overlap with VAL/TEST, fev-bench and
GIFT-Eval), the world-model test (`wm_ssm_none_s*` vs `wm_ssm_direct_s*`), and the headline metric.
Branch `alma/investor-sprint`, HEAD `bb8983f`. I did not edit any training or evaluation code, and I did not open TEST.
The only code I ran is `tools/audit_sprint_probe.py`, on CPU under `nice -n 10`: about 6 s and 176 MB peak RSS per
subcommand. It loads checkpoint configs, parses logs, and builds the SSM family on random tokens.

## Checklist

| # | Item | Verdict |
|---|---|---|
| 1a | TRAIN / DEV / VAL / TEST construction; no training path reads VAL or TEST | **PASS** |
| 1b | Frozen TEST: has it been touched? (`data/bench/test_access.log`) | **PASS** (no forecast or score on TEST). **CONCERN**: the log only records access through `bench._guard` |
| 1c | Upcoming official fev-bench run (100 tasks = 48 TEST + 30 VAL + 22 other) | **CONCERN, veto-armed**: it consumes TEST and bypasses the guard (see the fixes) |
| 2a | Real data in training under `TSAR_GIFT_CLEAN=1`; overlap with VAL/TEST | **PASS** (VAL side checked by hashes only, as in audit_r1) |
| 2b | Overlap with the 100 fev-bench tasks | **CONCERN**: 7 exact tasks (m5 ×3, ercot ×4) plus possibly `australian_tourism` must be self-flagged. The 30 VAL tasks are selection-exposed |
| 2c | Overlap with GIFT-Eval | **PASS** for training (the 14 datasets are GIFT-free). **CONCERN**: 9 VAL families are GIFT test datasets (selection exposure; disclose) |
| 2d | Synthetic generators (synth3, synth5, built-ins) | **PASS**: no file reads |
| 2e | Normalisation and statistics on future values; TTA / paths / RECAL | **PASS** |
| 3a | WM test: recipe, environment, data, parameters, MACs, steps, eval identical except `TSAR_WM_ABL` | **PASS** for s0. **CONCERN** for s3/s4 (training code differs from the rollout runs; verified inert by inspection and probe, but `wm/*.py` is not hashed) |
| 3b | Is 'direct' a fair non-rollout twin? | **PASS**, with two disclosed limits |
| 3c | Seed selection for the WM test (seeds 0, 3, 4) | **FAIL**: the stated reason is false, and the choice favours the rollout |
| 3d | WM test status | **NOT YET A RESULT**: 1 of 3 paired seeds done (s0: rollout better by 0.022) |
| 4a | `val_wql_rel` / `family_wql_rel` computation; re-derivation | **PASS** |
| 4b | Comparability with `docs/goal.md` | **PASS** (same harness, same tasks and denominators) |
| 4c | Comparability with `docs/alma_goal.md` (fev SQL skill, GIFT rel. CRPS) | **FAIL if conflated**: no current number is on the alma_goal scale |

---

## 1. Splits and TEST access

### 1a. Construction: PASS
- **VAL and TEST tasks.** VAL is 30 tasks in 13 families and TEST is 48 tasks in 18 families. They are fev-bench task
  specs at f323f6c, and the data is pinned to `fev_datasets@d3e7e72` (`tools/bench.py:71`, `:86-167`).
  - The families are disjoint: `audit()` asserts it (`:926`).
  - EXCLUDED_FAMILIES (`:171-183`) keeps every TRAIN source (ercot, m5, hospital and others) out of both splits.
  - Windows are built as `ctx = y[max(0, c-2048):c]` and `target = y[c:c+H]` (`:770`, `:775`). Contexts end strictly
    before the cutoff.
- **TEST guard** (`bench._guard`, `:255-275`).
  - It needs `TSAR_SPLIT=test`.
  - Model scripts must be `train.py` or `train_v2.py` with a sha-matched frozen checkpoint.
  - Every access is appended to `data/bench/test_access.log`.
  - `_guard_config` (`:277-281`) blocks `_build_task` on a TEST config even if the caller skipped `_guard`.
  - `train_wm.py:150` refuses any `TSAR_SPLIT` other than val, even in eval-only mode.
- **TRAIN.** `load_train` (`:670-705`) reads only `data/bench/train/*.parquet`. These are whole series: no window of a
  TRAIN series is ever scored.
- **DEV** (`tools/devsplit.py`). It holds out whole TRAIN families: traffic, aus_electricity, nn5, tourism, and
  dominick under GIFT_CLEAN. Import-time asserts (`devsplit.py:64-68`) forbid any VAL/TEST/excluded/TRAIN_DROP collision.
  DEV calls `bench.evaluate(tasks=dev)`, so VAL and TEST are never loaded.
- **Selection during training.** In `train_wm.py` the eval happens once, after the fixed `TSAR_STEPS`
  (`TSAR_EVAL_MIN` is unset in every sprint `.prov.json`). There is no early stopping on VAL.

### 1b. TEST access log: PASS, with a CONCERN
- There are 51 entries, from 2026-10-29 13:22 to 2026-10-31 11:39.
- **Every entry is an overlap scan, marked target-free:**
  - `bench.py overlap --split test`;
  - `overlap_scan.py stamp-extra`, `stamp-gep`, `confirm-gep` and `selftest --known-tp`;
  - three 2026-10-29 "growth-corr (contexts only)" scratch scans.
- **None is a forecast, a baseline or a score.**
- "Target-free" means `_series_spans(include_target=False)` (`bench.py:987-1004`, `:1025`). This is the union of
  contexts. Because window k+1's context contains window k's target, it exposes all TEST values except the last
  window's target.
  - That exposure was used only to *remove* TRAIN series (TRAIN_DROP) and to vet GEP. It is acceptable and was
    accepted in audit_v2 and audit_v3.
- **CONCERN 1: the log only covers the guard.** Any read outside `bench._guard` leaves no trace: a direct
  `pd.read_parquet` on `data/bench/fev/*`, the official `fev` package, or a scratch script.
  - File access times cannot back the log up. APFS does not update atime here: every fev parquet, VAL included, still
    shows its 2026-10-29 13:26 download time.
  - So "TEST untouched" rests on the guard plus team discipline.
- **CONCERN 2: the GEP filter is being tuned against TEST.** `confirm-gep` was run 35 times on 2026-10-31,
  across 4 commits (89a36e8 → 261c88b), on TEST tasks (redset, boomlet, epf_fr).
  - The aim is to decide which TEST lag hits count as "filtered" rather than "residual".
  - This tunes a contamination filter on TEST hit outcomes, and the uncommitted `tools/overlap_scan.py` holds that
    filter.
  - It is not label leakage into a model. But it weakens the guard in the permissive direction.
  - GEP is unusable today: there is no `overlap_gep.json` stamp, and `TSAR_BENCH_GEP` is absent from every sprint run.

### 1c. Official fev-bench full eval: CONCERN, veto-armed
- `docs/benchmark_contract.md` §0.1: of the 100 tasks, 48 are our TEST tasks, 30 are VAL and 22 are other.
- `docs/alma_goal.md` §5 requires that "the frozen final model is scored on test once", yet P0 says "score the current
  Alma SSM on both benchmarks".
- Running the 100-task fev harness on a non-final model therefore spends the single TEST look. The `fev` package does
  this outside `bench._guard`, so it is also unlogged.
- **I will not accept a 100-task fev number unless the fix in R1 is followed.**

## 2. Leakage

### 2a. GIFT-clean TRAIN: PASS
- **Checkpoint configs** (probe `wm`). All 6 finished sprint checkpoints have the same values:
  - `train_sha256 bc78d74b5a3f`, 10,758 series, 14 datasets, `train_drop_sha256 372a45bd827a`.
  - The datasets are dominick, ercot, m5, monash_australian_electricity, cif_2016, m1_monthly, m3_monthly,
    m3_quarterly, nn5_weekly, tourism_monthly, tourism_quarterly, traffic, weather and nn5.
  - `bench_extra` 0, and GEP is off.
- **Overlap with VAL/TEST.** `docs/bench_integrity.md` §1 scanned these datasets (a subset of the 17 LOCAL datasets):
  0 VAL hash hits, and TEST macro duplicates are removed by TRAIN_DROP.
  - The VAL side was checked by hashes only (exact + z-normalised, L=32). No lag scan was run on VAL. This is the
    same caveat as audit_r1 §5.

### 2b. fev-bench: CONCERN
- **TRAIN ∩ fev (exact).** m5 (`m5_1D/1W/1M`) and ercot (4 tasks). `monash_tourism_*` may overlap
  `australian_tourism`. `hospital` no longer overlaps, because GIFT_CLEAN drops monash_hospital.
  - Under fev rules these 7 (or 8) tasks must be self-flagged and imputed with Chronos-Bolt (Base) scores.
- **Selection exposure.** The 30 VAL tasks are fev tasks Alma was selected on: about 60 or more VAL screens, plus the
  family choice (ssm was promoted on its seed-0 VAL score).
  - A fev score over all 100 tasks is therefore not zero-shot in the selection sense on 30% of tasks.
- **Related, unflagged.** `kdd_cup_2022_10T/30T` and `LOOP_SEATTLE_5T` share a source with VAL families.

### 2c. GIFT-Eval: PASS for training, CONCERN for disclosure
- None of the 14 TRAIN datasets is a GIFT-Eval dataset. All 13 GIFT overlaps are dropped by `bench.py:689`. A
  `testdata_leakage: No` flag is justified.
- 9 VAL families are GIFT test datasets: ETT, LOOP_SEATTLE, M_DENSE, SZ_TAXI, bizitobs_l2c, hierarchical_sales,
  jena_weather, solar and restaurant. That is selection exposure. Disclose it next to every GIFT number, and report the
  GIFT score with and without those configs.

### 2d. Synthetic generators: PASS
- `tools/synth3.py` and `tools/synth5.py` (imported at `train_wm.py:220`, `:224`) and the built-in `synthetic` /
  `synthetic2` read no files.
  - `grep read_parquet|open(|np.load|json.load|pickle` finds only a `__main__` PNG writer in synth3.
- The frequency mix in synth3 `FREQS` is a hand prior over the fev/GIFT seasonality convention. It is not fitted to any
  split.

### 2e. Future values at inference: PASS
- `causal_stats` (`train_wm.py:306`) uses cumulative sums read at patch ends. `seasonal_ref` (`:350`) indexes
  `ends + k - m*ceil(k/m) <= origin` and uses a causal forward fill.
- Eval de-normalises with `ctx.loc[:, -1]` and `ctx.scale[:, -1]` of the last context patch (`:856`).
- The array holds only the ≤256 context points, so nothing beyond the origin exists to read.
- Long horizons feed back the model's own median (`:825-832`).
- `TSAR_WM_LONG=paths`, TTA `sym` / `clip` and row independence were all verified in audit_r1 §1, §3 and the probe
  table. The code is unchanged since.
- `TSAR_WM_RECAL` fits only on `CAL_SRC`, which is the TRAIN pool or the DEV holdout, both TRAIN families
  (`train_wm.py:485-491`, `fit_recal`). It is off in every sprint run.
- `TSAR_TTA_STEPS > 0` (the shared-weight adaptation leak, audit_v3 H-1) is refused by `bench._guard` (`:257`).

## 3. World-model test

### 3a. Recipe identity
- **Environment.** Every run has `TSAR_AUG=0.5 BENCH=1 GIFT_CLEAN=1 SPLIT=val STEPS=2500 SYN3=1 SYN5=1 SYN_CURR=0.9
  SYN_FRAC=0.3 WM=ssm`. Only `TSAR_WM_ABL`, `TSAR_SEED` and `TSAR_RUN` differ: this is from the probe `cfg_mismatch`
  over the checkpoint `cfg.env`, and from `runs/jobs/sprint.txt` for s3/s4.
- **Data.** Identical train hash (2a). The numpy data rng depends only on the seed, so a none/direct pair with the same
  seed sees the same batches.
- **Model.** `params=365937` and `macs_per_forecast=13765632` for both arms, with the same state_dict shapes.
  - In direct mode each of the K=8 leads takes one mixer step from the origin state. In rollout mode it takes K
    sequential steps. That is the same number of steps and the same matmuls (`wm/ssm.py` at HEAD c508393 `:221-229` vs `:230-248`).
- **Compute.** Training time on the same machine: none_s0 2077 s vs direct_s0 2008 s. Peak MPS memory 3352 vs
  3250 MB.
- **Eval.** The same 30 tasks and the same `n_forecasts` in every log (probe). Raw median long-horizon mode, no TTA or
  paths on either arm.
- **Code versions.** Taken from `.prov.json` `sha256` and matched against git.

| run | train_wm.py | bench.py | wm/ssm.py |
|---|---|---|---|
| none s0-s3, direct s0 | 12c51ddd (9c572d5) | 163af08f | **not recorded** (b464a5de at those commits) |
| none s4 | **3dc5f7ad (uncommitted, unrecoverable)** | 1bb0b510 | not recorded |
| direct s3 (running), s4 (queued) | 3c04c42f (1c1f823/c508393) | 1bb0b510 | not recorded; the working tree changed again *during* direct_s3 (period-lever functions added, uncommitted) |

- **Verification of the code-version differences.**
  - `bench` 163af08f → 1bb0b510 adds only `GEP_DROP`, which is inert with GEP off (audit_r1).
  - `train_wm` 12c51ddd → 3c04c42f, training path: `TSAR_WM_ORIGINS` (0 → branch skipped, no rng draw) and
    `TSAR_WM_LEADW` (1.0 → mask unchanged). Both are no-ops at their defaults by inspection.
  - The probe (`ssmbit`) builds SSMFamily with torch seed 0 from three versions: 9c572d5 (b464a5de), HEAD (f733243d)
    and the current working tree. All four ablations give identical initial state_dicts and max |Δq| = 0.0 on random
    tokens. So the ssm.py edits are bit-identical at default env.
- **Verdict: PASS for the s0 pair, CONCERN for s3/s4.** The residual risk is unhashed `wm/*.py`, `synth*.py` and the
  unrecoverable none_s4 `train_wm.py`.
  - `tools/jobq.py:87` hashes only the script, bench, prepare and jobq.

### 3b. Is 'direct' a fair twin? PASS, with two limits to disclose
- It shares the encoder, every parameter and the per-lead input `U_k = step_emb[k] + drive(ref_k)`.
- Lead k's output depends only on the origin state and `U_k`: there is no carried state. The selective `dt` lets
  `step_emb[k]` learn a k-patch jump.
- `wm/base.py` requires a parameter- and MAC-matched replacement for the transition. This one is exactly matched.
- **Limit 1: it tests one architecture, not every direct decoder.** It is a one-step jump of the *same* diagonal SSM,
  not a generic direct head (for example an MLP or attention head from `o_j` to all K patches).
  - A win shows that within this architecture the rollout beats the matched jump.
  - It does not show that the rollout beats any direct decoder. For comparison, the GRU line's direct head *tied* its
    rollout (research_log 2026-10-30 20:10, +0.0020 over 3 seeds).
- **Limit 2: horizons beyond 64 steps.** The rollout covers K·PATCH = 64 steps. Beyond that, both arms are chained by
  the harness, which feeds back the median (`train_wm.py:825`).
  - On the 8 VAL tasks with h > 64, 'direct' is therefore not "one shot over the whole horizon" as alma_goal §2 words
    it.
  - Report the h ≤ 64 and h > 64 subsets separately.

### 3c. Seed selection: FAIL
The brief says "seeds 1,2 of the rollout used a different recipe (no SYN3/SYN5) and must not be mixed in". **The
evidence says otherwise:**
- `runs/wm_ssm_none_s1.prov.json` and `_s2.prov.json`, `runs/jobs/wave5.txt:28-29`, and the checkpoint `cfg.env` all
  show `TSAR_SYN3=1 TSAR_SYN5=1`.
- Train hash, git, code sha and every env key except seed match s0 and s3. Seeds 1 and 2 *are* the same recipe.
- `wm_ssm_direct_s1` and `_s2` were queued with that same recipe and then parked (`runs/jobs/wave6.txt:38-39`,
  `#PARK`). `sprint.txt` replaced them with s3 and s4.

**Effect of the choice:**

| rollout seeds | mean wql_rel |
|---|---|
| {0,3,4} (sprint) | **0.6760** |
| {0,1,2,3,4} | 0.6804 |
| {1,2,3,4} (fresh: excludes s0, the seed that selected ssm) | 0.6828 |
| {1,2} (the dropped seeds) | 0.6870 |

- The two dropped seeds are the rollout's worst (0.6936 and 0.6804).
- Seed 0 is the seed that promoted ssm (wave5.txt:27). Preregistration §3 says "the seed that selected a config never
  counts in its confirmation".
- The sprint design keeps the selection seed and drops the two worst seeds. That biases the test toward the rollout by
  about 0.004-0.007, which is a third of the 0.022 seed-0 margin.
- Whatever the intent, a WM-test result on seeds {0,3,4} would not be accepted.

### 3d. Current evidence (s0 only)
- Seed 0: rollout 0.6707 vs direct 0.6927, Δ = +0.0220 in favour of the rollout.
- Per-task seed-0 log(direct/none): mean +0.032. The family-cluster 95% CI is +0.0008..+0.065, which barely excludes 0.
  Wins / ties / losses at ±2% are 16 / 11 / 3.
- The margin is concentrated. uk_covid_nation contributes +0.134, the noisiest family (per-task seed SD up to 0.128,
  preregistration §0), and M_DENSE +0.101. solar goes the other way (−0.073).
- A leave-one-family-out check is needed before any claim.

## 4. Metric

### 4a. Computation: PASS
- `bench.evaluate` (`tools/bench.py:807-866`):
  - per task, `wql_rel = wql / sn_wql`. WQL is pooled over every window and series of the task, at quantiles
    0.1..0.9 (`prepare._score`). SN is the degenerate seasonal naive.
  - `wql_rel` = geometric mean over the 30 tasks, clipped below at 1e-6 with no upper clip (`:802`, `:847`).
  - `wql_rel_family` = geometric mean over the 13 families of the per-family geometric mean (`:844`, `:850`).
- `train_wm.py` prints them as `val_wql_rel` (`:981`) and `family_wql_rel` (`:996`).
- The probe re-derived both from the logged per-task tables for all 6 finished runs. They agree within the 3-decimal
  rounding of the tables (≤ 1e-4).

### 4b. vs `docs/goal.md`: PASS
- goal.md's table has the 5-seed ssm mean 0.680, seeds 0-2 raw → TTA 0.6815 → 0.6712, and baselines Chronos-2 0.559
  and Timestral-209M 0.564.
- These use the same task-weighted `wql_rel`, tasks, windows and SN denominators. audit_r1 §4 verified the same CSVs.

### 4c. vs `docs/alma_goal.md`: FAIL if conflated
- alma_goal's tiers are **fev SQL skill** (100 tasks, per-window scaled quantile loss, statsforecast SN,
  `1 - gmean(clip(·, 0.01, 100))`) and **GIFT relative CRPS** (97 configs).
- `val_wql_rel` differs in several ways:
  - the metric (WQL, not SQL);
  - window pooling;
  - the SN denominator (degenerate, no intervals);
  - no upper clip;
  - a 1000-forecast cap on 13 of 30 tasks;
  - only 30 VAL tasks.
- `1 - 0.680 = 32%` is **not** an fev SQL skill and must never be put next to the 46.2-51.8 values.
- The leaderboard values in alma_goal §3 (for example TimeRSI-1.5 51.8, TimeH2O-v1 50.8) have no local source file.
  Before any claim, fetch the fev and GIFT result CSVs at pinned commits (benchmark_contract §4, item 5).

---

## Required fixes (blocking before the WM test is reported)

1. **W1 (FAIL 3c).** Un-park `wm_ssm_direct_s1` and `wm_ssm_direct_s2` (exact lines in `runs/jobs/wave6.txt:38-39`,
   about 35 min each).
   - Report the WM test on **all 5 paired seeds** (0-4).
   - Make the primary analysis the 4 fresh seeds {1,2,3,4}, with s0 shown separately as the selection seed.
   - Use paired per-seed Δ with a seed t-interval, plus a family-cluster bootstrap of seed-mean per-task log ratios.
     Both must exclude 0.
   - Correct the "seeds 1,2 used a different recipe" statement wherever it was written.
2. **W2 (CONCERN 3a).**
   - Re-score all 10 checkpoints eval-only under one pinned code version. This removes eval-code differences, including
     none_s4's unrecoverable `train_wm.py`.
   - Do not change `train_wm.py` or `wm/ssm.py` on disk while `direct_s4` is queued, or record the diff.
   - (jobq owner) Add `wm/*.py`, `tools/synth3.py` and `tools/synth5.py` to the prov sha256 list.
3. **W3 (3b).** State the two limits of the direct twin next to the result: same-SSM one-step jump, and h > 64 chained
   in both arms. Report h ≤ 64 and h > 64 separately, and leave-one-family-out.
4. **W4 (rules §5).** Raw and TTA numbers on the same seeds for both arms. Direct has no TTA score yet.

## Required before the public benchmarks

- **R1 (1c).** Choose one option and write it down before any run:
  - (a) Score only the 52 non-TEST fev tasks during development. Keep the 48 TEST tasks sealed until the frozen P4
    model.
  - (b) Declare a fixed, small number of full-fev looks (for example a P0 baseline look and the final), log each one in
    `test_access.log` by hand with the checkpoint sha, and report every look.
  - In either case the official fev runner must refuse TEST configs without `TSAR_SPLIT=test` and must log the access.
- **R2 (2b).** Self-flag m5 ×3 and ercot ×4 (and `australian_tourism` unless the source is shown to differ) and impute
  them per the fev policy. Report fev with and without the 30 VAL tasks.
- **R3 (2c).** GIFT: `testdata_leakage: No` is justified. Disclose the 9 VAL-family configs, and report the score with
  and without them. Limit the number of GIFT looks, because each look is selection.
- **R4 (1b).** GEP: do not stamp on a filter tuned against TEST hits. Use the conservative EXTRA policy (drop every
  flagged key), or freeze the confirm filter on VAL and selftest data before running it on TEST.
- **R5 (4c).** Never map `val_wql_rel` to SQL skill. Fetch the public result CSVs before quoting leaderboard values.

---

## Result acceptance checklist (reusable)

A result is accepted only when every applicable line is checked with evidence (file:line or numbers).

**Provenance**
- [ ] Every run has `.prov.json` with `exit 0`, `done`, `ckpt_sha256`, the full TSAR env and code sha256. The script,
      bench, prepare, `wm/<family>.py` and generators are committed or diffable.
- [ ] Arms differ only in the declared key(s). Diff the checkpoint `cfg` (train_sha256, train_series, drop hash) and the
      env. Code differences are shown to be inert, by bit-identity probe or by re-scoring under one version.

**Splits and leakage**
- [ ] `TSAR_SPLIT` is val, or dev under `TSAR_DEV`. TEST appears in `test_access.log` only for an authorised, declared
      look with a sha-matched frozen checkpoint.
- [ ] Training data is TRAIN only (`train_sha256` matches the known corpus, or a new corpus has a clean overlap stamp).
      `TSAR_BENCH_EXTRA` / `GEP` are used only with a current clean stamp.
- [ ] No selection on TEST. VAL screens are counted (K) and the winner's curse is stated.

**Inference-only tricks (TTA, paths, RECAL, ensembles, period detection, test-time adaptation)**
- [ ] Every input is the row's own context (≤ the 2048 points the bench passes) or the model's own outputs. No target,
      no other row, no future index. Prove row independence (a row-swap probe as in audit_r1).
- [ ] No shared-weight adaptation over a batch of rolling windows (`TSAR_TTA_STEPS` is refused for this reason).
- [ ] Anything fitted (calibration maps, thresholds) is fitted on TRAIN/DEV only. The setting is chosen on VAL and
      counted as a screen.
- [ ] Raw and trick numbers on the same seeds. Gains of combined tricks are measured jointly, not added.

**Periodicity / new features**
- [ ] The period detector, phase and template for origin j use only points ≤ the end of patch j, in training (every
      origin of a window) and in eval. Probe: perturb points after origin j and check the outputs are unchanged.
- [ ] The detector has no free parameter tuned on VAL or TEST. Its window (`PER_LW`) is the same in train and eval.
      The season metadata it may read is the same metadata the baselines get.

**Statistics and aggregation**
- [ ] ≥ 3 seeds (5 for claims, and when 0.013 ≤ Δ < 0.020). Seeds are paired by id. No seed is dropped. The selection
      seed is excluded from confirmation. Every finished seed of the recipe is reported.
- [ ] The headline is the task-weighted geometric mean of `wql_rel` over all 30 VAL tasks (no subsets), with
      `family_wql_rel`, `mase_rel` and `cov80_error` alongside. The CI is a family-cluster bootstrap plus a seed
      t-interval. Leave-one-family-out shows no family carries > 40%.
- [ ] Baselines are re-derived from `baselines/bench/per_task/*_val.csv` with identical tasks, `n_forecasts` and
      `sn_wql`.

**Public benchmarks (fev-bench, GIFT-Eval)**
- [ ] Official harness at a pinned version, with the full task list and the leaderboard's failure-imputation policy.
      fev headline is SQL skill. GIFT headline is CRPS/MASE gmean vs SN.
- [ ] TEST policy R1 is followed and the look is logged. TRAIN-overlap tasks are self-flagged and imputed. Scores are
      shown with and without the VAL-exposed tasks/configs.
- [ ] Published baseline values come from pinned result files, not memory. "Significant" means a paired bootstrap over
      tasks with Holm correction across the named baselines (alma_goal §4).
- [ ] Efficiency is reported (params, training compute, CPU latency, peak memory), with each model's effective context.

**World-model claims**
- [ ] The matched direct twin has the same params, MACs, data and steps. Paired seeds as above. The rollout wins with
      both CIs excluding 0. The limits of the twin and the h > native-rollout chaining are disclosed.

## Probe

`tools/audit_sprint_probe.py`:
- `wm`: checkpoint configs, log re-derivation, paired stats.
- `ssmbit`: ssm.py bit-identity across versions.

Re-run `wm` once direct s1-s4 finish. It picks up any `wm_ssm_direct_s*` checkpoint and log automatically.
