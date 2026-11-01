# Evaluation rigor: decision rules and tooling

These rules sit on top of the keep rule in `program.md` and the pre-registered gates in
`RESEARCH_PLAN.md`. Where they differ, the stricter rule applies.

## Where the noise comes from

The headline is `val_wql_rel`: per-dataset WQL divided by seasonal naive's WQL, then a
geometric mean over the 15 development datasets. A difference between two runs mixes three
sources of noise:

| Source | What varies | How we measure it |
|---|---|---|
| Training noise | seed, MPS non-determinism, steps reached in a wall-clock budget | repeat seeds: `tools/seeds.sh`, then `compare.py seeds` |
| Eval-sample noise | which ≤300 series are scored per dataset | series bootstrap (`CI series`), needs `TSAR_DUMP` |
| Dataset choice | which 15 datasets are in the suite | dataset bootstrap (`CI datasets`) and the t-interval |

Measured so far: about 0.007 in `val_wql_rel` between 4-minute runs. Per-dataset spread is much
larger on a few datasets. `ercot`, `monash_cif_2016` and `nn5` move by 10% or more between runs, so
a single-dataset "win" there means little. Note that even the same seed is not bit-reproducible
on MPS, so a "seed" here really means one independent run.

The dataset bootstrap and the t-interval answer one question: does the gain generalize across
datasets? The seed interval answers another: is it more than training luck? Neither replaces the
other. A single-seed comparison has no estimate of training noise at all, and `compare.py`
says so when that is the case.

## Decision rules

**R1. Iteration (4-minute loop, val only).** Keep a change if it meets either of these:
- **(a) Single seed:** `val_wql_rel` improves by more than **2× the noise floor**, which is
  currently 2 × 0.007 ≈ **0.014**, about 2% relative. It must also lose on no more than 5 of 15
  datasets (`compare.py pair`, wins/losses line).
- **(b) Smaller gains:** the gain holds on **3 seeds** of both the candidate and the parent:
  `compare.py pair "runs/new_s*.log" "runs/parent_s*.log"` shows a ratio < 1 for every seed,
  and the seed 95% CI upper bound is below 1.

  Anything else counts as a tie, and the simpler, smaller or faster model wins (`program.md`).
  A gain under 0.007 on one seed is never a keep.

**R2. Noise floor upkeep.** Re-measure the noise floor with `tools/seeds.sh <name> 3` whenever
the default model changes materially: more than 1.5× the params, a new architecture family, or a
new budget. Record `compare.py seeds` output (mean, std, 2×std) at the top of `research_log.md`.

**R3. Reported numbers.** Every number that leaves the research log, such as a table, a claim or
a plot, is reported as **mean ± std over ≥ 3 seeds**, with n stated. Baselines from
`baselines.py` are deterministic forecasts and are reported as single values.

**R4. Claims against a baseline.** *On the bench (`TSAR_BENCH=1`), R4 is replaced by Gate G
below; the rest of this paragraph applies to the old `prepare.py` suite only.*
"Model X beats baseline B" may be written only if
`compare.py claim "runs/X_s*.log" B` returns **PASS**. That requires the upper end of the 95% CI
of the ratio X/B to be **< 1** for every interval available:
- the dataset bootstrap (always),
- the hierarchical dataset→series bootstrap (when both sides have dumps),
- and the seed t-interval (when both sides have ≥ 2 seeds).

Also report wins/losses and the Wilcoxon p-value. A ratio below 1 with a CI that crosses 1 is
written as "comparable to B", not "better than B".

**R5. Confirmatory runs (test and holdout) happen once, on frozen checkpoints.** Candidates and
the baseline set are frozen **before** any test or holdout score is seen (`docs/audit_v2.md` §5).
TEST never retrains: `train.py` refuses `TSAR_SPLIT=test` unless it scores a frozen checkpoint.

1. Train each seed with a checkpoint name: `TSAR_RUN=final_s101 TSAR_SEED=101 TSAR_DUMP=... uv run
   tools/run.py`. At the end of training the checkpoint is saved with `complete=True` and its full
   run config. The val score is written to `data/ckpt/final_s101.pt.val.json`, bound to the
   checkpoint's sha256. Record that sha256 in the pre-registration.
2. Score TEST from those exact bytes:
   `TSAR_EVAL_ONLY=data/ckpt/final_s101.pt TSAR_CKPT_SHA256=<sha> TSAR_SPLIT=test TSAR_DUMP=... uv run tools/run.py`.
   `train.py` restores the stored `TSAR_*` config and refuses any explicitly set value that
   differs. It then checks the sha256, requires `complete=True`, and re-scores VAL in the same
   session. If VAL does not reproduce the recorded score within 1e-3, it stops before TEST.
3. Apply Gate G with `--split test`. Results are reported whether they pass or fail. No retuning
   after looking. If a bug forces a re-run of the same frozen artefacts, log why in
   `research_log.md`.

`TSAR_EVAL_ONLY` on VAL is also the way to re-score any guarded checkpoint. On VAL only, the
inference-time settings (`TSAR_ENS`, `TSAR_TTA_*`, `TSAR_SELFCAL*`) may differ from training,
with a warning. That counts as exploratory and is never allowed on TEST.

**Checkpoint guard.** Every checkpoint stores its run config:

- the bench flag, `TSAR_BENCH_EXTRA` and `TSAR_BENCH_MAXLEN`;
- `TSAR_ZS`, the context, prefix and model sizes;
- the sorted train-dataset list and a sha256 of the exact training corpus (after `TRAIN_DROP`);
- the `TRAIN_DROP` hash, the full `TSAR_*` env and the git hash.

Resume refuses any mismatch except run-control keys: `TSAR_RUN`, `TSAR_SPLIT`, `TSAR_EVAL_MIN`,
`TSAR_DUMP`, `TSAR_DEVICE`, `TSAR_MINUTES`, `TSAR_EVAL_ONLY` and `TSAR_CKPT_SHA256`. A git
difference is printed, not refused. Under `TSAR_BENCH=1`, a checkpoint without a stored config
(any pre-guard or old-suite file) is refused. `TSAR_ZS=1` together with `TSAR_BENCH=1` is refused.

**`TSAR_BENCH_EXTRA`.** `bench.load_train(extra=True)` refuses to run unless
`data/bench/overlap_extra.json` exists, says `clean`, and matches the current TRAIN_EXTRA files,
lists and `TRAIN_DROP`. Only `tools/overlap_scan.py stamp-extra` writes that file. It runs the
exact/znorm hash scan on VAL and TEST plus the lag scan on the TEST macro/world tasks, all with
the extras included, and writes the stamp only if no unremoved hit remains.

**R6. Subsets.** The 4 ZS datasets (`--zs4`) and the 5 holdout datasets are too few for a
dataset bootstrap to mean much: with 5 datasets there are only 126 distinct resamples. On these,
report the per-dataset table and the series-level CI, and treat a dataset-level CI as indicative
only.

**R7. Multiple comparisons.** If you ran k variants and kept the best one, its single-seed val
gain is biased upward. Before you promote it to a candidate, re-run the winner on fresh seeds
(R1b).

## Gate G: the single claim gate on the bench

This is the only gate for a claim on the leak-audited bench. It supersedes R4 there and the
"claiming a win" rule in `docs/protocol.md` §6. It implements `docs/audit_v2.md` §5.3 gate A.
`compare.py claim` computes it; nothing else counts.

```sh
uv run tools/compare.py claim "runs/dumps/final_s*" --set final --min-seeds 5 [--split test]
```

**Notation.**

- A is our model. Its per-task log metric is averaged over seeds.
- B is one deterministic baseline.
- Each metric is gated separately: `wql_rel` and `mase_rel`.
- Bootstrap: B = 10,000 draws, seed 0.
- Tasks are clustered by source family: 13 families on VAL, 18 on TEST.

**For each B and each metric, all of these must hold:**

| Item | Condition |
|---|---|
| G0 | A and B were scored on the same task set and the same windows (`n_forecasts` and horizon match). |
| G1 | The upper end of **CI_fam**, the 95% family-cluster bootstrap interval of the task-weighted geometric-mean ratio, is below 1. This is the primary interval. (It equals `bench.compare`'s `ci95_family`; a unit test checks this.) |
| G2 | The upper end of **CI_famw**, the same interval for the family-weighted ratio (mean over families of the per-family mean log ratio), is below 1. This stops 15 boomlet tasks from carrying the verdict. |
| G3 | A wins on a strict majority of families. A family is a win when its mean log ratio is below −0.001. |
| G4 | **Seed condition.** Every seed's ratio is below 1. With 2 or more seeds, the seed t-interval upper bound is below 1. At least `--min-seeds` seeds are required: 3 to report anything, 5 for the final claim. |
| G5 | `n_nonfinite = 0` on every task, for every seed of A. |

**Verdicts.**

- **"A beats every baseline in the set"** is an intersection-union test. Every (B, metric) gate
  must pass at the unadjusted level. No correction is needed across baselines or metrics for
  this conjunction.
- **"A beats B"** as an individual statement requires gate G on both metrics for B. It also
  requires a Holm-adjusted p-value below α, with Holm applied across every (B, metric) statement
  in the run. Each statement's p-value is the larger of the two family-bootstrap p-values (G1
  and G2), so the p-value agrees with the intervals.
- **"Large margin"** requires a `wql_rel` ratio of at most 0.95 and a CI_fam upper bound of at
  most 0.98, against every B.
- **Calibration C1** requires the family-bootstrap CI upper bound of mean(|cov80 − 0.8|_A −
  |cov80 − 0.8|_B) to be below 0, and the point value to be no worse than B's. It is claimable
  only if accuracy passes.
- **Across candidates:** if more than one candidate carries a claim, pass `--candidates K`. All
  intervals then use level 1 − 0.05/K (Bonferroni).
- **Reported but not gated:**
  - the task bootstrap CI (too narrow, because tasks within a family are correlated);
  - the family t-interval (G − 1 df, a cross-check);
  - task and family sign tests (ties within 0.1%, with Holm-adjusted family sign-test p);
  - the Wilcoxon test.
- If a gate fails, write "comparable to B". Failed gates are printed, never omitted.

**Not covered by this tool.** These need TEST dumps on both sides: the multi-level
calibration table (gate C, 4 central levels), efficiency (gate E), the zero-shot report (gate Z)
and the world-model ablations (gate W). See `docs/audit_v2.md` §5.3.

## Tooling

| File | Purpose |
|---|---|
| `tools/run.py` | Drop-in for `uv run train.py`. Adds `TSAR_SEED` and `TSAR_DUMP=<dir>`. On the old suite it saves `per_dataset_<split>.csv`; on the bench (`TSAR_BENCH=1`, by patching `bench.evaluate`) it saves `per_task_<split>.csv`. It also saves the raw quantile forecasts (`preds_<split>.npz`) and `meta.json`, which includes the checkpoint sha256 under `TSAR_EVAL_ONLY`. It does not modify train.py on disk. |
| `tools/seeds.sh` | Runs N seeds of one config one after another, with dumps. Resumable, and refuses `TSAR_RUN` and `TSAR_SPLIT`. |
| `tools/compare.py` | `pair`, `claim`, `seeds`, `board`. Reads train.py logs (old-suite and bench tables), per-dataset/per-task CSVs, dump dirs, baseline names and `sn`. Bench mode is detected from the files, or set with `--bench`/`TSAR_BENCH=1`. Baseline names then resolve to `baselines/bench/dumps/<name>/` or `baselines/bench/per_task/<name>_<split>.csv`. Mixing bench and old-suite runs is refused. |
| `tools/test_compare_bench.py` | Unit tests for the above, CSV only (about 5 s): bench log parsing, name resolution, CI_fam equal to `bench.compare` (and the audit's power-table numbers), Holm, the gate, and the run.py bench dump. |
| `tools/dump_baseline.py` | Re-scores a baseline and keeps its forecasts (`--bench` writes to `baselines/bench/dumps/<name>/`). Heavy: run it only when the machine is idle. |
| `tools/overlap_scan.py` | Lag-searching growth-correlation scan of TRAIN against VAL/TEST (the method behind `TRAIN_DROP`), `selftest` controls, and `stamp-extra`, the gate for `TSAR_BENCH_EXTRA`. Heavy except `selftest`. |

```sh
# noise floor / multi-seed result
tools/seeds.sh base 3 TSAR_MINUTES=4
uv run tools/compare.py seeds "runs/dumps/base_s*"

# is a change real?
tools/seeds.sh latent 3 TSAR_MINUTES=4 TSAR_LATENT=0.3
uv run tools/compare.py pair "runs/dumps/latent_s*" "runs/dumps/base_s*"

# may we claim we beat Timestral-28M?
uv run tools/compare.py claim "runs/dumps/base_s*" timestral_28m

# leaderboard of all baselines + our runs vs seasonal naive
uv run tools/compare.py board "runs/dumps/base_s*" --ref sn
```

Notes:
- **Log precision.** Logs print per-dataset `wql_rel` to 3 decimals, so log-only comparisons are
  quantized at about 0.1–0.3% per dataset. Prefer dump dirs.
- **Pairing seeds.** Seeds are paired by position: sorted glob order on each side. Use matching
  seed lists on both sides.
- **Series bootstrap.** It recomputes each dataset's WQL ratio from resampled series sums. It is
  checked to match `prepare.evaluate` to about 3e-7 relative.
