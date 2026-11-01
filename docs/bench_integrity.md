# Benchmark integrity checks (tools/bench.py)

Date: 2026-10-29. Scope: the four open items on the leak-audited protocol (`docs/protocol.md`).
Nothing was scored on TEST. `train.py`, `prepare.py` and `baselines.py` were not changed.
The checks used little CPU (`nice -n 19`). No foundation model was run.

## 1. Duplicate and near-duplicate check, TRAIN vs VAL/TEST

### Method

The `bench.py overlap` command was rewritten. Run it with
`uv run tools/bench.py overlap --split val|test [--L 32]`.

**Train bank.** Every series of all 17 TRAIN datasets is hashed, with no subsampling: 165,718
series. Runs of 32 points are hashed at stride 32, so any shared stretch of 63 or more points is
found. The probe side hashes at stride 1. There are two matchers:

- **exact**: the values rounded to 1e-3;
- **znorm**: each run is z-normalised and rounded to 0.01, so it catches copies that were
  rescaled or shifted.

Flat runs, step runs and ramps are skipped: a run needs at least 8 distinct values and at least 8
distinct first differences.

**Probe.** The probe is the data the windows actually expose for each series. It is rebuilt from
the exact context arrays the models receive (`_series_spans`). This was checked against raw
slices, including on tasks whose contexts are cropped.

- VAL: the union of all contexts plus the targets.
- TEST: the union of contexts only. The final window's target is never hashed, and no forecast or
  score is made. The `_build_task` call does hold the target arrays in memory, but they are not
  used.

Positive controls passed. A copied run gave exact hits. A copy transformed as 7.3·y+100 gave
znorm hits only. A random walk gave 0 hits.

**Lag-searching scan (scratch script, macro tasks only).** Some series are re-releases with a
different vintage or deflator, so no run matches exactly. These were found by searching every
alignment with at least 60 overlapping points for the highest Pearson correlation of first
differences, with a threshold of 0.95. Pairs scanned (TEST contexts only):

- `us_consumption_1Q` against m4_quarterly, m3_quarterly and tourism_quarterly;
- `gvar` against m4_quarterly and m3_quarterly;
- `us_consumption_1M` against m1/m3/tourism monthly, hospital, car_parts and cif.

### Results

| Check | VAL (30 tasks) | TEST (48 tasks, target-free) |
|---|---|---|
| exact 32-pt runs | 0 series | 0 series |
| znorm 32-pt runs | 0 series | 2 tasks, 1 run each |
| lag-searching scan | not run (no macro families) | 17 TEST/TRAIN pairs |

The two TEST znorm hits:

- **boomlet_772 vs dominick[58456], dominick[81680].** This is a false positive. The matching run
  is one step change (about 96 down to 0 in dominick, about 2.9 down to 0.07 in boomlet) with
  noise below the rounding. The match does not extend past the 32 points.
- **us_consumption_1Q vs m4_quarterly[13328].** This is a real near-duplicate. The scale ratio is
  about 300 (296 to 318) over 230 aligned quarters. The log-growth correlation is 0.98 at the same
  unique lag at which the hash matched.

The lag scan then found 17 TEST/TRAIN pairs, involving 18 TRAIN series in total. The PCE
aggregates and the clothing, gasoline and motor-vehicle series in `us_consumption` (BEA) appear
in M4 Quarterly and M3 Monthly under anonymous IDs. The GVAR long-run rates (Switzerland, South
Africa) match M3 Quarterly with r = 1.000 and 0.9999, and the short rates (UK, Brazil) match
M3 Quarterly and M4 Quarterly. This is the FRED/BEA risk that `docs/protocol.md` already
documents, arriving through M3/M4 rather than through `monash_fred_md`.

### Fix (implemented)

`bench.TRAIN_DROP` lists the 18 series:

- m4_quarterly: 13 series;
- monash_m3_monthly: 2 series;
- monash_m3_quarterly: 3 series.

`load_train(drop_overlap=True)` is the default, and `train.py` (TSAR_BENCH=1) picks it up with no
change. The drop is applied after the fixed-seed subsample, so every other TRAIN series is
identical to before.

- Full corpus: 165,718 → 165,700 series.
- Default (subsampled) corpus: 15,530 → 15,524 series.

TRAIN series now carry a `"key"` field, which is their position within the dataset.

### Limits

- The lag scan covered only the macro families listed above. Other TEST/VAL families are covered
  by the 32-point exact and znorm hashes only, so a re-release with other rounding or a deflator
  could escape. Domain review makes this unlikely, since the remaining families do not share a
  source.
- The scan did not cover TRAIN_EXTRA, which is not downloaded. Re-run `overlap` and the lag scan
  if `--extra` or GiftEvalPretrain is added.
- Every VAL series is at least 32 points long. Some yearly TEST series (`world_*`,
  `us_consumption_1Y`) are shorter than 60 points and could not be scanned.

## 2. AutoETS / AutoTheta: long seasons and short series

`baselines.py` feeds statsforecast `P.fill_nans(c)[-512:]` with `season_length = m`. This was
checked one process at a time on four contexts per task, using the exact same input preparation
(statsforecast 2.0.1):

| Input | m | AutoETS picks | AutoTheta | NaN/inf |
|---|---|---|---|---|
| synthetic sine, n=512 | 24, 48, 96 | seasonal (A,N,A) | decomposed | 0 |
| jena_weather_10T (VAL) | 144 | **(A,Ad,N), non-seasonal** | not decomposed | 0 |
| synthetic sine, n=512 | 144 | **(A,Ad,N)** | **not decomposed** | 0 |
| bizitobs_l2c_5T (VAL) | 288 | **(A,Ad,N) / (A,N,N)** | not decomposed (512 < 2m) | 0 |
| synthetic sine, n=512 | 288, 1440 | **non-seasonal** (A,A,N) / (M,A,N) | **not decomposed** | 0 |
| favorita_stores_1M (VAL), ctx 30 | 12 | (A,N,N), flat | OTM, not decomposed | 0 |
| synthetic, n=18 (< 2 seasons) | 12 | (A,N,N), flat | OTM, not decomposed | 0 |

Findings:

- **No failures and no NaNs.** The models do not crash and return finite values.
- **For m ≥ 144 both models silently fall back to non-seasonal models.** At m = 288 or 1440 the
  512-point context is shorter than 2 seasons, which Theta needs before it will decompose. At
  m = 144 both models still choose non-seasonal fits, even on a clean sine.
- **Affected tasks.** VAL: jena_weather_10T and bizitobs_l2c_5T. TEST: boomlet m = 288/1440 and
  redset_5T. On these tasks AutoETS and AutoTheta are effectively trend models and lose to
  seasonal naive by construction. Every model gets the same context, so the comparison is fair,
  but these models are weak references on those tasks. Do not choose "the best statistical
  baseline" on those tasks.
- **Runtime.** AutoETS took 10–28 s per series at m = 144/288 on this loaded machine. At 1000
  forecasts per boomlet task, `auto_ets` on TEST will take hours. Budget for it.
- **Short series** (fewer than 2 seasons) fall back to non-seasonal level models (flat forecasts),
  with no error.
- **Process spawning.** statsforecast uses `n_jobs=-1` with spawned processes. A script without an
  `if __name__ == "__main__":` guard crashes with `BrokenProcessPool`. `bench.py`'s CLI has the
  guard.
- **NaN forecasts were silently zeroed.** `bench.evaluate` turns NaN/inf in any model's forecast
  into 0. It now counts them per task: a `n_nonfinite` column in `per_dataset`, plus a printed
  warning.

## 3. Evaluation correctness

- **Identical inputs.** Two models recorded a SHA-1 hash of every (contexts, h, m, freq) call over
  all 30 VAL tasks, and the hashes were identical. Targets never reach the model.
- **Mutation safety.** A test model that zeroed its input arrays in place previously would have
  corrupted the contexts used for the MASE scale, for seasonal naive, and for the next model in
  `run_baselines`, which shares the task objects. `evaluate` now passes copies. After running
  the hostile model, all contexts were unchanged.
- **Seasonal naive = 1.0 exactly.** On all 30 tasks (16,885 forecasts) both `wql_rel` and
  `mase_rel` are min = max = 1.0, and the headline and family geometric means are also 1.0.
  The queue's `seasonal_naive_pi` run (statsforecast) gives mase_rel = 1.0000, which is
  consistent: the same median.
- **Context checks.** All contexts are at most 2048 points, none starts with NaN, and no target
  window is entirely NaN.
- **Task definitions match fev-bench.** All 78 task specs (30 VAL + 48 TEST) were compared with
  `benchmarks/fev_bench/tasks.yaml` at fev commit `f323f6c`: horizon, num_windows,
  initial_cutoff (fev default −H·W), step (default H), seasonality, min_context_length, and the
  target columns, including the count of `target_0..N-1`. There were **0 mismatches**.

## 4. TEST access

TEST data was accessed only for content checks. No forecast was made and no score was computed.
Each access is logged in `data/bench/test_access.log` (6 entries on 2026-10-29: `overlap(test,
target_free=True)`, 2× `overlap-investigate`, 3× growth-correlation scan). All of them read
contexts only.
