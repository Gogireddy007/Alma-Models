# Audit r1 (Auditor, 2026-10-31)

Scope: TTA (0f32857), AlmaTS FRAME=origin (`wm_tswm_origin_s0`), `TSAR_WM_LONG=paths` (ee84ed6), the headline
numbers, and corpus leakage. Read-only on `train_wm.py`, `wm/` and `tools/`. Line numbers refer to the committed
`train_wm.py` at 0f32857 (sha256 2fac1ad7…, the file every TTA/tswm run below used), `wm/tswm.py` at HEAD
(unmodified in the working tree) and `tools/bench.py` in the working tree (sha256 1bb0b510…, the file those runs used).
TEST was not touched. The only model code I ran is `tools/audit_r1_probe.py` (random weights, CPU, through
`tools/smoke.py`). `tools/audit_r1_numbers.py` reads text files only.

## Verdicts

| # | Item | Verdict |
|---|---|---|
| 1 | TTA (`TSAR_WM_TTA`) causality | **PASS** |
| 2 | AlmaTS FRAME=origin: causality, harness check, comparison | **CAVEAT** (causal: PASS; claim/comparison: caveats) |
| 3 | `TSAR_WM_LONG=paths` feeds back only own predictions | **PASS** |
| 4 | Headline numbers re-derived | **PASS** with a **CAVEAT** on how TTA is quoted and on a stale table |
| 5 | Corpus leakage (GIFT_CLEAN, GEP off) | **PASS** with a **CAVEAT** (VAL scan is hash-only) |

No veto on anything audited here. Two conditions on promotion are in items 2 and 4.

## 1. TTA: PASS

What the forecast can read, end to end:
- The bench builds each window as `ctx = y[max(0, c - 2048):c]` and `target = y[c:c + H]`
  (`tools/bench.py:770`, `:775`). The context ends strictly before the origin `c`, for every task and window.
  `MAX_CONTEXT = 2048` (`:65`). The VAL task definitions, `_build_task` and `evaluate` have not changed since the
  baselines were scored (`git diff f983682 HEAD -- tools/bench.py` touches only TRAIN/GEP code).
- `evaluate` passes copies of the contexts and nothing else (`tools/bench.py:828`). The targets are used only after
  `predict_fn` returns. `predict_fn` never receives a target or an index into the raw series.
- `predict` (`train_wm.py:763-774`). `sym` calls `predict_one` on `-ctx` (`:769`), the same code path on the same
  context, negated. It then reflects and averages (`:770`). `clip` (`:771-773`) computes `nonnegative_context`
  (`:752-759`) per row from the same `ctxs` list. So "the whole observed context up to 2048 points" is exactly
  `y[c-2048:c]`, which is all before the origin. The model itself reads only the last `CONTEXT` = 256 points
  (`:800`). The clip reading more history than the model is legitimate: it is context, and the Timestral baselines
  received the same 2048 points.
- Cross-window contamination. Window k+1's context contains window k's target (rolling origins), and
  `predict_chunk` batches up to 512 windows together (`:792-803`). So a family that mixed rows in a batch could
  leak. `nonnegative_context` and the TTA arithmetic are row-wise. A grep finds no batch-dimension reductions
  (BatchNorm, `.mean(0)` and similar) in `wm/ssm.py`, `wm/gru.py`, `wm/tswm.py` or `train_wm.py`. The probe
  (below) checks row independence numerically for ssm, gru and tswm.
- The re-scored checkpoints are the trained ones: the sha256 in each `runs/wm_*_tta.log` matches `ckpt_sha256` in
  the training run's `.prov.json` for ssm s0-s2 and gru s0-s2. The model is frozen and eval-only, and no VAL
  statistic feeds back (`TSAR_EVAL_MIN` is unset, so `train_wm.py:887` never scores during training).

Not leakage, but noted: the commit message reports cov80_err 0.132 → 0.113 on the 5-task smoke. On full VAL, TTA
leaves cov80_err essentially unchanged (s0 0.1250 → 0.1249; seeds 0-2 mean 0.1338 → 0.1304).

## 2. AlmaTS FRAME=origin: CAVEAT

**Causality: PASS.**
- `_to_origin_frame` returns `q` unchanged when `frame == "origin"` (`wm/tswm.py:447-450`). The lead quantiles stay
  in the harness model space of origin (b, j). That space is `asinh((x - loc_j) / scale_j)`, with `loc_j`/`scale_j`
  from `causal_stats` (`train_wm.py:276`): cumulative sums read at patch ends, so they use points ≤ the end of
  patch j only. De-normalisation at eval uses `ctx.loc[:, -1]`/`ctx.scale[:, -1]` of the last context patch
  (`:812`).
- The encoder mask is causal over valid keys (`wm/tswm.py:325`). Lead tokens read encoder keys 0..j only, on both
  decode paths (dense `:396`, gather `:403`).
- At eval the input array holds only the 256 context points, so nothing past the origin exists to be read.

**Does the harness check exercise it? Partly.**
- `causality_check(model)` (`train_wm.py:385-413`, called at `:440`) runs on the very model built with
  `TSAR_WM_TSWM_FRAME=origin`, so it covers the frame. `runs/wm_tswm_origin_s0.log` shows
  `wm_causal: ok (max change 0.0e+00 at 40 early origins)`.
- It has two gaps:
  - It perturbs both rows identically, so it cannot detect mixing across rows.
  - With 2 rows × 20 origins it takes the **dense** decode path (`wm/tswm.py:390`). Eval uses the **gather** path.
- `tools/audit_r1_probe.py` closes both gaps: see "Probe results" below.
- Recommendation for the harness owner (not a blocker): add a row-swap test and a `TSAR_WM_TSWM_PATH=gather` pass to
  `causality_check`.

**Comparison with `wm_ssm_none_s0` (0.6707): like-for-like in setup, but not in what is claimed.**

Same:
- **Data.** Both checkpoints store the same training-corpus hash: `train_sha256` bc78d74b5a3f, 10,758 series, 14
  datasets, `train_drop_sha256` 372a45bd827a, `bench_extra` 0. So do ssm s1, gru s0 and tswm none.
- **Recipe.** Identical `.prov.json` env apart from `TSAR_WM`/`TSAR_WM_TSWM_FRAME`: STEPS 2500, SEED 0, AUG 0.5,
  SYN3/SYN5 1, SYN_CURR 0.9, SYN_FRAC 0.3, GIFT_CLEAN 1, no GEP, no EXTRA.
- **Size.** 368,361 vs 365,937 params; 12.2M vs 13.8M MACs.
- **Eval.** The same 30 VAL tasks with the same `n_forecasts` per task as every baseline (re-checked from the logs).
  Raw median long-horizon mode, no TTA or paths on either.

Different:
- **Code version.** ssm s0 ran with train_wm 12c51ddd (9f3cb6e) and bench 163af08f (22ea1fe). tswm origin ran with
  train_wm 2fac1ad7 (0f32857) and the *uncommitted* bench 1bb0b510. The only bench difference is `GEP_DROP`
  (`tools/bench.py:513`), which is read only inside `load_gep`, behind `gep_on()`. That makes it inert here, and the
  identical `train_sha256` confirms it. Reproducibility caveat: the run used a dirty tree (`git_dirty: true`), so
  `GEP_DROP` should be committed.
- **Seeds.** One seed (tswm) against one seed (ssm s0). ssm seed sd = 0.0084 (5 seeds). AlmaTS origin 0.6806 is
  therefore **indistinguishable** from the ssm 5-seed mean 0.6804, and it is 0.0099 *worse* than ssm s0 on the same
  seed.
- **The "0.6806 vs 0.948" contrast.** That number is FRAME=origin vs FRAME=loc (`wm_tswm_none_s0`, 0.9478). It shows
  that the loc frame was broken. It is not evidence that the AlmaTS world model works.
- **The world-model check is still missing.** The matched control `wm_tswm_origin_direct_s0` is still training (step
  700/2500 at audit time). The FRAME=loc direct (0.8244) is not a valid control for FRAME=origin none.

**Conditions before AlmaTS origin is promoted or shown as a WM on the scoreboard:**
- (a) `wm_tswm_origin_direct_s0` finishes, and none beats it by more than seed noise.
- (b) Seeds 1-2 are run before any claim relative to ssm (R2 rule: ≥ 0.008 on the seed mean).

## 3. `TSAR_WM_LONG=paths`: PASS

`predict_paths` (`train_wm.py:727-746`) does three things:
- Chunk 1 is `predict_chunk` on the bench contexts (`:732`).
- Every later chunk extends each path with `q[i, qi]` (`:737`). `q` is the model's own sorted output for chunk 1,
  and after that it is the `mixture_quantiles` of the model's own component forecasts (`:739-740`).
- `mixture_quantiles` (`:681-723`) is a deterministic row-wise function of those component quantiles and the fixed
  `PATH_W`.

Nothing else enters: no target and no raw series. The order `[c for pp in paths for c in pp]` → `reshape(S, n, …)`
keeps each component on its own series (path-major on both sides). The median mode (`:783-787`) feeds back
`q[i, NQ // 2]`, also the model's own output. With TTA, the `-x` pass runs the same code on the negated context.
The training-side additions in ee84ed6 (`TSAR_MAX_H`, `LEADW`, `ORIGINS`) affect the loss on TRAIN windows only.

## 4. Headline numbers: PASS (re-derived), CAVEAT on quoting

Method (`tools/audit_r1_numbers.py`, which imports nothing from the repo):
- Parse the last per-task table of each `runs/*.log`.
- Recompute the geometric mean of `wql_rel` and `mase_rel` and the mean |cov80 − 0.8|, and compare with the printed
  `val_*` lines.
- Re-derive the baselines from the full-precision `baselines/bench/per_task/*_val.csv`, with `wql_rel` recomputed as
  `wql / sn_wql`. Check that every CSV and every run log has the same 30 tasks and the same `n_forecasts` (and the
  same `sn_wql` for the baselines).

The derived values agree with the printed ones to within the 3-decimal rounding of the tables (≤ 1e-4).

| claim | re-derived | per-seed (printed wql_rel) | status |
|---|---|---|---|
| ssm none 5-seed mean 0.6804 | **0.6804** (sd 0.0084); mase 0.8376, cov 0.1317 | 0.6707, 0.6936, 0.6804, 0.6808, 0.6765 | confirmed |
| gru 5-seed 0.6958 | **0.6958** (sd 0.0031); mase 0.8504, cov 0.1180 | 0.6935, 0.7002, 0.6941, 0.6934, 0.6980 | confirmed |
| ssm +TTA (3 seeds) "~0.671" | **0.6712** (sd 0.0090) | 0.6636, 0.6811, 0.6688 | confirmed |
| gru +TTA (3 seeds) | 0.6922 | 0.6868, 0.6974, 0.6924 | |
| ssm +paths (3 seeds) | 0.6750; cov 0.1037 | 0.6649, 0.6859, 0.6743 | |
| AlmaTS origin s0 0.6806 | 0.6806 (derived 0.6805); mase 0.8398, cov 0.1310 | | confirmed |

Baselines, all confirmed against `baselines/bench/results.tsv` and `research_log.md` ("2026-10-29 bench val
baselines"); same tasks and `sn_wql` in every CSV:

| model | wql_rel | mase_rel | cov80_err |
|---|---|---|---|
| Chronos-2 | 0.5586 | 0.7195 | 0.0680 |
| Timestral-209M | 0.5639 | 0.7062 | 0.0342 |
| Timestral-28M | 0.5718 | 0.7124 | 0.0360 |
| Chronos-Bolt base | 0.5817 | 0.7391 | 0.0579 |
| TiRex | 0.5818 | 0.7267 | 0.0477 |
| Timestral-5M | 0.5846 | 0.7299 | 0.0466 |
| AutoTheta | 0.9594 | 0.8943 | 0.1296 |
| Seasonal naive | 0.9876 | 1.0000 | 0.1423 |
| AutoETS | 2.6842 | 0.9150 | 0.0925 |

Caveats:
- **TTA gain, quoted like-for-like.** `docs/goal.md` writes "0.680 / ~0.671". That puts a 5-seed raw mean next to a
  3-seed TTA mean. On the same seeds 0-2, raw is 0.6815, so TTA gives **−0.0103** (−0.0071 to −0.0125 per seed;
  gru −0.0037). That is a real, consistent eval-only gain, but quote it as 0.6815 → 0.6712 (seeds 0-2).
- **paths + TTA together has never been scored on full VAL.** The R1 recipe in `docs/autoresearch.md` ("paths + TTA
  inference") is not yet backed by a full-VAL number. The two gains (−0.0065 paths, −0.0103 TTA) should not be
  added. Score the combination on the incumbent before using it as the reference.
- **Stale table in `research_log.md`.** The "Bar to beat" table (`research_log.md:16-30`, for example Chronos-2
  0.621, Timestral-209M 0.569, AutoETS 0.781) does not match the current bench. It is from an earlier evaluation.
  It should be marked superseded so no-one compares against it. The "bench val baselines" block (`:5-11`) and
  `docs/goal.md` are correct.

## 5. Leakage: PASS with CAVEAT

GEP is off:
- `TSAR_BENCH_GEP` is absent from every audited `.prov.json`. jobq records the full merged env, inherited keys
  included (`tools/jobq.py:88`, `:269-270`), so its absence is real.
- `gep_on()` defaults to off (`tools/bench.py:525`).
- `load_gep` is called only behind `gep_on()` (`:703`).
- `TSAR_BENCH_EXTRA` is also unset, so `bench_extra` = 0 in every checkpoint config.

Corpus:
- With `TSAR_GIFT_CLEAN=1`, `load_train` drops the `GIFT_TEST_OVERLAP` datasets (`tools/bench.py:688`).
- The checkpoints confirm 14 TRAIN_LOCAL datasets: dominick, ercot, m5, monash_australian_electricity, cif_2016,
  m1_monthly, m3_monthly, m3_quarterly, nn5_weekly, tourism_monthly, tourism_quarterly, traffic, weather, nn5.
  m4_quarterly, hospital and car_parts are absent.
- The rest is synthetic. Training does no VAL-based selection (fixed steps, final checkpoint, `TSAR_EVAL_MIN` unset).

Overlap with VAL/TEST:
- `docs/bench_integrity.md` §1 scanned all 17 LOCAL datasets (a superset of these 14) against VAL contexts+targets
  and TEST contexts. Exact and z-normalised 32-point hashes found 0 VAL hits.
- TEST macro near-duplicates are removed by `TRAIN_DROP` (m3_monthly/m3_quarterly keys apply here). The drop hash is
  identical across runs.
- The TRAIN_EXTRA stamp (`data/bench/overlap_extra.json`) is irrelevant: EXTRA is not used.

Caveats:
- The VAL side was checked by hashes only. No lag/correlation scan was run against VAL (`bench_integrity.md`
  "Limits"), so a re-release with different rounding would escape.
- Domain review makes such an overlap unlikely: LOCAL sources (PeMS traffic, Walmart M5, Dominick's, ERCOT, BoM
  weather, NN5, tourism, M1/M3, CIF, AEMO) do not share a source with the VAL families (ETT, Seattle loop, M_DENSE,
  SZ taxi, bizitobs, Favorita, hierarchical sales, Jena, KDD-2022 wind, restaurant, NREL solar, UK COVID).
- VAL family selection exposure for the foundation-model baselines is disclosed separately (`VAL_EXPOSURE`). It is
  not training leakage for Alma.

## Probe results (`tools/audit_r1_probe.py`)

Run with `uv run tools/smoke.py --timeout 60 -- uv run python tools/audit_r1_probe.py`: exit 0, 3 s, peak RSS 222 MB.

Setup:
- Every family weight is randomised (N(0, 0.1)), so the zero-initialised transitions, drive and SSM `C` are live.
- 3 rows. Row 1 has left padding. Origins 0..19.
- `causal`: perturb every per-patch input (tok, ref, x, loc, scale) at patch ≥ 20.
- `rowind`: replace rows 1-2 entirely and check row 0's outputs.

Max |Δ| in each case:

| family | causal (dense) | rowind | causal (gather, = eval path) | dense vs gather |
|---|---|---|---|---|
| tswm none, FRAME=origin | 0 | 0 | 0 | 2.4e-07 |
| tswm direct, FRAME=origin | 0 | 0 | 0 | 1.2e-07 |
| tswm none, FRAME=loc | 0 | 0 | 0 | 3.6e-07 |
| ssm none | 0 | 0 | n/a | n/a |
| gru none | 0 | 0 | n/a | n/a |

Conclusions:
- All families are causal on both decode paths.
- No row of a batch can see another row. Rolling windows batched together therefore cannot leak one window's target
  into another's forecast.
- The dense path, which the harness check tests, and the gather path, which eval uses, agree to float rounding.

## Heads-up for the next audit (not audited here)

The working tree has uncommitted `train_wm.py` edits adding `TSAR_WM_RECAL`/`TSAR_WM_HOLD` (calibration fitted on
TRAIN tails) and a new `tools/wmrecal.py`. Any run using them needs its own audit before promotion. Points to check:
- The calibration windows come only from `CAL_SRC` (TRAIN).
- `HOLD` removes exactly those tails.
- The `freq` key used at apply time does not let VAL-task metadata select a map that was tuned on VAL.
