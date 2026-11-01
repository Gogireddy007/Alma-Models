# Gap analysis: syn5_only vs Chronos-2, Bolt, TiRex (VAL, 30 tasks, read-only)

Author: Dr. Lena Park. All numbers are VAL, val-selected, single seed (s0). Inputs: `runs/syn5_only.log`,
`runs/sc_d96_l2_10k.log`, `runs/b_curr.log`, `baselines/bench/per_task/*_val.csv`. Series statistics were
computed in-process from VAL arrays only (`bench.load_val`, 300 windows per task); no test or holdout touched.
"gap" = mean over tasks of log(our wql_rel / baseline wql_rel). "share" = a group's summed log ratio / total.

Headline wql_rel (geometric mean): ours 0.685, Chronos-2 0.559, Bolt 0.582, TiRex 0.582, Timestral 5M/28M/209M
0.585/0.572/0.564, AutoTheta 0.959. Gap vs Chronos-2 is +0.205 log (x1.227), vs Bolt +0.164 (x1.178),
vs TiRex +0.164 (x1.178). MASE gap vs Chronos-2 is +0.159 log. We beat Chronos-2 on only 4 of 30 tasks.
Seed noise: b_curr s0/s1/s2 = 0.6932/0.6933/0.6994, so about 0.005 on the headline; syn5_only has one seed.

## 1. Gap by task and family

Top tasks vs Chronos-2 (our wql_rel / theirs, log gap, share of total):

| task | ours | C2 | log gap | share |
|---|---|---|---|---|
| bizitobs_l2c_1H | 0.660 | 0.317 | 0.734 | 12.0% |
| M_DENSE_1H | 0.938 | 0.456 | 0.721 | 11.7% |
| LOOP_SEATTLE_1H | 0.880 | 0.469 | 0.628 | 10.2% |
| M_DENSE_1D | 0.824 | 0.530 | 0.440 | 7.2% |
| favorita_stores_1D | 0.688 | 0.449 | 0.428 | 7.0% |
| uk_covid_nation_1D/cumulative | 0.340 | 0.227 | 0.406 | 6.6% |
| jena_weather_10T | 0.871 | 0.588 | 0.394 | 6.4% |
| solar_1W | 0.674 | 0.473 | 0.354 | 5.8% |
| ETT_15T | 0.831 | 0.591 | 0.342 | 5.6% |
| jena_weather_1H | 0.770 | 0.551 | 0.334 | 5.4% |

The top 3 carry 34%, the top 8 carry 67%, the top 7 hourly/sub-hourly or M_DENSE tasks (the three above, M_DENSE_1D,
jena_10T, ETT_15T, jena_1H) carry 59%. Closing only those 7 would leave us at x1.088 of Chronos-2.

By family vs Chronos-2 (share): M_DENSE 18.9%, bizitobs_l2c 14.3%, LOOP_SEATTLE 13.9%, jena_weather 13.2%,
ETT 11.0%, favorita 7.3%, solar 6.9%, hierarchical_sales 4.7%, SZ_TAXI 3.9%, uk_covid_nation 2.7%,
restaurant 2.2%, utla 0.8%, kdd 0.1%. Five families (M_DENSE, bizitobs, LOOP, jena, ETT) carry 71% of the gap.

Against Bolt and TiRex the same hourly tasks lead, but the tail differs:

| | vs Bolt | vs TiRex |
|---|---|---|
| gap (log) | 0.164 | 0.164 |
| top tasks (share) | M_DENSE_1H 14.7%, bizitobs_1H 13.9%, LOOP_1H 9.1%, utla_1D/new 7.9% | M_DENSE_1H 14.6%, bizitobs_1H 12.8%, LOOP_1H 12.2%, jena_10T 7.5% |
| top-8 share | 74% | 76% |
| tasks where we win | 5 (favorita family is -0.034, kdd -0.009) | 4 (uk_covid_nation -0.091, utla -0.009) |
| top families | M_DENSE 22%, jena 16%, bizitobs 14%, LOOP 13%, ETT 12% | M_DENSE 22%, jena 17%, LOOP 16%, bizitobs 15%, ETT 13% |

Bolt and TiRex are weaker than Chronos-2 on favorita_1D and on the covid cumulative tasks, so those tasks are
not a stable "gap"; the three hourly tasks and M_DENSE are.

## 2. Gap by task characteristic (vs Chronos-2)

| slice | n | mean log gap | ratio | share |
|---|---|---|---|---|
| freq H / sub-hourly (H,15/10/5min) | 10 | 0.379 | 1.46 | 61% |
| freq D | 12 | 0.226 | 1.25 | 44% |
| freq W / M | 8 | -0.040 | 0.96 | -5% (we beat C2) |
| horizon <=14 | 9 | -0.035 | 0.97 | -5% |
| horizon 28 | 13 | 0.291 | 1.34 | 62% |
| horizon 96-144 | 2 | 0.244 | 1.28 | 8% |
| horizon 168-288 | 6 | 0.364 | 1.44 | 36% |
| median context available <100 | 6 | -0.093 | 0.91 | -9% |
| context 100-300 | 7 | 0.108 | 1.11 | 12% |
| context 300-1000 | 6 | 0.278 | 1.32 | 27% |
| context >1000 (capped at 2048) | 11 | 0.388 | 1.47 | 70% |
| zeros share >10% (intermittent) | 8 | 0.281 | 1.32 | 37% |
| zeros share <1% | 16 | 0.197 | 1.22 | 51% |
| trend R2 (linear) >0.3 | 5 | -0.035 | 0.97 | -3% |
| trend R2 <0.1 | 10 | 0.352 | 1.42 | 57% |
| multi-seasonal (ACF of detrended series at 7x season >0.2) | 5 | 0.526 | 1.69 | 43% |
| no multi-seasonality | 25 | 0.140 | 1.15 | 57% |
| seasonal (season >1) / non-seasonal | 19 / 11 | 0.304 / 0.033 | 1.36 / 1.03 | 94% / 6% |

Spearman of per-task gap with: median context length +0.70, H/context ratio -0.57, trend R2 -0.40, seasonal ACF +0.43,
zeros share +0.15. The one clear driver is how much history exists: the gap grows monotonically with context
length and flips sign for short series. Two caveats. Context, frequency, horizon and multi-seasonality are
strongly collinear (long contexts are hourly or sub-hourly with H of 96-288), so these rows are one signal seen
from several sides. Our inference CONTEXT is 256 (32 patches), MAX_H is 64 (`train_v2.py:38-39`), while baselines see
up to 2048 points (`bench.py:65`): on a hourly task with weekly structure we see 1.5 weeks and roll 21 patches.
Intermittency is a secondary effect: the 8 zero-heavy tasks (favorita, hierarchical_1D, bizitobs) are x1.32,
but bizitobs and favorita_1D are also long-context tasks. Short, trending, non-seasonal tasks are already at or
below Chronos-2: W/M tasks are 4% better, and the covid cumulative 1W tasks are well ahead of Chronos-2 (0.246 vs 0.322).

## 3. Calibration (cov80 minus 0.80, negative = under-coverage)

Mean |error|: ours 0.104, Chronos-2 0.068, Bolt 0.058, TiRex 0.048, Timestral-28M 0.036, AutoETS 0.093, AutoTheta 0.130.
Signed mean: ours -0.094 (we are almost entirely under-coverage), C2 -0.059, Bolt -0.030, TiRex -0.027.
Tasks below -0.05: ours 14, C2 12, Bolt 6, TiRex 5, Timestral-28M 3. Only 4 of our tasks over-cover, max +0.09.

| task | ours | C2 | Bolt | TiRex | TS-28M |
|---|---|---|---|---|---|
| bizitobs_l2c_5T | -0.64 | -0.21 | -0.33 | -0.23 | -0.12 |
| jena_weather_10T | -0.31 | -0.03 | -0.04 | -0.05 | +0.02 |
| solar_1W | -0.26 | +0.08 | +0.10 | 0.00 | -0.02 |
| uk_covid_utla_1D/new | -0.23 | -0.19 | -0.10 | -0.15 | -0.11 |
| bizitobs_l2c_1H | -0.20 | -0.07 | -0.04 | -0.10 | +0.02 |
| ETT_1H | -0.19 | -0.03 | -0.10 | -0.01 | 0.00 |

By group, mean |error| ours / C2 / TiRex: H and sub-hourly 0.175 / 0.063 / 0.055; D 0.063 / 0.056 / 0.035;
W and M 0.077 / 0.091 / 0.057. So the calibration deficit is a long-rollout, hourly/sub-hourly problem; on D and W/M
we are within 0.01 of Chronos-2 (and better on W/M). Per-task calibration error is nearly uncorrelated with the
accuracy gap (r = -0.08), so fixing coverage would not by itself close wql_rel.

## 4. syn5_only vs sc_d96_l2_10k vs b_curr

Configs (prov.json): b_curr = SYN3, 2500 steps (0.693); syn5_only = SYN3+SYN5, 2500 steps (0.685);
sc_d96_l2_10k = SYN3 (no SYN5), 10000 steps, d96 / 2 layers, 0.369M params (0.685). All three are the same 0.369M
network. Gap vs Chronos-2 (log): b_curr 0.216, syn5_only 0.205, sc_10k 0.204. Mean change from the prior:
-0.011 log; from 4x training steps: 0.000 (vs b_curr: -0.012). The headline is flat to within seed noise (+-0.005).
The effect is mixed per task, not uniformly small.

| freq group (n) | gap b_curr | gap syn5 | gap 10k | prior effect | long-train effect |
|---|---|---|---|---|---|
| H / sub-hourly (10) | 0.379 | 0.374 | 0.369 | -0.005 | -0.005 |
| D (12) | 0.200 | 0.226 | 0.179 | +0.027 | -0.048 |
| W / M (8) | 0.036 | -0.040 | 0.036 | -0.077 | +0.077 |

What each change fixes (log change in wql_rel, negative = better):

| change | helps | hurts |
|---|---|---|
| SYN5 prior (vs b_curr) | uk_covid_nation_1W/new -0.19, 1W/cum -0.14, solar_1W -0.14, utla_1W/cum -0.10, solar_1D -0.08, ETT_15T -0.07 | uk_covid_nation_1D/cum +0.27 (0.259 to 0.340), jena_1H +0.11, utla_1D/new +0.05, nation_1D/new +0.05, hier_1D +0.04 |
| 4x steps (vs syn5_only, but no SYN5) | covid_nation_1D/cum -0.28, jena_1H -0.16, M_DENSE_1H -0.08, hier_1D -0.08, bizitobs_1H -0.06 | solar_1W +0.22, nation_1W/cum +0.13, utla_1W/cum +0.12, ETT_15T +0.10, nation_1W/new +0.08 |

Reading: the prior helps short weekly/monthly tasks (which we already beat Chronos-2 on) and hurts daily
trend/cumulative tasks and jena_1H; longer training recovers exactly those daily losses and gives back the weekly
gains. Neither moves the hourly and sub-hourly tasks that carry the gap (M_DENSE_1H 0.917/0.938/0.866,
LOOP_1H 0.920/0.880/0.881, ETT_15T 0.889/0.831/0.914 for b_curr/syn5/10k). More steps and a richer prior do not
reduce the gap, so it is a structural limit (context, capacity, rollout), not an undertrained one. Seed noise on
single weekly tasks with 1-5 windows is large, so per-task weekly deltas should not be trusted individually.
Past evidence on context: `b_base512` (0.740) lost to `b_base256` (0.726), but `docs/arch_plan.md:189` says
exp14 lost because steps were halved, so context at fixed steps is untested.

## 5. Ranked hypotheses (each is a 2500-step VAL screen, 1 seed first, then 3 seeds if it moves)

Baseline for all screens: syn5_only recipe (TSAR_SYN3=1 TSAR_SYN5=1 TSAR_SYN_CURR=0.9 TSAR_SYN_FRAC=0.3
TSAR_AUG=0.5 TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SPLIT=val), seed-matched, 0.685 (s0) or 0.693 mean-ish.

1. **Inference and training context is too short (256 vs 2048).** Targets the 11 tasks with >1000 points of
   history: 70% of the gap vs C2 (the 5 multi-seasonal tasks alone hold 43%, where lag 168 is barely
   inside a 256 window). Screen: `TSAR_CONTEXT=1024 TSAR_BATCH=32` against control `TSAR_CONTEXT=256 TSAR_BATCH=32`
   (equal samples and steps, so context is the only change). Falsified if the long-context-task subset log gap does not
   shrink by at least 0.05 vs the control, or the headline stays above 0.68. Check the position encoding and the
   NORM statistics handle a longer window first; the 512 loss may be a step-count artifact, not a context one.
2. **Rollout length and compounding error on H >= 96.** Targets 6 tasks with H 168-288 (36% of the gap, x1.44) and
   the 14 under-covered tasks. Screen: `TSAR_LONG_H=176 TSAR_MS_COARSE=2` (one-shot fine plus coarse lead) vs
   default MAX_H=64 rollout. Falsified if the 6 long-H tasks do not improve by 0.05 log or the short-H tasks
   (<=14, currently better than C2) regress by more than 0.02. Overlaps with 1, so run 1 and 2 as a 2x2 grid.
3. **Capacity (0.369M parameters vs 5M-205M baselines).** 4x steps changed nothing, which points at width or
   depth rather than steps. Targets the mean gap (up to 100%, but least specific). Screen:
   `TSAR_D_MODEL=192 TSAR_N_LAYERS=4` at 2500 steps, defaults otherwise. Falsified if the headline is above 0.67
   (capacity should give at least 0.015 given the seed noise) or if the hourly tasks do not move more than the
   others. Confounded with wall time, so also report per-step loss.
4. **The prior trades daily/cumulative accuracy for weekly accuracy.** Targets about 13% of the gap (covid
   cumulative, jena_1H, utla_1D/new and nation_1D/new regressions from SYN5: +0.27/+0.11/+0.05/+0.05) and would
   take the best of both runs. Screen: `TSAR_SYN5=1 TSAR_SYN_FRAC=0.15` (and 0.5 for the curve) vs SYN5 at 0.3.
   Falsified if covid_nation_1D/cumulative stays above 0.30 while solar_1W and covid 1W stay at syn5 levels.
   Logs `syn5_curr_25` and `syn5_curr_50` may already hold a version of this and should be read before running it.
5. **Quantile spread is too narrow at long leads (calibration).** Targets cov80 (0.104 to under 0.07) and
   a small share of wql (under 10%; per-task coverage error and gap are uncorrelated). Screen: `TSAR_HDEC=1`
   (lead-widening) and separately `TSAR_SELFCAL=0.5`, both on the baseline recipe. Falsified if cov80_error on the 10
   hourly/sub-hourly tasks stays above 0.12 or wql_rel on the short-H tasks worsens by over 0.01.

Not recommended as a priority: more training steps (0.000 gain at 4x), and intermittency-specific modelling
(the zero-heavy slice is mostly long-context tasks, so it is not independent evidence).
