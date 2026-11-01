# Alma-Power world-model search (DE-LU day-ahead prices)

Owner: power world-model architect. Files: `tools/power/wm_*.py`, `tools/power/search/**`, `runs/power/search/**`
(force-added selectively), this document. Binding harness protocol: `docs/power_prereg.md` (lead).

## 0. Protocol (written 2026-10-31 20:20 EDT, BEFORE any DEV score of any model; committed before the first run)

**Data for the search.** `tools/power/search/fetch_smard.py` downloads hourly DE-LU day-ahead price and German
load / wind onshore / wind offshore / solar actuals from the SMARD.de chart-data API (Bundesnetzagentur, the source
Energy-Charts cites for its prices), truncating in memory at the split end BEFORE writing (DEV cache
`runs/power/search/smard/dev/`, nothing after 2025-06-30 23:00 local; asserted again at load). Reason: the
Energy-Charts API returned HTTP 429 while the lead was fetching; the lead's data (`data/power/hourly.csv`) is used
for the harness and for the final refit (below). Days 2018-10-01..2025-06-30 (DE-LU zone only), local delivery
days x 24 slots (`tools/power/wm_data.py`).

**Information set (asserted in `wm_data.batch`).** For target day D: prices of days <= D-1, fundamentals (load,
wind, solar, residual load = load - wind - solar) of days <= D-2, calendar of D (weekday, German public holidays,
bridge days, season) which is known in advance. Realised residual load of D-1 / D appears only as a training target
of the physical model's auxiliary loss, never as an input. In the harness wrapper the same feature builder
(`wm_data.batch_from_arrays`) reads only the harness context (covariate rows of D-1 are NaN there and are not used).

**Backtest (the search metric).** Rolling origin, one origin per target day 2024-01-01..2025-06-30 (all days with
24 prices), models trained once on target days 2018-10-16..2023-12-31 (asserted `train_end <= 2023-12-31` for any
scored run). Quantile grid = the harness grid (13 levels 0.025..0.975).

**Selection score (pre-declared, higher is better):**
`S = 0.35 * P1 + 0.35 * P2 + 0.30 * QSskill`, where (docs/power_prereg.md section 4 definitions)
P1 = daily direction accuracy (mean of hourly medians vs base(D-1), label base(D) > base(D-1), label ties excluded),
P2 = hourly direction accuracy (median(D,h) vs P(D-1,h), all slots), QSskill = 1 - QS / QS(naive_epf),
QS = mean pinball over levels 0.1..0.9 in EUR/MWh, naive_epf as defined in the prereg (naive_w on Mon/Sat/Sun,
else naive_d, empirical same-slot difference quantiles over the last 364 days).

**Loop.** autoresearch pattern: every run has a fixed `--steps` budget (default 1500 Adam steps, batch 64,
OneCycle lr 2e-3, wd 1e-4, grad clip 1, CPU 2 threads, nice 10), one scalar S, every run (crashes included)
appended to `runs/power/search/results.tsv`. Keep / discard: a configuration is kept as the new incumbent of its
architecture only if S beats the incumbent by > 0.003 (below that is seed noise); discarded configurations stay
in the log and count in K.

**Twins.** Every world model (`--abl none`) has a matched twin (`--abl direct`): identical module list and
parameter count, same data, steps, seeds, lr; every hour is ONE step of the same transition from the origin
state (no carried latent state, no closed-loop feedback), i.e. all 24 hours decoded at once from the state.

**Finalists and freeze.**
1. The 3 best world-model recipes by single-seed S are re-run with seeds 1 and 2; the decision uses the 3-seed
   mean S. The frozen candidate is the 3-seed ensemble (average of quantiles, re-sorted) of the best recipe; its
   twin is the 3-seed ensemble of the same recipe with `--abl direct`, identical budget.
2. "Beats the twin" on DEV = higher ensemble S (components and a 7-day block-bootstrap CI reported).
3. Frozen checkpoints = the chosen recipe REFIT with the same steps and seeds on the lead's panel
   (`tools/power/data.train_panel()`, Energy-Charts actuals, target days <= 2025-06-30), so the harness inputs and
   the training inputs come from the same source and the 2024-25 regime (more negative-price hours) is in
   training. No selection happens on the refit; its harness-DEV score is in-sample and reported as a sanity check
   only. The DEV evidence for the recipe is the backtest above.
4. `runs/power/FROZEN.json`: checkpoints + sha256, K, winner's-curse note; then VAL once through
   `tools/power/run_eval.py --split val` for the candidate and its twin. TEST is never touched by the architect.

## 1. Architectures (tools/power/wm_models.py)

Common observer: per-day token MLP([24 z-prices, 4x24 lagged fundamentals, calendar]) -> GRU over L context days ->
world state s at the end of D-1. z-frame per origin: mean / (std + 5 EUR) of the prices of D-7..D-1.

| arch | world model (abl=none) | twin (abl=direct) |
|---|---|---|
| cycle | daily-cycle SSM WM: GRUCell rolled hour by hour through day D on a phase clock (hour harmonics, weekday, holiday, bridge, season), same-hour prices of D-1 / D-7, same-hour fundamentals of D-2, closed-loop feedback of its own previous-hour median | each hour = one step of the same cell from s, no feedback |
| phys | physical-driver WM: latent system state rolled 48 h from the end of D-2 (last known fundamentals) through D-1 (assimilating the observed D-1 prices) into D; per hour it emits residual load rl_hat (aux loss vs realised residual load, training target only) and a 3-way regime (normal / scarcity / surplus); price median = learned monotone merit-order curve of rl_hat (positive-weight sigmoid sum; curve parameters from s) + regime offsets + quantile spreads | each of the 48 hours = one step from s |
| regime | regime-switching WM: cycle rollout whose quantiles are a mixture of 3 regime heads with a learned carried transition pi_t = softmax(T(h_t, pi_{t-1})) | uniform carried regime, one step from s |
| alma | Alma's pretrained SSM (data/ckpt/wm_ssm_none_s0.pt) fine-tuned on hourly prices; twin from wm_ssm_direct_s0.pt | (as in tools/commodity/wm_alma.py) |

Amendment to section 0 (20:35, after r1-r2, disclosed): `tools/power/search/q.sh` was rewritten (the xargs queue
died silently after 2 runs of r3; the 7 missing r3 runs were relaunched unchanged as `r3b.txt`). The harness-side
naive_epf puts its median exactly at the naive value; the search metric's naive_epf adds the median difference
(tiny QS difference; same reference for every row, so rankings are unaffected).

## 2. Experiment log (all rows: `runs/power/search/results.tsv`; logs `runs/power/search/logs/`)

Reference rows (search metric): naive_epf S 0.430 (P1 0.638, P2 0.591, QSskill 0), naive_w 0.401, naive_d 0.369.
Alma pretrained zero-shot: S 0.5246 (P1 0.689, P2 0.664, QSskill 0.170); direct twin 0.5185.

| round | what | finding |
|---|---|---|
| smoke | 30-step plumbing runs of cycle / phys / regime | (not selection rows) |
| r1 | 3 archs x {WM, twin}, d64 L14 lr 2e-3 1500 steps; Alma zero-shot both | all trained archs 0.619-0.624 (WM vs twin: cycle 0.6209/0.6162, phys 0.6244/0.6229, regime 0.6189/0.6214) |
| r2 | phys: d96, L7, L28, 3000 steps, dirw 0.3, aux 0, aux 1, lr 1e-3; cycle: dirw, L28; Alma fine-tuned 600 steps (WM, twin) | bigger / longer = worse (overfit: d96 0.589, 3000 steps 0.576); the direction loss hurts QS (cycle 0.551); lr 1e-3 best (0.6435); Alma FT 0.569 / twin 0.573 |
| r3 | phys lr 5e-4, 3e-4, lr1e-3 x {750 steps, d32, aux 1, L28}; cycle lr 1e-3 / 5e-4; regime lr 1e-3 | phys lr1e-3 aux1 0.6464 (best single row), lr1e-3 L28 0.6421, cycle lr5e-4 0.6408, regime 0.6366 |
| f | finalists A/B/C seeds 1,2 + twins seeds 0,1,2 | below |

One crash row (`r1_alma_zs_none`, import-name clash with tools/commodity/wm_data.py; fixed and re-run).

## 3. Finalist selection and freeze (written 21:05, before any VAL number was read by the architect)

Top-3 world-model recipes by single-seed S (rule 1): A = phys lr 1e-3 aux 1.0; B = phys lr 1e-3 (aux 0.3);
C = phys lr 1e-3 L 28 (all d 64, 1500 steps).

| recipe | WM S seeds 0/1/2 | WM mean | twin S seeds 0/1/2 | twin mean | 3-seed ensemble WM / twin | ensemble diff 90% CI (7-day block bootstrap) |
|---|---|---|---|---|---|---|
| A | 0.6464 / 0.6450 / 0.6346 | 0.6420 | 0.6352 / 0.6410 / 0.6360 | 0.6374 | 0.6449 / 0.6453 | [-0.0067, +0.0060] |
| **B** | 0.6435 / 0.6415 / 0.6421 | **0.6424** | 0.6416 / 0.6438 / 0.6391 | 0.6415 | **0.6451** / 0.6457 | [-0.0063, +0.0057] |
| C | 0.6421 / 0.6380 / 0.6396 | 0.6399 | 0.6341 / 0.6448 / 0.6430 | 0.6406 | - | - |

By the pre-declared rule the frozen recipe is **B** (best 3-seed mean, 0.6424 vs A 0.6420: a coin flip inside
noise). Ensemble B on the DEV backtest: S 0.6451 [90% CI 0.626, 0.664], P1 0.808, P2 0.753, QSskill vs naive_epf
0.329, MAE 19.3 EUR/MWh, 80% coverage 0.769. **The world model does NOT beat its twin on DEV** (twin ensemble S
0.6457, P1 0.808, P2 0.756, QSskill 0.327; difference CI straddles 0). The single seeds favoured the WM in 2 of 3
recipes by <= 0.005; seed ensembles erase it. Plainly: the hourly latent rollout and the one-step direct decode are
indistinguishable on this metric; the gain over naive / Alma comes from the shared observer, the calendar and the
fundamentals, not from rolling the state forward.

K = configurations tried: 49 training/eval runs (45 scored model rows incl. 9 finalist seeds and 9 twins, 3 smoke,
1 crash) + 3 reference rows, 33 distinct recipes. Winner's curse: the best single row (0.6464, A seed 0) did not
survive its seeds (A mean 0.6420); with ~30 recipes inside a +-0.005 seed band, the selected recipe's DEV S is
optimistic by roughly that band. The ensemble's 90% CI is +-0.019.

**Frozen (rule 3):** recipe B refit on the lead's panel (`--data lead`, target days 2018-10-16..2025-06-30, spring-
forward days dropped as targets), 1500 steps, seeds 0/1/2, WM and twin with identical budgets
(`tools/power/search/refit.txt`). Harness wrappers: `tools/power/wm_power.py` (`wm_power`, `wm_power_direct`; sha256
verified at load; deterministic; all 13 levels; uses only the harness arguments + calendar via `dates`).
Consistency check: the harness path reproduces the search path for the backtest ensemble to 0.74 EUR/MWh mean abs
(difference = SMARD vs Energy-Charts fundamentals, corr > 0.999). Information-set perturbation test
(`tools/power/search/test_leak.py`): forecasts unchanged when prices of D / D+1 or fundamentals of D-1 / D are
altered, changed when prices of D-1 or fundamentals of D-2 are altered, all 3 archs x {WM, twin}: ALL OK.

| name | members (runs/power/search/ckpt/) | sha256 |
|---|---|---|
| wm_power | frz_phys_none_s0.pt / s1 / s2 | be295a71... / 12bdf269... / 3b751c97... |
| wm_power_direct | frz_phys_direct_s0.pt / s1 / s2 | abc82742... / f96cf84f... / f0b86f8c... |

Full hashes in `tools/power/wm_power.py` and `runs/power/FROZEN.json`. DEV-diagnostic wrappers `wm_power_bt` /
`wm_power_bt_direct` (the backtest ensembles, trained to 2023-12-31) register only with `POWER_WM_DIAG=1`.

## 3b. Harness DEV diagnostic (backtest ensembles, out-of-sample; run during the VAL window, no selection)

`POWER_WM_DIAG=1 run_eval.py --split dev --models wm_power_bt,wm_power_bt_direct,naive_epf,lear,alma_s0,alma_direct_s0`
(365 target days 2024-07-01..2025-06-30; the bt ensembles were trained on targets <= 2023-12-31 and never
recalibrated, while LEAR recalibrates daily on the last 728 days):

| model | P1 | P2 | QS skill vs naive_epf | MAE |
|---|---|---|---|---|
| lear | 0.805 | 0.757 | 0.339 | 20.85 |
| wm_power_bt | 0.816 | 0.746 | 0.311 | 21.88 |
| wm_power_bt_direct | 0.811 | 0.751 | 0.312 | 21.82 |
| alma_s0 / alma_direct_s0 | 0.699 / 0.704 | 0.660 / 0.643 | 0.178 / 0.166 | 25.9 / 26.2 |
| naive_epf | 0.625 | 0.582 | 0 | 30.63 |

wm_power_bt vs lear: P1 +0.011 (p 0.53), P2 -0.011 (p 0.16), QS +0.36 EUR worse (p 0.12): level with LEAR on
direction, slightly behind on QS. vs twin: P1 +0.006, P2 -0.005, QS +0.007, all n.s. vs alma_s0: better on all
three (p < 0.001).

## 4. VALIDATION (run once, 21:00 EDT, after the freeze commit 7e21933)

`tools/power/run_eval.py --split val --models wm_power,wm_power_direct` (123 target days 2025-07-01..2025-10-31;
forecasts `runs/power/val/wm_power*.npz`, metrics in `runs/power/val/metrics.json`). The baseline / FM rows below
were produced by the lead's earlier VAL runs in the same file; the architect read them only after the freeze.

| model | P1 daily dir. | P2 hourly dir. | P3 spread dir. | P4 QS skill vs naive_epf | MAE EUR/MWh | cov80 |
|---|---|---|---|---|---|---|
| chronos2 | 0.780 | 0.759 | 0.780 | 0.420 | 16.63 | 0.765 |
| chronos2_cov | 0.732 | 0.759 | 0.724 | 0.403 | 17.20 | 0.790 |
| lear_nocov | 0.780 | 0.757 | 0.764 | 0.402 | 17.09 | 0.797 |
| lear | 0.789 | 0.748 | 0.740 | 0.398 | 17.24 | 0.783 |
| timestral_5m (contaminated) | 0.772 | 0.743 | 0.732 | 0.389 | 17.75 | 0.784 |
| tirex | 0.732 | 0.733 | 0.756 | 0.384 | 17.73 | 0.773 |
| **wm_power_direct (twin)** | 0.772 | 0.723 | 0.756 | 0.372 | 18.06 | 0.780 |
| arx | 0.756 | 0.741 | 0.683 | 0.370 | 18.07 | 0.826 |
| **wm_power (candidate)** | 0.740 | 0.732 | 0.756 | 0.369 | 18.03 | 0.775 |
| chronos_bolt_base | 0.724 | 0.722 | 0.724 | 0.354 | 18.55 | 0.809 |
| alma_s0 (zero-shot) | 0.650 | 0.663 | 0.642 | 0.220 | 21.98 | 0.748 |
| naive_w / naive_epf / naive_d | 0.724 / 0.634 / 0.528 | 0.651 / 0.566 / 0.522 | | -0.142 / 0 / 0.086 | | |
| kronos_small (n=25) | 0.640 | 0.623 | 0.640 | -0.332 | 35.94 | 0.482 |

Pairwise (harness DM-HLN, 7-day block bootstrap; Holm = the prereg's claim family):
- wm_power beats (Holm < 0.05 and CI excludes 0): naive_epf on P2 and QS, alma_s0 on P2 (Holm 0.032) and QS (Holm
  0.003), kronos_small on QS. Numerically better, not significant: chronos_bolt_base (QS, P1, P2), tirex (P1).
- wm_power is numerically WORSE than lear (QS +0.34 EUR, p 0.07; P1 -0.049, p 0.11), chronos2 (QS +0.59, p 0.068,
  bootstrap CI [0.12, 1.07] excludes 0 but Holm 0.95; P2 -0.028), tirex (QS +0.17, n.s.). It does not beat LEAR,
  Chronos-2 or TiRex on VAL.
- World-model test (candidate vs twin): QS +0.043 EUR (twin better, p 0.44), P1 -0.033 (twin better, p 0.10), P2
  +0.008 (WM better, p 0.15): **no significant difference**, consistent with DEV.

## 5. Application demo: wind scenarios (MODEL SIMULATION, not a validated forecast)

`tools/power/wm_scenario.py --model wm_power` -> `runs/power/scenario_demo.json`. DEV target days 2025-01-01..
2025-06-30 (180 days; lead panel through `build_context`, harness information set). Lever: an intervention on the
latent state, rl_hat(D,h) += X * wind(D-2,h) ("tomorrow's wind X% below its last known level"), read out through the
learned merit-order curve (monotone by construction; spreads independent of rl_hat).

| wind vs lagged level | -50% | -30% | -20% | -10% | 0 | +10% | +30% |
|---|---|---|---|---|---|---|---|
| mean daily median price, EUR/MWh | 97.5 | 95.6 | 94.6 | 93.6 | 92.5 | 91.5 | 89.3 |
| shift vs base | +4.9 | +3.0 | +2.1 | +1.0 | 0 | -1.1 | -3.3 |
| mean P(hour up vs D-1) | 0.554 | 0.531 | 0.518 | 0.505 | 0.492 | 0.478 | 0.450 |

- Internally consistent: 0 monotonicity violations in 336,960 (day, hour, quantile, adjacent-scenario) comparisons;
  the shift scales with the wind that is removed (-30% wind: +1.1 / +2.6 / +5.4 EUR/MWh on low / mid / high lagged-
  wind days, terciles at 9.3 and 16.0 GW), roughly uniform across hours (2.8-3.4 EUR/MWh).
- Limitations: the whole distribution shifts in parallel (80% width fixed at 58.5 EUR/MWh: the spread head does not
  see rl_hat, so the model cannot say that low wind also raises scarcity risk); the implied merit-order slope
  (~0.7 EUR/MWh per GW) is flatter than typical German estimates, i.e. attenuated, because the price median leans
  on the lagged-price channel; the lever is on D's latent residual load given a world where D-2 wind was observed,
  not a causal estimate. Not validated against realised wind shocks.

## 6. Dated correction (2026-10-31, added after the TEST audit)

The "written 21:05" label in section 3 and `frozen_at 21:08` in `runs/power/FROZEN.json` are wrong. The freeze commit 7e21933 is timestamped 21:01:41, and the VAL forecasts were written at 21:01:54, after it. The auditor verified this in docs/audit_power.md. FROZEN.json itself is not edited, because its sha is part of the audited record.
