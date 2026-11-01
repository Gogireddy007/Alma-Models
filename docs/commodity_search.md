# Commodity world-model search (WTI)

Owner: commodity world-model architect. Files: `tools/commodity/wm_*.py`, `tools/commodity/search/`,
`runs/commodity/search/` (git-ignored except what is force-added), this document.

## 0. Protocol (written 2026-10-31 18:45, after the reference rows, the Alma zero-shot rows and r1_obs_none were in,
before any sv / rs / cross-commodity result)

**Data.** FRED daily series fetched by `tools/commodity/wm_data.py` with `coed=2025-06-30` in the request itself
(DEV cache `runs/commodity/search/data/dev/`); every loader asserts max date <= 2025-06-30 (DEV) and nothing is
ever requested past 2025-10-31. VAL uses a separate cache (`.../data/val/`, `coed=2025-10-31`) and is read once, at
the end, for the frozen candidate and its twin. TEST is never requested. Rows = WTI trading days; the other series
are forward-filled causally (<= 10 days). WTI's -37.63 print (2020-04-20) is floored at 1.0 before the log (TRAIN
only; the backtest is 2023+).

**Backtest (the search metric).** WTI, origins every 5th trading day from 2023-01-03 such that the h=20 target is
<= 2025-06-30 (120 origins). Every search model is trained only on windows whose LAST TARGET is before 2023-01-01
(`wm_core.train_origins`, asserted). No retraining inside the backtest.

**Metric.** Quantiles 0.1..0.9 of log price at h = 1..20. CRPS ~ 2 x mean pinball over the 9 levels. RW = Gaussian
random walk on log price, sigma = RMS of the last 250 log returns. DA_h = sign agreement of the median change with
the realised change (ties 0.5). Vol-direction accuracy (next-20d realised vol > past-20d, prediction from the h=20
80% interval width) and 80% coverage at h=20 are reported, not scored.

**Selection score (pre-declared in the brief):** `score = mean_{h in 1,5,20} [0.5 * CRPS_h / CRPS_h(RW) + 0.5 * (1 - DA_h)]`,
lower is better, RW = 0.750 exactly.

**Noise rules (added now, because the first rows already show the score is dominated by DA noise: 2 fine-tuning
steps moved Alma from 0.731 to 0.763; the 90% block-bootstrap interval of one model is about +-0.035).**
1. Every row also reports the same score on DAILY origins over the same period (600 origins, `score_d`), as a
   robustness read; it does not replace the pre-declared score.
2. Finalists (best world-model recipes by weekly score) are re-run with 3 seeds; the decision uses the 3-seed mean of
   the weekly score. The frozen candidate is the 3-seed ensemble (quantile average) of the chosen recipe, and its
   twin is the 3-seed ensemble of the same recipe with `abl=direct`.
3. A world model "beats its twin" only if the seed-mean weekly score is lower AND the daily score is lower.
4. The frozen checkpoints are the backtest models themselves (trained on data before 2023-01-01) unless the
   expanding refit (same recipe, same step count, data to 2025-06-30) is declared below before VAL is read.

**Loop.** autoresearch pattern: fixed wall-clock budget per experiment (default 5 min, CPU, 2 threads, nice 10),
one scalar (the score), keep/discard, every run appended to `runs/commodity/search/results.tsv` (crashes included).

## 1. Architectures (all in `tools/commodity/wm_models.py`, `tools/commodity/wm_alma.py`)

Common encoder for the new models: per-step features (log return / window RMS, mask, log window scale; + spread
z-scores for cross-commodity) -> MLP -> GRU state d_t (the observer). Every world model (`abl=none`) has a twin
(`abl=direct`) with the identical module list and parameter count: lead k is ONE step of the same GRU from the origin
state, `d_k = GRU(step_emb[k], d_origin)`, and the heads decode the cumulative change to k directly.

| arch | world model (abl=none) | params |
|---|---|---|
| obs | observer-SSM: same GRU run open-loop over the horizon on step embeddings; quantile head on d_k | 25,970 |
| sv | stochastic-volatility WM: explicit latent log-vol h_t (mean-reverting, score-driven correction by abs(eta) and leverage eta) + GRU return state; Student-t returns; closed-loop imagination (sampled returns fed back, latent vol shock sigma_v eps); 512 paths -> quantiles; loss = one-step NLL (teacher-forced filter) + sample CRPS of the rollout | 25,191 |
| rs | regime-switching WM (RSSM-style): K=3 discrete regimes, state-dependent transition matrix, exact forward filtering, per-regime Student-t; regimes and returns sampled in imagination | 25,776 |
| sv+xc | cross-commodity sv: joint latent over WTI, Brent, Henry Hub, NYH gasoline, NYH heating oil, Gulf jet, Mont Belvieu propane (low-rank Gaussian joint emission, spreads recomputed from simulated prices) | 29,209 |
| obs+xc | cross-commodity observer | 26,738 |
| alma | Alma's pretrained SSM (`data/ckpt/wm_ssm_none_s0.pt`, never saw commodity data), zero-shot and fine-tuned on WTI | 365,937 |
| +pool | univariate model trained on windows of all 11 FRED energy series (each normalised by its own scale) | - |

Cross-commodity models cannot be frozen: the harness API passes only the target's own 1-D context (README:
"must use nothing but context"). They are DEV research rows only.

## 2. Experiment log

Every run is a row of `runs/commodity/search/results.tsv` (force-added to git), logs in `runs/commodity/search/logs/`.
Rounds: r1 = one 5-minute run per architecture x twin; r2 = Alma fine-tune with origin subsampling, sv with L=64;
r3 = Alma fine-tune at lr 3e-5, sv seeds 1-2; r4 = sv+pool seeds 1-2; f = final matched-step retrain (section 4).
`smoke_alma` is a 0.3-minute plumbing test. Failures: `r1_obs_xc_none` / `r1_obs_xc_direct` diverged (CRPS ratio
2-3; quantile head blow-up with 7 inputs, no fix attempted); `r1_alma_ft_*` got only 22-24 steps (12 s/step before
origin subsampling, machine load average 20-40); the 3 r4 runs were briefly mis-marked aborted and corrected (they had
finished). The machine was shared (load average 11-60), so wall-clock budgets gave very different step counts
between twins (e.g. r1_sv_none 286 steps vs r1_sv_direct 679); this is why the final pair uses matched max_steps.

## 3. Finalist selection (written 19:25, before the final retrain finished and before any VAL read)

Seed-mean weekly DEV score of the world-model recipes that got 3 seeds (rule 2), with their twins:

| recipe | WM seeds (s0, s1, s2) | WM mean | direct twin seeds | twin mean |
|---|---|---|---|---|
| sv, WTI only, L=128 | 0.7443, 0.7399, 0.7619 | **0.7487** | 0.7457, 0.7262, 0.7302 | 0.7340 |
| sv + pool (11 energy series) | 0.7234, 0.7743, 0.7690 | 0.7556 | 0.7287, 0.7628, 0.7412 | 0.7442 |

The single best world-model row, `r1_sv_pool_none` (0.7234), does not survive its seeds (mean 0.7556): a textbook
winner's curse. By the pre-declared rule the finalist recipe is **sv (stochastic-volatility WM), WTI only, L=128**,
seed-mean 0.7487, i.e. indistinguishable from the random walk (0.750; one model's 90% bootstrap interval is about
+-0.03). Its direct twin is better on these seed means (0.7340). So far the DEV evidence says the step-by-step
simulation does NOT help on this metric; the twin's edge is also within noise.

Not eligible: `r1_alma_zs_none` (0.7307, the lowest WM row) is the pretrained Alma run zero-shot; it has no matched
twin (`r1_alma_zs_direct` ran the rollout weights in direct mode, which is not a twin and is not quoted as a WM test),
and it is not commodity-trained. Fine-tuned Alma (0.752-0.782) got worse than zero-shot in every run, and its
direct rows started from the rollout weights (fixed in `wm_alma.py` after the audit; not re-run, search stopped).

K = configurations tried: 34 training/eval rows (incl. 1 smoke, 2 diverged) + 5 reference rows; with that many
tries, the best DEV row is optimistically biased by roughly the spread of the seed noise (~0.02-0.04).

**Final retrain (auditor B6).** The finalist and its twin are retrained with identical budgets: `--max_steps 600`,
same data (WTI windows with last target < 2023-01-01), seeds 0, 1, 2, same LR schedule. The frozen candidate is the
3-seed ensemble `wm_sv` (log-level quantile average) and the twin `wm_sv_direct` likewise. Sampling is re-seeded per
origin in the wrapper (forecasts do not depend on batch composition). DEV scores of the frozen ensembles are computed
through the harness wrapper (`tools/commodity/search/dev_final.py`).

**VAL declaration (before reading VAL).** The lead's harness (`tools/commodity/run_eval.py --split val`) is the
authoritative VAL run and decides inclusion; `tools/commodity/search/run_val.py` is a secondary read with this
document's metric only. Neither result changes the frozen checkpoints. The harness DEV split (origins from 2019) must
NOT be used for these models: they were trained on data up to 2022-12-30, so 2019-2022 DEV origins are in-sample.

## 4. Frozen candidate and twin (frozen 19:28, commit faa258f, before VAL)

| name | members (path, sha256) |
|---|---|
| `wm_sv` (world model) | `runs/commodity/search/ckpt/f_sv_none_s0.pt` 392a58a3a3596a1e8e71fe98061458e246bc5ab29a3326807f69af49edb8f2e2; `f_sv_none_s1.pt` 19a14e94e3834c3f35b42bd4bf9903c3d635548685401c8a78a9bdd15299ca27; `f_sv_none_s2.pt` 9d65eb3f6931001ef798c614a7a9cf6bcacf993755c106d07aa0c5461e13fdb1 |
| `wm_sv_direct` (twin) | `f_sv_direct_s0.pt` 8da818568e4e7ff344765ad31080c3e6236793c124d448f9faff91269dd80295; `f_sv_direct_s1.pt` 173c123ce6dfedafcf4a1531d15ae9c457deb8618bdae694f41ba17511604953; `f_sv_direct_s2.pt` e92697cf7a90a7a5aa7caaacafe84fd319fae2a1852ecf941d5219c473927390 |

25,191 parameters per member in both. Registered via `tools/commodity/wm_commodity.py` (`register()`; sha verified at
load; deterministic, per-origin seeding; all 13 harness quantiles; h <= 20, NaN beyond). Also in
`runs/commodity/FROZEN.json` (`models` + `wm_architect` block with shas, K and the winner's-curse note).

Final-retrain rows (matched 600 steps, weekly DEV score): WM s0/s1/s2 = 0.7322 / 0.7531 / 0.7542 (mean 0.7465);
twin = 0.7395 / 0.7765 / 0.7527 (mean 0.7562). Note this flips the r1-r3 seed means (where the twin was ahead):
the WM-vs-twin sign is not stable across retrains.

DEV backtest of the frozen ensembles through the harness wrapper (`runs/commodity/search/dev_final.json`):

| model | weekly score [90% CI] | CRPS ratio h1/h5/h20 | DA h1/h5/h20 | vol-dir | daily score |
|---|---|---|---|---|---|
| wm_sv | 0.734 [0.703, 0.760] | 0.986 / 0.997 / 0.996 | 0.525 / 0.542 / 0.508 | 0.633 | 0.7485 |
| wm_sv_direct | 0.755 [0.724, 0.790] | 0.988 / 1.003 / 1.000 | 0.525 / 0.458 / 0.475 | 0.542 | 0.7548 |
| random walk | 0.750 | 1 / 1 / 1 | 0.5 | 0.542 | 0.750 |

Rule 3 (lower weekly AND daily) is met by the WM over its twin, but the intervals overlap heavily and the daily score
is at the random walk. Plainly: **on DEV the candidate is within noise of the random walk.** The one consistent
signal is a small CRPS gain at h=1 (ratio 0.986-0.993, the volatility state) and vol-direction accuracy 0.61-0.63 vs
0.54 for the twin.

## 5. VALIDATION (run once, 19:31; 2025-07-01..2025-10-31)

Authoritative: lead's harness `run_eval.py --split val --models wm_sv,wm_sv_direct --series wti` (86 daily origins;
forecasts in `runs/commodity/val/wti/wm_sv*.npz`, metrics in `runs/commodity/val/metrics.json`):

| model | T1 dir. acc h1 / h5 / h20 | T4 CRPS skill vs RW h1 / h5 / h20 |
|---|---|---|
| wm_sv | 0.465 / 0.494 / 0.149 | -0.0005 / +0.0022 / -0.041 |
| wm_sv_direct | 0.500 / 0.444 / 0.149 | -0.011 / -0.027 / -0.049 |
| rw_gauss | 0.500 / 0.444 / 0.149 | 0 |
| chronos2 | 0.500 / 0.469 / 0.373 | -0.024 / +0.025 / +0.049 |
| tirex | 0.535 / 0.432 / 0.149 | -0.006 / -0.039 / -0.311 |
| alma_s0 (zero-shot, lead's settings) | 0.628 / 0.667 / 0.433 | -0.145 / +0.042 / -0.011 |

Secondary (this document's metric, `runs/commodity/search/val_results.json`): weekly (n=14) score wm_sv 0.781,
wm_sv_direct 0.781, RW 0.750; daily (n=66) 0.825 / 0.840 / 0.750. Both lose to the random walk on VAL, mainly at
h=20: the models' median carries a small positive drift learned from 1986-2022 (frac of "up" medians = 1.0 at h=20),
and WTI fell over the window. The WM is marginally better than its twin on VAL CRPS at every horizon (h1 -0.0005 vs
-0.011, h5 +0.002 vs -0.027, h20 -0.041 vs -0.049), consistent in sign with DEV but small.

## 6. Conclusions and flags for the auditor

- No commodity world model here beats the random walk with any confidence, on DEV or VAL; none should be expected to
  beat Chronos-2 on TEST. The WM-vs-twin gap is small and changed sign across retrains.
- Suspicious / limiting: (1) the learned positive drift makes the median always "up" at h=20; (2) harness DEV
  origins 2019-2022 are in-sample for these checkpoints (trained to 2022-12-30); score them only from 2023;
  (3) wall-clock budgets on a machine at load 11-60 gave twins unequal steps in r1-r4 (fixed for the final pair);
  (4) sample-based models' DA depended on the sampling seed before per-origin seeding (r1_sv_direct 0.7457 vs
  0.7510 re-sampled); (5) `runs/commodity/FROZEN.json` did not exist and was created here (`models` list + my block);
  the lead should check it against the strict TEST format; (6) the harness VAL run regenerated
  `runs/commodity/val/metrics.json` (adds the two models), `report.md` was not regenerated.
