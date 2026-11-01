# Alma-Fin world-model search (crypto perps: day-ahead vol + risk-sizing policy)

Owner: Alma-Fin architect (agent). Files: `tools/fin/wm_*.py`, `tools/fin/agents/**`, `tools/fin/search/**`,
`runs/fin/search/**` (force-added selectively), `runs/fin/FROZEN.json`, this document. Binding harness protocol:
`docs/fin_prereg.md` (36d2232); build spec `docs/fin_research.md` (f285496); interface `tools/fin/README.md`.

## 0. Protocol (written 2026-10-31, BEFORE any DEV score of any model by the architect; committed before the first run)

The architect has not read any VAL output (the lead's `runs/fin/val/` files are not opened) and never reads a
value dated after 2025-06-30 for search or training. Every search Panel is truncated to decision days <= 2025-06-30
right after loading (`tools/fin/search/common.py: dev_panel()`, asserted), so VAL rows are never in memory during
the search.

**Walk-forward DEV backtest (the search metric).** One fold:
- fit: training coins only (16), labels <= **2024-06-30** (`fit_end`, asserted on every training sample);
- backtest: decision days **2024-07-01..2025-06-30** (365 days), all 24 coins (the 8 held-out coins are never
  fitted, so their backtest is zero-shot) through the lead's `harness.run` / `harness.pnl` (same contexts, clock,
  fills, costs and funding as VAL/TEST), and the 13 ETFs (daily-only, zero-shot) for the transfer columns.
- References on the same fold, refit with `fit_end` = 2024-06-30 by temporarily setting `data.TRAIN_END` (the
  lead's factories cut through `models.train_view()`): majority, rv_persist, har, harx, har_roll, garch, lgbm,
  lgbm_asset (each asset's own history <= 2024-06-30), buy_hold, tsmom_1/7/30, meanrev_1, voltgt_tsmom,
  lgbm_policy. Foundation models are not run on DEV (Kronos saw DEV; cost); they are compared on VAL/TEST by the lead.

**Selection metrics (pre-declared).**
- Forecasters: `S_F = P1` = pooled coin-day hit rate, all 24 coins, on the backtest (direction as in the harness:
  `p_up > 0.5`). Tie-break: P3 mean pinball (levels 0.1..0.9) on log RV, lower is better.
- Continual rule: `S_C = P1(continual) - P1(frozen)` on the backtest, with the online rule applied through
  `observe()` exactly as in VAL/TEST (the frozen weights are the fold model).
- Agents: `S_A` = annualised net Sharpe on the 8 held-out coins (prereg P2 definition, harness costs + funding).
- Every run (crashes included) is a row in `runs/fin/search/results.tsv` with its config, seed, fold and metrics.

**Search space (small on purpose; K is counted).**
- WM forecaster `wm_fin`: latent size d in {64, 128}, each with its direct twin -> 4 single-seed rows.
- The best d by S_F of the ROLLOUT candidate (not of the twin) is the recipe; twins always use the same recipe.
- Continual rule on the chosen recipe: S in {1, 3} SGD steps/day x lr in {1e-4, 3e-4} (Adam not used online: plain
  SGD with momentum 0), replay = the last 60 completed days of the context + the newly revealed day, L2-to-init
  weight 1.0 fixed -> 4 rows. Chosen by S_C; if no rule has S_C > 0, the rule with the largest S_C is still frozen
  (the claim is tested on VAL/TEST, and the DEV result is reported as is).
- Agents on the chosen WM: `wm_agent` (Dreamer-style actor-critic in imagination) with risk penalty
  lambda in {0.5, 2.0}; matched twins `ppo` and `dqn` with the same lambda, observations, horizon, batch, gradient
  steps and seeds; `wm_voltgt` (no RL: sign(30-day return) x min(1, sigma*/sigma_WM), sigma_WM from wm_fin's median).
  lambda is chosen by S_A of `wm_agent`; the twins use the same lambda.
- Finalists: the chosen recipe is re-run with seeds 1 and 2 (candidate and twin, both fold-trained); the DEV
  comparison of each pillar uses the 3-seed ensembles (forecasts: mean p_up, mean of quantiles re-sorted;
  agents: mean position). CIs: 7-day circular block bootstrap of the daily difference (prereg 6.1/6.3).

**Freeze.** The chosen recipe is REFIT with the same steps and seeds on all DEV (training coins, labels
<= 2025-06-30, `models.train_view()`); no selection happens on the refit. Frozen models (each a 3-seed ensemble
in one checkpoint file, sha256 in the wrapper meta and in FROZEN.json): `wm_fin` (claim model), `wm_direct`
(twin), `wm_fin_cl` (continual; its twin is `wm_fin_cl:frozen`, i.e. the identical weights with observe disabled),
`wm_agent` (claim model), `ppo`, `dqn` (twins), `wm_voltgt` (non-RL control). `runs/fin/FROZEN.json` records
claim_models, K (every configuration scored on DEV, crashes and smoke runs listed separately),
`var_sharpe_trials` (variance of the per-period net Sharpe of every agent configuration scored on the DEV fold,
held-out coins), the online rule, twin-matching evidence (param counts, steps, seeds), and the go/no-go rule for VAL.
It is committed together with the checkpoints and wrappers BEFORE VAL. VAL is then run once
(`run_eval.py --split val --models <frozen list>`), results appended to section 4 below; no change after that.
TEST is never run or touched by the architect.

**Leak test** `tools/fin/search/test_leak.py` (must pass before freeze): perturbing any daily value of days >= D,
any hourly value of hours >= D 00:00, or the fill-bar prices of D does not change any frozen model's output at D
(bit-identical); perturbing day D-1 / the last hour before D changes them; a continual model's observe() buffer
never holds a row dated >= the next decision day, and its decisions up to a perturbation day are bit-identical.

## 1. Architecture (tools/fin/wm_core.py)

Per asset, per day t, 17 asset-agnostic features (no asset ID; anchor A = mean log RV of the 30 days before D):
log RV - A, log Parkinson - A, return / exp(A/2), |return / exp(A/2)|, log volume vs its 30-day mean, market
factor (cross-sectional mean over the context's assets of the first and third features), taker-buy ratio - 0.5,
funding, the four 6-hour-block log RVs of day t (crypto, from hourly rv5), masks (crypto flow, intraday, valid).
ETFs: flow/funding/intraday features are masked; training applies the same masks at random (p = 0.3) so the
daily-only regime is seen in training. Window W = 28 days.

- RSSM (DreamerV3-lite): deterministic GRU state h (d) + Gaussian stochastic z (16). The transition over one day is
  FOUR 6-hour sub-steps of a GRUCell with prior samples z; each sub-step decodes the block log RV and block return
  (supervised by crypto hourly data in training). At the end of a day the posterior z is inferred from the day's
  features, which are reconstructed by a decoder; KL balancing (0.8/0.2) with 1 free nat.
- `wm_fin` (rollout): from the filtered state at D-1, roll the four sub-steps of day D (16 prior samples at
  inference); the day heads (13 quantiles of log RV(D) - A, P1 logit, return-up logit) read the rolled latents.
- `wm_direct` (twin): identical modules, losses, data, steps, seeds; the day heads read the filtered state at
  D-1 (no rollout). Parameter counts are identical (same module list).
- `wm_agent`: actor pi(a | latent, a_prev) and critic V(latent, a_prev), a in [-1, 1] as a fraction of the vol
  budget: position = clip(a * sigma* / sigma_WM, -1, 1), sigma* = 40%/yr (crypto), 10%/yr (ETF). Trained on imagined
  10-day latent trajectories from real DEV start states (dynamics backprop + lambda-returns, Dreamer-style).
  Reward (in units of sigma*): a * r_hat * sigma*/sigma_WM - cost * |delta position| - position * funding -
  lambda * (position * r_hat)^2, cost = 10 bp (7 bp BTC/ETH) per unit, funding = last known rate.
  a_prev at inference is recomputed deterministically by running the policy over the context's last 10 filtered
  states from a = 0 (no hidden state across calls).
- `ppo` / `dqn` (twins, `tools/fin/agents/`): the same observation (frozen WM latent + a_prev), same nets, horizon,
  batch, gradient steps, seeds and reward, trained on REAL DEV transitions only (historical next-day returns and
  funding). DQN: 5 sizes {-1, -0.5, 0, 0.5, 1}.
- `wm_fin_cl`: `wm_fin` + the frozen online rule through `observe()`.

## 2. Experiment log

All rows: `runs/fin/search/results.tsv` (A, 116 rows) and `runs/fin/search_b/results_b.tsv` (B, 923 rows).
Loop states: `runs/fin/search/loop_{wm,cl,rl}.json`. Logs: `runs/fin/search/logs/`.

| stage | rows | outcome |
|---|---|---|
| round 1 (before Am1, scored on F1) | 4 | RSSM d64/d128 rollout/direct P1 0.678/0.681/0.685/0.680 |
| WM loop (E1-E3), no calendar | 31 + 1 aborted + 1 crash | START EM 0.6649; KEEPs lr 2e-3 (0.6690), 800 steps (0.6762) |
| WM dow extension (Am2) | 19 + leak-gate row | dow=True EM 0.7018; w_rec 1.0 EM **0.7077** (E1 0.691 / E2 0.714 / E3 0.718) |
| continual loop | 12 | every S_C in [-0.0014, +0.00001]; meta lr_adapt -0.0003; nothing kept |
| RL loop (Dreamer) | 13 | lam 0.25, lr 1e-4: EM held-out Sharpe 1.85; stopped when the neighbourhood was exhausted |
| holdout F1 | 3 WM + 1 CL + 3 RL + voltgt ref | see section 3 |
| 3-seed F1 + refit | 9 + 5 | see section 3 |
| F1 agent references (lead factories, fit <= 2024-06-30) | 7 | held-out Sharpe: buy_hold 0.69, tsmom_1 1.02, tsmom_7 -0.29, tsmom_30 0.46, meanrev_1 -2.10, voltgt_tsmom 0.08, lgbm_policy -0.31 |

Calendar-aware P1 references (lead, `runs/fin/fold_refs.json`):

| model | EM (E1-E3) | F1 |
|---|---|---|
| lgbm_dow | 0.6930 | 0.7087 |
| har_intra_dow | 0.6815 | 0.7167 (post hoc) |
| har_dow | 0.6751 | 0.6968 |
| lgbm | 0.6502 | 0.6686 |
| har / har_roll | 0.627 / 0.619 | 0.664 / 0.668 |

What helped:
- the weekday-of-D input (+2.6 pp EM on the RSSM);
- lr 2e-3 with 800 steps;
- reconstruction weight 1.0.

What did not help:
- fewer sub-steps (sub 1/2 tie);
- larger d;
- zd 8/32;
- ETF-mask changes;
- removing the market factor.

The p_up BCE head is essential: without it, P1 drops to about 0.565.

## 3. Selection and freeze

Mechanical decisions (`runs/fin/search/{wm,cl,rl}_final.json`). F1 numbers are post-selection (RT-11).
- **WM:** top-3 kept configurations were scored on F1: 0.7194 (dow + w_rec 1.0), 0.7179 (dow), 0.6885 (no dow). Pick: `50b221e074`, confirmed (>= 0.6779). B's best F1 was 0.7152 (moe4), so the pooled P1 finalist is A's `wm_fin`.
- **P1 gate (Am2):** `wm_fin`'s recipe EM 0.7077 beats lgbm_dow's 0.6930 overall and on every fold (E3 by only 0.001). B's moe4 passes by 0.0004 EM, a **tie**; it is frozen as a non-claim model. Both EMs were known when the gate was committed.
- **Continual:** only the START rule was ever kept, F1 S_C +0.0021 (> 0, confirmed). Frozen rule: plain SGD, 1 step per day, lr 1e-4, L2-to-init 1.0, 60-day replay, all parameters.
- **RL:** Dreamer `fa2ea8f71c` (lam 0.25, lr 1e-4, horizon 10, gamma 0.9, 500 steps). F1 held-out Sharpe 0.59 vs 0.07 for `wm_voltgt` on the same WM, so confirmed.
- **Refit** on labels <= 2025-06-30, 3 seeds, no selection. sha256 values are in FROZEN.json and asserted by the wrappers.

DEV F1 3-seed pillar comparisons (descriptive only; 7-day block-bootstrap 95% CI, DM p):

| pillar | candidate | twin | difference |
|---|---|---|---|
| rollout vs direct, P1 all | 0.7171 | 0.7159 | +0.0013 [-0.0038, +0.0061], p 0.60 |
| rollout vs direct, P3 QS all | 0.1514 | 0.1525 | -0.0011 [-0.0021, -0.0001], p 0.055 |
| rollout vs direct, ETF P1 / QS | 0.591 / 0.228 | 0.584 / 0.232 | +0.0065 [+0.0003, +0.0129] / -0.0043 [-0.0064, -0.0024] |
| continual vs frozen, P1 all | 0.7188 | 0.7171 | +0.0017 [-0.0008, +0.0043], p 0.20 |
| continual vs frozen, P3 QS all | 0.1507 | 0.1514 | -0.0006 [-0.0013, +0.0001], p 0.12 |
| wm_agent vs ppo, held-out Sharpe | 0.46 | -0.17 | +0.62 [-0.22, +1.44], p 0.15 |
| wm_agent vs dqn, held-out Sharpe | 0.46 | 0.97 | -0.51 [-2.08, +0.93], p 0.55 |
| wm_agent vs wm_voltgt (no RL), held-out Sharpe | 0.46 | 0.06 | +0.40 [-1.89, +2.89], p 0.72 |

Reading:
- **Rollout vs direct.** The WM is not distinguishable from its direct twin on P1; the rollout is slightly better on P3 QS.
- **Continual vs frozen.** Continual learning is +0.17 pp P1, not significant.
- **RL.** The Dreamer agent beats PPO numerically, but DQN is numerically better than the Dreamer agent. RL is not shown to beat the non-RL vol-target control (wide CI). All F1 agent Sharpes are below 1, and F1 buy_hold is 0.69.
- **Portability.** F1 held-out coins score 0.7185 vs 0.7164 for the training coins. ETF P1 is 0.59 (descriptive).

**Declared outcome, P1 gate:** `wm_fin` qualifies as the P1 finalist on E1-E3. Its F1 edge over har_intra_dow (0.7194 vs 0.7167) is within noise and post-selection.

K = 985 (A 109 + B 876 non-reference rows, crashes, screens and holdout looks included). var_sharpe_trials = 0.00218 (per-day Sharpe units, 74 per-fold or F1 agent rows; the row list is in FROZEN.json).

## 4. VAL (run once after the freeze commit)

Freeze of record: f2bbdbc (2026-11-01 07:46:20 UTC). The red team gave GO (docs/fin_redteam.md pass 13). The rehearsal
`run_eval.py --split val --rehearse` (full list, `runs/fin/rehearsal_val/`) completed at ~08:50 UTC with no crash;
its metrics were not read. VAL was then run ONCE:
`run_eval.py --split val --models wm_fin,wm_direct,wm_fin_cl,wm_fin_cl:frozen,wm_agent,ppo,dqn,wm_voltgt,wm_b_moe4`
(~08:55 UTC; `runs/fin/val/metrics.json`). VAL has 123 days. It is for debugging and confirmation only; no claim is made
on it (prereg s7). Go/no-go: no crash and no non-finite outputs, so **GO** for TEST under the pre-written rule. Nothing
was changed after VAL.

P1 (crypto, all 24 coins; base rate 0.504):

| model | P1 all | P1 held-out | P3 QS |
|---|---|---|---|
| wm_fin_cl | 0.7375 | 0.7154 | 0.1543 |
| wm_fin (= wm_fin_cl:frozen) | 0.7371 | 0.7144 | 0.1543 |
| lgbm_dow | 0.7358 | 0.7266 | 0.1590 |
| wm_direct | 0.7348 | 0.7215 | 0.1533 |
| wm_b_moe4 | 0.7314 | 0.7033 | 0.1606 |
| har_intra_dow | 0.7293 | 0.7205 | 0.1647 |
| har_dow | 0.7141 | 0.7073 | 0.1706 |
| tirex / chronos2 | 0.7043 / 0.6992 | | 0.171 / 0.170 |
| lgbm / har / har_roll | 0.675 / 0.662 / 0.666 | | |

ETF (E1, descriptive): wm_fin 0.574, below the best (chronos2 0.596, har_roll 0.592, garch 0.592).

Pillar tests on VAL (day-level DM p, 7-day block-bootstrap 95% CI; descriptive):

| test | difference | CI | p |
|---|---|---|---|
| rollout vs direct, P1 | +0.0024 | [-0.0058, +0.0112] | 0.63 |
| rollout vs direct, P3 QS (lower is better) | +0.0011 (direct better) | [-0.0003, +0.0024] | 0.23 |
| continual vs frozen, P1 | +0.0003 | [0.0000, +0.0010] | 0.32 (3 updates/day) |
| continual vs frozen, P3 | -0.00002 | [-0.00008, +0.00005] | 0.70 |
| wm_fin vs lgbm_dow, P1 | +0.0014 | [-0.025, +0.030] | 0.92 |
| wm_fin vs har_intra_dow, P1 | +0.0078 | [-0.016, +0.032] | 0.63 |
| portability: wm_fin vs lgbm_asset, P1 held-out | +0.051 | [+0.012, +0.088] | 0.022 (Holm-own 0.30) |
| portability: wm_fin vs har_roll, P1 held-out | +0.060 | [+0.015, +0.104] | 0.018 (Holm-own 0.27) |
| wm_agent vs ppo, Sharpe held-out | -0.16 | [-0.92, +0.53] | 0.72 |
| wm_agent vs dqn, Sharpe held-out | -2.26 | [-4.92, +0.96] | 0.19 |
| wm_agent vs wm_voltgt, Sharpe held-out | -1.02 | [-4.40, +2.58] | 0.59 |

P2 (held-out coins, net Sharpe, VAL):

| agent | Sharpe | PSR | DSR (K=985) |
|---|---|---|---|
| wm_agent | -2.39 | 0.07 | 0.0004 |
| ppo | -2.23 | | 0.0012 |
| dqn | -0.13 | | 0.041 |
| wm_voltgt | -1.37 | | 0.006 |
| buy_hold | 0.57 | | |
| tsmom_1 | 1.45 | | |
| tsmom_7 | 1.20 | | |

DSR sensitivity at K=1,038 (reference rows included): wm_agent 0.0004, ppo 0.0011, dqn 0.040, wm_voltgt 0.0062.

Reading (VAL, 123 days, no claims):
- The WM forecaster is top on crypto P1, tied with lgbm_dow.
- Rollout is not distinguishable from direct.
- Continual adds almost nothing.
- The zero-shot pooled model beats the per-asset twins on the held-out coins by 5-6 pp. Bootstrap CIs exclude 0, but Holm-own p is 0.27-0.30.
- The RL agents lose money on held-out coins in VAL. RL does not beat the non-RL control.

**Reconciling the "wm_fin F1 P1" numbers.** All of these are post-selection:
- 0.7194: single seed 0, recipe 50b221e074 fit <= 2024-06-30, scored on F1 (holdout_wm row; `runs/fin/search/holdout_wm.json`).
- 0.7179: the second-ranked recipe 0dbccd3e62 (dow, w_rec 0.5), single seed 0, on F1.
- 0.7171: the 3-seed F1 ensemble of 50b221e074, seeds 0-2 (f1ens_wm row; `runs/fin/search/preds/f1ens_wm_50b221e074_none_crypto.npz`).
- 0.7077: NOT an F1 number. It is the recipe's E1-E3 loop EM (mean of 0.691 / 0.714 / 0.718, seed 0, fold models).
- The refit (frozen, fit <= 2025-06-30) was never scored on DEV.

### Amendment Am0 (before any DEV score; 2026-10-31 ~23:50 EDT, committed in the commit that adds it)
CPU timing on the shared Mac (0.45 s per Adam step with W = 28): the window is W = **14** days (the last 336 of the
672 context hours), training budget **1000** Adam steps (batch 256, OneCycle lr 1e-3, wd 1e-5, clip 1, 3 threads)
for every WM row, twin and refit. No score of any kind had been computed when this was decided.

### Amendment Am1: scaled autoresearch loop with nested selection (2026-11-01 ~00:10 EDT, before the scaled search; committed in the commit that adds it)

Reason: the user asked, via the coordinator at ~00:05 EDT, for a self-improving autoresearch loop over hundreds of
variations across all four pillars, plus a self-improving continual variant. Am1 supersedes section 0's search
space and selection. The info set, twins, freeze and VAL rules are unchanged.

**Disclosure.** Before this instruction, four round-1 rows were scored on the window 2024-07-01..2025-06-30:
RSSM d64/d128 x rollout/direct, seed 0 (results.tsv; P1 0.6779 / 0.6808 / 0.6853 / 0.6804). Am1 now makes this window
the selection holdout. Those four looks count in K. They do not choose anything. The loop's starting incumbent is
**d = 64**, the Am0 default, even though d128-rollout scored higher on that window.

**Nested selection.**
- Loop folds (shared with architect B, `runs/fin/search_b/`):
  - E1: fit <= 2022-12-31, test 2023-01-01..2023-06-30;
  - E2: fit <= 2023-06-30, test 2023-07-01..2023-12-31;
  - E3: fit <= 2023-12-31, test 2024-01-01..2024-06-30.
  - EM = the mean over E1..E3.
- Every loop panel is cut at 2024-06-30 at load (asserted). The loop never sees a later value.
- Holdout F1: fit <= 2024-06-30, test 2024-07-01..2025-06-30.
  - Only finalists are scored on it, once each.
  - The loop never optimises on it.
- VAL stays run-once after the freeze. TEST is untouched.

**Loop.** Autoresearch style, automated (`tools/fin/search/loop.py`):
1. Propose a one-knob mutation of the current incumbent. The proposer reads the log: it skips configurations already tried and prefers knobs whose past mutations were kept.
2. Train small and fast.
3. Screen on E3.
4. If E3 >= incumbent E3 - tol_screen, also run E1 and E2.
5. KEEP (new incumbent) only if EM > incumbent EM + delta. Otherwise REVERT.
6. Every row (screen-only, full, crash) is logged to `runs/fin/search/results.tsv` (fold E3 or EM).

**Search spaces, scores, keep thresholds.**
- **WM** (RSSM, rollout candidate), score P1 (all 24 coins on the fold; held-out coins zero-shot):
  - d in {32, 48, 64, 96, 128}; z in {8, 16, 32}; sub-steps/day in {2, 3, 4, 6}; W in {7, 10, 14, 21, 28};
  - lr in {3e-4, 1e-3, 2e-3}; steps in {300, 500, 800}; batch in {128, 256};
  - loss weights: w_q in {1, 2, 4}, w_p in {0, 0.5, 1, 2}, w_blk in {0, 0.5, 1}, w_kl in {0.03, 0.1, 0.3}, w_rec in {0.25, 0.5, 1};
  - ETF-mask probability in {0, 0.3, 0.5}; market-factor features on/off.
  - Starting incumbent: Am0 recipe with 500 steps. tol_screen 0.003, delta 0.002.
- **Continual** (on the WM loop winner's fold models), score S_C = P1(online) - P1(frozen):
  - SGD steps/day in {1, 2, 3}; lr in {3e-5, 1e-4, 3e-4, 1e-3}; l2-to-init in {0.1, 1, 10}; replay days in {20, 60, 120};
  - params in {all, heads}.
  - **Self-improving meta-rule** (`meta=lr_adapt`): after each revealed day, compare the newest day's loss under the online weights with its loss under the frozen weights (both computed before any update on that day, so it is a pure out-of-sample past error). Then lr <- clip(lr * exp(+eta) if online won else lr * exp(-eta), lr_min, lr_max). eta in {0.05, 0.2}; lr range fixed [1e-5, 1e-3].
  - The meta-rule and its constants are hyperparameters, frozen before TEST.
  - tol_screen 0.002, delta 0.001.
- **RL** (on the WM loop winner's fold models, seed 0), score = held-out net Sharpe (annualised):
  - lambda in {0, 0.25, 0.5, 1, 2}; horizon in {5, 10, 20}; gamma in {0.8, 0.9, 0.97}; actor lr in {1e-4, 3e-4, 1e-3};
  - steps in {500, 1000}; cost multiplier in reward in {1, 2}.
  - Twins: PPO/DQN get the selected shared knobs. Each also gets <= 4 algorithm-specific rows (PPO clip/epochs, DQN target period) on the E folds, logged and counted.
  - tol_screen 0.3, delta 0.15 (Sharpe units).
- **Cross-asset** is not optimised separately. ETF and held-out columns are reported at the holdout.

**Budget and stop rule.** Per loop, stop at whichever comes first:
- the run budget: WM 150, continual 80, RL 80;
- the clock: WM loop ends by 02:00, continual by 02:35, RL by 03:05 EDT;
- 25 consecutive REVERTs.

**Finalists.**
- Per pillar, the top-3 distinct kept configurations by EM go to F1 once (plus architect B's <= 3 per pillar).
- Pooled finalist = the best F1, ties broken by P3. It is confirmed only if F1 >= F1 of the starting incumbent (WM: the logged d64 row; continual: 0, i.e. S_C > 0; RL: the F1 Sharpe of `wm_voltgt`). Otherwise the starting incumbent is frozen.
- Confirmed finalists, with their twins (same recipe), go to the 3-seed F1 stage and then the refit on all DEV (<= 2025-06-30). No selection happens at either step.

**K.** K counts every row of results.tsv + `runs/fin/search_b/results_b.tsv` (screens, crashes, holdout, references excluded only if marked kind=ref). var_sharpe_trials = variance of the per-period net Sharpe over all agent rows of both logs.

**Leak gate.** Each family must pass `tools/fin/search/test_leak.py` (`--family` mode, on fold checkpoints) before its loop scores count:
- RSSM forecaster rollout/direct;
- continual SGD;
- meta-continual;
- Dreamer/PPO/DQN agents.

Results go to `runs/fin/search/test_leak_<family>.txt`. Red-team findings are fixed before the freeze. Memory: one job of mine at a time, <= ~400 MB, swap and ps checked by the loop before each run.

### Amendment Am2: post-hoc weekday extension and new finalist gate (2026-11-01 06:15 UTC, before the extension starts)

Times are in UTC from here on (RT-14). DST ended at 06:00 UTC (02:00 EDT -> 01:00 EST). The Am1 clock stops were
meant as EDT:
- WM loop: stopped at 06:05 UTC (5 minutes late; the in-flight proposal is logged as `aborted`, counted in K).
- Continual loop: stops at 06:30 UTC.

**Why.** The red team found (RT-15) that most of the P1 edge over the registered baselines comes from a weekday effect.
- F1 hit rates: weekday-only rule 0.609, HAR 0.664, HAR + weekday 0.693, red team's `har_intra_dow` 0.7154.
- The lead is registering har_dow / lgbm_dow / har_intra_dow in a prereg amendment.
- The RSSM had no calendar input.

**Extension (post hoc, disclosed as such).**
- New WM knob `dow`: a weekday one-hot of the decision day D, appended to the head inputs. It is calendar data known in advance, not an asset id.
- The leak gate (`test_leak.py --family rssm` on a dow fold checkpoint) must pass first.
- Then the WM loop resumes from its incumbent. The first proposal is forced to dow=True; after that the proposer runs as before.
- Stop at **07:05 UTC**, 30 proposals, or 25 consecutive reverts, whichever comes first.
- The RL loop then runs on the resulting WM incumbent until **07:35 UTC**. The continual loop ran on the pre-extension incumbent (rules transfer, disclosed).
- Every extension row is logged and counted in K.

**P1 finalist gate (coordinator).** A P1 ("beats every strong baseline") finalist must numerically beat the best calendar-aware reference on E1-E3 (the lead's registered `har_intra_dow`). A config whose edge disappears against it is not a finalist. If no config qualifies, this document and FROZEN.json will say: "No P1 finalist beats har_intra_dow on E1–E3."

**Pillar tests** do not need a P1 finalist:
- RSSM rollout vs direct;
- continual vs frozen;
- WM agent vs PPO/DQN.
They use matched twins (same seeds, params, data) and are pre-declared in FROZEN.json before VAL.

**F1 caveats (RT-11).** F1 numbers of chosen models are post-selection: the maximum of the F1 looks from A and B (winner's curse). F1 candidate-vs-twin differences are descriptive only. The only F1 decision is the confirmation gate.

**Disclosures from architect B (relayed).**
1. Six c4/c5 cross-attention rows diverged to NaN and were logged as P1 = 0 with status ok. They count as crashes in K.
2. har_intra_dow was specified by the red team after it had seen F1, so its F1 score is post hoc.
3. B saw A's round-1 F1 rows before Am1. Nothing was tuned on them.
4. The cache label on B's three F1 checkpoints says loop3, but they are loop4.
5. B's edge comes from calendar and intraday features.
   - B's F1 numbers: moe4 0.7152 vs har_intra_dow 0.7154.
   - On E1-E3, moe4 beats B's re-implementation of har_intra_dow by +0.0109, 95% CI [-0.0038, 0.0258]. The decision uses the lead's registered implementation.
   - B's agents failed on F1 (Sharpe -0.32 / -0.37), and B's continual rules gained nothing.
6. RT-4 (held-out coins in the market-factor channels) and RT-9 (a one-day purge) were fixed mid-search. Pre-fix rows stay in K.

**Other disclosures.**
- E-fold selection scores P1 on all 24 coins, so the held-out coins influence selection. DEV held-out numbers are not zero-shot-clean (TEST is unaffected).
- RL real-transition rewards were switched to fill-to-fill returns and realised funding (RT-8) before the RL loop. Dreamer rewards stay WM-imagined.
- Before 06:15 UTC, the loop's RL `real_states` filtered the full 28-day dataset window instead of the model's w = 14 days. No RL score had been computed with that code, and it is fixed.

### Am2 correction (RT-16 + coordinator; 2026-11-01 ~06:35 UTC; written before B's per-fold numbers were examined)
The P1 finalist gate is defined against the MAXIMUM over the lead's registered calendar-aware baselines {har_dow,
lgbm_dow, har_intra_dow}, using the lead's implementation (`runs/fin/fold_refs.json`). On E1-E3 that is lgbm_dow,
EM 0.6930 (E1 0.666, E2 0.696, E3 0.717). A P1 finalist must satisfy two conditions:
(a) its EM is strictly higher than that EM;
(b) it does not lose to the per-fold maximum reference on more than one of E1, E2, E3.
Otherwise no P1 claim model is frozen, and the record says: "No P1 finalist beats the best registered baseline on E1–E3."
The architects did not consult the lead's VAL numbers (`docs/fin_results.md`, `runs/fin/val/`).
