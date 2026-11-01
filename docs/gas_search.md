# Alma-Gas world-model search (weekly EIA storage)

Owner: gas world-model architect. My files: `tools/gas/wm_*.py`, `tools/gas/search/**`, `runs/gas/search/**`,
`runs/gas/FROZEN.json`, `runs/gas/scenario_demo.json` and this document. The binding harness protocol is
`docs/gas_prereg.md` (lead, f9e01bd, plus amendments).

## 0. Protocol

Written and committed BEFORE any DEV score of any model. Up to this commit, the only things run were plumbing smoke
tests: about 24 contexts, 5 training steps, output shapes checked, no metric computed.

**Data.** Contexts come only from `storage.build_context(k, kind)` for `k` in `storage.train_weeks()`, i.e. target
weeks released 2014-01-15 .. 2025-06-30. They are cached in `runs/gas/search/cache/ctx_train.pkl` (not committed;
it can be rebuilt with `run.py cache`). Labels (dS, dev per series) are read only for those weeks. I never build,
load or score a VAL or TEST week; the only exception is the single VAL run through the lead's harness after the
freeze. Features: `tools/gas/search/feats.py`, a pure function of one context. The same function runs in training
and in the harness wrapper.

**Information set.** Exactly the lead's (prereg §2), because the model sees nothing but the context dict:
- Weather: observed up to origin-1; after that, the CPC forecast issue usable at the origin.
- Storage history: up to week W-1.
- HH price: up to the origin date.
- Calendar: US federal holidays, day of year.

Week W-1's daily weather is read from `obs_daily`. Those days all end on or before origin-1 at both origins.
`tools/gas/search/test_leak.py` checks this end to end:
- Forbidden perturbations must leave the forecasts of every architecture bit-identical: storage of week W and
  later, observed weather from the origin day on, unusable CPC issues, and HH prices after the origin.
- Allowed perturbations must change them: dS_{W-1}, observed weather of origin-1, the used CPC issue, and the
  origin-day HH price.

**DEV backtest (the search metric).** Rolling origin with 6 yearly folds by release date: Jul 1 Y .. Jun 30 Y+1,
Y = 2019..2024. Together they cover the lead's DEV window, weeks released 2019-07-01 .. 2025-06-30, about 313
weeks x 2 origin kinds. Each fold is trained only on train weeks released strictly before the fold start
(asserted). Training scales (dev sd per series) and the CPC forecast-error sd used by Monte Carlo variants are
computed inside the fold, from those training weeks only. One model serves both origins: it is trained on thu and
fri contexts together, and per-day source flags mark which days are observed and which are forecast.

**Metrics (L48 unless stated, per origin kind).**
- G2: MAE of the median (Bcf), and QS = mean pinball over levels 0.1..0.9.
- G1: accuracy of sign(median - avg5) against sign(dev), with dev = 0 excluded.
- G3: pooled regional accuracy.
- cov80.

**Selection score** (pre-declared; lower is better):

`S = mean over {thu, fri} of ( MAE / MAE_ref + QS / QS_ref ) / 2`

`ref` is `ref_ols_wx`, the prereg's `st_ols_wx` (dev ~ 1 + HDD anomaly + CDD anomaly + dev_{W-1}, per series and
kind, residual quantiles) **refit per fold**. It is therefore out-of-sample, unlike the harness DEV row of
st_ols_wx, which is fitted on all DEV weeks. G1 accuracy is reported but is not part of S: on about 313 weeks it is
too noisy to rank close models.

**References (same folds; not counted in K):**
- ref_ols_wx (the bar);
- ref_ols_seas (adds anomaly x season interactions);
- ref_ridge_daily (ridge on the 7 daily anomalies + season interactions + the WM's encoder vector, i.e. the
  strongest linear user of the WM's inputs);
- ref_persistence and ref_climatology (the harness baselines; no fit).

**Search loop.** Seed 0 for every exploratory row, fixed budget (1200 AdamW steps, batch 64, OneCycle lr 3e-3,
wd 1e-4, 2 CPU threads, nice 10). Every run, crashes included, is appended to `runs/gas/search/results.tsv`.
Every WM row gets its matched direct-twin row (same cfg with arch=direct, same seed); twins count in K.
1. R1, core grid: assim in {0, 1} x regime in {0, 1}, at d = 32.
2. R2: single-factor changes around the best R1 WM by S: d in {16, 64}, steps in {600, 2400}, aux in {0, 1.0},
   wl48 = 1, drop_hh = 1. A change is kept as the new incumbent only if S improves by > 0.01 (smaller is
   treated as seed noise).
3. R3: Monte Carlo variant (mc = 1, 200 weather paths) of the incumbent.
4. Cap: about 40 scored rows. Family 4 (stochastic-vol WM for the vol-direction secondary) only if time allows.

**Finalists and freeze.**
1. The 3 best WM recipes by seed-0 S are re-run with seeds 1 and 2, and so are their twins. The decision uses the
   S of the 3-seed ensemble: quantiles averaged per level, then re-sorted.
2. The frozen candidate `wm_gas` is the 3-seed ensemble of the best WM recipe. The twin `wm_gas_direct` is the
   3-seed ensemble of the same recipe with arch=direct. The candidate is the best world model whether or not it
   beats its twin or the references; that comparison is reported, not optimised.
3. WM vs twin on DEV:
   - paired weekly |error| difference of the two ensembles (L48, each origin kind), with DM-HLN p and a 95%
     bootstrap CI (weeks resampled, 2000 draws, seed 0);
   - the same for QS;
   - exact McNemar for G1.
4. Frozen checkpoints = the chosen recipe refit with the same steps and seeds on ALL train weeks (released
   2014-01-15 .. 2025-06-30). No selection happens on the refit. Its harness-DEV score is in-sample and reported as
   a sanity check only; the DEV evidence for the recipe is the rolling backtest. Each model's 3 seeds are stored in
   ONE checkpoint file, so `meta = {ckpt, ckpt_sha256}` lets the harness preflight cross-check it against
   FROZEN.json. The wrapper also verifies the sha256 at load.
5. `runs/gas/FROZEN.json` records the checkpoints and sha256, the recipe, K, the winner's-curse note and the VAL
   go/no-go rule. It is committed with the checkpoints and wrappers (`git add -f`). The freeze time is written as
   "see the freeze commit time", with no hand-written timestamp. After that commit: VAL exactly once via
   `tools/gas/run_eval.py --task storage --split val --models wm_gas,wm_gas_direct`, results appended below.
   TEST is never touched by the architect.

## 1. Architecture (`tools/gas/search/nets.py`)

The daily storage-balance state-space world model (`GasNet`) has these parts:
- **Encoder:** MLP(released history: 8 dev lags x 6 series, avg5, storage-level deficit vs the 5 previous years,
  4 lagged weekly HDD/CDD anomalies, day of year, HH log level and 20-day change, origin kind) -> latent z0.
- **Transition:** GRUCell, one step per DAY, driven by that day's HDD, CDD, causal normals, anomalies,
  observed/forecast flag and lead, day of week, day of year and holiday.
- **Assimilation (assim = 1):** the model first rolls week W-1 (observed weather) and sums its regional flows. The
  innovation (released dS_{W-1} - model's dS_{W-1}) / sd is written into the state, and then week W is rolled.
- **Readout:** daily regional net injection
  `a5_r / 7 + sd_r / 7 * base_r(z) - sum aH(z) [phi(HDD) - phi(HDDn)] - sum aC(z) [phi(CDD) - phi(CDDn)]`,
  with aH, aC >= 0 and a monotone basis. Demand is therefore monotone in HDD and CDD, and the flow equals the
  5-year average plus a state-dependent base under normal weather.
- **Aggregation:** weekly dS_r = sum of the daily flows; L48 = sum of the 5 regions.
- **Quantiles:** monotone offsets from the final state.
- **regime = 1:** base and coefficients are a softmax mixture of 3 regimes, with the gate computed from the state
  and the day.
- **mc = 1:** Monte Carlo rollouts with CPC forecast-error weather paths (fold-estimated sd by lead).
- **Direct twin (arch = direct):** identical modules and parameter count. Each day is ONE step of the same cell
  from the origin state, with no state carried between days. The assimilation innovation is computed the same way.

Honest prior (research §5): at these origins, 5-6 of the 7 target days are already observed. With a near-linear
readout the WM may reduce to the twin, and any WM gap should be small.

## Amendments to section 0 (each dated; written before the rows it governs were scored)

**Am1 (2026-10-31, after R1, before R2).**
- **Finding 1, collapsed quantiles.** In every R1 network row the learned quantile spreads collapsed: cov80 was
  0.02-0.05, against 0.78 for ref_ols_wx. About 20k parameters fitted to about 1,000 contexts overfit the
  in-sample residuals, so QS, and therefore S, are dominated by miscalibration. MAE is close to the bar: best WM
  21.0/20.9 vs ref 20.8/20.7.
- **Change 1, holdout calibration.** New option `cal = 1`. Inside each fold, a model trained on the first 80% of
  the fold's training weeks scores the last 20%, a time-ordered holdout. Those holdout median residuals give the
  empirical quantiles per series and origin kind. The model is then retrained on 100% of the fold's training
  weeks (same seed and steps), and its quantiles are its median plus those residual quantiles. All of this uses
  only pre-fold data; the same rule applies to the twin and to the final refit.
- **Change 2, regularisation knobs.** New options `wd` and `drop` (dropout in the encoder). The default is
  `drop = 0`, which leaves R1 models bit-identical: the layer is inserted only when drop > 0.
- **R2 (replaces the R2 list).** The incumbent is r1_wm_a1_r1 (assim = 1, regime = 1), the best R1 WM by S
  (1.1466; r1_wm_a1_r0 1.1484 is within the 0.01 noise band). Rows:
  - the incumbent with cal = 1, and its twin;
  - then single changes, all with cal = 1: (wd 0.01, drop 0.2), d = 16, steps = 600, aux = 1.0.
- **Compute.** Because cal doubles the cost of a run (about 4-5 min), finalists are the **top 2** WM recipes, not
  3. Each finalist gets seeds 1 and 2; the chosen recipe's twin gets seeds 0-2. MC (R3) runs only if time allows.
- **Finding 2, holiday-week origins** (reported to the lead, who disclosed it as prereg A2). 60 of the 313 DEV
  weeks are pre-2025 holiday weeks with the conservative F+10 release rule. For those weeks both origins fall
  after the week ends, so they are pure observed nowcasts. VAL and TEST have none. The selection score S stays as
  committed (all DEV weeks). The final DEV table also reports metrics excluding those weeks.
- **Discarded attempt.** A first reference run (2 reference rows, never looked at) was discarded. Its cache stored
  obs_daily as float32, unlike the harness's float64; the cache was rebuilt in float64 and all references re-run.
  The search itself had not started.

**Am2 (2026-10-31 ~22:30, after r2_wm_reg, before R2b).**
- **Result.** r2_wm_reg (wd 0.01, drop 0.2) improved S from 1.052 to 0.828. That is far beyond the 0.01 keep
  band, so it becomes the incumbent.
- **Rows stopped.** The remaining queued R2 rows were single changes from the old incumbent: d16 was already
  running and is logged; s600 and aux1 were never run.
- **R2b (single changes from the new incumbent):** the incumbent's twin, a stronger regulariser (wd 0.05,
  drop 0.3), regime 0 and d 16.
- **After R2b:** finalists = top 2 WM recipes by seed-0 S (Am1). No further exploration rows.

## 2. Experiment log (all rows in `runs/gas/search/results.tsv`; logs in `runs/gas/search/logs/`; predictions in `runs/gas/search/preds/`)

Rolling DEV backtest: 313 weeks per origin kind, released 2019-07-01 .. 2025-06-30. "S" is the pre-declared score
(lower is better; ref_ols_wx = 1). MAE and QS are L48 Bcf. Acc is G1, accreg is G3 pooled.

| round | rows | finding |
|---|---|---|
| refs | 5 | ref_ols_wx: MAE 20.8/20.7, acc 0.738/0.744. ref_ols_seas: S 0.948. ref_ridge_daily: S 1.025, but cov80 0.51/0.54. persistence: 1.78. climatology: 1.56. |
| R1 | 8 (4 WM + 4 twins) | Quantiles collapsed (cov80 about 0.03): overfitting. MAE about 21. Best WM: a1_r1, S 1.147. Twins with assim were much worse (S 1.35-1.46); dir_a0_r1 reached S 1.135. |
| R2 | 4 | With holdout calibration (cal), the incumbent reached S 1.052 and its twin 1.222. **Regularisation (wd 0.01, drop 0.2): S 0.828**, MAE 17.1, acc 0.818. d16 without regularisation: S 1.333. |
| R2b | 4 | Regularised twin: 0.824. Stronger regularisation (wd 0.05, drop 0.3): 0.827. regime 0: 0.838. d16: 0.842. |
| finalists | 4 + 3 | Seed variance is large: single-seed S ranges 0.79-0.88, so the seed sd (about 0.03) is far above the 0.01 keep band. 3-seed ensemble S: A (wd 0.01, drop 0.2) **0.8031**; B (wd 0.05, drop 0.3) **0.7966** -> B chosen by the rule. The twin of B, seeds 0-2, gives an ensemble S of 0.7765. |

K:
- 23 scored network runs (this count excludes the 5 reference rows and the 2 discarded, unseen reference rows).
- 17 distinct configurations: 10 WM recipes and 7 twin recipes.
- 0 crashes.

**Winner's curse.**
- The best single WM row was f_reg2_s2, S 0.7915. The chosen recipe's members score 0.8267, 0.8583 and 0.7915,
  mean 0.826.
- The choice between A and B (0.8031 vs 0.7966) lies inside seed noise.
- The DEV numbers of the candidate are optimistic: it is the maximum over about 10 WM recipes, and the
  architecture and regulariser were adapted on these same DEV weeks (6 yearly folds). The linear references were
  not tuned at all.

## 3. Selection and freeze (all numbers are from the DEV rolling backtest, before any VAL run by the architect)

**Rule outcome.** The finalists were A (wd 0.01, drop 0.2) and B (wd 0.05, drop 0.3), both WM with assim=1,
regime=1, cal=1. B has the better 3-seed ensemble S (0.7966 vs 0.8031), so **B is frozen.**
- `wm_gas`: B with seeds 0,1,2, refit on all 547 train weeks.
- `wm_gas_direct`: the twin (arch=direct), with the same refit.

Checkpoints and sha256 values are in `runs/gas/FROZEN.json` and `tools/gas/wm_gas.py`. There are 23,534
parameters per member for both models, and the only config difference is `arch`.

DEV table (313 weeks per origin; L48; ensembles of 3 seeds; references refit per fold):

| model | MAE thu / fri | QS thu / fri | G1 acc thu / fri | G3 acc thu / fri | cov80 thu / fri | S |
|---|---|---|---|---|---|---|
| **wm_gas (WM ensemble)** | 16.59 / 16.53 | 6.39 / 6.37 | 0.812 / 0.815 | 0.736 / 0.733 | 0.78 / 0.79 | 0.797 |
| wm_gas_direct (twin ensemble) | 16.17 / 16.08 | 6.25 / 6.20 | 0.844 / 0.834 | 0.743 / 0.737 | 0.81 / 0.81 | 0.777 |
| ref_ols_wx (the prereg bar, out-of-sample) | 20.78 / 20.69 | 8.03 / 8.03 | 0.738 / 0.744 | 0.684 / 0.690 | 0.79 / 0.78 | 1.000 |
| ref_ols_seas | 19.64 / 19.68 | 7.60 / 7.62 | 0.741 / 0.728 | 0.686 / 0.679 | 0.78 / 0.77 | 0.948 |
| ref_ridge_daily | 20.97 / 20.68 | 8.47 / 8.31 | 0.767 / 0.773 | 0.703 / 0.707 | 0.51 / 0.54 | 1.025 |
| ref_persistence | 35.80 | 14.78 | 0.633 | 0.627 | 0.80 | 1.784 |
| ref_climatology | 32.41 | 12.51 | 0.489 | 0.469 | 0.70 | 1.561 |

Excluding the 60 holiday-shifted weeks (prereg A2), 253 weeks remain:
- wm_gas: MAE 16.46/16.42, acc 0.814/0.814.
- ref_ols_wx: MAE 20.85/20.81, acc 0.727/0.731.

WM ensemble minus a competitor, per week (DM-HLN p; 95% bootstrap CI over weeks):

| comparison | origin | dMAE (Bcf) | dQS | G1 exact McNemar (WM-only / other-only, p) |
|---|---|---|---|---|
| WM - twin | thu | +0.42 [-0.49, 1.28], p 0.34 | +0.14 [-0.17, 0.44], p 0.36 | 6 / 16, p 0.052 |
| WM - twin | fri | +0.45 [-0.43, 1.32], p 0.31 | +0.17 [-0.15, 0.47], p 0.28 | 8 / 14, p 0.29 |
| WM - ref_ols_wx | thu | -4.19 [-6.06, -2.46], p < 1e-4 | -1.64, p < 1e-4 | 52 / 29, p 0.014 |
| WM - ref_ols_wx | fri | -4.16 [-5.99, -2.37], p < 1e-4 | -1.65, p < 1e-4 | 51 / 29, p 0.018 |
| WM - ref_ols_seas | thu | -3.05 [-4.78, -1.41], p 0.0004 | -1.21, p 0.0004 | 51 / 29, p 0.018 |
| WM - ref_ols_seas | fri | -3.16 [-4.91, -1.48], p 0.0003 | -1.25, p 0.0003 | 52 / 25, p 0.003 |

**Reading.**
- **The world model does NOT beat its direct twin on DEV.** The twin is numerically better on every endpoint,
  though not significantly. That matches the prior: 5-7 of the 7 target days are already observed at these
  origins, so there is little for a day-by-day rollout to compose.
- The gain over the linear bar comes from the shared parts that both networks have:
  - the monotone regional demand readout;
  - the regularised encoder of the released history;
  - the holdout calibration.
  It does not come from state rollout.
- These DEV p-values are optimistic: about 17 configurations were tried on the same folds (see winner's curse,
  section 2).

**Harness DEV sanity check (in-sample).** `run_eval --split dev --models wm_gas,wm_gas_direct` on the refit models
gives MAE 6.6/6.2 and acc 0.955/0.949, with no non-finite values. It is in-sample, because the refit uses all DEV
weeks, so it is a plumbing check only.

**Leak test.** `tools/gas/search/test_leak.py` gives ALL OK. It covers 9 DEV weeks x 2 kinds, all 8 arch variants,
and the frozen wrappers:
- Forbidden inputs changed 0 forecasts: storage of week W and later, observed weather from the origin on,
  unusable CPC issues, and HH after the origin.
- Allowed inputs changed all forecasts: dS_{W-1}, the latest allowed observed day, the used CPC issue, and the HH
  price of the origin day.

**Freeze integrity.** The freeze time is the commit time of the commit adding `runs/gas/FROZEN.json`; no
hand-written timestamps are used. The VAL go/no-go rule is in FROZEN.json.

**Deviation to disclose.** The swap was 4.7-6.3 GB during training, above the 4.5 GB guideline, because of other
processes. Memory pressure stayed normal (34-40% free), and only one small job (about 300 MB) ran at a time, so I
proceeded rather than wait.

## 4. VAL (run once, after the freeze commit 7678c65)

Command: `run_eval.py --task storage --split val --models wm_gas,wm_gas_direct` (commit 7678c65, scored
23:41:41). Window: 18 weeks released 2025-07-01 .. 2025-10-31 (17 G1 labels). Report: `runs/gas/val/report.md`.

| model | G1 acc thu / fri | MAE thu / fri | QS thu / fri | cov80 | G3 pooled thu / fri |
|---|---|---|---|---|---|
| wm_gas | 0.882 / 0.882 (15/17) | 11.0 / 11.0 | 4.43 / 4.38 | 0.83 / 0.83 | 0.807 / 0.807 |
| wm_gas_direct | 0.765 / 0.824 | 12.1 / 12.2 | 4.88 / 4.83 | 0.89 / 0.89 | 0.739 / 0.773 |
| st_ols_wx (harness) | 0.588 / 0.588 | 16.2 / 16.4 | 6.05 / 6.06 | 0.89 | 0.670 / 0.682 |
| tirex (best FM) | 0.824 / 0.824 | 14.5 / 14.5 | 5.67 / 5.67 | 1.00 | 0.761 / 0.761 |

WM minus twin on VAL (L48 |error|):
- thu: -1.11 Bcf, bootstrap 95% CI [-2.67, 0.34], DM-HLN p 0.18.
- fri: -1.22 Bcf, [-2.71, 0.17], p 0.13.

The sign is the opposite of DEV and it is not significant. 18 weeks is far too few for claims (prereg §6), and
McNemar vs st_ols_wx gives p = 0.125.

**Go/no-go (rule in FROZEN.json): GO.**
- All quantiles are finite.
- MAE 11.0 <= 1.5 x 16.2 / 16.4.
- cov80 0.83 lies in [0.5, 1.0].

No selection or change follows VAL.

## 5. Cold-snap scenario demo (simulation only, not scored): `tools/gas/wm_scenario.py` -> `runs/gas/scenario_demo.json`

**Setup.**
- Model: wm_gas, 3 members, 1,000 paths per arm.
- Origin: thu. Week: the DEV winter week (Dec 2024 - Feb 2025) whose baseline median deviation is closest to
  zero, which turned out to be the week ending 2025-02-28.
  - That rule was fixed after a first run on the week ending 2025-01-10, which was already extremely cold
    (P = 1.0 in both arms, so uninformative). This is disclosed.
- Shock: +2 sd of the daily HDD anomaly, i.e. +10.5 HDD/day on all 7 days.

**Results.**

| arm | dS_W median (p10, p90) | dev vs avg5 | P(draw > 5-yr average draw) | L48 level on Fri (from 1,840) |
|---|---|---|---|---|
| baseline | -75 (-96, -54) | -5 | 0.61 | 1,765 |
| cold snap | -218 (-240, -197) | -148 | 1.00 | 1,622 |

Limitations (also in the JSON):
- Only degree days are shocked. Supply, LNG and price feedback are not modelled.
- The +2 sd response is the learned monotone sensitivity, possibly extrapolated (about 1.8 Bcf per weekly HDD).
- Daily flows are latent and unvalidated, because EIA publishes weekly totals only.
- Uncertainty covers CPC forecast error + quantile spread + 3 seeds only.
- The direct twin can be shocked the same way. The WM adds a daily state path, not what-if ability per se.
