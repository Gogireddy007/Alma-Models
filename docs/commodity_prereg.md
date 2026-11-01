# Commodity (crude oil) forecasting: pre-registration

Author: commodity lead (agent), 2026-10-31. Branch `alma/investor-sprint`.
Status: written and committed BEFORE any TEST-window forecast or score was computed. At the time of writing, only
smoke forecasts on 3 validation-window contexts (2025-07-01 .. 2025-07-30) had been produced, to check that each model
loads; no score of any kind had been computed on any window.
Any change after this commit is a dated amendment at the bottom of this file, made before the TEST run, with its
reason. Harness: `tools/commodity/` (interface in `tools/commodity/README.md`).

## 1. Data

| name | source | unit | role |
|---|---|---|---|
| wti | FRED DCOILWTICO (WTI Cushing spot) | USD/bbl, business days | PRIMARY |
| brent | FRED DCOILBRENTEU (Brent Europe spot) | USD/bbl | secondary |
| hh | FRED DHHNGSP (Henry Hub spot) | USD/MMBtu | secondary |
| inv | EIA WCESTUS1 (US commercial crude stocks excl. SPR), weekly | thousand bbl | T3 (not on FRED; taken from eia.gov LeafHandler table) |

Raw files, sha256 and download date (2026-10-31): `data/commodity/manifest.json`. Days without a value (FRED ".")
are dropped (no fill). Spot series are taken as published on the download date; FRED spot prices are not revised in
practice, but this is not a vintage (real-time) dataset, which is disclosed. Weekly inventories can be revised by EIA;
we use the current vintage (disclosed limitation).

## 2. Splits

- TRAIN: all observations <= 2025-06-30. Only source of any fitting (statistical models fit on each context window
  only; GARCH for T2 fitted once on TRAIN; majority classes from TRAIN; any Alma fine-tuning through
  `data.train_windows`, which asserts no target after 2025-06-30).
- DEV backtest (model selection, architecture search): origins 2019-01-01 .. 2025-06-30, every 5th trading day
  (weekly for inv); only origin/horizon pairs whose target dates (and T2 future window) are <= 2025-06-30 are scored.
- VALIDATION: origins 2025-07-01 .. 2025-10-31, every trading day (every week for inv). Used for debugging and to
  confirm ONE frozen candidate (plus its direct twin). Only pairs whose target dates are <= 2025-10-31 are scored, so
  no TEST-period value enters any validation score.
- TEST: origins 2025-11-01 .. the last date in the downloaded data (2026-10-27 for the daily series, 2026-10-23 for
  inv), every trading day (every week for inv). A pair is scored if its target date is in the data.
  TEST is run exactly ONCE, all models in one run, after the candidate is frozen (section 8).
- Origin semantics: an origin is a date in the series; the model sees exactly the observations with date strictly
  before it (the last 1024 of them). Step k of the forecast is the k-th observation from the origin on.
  `P[t]` = last observed value; `P[t+h]` = the h-th future observation.

Model release dates vs the windows (training-data contamination risk):
Chronos-Bolt Nov 2024, TiRex May 2025, Kronos Aug 2025 (pre-trained on exchange K-line data; may include commodity
futures up to an undisclosed cutoff), Chronos-2 Oct 2025 (its training data may cover part of VALIDATION, not TEST),
Timestral 2026-10-15 (its training data MAY COVER THE TEST PERIOD: reported, marked "contaminated", and excluded from
any "beats all models" claim). Alma's training data: `docs/commodity_leakage.md`.

## 3. Targets

- T1 price direction, h in {1, 5, 20} trading days: y = sign(P[t+h] - P[t]). Pairs with P[t+h] == P[t] are excluded
  (count reported). PRIMARY on wti; brent, hh secondary.
- T2 volatility direction (wti primary; brent, hh secondary): y = 1 if the std of the 20 daily log returns over
  t+1..t+20 exceeds the std of the 20 daily log returns ending at t, else 0.
- T3 inventory direction (inv, weekly), h in {1, 4} weeks: y = sign(I[t+h] - I[t]).
- T4 probabilistic quality of log price, h in {1, 5, 20} (wti primary; brent, hh secondary): per origin, mean pinball
  loss over the 9 levels 0.1..0.9 of log(quantile) vs log(P[t+h]) ("QS"; 2 x QS approximates CRPS on that grid).
  Reported as skill vs the Gaussian random walk: 1 - mean QS(model) / mean QS(rw_gauss). Coverage of the central 80%
  interval [q0.1, q0.9] for every model; 90% [q0.05, q0.95] and 95% [q0.025, q0.975] for models that natively output
  those levels (others: N/A). Quantiles <= 0 are floored at 0.01 before the log (all models, all windows; DEV only:
  pairs whose target is <= 0 are dropped, which only affects April 2020).

## 4. Decision rules (fixed; no threshold is fitted on any window)

- Direction from a probabilistic model (T1, T3): sign(median_h - P[t]). If median_h == P[t] exactly, predict the TRAIN
  majority class.
- Volatility direction from a probabilistic model (T2): sigma_model = mean over k = 1..20 of
  (log q0.9_k - log q0.1_k) / (2 * 1.2816 * sqrt(k)); predict "higher" iff sigma_model > sigma_past (std of the last 20
  daily log returns of the context).

## 5. Models and fixed inference settings

All inference on CPU, `nice -n 10`. Every model gets the last 1024 observations of raw levels (it may crop).
Quantile levels requested: 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975 (NaN where not native).

| name | settings |
|---|---|
| alma_s0 (PRIMARY Alma) | `data/ckpt/wm_ssm_none_s0.pt`, frozen, zero-shot, train_wm.py eval-only light path, context crop 256, seasonality 5 (daily, freq B) / 52 (weekly), TTA off ("raw"), levels 0.1..0.9 |
| alma_s0_tta | same, TSAR_WM_TTA=1 (sym + clip), the "TTA" number required by docs/alma_goal.md |
| alma_direct_s0 | `data/ckpt/wm_ssm_direct_s0.pt` (direct twin), same settings |
| alma_s3, alma_direct_s3 | second paired seed; alma_s4 (+ alma_direct_s4 if that checkpoint exists when TEST runs); alma_s1, alma_s2 descriptive (no direct twin) |
| architect candidate (+ its direct twin) | frozen per section 8; name, checkpoint sha256 recorded in `runs/commodity/FROZEN.json` |
| chronos2 | amazon/chronos-2, context 1024, all 13 levels (interpolated by the pipeline inside its native grid) |
| chronos_bolt_base | amazon/chronos-bolt-base, context 1024, levels 0.1..0.9 |
| tirex | NX-AI/TiRex, context 1024, levels 0.1..0.9 |
| timestral_5m | mistralai/Timestral-5M, default forecaster options, context 1024, its native levels (CONTAMINATED for TEST) |
| kronos_small | NeoQuasar/Kronos-small + Kronos-Tokenizer-base, official code @67b630e, open=high=low=close=price, volume 0, context 512, 20 samples, T=1, top_p=0.9, seed 0, empirical quantiles; wti only (daily price model; not run on inv). If it is too slow for all origins, it is run on every 5th origin of each window (decided from validation timing only, recorded as an amendment before TEST) |
| auto_ets, auto_arima | statsforecast, last 512 observations, season_length 1, intervals at 20/40/60/80/90/95% |
| rw_gauss | driftless Gaussian random walk in log price, sigma from the last 250 log returns |
| majority | TRAIN majority class per target and horizon (direction-only baseline) |
| momentum | T1/T3: sign(P[t] - P[t-h]) (direction-only baseline) |
| T2 baselines | majority; vol_meanrev: "higher" iff sigma_past < median of the 20-day rolling vol over the last 252 days; vol_persist: "higher" iff sigma_past > std of the 20 returns before those (trend continuation); garch: GARCH(1,1) Gaussian fitted once on TRAIN daily log returns 2000-01-01..2025-06-30 (returns touching a non-positive price dropped), filtered forward, "higher" iff mean predicted variance over t+1..t+20 > sigma_past^2 |
| T3 baselines | majority; momentum (sign of the last h-week change); seasonal: sign of the mean of the same-horizon changes at t-52k, k = 1..5 |

## 6. Metrics and tests

- Directional accuracy with Wilson 95% CI; exact two-sided binomial test vs 0.5; Pesaran-Timmermann test (one-sided,
  H1: predictive). For h > 1 origins overlap, so the iid CIs/PT are optimistic; we also report accuracy on the
  non-overlapping subsample (every h-th origin).
- Model vs baseline (T1-T3): Diebold-Mariano on the 0/1 loss differential with the Harvey-Leybourne-Newbold correction
  and a Newey-West (Bartlett) HAC variance with lag h-1 (T2: lag 19), two-sided.
- T4: DM-HLN on per-origin QS differences vs rw_gauss, HAC lag h-1, two-sided.
- Holm correction within each (series, target) family across all model x baseline x horizon comparisons. Primary
  families: wti T1, wti T2, inv T3, wti T4. Brent/hh are secondary (Holm within each, reported separately).
- Base rates and the always-majority accuracy are printed next to every accuracy.

## 7. Primary metric per target and the "beats all models" criterion

| target | primary metric | series |
|---|---|---|
| T1 | directional accuracy, h = 1, 5, 20 (each reported; no pooling) | wti |
| T2 | directional accuracy | wti |
| T3 | directional accuracy, h = 1, 4 | inv |
| T4 | QS skill vs rw_gauss (equivalently CRPS/WQL ratio), h = 1, 5, 20 | wti |

"Alma beats model X on target T at horizon h" is claimed only if Alma's primary metric is better AND the two-sided
DM-HLN test of Alma vs X is significant at 5% after Holm over all competitor models for that (T, h). "Beats all models"
requires that against every non-contaminated model and baseline. Otherwise we say "numerically higher, not
significant" (or "lower"). Every target and horizon is reported, wins and losses alike.

## 8. Model-selection protocol (added on manager instruction before any score; still before any number)

1. Architecture search and selection use only TRAIN fitting and the DEV backtest (`run_eval.py --split dev`).
2. VALIDATION confirms the single frozen candidate (plus its direct twin); no further selection after it except the
   documented go/no-go to include it.
3. Freeze: `runs/commodity/FROZEN.json` lists the candidate name, checkpoint path and sha256, code commit. The TEST run
   refuses to start without it.
4. TEST runs exactly once with all models of section 5 together; `runs/commodity/test/RUN.json` records the commit
   sha, timestamp, model metas and checkpoint shas; a second TEST run is refused by the harness. If a bug is found
   after the TEST run, the fix and the re-run are both reported, with the original numbers kept.

## 9. World-model test

Rollout (`alma_sK`) vs its direct twin (`alma_direct_sK`), paired seeds K in {0, 3} (+4 if `wm_ssm_direct_s4.pt`
exists at TEST time; seeds 1, 2 have no twin). Endpoints: T4 QS on wti at h = 1, 5, 20 (DM-HLN, Holm over horizons x
seeds) and T1 accuracy on wti (DM on 0/1 loss). Outcome stated as: rollout better / worse / no significant difference,
per seed, and whether the sign is consistent across seeds. Same for the architect's candidate vs its direct twin.
Note (docs/audit_sprint.md 3c): the original seed choice (0, 3, 4) for the GIFT WM test was flagged as favouring the
rollout; here every paired seed that exists is used.

## Amendments

- A1 (2026-10-31 ~19:00, before any score on any window; reason: speed only). A Kronos-small smoke run with the
  section 5 settings (20 samples, context 512) took 1117 s for 3 validation contexts (~372 s per origin on this
  shared CPU), infeasible for ~250 TEST origins. Kronos settings are therefore: 10 samples, context 256, every 5th
  origin of VALIDATION and TEST (wti only); its comparisons use the common origins. Decided from wall time only.
- A2 (2026-10-31 ~19:45, audit B1; before any TEST computation). World-model test pairs: **s0 only**
  (`alma_s0` vs `alma_direct_s0`, and the same pair with TTA). `data/ckpt/wm_ssm_direct_s3.pt` is a half-trained
  checkpoint from a stopped run (step 565 of 2500, complete=False; its sha256 also changed during VAL) and
  `wm_ssm_direct_s4.pt` does not exist, so both are dropped; the VAL s3-pair numbers are void (forecasts deleted).
  `alma_s1..s4` rollouts are reported as descriptive seed spread only. `models._alma` refuses any checkpoint with
  complete != True or step != budget steps. Every Alma checkpoint used at TEST is pinned by sha256 in
  `run_eval.ALMA_PINS` (and `FROZEN.json["alma_sha256"]`, if present, must match); the TEST run verifies them.
  Disclosure (docs/audit_sprint.md): the s3 checkpoints were trained while the working tree was changing.
  TEST model list: alma_s0, alma_s0_tta, alma_direct_s0, alma_direct_s0_tta, alma_s1, alma_s2, alma_s3, alma_s4,
  chronos2, chronos_bolt_base, tirex, timestral_5m, kronos_small, auto_ets, auto_arima, rw_gauss, plus the frozen
  architect members in FROZEN.json.
- A3 (2026-10-31 ~19:45, audit B4; before any TEST computation). Overlap-robust statistics replace section 6 for
  inference. The auditor's Monte Carlo (and ours: DM-HLN with Bartlett lag h-1/2h/3h rejects 9%/8.5%/8% at h=5 and
  13.5%/11.5%/10.5% at h=20 under the null, n=230) shows the iid tests and the HAC DM over-reject for h>1.
  - Accuracy: point estimate on all origins, with a circular block-bootstrap 95% CI (block 2h; 40 for T2; 2000
    draws, seed 0). Inference on the non-overlapping subsample (origins whose position in the window is divisible by
    h; by 40 for T2): Wilson 95% CI, exact two-sided binomial test vs 0.5, Pesaran-Timmermann. (Our Monte Carlo:
    non-overlap binomial size 0.024/0.041 at h=5/20, Wilson coverage 0.95/0.96.)
  - Model vs model/baseline (T1, T2, T3): h=1: DM-HLN on the 0/1 loss (unchanged). h>1 and T2: exact McNemar on the
    non-overlapping subsample (size 0.032/0.004 at h=5/20 in our Monte Carlo). T4 vs RW and vs models: h=1: DM-HLN;
    h>1: DM-HLN (lag 0) on the non-overlapping subsample. DM-HLN with Bartlett lag 2h (40 for T2) is reported as
    supplementary only. Holm families as in section 6/7, applied to these primary p-values; every "beats" claim uses
    only them.
  - Declared in advance: on TEST, h=20 rests on ~12 independent windows and T2 on ~6, so T1 h=20, T4 h=20 and T2
    are DESCRIPTIVE unless the overlap-robust test passes (which is unlikely by construction). h=5 rests on ~49.
  - Kronos is reported as an every-5th-origin subsample (its own accuracy/CI rest on n/5; pairwise tests use the
    common origins). Sample-based models are seeded per origin (Kronos: seed 0 before each origin).
- A4 (2026-10-31 ~19:45, audit B2/B3). TEST is run in strict mode: refused unless FROZEN.json exists, the output
  directory holds no forecast file, `tools/commodity` and this file are committed (clean tree), every model resolves
  and every Alma checkpoint matches its pin (preflight, before RUN.json is written). Any load or forecast failure
  aborts the run; RUN.json then records status=failed (the TEST is still consumed). On success RUN.json gets
  status=complete and `test/hashes.json` (sha256 of every TEST forecast file and metrics.json). The exact command
  (`run_eval.py --split test`) was rehearsed end to end on VAL (`--split val --rehearse`, fresh directory
  `runs/commodity/rehearsal_val/`). FROZEN.json, RUN.json and hashes.json are force-added to git after the run.
