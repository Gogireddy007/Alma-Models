# Commodity world model, round 2: pre-registration

Author: round-2 commodity world-model architect (agent), 2026-10-31 ~20:10 ET. Branch `alma/investor-sprint`.
Status: written and committed BEFORE any DEV run was scored. At commit time only smoke tests had run (3 training
steps per model and a 256-origin forecast, to time the code; no metric of any kind was computed). Any later change
is a dated amendment at the bottom, with the number of DEV runs scored by then.
Code: `tools/commodity/r2/` (data_r2.py, core_r2.py, models_r2.py, train_r2.py, score_r2.py, val_r2.py, test_r2.py).
Outputs: `runs/commodity/r2/` (results.tsv = every run incl. failures; dev_scores.json/md; FROZEN.json).
The round-1 protocol (`docs/commodity_prereg.md`, amendment A3) and auditor rules (`docs/audit_commodity.md`) apply
unless overridden here. The round-1 TEST for WTI/Brent/Henry Hub was running while this was written; its results are
not read by round 2 (`runs/commodity/test/` is never opened by this code).

## 0. Scope change from the brief (manager deep-research note, received before any scoring)

The brief's grid (8 products x h in {1,5,10,20} x {price direction, vol direction, CRPS}) is narrowed BEFORE any
score, on the manager's instruction: price direction is dropped (literature and round 1: noise at 1-20 days).
- PRIMARY target: volatility direction at h = 5 and h = 10, POOLED over five refined products/NGL (POOL5):
  NY Harbor No.2 heating oil DHOILNYH, US Gulf Coast jet fuel DJFUELUSGULF, NY Harbor gasoline DGASNYH, US Gulf Coast
  gasoline DGASUSGULF, Mont Belvieu propane DPROPANEMBTX.
- SECONDARY target: probabilistic 5- and 10-day log-return range (CRPS, plus 80% / 95% coverage) for DHOILNYH and
  DJFUELUSGULF.
- NY Harbor / Gulf ULSD (DDFUELNYH, DDFUELUSGULF) and LA ULSD (DDFUELLA) are verified daily FRED series but are not
  in POOL5 (manager's panel); they are not used.

## 1. Data and splits

FRED daily spot series (EIA source), fetched with `coed=` in the request, re-cut, cached per split
(`runs/commodity/r2/data/<split>/`), max date asserted in the loader (`data_r2._guard`). Missing days (FRED ".")
are DROPPED, never filled; each series lives on its own trading calendar. FRED/EIA publish these series with a
weekly lag; this is an end-of-day backtest (disclosed). Spot prices are not revised in practice; not a vintage
dataset (disclosed). Propane has ~10% zero daily returns (stale prints), disclosed.
- TRAIN: every training target dated < 2015-01-01 (asserted in `Series.train_origins`). GARCH, GJR-GARCH-t and HAR
  parameters are also fitted on returns dated < 2015-01-01 only.
- DEV backtest (all search and selection): origin dates 2015-01-01 .. 2025-06-30 = the intersection calendar of the
  five POOL5 series (2619 dates), every date. A (series, origin, h) is scored only if its h-th future value is in the
  DEV data (<= 2025-06-30). The search models are NOT refitted inside the backtest; the frozen checkpoints are the
  backtest checkpoints themselves (trained on data < 2015-01-01). No refit is planned.
- VALIDATION: origin dates 2025-07-01 .. 2025-10-31, scored only where the target is <= 2025-10-31. Used ONCE, after
  the freeze, to confirm the frozen choice (plus twin, baselines, foundation models). No selection after it.
- TEST: origin dates >= 2025-11-01 for POOL5. Never loaded by the architect. Run once, later, by someone else with
  `tools/commodity/r2/test_r2.py` (written, never executed by the architect).
- Origin semantics: for series c and origin date D the model sees the observations of c dated strictly before D
  (t = last of them); horizon k = the k-th future observation of c.
- Exogenous inputs (only in the `sv_cal_exo` recipe): WTI DCOILWTICO, Brent DCOILBRENTEU, Henry Hub DHHNGSP daily log
  returns, dates <= 2025-06-30 (DEV) / <= 2025-10-31 (VAL), never as targets. For a product row dated d the input is
  the latest exo return dated STRICTLY before d (lag >= 1 day; as-of join, no filling of values; masked if older than
  5 calendar days). Non-positive prints (WTI 2020-04-20, -36.98) are dropped as missing days. If this recipe were
  chosen, its TEST run would need WTI/Brent/HH data from 2025-11-01 as inputs; that is a decision for the
  manager/auditor and is flagged in FROZEN.json.

## 2. Targets and decision rules (fixed, nothing fitted on DEV)

- vol h: past = RMS of the h daily log returns ending at t; fut = RMS of the h returns t+1..t+h; y = 1[fut > past];
  exact ties excluded. Rule for every quantile model (world models, twins, foundation models, Gaussian/t random-walk
  baselines): sigma_hat = mean_{k=1..h} (q0.9_k - q0.1_k) / (2 * 1.28155 * sqrt(k)) on log-change quantiles;
  predict 1 iff sigma_hat > past. Metric: accuracy pooled over all scoreable (series, origin) pairs.
- crps h: QS = mean pinball loss over the 9 levels 0.1..0.9 of the log change at h (2 QS ~ CRPS); lower is better.
  Coverage of [q0.1, q0.9] and [q0.025, q0.975] (models without the outer levels: N/A).
- Forecast format: quantiles (0.025, 0.1, ..., 0.9, 0.975) of lp[t+k] - lp[t], k = 1..10.

## 3. Baselines (all causal; parameters from data < 2015-01-01)

Vol direction: rw_gauss (sigma = RMS of last 250 returns, i.e. long-run vol), ewma (RiskMetrics lambda 0.94),
garch (GARCH(1,1)-Gaussian term structure), gjr_t (GJR-GARCH(1,1) with Student-t innovations, term structure);
these four through the quantile rule above. har / loghar (HAR-RV of Corsi on squared daily returns: RV_d, RV_w (5),
RV_m (22) -> mean squared return over t+1..t+h; OLS per product and per h; log version in logs), har_pooled /
loghar_pooled (same, one OLS on the stacked five products); rule: predict 1 iff predicted RV_h > past RV_h (log
space for log-HAR). meanrev: predict "down" iff past-h RV > median of the h-day RV over the trailing 252
observations, else "up". majority: TRAIN majority class per product and h.
CRPS: rw_gauss, ewma, garch, gjr_t (Gaussian / scaled-t quantiles of the cumulative change).
Strongest baseline per cell = the best of these on the DEV point metric of that cell (chosen on the same DEV data,
which favours the baseline). MS-GARCH: not implemented (no package offline; time), disclosed as missing.
Calendar features for baselines: not given; instead the gasoline robustness read (section 6) is reported.

## 4. World-model recipes (the architecture axis) and their twins

All world models advance a latent state step by step with a learned transition and forecast by Monte Carlo rollout;
each has a matched direct twin (identical module list and parameter count, same data, same max_steps, same seeds,
same LR schedule), see `models_r2.py`. Pooled univariate training over POOL5 (each window normalised by its own
return RMS), context L = 128, horizon 10, batch 64, AdamW lr 2e-3 cosine, max_steps 600 for every run, 2 CPU threads,
512 Monte Carlo paths at evaluation, deterministic seeding per series and origin chunk.

| recipe | world model | members |
|---|---|---|
| sv_pool | round-1 stochastic-volatility WM (latent log-vol AR state, score-driven correction, Student-t), no extra inputs | 3 seeds (0,1,2), quantile-averaged; twin likewise |
| sv_cal | sv + calendar inputs (Wednesday EIA report day, sin/cos day of year = seasonal vol), known ex ante, also fed in imagination | 3 seeds |
| sv_cal_exo | sv_cal + lagged WTI / Brent / HH daily returns (lag >= 1 day; unknown in imagination, masked) | 3 seeds |
| rs_pool | round-1 regime-switching WM (3 regimes, filtered belief, sampled regimes in imagination) | seed 0 |
| obs_pool | round-1 observer-SSM (deterministic GRU rollout, quantile head) | seed 0 |
| alma_zs | Alma SSM zero-shot (`data/ckpt/wm_ssm_none_s0.pt`) vs its twin `wm_ssm_direct_s0.pt` (round-1 pinned shas), harness settings, levels 0.1..0.9 | if it runs on all DEV origins in < 25 min; otherwise dropped and K reduced (amendment) |

Fine-tuned Alma is not run (round 1: fine-tuning made it worse in every run, and costs ~12 s/step here); disclosed.
Every training run (and any failure) is a row of `runs/commodity/r2/results.tsv`, with a run-level scalar (pooled
vol accuracy, mean of h 5 and 10) used only as a keep/discard guide; no hyper-parameter is tuned on it beyond the
fixed list above (if a run crashes it may be re-run once with the same settings; logged).

## 5. Selection rule (binding)

Cells: vol h=5 and h=10 (pooled POOL5; PRIMARY); crps h=5 and h=10 for DHOILNYH and for DJFUELUSGULF (SECONDARY).
Rows = (recipe, cell); K = number of rows scored (6 recipes x 6 cells = 36 if all run).
For each row:
- margin_base = WM metric minus the strongest baseline's (accuracy points for vol; relative QS skill for crps);
  margin_twin likewise vs the direct twin.
- Overlap-robust one-sided tests (H1: WM better). vol: date-clustered test: keep origin dates whose position is
  divisible by 2h (past and future windows do not overlap), average the 0/1 loss difference over the scoreable
  series of each such date, one-sided DM-HLN with lag 0 (= paired t) on those date means (cross-series correlation is
  absorbed by averaging within a date). crps: one-sided DM-HLN lag 0 on the QS differences at origins whose position
  is divisible by h. Circular block-bootstrap 95% CIs (block 2 x stride) of the margins on all origins, reported.
- p_cand = max(p_base, p_twin) (intersection-union: the WM must beat BOTH); z_min = min of the two one-sided z's.
- Eligible: n_indep (independent dates / windows of the primary test) >= 60.
- Multiplicity: Holm over all K rows on p_cand, reported; and a winner's-curse discount: margin_base minus
  z_{0.05/K} x its standard error (Bonferroni lower bound), reported.
Choice: among eligible PRIMARY rows with both point margins > 0, the one with the largest z_min (ties: larger
margin_base). If none, the same among eligible SECONDARY rows. If none either, the best eligible PRIMARY row by
z_min is frozen anyway and the result is declared NEGATIVE on DEV (a world model did not beat the strongest baseline
and its twin), and that is what goes to VAL/TEST. Holm significance is not required for the choice, but every claim
states the Holm-adjusted p and K.
The frozen comparator set for VAL and TEST: the chosen recipe, its twin, the DEV-strongest baseline (by name) and
every baseline of section 3, plus Chronos-2, Chronos-Bolt-base, TiRex and Kronos-small through the round-1 harness
model code (`tools/commodity/models.py`, HF_HUB_OFFLINE=1), scored with the same quantile rule. Kronos: the round-1
A1 settings (10 samples, context 256) on every 5th origin date if it is too slow for all; if that still exceeds
~30 min of CPU, Kronos is run on every 5th date of the frozen cell's series subset that fits (descriptive only,
labelled). IBM TTM: included only if
it loads offline; otherwise reported as missing.

## 6. VAL and TEST protocol

VAL: run once (`val_r2.py`), for the frozen choice only; reports the frozen cell's metric and test vs every model.
TEST (later, once, someone else, `test_r2.py`): origin dates >= 2025-11-01 on the POOL5 intersection calendar, all
models of section 5 in one run; primary claim: the frozen recipe beats the DEV-strongest baseline AND its twin on the
frozen cell by the section-5 test, one-sided 5%; "beats all" also requires every other baseline and foundation model
(Holm over those comparisons). On TEST at h = 5 (10) the non-overlapping sample is ~1 year / 2h dates x 5 series,
i.e. ~25 (12) independent dates: declared in advance as low power. Robustness (descriptive): pooled accuracy without
the gasoline RVP transition windows (origin dates Mar 15 - May 15 and Aug 25 - Sep 30 for DGASNYH, DGASUSGULF) and
per-product accuracies.

## Amendments
