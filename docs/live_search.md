# Alma-Live Gas: world-model search (Henry Hub daily, paper trading)

Owner: Alma-Live world-model architect. My files: `tools/live/wm_*.py`, `tools/live/search/**`,
`runs/live/search/**`, `runs/live/FROZEN.json` and this document. The data lead owns the rest of `tools/live/`
(point-in-time store `asof(t)`, paper broker, harness). The red team hunts leaks.

## 0. Protocol (written and committed BEFORE any DEV score; only plumbing smoke tests run before this commit)

### 0.1 Clock stops (UTC, 2026-11-01)

| Stage | Stop |
|---|---|
| plumbing (data adapter, leak tests, model code; smoke only, no metric) | 13:30 |
| WM loop (forecasting families: structural+graph, RSSM, transformer, direct/no-graph twins, ensembles) | 15:45 |
| continual loop (online rules on the WM incumbent) | 16:15 |
| agent loop (Dreamer actor, PPO, DQN, vol-target rule) | 16:50 |
| finalist seeds + feature-matched baseline gate, ping manager before VAL | 17:20 |
| VAL (once per finalist, post-selection label), ping manager before freeze, FROZEN.json | 18:00 |
| v2 search (same protocol, separate K block, same results.tsv) | 23:30 |

A stop is a hard stop: a run that started before it may finish and is logged; nothing new starts in that stage.

### 0.2 Data and information set

The data lead's `asof(decision_time)` store had not landed when this was written. Until it does, the model reads an
interim point-in-time adapter `tools/live/wm_data.py` (mine) over the already-pinned Alma-Gas raw panels
(`data/gas/daily.csv`, `weekly.csv`, `cpc_obs.csv`, `cpc_fcst.csv`). It exposes the same contract: every input at
decision time t (16:00 America/New_York on trade date t) is built from rows whose availability time is <= t:
- Henry Hub spot of trade date t and earlier;
- an EIA weekly storage report when its release date (`avail`, holiday schedule aware) <= t;
- CPC observed degree days for calendar days <= t-1 (never the realized weather of t or later);
- CPC 7-day degree-day FORECASTS: the latest issue with issue date < t, or issue date == t and HTTP Last-Modified
  <= t 16:00 ET. Forecast days not covered by that issue are masked (value 0, mask 0); realized weather is never
  substituted for a forecast;
- causal degree-day normals (same day-of-year +/- 7 days, years strictly before the year of t);
- calendar (day of year, weekday, days to the next storage release).
Supply, rigs, LNG feedgas, COT and EDGAR event slots exist in the latent state and the shock graph but are masked
until the store provides them (disclosed limitation; they enter today only as scenario shocks).

**Instrument (interim).** The store's front-month / UNG series is not available yet, and EIA stopped publishing
RNGC1 in 2024. The interim tradable proxy is Henry Hub spot. Decision at the close of t, fill at the next morning's
spot assessment P[t+1] (next-open proxy), P&L = pos_t * (log P[t+2] - log P[t+1]) - cost * |pos_t - pos_{t-1}|,
cost 5 bp per unit turnover (10 bp sensitivity). Results are re-run on the store's front-month series when it
lands; the spot numbers are labelled "spot proxy" everywhere.

**Leak test (per checkpoint, before it is scored).** Randomized-future perturbation: for 64 random decision days t
per fold, every raw row whose availability time is after t (prices, storage reports, observed weather, CPC issues,
plus the targets) is replaced by random values; the rebuilt input window and the model output at t must be
bit-identical. Allowed perturbations (the price of t, the latest usable CPC issue, the latest released report)
must change the output. A checkpoint that fails is logged as `leakfail` and never ranked.

### 0.3 Splits

- DEV: decision days 2015-01-02 .. 2024-06-30 (targets <= 2024-06-30). Expanding-window inner folds, 4 folds by
  test year: Jul 1 Y .. Jun 30 Y+1 for Y = 2020, 2021, 2022, 2023. Each fold trains on decision days with all
  targets (t+7 calendar days margin) strictly before the fold start (asserted). All scores in the loop are on these
  folds only.
- VAL: 2024-07-01 .. 2025-10-31. Run once per finalist (models trained on all DEV days), for selection between
  finalists only; every VAL number is labelled post-selection.
- EXPOSED: >= 2025-11-01 (a prior project saw these HH prices). Never used for selection; reported only as
  "exposed, descriptive".

### 0.4 Model (world model)

Latent state per day: structured nodes (storage deviation vs 5-yr norm, weather-driven demand, supply, LNG / net
exports, price level regime, volatility regime, positioning) plus a learned latent z. Transition (daily step):
physics balance `stor_{k+1} = stor_k + supply_k - demand_k(HDD, CDD forecast) - netexp_k` with learned demand
coefficients and a learned residual, plus the N-order shock-propagation graph (signed, learned edge weights,
structural sparsity: weather -> demand -> storage -> price -> producer equities; LNG outage -> feedgas -> net
exports -> storage). Events enter as shocks to node values and propagate over the rollout. Readouts per rollout day
k = 1..7: daily return quantiles (k = 1 is next day; k = 2 is the traded day), 5-day return quantiles, log realized
5-day vol quantiles, next weekly storage change (vs its 5-yr average) quantiles.

### 0.5 Search metric (forecasting), lower is better

`S_F = mean over 4 DEV folds of mean_j QS_j / QS_j(ref)`, j in {r1, r2 (traded day), r5, lrv5, dsurp}; QS = mean
pinball over levels 0.05, 0.1, .., 0.95; ref = linear quantile regression (ridge-penalised) on the identical
flattened inputs, refit per fold (not counted in K). Also reported, never optimised alone: per-target QS, cov80.

### 0.6 Agent metric

`S_A` = annualised net Sharpe (252) of the concatenated DEV fold test days at 5 bp; turnover and max drawdown
reported. Position sizing by predicted vol (target 20% annualised, cap |pos| <= 2).

### 0.7 Search loop

Self-improving autoresearch loop: propose one change -> train -> leak test -> score on DEV folds -> keep (if S_F
improves by > 0.005; smaller is seed noise) or revert. Seed 0 for exploratory rows. EVERY run, crashes and leak
failures included, is appended to `runs/live/search/results.tsv`; every row counts in K for the deflated Sharpe.
Families: structural balance + residual + graph (`struct`), RSSM (stochastic latent, `rssm`), transformer encoder
(`tfm`), and ensembles. Every WM recipe that becomes incumbent gets its `direct` twin (same encoder, readouts
straight from the encoder state with the exogenous forecast inputs concatenated, no rollout) and `nograph` twin
(graph replaced by a parameter-matched MLP), params within +/- 5%, same steps and seeds.

### 0.8 Baselines and the feature-matched gate

Feature-matched baselines on identical inputs: ridge quantile regression and LightGBM quantile regression
(forecasting); a LightGBM policy (vol-targeted sign of predicted r2) and rules: buy-and-hold, tsmom20 vol-targeted,
vol-target long. Every finalist must beat the strongest feature-matched baseline on DEV (S_F for forecasting,
S_A for agents), or is reported as "does not beat <baseline>".

### 0.9 Pre-declared pillar tests (DEV for search, VAL post-selection; 3 seeds, ensemble of seeds)

Paired daily differences, stationary block bootstrap (mean block 10 days, 2000 draws, seed 0), two-sided p:
1. WM vs direct: per-day mean relative pinball (the S_F terms).
2. WM vs no-graph: same.
3. Continual vs frozen: same, with the online rule applied day by day through each fold.
4. WM agent vs PPO, vs DQN (both trained on real transitions, same observation = frozen WM latent + previous
   position, same nets and steps), vs the non-RL vol-target rule on WM forecasts: daily net P&L difference and
   Sharpe difference.
A pillar is "supported" only if the difference favours the WM and the 95% CI excludes 0.

### 0.10 Continual

After each day's outcome is known: replay (last 60 days + new day), k SGD steps, L2-to-init; a guard compares the
rolling 20-day loss of the online weights with the frozen weights and rolls back to the last good snapshot when the
online loss is worse by more than a margin. Meta rule (optional): lr multiplied by exp(+-eta) depending on whether
the last update reduced the next day's error.

### 0.11 Freeze

`runs/live/FROZEN.json`: checkpoint sha256s, recipe, K, var_sharpe_trials (variance of the per-period net Sharpe of
all agent rows), claim models, pillar tests. The manager is pinged before VAL and before the freeze.

## 1. Log

(appended as the search runs)

### Amendment A1 (12:40 UTC, before any DEV score; data lead and red-team inputs)

- The decision clock is `common.decision_time(day)`, which is 14:45 America/New_York (the manager's default), not 16:00. Inputs come from
  the data lead's `tools/live/store.py` `Store().asof(t)`, which is filtered on `published_at <= t`.
- HH spot is dropped entirely (red team LRT-2: EIA/FRED publish daily spot weekly). The price series is NG=F (with UNG
  and BOIL for execution) through d-1. The P&L/target is NG open(d+1) -> open(d+2), with roll-crossing intervals masked.
  Return targets are redefined accordingly: r1 = open(d+1)/close(d-1), r2 = traded open->open, r5 = open(d+5)/open(d+1),
  lrv5 = log RMS of the 5 open->open returns.
- Storage is first-print vintage only (LRT-1; the 33-week resubmission of 2026-04-23 enters only as its own vintage at
  that release).
- The interim adapter `tools/live/wm_data.py` stays as plumbing only. No score is computed on it.
- The clock stops shift by +90 min to absorb the store's ETA (13:40 UTC). Plumbing ends 15:00, the WM loop 17:15, continual
  17:45, agents 18:20, finalists 18:50, and VAL/freeze 19:30. v2 is unchanged (23:30).

### Amendment A2 (manager rulings, ~12:55 UTC, before any DEV score; supersedes A1 where they conflict)

- **Clock.** `common.decision_time(day)` = 15:30 ET, with prices in ctx through d-1 (a day-d bar is published at d 18:00 ET).
  The fill is at `common.fill_time(day)`, the 16:00 ET close of day d.
- **Instrument.** UNG is primary (BOIL secondary; NG=F only as a feature). The P&L is w_d * (log C_UNG(d+1) - log C_UNG(d)), at 10 bp per
  side, reported at 1x, 2x and 3x costs. HH spot is never a target (LRT-5); it may be a feature at its publication time.
- **Targets.**
  - r1 = UNG close(d) -> close(d+1) (traded; next-day distribution);
  - r5 = close(d) -> close(d+5);
  - lrv5 = log RMS of the 5 daily UNG returns d -> d+5;
  - dsurp = next first-print L48 weekly change minus its 5-yr average.
- **Selection score.** S_F is the mean over folds of the relative QS over the three primary claim targets {dsurp, r1, lrv5}. r5 is trained and
  reported but is not in S_F.
- **World model.** It builds on the prior wm_gas storage model, plus a CPC weather pathway (forecast + run-to-run revision; LRT-6
  fixed, pathway verified non-zero), a stochastic-vol block (log-vol node with an AR transition), a UNG price head, and an LNG/feedgas
  node (masked until data exists).
- **Graph.** The graph is hand-specified causal physics. It is used for scenarios and risk, and it is used for timing only if the graph WM beats
  its no-graph ablation on DEV.
- **Policy.** The champion is a fractional-Kelly (k = 0.25), vol-targeted rule on the WM forecast distribution. Challengers:
  - PPO trained in WM imagination with an ensemble-disagreement penalty;
  - PPO and DQN on raw features against real transitions;
  - the Dreamer-style actor.
  DEV baselines that the policy must beat:
  - buy_hold_ung and short_ung_carry (vol-targeted);
  - tsmom_20 and tsmom_252_21;
  - eia_day_long (Wed close -> Thu close);
  - mean-reversion -sign(r_{d-1}) and -sign(5d);
  - weekday rule (sign of the training-fold mean return by weekday);
  - seasonal_rule;
  - lgbm and ridge on the same features with the same sizing.
- **Run labels.** Runs made before the LRT-6 fix: none were scored (smoke only), so none are in results.tsv.

### Amendment A3 (12:35 UTC; coordinator re-scope; before any DEV score)

- **Single question.** Does the WM beat its matched direct twin and the strongest feature-matched baselines (ridge-QR,
  LightGBM-QR on identical inputs, and the prior frozen wm_gas on storage) on {r1 QS, dsurp, lrv5}?
- **Loop.** A Karpathy autoresearch loop per `tools/live/search/program.md`:
  - the single editable file is `tools/live/wm_core.py`;
  - the fixed wall-clock budget is 45 s per fold (~4 min per experiment);
  - the metric is skill = 1 - S_F (DEV folds only);
  - keep a change if it gains > 0.002, otherwise git revert;
  - the direct twin is trained at matched budget for every kept incumbent.
- **Data.** Core only: UNG, first-print storage, CPC. COT and NG=F are removed from features (38 -> 32 inputs).
- **Paused** until a DEV forecasting edge: Dreamer/imagination PPO, raw PPO/DQN, the equity graph for timing, and the continual
  search. The only policy is Kelly vol-target. There is one continual-vs-frozen check at freeze time.
- **Clock stops (UTC, binding).** The panel freezes when the data lead's pit CPC lands. The loop starts then and stops at
  **16:30**. Finalists (3 seeds of the incumbent and its twin, refs, the gate) run until **17:00**. Then I ping the manager
  before VAL. Freeze target is **17:45**.
