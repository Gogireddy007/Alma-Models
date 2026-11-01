# Alma-Fin (crypto perpetuals, day-ahead volatility + risk-sizing policy): pre-registration

Author: Alma-Fin lead (agent), 2026-10-31/11-01. Branch `alma/investor-sprint` (local only). Harness: `tools/fin/`
(`fetch.py`, `fetch_etf.py`, `universe.py`, `data.py`, `harness.py`, `models.py`, `metrics.py`, `run_eval.py`,
`test_infoset.py`, `label_corr.py`; interface in `tools/fin/README.md`). Research basis: `docs/fin_research.md`
(f285496), adopted with the changes listed in section 11. Leakage record: `docs/fin_leakage.md`.

Status: written and committed BEFORE any forecast, position or score of any kind was computed on any window (DEV, VAL
or TEST). Looks at the data before this commit: download counts and sizes, the universe ranking (30-day quote volume
to 2025-10-31), build-report counts (rows, gaps, outliers), and the cross-coin label correlation on DEV and VAL
(section 6.4; a property of the labels, not of any model). No TEST-period value was printed or summarised: TEST rows
are written to `*_SEALED.parquet`, hashed and counted only. Every change after this commit is a dated amendment at
the bottom, made before the TEST run, with its reason.

## 1. Data (free, no key; raw files + url + UTC download time + sha256 in `data/fin/manifest.json`)

| series | source | use |
|---|---|---|
| USD-M perpetual 5-min klines (OHLC, volume, quote volume, trade count, taker-buy volume) | `data.binance.vision` monthly zips 2019-09..2026-09, daily zips 2026-10-01..30; Binance `.CHECKSUM` (sha256) verified per file, mismatch -> rejected | labels (RV), fills, hourly/daily features |
| USD-M funding-rate settlements | `data.binance.vision` monthly `fundingRate` zips 2019-09..2026-09 | P&L (funding), features |
| SPY, QQQ, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY daily OHLCV + adjusted close | public Yahoo chart endpoint (unofficial; Stooq now serves a bot check, which we do not bypass), 2015-01..2026-10 | cross-market zero-shot arm |
| S3 listing of every USD-M symbol and its monthly files | `data.binance.vision` bucket listing (`data/fin/listing_um.json`) | universe; it KEEPS delisted symbols (LUNAUSDT to 2022-05, SRMUSDT to 2024-05, MATICUSDT, FTT, ...: 27 symbols end before 2025-06) |

Not available, therefore not used: the 2026-10 funding archive (published after month end) and the Binance REST API
(`fapi.binance.com` refuses this location). Consequence: TEST trading P&L (P2) is scored on decision days
2025-11-01..2026-09-29 (section 4); P1/P3 use all TEST days to 2026-10-30. The archive has no 2026-10-28 daily files
(404 for all 24 symbols at download time and on a retry before the pin): that day's labels are NaN, its 5-min
returns are absent, and the label reference of the following 7 days is NaN (excluded), by the generic gap rule.

Processing (`data.py build`): per symbol, 5-min bars de-duplicated by open time (no filling); hourly and daily
aggregates; `rv(D)` = sum of squared 5-min log returns of the bars opening 00:05..23:55 UTC of D (first return =
close(00:05 bar)/close(00:00 bar)), NaN if fewer than 274 of the 287 returns are present; `fill(D)` = (O+H+L+C)/4 of
the 00:05-00:10 bar of D; funding settlements assigned to the UTC day of the interval they END (a settlement at
D+1 00:00 belongs to day D; the settlement at 2025-11-01 00:00 is a TEST value and is not assigned to the open
2025-10-31 row). Outliers: 5-min |log return| > 0.25 are listed in `build_report.json` and kept unchanged; labels are
never clipped. ETFs: Garman-Klass session variance as `rv`; fill = session open; P&L on dividend-adjusted fills.
Rows dated >= 2025-11-01 go to `*_SEALED.parquet` (read only with `ALMA_FIN_TEST_GO=1`, which only
`run_eval.py --split test --i-have-auditor-go` sets); `data.load()` asserts the open files hold nothing later.
The processed files' sha256 are pinned in `run_eval.DATA_PINS` (section 12), before any score.

## 1b. Universe (fixed once, `tools/fin/universe.py`, from data <= 2025-10-31; `data/fin/universe.json`)

Candidates: USDT-margined perpetuals (no delivery suffix) with a 2025-10 daily file and a first daily bar <=
2024-11-01 00:00 UTC (343 listed by 2024-11 and alive in 2025-10; 319 eligible after removing listings after
2024-11-01 and index/stablecoin products BTCDOMUSDT, DEFIUSDT, USDCUSDT). Rank = sum of quote volume of the daily
bars 2025-10-02..2025-10-31. Top 24; held-out = ranks 3, 6, ..., 24 (every third); training = the other 16.

- training (16): ETHUSDT, BTCUSDT, BNBUSDT, DOGEUSDT, ZECUSDT, SUIUSDT, LINKUSDT, ENAUSDT, ADAUSDT, LTCUSDT, WLDUSDT,
  AAVEUSDT, WIFUSDT, NEARUSDT, CAKEUSDT, APTUSDT
- held-out (8): SOLUSDT, XRPUSDT, 1000PEPEUSDT, AVAXUSDT, TAOUSDT, DOTUSDT, BCHUSDT, 1000BONKUSDT
- held-out market (13 ETFs): SPY, QQQ, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY

Survivorship: the choice uses only information available before TEST, so TEST is free of survivorship bias; any
symbol delisted during TEST stays in the universe (position closed at its last close, P&L kept, slot earns 0
afterwards, no replacement). DEV is NOT survivorship-free (coins that died before 2025-10 cannot be in the universe);
DEV numbers carry this caveat. Several coins start after 2020 (e.g. SUI 2023-05, WIF 2024-01, ENA 2024-04, TAO 2024-04):
the panel is unbalanced; NaN before listing.

## 2. Information set (the leakage rule; enforced in code, tested by `tools/fin/test_infoset.py`)

Crypto, decision day D:
1. The decision is taken at D 00:00 UTC (= close of the last 5-min bar of D-1). A model sees daily rows of days
   <= D-1 and hourly rows of hours ending <= D 00:00 (`harness.build_context` asserts both with timestamps; the
   context is a fresh read-only copy).
2. Execution at the 00:05-00:10 UTC bar of D at (O+H+L+C)/4: one full bar after the decision bar, never the
   decision bar itself (no same-bar fill). The position is held until the fill of D+1.
3. Funding settled in (D 00:00, D+1 00:00] is charged to the position held from fill(D) (long pays a positive rate).
4. Labels of D (RV(D), return of D) are complete at D+1 00:00. A continual model receives day D's completed row
   through `observe()` only AFTER its decision for D is recorded and BEFORE its decision for D+1 (asserted:
   `available_from <= next decision time`); the runner materialises the row only then.
5. Models get data only through the arguments: while `predict`/`act`/`observe` run, an audit hook refuses any
   open() under `data/fin/` (tested with a deliberately snooping model).
6. Training (any fitting, any model, incl. the architect's): labels <= 2025-06-30 only, through
   `models.train_view()` (asserts the cut; training coins only unless the model is a pre-registered per-asset
   twin/baseline). Online updates during VAL/TEST runs through `observe()` are the only exception (continual
   pillar). ETFs are never used for fitting, except by the per-asset baseline `lgbm_asset` (each ETF's own DEV
   history) and the rolling per-asset baselines that refit on context history.
ETF, session D: decision at the close of session D-1; fill at the open of D; held to the open of D+1; context =
sessions <= D-1.

`test_infoset.py` (must pass before any score; output committed to `runs/fin/test_infoset.txt`):
perturbation (every daily value of days >= D and hourly value of hours >= D 00:00 multiplied by U(0.3, 3) ->
bit-identical contexts), sensitivity (day D-1 and the last hour before D must matter), raw re-derivation of fill
and RV from the zips, P&L timing (only fill(D), fill(D+1) used), training-view cut, data-access guard, model
future-invariance + determinism for the listed models, and an online toy model whose decisions up to a perturbation
day are bit-identical and whose updates are logged.

## 3. Splits (decision days)

- DEV: labels <= 2025-06-30. Harness DEV backtest: decision days 2022-07-01..2025-06-30 (selection only).
- VAL: 2025-07-01..2025-10-31 (123 days). Debugging and ONE confirmation run of the frozen candidates (+ twins); the
  go/no-go rule is written in FROZEN.json before VAL is read. VAL positions are marked at the 2025-10-31 close.
- TEST: 2025-11-01..2026-10-30 (P1/P3), 2025-11-01..2026-09-29 (P2). SEALED: no metric and no target value is
  computed or inspected before the single TEST run (section 9).
- ETF arm: the same calendar windows, NYSE sessions only.

Contamination of competitors (zero-shot FMs): Chronos-Bolt (2024-11) and TiRex (2025-05) predate VAL and TEST.
Chronos-2 was released 2025-10-17: its weights predate TEST, its corpus cut-off is not public and may overlap VAL.
Kronos-small was pretrained on K-lines from 45 exchanges up to June 2024 (crypto included): it has seen DEV, not
VAL/TEST, so its DEV numbers are flattering. Timestral-5M (released 2026-10-15) may have TEST in its training data:
reported, marked CONTAMINATED, excluded from every "beats" claim. Alma checkpoints of earlier rounds are not used
here. Prior TEST exposure in this repo: none for crypto or these ETFs (`docs/fin_leakage.md`).

## 4. Targets and decision rules (fixed; no threshold fitted on any window)

| id | target | label | model's prediction |
|---|---|---|---|
| P1 (PRIMARY) | next-day realised-vol direction, crypto, all 24 coins pooled (reported also for training / held-out coins) | `y(D) = 1[log RV(D) > mean_{k=1..7} log RV(D-k)]`; NaN if any input is NaN; exact tie excluded | `p_up > 0.5` if the model gives p_up, else median(log RV(D)) > the same reference; tie -> DEV majority |
| P2 (CO-PRIMARY) | net-of-cost daily P&L of the policy on the 8 HELD-OUT coins (equal 1/8 sleeves, fixed denominator), also on all 24 and the training 16 | net = pos x (fill(D+1)/fill(D) - 1) - pos x funding(D) - cost x abs(pos(D) - pos(D-1)) | positions in [-1, 1] per coin (clipped) |
| P3 (secondary) | log-RV distribution | log RV(D) | mean pinball over levels 0.1..0.9 (QS; CRPS ~ 2 QS), MAE of the median, 80% coverage |
| P4 (secondary, no claim expected) | next-day return direction | sign of the close-to-close log return of D | `p_ret_up > 0.5` |
| E1 (transfer, descriptive unless strong) | P1 on the 13 ETFs (Garman-Klass RV) | as P1 | as P1 |

Costs (per side, on abs(delta position) at the fill): taker fee 5 bp + slippage 2 bp for BTCUSDT/ETHUSDT, 5 bp for
the other coins (7 / 10 bp total); ETFs 2 bp. Stress run: costs x 2 (reported for every agent). Positions before the
first day of a split are 0; the closing trade after a delisting is charged; at the end of a split positions are
marked at the last close (no closing cost). Rebalancing to the target weight each day (no intraday drift of weights).

P2 metrics: annualised Sharpe (x sqrt(365); ETFs sqrt(252)) with a 7-day block-bootstrap 95% CI; PSR(0) (Bailey &
Lopez de Prado, skew/kurtosis-adjusted); DSR deflated by K (FROZEN.json `K`, and `var_sharpe_trials` = variance of
the per-period Sharpe over the K DEV trials); annual return and vol, max drawdown, Calmar, mean daily turnover
(mean over coins of abs(delta position)), gross / funding / cost decomposition, Sharpe at 2x costs.

## 5. Models (fixed settings; `tools/fin/models.py`; CPU, `nice -n 10`, HF offline)

Volatility forecasters (QS = 0.025, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.975):

| name | kind | settings |
|---|---|---|
| majority | baseline | DEV up-fraction of y (training coins, decision days <= 2025-06-30) |
| rv_persist | baseline | median log RV(D) = log RV(D-1); quantiles from DEV pooled daily changes |
| har | baseline | HAR-RV on log RV (1, lrv(D-1), mean 5d, mean 22d), pooled OLS on DEV training coins; Gaussian residual quantiles |
| harx | baseline | har + abs(funding(D-1)), taker imbalance(D-1), min(ret(D-1), 0) (crypto only) |
| har_roll | baseline (rolling re-estimation, per asset) | HAR refit every day per asset on its own trailing 365 context days |
| garch | baseline | GARCH(1,1) Gaussian per asset on the trailing 365 daily returns, refit daily; log RV = log sigma^2 + in-context mean gap |
| lgbm | competitor | LightGBM classifier (y) + regressor (log RV - ref) on 11 hand-crafted daily features (log RV levels/means, returns 1/7/30d, range, taker ratio, funding, volume z), pooled DEV training coins, fixed params (300 trees, lr 0.03, 15 leaves, seed 0) |
| lgbm_asset | competitor (per-asset twin) | same features, one model per asset on its own DEV history (200 trees, 7 leaves); ETFs: per ETF |
| chronos2 / chronos_bolt_base / tirex | FM (zero-shot) | univariate on the daily log RV series, last 512 days, h = 1 |
| timestral_5m | FM, CONTAMINATED | same |
| kronos_small | FM (zero-shot) | Kronos-small + Kronos-Tokenizer-base @67b630e on daily OHLCV K-lines (quote volume), 256-day context, 10 samples, T = 1, top_p 0.9, seed 0; predicted Parkinson variance of the sampled bar mapped to log RV by the in-context mean gap |
| architect candidates + twins | candidate | frozen per section 9 |

Trading agents (positions in [-1, 1] per coin):

| name | kind | rule |
|---|---|---|
| buy_hold | baseline | +1 in every asset |
| tsmom_1 / tsmom_7 / tsmom_30 | baseline | sign of the past 1 / 7 / 30-day log return |
| meanrev_1 | baseline | minus the sign of the previous day's return |
| voltgt_tsmom | baseline (the honest strongest) | sign(30-day return) x min(1, sigma* / sigma_HAR), sigma* = 40%/yr (crypto), 10%/yr (ETF), sigma_HAR = sqrt(exp(median log RV)) of the pooled DEV `har` |
| lgbm_policy | competitor | sign(P_lgbm(return up) - 0.5) x the same vol scaling |
| ppo, dqn | model-free RL twin (architect, `tools/fin/agents/`) | PPO (continuous) and DQN (5 sizes {-1, -0.5, 0, 0.5, 1}) on the SAME encoder features, real DEV transitions only, same reward, same budget/seeds as the world-model agent (section 7) |

## 6. Statistics

6.1 P1 (crypto coin-days; daily, non-overlapping; cross-sectionally correlated). Primary inference uses one number
per day: the day's mean hit rate over coins.
- Accuracy (pooled coin-days) with a 7-day circular block-bootstrap 95% CI (whole days resampled, 2000 draws, seed 0).
- vs 0.5: DM-HLN (h = 1) on the daily series 0.5 - daily accuracy (two-sided).
- Model vs competitor (PRIMARY pairwise test): DM-HLN (h = 1) on the daily difference of mean accuracy, two-sided,
  AND the 7-day block-bootstrap 95% CI of the mean difference must exclude 0.
- Descriptive only (they ignore cross-sectional correlation and are never quoted as evidence): coin-day Wilson CI,
  exact binomial, Pesaran-Timmermann, coin-day exact McNemar; Wilson with the effective sample size
  N_eff = n / (1 + (m - 1) rho) (rho from 6.4). Brier and log loss of p_up where given.
6.2 P3: DM-HLN (h = 1) on daily mean QS differences + 7-day block-bootstrap CI.
6.3 P2: paired 7-day circular block bootstrap of the Sharpe difference (2000 draws, seed 0), two-sided p =
2 min(P(diff* <= 0), P(diff* >= 0)), CI must exclude 0. Single-strategy significance: PSR(0) > 0.95 AND DSR > 0.95.
6.4 Cross-sectional dependence (measured before this commit by `tools/fin/label_corr.py`, DEV 2022-07..2025-06 and
VAL; written to `runs/fin/label_corr.json`): mean pairwise correlation of the coin labels y and of the daily
log-RV innovations; reported in section 12 with the implied N_eff and minimum detectable effects. Used only for
the descriptive N_eff; the primary tests (6.1) are day-level and do not depend on it.
6.5 Holm: for each claim model, over its comparisons with every non-contaminated competitor on the same target and
group ("beats" family); pillar tests (section 7) form one family of their own per pillar endpoint set.
6.6 Overlap: all samples are daily and non-overlapping (h = 1 day); no horizon > 1 is used, so no overlap
correction beyond the day-level aggregation and the 7-day blocks (which absorb short serial dependence) is needed.

## 7. Claims (each pillar against its matched twin)

General: "C beats X on T" only if C's metric is better AND the primary p, Holm-adjusted over C's comparisons with
every non-contaminated competitor on T, is < 0.05 AND the bootstrap CI of the difference excludes 0. "Beats every
strong baseline" requires that against EVERY non-contaminated model and baseline of section 5 on P1 (all 24 coins).
Otherwise: "numerically better/worse, not significant". All targets reported, wins and losses alike.

Twin matching (required for any pillar claim; evidence in FROZEN.json, checked by the auditor): identical inputs,
data, training budget (steps, samples), seeds, initialisation scheme, parameter count within +-5%; the twin differs
only in the pillar's mechanism.

| pillar | candidate | matched twin | endpoints (TEST) | claim if |
|---|---|---|---|---|
| world model vs model-free RL | WM agent (actor-critic trained in latent imagination) | `ppo` (and `dqn`) on the same encoder features, real transitions only | P2 net Sharpe, held-out coins (primary) and all 24 | Sharpe diff > 0, Holm over {held-out, all} x {ppo, dqn} < 0.05, CI excludes 0 |
| world-model rollout vs direct predictor | WM latent rollout -> day-RV heads | same encoder, direct day-ahead heads (`abl=direct`) | P1 (DM daily) and P3 QS, all 24 coins | better on P1 or P3 with Holm over the 2 endpoints < 0.05 and CI excluding 0; contradiction on the other endpoint is reported |
| continual vs frozen | candidate with `observe()` (frozen online rule: optimizer, lr, steps/day, replay, anchor) | the same weights with `observe()` disabled (`name:frozen`) | P1 and P3, all 24 coins | as above; update counts per day reported |
| zero-shot portability vs per-asset | pooled candidate, observe disabled, on the 8 held-out coins (and 13 ETFs) | `lgbm_asset` and `har_roll` (each asset's own history) | P1 on held-out coins; E1 on ETFs (descriptive) | better than both per-asset twins, Holm over the 2 < 0.05, CI excludes 0; "non-inferior" is not claimed |

K and the winner's curse: FROZEN.json records K (every configuration evaluated on DEV or VAL by the architect,
including discarded ones) and the variance of their DEV Sharpe/accuracy. DEV/VAL numbers of the selected candidate
are biased upward (the maximum of K noisy estimates); only TEST is a confirmation. DSR uses K. A pillar or "beats"
claim on VAL is never made (VAL has 123 days; it is for debugging).

## 8. Power (stated before any score; research section 4.3, updated with rho from 6.4 in section 12)

With the measured rho = 0.51 (section 12), N_eff ~ 1.88 coin-days per day for 24 coins: ~680 effective coin-days on
TEST; minimum detectable P1 difference vs 0.5 or vs a competitor with 25% discordance ~ 5.4 pp. Typical ML-over-HAR gains are 1-3 pp: a
significant P1 win over har/har_roll is possible but unlikely. P2 from one year: SE(annual Sharpe) ~ 1, MDE vs 0
~ 2.5, vs a 0.7-correlated twin ~ 1.9. No honest one-year test can confirm an ordinary trading edge; P2 is
reported with PSR/DSR and CIs and claimed only if it clears these bars.

## 9. Selection, freeze, TEST

1. Search and selection on DEV only (training labels <= 2025-06-30). The architect's own splits are inside DEV.
2. VAL (`run_eval.py --split val`, this harness, authoritative) is run ONCE for the frozen candidates + twins; the
   go/no-go rule is in FROZEN.json before VAL is read.
3. Freeze: `runs/fin/FROZEN.json` (architect): model names, `claim_models`, checkpoint paths + sha256, code commit,
   online rule, K, `var_sharpe_trials`, twin-matching evidence; committed (`git add -f`) before TEST.
4. TEST: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run_eval.py --split test --i-have-auditor-go`, exactly once, all
   section-5 models + FROZEN models, both markets, only after the manager relays the auditor's GO. Strict mode
   refuses: no flag; missing or uncommitted FROZEN.json; existing RUN.json or forecast files; `--models`, `--force`,
   or a single market; dirty `tools/fin`, this file or FROZEN.json (incl. untracked files); processed data whose
   sha256 differ from `DATA_PINS`; HF online; mismatching FROZEN checkpoints; any architect-module import error
   (preflight, before RUN.json). A crash after RUN.json consumes the TEST (status failed recorded). Rehearsed
   first on VAL into a fresh directory (`--split val --rehearse`). After the run: RUN.json, hashes.json, FROZEN.json
   are force-added and committed.

## 10. What is reported regardless of outcome

Every section-5 model on P1-P4 and E1 by group; every agent's P2 table incl. 2x costs; update counts of every
continual model; the four pillar tests; K and the winner's-curse note; contamination labels; rho and N_eff; data
caveats (DEV survivorship, Yahoo as an unofficial ETF source, 2026-10-28 archive gap, P2 TEST window ending
2026-09-29, the boundary funding settlement in the 2025-10-31 23:00 hourly row (known at the first TEST decision
time, so allowed, but it is the only TEST-dated value in the open hourly file)).

## 11. Differences from docs/fin_research.md (and why)

- P2 TEST window ends 2026-09-29 (no 2026-10 funding archive; REST API geo-blocked).
- Yahoo instead of Stooq for ETFs (Stooq bot check).
- P1 primary inference is day-level (DM-HLN on daily accuracy + 7-day blocks) rather than McNemar on coin-days,
  which would ignore the cross-coin correlation; McNemar is supplementary.
- Funding settlements at the split boundary are kept out of the open file's daily row (section 1).

## 12. Measured before this commit (DEV/VAL label properties; no model involved)

Data (raw files downloaded 2026-11-01 ~04:10-05:40 UTC; `data/fin/manifest.json`, 4,700 Binance files with
CHECKSUM verified, 0 rejected; 542 funding-month 404s are all months before a coin's listing (checked: no funding
month is missing where klines exist); 13 ETF JSON files). Build (`data/fin/build_report.json`): open files
crypto_daily 37,285 rows, crypto_hourly 894,635, etf_daily 34,361; sealed TEST rows (counted only) 8,712 / 209,088 /
3,250. Only partial days are each coin's listing day (RV NaN there); no 5-min gaps otherwise; 47 five-minute
|log return| > 0.25 across all coins and years (kept). The archive lacks 2026-10-28 (TEST, all coins): it is a NaN
row of the complete UTC calendar used by the harness.

Pins (`run_eval.DATA_PINS`, strict mode refuses any other file):

| file | sha256 |
|---|---|
| data/fin/crypto_daily.parquet | 1b5c9ee8cfe60bcc618d0a742027ff4318c6947b82e59895b9fb5af8af574904 |
| data/fin/crypto_hourly.parquet | 5bdccc569c4638e995469423fc233285afc95f94cabe9345049354c8499ac084 |
| data/fin/etf_daily.parquet | 964a1f50f5f74aeb368b6e48368d51bdbcd0f1ede0f1b3e487a688f10775f1b0 |
| data/fin/crypto_daily_SEALED.parquet | 6429949adf5aadc18c230845355b69f375cafdef3f5f39976a62a581a2258cc4 |
| data/fin/crypto_hourly_SEALED.parquet | b72f5d60d1abddff7c1c9819e973e464c346e6762160422000b0b3a5f154ac96 |
| data/fin/etf_daily_SEALED.parquet | 9a720627b71dc0b9b2e68580f49bbab946ef652399b79f37dc20d0b000c5dc06 |
| data/fin/universe.json | 361e0373e86822099dfcbe5e87f6849e240dc431d8ba9042178294fb0a9e92b0 |

The hourly parquets (55 MB / 14 MB) and the Binance zips (531 MB) are not committed; they are re-creatable from the
manifest (urls + sha256) with `fetch.py` + `data.py build`, and strict mode checks the pins.

Cross-sectional dependence (`runs/fin/label_corr.json`; label property, no model):

| market / window / group | days | coin-days | rho(y) | rho(log RV - ref) | N_eff per day | base rate |
|---|---|---|---|---|---|---|
| crypto DEV 2022-07..2025-06, 24 coins | 1,096 | 22,265 | 0.512 | 0.713 | 1.88 | 0.457 |
| crypto DEV, 16 training | 1,096 | 14,988 | 0.501 | 0.707 | 1.88 | 0.459 |
| crypto DEV, 8 held-out | 1,096 | 7,277 | 0.532 | 0.730 | 1.69 | 0.452 |
| crypto VAL, 24 coins | 123 | 2,952 | 0.520 | 0.794 | 1.85 | 0.504 |
| crypto VAL, 8 held-out | 123 | 984 | 0.527 | 0.824 | 1.71 | 0.496 |
| ETF DEV 2022-07..2025-06, 13 | 751 | 9,763 | 0.323 | 0.509 | 2.67 | 0.488 |
| ETF VAL, 13 | 87 | 1,131 | 0.266 | 0.398 | 3.10 | 0.502 |

rho(y) ~ 0.51 (research assumed 0.6), so N_eff ~ 1.88 x 363 ~ 680 coin-days on TEST (24 coins; 8 held-out ~ 620).
MDE (80% power, two-sided 5%) vs 0.5 ~ 2.8 x 0.5 / sqrt(680) ~ 5.4 pp; vs a competitor with 25% discordance
~ 2.8 sqrt(0.25 / 680) ~ 5.4 pp. ETFs (~250 sessions x 2.7) ~ 680 effective, MDE ~ 5.4 pp but descriptive (E1).
`run_eval.RHO = 0.512` (descriptive N_eff only). The labels' common factor is strong (rho of the continuous
log-RV innovation 0.71-0.82): day-level inference (6.1) is required.

## Amendments

- A1 (2026-11-01 ~01:30 ET; before any architect candidate was scored by this harness; prompted by the red-team
  log `docs/fin_redteam.md`; no change of any target, metric, test or claim rule). VAL baselines and FMs had been
  run (`runs/fin/val/`, debugging, section 3) when this was written.
  1. RT-1 (operational): `import lightgbm` now precedes the architect auto-import in `models.py` and `run_eval.py`
     (a LightGBM fit after `import torch` crashes the process with an OMP error on this Mac). No model output changes.
  2. RT-2 (disclosure): the universe rank uses quote volume of 2025-10-02..10-31 and requires the coin to be alive
     in 2025-10; those days are VAL. VAL, like DEV, is therefore NOT selection/survivorship-free (e.g. ZECUSDT
     entered on its Oct-2025 activity). TEST remains free of both (no TEST information selects coins). VAL is a
     go/no-go gate only; its numbers carry this caveat.
  3. RT-5: the data-access guard now also refuses open() under `runs/fin/**` (caches, VAL forecasts, labels) during
     predict/act/observe, except paths containing `/ckpt/`.
  4. RT-6: strict preflight now REQUIRES, for every model listed in FROZEN.json `models`, `meta["ckpt"]` and
     `meta["ckpt_sha256"]` equal to FROZEN.json `sha256[ckpt]` and to the file on disk (and the same for
     `wm_ckpt`/`wm_ckpt_sha256` when an agent declares a world-model checkpoint); otherwise it refuses before
     RUN.json.
  5. RT-7: K for DSR and the winner's-curse note must count every configuration of BOTH architects
     (`runs/fin/search/results.tsv` and `runs/fin/search_b/results_b.tsv` rows) plus every VAL look by any agent;
     FROZEN.json `K` is checked against those files by the auditor.
  `test_infoset.py` re-run after these changes (output in `runs/fin/test_infoset.txt`).
- A2 (2026-11-01 ~02:10 ET; operational, no output change). Red-team follow-up to RT-1: after torch work, a
  multi-threaded LightGBM fit still hangs on this Mac (toy repro: n_jobs=2 killed after 90 s, n_jobs=1 completes).
  All LightGBM models now use `n_jobs=1` (with `deterministic=True`). Verified: predictions are bit-identical to
  n_jobs=2 (toy, 5,000 rows), and lgbm, lgbm_asset and lgbm_policy reproduce their committed VAL npz bit-for-bit
  on 3 VAL days in a process that ran torch first. Acceptance test before TEST: `run_eval.py --split val --rehearse`
  with the full list (section 5 + FROZEN models) must complete in one process.
- A3 (2026-11-01 ~02:40 ET; operational, no output change). The A1 fix (import lightgbm before torch) made later
  torch inference segfault (TiRex, Chronos-2, Timestral and Kronos VAL runs exited 139; no forecast file was written). Toy
  repro: lightgbm-first + torch ops -> crash; torch-first + LightGBM fit with n_jobs=1 -> fine. Now `models.py`
  imports torch first and LightGBM lazily inside the factories (n_jobs=1, A2). Verified in one process: lgbm,
  tirex, lgbm_policy, chronos2 in sequence complete.
- A4 (2026-11-01 ~03:20 ET; red-team RT-13; scoring rule made abstention-proof, no number of any existing model
  changes because all have 100% coverage). P1: the scored set per target and group is every coin-day with a finite
  label; a missing prediction counts as the DEV-majority direction (crypto: "down"), so the daily accuracy series
  of all models cover the same coin-days. P3: pairwise DM/bootstrap use daily means over the coin-days where BOTH
  models have a finite forecast (intersection). Coverage (P1, P3) is reported per model in metrics.json; a claim
  model with coverage < 99% of the scored set on a target/group gets no "beats" on it (`coverage_void`).
- A5 (2026-11-01 ~05:00 ET; manager decision on red-team RT-15; BEFORE any freeze; adds registered baselines, no
  rule removed). Reason: crypto RV has a large weekday effect relative to its trailing 7-day mean (DEV up-rate of
  y: Mon-Fri ~0.51, Wed 0.58, Sat ~0.24, Sun ~0.26), plus a US-holiday effect and intraday recency (the last hours
  of D-1). The section-5 list had no calendar or intraday input, so any model with a weekday feature would "beat
  every strong baseline" for a trivial reason (F1: har 0.664 vs HAR + weekday 0.697). Added to section 5 (P1/P3,
  pooled OLS / LightGBM on the DEV training coins, same information set, same `train_view` cut, same tests):
  - `har_dow` (baseline): har + 6 weekday dummies of the target day D (Mon = base).
  - `lgbm_dow` (competitor): lgbm features + weekday one-hot (7) of D; same params.
  - `har_intra_dow` (baseline; red-team spec, crypto only): OLS of log RV(D) on 1, log RV(D-1), mean 5d, mean 22d,
    ref (mean log RV(D-7..D-1)), 6 weekday dummies, US federal holiday flag of D (pandas `USFederalHolidayCalendar`,
    fixed published rules, incl. Juneteenth from 2021), log of the 5-min RV of hours 18-24 UTC of D-1 and of hour
    23-24 UTC of D-1 (from `ctx["hourly"]["rv5"]`); Gaussian residual quantiles. Not run on ETFs (no hourly data;
    added to ETF_SKIP); `har_dow`/`lgbm_dow` run on ETFs zero-shot with their crypto fit.
  The P1 headline comparison is the candidate vs the BEST of all registered non-contaminated baselines and
  competitors, including these three ("beats every strong baseline" already required all of them; section 7).
  Tests: `test_infoset.py` (future-invariance, determinism; default list now includes the three) passes; the
  red-team's training-invariance R1-R3 and perturbation checks pass (`docs/fin_redteam.md`). DEV inner folds
  (`tools/fin/fold_refs.py`, `runs/fin/fold_refs.json`; fit cut moved to each fold's end):

  | P1 all coins (held-out 8) | E1 2023H1 | E2 2023H2 | E3 2024H1 | F1 2024-07..2025-06 |
  |---|---|---|---|---|
  | har | 0.610 | 0.637 | 0.635 | 0.664 (0.668) |
  | lgbm | 0.610 | 0.666 | 0.674 | 0.669 (0.674) |
  | har_dow | 0.647 | 0.672 | 0.706 | 0.697 (0.706) |
  | lgbm_dow | 0.666 | 0.696 | 0.717 | 0.709 (0.716) |
  | har_intra_dow | 0.661 | 0.680 | 0.704 | 0.717 (0.729) |

  VAL (debugging, all 24 coins / held-out 8): lgbm_dow 0.736 / 0.727, har_intra_dow 0.729 / 0.721, har_dow 0.714 /
  0.707 (vs best FM tirex 0.704). The calendar-aware baselines are the bar any P1 claim must clear.
