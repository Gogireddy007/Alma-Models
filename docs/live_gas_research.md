# Alma-Live Gas: research brief for a live, continually-learning US natgas world model (paper first)

Author: research lead, 2026-11-01. Branch `alma/investor-sprint`. Status: v1. No model was trained, no dataset was
downloaded beyond a few-KB probes of the endpoints listed in section 2, and no credential was entered anywhere.
Builds on `docs/gas_research.md`, `docs/gas_results.md` and `docs/audit_gas.md` (the prior gas project).

Evidence labels used throughout: **[PR]** peer-reviewed, **[PP]** preprint/working paper, **[OFF]** official
documentation from the data owner, **[CHK]** checked by me today with a live HTTP probe, **[BLOG]** trade press /
blog / vendor claim, **[MINE]** my own calculation, **[REPO]** a result already audited in this repo.

## 0. Bottom line

1. **What the prior project proved [REPO].** A daily storage-balance model (`wm_gas`) forecasts the Thursday EIA
   storage print against the 5-year average better than every foundation model, and better than the weather OLS on
   quantile score (QS). It did not beat OLS on MAE or direction, and the world-model rollout did not beat its
   direct twin. Separately, the DEV check showed that **storage-forecast skill gives no price edge**: hit rates of
   0.47 Fri-to-Fri and 0.50 Thu-to-Fri (`docs/gas_research.md` section 4). The market prices the weather before we
   do.
2. **Implication for a live trading system.** A "beats-the-baselines" claim is credible and testable on
   **forecasting endpoints** with known power: weekly storage deviation, daily return distribution (QS/CRPS) and
   realized-vol forecasts. It is not credible, in any test shorter than about 2 years, on **P&L direction** (section
   6.2: 12 months of daily paper trading can only detect an annual Sharpe of about 1.65 at 50% power).
   Pre-register both, and headline only what the sample can support.
3. **The tradable effects with real support** (net of costs) are few:
   - (a) the **negative roll yield / contango** that steadily drains long UNG/BOIL. Shorting it pays carry but has
     catastrophic tails.
   - (b) the **EIA announcement-day return premium** [PR].
   - (c) **volatility-managed sizing**, which reliably improves risk, but whose effect on Sharpe is contested
     [PR, contested].

   Weather-revision and storage-surprise effects are real but **contemporaneous**. Exploiting them needs speed or a
   paid consensus, so for us they serve as world-model inputs and pre-registered baselines, not as an assumed edge.
4. **Recommendation.**
   - **Instrument:** UNG, as the paper-traded instrument. Its price already embeds the roll, so ETF P&L is the true
     P&L with no splicing.
   - **Decision time:** daily at **15:30 ET**, filled at the 16:00 close, plus a weekly EIA-event leg.
   - **World model:** the existing storage-balance model, extended with GEFS ensemble weather, an LNG-feedgas state, a
     stochastic-volatility block and a small explicit causal graph.
   - **Policy:** a risk-constrained, vol-targeted mapping from the world model's predictive distribution. Model-based
     RL (Dreamer-style, trained in imagination) runs only as a pre-registered challenger, alongside DQN and PPO.
5. **Feasibility.**
   - **Within hours:** a credential-free *shadow* paper loop that logs forecasts and hypothetical fills for the
     baselines and the existing `wm_gas` nowcast.
   - **Once the user creates an Alpaca paper account themselves:** UNG/BOIL paper orders. No futures paper trading
     at Alpaca as of today; IBKR paper supports futures.
   - **About 2-4 weeks:** the extended world model, GEFS features, the RL challenger, audits and the pre-registration
     freeze.
   - **About 12-24+ months:** statistical confirmation of P&L.

## 1. What is tradable and predictable at daily or weekly horizons, net of costs

| effect | evidence | quality | tradable for us, net of costs? |
|---|---|---|---|
| **EIA storage surprise** (actual minus analyst consensus) moves front-month NG on Thursday at 10:30 ET | Strong negative return/surprise relation [S3][S4][S7]; analyst consensus is accuracy-weighted, with a mean error of about 13 Bcf in 1997-2004 [S5] | PR (strong, contemporaneous) | **No, as a forecast.** The reaction takes seconds to minutes, and the consensus is paywalled (Reuters/Bloomberg/WSJ polls). Our weather-only nowcast had RMSE 22.6-23.6 Bcf on DEV [REPO], worse than the analyst error, so "our forecast minus avg5" is **not** the market surprise. |
| **Announcement-day premium**: a large share of NG futures' total return is earned on EIA release days | Prokopczuk, Wese Simen and Wichmann, Energy Journal 2021 [S2]: more than half the annual return falls on announcement days, not explained by the surprise | PR | **Plausibly.** It needs only a calendar: long front-month exposure from the Wednesday close to the Thursday close. Known risks: the sample is pre-2019, the effect may have decayed after publication [S30], and UNG's roll drag still applies. **A mandatory pre-registered baseline** (`eia_day_long`). |
| **Weather-forecast revisions** (GFS/ECMWF/GEFS HDD/CDD changes between runs) drive front-month moves | Trade press consistently attributes daily moves to run-to-run HDD changes, e.g. "15-20 HDD colder" [S6b]. Temperature shocks affect NG futures returns [S7]; weather affects volatility [S6]. **I found no peer-reviewed study** isolating model-run revisions as a predictor of next-day returns. | PR for a contemporaneous link; BLOG for model-run attribution; no evidence of under-reaction | **Unproven.** NG trades about 23 h a day on Globex, so 00z/06z/12z runs are priced before UNG trades. Tradable only if the market under-reacts, and nobody has shown that. Use as a world-model input and as the `wx_revision` baseline. |
| **LNG feedgas / outages** (e.g. Freeport, June 2022) | Freeport fire, 2022-06-08: about 2 Bcf/d of feedgas lost; the front month fell 6.4% that day; HH spot fell from $9.46 to $7.72 over the week [S8][S9] | OFF (EIA) + BLOG (S&P/trade press) | **Event-driven, rare, and instantly priced.** Its value is as a world-model **state** (exports drain storage; outages are bearish for HH and bullish for TTF) and for scenarios, not for timing. A free daily feedgas series does not exist (section 2). |
| **Roll yield / contango** (UNG/BOIL decay) | Mechanism is textbook. UNG lost about 89% over 10 years and BOIL about 99.98%, both with reverse splits [S10][S11]. Spot and roll both drive commodity momentum [S12]. | BLOG for the magnitudes; mechanism is PR (Erb and Harvey 2006 [S13]) | **Yes, but dangerous.** A short-UNG carry earns the drag. Its tails are huge (winter squeezes; 2021-22), it pays borrow fees, and BOIL adds daily-reset volatility decay. **A mandatory baseline** (`short_ung_carry`, vol-targeted). Using UNG prices as the P&L series removes all roll-splicing artifacts. |
| **CFTC COT positioning** | Net positions do not lead NG returns; correlations are contemporaneous, or returns lead positions [S14]. Hedging pressure does price commodity futures [S15], but evidence specific to NG is weak and dated. | PR (mixed or negative for NG) | **Unlikely.** COT is a slow covariate: Tuesday positions released Friday 15:30 ET. Include it only as a low-weight input. |
| **Seasonality** | A 2026 out-of-sample test on 15 commodities including NG, with costs: no seasonal model beats the equal-weight long benchmark after Holm adjustment [S16]. Gas *volatility* and storage seasonality are strong [S6][REPO]. | PP for return seasonality (negative); PR for vol seasonality | **Returns: no.** Volatility and storage: yes, as structure inside the world model. Keep `seasonal_rule` as a baseline. |
| **Time-series momentum** | TSMOM is significant across 58 futures including commodities [S12] (JFE 2012). Replications show it weaker after 2009 [S12b]. | PR (portfolio); weak for a single contract | **Weak for one market.** A mandatory baseline (`tsmom_252_21`, plus a fast `tsmom_20`). |
| **Realized-volatility forecasting for sizing** | Volatility clustering, GARCH/HAR [S17]. Prior gas VAL: models reach 0.69-0.75 on 5-day vol direction [REPO, VAL only]. Volatility-managed portfolios raise Sharpe [S18], but out of sample the gain is fragile [S19]. | PR | **Yes, for risk.** Vol targeting reliably stabilizes drawdowns and the leverage of BOIL-like exposure. Treat any Sharpe gain as a hypothesis. |

**Net-of-costs reality check [MINE].** UNG is about $9.08 with roughly 27M shares a day. A 1-cent spread is about 11 bp
(5.5 bp per side), and for BOIL ($16.11, about 7M shares a day) it is about 6 bp. A daily-flip strategy (about 250
round-trips a year at about 11-15 bp each) pays **about 25-40% a year** in spread and slippage on gross exposure.
Any daily signal must therefore clear a large hurdle. Prefer low-turnover rules, no-trade bands, and the weekly
event leg.

**Is there any free proxy for the storage consensus?**
- No reliable one.
- Options I found:
  - Economic-calendar sites (Investing.com, ForexFactory) show a "forecast" column with visible history. Its
    terms of use are restrictive, scraping is fragile, and the as-issued integrity is unverified [BLOG].
  - Trade-press articles (FXEmpire, Nasdaq/RTT) quote the Reuters consensus after the fact [BLOG].
  - Kalshi lists an HH price contract but no storage contract [CHK, S20].
  - ICE lists EIA-storage-linked futures [OFF], but they are not free data.
- Recommendation: do not build on a consensus proxy. Define surprise against our own ex-ante world-model forecast,
  and say plainly that this is not the market surprise. Optionally the user may **hand-log** the Reuters/WSJ
  consensus each Wednesday from public headlines, as a forward-only covariate with timestamps.

## 2. Free, live data with point-in-time (as-published) history

All endpoints were probed on 2026-11-01 unless noted. Key handling: **all keys come from environment variables set by
the user** (e.g. `EIA_API_KEY`, `ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY`, IBKR via local TWS/Gateway login). Code
never stores or prints keys, and I entered none.

| source | URL / API | key? | update time (ET) | as-issued history? | leak risk and rule |
|---|---|---|---|---|---|
| **EIA WNGSR** (L48 + 5 regions) | https://ir.eia.gov/ngs/ngs.html (wngsr.csv/json, `ngshistory.xls`); revisions: https://ir.eia.gov/ngs/revisions.html (`revisions.xls` `original_data`); schedule: https://ir.eia.gov/ngs/schedule.html | no | Thu 10:30; holiday weeks shift [OFF] | **Yes, partly.** First prints are in `revisions.xls` from 2015-06-19 [REPO]. Weeks after the file's last update are unrevised, so the current vintage equals the first print. **Going forward: snapshot `wngsr.json` each Thursday at 10:31 with a sha256** to build our own vintage archive. | High: revisions (≥4 Bcf threshold), reclassifications, holiday release dates, the 2025 shutdown [REPO]. Rule: use the actual release datetime per week from the schedule archive. |
| **EIA API v2** | https://api.eia.gov/v2/ (e.g. `/natural-gas/stor/wkly/`, `/natural-gas/prod/`, `/natural-gas/move/`, STEO `/steo/`) | **yes, free** (register at eia.gov/opendata). Without a key it returns `API_KEY_MISSING` [CHK] | same as the underlying series | **No.** The API serves the **current vintage only**, with no release timestamp field. | High for anything revised (914 production, monthly supply/demand). Use only for unrevised or slowly revised series, or pair it with our own daily snapshots. Env var `EIA_API_KEY`. |
| **EIA STEO** (monthly balances, price forecasts) | https://www.eia.gov/outlooks/steo/ ; archives https://www.eia.gov/outlooks/steo/outlook.php (past editions) | no | about the 2nd Tuesday monthly | **Yes:** each archived edition is an as-issued vintage [OFF] | Low if each edition is keyed by its release date. A good slow prior for production and LNG exports. |
| **EIA-914 production** | https://www.eia.gov/petroleum/production/ ; API `/natural-gas/prod/` | API key | end of month, for M-2 | **Archived monthly report files only** (no API vintages) | High: revisions of 1-2% are common. Use at least a 2-month lag, or STEO vintages instead. |
| **EIA Natural Gas Weekly Update** (incl. LNG feedgas weekly averages, sourced from S&P Global pipeline data) | https://www.eia.gov/naturalgas/weekly/ ; archive https://www.eia.gov/naturalgas/weekly/archivenew_ngwu/YYYY/MM_DD [OFF] | no | Thu, about 13:00-14:00 | **Yes:** each dated archive page is as-issued text and tables | Low if keyed by publication date. **The best free feedgas proxy with history.** It is weekly and must be parsed from HTML. Licensing: the data is S&P's, so do not redistribute raw numbers. |
| **LNG feedgas, daily** | No free official daily series. Options: pipeline EBB nominations (Kinder/Williams/Cheniere-adjacent pipelines; scattered and fragile); paid vendors (S&P, SynMax, Kpler) [BLOG, S21] | n/a | n/a | n/a | Rely on weekly NGWU, plus outage **events** from news/GDELT and company 8-Ks, timestamped. |
| **Weather: CPC degree-day forecasts** | https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_forecasts_7day/YYYY/MM/DD/UtilityGas.Heating.txt [CHK] | no | issued 00 UTC, posted about 10:00-11:00 | **Yes:** 2014 onwards, genuine ex-ante issues, with Last-Modified checked [REPO] | Low (already gated in `tools/gas/storage.py`). Only 7 days ahead, and deterministic. |
| **Weather: GEFS operational archive** (31 members, about 35 days ahead) | AWS `s3://noaa-gefs-pds/` (gefs.YYYYMMDD/{00,06,12,18}/atmos/...) [CHK: earliest **2017-01-01**]; byte-range GRIB via `.idx` files keeps downloads small | no | about 3.5-5 h after each cycle [OFF] | **Yes:** archived real-time runs = ex-ante | Medium. S3 LastModified reflects the bulk upload, not real-time availability, so apply a conservative availability of cycle + 6 h. **Model change: GEFSv12 in Sep 2020** (a regime break). Large dataset: the data lead must subset (T2m at about 50 population/gas-weighted cities; daily max/min). |
| **Weather: GEFSv12 reforecast** | AWS `s3://noaa-gefs-retrospective/` (2000-2019; 5 members daily, 11 weekly) [OFF, S22] | no | static | Ex-ante by construction (the frozen v12 model rerun) | Low. It gives a consistent long history for training the forecast-revision features. |
| **Weather: GFS deterministic** | AWS `s3://noaa-gfs-bdp-pds/` [CHK: archive from **2021-02**]; NOMADS https://nomads.ncep.noaa.gov (about the last 10 days only) | no | about 3.5-4.5 h after each cycle | Yes (from 2021) | As for GEFS. |
| **Weather: ECMWF open data** (IFS/AIFS, CC-BY-4.0) | AWS `s3://ecmwf-forecasts/` [CHK: archive from **2023-01-18**] | no | about 7-9 h after each cycle | Yes, from 2023 | Short history: a forward-only feature, or 2023 onwards. |
| **Weather: Open-Meteo** | Historical Forecast API (2021+, stitched first hours, **near-analysis, not ex-ante multi-day**); Previous Runs API (fixed lead 1-7 days; 2024+, GFS T2m 2021-03+); Single Runs API (ECMWF IFS 2024-03+, others 2025-09+) [OFF, S23] | no for non-commercial use | n/a | **Previous Runs / Single Runs: yes.** Historical Forecast: **no** (it leaks) | High if the Historical Forecast endpoint is used for multi-day leads. **Licensing:** free tier is non-commercial; real-money trading needs a paid plan. |
| **Realized weather** | CPC observed degree days [REPO]; ERA5 (via Open-Meteo archive) | no | next day (CPC) | observations (preliminary, revised) | Observed day-d values only from d+1; never use them at leads > 0. |
| **NG front month NG=F; UNG, BOIL, KOLD, UNL** | Yahoo `query1.finance.yahoo.com/v8/finance/chart/{sym}` [CHK] | no (unofficial; personal-use terms) | real-time / EOD | Prices are not revised, but adjusted closes change with splits/dividends (BOIL has had reverse splits) | Medium: store **raw** OHLCV plus split events, compute returns ourselves, and keep daily snapshots. NG=F roll days are artifacts (mask). |
| **Contract-level NG futures** | Yahoo serves **live** contracts (`NGZ26.NYM`, `NGF27.NYM` work) but **expired ones return "Not Found"** [CHK]. EIA RNGC1-4 stopped 2024-04-05 [REPO]. CME DataMine is paid. | no | EOD | **Not for history.** Start a daily snapshot of the next 12-24 contracts now, to build a curve archive forward. | Historical curve slope proxies: EIA RNGC1-4 (to 2024-04); **UNG vs UNL** (front vs 12-month strip) return spread; the slope of live contracts going forward. |
| **ETF roll schedules** | UNG rolls over about 4 days, about 2 weeks before NG expiry (prospectus; verify the current one) [BLOG]. BOIL tracks the Bloomberg Natural Gas Subindex (BCOM roll, 6th-9th/10th business day; verify) [OFF/BLOG] | no | monthly | deterministic | Use roll calendars as features. They matter for UNG-vs-futures tracking. |
| **CFTC COT** (disaggregated, NAT GAS NYME code 023651) | Socrata: `https://publicreporting.cftc.gov/resource/72hh-3qpy.json?cftc_contract_market_code=023651` [CHK: latest row is report date 2026-10-27] | no (an app token is optional) | **Fri 15:30 for Tuesday positions**; holiday delays; publication was interrupted 2025-10-01..11-12 [OFF] | **No vintage field:** corrections overwrite rows. The release datetime must be reconstructed from the release-schedule page. | Medium: never use Tuesday data before Friday 15:30, and snapshot weekly going forward. |
| **Baker Hughes rig counts** | https://rigcount.bakerhughes.com (xlsx history) | no; **scripts are blocked** (HTTP 403 to curl [CHK]) | Fri 13:00 | Weekly levels are rarely revised | Low leak risk and low value at daily horizons (production lags rigs by months). Optional manual download by the data lead. |
| **SEC EDGAR** (EQT, CTRA, AR, RRC, EXE (ex-CHK), LNG, TELL, NFE, utilities) | `https://data.sec.gov/submissions/CIK##########.json` [CHK, works with a descriptive User-Agent]; XBRL `companyfacts`/`frames`; full-text search efts | no; User-Agent with contact required; ≤10 req/s [OFF] | continuous; every filing has an **acceptance datetime** | **Yes:** the acceptance timestamp is the as-published time | Low if keyed by `acceptanceDateTime`. **XBRL companyfacts includes later restatements**, so filter by `filed`. Use 8-K items (e.g. 1.05/7.01/8.01, outages and force majeure) as event flags. |
| **News (GDELT)** | DOC 2.0 API `api.gdeltproject.org/api/v2/doc/doc` (rolling about 3 months; ≤1 request / 5 s [CHK]); GKG/Events raw 15-min files (2015+; large) | no | 15 min | Crawl timestamps are as-observed, but source articles can be edited, tone models have changed, and deduplication varies | Medium: use **counts of outage keywords** ("Freeport LNG", "force majeure", "feedgas", "Sabine Pass") as event intensity, lagged by 1 h. No sentiment model fit on future data. |

**Rules carried over and added.** Every series is stored as `(value, observed_at_ET, source_vintage_sha256)`, and the
feature builder only reads rows with `observed_at <= decision_time`. This is the same information-set perturbation
test as `tools/gas/test_infoset.py`, extended to every new source. A live system creates its own clean vintage
archive from day 1, which is the main structural advantage of going live.

## 3. "N-order implications": graph and supply-chain propagation

**Credible [PR].**
- Shocks propagate along production networks: Acemoglu et al. 2012 [S24]; Barrot and Sauvagnat 2016, where natural
  disasters at suppliers hit customers [S25].
- Markets **under-react to linked-firm news**: Cohen and Frazzini 2008, customer-supplier momentum [S26]; Menzly and
  Ozbas 2010, cross-industry momentum [S27]. These are monthly, cross-sectional equity effects and have decayed after
  publication [S30].
- Gas producer equities load heavily on HH returns. This is the obvious first-order link and needs no graph to find.

**Hype or unproven [PP].**
- A wave of 2025-26 preprints builds LLM-extracted knowledge graphs or GNNs for stock prediction [S28].
- Common flaws:
  - graphs built with future information (full-sample correlations, today's supplier lists applied to past dates);
  - short test windows;
  - no costs;
  - no comparison against a simple sector-beta model.
- Only a few build **point-in-time** graphs from filing dates, e.g. [S29]. That is the correct design.

**For natgas specifically**, the chain LNG outage → feedgas ↓ → storage ↑ vs norm → HH ↓ → producers ↓ / TTF ↑ /
LNG exporter (LNG) ↓ is **physically real and contemporaneous**. HH and producers reprice within minutes (Freeport:
-6.4% on day 1 [S9]), so the 2nd-to-3rd-order links offer little *timing* alpha. Their value lies in:
- consistent scenario analysis, the thing a direct model cannot do;
- sizing exposure across correlated holdings;
- possibly slower-moving nodes (utilities, midstream) where under-reaction is plausible but unproven.

**Recommended implementable design** (fits easily in 8 GB on an M3; about 10^4 parameters; numpy/PyTorch CPU or MPS):

1. **Explicit causal DAG, hand-specified from physics. Not a learned GNN.**
   - Nodes (daily): weather demand (gas-weighted HDD, population CDD, from forecast ensembles), power burn,
     res/comm demand, dry production, LNG feedgas, Mexico exports, storage, HH front-month return, curve slope (front
     minus 12-month), realized vol.
   - Weekly: EIA print, COT.
   - Equities (daily returns): EQT, CTRA, AR, RRC, EXE, LNG, TELL, NFE, plus 2-3 gas utilities (e.g. ATO, SWX) and
     XLE as control.
2. **Edges and parameters.** The physical block is the existing storage-balance world model
   (`tools/gas/wm_gas.py`). The price block maps the **world-model storage-surprise distribution and the
   weather-forecast revision** to HH return mean and volatility, through a small learned head. The equity block uses
   **rolling 252-day betas** of each equity to HH and XLE, re-estimated only from past data. The beta matrix is the
   graph; no edge is learned from the future.
3. **Propagation.** For a scenario (e.g. "Freeport offline: feedgas -2 Bcf/d for 3 weeks"), roll the world model
   daily, with 1,000 paths from the GEFS member spread. Push the HH path through the betas. Output distributions for
   storage vs norm, HH, UNG/BOIL and each equity.
4. **Testable claim, pre-registered.** On historical outage and cold-snap events (Freeport 2022-06 and 2023;
   Uri 2021-02; Elliott 2022-12; Jan 2024 and Jan 2025 cold), check that the sign and magnitude of the scenario
   predictions lie inside the 80% interval more often than a sector-beta-only model. This is descriptive (few
   events), never a P&L claim.
5. **Equity trading** stays out of scope for v1. It serves as a sanity check and for risk only.

## 4. World models, Dreamer and model-based RL, and continual learning in finance

**Evidence of real out-of-sample success.**
- The Dreamer line (DreamerV3 [S31]) and offline model-based RL (MOPO [S32], MOReL [S33]) work on simulators and
  benchmarks. **I found no peer-reviewed, live, cost-inclusive trading result for Dreamer-style agents.**
- Finance RL papers (DQN/PPO/actor-critic) are overwhelmingly preprints with:
  - short backtests (often 1-2 years);
  - no or toy costs;
  - selection of the best of many runs.
- Examples: [S34] reports Sharpe of about 1.3-1.4 on a 2019-2020 equity backtest, a strong bull window, with high
  beta. [S35] compares DDQN/PPO to buy-and-hold on 2019-2023. The survey of 167 RL-finance papers [S36] (ACM CSUR) is
  the best map of the field.
- A 2026 crypto benchmark explicitly measures the **backtest → paper → live gap** [S37] and treats it as the central
  open problem.
- Market-impact-aware RL environments show that cost-free training produces strategies that fail with costs [S38].

**Known failure modes and the mitigation in this design.**

| failure | evidence | mitigation |
|---|---|---|
| Backtest overfitting / multiple testing | PBO and the Deflated Sharpe ratio [S39][S40]; "...and the cross-section" t > 3 [S41]; in-sample Sharpe barely predicts OOS Sharpe in large strategy samples [S42, PP] | Pre-register K (number of configurations), report DSR, run a single frozen TEST, and score forward paper as the only P&L evidence |
| Post-publication / regime decay | Anomaly returns are 26% lower out of sample and 58% lower after publication [S30] | Assume the literature effects (announcement-day premium, TSMOM) are halved; budget for it |
| Regime shift | Gas: GEFSv12 model change in 2020; LNG export growth from 2016 to 14-16+ Bcf/d (with Golden Pass and Plaquemines ramping); 2022 volatility; 2025 shutdown data gaps [REPO] | Train on an expanding window with recency weighting; regime and vol state in the world model; drift monitors (section 6) |
| Model exploitation in imagination (Dreamer/MBRL) | The policy exploits world-model errors; MOPO/MOReL penalize ensemble disagreement [S32][S33] | Ensemble world model (≥5 members), reward penalized by disagreement, rollout horizon ≤ 10 days, and the policy action space restricted to {-1, -0.5, 0, 0.5, 1} × vol-target |
| Too little data for RL | About 2,500 daily observations since 2017 (GEFS); fewer than 10 winters | The RL policy is a *challenger*. The champion is a closed-form decision rule on the world-model distribution (section 5.4). |
| Continual learning: catastrophic forgetting vs stale models | Concept drift is the norm in markets (adaptive-markets view) | **Scheduled, versioned refits:** weekly recalibration of the heads (quantile calibration on the trailing 52 weeks), monthly full refit on an expanding window. Each version is frozen with a sha256 *before* it trades. No online gradient step touches live decisions without a version bump. Bayesian online changepoint detection [S43] on forecast residuals only *triggers a review*, never an automatic architecture change. |

**Lesson from this repo [REPO].** The world-model rollout did not beat its direct twin. For Alma-Live the claim
must be decided again by a matched-twin test. The world model's distinct value is (i) scenarios and (ii)
imagination rollouts for policy training. Neither has to be a forecasting win, but neither can be sold as one
without the test.

## 5. Recommendation (concrete, pre-registrable)

### 5.1 Targets (forecasting, primary for claims)

| id | target | origin | metric | power (forward) |
|---|---|---|---|---|
| **L1** | weekly storage deviation dS_W - avg5_W (L48; 5 regions secondary), first print | Wed 15:30 ET (h = 0 nowcast) and the previous Thu 15:30 (h = 1) | QS and MAE (Bcf), direction; vs st_ols_wx, wm_gas (frozen prior), and FMs | 52 weeks/yr; the prior TEST showed QS gains detectable at n = 52 [REPO] |
| **L2** | next-day UNG log return distribution (close → close) | 15:30 ET | CRPS/QS vs a Gaussian random walk with EWMA vol and vs GARCH-t; direction as a **negative control** | about 250/yr; continuous loss, DM-HLN |
| **L3** | 5-day realized variance of UNG (and NG=F) | 15:30 ET | QLIKE vs GARCH(1,1), GJR, HAR-RV (daily), EWMA | about 50 non-overlapping/yr; daily overlapping with HAC |
| **L4** | EIA release-day UNG return (Wed close → Thu close) | Wed 15:30 | sign and QS | 52/yr (low power; descriptive) |

### 5.2 Decision time, instrument, execution

- **Instrument.**
  - **UNG**, long or short, for v1. It is liquid (about 27M shares a day), unlevered, and its price embeds the roll,
    so its P&L is exact.
  - **BOIL/KOLD only as a sizing alternative.** Their daily-reset leverage decays, so a 2x position should normally
    be expressed as 2x UNG notional (needs margin), not as BOIL held for days.
  - **MNG/QG micro and e-mini futures** (MNG: 1,000 MMBtu, $0.001 tick = $1 [S44]) only in phase 2, through IBKR.
- **Decision time.** Every NYSE trading day at **15:30 ET**:
  - all inputs are stamped ≤ 15:15 ET;
  - the NG settle (14:30 ET) and the 12z GEFS (if posted) are included;
  - the order is MOC or limit-at-close for the 16:00 close; the paper fill is the official close ± cost.
- **Event leg.** The Wednesday 15:30 decision may add the `eia_day` exposure, chosen by the policy and capped at
  0.5 × vol-target.
- **Position limits.** |position| ≤ 1.0 × vol-target notional (target 20% annualized portfolio vol). Hard caps:
  - gross ≤ 100% of equity in UNG;
  - no position over the NG expiry week unless the policy is explicitly roll-aware;
  - no trading on half days.
- **No-trade band.** Trade only if |target - current| > 0.25 × vol-target, which cuts turnover.

### 5.3 Cost model (pre-registered; recalibrated from paper fills, never loosened)

- **UNG.** Half-spread is max(0.5 cent, observed quoted half-spread at 15:55), about 5.5 bp. Slippage at the close
  is 3 bp, commission $0 (Alpaca) or IBKR tiered (about $0.0035/share). Total **about 10 bp per side, 20 bp round
  trip**. Short borrow: assume 1% a year unless the broker quotes more; HTB means no short.
- **BOIL/KOLD.** About 6 bp half-spread plus 3 bp. The expense ratio of about 0.95% a year is already in the price.
- **Futures (phase 2).** 1 tick plus about $1.5-2.5 per side all-in (exchange plus broker) per NG-equivalent.
  Model expiry rolls explicitly.
- **Stress test.** Report every P&L at 1×, 2× and 3× costs. A strategy that dies at 2× is not deployable.

### 5.4 World model (Alma-Live WM) and policy

**State (daily latent):**
- storage S (L48 + 5 regions, mass-balanced);
- production level;
- LNG feedgas level plus an outage indicator;
- weather-demand sensitivity (convex in HDD, piecewise in CDD);
- power-burn regime;
- log-volatility h;
- curve-slope state.

**Inputs at decision time:**
- *Weather:* the GEFS 31-member HDD/CDD paths for days 1-15 (gas- and population-weighted), and the **revision**
  versus the previous day's same-cycle run (and versus 24 h ago). CPC 7-day forecasts are a fallback, available back
  to 2014.
- *Fundamentals and market:*
  - weekly NGWU feedgas and outage flags (GDELT counts, 8-K items);
  - COT net managed money (lagged to its Friday release);
  - roll calendar;
  - HH, UNG and UNL returns.

**Transitions:**
- S_{d+1} = S_d + prod - demand(wx, z) - feedgas - other + ε. Weekly sums are observed against the EIA first print.
- h_{d+1} = AR(1) with seasonal mean and loadings on the storage-deviation, weather-revision and EIA-day dummies.
- The UNG return is t-distributed. Its mean comes from a small head on (Δ E[storage surprise], weather revision,
  slope/roll), and its scale is exp(h).

**Ensemble and calibration:** 5 seeds, plus split-conformal calibration of the quantiles on the trailing 52 weeks.

**Matched direct twin:** the same inputs mapped straight to each target, with the same parameter budget, as in the
prior project.

**Policy (champion).** A closed-form rule on the world model's predictive distribution:

    w = clip( k × E[r_{t+1}] / Var[r_{t+1}], -1, 1 ) × vol_target / σ̂

- It applies only when the conformal 80% interval of r excludes zero by more than the cost, or when the
  storage-surprise posterior is decisive; otherwise w = 0.
- k = 0.25 (fractional Kelly).
- The no-trade band from section 5.2 applies.

**Challengers (pre-registered, all run in the same paper loop):**
- **RL-WM**: PPO trained in world-model imagination with a MOPO-style disagreement penalty.
- **DQN** and **PPO** trained on raw features against a historical backtest environment with the cost model.
- **LightGBM** and **ridge** on the same features → sign → the same sizing rule.

### 5.5 Baselines (all must be in the forward test, same cost model)

| id | rule | reference |
|---|---|---|
| `buy_hold_ung` | long UNG 1× | — |
| `cash` | flat | — |
| `short_ung_carry` | short UNG, vol-targeted, flat in Dec-Feb if storage deficit > 1 sd (pre-declared variant) | roll-yield mechanism [S13] |
| `tsmom_252_21`, `tsmom_20` | sign of past 12-1 month or 20-day UNG return × vol-target | [S12] |
| `seasonal_rule` | long Sep-Dec, short Jan-Mar/Apr (fixed calendar; no fitting) and DVR seasonal | [S16] |
| `eia_day_long` | long from Wed close to Thu close only | [S2] |
| `wx_revision` | sign(Δ GEFS-mean 1-15 d gas-weighted HDD, winter; Δ CDD, summer) vs the previous day's 12z run, × vol-target | trade-press mechanism [S6b] |
| `storage_surprise_rule` | sign(-(world-model or OLS nowcast of dS_W - avg5_W)), held Wed close → Thu close | [S3][S5]; our nowcast stands in for the paywalled consensus |
| `lgbm_feat`, `ridge_feat` | same feature set as the world model, direct | — |
| `dqn_raw`, `ppo_raw` | standard implementations (stable-baselines3), same environment and costs | [S34][S35] |
| FMs | **Chronos-2** [S45], **TiRex** [S46], **TimesFM 2.x** [S47], **Kronos** (OHLCV K-line model) [S48] → quantiles → the same sizing rule; zero-shot, contamination flagged by pretraining cut-off | [REPO harness] |
| vol baselines | GARCH(1,1)-t, GJR, HAR-RV, EWMA(0.94) | [S17] |
| prior Alma | frozen `wm_gas` / `wm_gas_direct` for L1 | [REPO] |

### 5.6 Pre-registrable forward-test plan

1. **Freeze** (commit with sha256):
   - the code, configs and world-model checkpoints;
   - the cost model;
   - the baselines list;
   - the decision schedule;
   - the metrics;
   - K, the number of configurations tried in development.
2. **Start date and duration** are fixed in advance.
   - **Primary forecasting analysis at 52 weeks (L1-L3).**
   - P&L analysis at 12 and 24 months.
   - No early stop for success. Early stop for harm only, via the kill rules in section 6.1.
3. **Primary claims**, with Holm correction across the pre-declared set:
   - **L1 QS:** WM vs st_ols_wx and the best FM.
   - **L2 CRPS:** WM vs RW-EWMA and vs GARCH-t.
   - **L3 QLIKE:** WM vs GARCH.
   - **WM vs twin** on L1-L3. This decides the "world model" claim.
4. **P&L claims (secondary).** The net Sharpe of the champion vs each baseline, using:
   - the Ledoit-Wolf Sharpe-difference test, or a stationary bootstrap;
   - the Probabilistic/Deflated Sharpe ratio [S39][S40] with K disclosed;
   - the 2× cost sensitivity.

   The power table in section 6.2 is printed next to every number.
5. **Monthly refits** are allowed only by a frozen *procedure* (window, hyperparameters, seeds). The refit code is
   part of the freeze. Every live decision logs the model version hash.
6. **Reporting.** An append-only `runs/live_gas/decisions.jsonl` records the inputs' vintage hashes, the forecast,
   the target position, the paper fill and the realized return. A weekly auditor re-derivation recomputes the
   metrics from the raw snapshots. Shadow mode (no broker) and broker paper are logged separately; the broker fills
   calibrate the cost model.

## 6. Real-money readiness

### 6.1 Requirements before any real money

**Brokers with paper modes:**
- **Alpaca paper:** US equities and ETFs (UNG, BOIL, KOLD) through a REST API; free. **No futures today.** Alpaca
  Derivatives registered as an FCM in Aug 2026 but has not started operations, and its stated focus is prediction
  markets [S49][BLOG].
- **IBKR paper:** a full simulated account through TWS/IB Gateway (local login by the user). It supports **futures
  (NG, QG, MNG), ETFs and options**. Fills are simulated from top of book and are optimistic for size. Market-data
  subscriptions are needed for futures.
- **Others:** Tradier (equities/options), TradeStation (futures, paper), NinjaTrader/Tradovate (futures). Not
  evaluated in depth.

**Risk limits (hard-coded, outside the model):**
- max position (gross and net);
- max daily loss of 2% of equity, which flattens and halts for the day;
- max drawdown of 10%, which flattens and halts until a human review;
- max orders per day (5);
- price-collar orders (limit at close ±1%);
- no trading when any input is stale beyond its SLA (e.g. GEFS missing, price feed delayed, EIA release not
  ingested), or when the world model's ensemble disagreement exceeds its 99th percentile in-sample.

**Kill switch:** a file flag plus a CLI command plus an automatic trigger on the breaches above. It cancels all open
orders, flattens, and requires a manual reset. Test it weekly in paper.

**Slippage and cost monitoring:** compare each paper fill with the 16:00 official close and the quoted spread; the
cost model is re-estimated monthly from paper fills (may only get more conservative).

**Operations:**
- *Scheduling:* launchd/cron on the Mac, plus a cloud fallback later.
- *Clock:* NTP-synced; all timestamps in UTC with ET derived.
- *Idempotent order IDs;* reconcile broker positions daily.
- *Holidays:* NYSE and CME holiday calendars, and the EIA schedule.
- *Logging and alerting:* email or push on any halt.

**Compliance:** personal account only; no redistribution of S&P-sourced NGWU numbers or licensed weather data;
Open-Meteo commercial terms if used.

### 6.2 How long must the forward paper test run? [MINE]

With iid daily returns, the t-statistic of the Sharpe ratio is about SR_annual × √years. A one-sided test at α =
0.05:

| forward months | min SR detectable (50% power) | min SR at 80% power |
|---|---|---|
| 3 | 3.29 | 4.97 |
| 6 | 2.33 | 3.52 |
| 12 | 1.65 | 2.49 |
| 18 | 1.34 | 2.03 |
| 24 | 1.16 | 1.76 |
| 36 | 0.95 | 1.44 |
| 60 | 0.74 | 1.11 |

Minimum track record length (Bailey and López de Prado [S39]; skew 0.3; it barely changes for kurtosis 3 to 10),
needed to reject SR ≤ 0 at 95%:

| true SR | MinTRL |
|---|---|
| 0.5 | about 129 months |
| 1.0 | about 32 months |
| 1.5 | about 14 months |
| 2.0 | about 8 months |
| 3.0 | about 3.5 months |

Comparing against a correlated baseline (e.g. vs buy-hold) needs longer still. **Conclusion.** For a realistic
single-market daily strategy (true net SR 0.5-1.0), **no paper test of practical length can establish the edge
statistically**. Forward paper testing is for verifying operations, costs and the absence of leakage (live ≈
backtest), and for testing the forecasting endpoints, which do have power at 52 weeks. Real money, if ever, should
start small, sized so that a full drawdown is acceptable, with a pre-declared "decay" rule. For example: stop if
the 6-month live Sharpe falls below backtest SR - 2/√0.5.

## 7. Feasibility

**Paper-deployable within hours (no new credentials, local only):**
1. Fetchers with vintage snapshots:
   - Yahoo daily OHLCV (UNG, BOIL, KOLD, UNL, NG=F, live NG contracts);
   - CPC 7-day forecasts;
   - the EIA WNGSR html/json (no key);
   - the CFTC Socrata endpoint;
   - EDGAR submissions for 10 tickers.
2. **Shadow paper loop** (cron at 15:30 ET), logging decisions and hypothetical close fills with the cost model.
   Rules:
   - `buy_hold`, `short_ung_carry`, `tsmom`, `seasonal_rule`, `eia_day_long`;
   - a CPC-based `wx_revision` (7-day; a GEFS version comes later);
   - `storage_surprise_rule` from the **frozen prior `wm_gas`** nowcast, which already exists with an audited
     information set;
   - GARCH vol targeting.
3. A kill-switch and risk-limit module, plus the append-only decision log.
4. Broker paper (UNG/BOIL) as soon as **the user** creates an Alpaca paper account and sets
   `ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` (paper endpoint) as environment variables. I will not create accounts
   or enter keys.

**Needs days to about 2 weeks:**
- **GEFS subsetting pipeline:** byte-range GRIB2 for T2m at about 50 cities, 4 cycles a day, 2017 onwards. That is
  tens of GB raw, but only about 100 MB after subsetting. It is a data-lead job, with a conservative cycle + 6 h
  availability gate.
- **NGWU archive parser** (feedgas weekly, as-issued).
- **COT release-datetime reconstruction.**
- **Point-in-time UNG/UNL curve proxy.**
- **Information-set perturbation tests** for every new source.
- **The extended world model and matched twin:** storage plus vol plus price head; ensemble and conformal
  calibration.
- **LightGBM/ridge/DQN/PPO baselines** in a cost-aware backtest environment.
- **FM wrappers** (Chronos-2, TiRex, TimesFM, Kronos) adapted to the daily UNG target.

**Needs about 2-4 weeks:**
- the RL-in-imagination challenger with a disagreement penalty;
- historical DEV/VAL/TEST evaluation under the prior project's protocol (prereg, audit GO, single frozen TEST);
- the scenario engine for the causal graph and the event back-check;
- IBKR paper integration for MNG/QG (needs the user's own IBKR login).

**Needs months to years:**
- statistically meaningful P&L evidence (section 6.2);
- a contract-level curve history (we must snapshot forward; there is no free historical source);
- confirmation of any weather-revision under-reaction effect.

**Honest risks:**
1. The strongest prior is that **no daily price-direction edge survives costs** (prior DEV check; commodity round 1).
   The likely live outcome is that the world model wins on forecasting endpoints and the policy roughly matches
   vol-targeted carry/TSMOM baselines.
2. Short-UNG carry has the best historical net returns of the simple rules, but it can lose 50%+ in weeks during a
   winter squeeze.
3. Feedgas, the most important new state, has no free daily feed.
4. Open-Meteo, NGWU (S&P data) and Yahoo all carry licensing caveats for real-money or commercial use.

## Sources

Prior-project sources [S2]-[S7] are as in `docs/gas_research.md`; key ones are repeated here.

- [S2] Prokopczuk, Wese Simen, Wichmann, "The Natural Gas Announcement Day Puzzle", Energy Journal 42(2), 2021. [PR] https://ideas.repec.org/a/aen/journl/ej42-2-wichmann.html
- [S3] Linn, Zhu, J. Futures Markets 24 (2004), storage report and volatility. [PR]
- [S4] Chiou-Wei, Linn, Zhu, storage surprises and futures price changes (cited in [S7]). [PR]
- [S5] Gay, Simkins, Turac, J. Futures Markets 29(5) (2009), analyst storage forecasts. [PR] https://ideas.repec.org/a/wly/jfutmk/v29y2009i5p451-477.html
- [S6] Mu, "Weather, storage, and natural gas price dynamics", Energy Economics 29(1) (2007). [PR] https://iaee.org/en/students/best_papers/Xiaoyi_Mu2.pdf
- [S6b] NGI market reports attributing moves to GFS/ECMWF/GEFS HDD revisions. [BLOG] https://www.naturalgasintel.com/?p=257308 ; https://www.naturalgasintel.com/?p=209687
- [S7] Hu, Hu, Lin, temperature shocks and inventory surprises on NG/HO futures returns (2014). [PR] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4122141/
- [S8] EIA Natural Gas Weekly Update, 2022-06-16 (Freeport feedgas 1.83 Bcf/d; HH $9.46 → $7.72). [OFF] https://www.eia.gov/naturalgas/weekly/archivenew_ngwu/2022/06_16 ; EIA Today in Energy https://www.eia.gov/todayinenergy/detail.php?id=53079
- [S9] Gas Strategies pricewatch 2022-06-09 (July NG -6.4%); S&P Global 2022-06-14. [BLOG] https://gasstrategies.com/information-services/gas-matters-today/pricewatch-09-jun-2022-gas-matters-today ; https://www.spglobal.com/commodity-insights/en/news-research/latest-news/natural-gas/061422-freeport-lng-in-texas-wont-return-to-full-service-until-late-2022-operator
- [S10] 24/7 Wall St, UNG contango cost (2026-05-28). [BLOG] https://247wallst.com/investing/2026/05/28/ung-holds-front-month-natural-gas-futures-and-contango-has-cost-holders-90-percent-over-a-decade-without-a-single-price-drop/
- [S11] 24/7 Wall St, BOIL decay (2026-05-25). [BLOG] https://247wallst.com/investing/2026/05/25/boil-promises-2x-natural-gas-but-contango-has-eaten-99-percent-of-its-value-over-the-past-decade/
- [S12] Moskowitz, Ooi, Pedersen, "Time series momentum", JFE 104(2) (2012). [PR] https://econpapers.repec.org/RePEc:eee:jfinec:v:104:y:2012:i:2:p:228-250 ; [S12b] replication showing weaker post-2009 results (NOVA thesis, 2025). [PP] https://run.unl.pt/entities/publication/fdc6f042-44dd-42ee-869d-10b4dbce7dbf
- [S13] Erb, Harvey, "The strategic and tactical value of commodity futures", Financial Analysts Journal (2006). [PR]
- [S14] Sanders, Boris, Manfredo, "Hedgers, funds, and small speculators in the energy futures markets", Energy Economics 26(3) (2004). [PR] https://ideas.repec.org/a/eee/eneeco/v26y2004i3p425-445.html
- [S15] De Roon, Nijman, Veld, "Hedging pressure effects in futures markets", J. Finance 55(3) (2000). [PR]
- [S16] Kosch, Forsberg, "Seasonal Trading in Commodity Futures" (2026), arXiv 2609.12227 (15 futures incl. NG; no Holm-significant win over long benchmark with costs). [PP] https://www.alphaxiv.org/abs/2609.12227
- [S17] Corsi, HAR-RV, J. Financial Econometrics (2009) [PR]; Bollerslev GARCH (1986) [PR]; energy GARCH vs ML, arXiv 2405.19849 [PP].
- [S18] Moreira, Muir, "Volatility-managed portfolios", J. Finance 72(4) (2017). [PR]
- [S19] Cederburg, O'Doherty, Wang, Yan, "On the performance of volatility-managed portfolios", JFE 138(1) (2020). [PR]
- [S20] Kalshi NGAS contract terms (HH price, not storage). [OFF] https://kalshi-public-docs.s3.amazonaws.com/contract_terms/NGAS.pdf ; ICE EIA-linked contracts https://www.ice.com/products/69723154/US-EIA-Financial-Weekly-Index-100-USD
- [S21] SynMax feedgas product description. [BLOG/vendor] https://blog.synmax.com/hyperion-client/new-dataset-us-gas-demand
- [S22] NOAA GEFSv12 reforecast on AWS. [OFF] https://registry.opendata.aws/noaa-gefs-reforecast ; https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf ; operational GEFS https://docs.opendata.aws/noaa-gefs-pds/readme.html
- [S23] Open-Meteo Previous Runs and Single Runs APIs. [OFF] https://open-meteo.com/en/docs/previous-runs-api ; https://open-meteo.com/en/docs/single-runs-api ; https://openmeteo.substack.com/p/single-runs-api
- [S24] Acemoglu, Carvalho, Ozdaglar, Tahbaz-Salehi, "The network origins of aggregate fluctuations", Econometrica 80(5) (2012). [PR]
- [S25] Barrot, Sauvagnat, "Input specificity and the propagation of idiosyncratic shocks in production networks", QJE 131(3) (2016). [PR]
- [S26] Cohen, Frazzini, "Economic links and predictable returns", J. Finance 63(4) (2008). [PR]
- [S27] Menzly, Ozbas, "Market segmentation and cross-predictability of returns", J. Finance 65(4) (2010). [PR]
- [S28] Examples of 2026 LLM/KG/GNN propagation preprints: https://www.alphaxiv.org/abs/2606.29290 ; https://www.alphaxiv.org/abs/2607.10932 ; https://www.alphaxiv.org/abs/2603.24236 [PP]
- [S29] "LLM Latent Edge Measurement: Point-in-Time Economic Graphs ..." (2026). [PP] https://www.alphaxiv.org/abs/2607.15640
- [S30] McLean, Pontiff, "Does academic research destroy stock return predictability?", J. Finance 71(1) (2016). [PR]
- [S31] Hafner et al., DreamerV3, arXiv 2301.04104 (Nature 2025). [PR] https://www.alphaxiv.org/abs/2301.04104
- [S32] Yu et al., MOPO: model-based offline policy optimization, NeurIPS 2020. [PR] https://www.alphaxiv.org/abs/2005.13239
- [S33] Kidambi et al., MOReL, NeurIPS 2020. [PR] https://www.alphaxiv.org/abs/2005.05951
- [S34] "Deep RL for Equity Trading: Benchmarking Actor-Critic Methods with Forward Retraining" (2026). [PP] https://www.alphaxiv.org/abs/2609.31870
- [S35] "Can Artificial Intelligence Trade the Stock Market?" (DDQN/PPO vs buy-and-hold, 2025). [PP] https://www.alphaxiv.org/abs/2506.04658
- [S36] "The Evolution of Reinforcement Learning in Quantitative Finance: A Survey" (167 papers; accepted ACM CSUR). [PR] https://www.alphaxiv.org/abs/2408.10932
- [S37] "Can AI Make Money in Crypto? Measuring the Gap from Backtests to Real Markets" (2026). [PP] https://www.alphaxiv.org/abs/2609.34510
- [S38] "Realistic Market Impact Modeling for RL Trading Environments" (2026). [PP] https://www.alphaxiv.org/abs/2603.29086
- [S39] Bailey, López de Prado, "The Sharpe ratio efficient frontier", J. Risk 15(2) (2012), PSR and MinTRL. [PR] https://ideas.repec.org/a/rsk/journ4/2223785.html
- [S40] Bailey, López de Prado, "The Deflated Sharpe Ratio", J. Portfolio Management (2014) [PR] https://papers.ssrn.com/abstract=2460551 ; Bailey, Borwein, López de Prado, Zhu, "The probability of backtest overfitting", J. Computational Finance (2017). [PR]
- [S41] Harvey, Liu, Zhu, "... and the cross-section of expected returns", RFS 29(1) (2016). [PR]
- [S42] Wiecki et al., "All that glitters is not gold: comparing backtest and out-of-sample performance on a large cohort of trading algorithms" (Quantopian, 2016). [PP]
- [S43] Adams, MacKay, "Bayesian online changepoint detection" (2007). [PP] https://www.alphaxiv.org/abs/0710.3742
- [S44] CME Micro Henry Hub (MNG) contract specs. [OFF] https://www.cmegroup.com/markets/energy/files/getting-started-with-micro-henry-hub-natural-gas.pdf ; https://cmegroup.com/articles/faqs/micro-henry-hub-natural-gas-futures-and-options-frequently-asked-questions.html
- [S45] Ansari et al., Chronos-2, arXiv 2510.15821. [PP] https://www.alphaxiv.org/abs/2510.15821
- [S46] Auer et al., TiRex, arXiv 2505.23719. [PP] https://www.alphaxiv.org/abs/2505.23719
- [S47] Das et al., TimesFM, ICML 2024, arXiv 2310.10688. [PR] https://www.alphaxiv.org/abs/2310.10688
- [S48] Shi et al., Kronos: foundation model for financial K-lines, arXiv 2508.02739. [PP] https://www.alphaxiv.org/abs/2508.02739
- [S49] Alpaca FCM registration announcement (Aug 2026). [OFF/BLOG] https://alpaca.markets/blog/alpaca-registers-as-futures-commission-merchant-with-the-cftc-and-nfa-to-advance-launch-of-prediction-markets/ ; forum https://forum.alpaca.markets/t/futures-trade-in-alpaca/11632
- CFTC COT release timing and the 2025 interruption. [OFF] https://www.cftc.gov/node/128971 ; https://cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm ; https://mondovisione.com/media-and-resources/news/cftc-commitments-of-traders-reports-update-report-data-for-12092025-20251219/
- Live probes [CHK], 2026-11-01: `noaa-gefs-pds` earliest prefix gefs.20170101; `noaa-gfs-bdp-pds` from 2021-02; `ecmwf-forecasts` from 20230118; CFTC Socrata 72hh-3qpy NG row 2026-10-27; EIA API v2 → API_KEY_MISSING without a key; Yahoo NGZ26.NYM/NGF27.NYM served, NGV26.NYM/NGZ25.NYM not found; EDGAR submissions OK with a User-Agent; GDELT rate limit message; Baker Hughes 403 to scripts; UNG 9.08 (median vol 26.7M), BOIL 16.11 (7.0M), UNL 5.16 (64k), NG=F 2.898.
