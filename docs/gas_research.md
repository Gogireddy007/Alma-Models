# Alma-Gas: natural gas deep research (targets, data, information set, world-model designs)

Author: deep-research agent, 2026-10-31. Branch `alma/investor-sprint`. Status: v2 (complete). No model was trained and no TEST-period value (> 2025-10-31) was inspected. Files were downloaded
only to check URLs and formats; for files that contain post-2025-10-31 rows, only headers, date ranges or
pre-2025 rows were printed.

Bottom line up front: **daily/weekly Henry Hub price direction is very likely noise** (the oil round 1 finding,
`docs/audit_commodity.md`, carries over; no peer-reviewed gas paper reports a stable out-of-sample direction edge at
1-20 days). The places with documented, physically grounded predictability are (a) the **weekly EIA storage change
relative to its 5-year average** (driven by weather that is observed or forecast before the release) and (b)
**volatility direction** (strongly persistent and seasonal). A storage-balance world model fits (a) naturally.

## 1. Which targets are predictable? (evidence strength)

| target | evidence | strength | notes / typical numbers |
|---|---|---|---|
| HH spot/front-month price direction, 1-20 d | Efficient-market evidence: naive/RW is a "notoriously difficult" benchmark for financial series [S1]; oil round 1 here: RW won at 1-20 d (`docs/audit_commodity.md`). Gas futures returns are concentrated on EIA announcement days: >50% of the annual return is earned on them, and the pre-announcement drift is not explained by the surprise [S2]. | weak / noise | No reproducible, peer-reviewed out-of-sample hit rate > 55% found. Treat as a negative control. |
| Report-day futures return sign given storage surprise | Strong, significant negative relation between futures returns and storage surprise (actual minus analyst expectation) [S2][S3][S4]; spot reacts the day after the release [S4]. Market conditions on individual analyst forecasts weighted by past accuracy [S5]. | strong (contemporaneous) | It is a reaction to news, not a forecast: it requires the consensus, which is not free (Bloomberg/Reuters/WSJ polls). Without consensus, it cannot be scored honestly. |
| Weekly storage change (net injection/withdrawal) and its deviation from the 5-yr average | Storage is weather-driven; weather shocks move both mean and volatility of gas futures; volatility is higher on storage-report days and Mondays [S6][S7]. Analysts forecast the print with ~13 Bcf/week average error in 1997-2004, smaller during the injection season [S5]. | strong (physical) | The raw sign (injection vs withdrawal) is almost pure seasonality, so it is a trivial target. The non-trivial target is sign(dS_w - avg5_w), i.e. whether the surplus/deficit vs 5-yr average widens. |
| Weather-driven demand (HDD/CDD) | Temperature shocks significantly affect near and far-month gas futures returns; population-weighted multi-station temperature works better than one station [S7]. | strong for demand/storage, weak for price | Used as an input (exogenous driver), not a price target. |
| Realized-volatility direction | Volatility clustering and GARCH dominance in energy [S8]; heavy tails and conditional heteroscedasticity in gas prices [S9]; oil round 1: GARCH was best on vol direction (`docs/audit_commodity.md` §"Volatility direction"). | strong | Gas vol is also seasonal (winter). 20-day windows give only ~6 independent windows/yr (audit B4); use shorter windows. |
| Winter/summer spread, curve shape/roll | Storage and the forward curve are linked (theory of storage; price-storage SV models [S10]). | medium, but data not free | Needs a full futures curve; EIA stopped publishing NYMEX contracts 1-4 after 2024-04-05 (checked, §2). |
| TTF vs HH | European gas is modelled with temperature, storage, EUA, oil and coal, with skewed heavy-tailed state-space models; gains are in CRPS (13%/9% vs best benchmark), mainly from volatility modelling, not direction [S9]. | medium (distribution), weak (direction) | Since 2022, TTF and HH are linked mainly by LNG export capacity; spread direction has no documented edge. |

## 2. Free, reproducible data (URLs checked 2026-10-31)

| series | URL / API | frequency, coverage | publication timing and quirks |
|---|---|---|---|
| Henry Hub spot (EIA RNGWHHD = FRED DHHNGSP) | https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP ; https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm | business days, 1997-01-07 .. 2026-10-27 (already in `data/commodity/hh.csv`, 5,717 obs, sha256 in `data/commodity/manifest.json`) | **Still updated** (not discontinued; the discontinued FRED series is GASPRICE, ended 2014 [S11]). Vendor: Refinitiv/LSEG [S12]. EIA updates it **weekly** (page: "Release Date 10/28/2026, Next 11/4/2026"), so a real-time system would read it from a vendor; prices themselves are public on the trade day. The trade-date vs flow-date convention is **not documented** [S12]; weekend packages (Fri trade for Sat-Mon flow) mean Fri->Mon returns span 3 days. No holiday rows. |
| NYMEX NG futures 1-4 (EIA RNGC1..4) | https://www.eia.gov/dnav/ng/hist/rngc1d.htm | 1994 .. **2024-04-05, then stopped** (checked) | Unusable for 2025-2026. Contract expires 3 business days before the delivery month [S12]. |
| NG=F (Yahoo continuous front month) | https://query1.finance.yahoo.com/v8/finance/chart/NG=F | daily from 2000-08-30 (checked), America/New_York | Unadjusted front-month splice: **roll-day returns are artifacts** (the contract switch ~3 business days before month end, where contango/backwardation jumps can be several %). Unofficial API, no vintage, occasional missing/duplicate days; terms of use are personal. Mask or back-adjust roll days; prefer spot for the primary series. |
| EIA weekly storage, Lower 48 working gas (NW2_EPG0_SWO_R48_BCF) | https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm ; API v2 (free key) https://api.eia.gov/v2/natural-gas/stor/wkly/ ; report files https://ir.eia.gov/ngs/ngs.html (wngsr.csv/json, `ngshistory.xls`, `ngsstats.xls` = 5-yr avg/max/min, `cvhistory.xls` from 2015) | weekly (week ends Friday), current-methodology series from 2010-01-01 (checked); 5 regions (East, Midwest, Mountain, Pacific, South Central) | Released **Thursday 10:30 ET** for the week ending the previous Friday (6-day lag); moves to Wednesday (or another day) in holiday weeks [S13]. Revisions: published when cumulative changes >= a threshold (10 Bcf in 2015 [S14], lowered from 7 to 4 Bcf per a 2017 evaluation [S13]); reclassifications appear as an "implied flow" that differs from the net change. The current-vintage series contains these revisions; the first print is in the weekly report files. The 5-yr average (e.g. "2021-25") switches window every January. |
| CPC degree days, observed (gas-utility weighted) | https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{YYYY}/UtilityGas.Heating.txt (also Population.Heating/Cooling.txt, StatesCONUS, CensusDivisions) | daily, 1981 .. now, 9 Census divisions + CONUS rows | **`UtilityGas.Heating.txt` weights HDD by gas-heating households: the right demand driver.** Preliminary values; published the next day (no explicit timestamp). Normals 1981-2010 in `climatology/`. |
| CPC degree-day **forecasts (genuine ex-ante archive)** | https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_forecasts_7day/{YYYY}/{MM}/{DD}/UtilityGas.Heating.txt (and Population.Cooling.txt) | daily issues, 2014 .. now, 7 days ahead, from NDFD | Header "Issued 0000UTC YYYYMMDD"; HTTP Last-Modified of the 2015-01-02 file is 15:00 GMT the same day (i.e. 10:00 ET, before the 14:30 ET futures settlement). This is the key leakage-free weather input. |
| CPC 6-10 / 8-14 day outlooks | https://www.cpc.ncep.noaa.gov/products/archives/short_range/ (README.6-10day.txt) | daily, archive from ~2001 | Tercile probabilities vs 1991-2020 normal, gridded; coarse but ex ante [S15]. |
| Open-Meteo Historical Forecast / Previous Runs | https://historical-forecast-api.open-meteo.com/v1/forecast ; https://previous-runs-api.open-meteo.com/v1/forecast (`temperature_2m_previous_dayN`, N = 1..7) | Historical forecast ~2021/2022+; Previous Runs from Jan 2024 (GFS T2m from Mar 2021) [S16][S17] | The historical-forecast series stitches the first hours of each run (close to analysis, i.e. near-observed): **not an ex-ante multi-day forecast**. Previous Runs gives fixed lead times but only from 2024 (too short for DEV training). Use CPC forecasts instead; Open-Meteo archive (ERA5) only as observed weather. |
| TTF | Yahoo TTF=F (ICE Endex front month, EUR/MWh), daily from ~2017-10 (checked); FRED PNGASEUUSDM (IMF, monthly) | | Same roll issues as NG=F; EUR and MWh units; different holidays. Secondary only. |
| EIA consumption / supply | EIA Natural Gas Monthly (API v2 `natural-gas/cons/sum`), lag ~2 months | monthly | Too slow for daily/weekly targets; DEV-only calibration of the demand model. |
| Analyst consensus for storage | Bloomberg/Reuters/WSJ polls | weekly | **Not free; not reproducible.** Do not use. |

## 3. Leakage pitfalls and information-set rules

Pitfalls specific to gas:
1. **Storage timestamp.** The week-W print (week ending Friday F) is public only on Thursday F+6 at 10:30 ET. A model
   at an origin before that Thursday must not see dS_W. Daily price models must not use a storage value on its
   week-end date.
2. **Revisions and reclassifications.** The current-vintage EIA series embeds later revisions (>= 4-10 Bcf
   threshold) [S13][S14]. Targets should be scored against the first print where possible (report archive), or the
   current vintage with the limitation disclosed (as in the oil prereg).
3. **Observed vs forecast weather.** Using observed HDD of day t+k at an origin t is look-ahead. Only CPC forecasts
   issued on or before the origin may describe days after it. Observed HDD for day d may be used from day d+1.
4. **5-yr average window.** EIA's 5-yr window changes in January; compute avg5 ourselves from the history available
   at the origin (prior 5 years of the same ISO week) to keep it causal and reproducible.
5. **Continuous-contract rolls.** NG=F roll-day jumps are not returns; label and mask them, or use spot.
6. **Spot date convention.** Undocumented trade vs flow date [S12]; use the conservative rule below.
7. **Consensus not free.** Any "surprise vs consensus" target is out of scope; define surprise vs our own
   ex-ante forecast or vs avg5.
8. **Foundation-model contamination.** As in `docs/commodity_prereg.md` §2 (Chronos-2 Oct 2025 may cover part of VAL).
9. **Holiday weeks.** Release days shift (Wednesday or other days) [S13]; the info-set check must use the actual
   release date per week, not "Thursday".

Proposed information-set rules (to be enforced by a perturbation test, as in `tools/power/test_infoset.py`):

| data | at a daily origin t (forecast made at 14:30 ET on t-1 close, i.e. before day t) | at a weekly storage origin |
|---|---|---|
| HH spot | values dated <= t-1 | values dated <= origin day - 1 |
| NG=F settlement | <= t-1, roll days flagged | same |
| storage dS | weeks whose release datetime <= origin datetime | same (week W unknown until its release) |
| observed HDD/CDD | days <= t-2 (conservative; d+1 publication) | days <= origin - 1 |
| CPC forecast HDD/CDD | issues dated <= t-1 (issued 00 UTC, posted ~15 UTC) | issues dated <= origin day |
| calendar, normals, avg5 | anything (deterministic) | avg5 computed from years before the target year |

Weekly storage origins (recommended): **origin = Friday 15:00 ET after the week-W window closes** ("nowcast" of the
Thursday print, h = 0 weeks, 6 days before release: observed HDD for Sat..Thu of week W known, Fri from CPC
forecast) and **origin = Thursday 11:00 ET just after the week-(W-1) print** (h = 1, the week ending the next day
plus 7 days ahead, using CPC forecast issues of that Thursday).

## 4. DEV feasibility check (light, no training; DEV rows only)

Script: scratchpad `dev_storage_stats.py` (not committed; parses EIA storage html and CPC CONUS daily degree days,
drops every row after 2025-06-30 at parse time). Target: y_W = 1 if dS_W > avg5_W, where avg5_W is the mean net
change of the same week in the 5 prior years (own computation). Fit 2015-2019 (259 weeks), evaluate DEV
2020-01 .. 2025-06 (287 weeks). Weather = observed gas-utility-weighted HDD and population CDD of the report week
(Sat..Fri), as anomalies vs their own 5-yr same-week means, i.e. the **nowcast origin** of §3.

| predictor (eval 2020-01..2025-06, n = 287) | hit rate | p vs 0.5 | notes |
|---|---|---|---|
| base rate P(y = 1) | 0.481 | | train 0.541; roughly balanced |
| persistence sign(dev_{W-1}) | 0.606 | 2e-4 | already beats a coin |
| OLS dev ~ HDD anomaly + CDD anomaly | 0.760 | 2e-19 | corr(pred, dev) = 0.855, RMSE 23.6 Bcf (sd(dev) = 44 Bcf) |
| OLS + dev_{W-1} | 0.763 | 6e-20 | RMSE 22.6 Bcf; winter (Nov-Mar, n = 121) **0.893**, summer (n = 166) 0.669 |
| same nowcast -> sign of HH spot change Fri->Fri | 0.466 (n = 279) | 0.88 | **no price edge** |
| same nowcast -> HH spot change Thu->Fri (day after release) | 0.500 (n = 254) | 0.53 | **no price edge** |

Reading: a plain weather regression already recovers most of the storage signal (the market pays analysts for
exactly this, [S5]). The price does not follow our nowcast because the market already prices the weather; this is
direct DEV evidence that **price direction is the wrong headline target**. Summer is harder (power burn depends on
the gas-coal price switch, LNG feedgas and wind/solar), so that is where a richer state can add value.
These numbers use observed weather (legal at the nowcast origin only); the h = 1-week target with CPC forecasts
will be lower and must be measured on DEV by the gas lead.

## 5. World-model designs and their direct twins

All twins see exactly the same inputs (same windows, same CPC issues, same lags) and the same training data; only
the computation differs: the WM rolls a latent state forward one **day** at a time and aggregates, the twin maps the
inputs to the weekly target in one shot.

| id | world model (latent state, step) | direct twin | target(s) | why the WM could win |
|---|---|---|---|---|
| WM-S (primary) | **Storage-balance state-space WM.** Daily latent z_d = [storage S_d, supply level, export/LNG level, demand-sensitivity, power-burn regime]; transition S_{d+1} = S_d + supply_d - demand(HDD_d, CDD_d, z_d) - exports_d + e_d, with learned nonlinear demand(.) (heating demand is convex in HDD and saturates; holiday/weekend effects). Observation: weekly sum of daily flows = EIA dS_W (5 regions + L48, so regions must add up). Inputs per day: observed HDD/CDD up to the info-set cut, CPC forecast HDD/CDD after it. Rolled 7-14 days. | MLP/linear head on the concatenated 7 daily HDD/CDD (+ forecasts), lags of dev, week-of-year -> dev_W directly (one head per horizon) | dev_W sign and Bcf, h = 0 (nowcast) and h = 1, 2 weeks; 5 regions secondary | Composition across days and weeks (h = 2 reuses the rolled state), mass balance across regions, nonlinear demand, partial weeks and holiday shifts. Honest caveat: with linear demand and h = 0 the WM reduces to the twin; expect the WM gap at h = 1-2 and in summer, not in the winter nowcast. |
| WM-V | **Stochastic-vol WM with storage-deficit drive.** Latent log-vol h_t: AR(1) with seasonal mean + loading on the storage deviation (low inventory -> higher vol; theory of storage, price-storage SV [S10]) and on weather anomaly; Student-t returns; rolled daily, k-day RV = mean of simulated squared returns (Monte Carlo). | same features -> direct regression of log RV_{t+1..t+5} (and the up/down label) | vol direction, 5 d (and 20 d descriptive) | Seasonality and storage enter the state, GARCH lacks them. Oil lesson: wm_sv lost to GARCH (0.612 vs 0.744); GARCH must be in the baselines. |
| WM-R | **Regime switching** (Markov: injection / withdrawal / shoulder; or tight / loose storage) inside WM-S or WM-V | same inputs with a regime one-hot | as above | Only if it improves DEV; otherwise drop. |
| WM-P | Price via the latent: HH log-return = f(z_d) + SV noise, rolled with WM-S | direct return/direction head | HH direction 1, 5 d | Negative control. §4 shows ~50%; report it, do not headline it. |
| Demo | **Cold-snap scenario**: shift the CPC 7-day HDD forecast by +2 sd for days 1-7, roll WM-S/WM-V 1,000 paths -> fan of storage dev, end-of-season (Mar 31) storage, and 5-day vol. Also a "mild winter" and "LNG outage" (export state -2 Bcf/d) scenario. | none (a direct model cannot answer "what if") | qualitative | This is what the WM can do that the twin cannot; show it, but do not score it. |

## 6. Baselines and foundation models

| target | baselines (field standard) | FMs already in the harness |
|---|---|---|
| storage dev_W | avg5 (= base rate), persistence sign(dev_{W-1}), seasonal naive (last year's same-week dS), **OLS on weather anomalies (+dev lag)** (§4, the bar to beat), AR(1)/ARIMA on dev | Chronos-2 (covariate-capable: give HDD/CDD), Chronos-Bolt, TiRex on the weekly dS series (univariate); Kronos is for K-line prices, skip |
| HH price direction 1/5/20 d | RW (sign = majority/drift), momentum sign(r_{t-h..t}), AR(1), seasonal naive | Chronos-2, Chronos-Bolt, TiRex, Kronos-small (as in `docs/commodity_prereg.md`) |
| vol direction 5 d | GARCH(1,1) and GJR (best in oil round 1), HAR-RV on squared daily returns (no free intraday data), EWMA, seasonal vol mean | same FMs via quantile spread (oil harness rule) |

## 7. Recommendation and statistical power

TEST = 2025-11-01 .. ~2026-10-30, about **52 weekly** and **~245 daily** origins. Minimum detectable hit rates
(one-sided exact binomial p < 0.05 / Wilson 95% lower bound > p0), non-overlapping samples:

| n (independent) | vs 0.5 | Wilson LB > 0.5 | vs p0 = 0.6 | vs p0 = 0.7 |
|---|---|---|---|---|
| 250 (daily h = 1) | 0.556 | 0.564 | | |
| 120 | 0.583 | 0.592 | | |
| 52 (weekly; daily h = 5 non-overlap) | 0.635 | 0.654 | 0.731 | 0.827 |
| 24 | 0.708 | 0.708 | | |
| 12 (h = 20 non-overlap) | 0.833 | 0.833 | | |

1. **Primary: weekly storage deviation direction, sign(dS_W - avg5_W), at the h = 1-week origin (Thursday after
   the previous print, CPC forecast weather only).** DEV evidence: the nowcast analogue reaches 0.76 (0.89 in winter)
   vs a 0.65 detection bar at n = 52; TEST covers winter 2025-26. Also report the h = 0 nowcast (Friday origin) as
   co-primary. Beating persistence (0.61) is the real bar: model-vs-model is an exact McNemar on discordant weeks
   (e.g. 11 of 13 discordant in favour gives p ~ 0.01), so pre-declare it.
2. **WM vs direct twin: use a continuous loss** (MAE in Bcf of dev_W, DM-HLN on 52 weeks, plus CRPS from 200 WM
   paths) as the primary comparison; hit rate alone has too little power to separate two good models on 52 weeks.
   Secondary pooling: 5 EIA regions x 52 weeks (260 labels, week-clustered bootstrap CI).
3. **Secondary: 5-day volatility direction of HH spot** (non-overlapping n ~ 49; need >= ~0.65 vs 0.5, or vs the
   TRAIN base rate). GARCH is the bar.
4. **Negative control (reported, not headlined): HH price direction at 1, 5, 20 days** and post-release price
   direction. Expect ~0.50 (oil round 1; §4 DEV check). Saying so is part of the honest pitch.
5. Investor framing: "we forecast the Thursday EIA storage print against the 5-year norm, a week ahead, from
   ex-ante NOAA forecasts, with a world model that can also run cold-snap scenarios", not "we predict gas prices".

Data to pull for the pre-registration (all free): EIA storage L48 + 5 regions (html/API v2, plus `ngshistory.xls`
for first prints and pre-2010 history: needs `xlrd`, not installed here); CPC daily `UtilityGas.Heating` and
`Population.Cooling` (1981+) and the 7-day forecast archive (2014+); FRED DHHNGSP (already local). Training data
for forecast-based models therefore starts in 2014 (CPC forecast archive), i.e. ~10 winters on DEV.

## Sources

- [S1] Movement Prediction-Adjusted Naive Forecast: Is the Naive Baseline Unbeatable in Financial Time Series Forecasting? (2024) https://www.alphaxiv.org/abs/2406.14469
- [S2] Prokopczuk, Wese Simen, Wichmann, "The Natural Gas Announcement Day Puzzle", The Energy Journal 42(2), 2021. https://ideas.repec.org/a/aen/journl/ej42-2-wichmann.html ; https://livrepository.liverpool.ac.uk/3083246/ ; exec. summary https://iaee.org/ej/ejexec/ej42-2-Wichmann-exsum.pdf
- [S3] Linn, Zhu, "Natural gas prices and the gas storage report: Public news and volatility in energy futures markets", J. Futures Markets 24 (2004).
- [S4] Linn, IMF commodity seminar slides (2012). https://0-www-imf-org.library.svsu.edu/external/np/seminars/eng/2012/commodity/pdf/linn2.pdf
- [S5] Gay, Simkins, Turac, "Analyst forecasts and price discovery in futures markets: The case of natural gas storage", J. Futures Markets 29(5):451-477 (2009). https://ideas.repec.org/a/wly/jfutmk/v29y2009i5p451-477.html ; NGI coverage of the 13.2 Bcf error: https://www.naturalgasintel.com/study-finds-average-13-2-bcf-error-in-analysts-weekly-storage-predictions-3/
- [S6] Mu, "Weather, storage, and natural gas price dynamics: Fundamentals and volatility", Energy Economics 29(1):46-63 (2007). https://ideas.repec.org/a/eee/eneeco/v29y2007i1p46-63.html ; https://discovery.dundee.ac.uk/en/publications/weather-storage-and-natural-gas-price-dynamics-fundamentals-and-v/
- [S7] Hu, Hu, Lin, effect of temperature shocks and inventory surprises on natural gas and heating oil futures returns, The Scientific World Journal (2014). https://search.emarefa.net/en/detail/BIM-1049684-effect-of-temperature-shock-and-inventory-surprises-on-natur
- [S8] Modelling and Forecasting Energy Market Volatility Using GARCH and Machine Learning Approach (2024). https://www.alphaxiv.org/abs/2405.19849
- [S9] Berrisch, Ziel, "Distributional Modeling and Forecasting of Natural Gas Prices", J. Forecasting 41(6) (2022). https://arxiv.org/abs/2010.06227
- [S10] Stochastic Path-Dependent Volatility Models for Price-Storage Dynamics in Natural Gas Markets (2024). https://www.alphaxiv.org/abs/2406.16400
- [S11] FRED GASPRICE (discontinued) https://fred.stlouisfed.org/series/GASPRICE ; DHHNGSP https://fred.stlouisfed.org/series/DHHNGSP
- [S12] EIA price definitions (Refinitiv source; contract expiry rule) https://www.eia.gov/dnav/ng/TblDefs/ng_pri_fut_tbldef2.asp
- [S13] EIA WNGSR schedule/holiday and revision notices: https://ir.eia.gov/ngs/ngs.html ; https://ir.eia.gov/ngs/revisions.html ; https://ir.eia.gov:443/ngs_test/wngsrevaluation.pdf
- [S14] EIA notice 2015-08-31 (10 Bcf revision threshold) https://ir.eia.gov/ngs/notice_08_31_2015.html
- [S15] CPC 6-10 day outlook data README https://www.cpc.ncep.noaa.gov/products/archives/short_range/README.6-10day.txt
- [S16] Open-Meteo Historical Forecast API https://open-meteo.com/en/docs/historical-forecast-api
- [S17] Open-Meteo Previous Runs API https://open-meteo.com/en/docs/previous-runs-api
