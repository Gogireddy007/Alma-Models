# Alma-Live Gas: leakage and vintage record

Data lead, 2026-11-01.
- Code: `tools/live/` (`sources.py` sets release rules; `store.py` implements `asof`).
- Tests: `tools/live/test_infoset.py`, output in `runs/live/test_infoset.txt`. Result: 70/70 checks passed at commit
  8242f93.
- Red-team log: `docs/live_redteam.md`. Its independent references are `tools/live/redteam/rt_ref.py` and
  `rt_vintage.py`, and its independent perturbation run matched (34/34).

## 0. Clock

- **Decision:** 15:30 ET on trading day d (`common.decision_time`).
- **Fill:** 16:00 ET close of d, plus 10 bp per side (`common.fill_time`). The fill is always after the decision.
- **What the model sees:** `ctx = Store().asof(decision_time)` returns only records with `published_at <= t`.
  - For each (series, obs), the latest vintage visible at t is returned.
  - In live/paper mode, `fetched_at <= t` is also required (LRT-8).
- **Labels:** come only from `Store.labels()` / `storage_labels()` and are never inside ctx.
- **Exposure:** 2025-11-01 onward is **EXPOSED**. A previous project's TEST showed Henry Hub prices for that period.
  The harness refuses that period unless the run is explicitly labelled. The only clean claim test is the forward
  paper test (`docs/live_prereg.md`).

## 1. Vintage and timing risks by source

| # | source / field | risk | rule in the store | residual risk | test |
|---|---|---|---|---|---|
| 1 | EIA storage level (L48 + 5 regions) | Current vintage embeds later revisions. 69 of 593 weeks since 2015-06-19 were revised (level \|diff\| mean 7.4 Bcf, max 14). This includes the 33-week resubmission of 2025-08-29..2026-04-10 (about -10.5 Bcf) released 2026-04-23 14:30Z (LRT-1) | First print from revisions.xls `original_data`, published at its release. Revised value published at the NEXT report's release. Resubmission values published at the 2026-04-23 release | (a) Pre-2015-06-19 weeks exist only in the current vintage (`current_backfill`). Typical revision size estimated from 2015+: 11% of weekly changes differ, mean \|diff\| 0.34 Bcf, p95 2 Bcf, max 10 Bcf. Excluding the resubmission, revised levels differ by a median of 4 Bcf. (b) Intermediate vintages (e.g. week 2026-03-27: 1865 -> 1861 -> 1851) are not in EIA's files; we show the first print until the final revision | LRT-1 checks: 1970 before / 1960 after 2026-04-23 14:30Z; matches rt_vintage exactly |
| 2 | Storage weekly change | The printed net change is not equal to FP(W) - FP(W-1) in 37 weeks (when W-1 was revised in the same report) | Separate series `<s>.chg_printed` = FP(W) - level(W-1) as known at A(W) | Same as row 1 for pre-2015 weeks | Week 2026-04-17 printed change is 103, not 93 |
| 3 | Storage release time | Holidays, DST (LRT-4), mourning days | Thursday 10:30 ET, converted DST-aware. EIA schedule alternates (2025-26) at 12:00 ET. Pre-2025 weeks with a Tuesday-Friday federal holiday (including 2018-12-05 and 2025-01-09) at F+10 12:00 ET | Conservative: a holiday week may become usable later than in reality | Every first print is at or after the rt_ref reference (878 weeks); DST checks for 2024-03-14 and 2024-11-07 |
| 4 | UNG / BOIL / NG=F daily bars (Yahoo) | (a) A day-d bar is not final until after the close. (b) Yahoo adjusts history for later splits. (c) Yahoo stamps futures bars at 00:00 exchange time; dating them with today's gmtoffset shifted EDT-period bars back one day (found and fixed before launch). (d) Yahoo can silently revise bars | (a) The bar of session d is published at d 18:00 ET, so ctx holds bars through d-1. (b) Stored UNADJUSTED; ctx re-applies only splits with ex-date <= t. (c) Session date comes from each bar's exchange time zone. (d) Every fetch is a raw snapshot; a changed value is appended as a new record with published_at = its fetch | Historical bars are a 2026-11-01 snapshot. Bar values before launch are "as now", not as first displayed (low risk for exchange closes) | px published_at check; UNG (4) and BOIL (7) split continuity checks |
| 5 | Henry Hub spot (EIA/FRED) | Published weekly with a lag (LRT-2) | **Not used at all.** Price inputs and P&L come from UNG and NG=F | none | none |
| 6 | NG front month / roll | Yahoo's NG=F switches contract on the session after the last trade, creating roll gaps (median \|ret\| 0.075 on switch sessions vs 0.028 elsewhere) | Labels `ng_cc` / `ng_oo` are NaN across switch intervals. Features zero the switch-day return. The live roll rule is calendar only: hold the nearest contract with last trade > d + 3 business days. NYMEX last trade = 3 business days before the delivery month | NG is for research and features only; the P&L instrument is UNG, whose price includes its own roll | Roll-rule and mask tests; identical under perturbation of future prices |
| 7 | Weather forecasts (CPC 7-day, NDFD) | (a) Posting time. (b) An "archived forecast" might actually be an analysis. (c) Realized weather substituted for a forecast (LRT-3) | (a) Issue 00Z. published_at = HTTP Last-Modified when within 3 days of issue (median issue + 15.0 h, min 13.5 h). Otherwise (2,548 rows, re-uploaded files) issue + 1 day 17:00Z. (b) Forecast error grows monotonically with lead (MAE 0.72 HDD at lead 0 to 1.87 at lead 6), the signature of genuine forecasts. (c) Forecast rows whose valid day had already ended at publication are dropped. Observed values live in a separate source and can never enter the forecast frame | 132 issue files missing on CPC's server; these show up as an older issue at decision time. Last-Modified could reflect a re-post of identical content (conservative) | Weather checks, including perturbing all obs leaving forecasts unchanged |
| 8 | Open-Meteo | The archive API is ERA5 realized weather. The historical-forecast endpoint stitches early forecast hours (close to realized) | **Not used.** The Previous-Runs API (genuine lead-k GFS, 2021-04+) is implemented in `sources.py` but **paused** per manager decision 4 (and rate-limited, HTTP 429) | none | none |
| 9 | CPC observed degree days | Yearly files hold final values (small revisions) | Used only for normals and scoring. Day D published at D+1 18:00Z | Small revision risk in normals | Obs published_at check |
| 10 | CFTC COT (**paused**) | Tuesday positions released Friday 15:30 ET. Holidays shift the release. Shutdown backlogs (LRT-10); corrections overwrite rows in CFTC's API | Friday 15:30 ET. A federal holiday Tuesday-Friday moves it to the next business day. Every Tuesday from a shutdown start until the catch-up is published at the catch-up date (2013-11-22, 2019-03-08, 2026-01-30) | LRT-11: 2018-12-04 and 2025-01-07 (mourning days) are probably a day late in reality; open, since COT is paused. Corrected rows are the current vintage | COT >= Friday check |
| 11 | SEC EDGAR (**paused**) | Period-end dating; time zone | acceptanceDateTime interpreted as US/Eastern (EDGAR's "Z" suffix is ET). XBRL facts use the accepting filing's time | none known | red-team audit |
| 12 | Baker Hughes rigs | none (not available) | rigcount.bakerhughes.com returns HTTP 403 bot protection, which is not bypassed. Marked unavailable in the manifest | none | none |
| 13 | EIA API v2 | none (not used) | Used only if `EIA_API_KEY` is set (it is not). A key is never requested | none | none |

## 2. Store and harness guarantees

- **Perturbation.** Randomizing every record with `published_at > t` leaves the ctx hash, features, forecaster
  outputs and rule actions bit-identical. This includes shuffled obs, 10% deleted rows and inserted fake rows.
  - Covered 17 decision times: DEV, VAL, release days, DST and holiday weeks, roll days, and one exposed day for the
    mechanism only.
  - The same holds for `fetched_at > t` in live mode.
  - Positive control: changing a record published at or before t changes ctx.
- **Append-only.** The PIT files are append-only and prefix-pinned in `data/live/manifest.json`.
  - Pins are anchored outside the file by git commits and by the manifest sha in each ledger decision line (LRT-8).
  - Pre-launch rebuilds before any decision existed: COT (shutdown rule), EDGAR (dedupe key), storage (2018-12-05
    holiday) and px (bar dating). After launch, a rule change can only append newer vintages. It cannot rewrite what
    was visible.
- **Fitting.** Fitted baselines fit only on days ≤ 2024-06-30 whose label was available before the refit time. DEV
  uses annual walk-forward. The harness asserts no fitting row goes beyond DEV.
- **Data access.** Models cannot open data files. An audit hook guards `data/live`, `data/gas`, `data/fin` and
  `runs/live` while predict, act or observe run.

## 3. Model contamination

- Chronos-2 (Oct 2025) and TiRex (May 2025) may have seen VAL-period prices. Their VAL scores are not claims.
- The forward paper test from launch (2026-11) is after every model release used here, except anything released
  later.
