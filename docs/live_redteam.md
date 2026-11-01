# Alma-Live Gas red-team log (running)

Red-team (agent), started 2026-11-01 (DST ended 02:00 EDT today: all times below are UTC unless marked). Scope:
data lead (`tools/live/**`: point-in-time store `asof(t)`, paper broker, harness, `docs/live_leakage.md`,
`docs/live_prereg.md`), world-model architect (`tools/live/wm_*.py`, `tools/live/search/**`, `runs/live/search/**`,
`runs/live/FROZEN.json`). The red-team writes only `tools/live/redteam/**` and this file, runs light `nice -n 10`
jobs, and reads no exposed-window (>= 2025-11-01) outcome to make choices. Prior bug classes carried over from
`docs/fin_redteam.md`: held-out data in shared factors (RT-4), reward purge (RT-8/9), abstention scoring (RT-13),
weak/calendar-free baselines (RT-15), selection on holdouts (RT-3/11/16), vacuous checkpoint pins (RT-6).

Independent references (written from raw files and exchange/agency rules, without reading team code):
- `tools/live/redteam/rt_ref.py`: WNGSR release times (Thu 10:30 ET, holiday overrides paired by holiday NAME from
  `data/gas/raw/ngs_schedule.html`, DST-correct UTC), COT release (Fri 15:30 ET, holiday -> next business day),
  NYMEX NG last trade (3 business days before delivery month) and calendar-only front contract, NG settle 14:30 ET,
  ETF close 16:00 ET, ERA5 latency (>= day + 5 d), EIA/FRED HH spot weekly publication.
- `tools/live/redteam/rt_vintage.py`: WNGSR first print / revision vintages and the printed net change.

## Findings

| id | severity | finding | evidence | owner | status |
|---|---|---|---|---|---|
| LRT-1 | HIGH if used (pre-emptive) | EIA storage 33-week resubmission: Lower-48 levels for weeks 2025-08-29 .. 2026-04-10 were revised down ~10.5 Bcf (all 33 weeks) in the report released 2026-04-23 14:30Z. The current vintage (`ngshistory.xls`, `data/gas/storage.csv` l48_current) embeds this. Any as-of value for these weeks at t < 2026-04-23 must be the first print (`revisions.xls` original_data); week 2026-03-27 also has an intermediate vintage (1865 -> 1861 at the 2026-04-02 report -> 1851) not recoverable from the files. Also: the printed net change differs from FP(W)-FP(W-1) in 37 weeks since 2015 (e.g. week 2026-04-17: 103 vs 93) | `rt_vintage.py` | data lead | **fixed + verified** (rt_pubaudit: store l48 first prints = rt_vintage 593/593, 69 revised rows = current with published_at >= rt_vintage revised_at (resubmission at 2026-04-23 14:30Z), l48.chg_printed = rt_vintage printed change 591/591; wm_store uses first prints only) |
| LRT-2 | MEDIUM (pre-emptive) | Henry Hub spot from EIA/FRED (RNGWHHD / DHHNGSP) is published weekly, days after the trade date; the Alma-Gas convention "spot of date t usable from the close of t" is not achievable live. published_at must follow the source's real publication, or the price input must come from a real-time source (NG futures settle 14:30 ET, ETF close 16:00 ET) | rt_ref.hh_spot_eia_avail_utc; gas data.py convention | data lead, architect | **fixed in architect adapter** (wm_data p_avail = Wednesday F+5 after the trade week; rt_ref uses the same Wednesday rule; perturbation passes); store: open |
| LRT-3 | MEDIUM (pre-emptive) | Open-Meteo archive weather is ERA5/ERA5T REALIZED reanalysis with ~5-day latency; it is never a forecast and is not available for day t-1 at decision t live. As-issued forecasts need an archived issue time (CPC 6-10/8-14 day issues, or GFS/ECMWF archived runs) | `data/gas/manifest.json` (archive-api URLs) | data lead | open (watch) |
| LRT-4 | INFO | WNGSR holiday schedule page covers only 2025 and Nov 2026 holidays; Jan-Oct 2026 holiday weeks with a Tue-Thu federal holiday (none found) would be unverified. Monday holidays do not shift the Thursday release. 2026-03-08 DST start and 2026-11-01 DST end must move the 10:30 ET release between 14:30Z and 15:30Z | rt_ref self-check | data lead | open (verify store's UTC conversion) |
| LRT-5 | **HIGH** (claim validity) | Architect's interim tradable proxy is Henry Hub SPOT (fill P[t+1], P&L P[t+1]->P[t+2], 5 bp). Spot is a cash-market assessment with untradable structure: DEV 2015-01..2024-06 daily log returns ac1 -0.091, ac2 -0.082; weekend-package weekday means Mon +1.0%, Fri -1.0%. Trivial rules under the same convention: -sign(r_t) net Sharpe 0.50, -sign(5d mom) 0.64, buy-hold -0.02. Any S_A / agent selection on spot rewards untradable artifacts; 5 bp is not a cost model for NG (tick 0.001 = $10/contract, roll spread) or UNG/BOIL (spread, BOIL 2x vol decay, expense) | red-team inline script (pass 1), docs/live_search.md 0.2 | architect (data lead for the instrument) | **accepted** (architect ~12:25Z): no spot P&L anywhere; scored panel from Store.asof via wm_store.py, target UNG close(d)->close(d+1), 10 bp/side at 1x/2x/3x, baselines add ac1/ac2 mean-reversion and weekday rules on UNG -> verify when wm_store lands |
| LRT-6 | MEDIUM (bug; not a leak) | CPC forecast inputs always zero: `_fc` keys are `datetime.date` (from `.tolist()`), lookups are `numpy.datetime64` -> `dd in cur` never true. panel.npz (2790 rows): U fmask 0.0 everywhere, fhdd/fcdd 0, rev_hdd/rev_cdd 0. Weather-forecast pathway and weather-revision features dead; WM weather/rollout pillars void until fixed | perturbing the used issue's values leaves u unchanged (2020-03-20, 2020-04-03, 2019-06-14) | architect | **fixed + verified** (wm_data _fc datetime64 keys; panel rebuilt, fmask ~0.92; rt_perturb_wm: the used issue now changes u, and every non-warm-up change is explained by LRT-9 issues only) |
| LRT-7 | MEDIUM (timing) | Clock mismatch: data lead `common.decision_time` = 14:45 ET; architect adapter = 16:00 ET close. At 14:45 the same-day CPC issues with Last-Modified after 14:45 (7 in archive, 2 in 14:45-16:00) and anything published 14:45-16:00 (ETF close) are not knowable | common.py, wm_data.close_utc | data lead + architect | **fixed** (manager: decision common.decision_time 15:30 ET, fill common.fill_time 16:00 ET close same session; Yahoo bars published d 18:00 ET so ctx holds prices through d-1) -> verify in harness |
| LRT-8 | MEDIUM (ledger/PIT, pending asof) | `append_records` dedupe key includes published_at: a later run can append an earlier-published row for already-decided times (retroactive change of asof(t)). Live asof must also filter fetched_at <= t; manifest prefix pins are self-certifying (file and manifest editable together) -> anchor in git / ledger chain | common.append_records | data lead | in progress: data lead: Store().asof(t, live=True) also requires fetched_at <= t; ledger lines record ctx sha256 + manifest sha + git HEAD, hash-chained and committed -> verify |
| LRT-9 | LOW | 115 CPC issues have HTTP Last-Modified days after the issue date (2014: 70, 2019: 4, 2020-03-19..04-10: 41, batch LM 2020-04-12 19:51 ET = likely re-upload). Adapter trusts issue date < t; true first availability unknown for these. Moot while LRT-6 zeroes forecasts; after the fix, disclose or drop | cpc_fcst.csv last_modified | architect, data lead | **accepted LOW** after fix of LRT-6 it bites: rt_perturb_wm 73/100 (all 27 changes = days using these issues). LM batches 2014-03-13 18Z (70 issues 2014-01-01..03-11), 2020-04-12 23Z (41 issues 2020-01-05..04-10), 2019 (4): bulk re-uploads, not late issues; store rule (LM if <= 3 d after issue else issue+1d 17:00Z) is reasonable; disclose in live_leakage |
| LRT-10 | **HIGH if used** (timing) | COT post-shutdown backlog: `sources.cot_release` delays only Tuesdays INSIDE the shutdown windows, but CFTC released the backlog chronologically for weeks after each shutdown: Tuesdays ~2025-11-18 .. mid-Jan 2026 (catch-up ~2026-01-30, exposed window), 2019-01-29 .. ~03-05 (catch-up 03-08, DEV), 2013-10-22 .. ~11-19 get normal Friday published_at, up to ~8 weeks early. Fix: published_at = max(rule, catch-up 15:30 ET) for every Tuesday from shutdown start to catch-up | sources.py COT_SHUTDOWNS / cot_release | data lead | **fixed + verified** (sources.cot_release: every Tuesday from shutdown start to catch-up -> catch-up 15:30 ET; rt_pubaudit: only 2 reports early by 3 d, see LRT-11) |
| LRT-11 | LOW | COT reports of 2018-12-04 and 2025-01-07: CFTC closed on the national days of mourning (2018-12-05, 2025-01-09); release likely shifted to the next Monday, store uses Friday (3 days early); store storage release for week 2018-11-30 = Thu 2018-12-06 (actual Thu release; red-team reference over-conservative: OK) | rt_pubaudit.py | data lead | open (COT paused) |
| LRT-12 | LOW | `store.ctx_hash` is not canonical: it hashes `astype(str)` of frames, so the same record values parsed as int vs float (e.g. storage '769' vs '769.0') give different input_sha256; a ledger replay from a re-parsed PIT file can fail to match | rt_perturb_store (first run: hash differed with identical features until dtypes were aligned) | data lead | **fixed + verified** (8242f93: canonical float64 / int64-ns / sorted columns; rt_perturb_store hashes stable) |
| LRT-13 | MEDIUM (live parity) | The model path `wm_store.features_at` reads data/live/pit directly with published_at <= t; it does not go through `Store.asof(t, live=True)` and has no fetched_at filter. In paper trading the ledger's ctx hash (Store.asof) is then not the model's input, and a backfilled row (LRT-8) could enter live features. Need: live inference builds features from the same records as asof(live=True) (fetched_at <= t) + a parity test | wm_store.load_raw / _avail | architect + data lead | **fixed + verified** (46d0206: wm_store.features_from_ctx(ctx) is the only feature function, panel built from Store.asof per day, paper predict uses asof(t, live=True); rt_perturb_store now also compares features_from_ctx x/u and next_storage: 31/31 identical, live 2/2) |
| LRT-14 | LOW | wm_store NG=F features (ng_r1, ng_r5, ng_lrv20, ng_lp) use raw Yahoo continuous closes: roll jumps at contract switches enter returns/vol (store.features zeroes them via yahoo_front_switch_days). Not a leak | wm_store.features_at lines "NG=F (feature only)" | architect | moot (NG=F removed from WM features) |
| LRT-15 | LOW (holdout boundary) | DEV fold 2023 test rows: 5 decision days (late Jun 2024) have labels ending up to 2024-07-08 (r5/lrv5 window, next storage release), i.e. VAL-period outcomes in the DEV loop metric. Drop te rows with LAST > 2024-06-30 | wm_core.fold_rows (te not filtered on LAST) | architect | **fixed + verified** (a50f1f6: wm_harness.fold_rows drops DEV test rows with LAST > 2024-06-30) |
| LRT-16 | MEDIUM (twin matching) | Twins are matched on wall-clock (`budget_s` 45 s per fold), not steps: the direct twin (no rollout) gets many more optimizer steps than the WM, and both step counts depend on machine load at run time (non-reproducible; WM and twin run at different times). `n_steps` is computed but not logged. The no-graph twin is `graph=0` (edges removed), not the pre-registered parameter-matched MLP replacement. Fix: log n_steps per fold; for the finalist pillar run WM and twin at equal fixed steps (and at equal wall clock) with 3 seeds; param-match the no-graph twin or label it an edge ablation | wm_core.train_model, matched_cfg, loop.twins | architect | **fixed (code)** (a50f1f6: per-fold optimizer steps logged in results.tsv column steps; run.py twin <rid> trains the direct twin at the WM row's per-fold steps, params +-5%); verify on the first twin rows. No-graph twin: open (label as edge ablation) |
| LRT-17 | MEDIUM (selection) | The loop selects the WM incumbent on the same 4 DEV folds on which the WM-vs-twin and WM-vs-ref gaps are then reported; the twin and refs are not selected alike (winner's curse in the WM's favour; keep threshold 0.002 on a single seed with a load-dependent budget is likely below seed noise). DEV pillar numbers are post-selection, descriptive only; the pillar claim must come from VAL (one shot) with the finalist's twin re-trained identically. Also: protected objects (fold_rows, Norm, windows, pinball/pinball_np used by run.score, predict) live in the loop's editable file wm_core.py -> guarded by `rt_loopguard.py` (baseline = first revision; 0 changes so far) | loop.py, program.md, run.score | architect | in progress (a50f1f6: fold_rows/Norm/windows/predict/pinball_np moved to locked wm_harness.py; run.py asserts LOCKED.sha over wm_harness + wm_store + run.py + panel at every run; rt_loopguard watches those files and flags any protected name re-defined in wm_core.py). Post-selection labelling of DEV pillar gaps: open until FROZEN/report |
| LRT-18 | MEDIUM (baseline validity) | Prior frozen wm_gas (baseline for dsurp) has train_end 2025-06-30 (runs/gas/FROZEN.json): it is in-sample on all 4 DEV folds and on 2024-07..2025-06 of VAL. Its DEV/VAL scores are not out-of-sample (biased in the baseline's favour, so not a pro-WM leak, but the comparison is invalid); refit it per fold or score only 2025-07..10 | runs/gas/FROZEN.json train_end | architect | **resolved** (architect: wm_gas is a forward-test comparator only, no DEV/VAL claim) |
| LRT-19 | **HIGH** (ledger integrity) | `paper.Ledger.verify()` is self-certifying: (a) truncating the tail (dropping a losing day's decision/fill/mark) verifies OK; (b) rewriting history from line k with recomputed hashes verifies OK; (c) `Book(name, mode="sim")` on an existing LIVE ledger accepts a backfilled past-day decision (mode taken from the caller, not genesis). Minor: sim books always fail verify (genesis stamped with wall clock, later lines simulated times) | `rt_ledger.py` A3/A4/A5 (temp books only) | data lead | **fixed + verified** (8242f93: genesis mode immutable, so sim-on-live is refused; Ledger.verify_anchor requires the git-committed ledger to be a byte prefix; CLI decide/settle auto-commit the ledger; sim genesis stamped 2000-01-01. rt_ledger: A1-A5 all detected, truncation and forged re-hash caught by the anchor). Residual LOW: the anchor is local git history (a repo reset/rewrite defeats it) and an anchor commit can fail on an index-lock race (printed, not fatal); red-team records ledger heads in this log at each pass |
| LRT-20 | **HIGH** (paper P&L) | `settle()` values positions with RAW (unadjusted) closes and never applies splits to held quantities: holding w = 0.5 BOIL across a 1:2 reverse split with a flat market books +50.00% equity; a forward split books a fake loss (and halt_day). Real splits: BOIL 2026-05-28, 2024-11-07; UNG 2024-01-24 | `rt_ledger.py` A6 | data lead | **fixed + verified** (8242f93: splits_from_store; held shares x factor and previous close / factor for ex-dates in (prev mark, day]; rt_ledger A6: flat market across BOIL 1:2 split gives ret -0.00%, no halt) |

## Planned tests (run as soon as the team code exists)

1. Store perturbation (`rt_perturb.py`): for sampled decision times t (DEV, release days, DST weeks, holiday weeks,
   COT Fridays, roll days), replace every record with published_at > t by random values (and delete/insert rows);
   `asof(t)`, feature vectors, model outputs and policy actions at t must be bit-identical; positive control: perturb
   a record published just before t -> must change.
2. published_at audit: every store record's published_at >= the rt_ref reference for its source (storage, COT,
   weather, settlements, ETF closes, EDGAR acceptance time not period end).
3. Roll rule: front contract chosen by calendar only; perturb later volume/OI -> identical.
4. Fills: decision at t fills at a price strictly after t (no same-bar fill); settlement used only after 14:30 ET.
5. Training: normalisation stats, fold purge/embargo, no record >= 2025-11-01 (or VAL) in fitting/selection;
   continual update at t sees only outcomes published <= t.
6. Claims: feature-matched ridge/LightGBM, storage-surprise and weather-revision rule baselines, calendar terms; K
   counts every look; twins matched; costs (NG tick 0.001 = $10, roll spread, UNG/BOIL spread, BOIL 2x decay, fees).
7. Ledger: hash chain re-verified independently; backfill / edit / reorder attempts must be detected.
8. Independent re-derivation of every reported score from raw data.

## Passes

- Pass 0 (2026-11-01, ~12:10Z): team code not yet present (`tools/live/` holds only `redteam/`, no `runs/live/`).
  Built rt_ref / rt_vintage from raw files; LRT-1..4 logged pre-emptively (12:13Z).
- Pass 1 (12:15-12:20Z): architect interim adapter `tools/live/wm_data.py` + data lead `common.py`, `fetch_cpc.py`.
  `rt_perturb_wm.py` (red-team availability: DST-correct EIA release per week, CPC Last-Modified, Wednesday spot
  publication; future rows randomised; 60 random + 20 release days + 20 late-LM CPC days): 96/96 non-warm-up days
  identical, positive controls (latest published price, last released report) change. weekly.csv avail vs rt_ref:
  1 week earlier than reference (2018-11-30 / mourning day 2018-12-05; real release was Thu Dec 6, reference
  over-conservative): OK. Storage levels = first print for every week since 2015-07 incl. the 33 resubmission
  weeks (LRT-1 not triggered in the adapter). Found LRT-5..9. common.ng_active_contract: calendar-only roll
  (3 bdays before last trade), no prices: OK.
- Pass 2 (~12:25Z): data lead `sources.py` read: storage first print + revised vintage at next report + 33-week
  resubmission at the 2026-04-17 report + printed net change (LRT-1 addressed in code, verify against rt_vintage when
  the PIT files exist); Yahoo px stored unadjusted, split re-applied as of t, bar published d 18:00 ET; CPC LM rule;
  cpc_obs D+1 18Z; Open-Meteo previous-runs GFS lead k published (D-(k-1)) 06Z (conservative vs GFS 18Z of D-k);
  EDGAR facts dated by accepting filing acceptance time (ET-localized 'Z', conservative either way), not period
  end: OK. COT backlog -> LRT-10. rt_perturb_wm re-run after LRT-6 fix: only LRT-9 issues change inputs.
- Pass 3 (~12:30Z): `store.py` (Store.asof), `wm_store.py`, PIT files (storage, px, cot, edgar).
  `rt_pubaudit.py`: storage first print vs eia_release_utc 5268 rows, 6 earlier (one week, 2018-11-30, reference
  over-conservative); storage vintages/printed change exact vs rt_vintage; px bars all published >= 16:00 ET close
  (NG=F >= 17:00 ET), fa >= pa everywhere; EDGAR facts never dated by period end (0 within 2 days of period end);
  COT after LRT-10 fix: 2 dates 3 days early (LRT-11). UNG/BOIL raw (unadjusted) closes jump x4/x2 on the split
  ex-dates as expected; asof re-applies splits with ex-date <= t.
  `rt_perturb_store.py` (Store.asof): 34/34 days identical with every published_at > t record randomised,
  future published_at shuffled, future 'revised' vintages + new obs + a fake split inserted (random DEV/VAL days,
  Thursdays, 2025-11-14/26, 2025-12-29/31, 2026-04-23 resubmission, DST 2024-03-11/11-04 and 2026-03-09, COT
  catch-up 2026-01-29/30, 2019-02-08); live=True with fetched_at > t randomised: 2/2 identical; positive control
  (revision published t-1 s) changes ctx + features. CPC/om PIT files not built yet -> re-run when they land.
  `rt_perturb_wmstore.py` (wm_store.features_at, own perturbation incl. inserted re-published old obs, a same-day
  CPC issue published after t, observed d-1 published after t): 53/53 identical incl. split days 2018-01-05,
  2024-01-24/25; positive control changes.
  UNG (target) has no spot-type artifacts: DEV ac1 -0.026, ac2 -0.001; rule baselines on the panel's r1 label at
  10 bp/side (DEV 2015-10..2024-06): buy-hold -0.52, -sign(r1) -0.30, -sign(r5) -0.21, tsmom20 0.24,
  -sign(storage surprise) 0.19, -sign(storage deviation) 0.35, sign(CPC HDD revision) -0.40, weekday-only long
  -0.51..-1.09 (UNG contango decay). Strongest simple rule: -sign(dev) 0.35 -> must be in the agent gate.
- Pass 4 (~12:40Z): search harness (run.py, loop.py, pillars.py, policy.py, wm_core.fold_rows/Norm/train_model,
  wm_continual.run_fold, wm_agent P&L/Kelly). Verified: folds expanding, train rows have every label date < fold
  start (asserted), test = Jul Y..Jun Y+1 for Y=2020..2023 (DEV only, see LRT-15); Norm and the refs'
  standardisation fit on train rows only; no early stopping/selection on test rows; per-checkpoint leak test
  compares the model output on a rebuilt perturbed window with the panel prediction; refs logged "not in K"; every
  exp/twin/crash/leakfail row is appended (K = all non-ref rows); continual updates only on rows whose LAST <
  decision day (all targets matured); P&L = w_d * UNG log close(d)->close(d+1) - 10 bp |dw|, folds concatenated;
  Kelly/vol-target sizing identical for WM and ridge/LightGBM (feature-matched gate). Note: short-UNG rules carry
  no borrow cost (UNG borrow is usually cheap but non-zero; add if a short-biased policy wins). LRT-15..18 logged.
  `rt_loopguard.py` baseline taken (b784147 and earlier: protected objects unchanged).
- Pass 5 (~12:50Z): `paper.py` ledger/broker. rt_ledger.py on throw-away books: edit and reorder detected;
  truncation, forged re-hash, sim-mode backfill into a live ledger and split-day P&L NOT detected -> LRT-19, LRT-20.
  Good: live decide refuses outside [15:30, +20 min] ET wall clock and re-decisions; settle refused before 18:00 ET;
  fills at the 16:00 close strictly after the 15:30 decision; models called under no_data_access with ctx =
  asof(t, live=True); stale inputs force flat.
- Pass 6 (~12:55Z): CPC PIT files landed (cpc_fcst 63,624 rows: published_at = Last-Modified for 61,076, issue+1d
  17:00Z for 2,548 (LRT-9 batches); every published_at >= issue + 13.5 h; fa >= pa; no forecast row for a day
  before its issue; cpc_obs 21,790 rows all published >= D+1). wm_store now uses PIT CPC (panel_store rebuilt, no
  NaN). Re-runs: rt_perturb_wmstore 43/43, rt_perturb_store 32/32 + live 2/2 identical, positive controls change.
- Pass 7 (~13:05Z): re-verification of data-lead fixes 8242f93 / 56a0536 and architect a50f1f6.
  rt_ledger: all 6 attacks detected (LRT-19, LRT-20 fixed). px rebuilt (Yahoo per-bar exchange tz): UNG/BOIL bars
  on non-trading days 0 (NG=F 7 on NYSE holidays = Globex holiday sessions, OK); all bars published >= 16:00 ET
  close (NG=F >= 17:00); storage first prints now 0 earlier than rt_ref (2018-12-05 mourning day handled).
  rt_perturb_store 32/32 + live 2/2 and rt_perturb_wmstore 43/43 identical after the rebuild. (rt_pubaudit's
  "285 missing revisions" = pre-2015-06 weeks with no first print in rt_vintage: red-team artefact, ignore.)
  **Verdict to manager: GO for the shadow PAPER loop with rule models only** (tools/live/shadow.sh, live books,
  ctx = asof(t, live=True), fills at the 16:00 close, splits handled, ledger anchored). Conditions: run update.py
  before 15:30 ET so fetched_at <= t; only mode=live books count; COT/EDGAR/om stay off (LRT-11); WM models enter
  the paper loop only after the freeze audit.
- Pass 8 (~13:10Z): wm_store rewritten (46d0206): features are a pure function of the Store.asof ctx, labels from
  Store.labels / storage_labels (dsurp = first-print change of ctx next_storage week minus the store expectation
  known 1 s before its release; lrv5 = log sum of 5 squared UNG returns). rt_perturb_store extended to the WM
  features: 31/31 identical under randomised/inserted future records (incl. 2025-11-14, 2025-12-29/31, 2026-04-23,
  DST days); features non-empty on sampled days. rt_rescore.py written (independent labels + folds + S_F from raw
  PIT); runs when the loop writes preds.
- Pass 9 (13:15Z): ledger heads recorded (independent anchor, LRT-19 residual). 8 live shadow books, genesis only,
  chain ok + git-anchored, mode live: buy_hold 80ef5ab2d61a, fc_ewma 48fc5de6c26b, fc_hist 77f27b3104bc,
  seasonal 41dcfb7d443e, short_ung_carry 4ed396e58f12, storage_surprise d953d1873a35, tsmom_20 7d9cbf46b5f5,
  weather_rev 091a7123e58c. `--sim` books live under runs/live/paper_sim (separate root) and Book refuses a mode
  mismatch; anchor retries on index.lock and logs to runs/live/paper/anchors.log (outside the ledger).
- Pass 10 (13:27Z): `rt_rescore.py` on the frozen panel (LOCKED.sha pinned): labels rebuilt from raw PIT
  (split-adjusted UNG closes, first-print storage + own 5-yr expectation) match panel Y on all 2782 rows
  (max |diff| r1 6e-9, r5 1e-8, lrv5 5e-7, dsurp 6e-6), LAST dates identical; own folds = refs.npz test rows;
  refs re-scored from preds: linqr 1.0000, lgbm 0.9616 = reported. K so far 0 (refs only).
- Pass 11 (13:38Z): first loop rows. Seed-noise rows (struct start recipe, seeds 0/1): S_F 1.6222 / 1.5823 (WM far
  worse than the ridge reference; lgbm ref 0.9616), steps logged per fold (5296-5405), LOCKED sha 51d16b5a7b7977c6,
  leak test pass; rt_rescore re-derives all 4 rows exactly (0 mismatches). K = 2 so far. rt_loopguard: protected
  objects not re-defined in wm_core.py.

## Status at hand-back (13:40Z)

Open: LRT-16 (verify twin steps on first twin rows; no-graph = edge ablation label), LRT-17 (post-selection labels
in FROZEN/report), LRT-11 (COT paused), LRT-9 (disclosed), LRT-4 (INFO). All HIGHs (LRT-5, -10, -19, -20) fixed and
verified; LRT-1 verified. Verdicts given: GO for the rules-only shadow paper loop. VAL / freeze: not yet requested.
Re-run kit before VAL/freeze (all light, nice -n 10): rt_rescore.py (S_F re-derivation + K), rt_loopguard.py,
rt_perturb_store.py (WM features under store perturbation), rt_pubaudit.py, rt_ledger.py (TMPDIR = scratch).

