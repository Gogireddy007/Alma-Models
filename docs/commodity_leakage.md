# Commodity leakage record

Commodity lead, 2026-10-31. Inputs: `tools/commodity/leakage_scan.py` -> `runs/commodity/leakage_scan.txt` / `.json`,
`tools/commodity/leakage_runs.py` -> `runs/commodity/leakage_runs.txt`, checkpoint configs, and the independent audit
`docs/audit_commodity.md` (section 1 and 3, which reached the same conclusions with its own probe).

## 1. Alma's training data (all Alma checkpoints used here)

All Alma checkpoints in the TEST list (`wm_ssm_none_s0..s4`, `wm_ssm_direct_s0`) share the same data config:
`train_sha256 = bc78d74b5a3f…`, `TSAR_GIFT_CLEAN=1`, `bench_extra=0`, no GiftEvalPretrain (`TSAR_BENCH_GEP` unset),
10,758 real series from 14 datasets, plus synthetic series generated in code (synth3/synth5/augmentation; no file
reads, docs/audit_sprint.md 2d). Training saw no commodity data and nothing after 2021-09:

| TRAIN dataset | series | timestamps (from the parquet bench.load_train reads) |
|---|---|---|
| dominick | 100,014 | 1989-09-14 .. 1997-05-01 (retail scanner prices/sales) |
| ercot | 8 | 2004-01-01 .. 2021-09-01 (electricity load) — latest real date in TRAIN |
| m5 | 30,490 | 2011-01-29 .. 2016-06-19 |
| monash_australian_electricity | 5 | 2002-01-01 .. 2015-04-01 |
| monash_cif_2016 | 72 | 1970..1979 (placeholder dates) |
| monash_m1_monthly | 617 | 1900..1991 (placeholder dates) |
| monash_m3_monthly | 1,428 | 1857..2005 |
| monash_m3_quarterly | 756 | 1946..1996 |
| monash_nn5_weekly / nn5 | 111 / 111 | 1996-03 .. 1998-05 (ATM withdrawals) |
| monash_tourism_monthly / quarterly | 366 / 427 | 1975 .. 2007-09 |
| monash_traffic | 862 | 2015-01-01 .. 2016-12-31 |
| monash_weather | 3,010 | 1970-01-01 .. 2150-08-25 (synthetic placeholder dates; Australian weather, pre-2017) |

(The bench subsamples at most 2000 series per dataset; the scan reads every series of every file, a superset.)
VALIDATION starts 2025-07-01 and TEST 2025-11-01, so no Alma training value is from either window.

**monash_fred_md** (FRED-MD macro panel, monthly, 1959-01 .. 2019-08) contains monthly WTI (FRED OILPRICEx). It is in
`bench.TRAIN_EXCLUDED` and is NOT in any Alma checkpoint's TRAIN list. It is the scan's positive control: it matches
monthly-averaged WTI with a run of 22 consecutive scale-free 6-grams and 17 consecutive exact-level 6-grams
(series #86), which shows the scan does find a real copy. It ends 2019-08 in any case.

## 2. Content scan for oil / gas / crude-stock series in TRAIN

Method (`leakage_scan.py`): every TRAIN series vs wti/brent/hh daily, their weekly and monthly means, and weekly
crude stocks, 6-grams of (a) levels rounded to 0.01 and (b) consecutive log-ratios rounded to 0.001 (scale-free, so a
rescaled copy matches); constant grams ignored. `leakage_runs.py` then measures the longest run of CONSECUTIVE matching
grams (a copied segment of length L gives a run of about L-6).

| dataset | gram hits | longest consecutive run | verdict |
|---|---|---|---|
| dominick | 38 ratio hits vs brent, 1,822 vs hh | 1 (brent), 3 (hh; e.g. gram (0,0,0,0,0,0.072)) | coincidences on a coarse price lattice (retail prices in cents; HH quoted to 0.01 at ~2-4 USD gives a small set of log-ratios, e.g. many zeros) |
| monash_weather | 5 vs brent, 82 vs hh (ratio) | 1 (brent), 3 (hh) | coincidental; e.g. gram (0,0,0,0,0,0.072) |
| monash_traffic | 2 vs hh (ratio) | 1 | coincidental |
| all other TRAIN datasets | none | – | clean |
| monash_fred_md (excluded) | 36 level + 37 ratio vs monthly WTI | 17 / 22 | real copy of monthly WTI; not in TRAIN |

Conclusion: no oil, gas or crude-stock series is in Alma's training data, and no training value is dated in
VALIDATION or TEST. The independent audit (docs/audit_commodity.md 3a) agrees.

## 3. Time leakage in the harness

Verified by the auditor's perturbation probe (docs/audit_commodity.md 1b-1h): context = observations strictly before
the origin; VAL scoring uses only targets <= 2025-10-31 and DEV only <= 2025-06-30; multiplying or truncating all
post-cutoff values left every VAL forecast and score bit-identical. GARCH (T2 baseline) is fitted on TRAIN only and
filtered causally; vol baselines use returns ending at the last observation; rw_gauss sigma comes from the context.
EIA stocks: the origin is the week-ending Friday W; the context ends at I[W-1], released the Wednesday of week W
(known at the origin); the h=1 target is the next, still unreleased print. Data are current vintage (spot prices are
not revised in practice; EIA stocks can be), not a real-time dataset.

The architect's commodity models (tools/commodity/wm_*.py, search/) train on data <= 2025-06-30 through their own
loaders with date assertions (audit 1e-1g); their freeze and twin matching are covered in the audit (B6), not here.

## 4. Competitor models: release dates vs the windows

| model | release | training data could include VALIDATION (2025-07..10)? | TEST (2025-11 ..)? | treatment |
|---|---|---|---|---|
| Chronos-Bolt-base | Nov 2024 | no | no | clean |
| TiRex | May 2025 | no | no | clean |
| Kronos-small | Aug 2025 (exchange K-line pre-training; commodity futures may be included up to an undisclosed cutoff) | possibly (July 2025) | no | clean for TEST |
| Chronos-2 | Oct 2025 | possibly | no | clean for TEST |
| Timestral-5M | 2026-10-15 | possibly | **possibly** | marked CONTAMINATED; reported, excluded from every "beats" claim. A claim "significantly better than Timestral" (docs/alma_goal.md) cannot be made on this TEST window. |
| AutoETS / AutoARIMA / rw_gauss / GARCH / majority / momentum / seasonal | fitted on each context (GARCH and majority on TRAIN) | no | no | clean |

## 5. Residual risks (not provable either way)

- TEST prices are on disk (`data/commodity/*.csv` to 2026-10-27) and readable by every agent; mitigated by the
  pre-registered fixed model list, the frozen candidate sha, strict TEST mode and the single-run RUN.json.
- The architect's DEV search tried K configurations (count to be stated in FROZEN.json); winner's curse applies to its
  candidate.
