# GIFT-clean training corpus (TSAR_GIFT_CLEAN) and the GiftEvalPretrain proposal

Written 2026-10-30 by Yuki Tanaka (training data). No training run, no download, and no TEST access
(TEST was not read at all). Smoke tests used `tools/smoke.py`. The only remote data read was HF API
metadata and the GIFT-Eval paper.

**User decision:** Alma is trained on a GIFT-clean corpus, so that GIFT-Eval, TIME and fev-bench
results can be called zero-shot. GIFT rule (gift-eval README @ `82f89c3`): a model is `zero-shot` only
if no GIFT test dataset is in training and no GIFT train split is used.

## 1. Verified exclusion list

I checked this against two sources:
- the 28 folders of `Salesforce/GiftEval` @ `30841734` (HF API tree);
- the test-dataset table (Table 13) of arXiv:2410.10393.

The result matches Iyer's list in benchmark_contract.md section 2: 13 TRAIN datasets.

| TRAIN dataset | Where | GIFT test folder | Kind |
|---|---|---|---|
| m4_quarterly | LOCAL | m4_quarterly | exact |
| monash_hospital | LOCAL | hospital | exact |
| monash_car_parts | LOCAL | car_parts_with_missing | exact |
| m4_daily, m4_hourly, m4_monthly, m4_weekly, m4_yearly | EXTRA | m4_* (5 folders) | exact |
| monash_kdd_cup_2018 | EXTRA | kdd_cup_2018_with_missing | exact |
| monash_saugeenday | EXTRA | saugeenday | exact |
| monash_temperature_rain | EXTRA | temperature_rain_with_missing | exact |
| monash_electricity_hourly, monash_electricity_weekly | EXTRA | electricity (15T/H/D/W) | related: same UCI ElectricityLoadDiagrams clients (Table 13 source). Weekly is an aggregate of the same series. |

Other GIFT test folders and how they relate to our data:
- `covid_deaths` is already in TRAIN_EXCLUDED.
- `us_births`, `bitbrains_*`, `bizitobs_application` and `bizitobs_service` are not in our data.
- The other folders (ETT, LOOP_SEATTLE, M_DENSE, SZ_TAXI, bizitobs_l2c, hierarchical_sales, jena_weather,
  solar, restaurant) are VAL families. Selecting on them is selection exposure, not training leakage, and
  is disclosed as such (eval_contract.md section 4).

**Not covered by the flag:** non-bench training. Scripts without `TSAR_BENCH=1` train on
`prepare.load_train_series()`, which still holds M4, hospital and car_parts. `TSAR_GIFT_CLEAN` acts only
inside `bench.load_train`, so a GIFT-clean run must also set `TSAR_BENCH=1`. Synthetic data (synth4,
syn5) is GIFT-free by construction.

## 2. TSAR_GIFT_CLEAN=1 (tools/bench.py)

How it works:
- The 13 datasets are listed in `bench.GIFT_TEST_OVERLAP`, each with its reason.
- `bench.gift_clean()` reads the flag.
- `load_train` drops those datasets from its spec list.

The `TRAIN_EXTRA` stamp guard is unchanged:
- `extra=True` still calls `require_extra_stamp()`.
- `_train_specs`, `extra_scan_key` and both overlap scans still use the full lists. A stamp is therefore
  always made on the full TRAIN_EXTRA, and the flag cannot be used to make a narrower stamp.

Every script that calls `bench.load_train` inherits the flag (train*.py, overlap tools).

**Default is bit-identical.** I ran `load_train(max_len=20000)` (the train scripts' call) before and after
the edit, with the flag off. I hashed it with sha256 over (dataset, key, horizon, season, freq) plus the
float32 bytes of each target:

| | series | finite points | sha256 before | sha256 after |
|---|---|---|---|---|
| LOCAL (extra=False) | 15,524 | 43,606,723 | f470b47a4a36f028... | f470b47a4a36f028... (identical) |
| LOCAL + EXTRA | 32,021 | 68,435,707 | fda6e4f228a8f613... | fda6e4f228a8f613... (identical) |

The EXTRA hash bypassed the stamp guard inside the smoke script, for counting only. No stamp exists yet,
and nothing was trained.

**With the flag on:**

| | datasets | series | finite points | sha256 |
|---|---|---|---|---|
| LOCAL | 14 (17 - 3) | 10,758 | 43,316,671 | 96da45b4bb672dc4... |
| LOCAL + EXTRA | 25 (38 - 13) | 17,569 | 51,786,331 | 5d4cac8873301ff4... |

The flag removes 30% of LOCAL series but only 0.7% of LOCAL points: those sets are short M/Q series.
With EXTRA it removes 24% of points, mostly m4_daily (4.7M), electricity_hourly (6.4M) and
kdd_cup_2018 (2.4M).

The kept datasets get slightly different 2000-series subsamples (monash_weather, m5, dominick,
rideshare, taxi_30min). The subsampling rng is shared across datasets in sequence, and that was left
as is so the default stays bit-identical.

The GIFT-clean hourly and sub-hourly TRAIN sets are:
- LOCAL: ercot, monash_traffic, monash_australian_electricity;
- EXTRA: pedestrian_counts, rideshare, taxi_30min, uber_tlc_hourly, wind_farms_hourly.

Memory caveat: these smokes peaked at 1.8 to 2.4 GB RSS (whole-parquet reads), which is over the 2 GB
smoke limit by up to 0.4 GB.

## 3. DEV split under the flag (tools/devsplit.py, coordinated with Kim)

`monash_car_parts` is GIFT test data. Under `TSAR_GIFT_CLEAN=1`, the `car_parts_1M` DEV task is
replaced by `dominick_1W`:
- source: `dominick`, store x UPC weekly retail;
- size: 100k series, median 272 points, 42% zeros;
- task: h=8, 4 windows, 300 forecasts, 75 series.

Why dominick:
- It keeps DEV's intermittent-retail role (car_parts had 74% zeros).
- It is GIFT-free and is not a VAL/TEST or EXCLUDED family; the import-time asserts check this, and a
  new assert checks that no DEV set is in GIFT_TEST_OVERLAP.
- It leaves m5 (60% zeros, daily) in training for the retail VAL tasks.

Hourly coverage is unchanged. traffic_1H (h=168, h=24) and aus_electricity_30T (h=96, h=336) are still
4 of the 9 DEV tasks. A smoke run of `devsplit.evaluate(naive)` with the flag on scored 9 tasks over
5 families.

In DEV screens the training side keeps only ercot as sub-daily data, with or without the flag.
Recommendation: once the stamp exists, run GIFT-clean DEV screens with `TSAR_BENCH_EXTRA=1` to add the
5 GIFT-free hourly and 30-min extras.

The flag-off DEV is untouched.

Follow-up for Kim: the help text in `tools/search.py` (line 317) still names car_parts as a DEV family.
That is accurate only with the flag off.

## 4. VAL_EXPOSURE (documentation only)

I added Andrade's eval_contract.md section 4 notes to `bench.VAL_EXPOSURE`:
- solar: Chronos-2, Chronos-Bolt and TiRex are each "likely" to have trained on it;
- kdd_cup_2022: TiRex "likely" (GIFT pretrain kdd2022);
- favorita_stores: TiRex "possible" (favorita sales).

The keys are unchanged. `bench.py audit` prints the new text and passes.

## 5. Salesforce/GiftEvalPretrain: listing and subset proposal

(Approved and downloaded by the lead on 2026-10-30; loader and scan in section 6.)

The repo @ `6830b624` (2025-01-21) is Apache-2.0 and holds 6,682 files (HF `save_to_disk` arrow shards)
in 152 dataset folders, **975 GB** in total. Three families make up 99% of it:
- cmip6, 33 folders: 469 GB;
- era5, 30 folders: 387 GB;
- buildings_900k: 63 GB.

The full size listing is in the appendix. Frequencies come from paper Table 14.

### Excluded from any subset

| Reason | Folders |
|---|---|
| VAL family source | kdd2022 (VAL kdd_cup_2022); favorita_sales and favorita_transactions (VAL favorita_stores; fev-excluded); solar_power (VAL solar, same domain; a 4 s series we cannot use anyway); covid_mobility (VAL uk_covid period) |
| TEST family source or fev self-flag | fred_md (TEST us_consumption/gvar, already TRAIN_EXCLUDED); cdc_fluview_ilinet and cdc_fluview_who_nrevss (ILI, related to TEST ecdc_ili); spain (Spanish national load, likely ENTSO-E data = TEST entsoe); gfc12/14/17_load (fev proenfo_gfc tasks, would need fev self-flags); residential_pv_power (PV, related to TEST solar_with_weather) |
| Duplicate of our TRAIN | m1_*, monash_m3_*, cif_2016_*, tourism_*, nn5_*, traffic_hourly/weekly, australian_electricity_demand, elecdemand (Victoria demand, within australian_electricity), m5, weather, rideshare, pedestrian_counts, taxi_30min, uber_tlc_*, wind_farms. Several are DEV families (traffic, aus_electricity, nn5, tourism) and would break the DEV holdout. |
| Too big for 8 GB, or off-target | cmip6, era5, buildings_900k, largest_* (17.8 GB; also a superset of the PeMS sets), azure_vm_traces, borg, extended_web_traffic, residential_load_power (1-min), lcl (same London trial as london_smart_meters) |
| Allowed but not chosen | pdb, elf, covid19_energy (ProEnFo collection, next to the fev proenfo tasks); beijing_air_quality (stations likely inside china_air_quality); wind_power (4 s); project_tycho (healthcare); daily, weekly or monthly sets (wiki-rolling, kaggle_web_traffic_weekly, subseasonal*, godaddy, bitcoin, sunspot, vehicle_trips), which are outside the gap |

### Proposed subset: 1.58 GB, about 268M points (E), 25 folders, 1 shard each

The gap analysis puts 61% of the gap in H/sub-hourly tasks and 43% in multi-seasonal ones. The subset
therefore draws mostly from 5 to 30 min data with daily and weekly cycles:

| Domain | Folders (freq, MB) | MB | Points (E from Table 14) |
|---|---|---|---|
| Transport, sub-hourly road traffic | PEMS03 (5T, 37.5), PEMS04 (5T, 62.6), PEMS07 (5T, 99.7), PEMS08 (5T, 36.4), PEMS_BAY (5T, 67.8), LOS_LOOP (5T, 28.4), Q-TRAFFIC shard 0 of 3 (15T, 353.0) | 685.4 | 155M |
| Transport, metro ridership | BEIJING_SUBWAY_30MIN (30T, 22.6), SHMETRO (15T, 20.3), HZMETRO (15T, 1.5) | 44.4 | 2.3M |
| Energy, buildings and households | london_smart_meters shard 0 of 2 (30T, 339.9); bdg-2 bear/fox/panther/rat, hog, bull, cockatoo (H, 59.8); ideal, smart, borealis, sceaux (H, 5.8) | 405.5 | 95M |
| CloudOps | alibaba_cluster_trace_2018 shard 0 of 10 (5T, 305.1) | 305.1 | 9.5M |
| Nature | china_air_quality (H, 137.8), oikolab_weather (H, 3.2) | 141.0 | 6.5M |
| **Total** | | **1,581.7** | **about 268M (5x our GIFT-clean 52M)** |

**Lite tier: 0.58 GB, about 88M points.** It drops Q-TRAFFIC, london_smart_meters and alibaba and
keeps all the other folders. Approve this one first if 1.6 GB is too much disk or scan time.

Loader work: HF `save_to_disk` arrow files with a `target` field. The 17 multivariate sets (PEMS,
LOS_LOOP, metro, alibaba) need their variates split, as bench does. This would come as a new
`TRAIN_GEP` list, behind its own flag and stamp, in `bench.load_train`.

### Overlap risk against our VAL and TEST fev families

| Subset family | Nearest VAL/TEST family | Risk |
|---|---|---|
| PEMS*, LOS_LOOP, Q-TRAFFIC (LibCity / Caltrans / Baidu) | VAL LOOP_SEATTLE (WSDOT), M_DENSE, SZ_TAXI (also LibCity) | Low: different cities and sensors. LibCity is a common packager, though, so the scan must confirm. |
| metro ridership | none | Very low |
| london_smart_meters, bdg-2, ideal, smart, borealis, sceaux | TEST entsoe (national load), VAL ETT (transformer) | Low: building and household scale, not grid scale |
| alibaba cluster trace | VAL bizitobs_l2c, TEST redset | Low to medium: same CloudOps domain, different operators |
| china_air_quality | TEST uci_air_quality (Italy, 2004-05) | Low: different country and period. Any Italian station must be checked. |
| oikolab_weather (Melbourne) | VAL jena_weather | Very low |

Because GIFT built GiftEvalPretrain to exclude GIFT test data, the residual risk is against fev
VAL/TEST, not against GIFT.

### Scan plan before any training use (tools/overlap_scan.py)

1. After approval, download only the listed shards, pinned to rev `6830b624`, with a sha256 manifest.
2. Add `TRAIN_GEP` to `bench.extra_scan_key` (or a separate GEP stamp keyed the same way), so that
   `require_*_stamp` blocks training until the scan is clean.
3. Run `bench.overlap` (exact plus znorm, L=32) of every GEP series against VAL contexts and targets and
   TEST contexts (target-free, `TSAR_SPLIT=test`, logged).
4. Run `overlap_scan.py lag` (growth correlation >= 0.95 over >= 60 points) on the same-frequency
   classes. The ones that matter are 5T/15T/30T/H against LOOP_SEATTLE, SZ_TAXI, M_DENSE, ETT,
   bizitobs, jena, solar, entsoe, uci_air_quality and redset. Long 5T series make the FFT costly, so
   chunk them and queue the scan through jobq as one heavy job, not as a smoke test.
5. Each hit is either dropped (`TRAIN_DROP`) or documented as a false positive. Write the stamp only
   when the scan is clean.
6. Then re-run the GIFT-clean dataset and point-count log for flag on/off plus GEP, and update
   `config.json` (`testdata_leakage: No`, `model_type: zero-shot`).

## 6. TSAR_BENCH_GEP=1: loader and scan (2026-10-30, steps 2-5 of the scan plan)

**Status: loader and stamp-gep implemented; the full scan has NOT been run; training on GEP is blocked
until it writes `data/bench/overlap_gep.json`.** No TEST access in any smoke (partial scans used
`--splits val`).

### Data on disk

`data/bench/gep/` holds 25 folders at rev `6830b624de7e...`, one arrow shard each. Its sha256 manifest is
1,581,673,939 bytes over 75 files, and the manifest's own sha256 `abe5ed94efca...` is pinned in
`bench.GEP_MANIFEST_SHA256`. `bench.gep_verify()` checks the pinned manifest, the repo and revision, the
total bytes and the size and sha256 of every file, once per process (about 3 s). The loader, the stamp
key and stamp-gep all call it.

Measured layout (`target` column; multivariate rows are `fixed_size_list<list<float>>`):

| Folder | Freq | Rows x variates | Univariate series | Length (min/median/max) | Points |
|---|---|---|---|---|---|
| PEMS03 | 5T | 358 x 1 | 358 | 26,208 | 9.4M |
| PEMS04 | 5T | 307 x 3 | 921 | 16,992 | 15.6M |
| PEMS07 | 5T | 883 x 1 | 883 | 28,224 | 24.9M |
| PEMS08 | 5T | 170 x 3 | 510 | 17,856 | 9.1M |
| PEMS_BAY | 5T | 325 x 1 | 325 | 52,128 | 16.9M |
| LOS_LOOP | 5T | 207 x 1 | 207 | 34,272 | 7.1M |
| alibaba_cluster_trace_2018 (shard 0/10) | 5T | 5,841 x 2 | 11,682 | 104/1,728/1,728 | 19.1M |
| Q-TRAFFIC (shard 0/3) | 15T | 15,050 x 1 | 15,050 | 5,856 | 88.1M |
| SHMETRO | 15T | 288 x 2 | 576 | 8,809 | 5.1M |
| HZMETRO | 15T | 80 x 2 | 160 | 2,377 | 0.4M |
| BEIJING_SUBWAY_30MIN | 30T | 276 x 2 | 552 | 1,572 | 0.9M |
| london_smart_meters_with_missing (shard 0/2) | 30T | 2,760 x 1 | 2,760 | 10,176/31,104/39,648 | 84.9M |
| bdg-2 bear/fox/panther/rat | H | 611 x 1 | 611 | 8,760-17,544 | 9.4M |
| hog, bull, cockatoo | H | 66 x 1 | 66 | 17,544 | 1.1M |
| ideal, smart, borealis, sceaux | H | 238 x 1 | 238 | 393-34,223 | 1.5M |
| china_air_quality | H | 437 x 6 | 2,622 | 1,950/8,759/397,335 | 34.4M |
| oikolab_weather | H | 8 x 1 | 8 | 100,057 | 0.8M |
| **Total** | | | **37,529** | | **about 328M** |

The proposal estimated 268M points from Table 14; the shards hold more, mostly Q-TRAFFIC and london.
SHMETRO, HZMETRO and BEIJING_SUBWAY are 23-43% NaN (closed hours), and china_air_quality is 18% NaN.

### Loader (tools/bench.py)

- `TRAIN_GEP` lists (folder, training dataset, season, freq, horizon). Folders of one domain and frequency
  share a dataset name, so the dataset-uniform samplers see 9 GEP datasets, not 25:
  `gep_traffic_5min` (PEMS*, LOS_LOOP), `gep_cloud_5min`, `gep_qtraffic_15min`, `gep_metro_15min`,
  `gep_metro_30min`, `gep_london_30min`, `gep_buildings_H` (bdg-2, hog, bull, cockatoo, ideal, smart, borealis, sceaux),
  `gep_air_H` and `gep_weather_H`. Under GIFT_CLEAN + EXTRA that is 9 of 34 real datasets.
- `_GepShard` memory-maps the shard (`pa.memory_map` + IPC stream) and works per record batch, so nothing
  is concatenated or converted through pandas. Each univariate series is a float32 view into the map.
  Multivariate rows are split into series keyed by row-major (row, variate) position, before any cap,
  as bench does with `_target_arrays`.
- `load_gep()` caps per folder. It goes through the series in a fixed random order (rng seeded by
  `[SEED, crc32(folder)]`, independent of the LOCAL/EXTRA rng) and keeps series with at least
  MIN_CONTEXT finite points until it has `GEP_MAX_SERIES` = 2000 series or `GEP_MAX_POINTS` = 4M points.
  Each kept series is a random contiguous crop of at most `GEP_MAX_LEN` = 4096 points (also at most
  load_train's `max_len`). A random crop, not the tail, because co-recorded sensors share one calendar.
- `GEP_DROP` ({folder: [key]}, empty) is applied after selection. It is kept apart from TRAIN_DROP, because
  `extra_scan_key()` hashes TRAIN_DROP and a GEP drop would otherwise make the EXTRA stamp stale.
- `load_train` reads `TSAR_BENCH_GEP` (default off). When it is set, it calls `require_gep_stamp()` first
  and appends `load_gep()` after LOCAL/EXTRA, so the LOCAL/EXTRA part of the list does not change.
- `require_gep_stamp()` needs `data/bench/overlap_gep.json` with result clean and a key equal to
  `gep_scan_key()`: sha256 over GEP_REV, TRAIN_GEP, the verified file sha256s and GEP_DROP.
  Without the stamp, `TSAR_BENCH_GEP=1 ... load_train()` raises PermissionError (checked).
- `bench.overlap()` gained `source=` (a (name, key, y) iterable in place of the TRAIN bank) and `tasks=`.
  The TRAIN bank code moved into `_train_source()` unchanged.
- The key `TSAR_BENCH_GEP` is in the strict-key lists of train_search.py, train_v2.py and train_arch2.py.
  train_wm.py (Haddad) must add it to its `_TOOL_KEYS`.

**Default is bit-identical.** `load_train(max_len=20000)` with HEAD (8062347) bench.py and with the new
one. The sha256 is over (dataset, key, horizon, season, freq) plus the float32 bytes of each target:

| Config | series | finite points | sha256 (HEAD = new) |
|---|---|---|---|
| LOCAL | 15,524 | 43,606,723 | e0e9f0914742a81c... |
| LOCAL + EXTRA | 31,976 | 68,397,846 | a8b092ae04aadc8a... |
| GIFT_CLEAN, LOCAL + EXTRA | 17,554 | 51,749,820 | 609cb287d0005204... |

**With GEP** (TSAR_GIFT_CLEAN=1, EXTRA, TSAR_BENCH_GEP=1; the stamp check was bypassed inside the smoke
script for counting only, and nothing was trained):
- 27,926 series, 85,678,676 finite points, sha256 8941cc29e60c933e....
- GEP adds 10,372 series and 33.9M finite points, so the GIFT-clean real corpus grows from 51.7M to 85.7M
  finite points (+66%).
- load_train peak RSS was 1,318 MB (1,169-1,334 MB without GEP), and it took 10.5 s.

| Folder | series | points | finite |
|---|---|---|---|
| PEMS03 | 358 | 1,466,368 | 1,466,368 |
| PEMS04 | 921 | 3,772,416 | 3,772,416 |
| PEMS07 | 883 | 3,616,768 | 3,616,768 |
| PEMS08 | 510 | 2,088,960 | 2,088,960 |
| PEMS_BAY | 325 | 1,331,200 | 1,330,996 |
| LOS_LOOP | 207 | 847,872 | 847,872 |
| alibaba_cluster_trace_2018 | 2,000 | 3,278,958 | 3,278,958 |
| Q-TRAFFIC | 977 | 4,001,792 | 4,001,792 |
| SHMETRO | 576 | 2,359,296 | 1,794,121 |
| HZMETRO | 160 | 380,320 | 292,000 |
| BEIJING_SUBWAY_30MIN | 552 | 867,744 | 496,800 |
| london_smart_meters_with_missing | 977 | 4,001,792 | 4,001,648 |
| bdg-2 bear / fox / panther / rat | 91 / 135 / 105 / 280 | 2,502,656 | 2,502,656 |
| hog / bull / cockatoo | 24 / 41 / 1 | 270,336 | 270,336 |
| ideal / smart / borealis / sceaux | 217 / 5 / 15 / 1 | 858,579 | 858,579 |
| china_air_quality | 1,003 | 4,002,193 | 3,275,818 |
| oikolab_weather | 8 | 32,768 | 32,768 |

### stamp-gep (tools/overlap_scan.py)

`TSAR_SPLIT=test uv run tools/overlap_scan.py stamp-gep --out runs/overlap_gep_hits.json`

It scans every full GEP series, before caps and GEP_DROP:
1. **Hashes.** `bench.overlap` (exact + znorm, L=32) against VAL contexts + targets and TEST contexts
   (target-free, logged).
2. **Lag scan.** Growth correlation >= 0.95 against every VAL/TEST task of the same frequency class
   (5min, 15min, 30min, H). This covers bizitobs_l2c, boomlet, redset, ETT, SZ_TAXI, entsoe,
   solar_with_weather, LOOP_SEATTLE, M_DENSE, jena, epf and uci_air_quality.
   - Long series are cut into overlapping segments, with overlap at least the longest probe. Every
     alignment then lies inside one segment, and Pearson r does not change under per-segment
     standardisation. A segmenting unit test found a copy at absolute lag 41,000 in a 60k series.
   - Probes are grouped by length, so short probes get small FFTs.
   - The FFTs are scipy with 4 threads; `--workers 0` uses numpy. The selftest gives the same hits on
     both paths.
   - Sub-daily classes need n >= max(60, 2 x season) aligned points: 576 at 5min, 192 at 15min, 96 at
     30min, 60 at H. Smooth diurnal series (temperature, load) correlate by phase over a day. A
     re-released copy matches over its whole span, and short exact or affine shared stretches are
     already caught by the hashes.
3. **Stamp.** It writes the stamp only when no residual hit remains (GEP_DROP- and GEP_KNOWN_FP-aware).
   Otherwise it lists the hits and exits with code 1.
   - `--folders`, `--tasks` and `--splits val` make a partial scan that never stamps.
   - A full scan caches each part (hash_val, hash_test, lag_<class>) in `runs/gep_scan/`, keyed by the
     verified data, so a crashed run resumes. After review, a rerun with new drops re-uses all parts.

**Smokes (VAL only, partial, all clean, 0 hits):**

| Folders | Tasks | Time | Peak RSS |
|---|---|---|---|
| HZMETRO, sceaux | 4 VAL tasks | 14 s | 365 MB |
| bdg-2_bear, LOS_LOOP | 3 tasks | 58 s | 687 MB |
| bdg-2_panther, sceaux, HZMETRO | grouped path | 35 s | 320 MB |

Measured lag rates were about 1,700-2,900 pairs/s at FFT 4-8k, 530-800 at 16k and 130-270 at 32k (the
machine was shared with a training job).

**Full-scan estimate:** about 4-9 h and 1.5-2.5 GB. Lag 5min 2.3-4.3 h, 15min 1.7-2.6 h, 30min
0.5-0.7 h, H 0.5-1.5 h; hashes about 10 min. This is too heavy for a smoke. The command is in
`runs/jobs/gep_scan.txt`. jobq cannot run it: it refuses non-VAL splits and passes no arguments. The lead
runs it by hand, like stamp-extra.

### Remaining steps

1. The lead runs the scan.
2. Review the hits. Add real duplicates to `GEP_DROP`, or add false positives to `GEP_KNOWN_FP` with a
   note.
3. Re-run until the stamp is written.
4. Screen `TSAR_BENCH_GEP=1` (GIFT_CLEAN + EXTRA) against gc_extra under the usual 3-seed rule.
5. Update `config.json`: `testdata_leakage: No`, `model_type: zero-shot`.

## Appendix: GiftEvalPretrain folder sizes (MB, HF API, rev 6830b624)

BEIJING_SUBWAY_30MIN 22.6; HZMETRO 1.5; LOS_LOOP 28.4; PEMS03 37.5; PEMS04 62.6; PEMS07 99.7; PEMS08 36.4; PEMS_BAY 67.8; Q-TRAFFIC 1059.1; SHMETRO 20.3; alibaba_cluster_trace_2018 3049.4; australian_electricity_demand 4.6; azure_vm_traces_2017 10632.1; bdg-2_bear 5.9; bdg-2_fox 9.3; bdg-2_panther 3.7; bdg-2_rat 18.9; beijing_air_quality 18.5; bitcoin_with_missing 0.3; borealis 0.3; borg_cluster_data_2011 15059.0; buildings_900k 63020.1; bull 11.5; cdc_fluview_ilinet 1.3; cdc_fluview_who_nrevss 0.7; china_air_quality 137.8; cif_2016_12 <0.1; cif_2016_6 <0.1; cockatoo 0.4; covid19_energy 0.9; covid_mobility 0.6; elecdemand 0.1; elf 0.1; extended_web_traffic_with_missing 1487.7; favorita_sales 560.2; favorita_transactions 0.3; fred_md 0.3; gfc12_load 12.6; gfc14_load 0.1; gfc17_load 2.3; godaddy 1.1; hog 10.1; ideal 5.0; kaggle_web_traffic_weekly 70.7; kdd2022 189.1; lcl 38.2; london_smart_meters_with_missing 665.1; m1_monthly 0.2; m1_quarterly <0.1; m1_yearly <0.1; m5 234.8; monash_m3_monthly 0.6; monash_m3_other 0.1; monash_m3_quarterly 0.2; monash_m3_yearly 0.1; nn5_daily_with_missing 0.3; nn5_weekly 0.1; oikolab_weather 3.2; pdb 0.3; pedestrian_counts 12.5; project_tycho 5.6; residential_load_power 1752.0; residential_pv_power 1504.1; rideshare_with_missing 3.5; sceaux 0.1; smart 0.4; solar_power 29.6; spain 0.6; subseasonal 227.2; subseasonal_precip 39.1; sunspot_with_missing 0.3; taxi_30min 221.9; tourism_monthly 0.4; tourism_quarterly 0.2; tourism_yearly 0.1; traffic_hourly 59.9; traffic_weekly 0.4; uber_tlc_daily 0.2; uber_tlc_hourly 4.5; vehicle_trips_with_missing 0.1; weather 171.8; wiki-rolling_nips 163.7; wind_farms_with_missing 688.7; wind_power 29.6; cmip6_1850..2010 (33 folders) 469.2 GB; era5_1989..2018 (30 folders) 386.5 GB; largest_2017..2021 (5 folders) 17.8 GB.
