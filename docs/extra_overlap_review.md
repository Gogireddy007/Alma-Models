# TRAIN_EXTRA overlap review (stamp-extra residuals, 2026-10-30)

Reviewer: L. Andrade. Input: `runs/overlap_extra_stamp.log` (residual JSON at lines 917-8935, 806 hits).
**No TEST data was read for this review.** The classification uses the target-free hit list the scan already
logged, plus the TRAIN series alone (scratch scripts, not committed).

## Residuals, by flagged TRAIN key (unique keys)

| Scan | Hits | TRAIN keys |
|---|---|---|
| VAL hash | 0 | — |
| TEST hash (znorm, L=32) | 16 | wind_farms_hourly 5 (18, 100, 143, 218, 219), m4_daily 2, m4_monthly 1, m4_yearly 3 |
| TEST lag (diff corr ≥ 0.95) | 790 | m4_yearly 346, monash_m3_yearly 10, m4_monthly 10 |

Total: 384 unique series. All of them are in TRAIN_EXTRA, none in TRAIN_LOCAL.

## Classification

- **wind_farms_hourly vs boomlet (probable false positives, not proven).** By provenance (Australian wind farm
  power vs Datadog observability metrics) a real copy is implausible. Keys 218 and 219 are heavily quantized: 281
  and 232 unique values in 7250 points, a median of 9-13 distinct values per 32-point chunk, and idle stretches
  of 0, -0.015, -0.03, … (steps of 0.005). Such chunks pass the low-information filter (≥ 8 distinct values and
  differences), and after z-normalization they coincide with small-integer count metrics. Key 218 alone
  "shares" 39 runs with one boomlet task, which is the signature of a recurring quantized pattern, not a copy.
  Keys 18, 100 and 143 are 46-73% zeros with 0.38-0.56 median mode fraction per chunk (typical series: 0.06).
  Showing this conclusively would need the matching TEST chunks, which I did not read.
- **m4_daily vs boomlet (2 keys).** The same reasoning applies (M4 daily is mostly finance/macro). Probable FP.
- **monash_m3_yearly vs world_life_expectancy and us_consumption_1Y (10 keys, 31 hits).** Probable false positives:
  - every hit is at the minimum overlap (n = 24-34) with r = 0.950-0.985;
  - the same 4 M3 keys (472-475) match 10 different countries at different lags (up to 6 countries per key).
    One series cannot be a copy of six countries. Smooth demographic trajectories have near-collinear first
    differences over 24 years.
- **m4_yearly (346 keys).** Mixed:
  - 99 keys match ≥ 3 different probe series (the same many-to-one smooth-trend pattern, mostly
    world_life_expectancy at n ≈ 24-27), so most are FPs;
  - 20 hits with n ≥ 40 and r ≥ 0.98 are against us_consumption_1Y (gasoline, PCE, motor vehicles: r up to
    0.9996 at n = 46). These are **real duplicates**, the same BEA series that M4 contains, as already found for
    m4_quarterly.
- **m4_monthly (10 keys).** Plausibly real BEA vintages: key 6216 matches PCE at n = 575, r = 0.959, and is also a
  hash hit; the food series match at n = 72-86. Each key matches one probe.

## Action (conservative policy)

- I added all 384 keys to `bench.TRAIN_DROP`, grouped per dataset with a comment citing this scan.
- I added nothing to `overlap_scan.KNOWN_FP`. It is keyed by (task, dataset), so an entry would whitelist whole
  pairs (for example every m3_yearly vs world_life_expectancy hit) instead of the reviewed series. The FPs above
  are argued, not proven, and dropping them costs only 5 of 337 wind_farms_hourly and 10 m3_yearly series.
- **Cost:** none under `TSAR_GIFT_CLEAN` for M4, which is excluded anyway. 15 series (5 wind_farms_hourly,
  10 m3_yearly) are lost otherwise.
- **Check:** default `load_train()` (LOCAL only, flag off, 15,524 series) has the same sha256 of
  (dataset, key, h, season, freq, values) before and after: `e87ea884…a900e`. Only the TRAIN_DROP hash changed
  (a41be7ad → 372a45bd), so `extra_scan_key` changed and the stamp must be re-run.
- Every residual key is now in TRAIN_DROP, so a re-run of stamp-extra with unchanged data and settings should
  report 0 residual. **The re-run is left to the lead** (not run here).

## Follow-ups

- If the FP claim for wind_farms/boomlet must be proven: under `TSAR_SPLIT=test`, logged, print the matched
  z-normalized 32-point TEST chunk next to the TRAIN chunk (contexts only).
- Consider raising the low-information filter (for example the mode fraction per chunk) so quantized idle runs
  stop matching.
