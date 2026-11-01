# Audit: Alma-Power (DE-LU day-ahead electricity), pre-TEST (2026-10-31, ~21:25 ET)

Auditor: independent auditor with veto, held to the same standard as `docs/audit_commodity.md`.

Scope:
- Prereg: `docs/power_prereg.md` (5730a54) and amendment A1 (4085a95).
- Harness: `tools/power/*.py` (fc391d7, unchanged since 7e21933).
- Data pin: `data/power/manifest.json` (03ffa6d).
- Lead VAL: `docs/power_results.md` (32fe55c).
- Architect: `docs/power_search.md`, `runs/power/FROZEN.json`, `tools/power/wm_*.py`, `tools/power/search/**`
  (freeze 7e21933).

Everything was checked at HEAD 32fe55c. `git status` is clean for `tools/power`, `docs/power_prereg.md` and
`runs/power/FROZEN.json`.

Checks I ran myself:
- **Info-set tests:** `tools/power/test_infoset.py` with the models naive_epf, arx, lear, wm_power, wm_power_direct,
  chronos2_cov and alma_s0 (peak memory 757 MB), and `tools/power/search/test_leak.py`.
- **Strict-mode preflight:** `run_eval.preflight()` on the exact TEST model list. It writes no RUN.json.
- **Hashes:** sha256 of every pinned file.

## Verdict: **GO**

There are no blocking findings. The non-blocking items listed under "Concerns" must be disclosed in the write-up.
Run the command exactly as pre-registered. I recommend `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, because the
foundation models load lazily after RUN.json is written.

## Checklist

| # | item | status | evidence |
|---|---|---|---|
| 1 | Prereg before any score | PASS | 5730a54 at 20:13:28. The first forecast file (`runs/power/val/naive_d.npz`) is from 20:24:35, and the architect's protocol (b3db266, 20:16:33) precedes every DEV score. `runs/power/test/` does not exist, and no TEST forecast or score file exists anywhere. |
| 2 | Lead flag 1: TEST-month means printed (A1) | CONCERN, disclosed | The means were seen between 20:13 and 20:24: prices rose from about 85 to about 170 EUR/MWh. That was after the prereg had fixed every lead model setting. The only later lead code change (4085a95, LEAR `threadpool_limits(1)` for determinism) has no tunable content. The architect's selection rule (S, top-3, 3-seed mean) was declared at 20:20 and applied mechanically (item 6). Residual risk: the level shift was known to both agents before the freeze. Keep the A1 disclosure in every result. |
| 3 | Data pin | PASS | `hourly.csv` sha256 `79a2c96e…6dea` equals `run_eval.DATA_SHA256`, which strict mode enforces (`run_eval.py:477-478`). The manifest records the raw URLs, download times and sha256. |
| 4 | Information-set gate: prices ≤ D-1, fundamentals ≤ D-2 | PASS | `data.build_context` (`data.py:184-218`) asserts this with UTC times. The D-1 price was cleared at noon D-2, before the noon D-1 forecast time, and no context hour overlaps D. Covariate rows for D-1 are set to NaN, and the last known covariate day must be ≤ D-2. My `test_infoset.py` run, on 16 days including DST days: the perturbation, sensitivity and training-helper checks pass. Forecasts with the future perturbed are bit-identical for naive_epf, arx, wm_power, wm_power_direct, chronos2_cov and alma_s0; LEAR differs by 2.2e-12, which is solver noise also present between repeat calls. `test_leak.py` (architect's batch builder, all 6 arch × abl): perturbing P_D, P_D+1, F_D-1 or F_D changes 0/20 forecasts; perturbing P_D-1 or F_D-2 changes 20/20. ALL OK. Code review: chronos2_cov's 24-hour shift puts the last non-NaN covariate at the final hour of D-2 (`models.py` `_chronos`). LEAR uses P[D-1..D-7] and C[D-2], C[D-7]. The wm_power wrapper slices covariates D-L-1..D-2 and asserts the context ends at D-1 (`wm_power.py:55-73`). |
| 5 | Lead flag 4: architect training data | PASS | Search data come from the SMARD files `runs/power/search/smard/dev/{price,load,wind_on,wind_off,solar}.json`. All have `end_exclusive` 2025-07-01, and the last timestamp is 2025-06-30 23:00 Europe/Berlin (non-null to the end). There is no VAL or TEST cache: no `search/raw/`, no `smard/val/`. The **frozen** checkpoints were refit on the lead panel (`args.data == "lead"`, `wm_data.LeadData` → `data.train_panel()`, which raises past 2025-06-30). Their `train_end` is 2025-06-30. Search backtest models were trained on targets ≤ 2023-12-31. |
| 6 | Lead flag 2: freeze vs VAL timing | PASS (doc timestamps wrong) | Commit 7e21933 (21:01:41) contains the final `FROZEN.json`, `wm_power.py` with all six sha256 values, the six checkpoints, `results.tsv`, `ens_A/B.json` and §3 of `docs/power_search.md` (selection table and rule). None of these files has changed since (`git diff 7e21933 HEAD` is empty for them). The VAL npz for `wm_power*` were created at 21:01:54 and `val/metrics.json` at 21:01:59, both **after** the commit. Every selection input is in `results.tsv` and `ens_*.json` (last write 20:58), from DEV only. The architect never fetched VAL data. So no VAL number could have shaped the selection. The rule outcome checks out against `results.tsv`: B scores 0.6435 / 0.6415 / 0.6421 (mean 0.6424), A 0.6420, C 0.6399. "Written 21:05" (§3) and `frozen_at 21:08` (FROZEN.json) are later than the commit that contains them, so they cannot be true. They are mislabelled times; correct them in a dated note (non-blocking). |
| 7 | Lead flag 3: wm_power pinning | PASS | The preflight FROZEN.json cross-check keys on `meta["ckpt"]`, which the ensembles do not have, so it does not apply to them. Instead, `wm_power.Ensemble` asserts each member's sha256 at load (`wm_power.py:83-84`). Those hashes live in the committed `wm_power.py`, which is covered by the strict dirty-tree check. My preflight confirmed three things: the 6 wrapper hashes equal the 6 in `FROZEN.json["sha256"]`; they equal `shasum` of the files; and RUN.json will record them in `model_meta.members`. Recommended, not blocking: also cross-check ensemble members against FROZEN.json in `preflight`. |
| 8 | Matched direct twin | PASS | The six `frz_phys_{none,direct}_s{0,1,2}.pt` share arch phys, d 64, L 14, lr 1e-3, aux 0.3, 1500 steps, data "lead", train_end 2025-06-30 and 91,845 params. They differ only in `abl` and are paired by seed. |
| 9 | K and winner's curse | PASS | FROZEN.json and `docs/power_search.md` §3 record: 49 training/eval runs, 45 scored rows, 33 distinct recipes, plus smoke, crash, reference and refit rows. Best single row 0.6464 vs selected 3-seed mean 0.6424, with about 30 recipes inside ±0.005. "The WM does not beat its direct twin on DEV": ensemble difference 90% CI [-0.0063, +0.0057]. |
| 10 | TEST command scores every pre-registered competitor in one shot | PASS | `PREREG_MODELS` (`run_eval.py:44-45`) matches prereg §5 exactly (14 models) plus the 2 FROZEN models, 16 in total. `majority` and `dow_majority` are computed inside `score()`. Preflight on that list: no import errors, every model resolves, Alma pins match, data pin matches. Kronos runs on every 5th day, as pre-registered. Strict-mode guards (`run_eval.py:466-492`) are the same design I broke-tested for commodity: FROZEN required, no RUN.json, no npz, no `--models/--force`, clean tree, data sha, pinned Alma, preflight before RUN.json. |
| 11 | Contamination | PASS | Timestral is flagged CONTAMINATED and excluded from claims (`run_eval.py:50`, `add_holm`). Chronos-2 (Oct 2025) and Kronos (Aug 2025) predate TEST. Alma's pre-training data contain no electricity prices (`docs/commodity_leakage.md`). |
| 12 | Statistics | PASS | P1 and P3 each have one label per day, tested with DM-HLN h=1 plus McNemar. P2 is aggregated to a day-level hit rate with a 7-day circular block bootstrap. P4 uses daily QS/AE differences. A claim needs Holm-own p < 0.05 **and** a block-bootstrap CI that excludes 0 (`add_holm`). |

## Concerns (disclose; not blocking)

1. **The TEST price level was known before the freeze (A1).** The doubling of the mean was visible to the lead and,
   through the prereg, to the architect. Both selection procedures were mechanical or pre-declared, but say this
   plainly.
2. **The VAL rehearsal predates the freeze.** `rehearsal_val/RUN.json` started at 20:52 at commit 03ffa6d, with
   `tools/power` dirty, and **without** `wm_power*`. Mitigation: the code is identical to the frozen state, the
   strict preflight on the full TEST list passed (item 10), and `wm_power*` ran through `forecast()` on all 123 VAL
   days. If time allows, re-rehearse into a fresh directory first.
3. **Mislabelled freeze timestamps (item 6).** Fix them with a dated correction.
4. **`test_infoset.py` makes model forecasts on sampled TEST days.** They are in-memory invariance checks: never
   saved, never compared with targets. Restrict them to DEV/VAL days in future.
5. **Regime change.** TEST is entirely in the 15-minute-product regime; DEV is entirely hourly. The level has roughly
   doubled. The direction and level results must be read with that in mind (the prereg discloses this).
6. **Lazy foundation-model loads after RUN.json.** Use HF offline mode so a network failure cannot consume the TEST.

---

# TEST results audit (2026-10-31, ~22:20 ET)

Inputs: `runs/power/test/{RUN.json, hashes.json, metrics.json, report.md, test_run.log}` and 16 forecast npz files,
committed in e57c949. The independent re-derivation is `tools/audit_power_test.py`. It reads only
`data/power/hourly.csv` and the npz files, and rebuilds the panel, labels, DEV majorities, metrics, DM, block
bootstrap and Holm itself; it imports nothing from `tools/power`.

## Integrity: PASS

- **One complete run.** RUN.json: status complete, 21:18:40 to 22:06:31, commit 06b9a64 (my GO commit; `tools/power`
  unchanged since 7e21933), `tools_power_dirty == ""`, argv `run_eval.py --split test`, the 16 pre-registered and
  frozen models.
- **Pins match.** `data_sha256` = the A1 pin and equals the file. `frozen_sha256` equals `runs/power/FROZEN.json`.
  The wm_power / wm_power_direct member shas in the npz metas equal FROZEN.json. The Alma shas equal the pins.
- **Hashes match.** `hashes.json` (17 entries) lists exactly the npz on disk plus metrics.json; every sha256 matches;
  its own hash equals `RUN.json.hashes_sha256`.
- **No file predates the run.** Every npz has `created >= started` and `split == "test"`. Days run 2025-11-01 to
  2026-10-31 (365; Kronos 73).
- **The log is clean.** It contains no error, failure or skip.

## Re-derivation: 243 of 243 numbers match exactly

What was re-derived:
- For all 18 entries (16 models plus majority and dow_majority): P1, P2 and P3 accuracy.
- For all 16 models: P4 QS, MAE, 80% coverage, 95% coverage where native, and QS skill vs naive_epf.
- wm_power vs lear, chronos2, chronos2_cov, tirex, chronos_bolt_base, wm_power_direct, naive_w and dow_majority on
  P1, P2, P3, P4-QS and P4-MAE: DM p, own-Holm p, block-bootstrap CI and the §7 "beats" flag.

Everything agrees to 1e-9, so the harness numbers can be quoted.

A coincidence I checked: wm_power has P1 = P3 = 0.7315, and wm_power_direct and tirex (P1) have the same value.
These are independent label sets, recomputed from the raw panel, that each happen to give 267 of 365 hits.

## Headline table (TEST, 365 days; higher accuracy is better, lower QS/MAE is better; Timestral is contaminated)

| model | P1 | P2 | P3 | QS (skill vs naive_epf) | MAE | cov80 / cov95 |
|---|---|---|---|---|---|---|
| **wm_power** | 0.732 | **0.747** | 0.732 | 9.02 (0.372) | 22.43 | 0.752 / 0.913 |
| wm_power_direct | 0.732 | 0.739 | 0.732 | 9.08 (0.369) | 22.66 | 0.740 / 0.912 |
| chronos2 | 0.726 | 0.740 | 0.701 | **8.67 (0.397)** | **21.82** | 0.727 / 0.943 |
| chronos2_cov | 0.723 | 0.727 | **0.745** | 8.85 | 22.16 | 0.750 |
| tirex | 0.732 | 0.737 | 0.693 | 8.80 (0.388) | 22.07 | 0.764 |
| lear | 0.737 | 0.731 | 0.704 | 9.18 (0.362) | 22.77 | 0.738 / 0.906 |
| lear_nocov | **0.742** | 0.736 | 0.696 | 9.17 | 22.67 | 0.748 |
| arx | 0.740 | 0.728 | 0.701 | 9.49 | 23.35 | 0.793 |
| chronos_bolt_base | 0.723 | 0.730 | 0.699 | 9.08 | 22.75 | 0.784 |
| alma_s0 (zero-shot) | 0.668 | 0.661 | 0.644 | 11.04 | 27.22 | 0.703 |
| naive_epf / naive_w | 0.578 / 0.603 | 0.577 / 0.632 | 0.616 / 0.655 | 14.38 / 16.75 | 34.13 / 40.16 | 0.763 / 0.750 |
| dow_majority / majority | 0.644 / 0.545 | 0.597 / 0.504 | 0.625 / 0.526 | - | - | - |

## What can be claimed (prereg §7: better, AND own-Holm p < 0.05, AND the 7-day block-bootstrap CI excludes 0)

1. **wm_power significantly beats the naive baselines.**
   - It beats naive_d, naive_epf, naive_w and majority on every target. For example, vs naive_w: P2 +11.6pp, P1
     +12.9pp, QS -7.7 EUR/MWh.
   - It beats dow_majority on P2 (+15.0pp) and P3 (+10.7pp). It does **not** beat dow_majority on P1: raw p 0.006
     but own-Holm ≥ 0.05.
   - It beats kronos_small on every target. Kronos is scored on 73 days with reduced settings, which must be stated.
   - It beats zero-shot Alma (alma_s0, alma_direct_s0) on P2, P3, QS and MAE, but not P1.
2. **Numerical rank (no significance attached).**
   - P2 hourly direction: #1 at 0.747, ahead of chronos2 0.740, tirex 0.737 and lear_nocov 0.736.
   - P3: #2.
   - P1: it ties tirex and its twin at 0.732, *behind* lear_nocov 0.742, arx 0.740 and lear 0.737. It is not tied
     for first.
   - P4: #4 on QS and on MAE, behind chronos2, tirex and chronos2_cov.
3. **Skill.** wm_power's QS skill vs the EPF-standard naive is 0.372 (rMAE 0.657) in a year when price levels
   roughly doubled.

## What cannot be claimed

1. **That wm_power beats LEAR, ARX or any foundation model.** It has no significant win over lear, lear_nocov, arx,
   chronos2, chronos2_cov, tirex or chronos_bolt_base on any target.
   - The closest case is P2 vs chronos2_cov: +2.1pp, raw p 0.040, CI excludes 0, but own-Holm well above 0.05.
   - The P4 QS gap to chronos2 is +0.36 EUR/MWh against wm_power (p 0.09).
   - Never say "best model" or "beats state of the art".
2. **Any world-model claim.**
   - The prereg (§9) tests the candidate vs its twin with "Holm over the endpoints" (P4 QS, P1, P2), and §7
     requires Holm < 0.05 and a CI that excludes 0.
   - Raw p-values: P2 0.0168 (+0.87pp, CI [+0.25, +1.54]pp), P4 QS 0.146, P1 1.0, plus P4 MAE 0.0169 (-0.23 EUR/MWh,
     CI excludes 0).
   - Holm over the 3 §9 endpoints puts P2 at 3 × 0.0168 = **0.0503**. With MAE included as a 4th endpoint, the
     adjusted values are P2 0.067 and MAE 0.051. The harness's own-Holm, taken over all competitors, gives 0.134.
   - **Under every reading the pre-registered threshold is missed**, and on DEV the world model did not beat its twin
     (CI straddled 0).
   - Allowed wording: "the rollout model was slightly more accurate than its matched direct twin on hourly direction
     and MAE (raw p ≈ 0.017; not significant after the pre-registered correction); the world-model hypothesis is not
     confirmed".
   - alma_s0 vs alma_direct_s0 shows no difference on any endpoint (all Holm ≥ 0.36).
3. **Calibration.** wm_power's 80% interval covers 75.2% and its 95% interval 91.3%, so it is under-dispersed.
   Most models under-cover in this regime.
4. **Timestral.** It is reported but contaminated.

## Context to show with every number

- **Regime shift.** TEST is entirely 15-minute products, while every training and DEV day was hourly. The mean price
  rose from about 85 to about 170 EUR/MWh.
- **A1 disclosure.** The TEST-month means were seen before the freeze.
- **Selection.** K = 49 runs and 33 recipes; the frozen recipe won by 0.0004 in DEV S (winner's curse).
- **Timestamps.** The mislabelled freeze times (21:05 / 21:08) still need their dated correction.
- **Rehearsal.** The VAL rehearsal predated the freeze.

None of these change the numbers above. They bound how far the numbers generalise.
