# Audit: Alma crude-oil forecasting, pre-TEST (2026-10-31, ~19:15 ET)

Auditor: independent auditor agent with veto. Scope: `docs/commodity_prereg.md` (lead), `tools/commodity/**`
(harness, architect's `wm_*.py`, `search/`), `data/commodity/`, `runs/commodity/**`. Read-only on others' code.
Probe: `tools/audit_commodity_probe.py` (`stats`, `report`, `leak`, `arch`). It writes nothing under `runs/commodity/`.
Peak RSS of the `leak` probe was 805 MB; the other probes used less.
Code was still changing during the audit (`report.py` 19:06, `wm_commodity.py` 19:04, `leakage_runs.py` 19:09,
search rounds r2/r3 running). File:line references are to the working tree at about 19:10.

## Verdict: NO-GO for the TEST run, for now

No leakage was found and no TEST number exists yet. Six blocking fixes (B1 to B6 below) must land first. The two
heaviest are B1 and B4. In B1, one world-model "twin" is a half-trained checkpoint whose hash changed between VAL and
now. In B4, the pre-registered statistics give false positives 2 to 10 times too often at h = 5 and h = 20. Once
B1 to B6 are closed and committed, the auditor re-checks and can give GO without re-auditing everything else.

## Checklist

### 1. Time leakage

| # | item | status | evidence |
|---|---|---|---|
| 1a | Harness training helper cannot emit a target after 2025-06-30 | PASS | `data.py:116-135`: `end > TRAIN_END` raises, and the last target date is asserted. |
| 1b | Harness origin semantics (context strictly before origin; `P[t]` = v[i-1]; target v[i-1+h]) | PASS | `run_eval.py:85` `ctxs = v[i-CTX:i]`; `:194`, `:237-238` (T2 windows: 20 returns ending at obs i-1, and obs i..i+19). |
| 1c | VAL scoring only uses targets ≤ 2025-10-31; DEV scoring only ≤ 2025-06-30 | PASS | `run_eval.py:46-58` (`score_end`), `:192`, `:235`, `:275`. **Probe `leak`**: post-cutoff values were multiplied by U(0.3, 3) or truncated. VAL forecasts and every VAL score (wti and inv; rw_gauss, auto_ets, architect sv checkpoint) stayed bit-identical. The same holds for DEV wti (cut 2025-06-30). |
| 1d | Existing VAL forecasts were made from proper contexts | PASS | `runs/commodity/val/{wti,brent,hh}/rw_gauss.npz` re-derived exactly (`np.allclose`); origins 2025-07-01..2025-10-31 (86/88/87). Every non-search npz under `runs/commodity/` has max origin 2025-10-31. |
| 1e | Architect data: never loads > 2025-06-30 (DEV) or > 2025-10-31 | PASS | `wm_data.py:47-53` (FRED request carries `coed=`, re-cut) and `:32-37` (assert on max date). Every DEV cache file has max date 2025-06-30. No `data/val/` cache exists and `val_results.json` is absent, so the architect has not read VAL. No `wm_*`/`search/*` file reads `data/commodity/` (grep). |
| 1f | Architect training windows end before the backtest | PASS | `wm_core.py:60-71` asserts the last target < cut, and `run_exp.py:70` asserts cut ≤ 2023-01-01. Probe `arch`: last train target is 2022-12-30 for sv, Alma-ft and obs. The `refit*` exemption at `run_exp.py:70` would score a refit model in-sample on the 2023-2025 backtest. No refit rows exist yet; if any appear, they must be excluded from the leaderboard. |
| 1g | Architect models are causal at inference | PASS | Probe `arch`: lp[t+1:] was replaced by N(0, 5) noise, one origin per call. max \|Δq\| = 0 for sv, Alma-ft and obs-direct (10 origins each). Feature scales come from the context only (`wm_core.py:87-104`, `wm_alma.causal_stats`). |
| 1h | Normalisation / vol estimates | PASS | rw_gauss sigma comes from the context's last 250 returns (`models.py:48-64`). Kronos normalises on the context only (`models.py:190-191`). GARCH filter: s2[i-1] uses r[:i-1] (`metrics.py:104-111`, `run_eval.py:151`); checked by index. vol_meanrev and vol_persist use returns ending at obs i-1. |
| 1i | Cross-commodity inputs | PASS (not usable in harness) | Harness models are univariate by API. `wm_commodity.py:69` refuses a multivariate checkpoint, so xc models cannot be frozen. In the search, row t holds same-date closes (Brent/products settle at or before the WTI close) and the ffill is causal (`wm_data.py:72`, limit 10). No future value is used. |
| 1j | EIA release lag | PASS (disclose) | All 2300 inv dates are Fridays (week ending). An origin is the Friday W, and the context ends at I[W-1], released on Wednesday of week W. So it is known at the origin. The h=1 target I[W] is released about 5 days after the origin. Revisions: current vintage (disclosed in prereg §1). |

### 2. TEST untouched

| # | item | status | evidence |
|---|---|---|---|
| 2a | Pre-registration predates any TEST computation | PASS | `d96ee1f` (18:40:49) contains `commodity_prereg.md`. No `runs/commodity/test/` exists, no npz has an origin > 2025-10-31, and grep of search logs/results finds no date > 2025-10-31 except today's timestamps. |
| 2b | Amendment A1 (Kronos 10 samples, ctx 256, every 5th origin) | CONCERN | It is dated "before any score" and justified by wall time, which is plausible: the VAL metrics.json was written 19:05, after A1, and no Kronos npz existed before A1. But A1 is **uncommitted** (`git diff docs/commodity_prereg.md`). It must be committed before TEST (B2). |
| 2c | TEST prices are on disk (`data/commodity/*.csv` to 2026-10-27) and readable by every agent | CONCERN (unverifiable) | This cannot be proven absent. It is mitigated by the fixed model list, RUN.json and the single-run rule. |
| 2d | VAL has been scored (metrics.json/report.md 19:05) | PASS (allowed) | Prereg §2 allows VAL for debugging. The primary Alma (`alma_s0`, TTA off) is fixed in prereg §5, so no swap based on VAL is allowed. |

### 3. Training-data contamination

| # | item | status | evidence |
|---|---|---|---|
| 3a | Alma pre-training data vs oil | PASS | `runs/commodity/leakage_scan.txt`: all 8 checkpoints share train_sha256 `bc78d74b5a3f` and the same 14 datasets. Real-data timestamps end by 2021-09 (ercot), and every other dataset ends ≤ 2016 (monash_weather's 2150 timestamps are synthetic). So none overlaps VAL or TEST. `monash_fred_md` holds monthly WTI (a 22-gram exact run, `leakage_runs.txt`) but is excluded from TRAIN. Other gram hits are isolated (run length 1, coarse lattice). |
| 3b | `docs/commodity_leakage.md` (cited by prereg §2) | FAIL (B5) | The file does not exist. |
| 3c | Competitor release dates vs TEST (from 2025-11-01) | PASS | Chronos-Bolt Nov 2024, TiRex May 2025, Kronos Aug 2025, Chronos-2 Oct 2025: all predate TEST, though Kronos and Chronos-2 may cover part of VAL. Timestral 2026-10-15 is flagged CONTAMINATED (`run_eval.py:40`) and excluded from beats-all (`report.py:112`). Note: `docs/alma_goal.md` asks for "significantly better than Timestral". That claim **cannot** be made on this TEST window. |

### 4. Fairness

| # | item | status | evidence |
|---|---|---|---|
| 4a | Same origins / horizons / quantile grid / context for all | PASS (disclose) | Every model gets the last 1024 obs (`run_eval.py:85`) and is scored on the 9 levels 0.1..0.9. Own crops differ: Alma and Kronos 256, stats 512, other FMs 1024. Kronos runs on every 5th origin (A1). Pairwise tests use common origins (`run_eval.py:226`), but Kronos's own accuracy and CI rest on n/5 and must be labelled so. |
| 4b | Baselines | PASS | Majority = TRAIN up-fraction; momentum = sign(P[t] - P[t-h]); ties go to "down", which is immaterial. Seasonal inv uses lags of 52k weeks, k ≥ 1. GARCH(1,1) is fitted once on TRAIN and filtered causally; `garch_mean_var` matches E[σ²_{t+k}] (`metrics.py:114-118`). Note: on DEV, GARCH parameters are in-sample (DEV ⊂ the fit window), a small advantage to that baseline on DEV only. |
| 4c | Direction and vol rules as pre-registered | PASS | Direction is sign(median - P[t]), with ties to the TRAIN majority (`run_eval.py:212`). Vol rule: mean over k=1..20 of (log q.9 - log q.1)/(2·1.2816·√k), compared with the std (ddof 1) of the last 20 returns (`:255-258`). The quantile floor of 0.01 matches the prereg. |
| 4d | Pre-registered WM test pairs (alma_sK vs alma_direct_sK) | **FAIL (B1)** | `data/ckpt/wm_ssm_direct_s3.pt` has `step=565, complete=False` (budget 2500), while `wm_ssm_none_s3.pt` has 2500 steps. No training process is running. Its sha256 also **changed**: the VAL npz meta records `ef71c492…` (18:51) and the file now hashes `ef61c00c…` (mtime 18:53). The s3 pair in `runs/commodity/val/report.md` therefore compares a full model with a partial one, and the TEST run would load whatever file is present. `models.py:277-283` registers any checkpoint that exists, without checking `complete`. The s0 pair is matched: same cfg, 2500 steps each, env differs only in `TSAR_WM_ABL`, 365,937 params each. |
| 4e | Architect WM twin matching | FAIL (B6) | (i) The budget is wall-clock (5 min) on a shared CPU, so step counts differ by up to 2.4x: sv 286 vs 679, obs 1290 vs 720, rs 464 vs 679, obs+xc 2487 vs 1082. (ii) The Alma-ft "direct" twin is initialised from the **rollout** pretrained weights (`wm_alma.py:102,112`, `init=CKPT0` for both abl), not from `wm_ssm_direct_s0.pt`, so it is not a matched twin. (iii) `r1_alma_zs_direct` is the rollout weights run as direct and is not a twin. It must never be quoted as a WM test. |
| 4f | Architect candidate in harness | PASS (design) | It enters through `FROZEN` in `wm_commodity.py:25` (path + sha, verified at load) and `runs/commodity/FROZEN.json`. Both are still empty or absent, as expected before freezing. |

### 5. Selection

| # | item | status | evidence |
|---|---|---|---|
| 5a | Selection used DEV only | PASS so far | No VAL cache and no `val_results.json`; `run_val.py:9` refuses a second VAL run. |
| 5b | Number of configurations | CONCERN (state it) | As of 19:15, `results.tsv` has 23 rows: 5 random-walk references, 1 smoke run, and 17 trained or zero-shot configurations, with rounds r2/r3 still queued. Weekly scores of the non-broken configurations span 0.730 to 0.782. The single-model 90% block-bootstrap half-width is about ±0.035, and RW = 0.750 sits inside every model's interval. The best DEV rows (Alma zero-shot, 0.730/0.731) are not distinguishable from RW. Any TEST claim must cite the count K and the winner's curse. |
| 5c | Search protocol timing | CONCERN (minor) | `docs/commodity_search.md` §0 was written at 18:45, after the reference, Alma zero-shot and obs rows. The noise rules were added after seeing those rows. This is DEV-only, so it does not block. |
| 5d | VAL used once | CONCERN (B6) | Two VAL procedures exist: the architect's `search/run_val.py` (search metric, its own FRED VAL cache) and the lead's `run_eval.py --split val` (prereg metrics). Before either is read for the candidate, it must be declared which one decides go/no-go and by what rule. |

### 6. Statistics

Toy checks (probe `stats`):
- Wilson(60, 100) = (0.50200, 0.69060), identical to statsmodels.
- Holm is identical to statsmodels `multipletests` and keeps NaN.
- DM-HLN at h=1 equals the scipy one-sample t (1.69166).
- PT has size 0.058 under the iid null (n = 200).
- The PT and HLN formulas match Pesaran-Timmermann (1992) and Harvey-Leybourne-Newbold (1997).
- `report.beats_all` once read a metric of exactly 0 (the rw_gauss skill) as -9 through `x or -9`, which could list RW as "beaten" by a worse model. This was fixed in the working tree at 19:06 (`report.py:115-116`, `nz()`), and probe `report` confirms the fix.

Overlap Monte-Carlo (probe `stats`): a driftless random walk with daily origins and n = 230, about TEST size. Two zero-skill persistent predictors (momentum-h and momentum-60), 600 reps, nominal 5%:

| h | binomial test false-positive | PT false-positive | Wilson coverage of 0.5 | DM-HLN (Bartlett, lag h-1, as implemented) false-positive | circular block bootstrap (block 2h) coverage |
|---|---|---|---|---|---|
| 1 | 0.038 | 0.048 | 0.942 | 0.047 | 0.940 |
| 5 | 0.177 | 0.112 | 0.790 | 0.100 | 0.912 |
| 20 | **0.498** | 0.190 | **0.477** | 0.095 | 0.862 |

| # | item | status | evidence |
|---|---|---|---|
| 6a | Implementations correct | PASS | Toy checks above; `metrics.py:6-75`. |
| 6b | Accuracy CI / binomial / PT under overlapping horizons | **FAIL (B4)** | `run_eval.py:172-177` uses the iid Wilson, binomial and PT. At h=20 the "95%" CI covers the truth only 48% of the time, and the binomial test flags a zero-skill model half the time. `acc_nonoverlap` is printed without any CI or test. T2 is worse: each label uses a 40-day span. |
| 6c | DM-HLN HAC bandwidth | **FAIL (B4)** | A Bartlett kernel truncated at h-1 down-weights the MA(h-1) autocovariances (weight 0.05 at lag 19). Persistent predictors add dependence beyond h-1. Result: false-positive rate about 2x nominal at h=5 and h=20. T2 uses lag 19 (`run_eval.py:269`), but its loss is dependent over about 39 lags. This matters in practice: VAL `report.md` already shows `alma_s0` at h=5 "beating 5 competitors (Holm p<0.05)" on 81 overlapping origins. |
| 6d | Holm families | PASS | `run_eval.py:319-342` (model × baseline × h within each series/target) and `report.py:112-114` (all non-contaminated competitors per (T, h)), as in prereg §6-7. |
| 6e | Determinism of sample-based architect models | CONCERN | `wm_commodity.make_predict` seeds once per batch, so an origin's quantiles depend on which other origins share its batch. The probe saw max \|Δq\| = 0.034 for sv when the batch composition changed. The fix is to seed per origin. |

## Required fixes before GO (blocking)

- **B1 (WM pairs).** Do one of two things for `wm_ssm_direct_s3`. Either finish it to 2500 steps from a committed code
  state and record its sha256, or drop the s3 pair by a dated prereg amendment. In both cases, rebuild the VAL s3
  forecasts with the final file and do not quote the current VAL s3 numbers. Make `models._alma` refuse any
  checkpoint with `complete != True` or `step != budget["steps"]`. Pin the sha256 of every Alma checkpoint used at
  TEST in `FROZEN.json`, and have the TEST run verify them. Disclose that the direct_s3 checkpoint was trained under
  a changing working tree (`docs/audit_sprint.md:156`).
- **B2 (provenance).** Commit `tools/commodity/**`, including the untracked `run_eval.py`, `metrics.py`, `report.py`,
  `leakage_*.py`, `wm_*.py` and `search/*`, plus the modified `models.py` and prereg amendment A1, before TEST. RUN.json
  must show `tools_commodity_dirty == ""`. Because `runs/` is git-ignored, force-add `FROZEN.json` and `test/RUN.json`
  right after the run, along with a sha256 list of every `test/*/*.npz` and `test/metrics.json`.
- **B3 (TEST guard holes).**
  (a) `forecast()` reuses any cached npz whose origins match (`run_eval.py:75-79`). In TEST mode, abort if
  `runs/commodity/test/` holds any npz before RUN.json is written.
  (b) An unknown model is skipped with a print (`run_eval.py:80-84`), and a broken architect module is skipped silently
  (`models.py:289-294`). In TEST mode, fail before writing RUN.json unless every model in `PREREG_MODELS` + `FROZEN`
  resolves, except a declared list of absent ones (for example `alma_direct_s4`).
  (c) A crash after RUN.json is written burns TEST on an 8 GB shared machine. Rehearse the exact TEST command on VAL
  (same list, fresh output directory) before the real run.
- **B4 (statistics, dated amendment before TEST).** For T1 at h = 5 and 20, and for T2:
  - Replace or supplement the iid Wilson CI, binomial test and PT with a moving or circular block bootstrap of the
    hit series (block ≥ 2h; 40 for T2) or a HAC-based CI.
  - Add the exact binomial and Wilson CI on the non-overlapping subsample.
  - Use a DM bandwidth that controls size: rectangular kernel to h-1, or Bartlett/fixed-b with bandwidth ≥ 2h, or a
    block bootstrap of the loss differential. Use about 39 lags for T2.
  - State in advance that h=20 and T2 on TEST rest on about 11 to 12 independent windows and are descriptive unless
    the overlap-robust test passes.
  - The headline "beats" claim may only use the overlap-robust p-values.
- **B5 (contamination record).** Write `docs/commodity_leakage.md` from `leakage_scan.txt` and `leakage_runs.txt`
  (including the dominick/weather run lengths), as prereg §2 promises. Include the competitor cut-off table and the
  Timestral exclusion.
- **B6 (architect freeze).**
  - Retrain the finalist and its direct twin with identical `--max_steps`, seeds and data, instead of wall-clock.
  - For an Alma-ft recipe, initialise the twin from `wm_ssm_direct_s0.pt`.
  - Write `FROZEN.json` with the members' sha256 and the 3-seed DEV evidence, including the count K of configurations
    tried.
  - Declare, before reading VAL, which VAL procedure and rule decides inclusion.
  - Keep `refit*` rows out of the DEV leaderboard.

## Non-blocking (disclose in the write-up)

GARCH parameters are in-sample on DEV. Kronos is scored on every 5th origin. Effective context crops differ by model.
FRED spot data and EIA stocks are current vintage, not real-time. h=1 for inventories is the next unreleased print,
about 5 days after the origin. Timestral is contaminated, and Chronos-2 and Kronos may cover part of VAL. The DEV
winner's curse applies, with K configurations tried. The architect's search metric (weekly 2023-2025, its own score)
differs from the prereg metrics, so the TEST report must use the prereg metrics only.

## For the TEST-results audit (next round)

The auditor will check the following:
- RUN.json: commit, clean tree, frozen shas, model list.
- That no `test/` npz predates RUN.json.
- Re-derivation of the headline accuracies, QS skills and DM p-values from the npz files with an independent script.
- That every target and horizon is reported, wins and losses alike.
- That the overlap-robust CIs from B4 are the ones quoted.
- Seed consistency of the WM test.

---

# Re-audit for TEST launch (2026-10-31, ~19:55 ET): **GO**, on one pre-run condition

What I checked: lead commits 290d46e, ec531f0 and 21b90eb (amendments A1-A4, `docs/commodity_leakage.md`, strict TEST
mode, `ALMA_PINS`, the VAL rehearsal) and architect commits faa258f and 2ee0086 (freeze, `FROZEN.json`,
`val_results.json`). Probe: `tools/audit_commodity_probe.py stats2` (new) and a sandbox copy of the harness in the
auditor's scratchpad (a git repo holding a copy of `tools/commodity`, `data/commodity`, the prereg and `FROZEN.json`),
used to try to break the guards. Nothing was written to the real `runs/commodity/`.

**Pre-run condition (one line).** `runs/commodity/FROZEN.json` is modified and uncommitted: the lead added the
`alma_sha256` block after faa258f. The strict dirty-tree check covers only `tools/commodity` and the prereg, not
`FROZEN.json`. RUN.json does record the file's content and sha256, so nothing would be lost, but commit `FROZEN.json`
before the run so the committed and recorded versions agree. All its shas already match the files on disk (below).

## B1-B6 status

| item | status | evidence |
|---|---|---|
| B1 Alma WM pairs | RESOLVED | A2 keeps only the s0 pair. The s3 pair is void: `runs/commodity/val/wti` has no `*direct_s3*`, and `alma_direct_s3` is not in `PREREG_MODELS`. `models._alma` refuses incomplete checkpoints; tested on `wm_ssm_direct_s3.pt`, it gives "incomplete checkpoint (step=565, complete=False …); refused". All 6 pinned Alma shas (`run_eval.ALMA_PINS` = `FROZEN.json["alma_sha256"]`) equal `shasum -a 256` of the files. The s0 pair was already shown matched (same cfg, 2500 steps each, env differs only in `TSAR_WM_ABL`). |
| B2 provenance | RESOLVED (+ condition) | `tools/commodity/**`, the prereg (A1-A4) and `docs/commodity_leakage.md` are committed, and `git status` is clean for both strict paths. The rehearsal RUN.json records commit ec531f0 with `tools_commodity_dirty == ""`. Only open item: the `FROZEN.json` commit above. |
| B3 TEST guards | RESOLVED | Sandbox, each case refused **before** RUN.json is written or `test/` is created: an npz present in `test/` ("already holds forecast files"); `--models`; a modified `metrics.py`; a modified prereg; an untracked `tools/commodity/wm_evil.py`; an Alma sha in `FROZEN.json` that differs from the pin; `FROZEN.json` missing; RUN.json already present; `--metrics-only` with no run; an architect module that fails to import (preflight `LOAD_ERRORS`); a pre-registered model missing from the registry (preflight `KeyError: 'alma_direct_s0'`). Inside `forecast()`, strict mode refuses any existing destination file and re-raises load errors (`run_eval.py` diff ec531f0). Rehearsal `runs/commodity/rehearsal_val/RUN.json`: status complete, 18 models (A2 list + `wm_sv`, `wm_sv_direct`), 19:35-19:40, EXIT 0. |
| B4 statistics | RESOLVED | A3 makes the non-overlap subsample the basis of inference (exact binomial, Wilson CI, PT; exact McNemar for model-vs-model at h > 1 and T2; DM-HLN lag 0 on the subsample for T4 at h > 1). The block bootstrap gives the full-sample CI. `report.py` and `beats_all` use only the primary `p`. Size, re-simulated by calling the new `run_eval.dir_stats` / `pair_test` directly (table below), holds at or below nominal for every primary test. |
| B5 leakage record | RESOLVED | `docs/commodity_leakage.md` (21b90eb) matches §3 above: TRAIN data ends by 2021-09; `monash_fred_md` is excluded; dominick runs reach at most 3 and are coincidental; competitor cut-offs are listed; Timestral is contaminated. |
| B6 architect freeze | RESOLVED | **Twin matched:** the 6 checkpoints `f_sv_{none,direct}_s{0,1,2}.pt` are identical in arch sv, L 128, cols [0], kw {}, lr 2e-3, batch 64, cut 2023-01-01, `max_steps` = steps = 600, and 25,191 params each. They differ only in `abl` and are paired by seed. **Shas:** all 6 files equal the hashes in `FROZEN.json` and in `wm_commodity.FROZEN` (committed), and they are verified at load. **Timing:** frozen in faa258f at 19:28:13; harness VAL npz for `wm_sv`/`wm_sv_direct` were written at 19:30:51/55, after the freeze. **VAL procedure:** the declaration is in the faa258f version of `docs/commodity_search.md:102` (before VAL): the lead's harness `run_eval.py --split val` is authoritative; `search/run_val.py` is a secondary read; neither changes the checkpoints. **K:** 34 search rows + 5 references + 6 final retrains, with the winner's curse stated in `FROZEN.json`. **Determinism:** the `wm_sv` ensemble is batch-independent (4 origins: batched, one at a time and reversed order are bit-identical). |

## A3 size (probe `stats2`, driftless RW, constant vol, n = 230 daily origins, 500 reps; nominal 5% / 95%)

The predictors carry zero skill and the pairs have equal expected loss:
- T1: momentum-h vs momentum-60.
- T2: two persistent random sign predictors.
- T4: two Gaussian forecasters from disjoint, equal-length windows.

| target | subsample binomial | subsample PT | subsample Wilson coverage | block-bootstrap CI coverage | primary pair test | supplementary DM (Bartlett 2h) |
|---|---|---|---|---|---|---|
| T1 h=1 | 0.050 | 0.038 | 0.936 | 0.948 | 0.060 (DM-HLN) | 0.056 |
| T1 h=5 | 0.032 | 0.042 | 0.934 | 0.930 | 0.020 (McNemar) | 0.046 |
| T1 h=20 | 0.020 | 0.036 | 0.980 | **0.872** | 0.000 (McNemar) | **0.148** |
| T2 | 0.036 | – | – | – | 0.000 (McNemar) | 0.036 |
| T4 h=5 | – | – | – | – | 0.050 (DM lag 0, subsample) | 0.058 |
| T4 h=20 | – | – | – | – | 0.054 (DM lag 0, subsample) | 0.092 |

Monte-Carlo SE is about 0.01. The primary tests hold size. Two limits apply when the TEST results are written up:
1. The supplementary wide-lag DM over-rejects at h=20 (0.148). It must never be quoted as evidence.
2. The full-sample block-bootstrap "95%" CI covers about 87% at h=20. Label it descriptive; the subsample Wilson CI is
   the inferential one.

The McNemar tests at h=20 and T2 are very conservative (about 12 and 6 independent windows). That is why A3 declares
those targets descriptive in advance.

## Other checks

- **No TEST file exists.** `runs/commodity/test/` is absent, and every forecast file under `runs/commodity/` (val,
  rehearsal_val, rehearsal_val_prefreeze) has origins ≤ 2025-10-31.
- **VAL is not encouraging, and that is fine.** It is recorded honestly in `docs/commodity_search.md` §5. `wm_sv`
  T1 accuracy is 0.465 / 0.494 / 0.149 at h = 1 / 5 / 20. Its median is "up" at every h=20 origin (a learned drift),
  and it is within noise of the random walk on DEV. The candidate goes into TEST as pre-registered; no further change
  is allowed.

## Residual risks (non-blocking)

1. **Lazy model loads.** Chronos, TiRex, Timestral and Kronos load lazily inside `forecast()`, so a network or HF Hub
   failure would happen after RUN.json is written and would consume the TEST. The rehearsal shows the local cache
   works. Run the TEST with `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) so it uses only the cache that the
   rehearsal proved.
2. **Memory.** The rehearsal took about 5 min for about 86 origins per series. TEST has about 3x the origins
   (Kronos about 50). Avoid running it alongside other memory-heavy jobs.
3. **Write-up wording for `wm_sv`.** It was trained on WTI only and is also run on brent/hh/inv; treat those results
   as descriptive. Do not score the frozen search models on harness DEV origins before 2023 (in-sample).

## Verdict

**GO** for `run_eval.py --split test`, once `runs/commodity/FROZEN.json` is committed. Recommended: run it with
offline HF. Immediately after the run, force-add `test/RUN.json`, `test/hashes.json` and `test/metrics.json`. The
TEST-results audit will then re-derive the headline numbers independently from the npz files.

---

# TEST results audit, round 1 (2026-10-31, ~20:15 ET)

Inputs: `runs/commodity/test/{RUN.json, hashes.json, metrics.json, report_wm_sv.md, report_alma_s0.md}` and the
TEST forecast npz files (committed in 45be6d4; the npz files themselves are git-ignored and pinned by `hashes.json`).
The independent re-derivation is `tools/audit_commodity_test.py`. It reads only `data/commodity/*.csv` and the npz
files and re-implements every statistic with numpy/scipy; it imports neither `run_eval.py` nor `metrics.py`.

## Integrity: PASS

- **One complete run.** RUN.json: status complete, started 19:46:01, finished 20:00:57, commit 97be9e5,
  `tools_commodity_dirty == ""`, argv `run_eval.py --split test`, the 18 models of A2 plus `wm_sv`/`wm_sv_direct`.
  97be9e5 differs from the audited 9eaab83 only in `FROZEN.json`, which was the pre-run condition.
- **No file predates the run.** Every one of the 70 TEST npz has `meta.created >= started` and `split == "test"`;
  the Alma files were written from 19:46:02.
- **Hashes match.** `hashes.json` lists exactly the npz on disk plus metrics.json; all 70 sha256 match; its own
  sha256 equals `RUN.json.hashes_sha256`.
- **Frozen file matches.** `FROZEN.json` sha256 equals `RUN.json.frozen_sha256`. The Alma checkpoint shas in the
  npz metas equal the pins.
- **The log is clean.** `test_run.log` contains no error, failure or skip. The only warnings are the expected
  "TTA differs from training" notes for the `*_tta` models.

## Re-derivation: 157 of 158 numbers match exactly

What was re-derived on WTI:
- T1 at h = 1/5/20: accuracy, non-overlap accuracy, binomial p and Wilson lower bound.
- T2 accuracy.
- T4 at h = 1/5/20: skill vs RW and the primary p.
- Every world-model pair test.
- wm_sv vs rw_gauss and vs chronos2, and alma_s0 vs rw_gauss and vs chronos2.

Models: wm_sv, wm_sv_direct, alma_s0, alma_direct_s0, chronos2, tirex, rw_gauss and garch. All agree to 1e-9 except
GARCH T2 accuracy: 0.7357 here vs 0.7445 in the harness. That gap comes from my deliberately different optimiser
(Powell vs Nelder-Mead), which flips 2 of 227 labels; garch is the best T2 model under both. I also re-derived the
"beaten (Holm)" claims for wm_sv at T4 h=1 and h=5 with statsmodels Holm, and they match.

## Coverage and test choice: PASS, with two labelling notes

- **Every target and horizon is reported, wins and losses alike.** That covers wti/brent/hh T1 (h = 1/5/20), T2, T4
  (h = 1/5/20) and inv T3 (h = 1/4).
- **The p-values quoted are the A3 primary tests.** Those are DM-HLN at h=1; exact McNemar on the non-overlapping
  subsample for direction at h > 1 and for T2; DM lag 0 on the subsample for T4 at h > 1. The `beats_all` and
  world-model tables use only `p`; the supplementary wide-lag DM is not used for any claim.
- **Note 1.** The world-model table header says "DM p" even where the test is McNemar. Fix the label in any external
  copy.
- **Note 2.** `beats_all` counts `wm_sv_direct` as a competitor of `wm_sv`. This is harmless, but the twin is not an
  external model.

Effective sample sizes: TEST WTI has 246 daily origins. The non-overlapping subsample has 49 origins at h=5,
12 at h=20 and 6 for T2. As pre-declared, h=20 and T2 are descriptive.

## What can and cannot be claimed (WTI unless stated; p = primary, Holm as in prereg §6-7)

**CAN be claimed**

1. **wm_sv probabilistic forecast at h = 1 (T4).** Its skill vs the random walk is +0.45%, rank 1 of 9 among the
   non-contaminated models. It is significantly better (Holm < 0.05) than auto_arima (Holm p 0.001),
   chronos_bolt_base (0.0001) and kronos_small (0.002). Kronos is scored on 50 origins with reduced settings
   (amendment A1), and that must be stated alongside the result.
2. **wm_sv at h = 5 (T4).** It is significantly better than kronos_small only (Holm 0.0009, 49 origins).
3. **The TEST was clean.** It was run once, pre-registered, sha-pinned and audited. Losses and wins are reported
   alike.

**CANNOT be claimed**

1. **That wm_sv beats all models, or beats the random walk.** At T4 h=1 wm_sv vs rw_gauss gives p = 0.48 and
   wm_sv vs chronos2 gives p = 0.039 unadjusted but Holm 0.19 (tirex: 0.045, Holm 0.19). At h = 5 and 20 its skill
   vs RW is negative (-0.76%, -0.53%).
2. **Any directional accuracy edge, for any model.** No model has a non-overlap binomial p < 0.05 or a significant
   win over a baseline at any horizon. wm_sv's accuracy is 0.492 / 0.471 / 0.590 at h = 1/5/20. The random walk's
   tie rule (TRAIN majority, which is up) gives 0.549 / 0.554 / 0.639, and the up-majority baseline matches it.
   wm_sv vs rw_gauss at h=1 is -5.7pp (p 0.039, worse; not significant after Holm).
3. **That world models help on oil (the world-model hypothesis).**
   - wm_sv vs wm_sv_direct shows no significant difference on any endpoint after Holm. The one unadjusted signal
     favours the direct twin: T1 h=1, -4.9pp for the world model, p 0.034, Holm 0.30.
   - alma_s0 vs alma_direct_s0 shows no significant difference either. Its T4 h=1 rollout is slightly worse
     (p 0.074, Holm 0.59).
   - The DEV and VAL pattern of "WM marginally better than its twin" did not replicate.
4. **Volatility direction (T2).** wm_sv scores 0.612, below the 0.621 base rate. GARCH is best at 0.744 (0.736 in
   my re-derivation) and chronos2 scores 0.634. All of this rests on 6 independent windows and is descriptive.
5. **Zero-shot Alma (alma_s0) is significantly worse than the random walk at T4 h = 1.** Its skill is -11.4%,
   p = 0.0001; at h = 5 it is -6.2% (p 0.044, worse). Its VAL direction numbers (0.628 / 0.667) were noise; on TEST
   it scores 0.476 / 0.471.
6. **Anything about Timestral.** It is contaminated and excluded.
7. **Secondary series (brent, hh, inv).** They are reported in the report files but were not independently
   re-derived; they are secondary families and must be presented as such.

**Bottom line for investors.** On a clean, pre-registered year of out-of-sample crude-oil data, no model was
reliably better than a random walk. That includes Alma, the oil-specific world model, Chronos-2 and TiRex. The one
defensible positive is narrow: wm_sv's 1-day probabilistic forecast ranks first and significantly beats three
weaker models, but it ties the random walk and Chronos-2.

## Round-2 search (tools/commodity/r2/)

- **Code check: PASS.** No file in `tools/commodity/r2/` references `runs/commodity/test`, `data/commodity` or any
  TEST date. `data_r2.py` requests FRED with `coed=` set to 2025-06-30 for dev and 2025-10-31 for val, asserts max
  date ≤ 2025-10-31, and lists WTI, Brent and Henry Hub as FORBIDDEN targets. Those three series are used only as
  input covariates capped at 2025-10-31. `test_r2.py` does not exist yet.
- **Rule.** Round 2 must never read `runs/commodity/test/`, the round-1 reports, or this TEST section.
- **Disclosure.** Round 2's TEST window (refined products, from 2025-11) overlaps round 1's in calendar time, and
  products move with crude. The round-1 TEST outcome, including the regime (WTI up and more volatile), is now public
  in the repo. Round 2's TEST is therefore not blind to the period. Disclose this, and if possible run round 2 with
  an agent that has not read these files.
