# Alma Models: independent audit and improvement experiments (2026-11-01)

Repo: https://github.com/Bhavith-Chandra/Alma-Models (commit 731edfa), cloned to `repo/`. All my code and outputs are in
`audit/`. Nothing was pushed anywhere.

## 1. How it was tested (no faking)

For every project I rebuilt the data from the public sources, ran the **frozen checkpoints** through their own
wrappers, scored them with **my own metric code**, and ran a **leak test** (scramble every value the model must not
know at forecast time; the forecast must not change).

| Project | Data rebuilt vs original sha256 pins | Checkpoint genuine? | Leak test | Reported TEST numbers reproduced? |
|---|---|---|---|---|
| Oil (wm_sv) | FRED files byte-identical | **Yes**: retrained from recipe, weights match to 7e-7 | 0/25 changed | **Exactly** (QS 0.01052 / 0.02383 / 0.04994) |
| Gas (wm_gas) | EIA storage + CPC forecast archive byte-identical (`weekly.csv`, `cpc_fcst.csv` match pins) | **Yes**: refit seed 0 outputs corr 0.99996 with frozen | 0/26 changed | **Exactly** (MAE 11.53/11.56, acc 0.788/0.808) |
| Power (wm_power) | Energy-Charts 2015-2025 byte-identical (2026 file now 1 day longer) | n/a (wrapper sha-checked) | 0/25 changed | Yes (P2 0.746 vs 0.747, P1 0.731 vs 0.732; 1-day window difference) |
| Fin (wm_fin) | All 4,700 Binance zips + `universe.json` + sealed TEST parquets byte-identical | DEV preds re-run: corr 0.99997 | via harness asserts | Yes (P1 0.727/0.728, QS 0.1459/0.1496) |

**Verdict: I found no data leakage and no fabricated numbers in the four headline models.** The checkpoints are
real, trained only on pre-test data, and their TEST scores reproduce.

## 2. What *is* wrong: misleading claims (cherry-picked framing)

1. **Fin "transfers to unseen coins, +8-9 pp, Holm < 1e-4" (README + model card)** is measured only against
   `lgbm_asset` and `har_roll`, which have no weekday input. The repo's own weekday-aware HAR (`har_intra_dow`)
   scores **0.731 on the held-out coins vs wm_fin 0.728**. There is no transfer advantage over a simple baseline.
2. **Oil "wm_sv ranks #1 of 9"** is a noise-level win: skill vs random walk is +0.45% (p 0.48). My own
   EWMA(0.94) random walk scores 0.01054 vs wm_sv 0.01052 at h=1 and beats it at h=5. wm_sv is a random walk.
3. **The "world model" idea is not supported anywhere**: the matched direct twin (no rollout) is equal or better in
   gas (TEST MAE 11.40 vs 11.53), fin (QS 0.1465 vs 0.1471), oil and power. The README does disclose this.
4. Power "#1 numerically on hourly direction" (0.747 vs Chronos-2 0.740) is not significant (disclosed).
5. Older research track (`train*.py`, `runs/leaderboard.md`): the repo's own audit_v3 says none of its 38
   leaderboard numbers is traceable to code + config. Not re-run here (needs the benchmark corpus).

## 3. Improvement experiments (small on DEV/VAL first, then scaled, then one TEST look)

Caveat on all TEST numbers below: the TEST year was already scored by the repo and re-scored by this audit, so it is
a **second look**, not a pristine hold-out. Every weight/choice was fixed on DEV or VAL before the TEST score.

### Gas (biggest win)
- **Small** (existing DEV backtest, 6 rolling folds 2019-07..2025-06): pool rollout+direct seeds and blend with OLS,
  blend weight chosen leave-one-fold-out. DEV MAE 16.59 → 15.72, better in 5/6 folds, fold weights stable 0.2-0.3.
- **Scaled**: trained 8 new DEV-backtest runs (seeds 3-6 of both recipes) → 14-member pool + 0.20 OLS.
  DEV MAE 15.39 (thu) vs frozen recipe 16.59 (−7%). Frozen in `GAS_FREEZE.md`, then refit seeds 3-6 on all
  train weeks.
- **TEST (L48, 52 weeks)**:

| model | MAE thu / fri (Bcf) | QS thu / fri | acc thu / fri |
|---|---|---|---|
| wm_gas (frozen) | 11.53 / 11.56 | 4.59 / 4.56 | 0.788 / 0.808 |
| st_ols_wx | 14.35 / 14.46 | 6.14 / 6.09 | 0.808 / 0.808 |
| **14-seed pool + 0.2 OLS** | **10.63 / 10.46** | **4.30 / 4.23** | 0.808 / 0.788 |

  vs OLS: MAE −3.7 / −4.0 Bcf, CI excludes 0 (the frozen model was **not** significant vs OLS on MAE).
  vs frozen wm_gas: MAE −0.9 / −1.1, p 0.13 / 0.08: better, but not significant on 52 weeks.
  Downside: 80% intervals over-cover (0.94), so they are too wide.

### Fin
- Quantile blend of wm_fin with `har_intra_dow`, weight 0.4 chosen on VAL (2025-07..10) only.
- TEST (24 coins): direction 0.740 vs wm_fin 0.724 (median rule) and har_intra_dow 0.731; QS 0.1459 vs 0.1471.
  vs wm_fin: +1.7 pp, CI [+0.5, +2.9]. vs har_intra_dow: +0.9 pp, CI [−0.6, +2.3] → **not significant** on
  direction, but QS −0.008, CI excludes 0. The result holds for weights 0.3-0.5 (all reported).

### Power
- Quantile blend of wm_power with LEAR, weight 0.5 chosen on VAL.
- TEST: QS 8.60 vs wm_power 9.03 / LEAR 9.22 (both significant); MAE 21.5 vs 22.4; hourly direction 0.755 vs
  0.746 (not significant vs wm_power, significant vs LEAR).

### Oil
- No improvement attempted beyond the check: the signal is indistinguishable from a random walk, so there is nothing
  for an ensemble to add.

## 4. Bottom line
- The models are honest; the **marketing is not always**: the fin transfer claim and oil "#1" are cherry-picked.
- The world-model (rollout) mechanism adds nothing measurable; the gains come from good features and ensembling.
- Cheap, honest wins: **seed ensembles + blending with the strongest simple baseline** improved gas (−8-10% MAE),
  power (−5% QS) and fin (better QS, +1.7 pp direction over wm_fin). None beats its best baseline by a significant
  margin on direction; gas now significantly beats OLS on MAE.

## 5. Follow-up work (same day)

All repo changes are on the local branch `audit-fixes` of `repo/` (not pushed).

1. **Claims corrected** in `README.md`, `winning_models/README.md` and the oil/fin model cards: oil is now described
   as a tie with the random walk; the fin transfer claim now names the weekday HAR that matches it on held-out coins;
   the README notes that the direct twin is equal or better everywhere and points to this audit.
2. **Gas intervals fixed.** Cause: interval width was ~constant (~52 Bcf) whatever the recent error level. New rule
   (chosen leave-one-fold-out on DEV, frozen in `GAS_FREEZE.md` amendment 1): scale the width by
   (recent 8-week |error| / DEV |error|)^0.5. DEV QS 5.97 → 5.92. TEST QS 4.30 → **4.14** (CI excludes 0),
   cov80 0.94 → 0.89/0.90, intervals 17% narrower. Still a little wide on TEST.
3. **Models packaged** with sha-checked wrappers and model cards: `wm_gas_ens` (`tools/gas/wm_gas_ens.py`,
   `winning_models/gas_wm_gas_ens/`), `wm_fin_blend`, `wm_power_blend`. Each reproduces the offline numbers exactly
   through the real harness. `wm_gas_ens` passes a leak test that scrambles the global data store (0/22 changed).
4. **Forward test** (`tools/forward/`): `FREEZE.json` pins 29 code/checkpoint files (2026-11-01 17:47 UTC);
   `forward_eval.py refresh|score` downloads new public data and scores only periods after 2026-11-01, refusing to
   run if a pinned file changed. Dry run on old data works for all three projects. First gas data: 2026-11-05.
5. **Research track** — see section 6.
6. **Reproducibility** (`docs/reproducibility.md`): training is already bit-identical run-to-run on one machine;
   the drift seen vs the original checkpoints is cross-hardware (multithreaded sums). New `ALMA_THREADS=1` switch in
   all trainers gives single-threaded deterministic training (oil, single-threaded, reproduced across machines to
   7e-7). Default behaviour unchanged (verified bit-identical).

## 6. Research track (zero-shot Alma)

- Re-scored the published checkpoints on the bench VAL split (30 tasks): `wm_ssm_none_s0` **0.6706** (claimed
  0.6707), `wm_ssm_direct_s0` **0.6926** (claimed 0.6926). Exact reproduction.
- Overclaim: `docs/almats_research_program.md` says "the SSM family is the first to pass the WM test" from seed 0
  alone, which is also the seed that selected the SSM family and its best seed (0.6707 vs 5-seed mean 0.6804).
  The repo's own audit_sprint flags this; direct twins for seeds 1-4 were never trained.
- **Settled by training the missing seeds** (8 runs, exact stored recipe): published rollout seed scores reproduce
  (0.6804 / 0.6808 / 0.6765 exact; seed 1 0.6947 vs 0.6936 after a resume). Rollout beats direct on 3/5 seeds;
  5-seed mean +0.7%, 95% CI [−0.03%, +1.3%], not significant; **+0.08% without the selection seed**. The SSM does
  not pass the world-model test. Written up in `repo/docs/ssm_wm_test_5seed.md`; the research-program doc now
  carries a correction.

## Files
- `oil_check.py`, `gas_check.py`, `power_check.py`, `fin_check.py` + `fin_score.py`: independent re-scores + leak tests
- `gas_exp.py` (DEV experiments), `run_gas_jobs.py`, `GAS_FREEZE.md`, `gas_final.py` (TEST)
- `fin_blend.py`, `fin_boot.py`, `power_blend.py`; result CSVs alongside
- Side effects in `repo/` (local clone only): rebuilt `data/`, copied checkpoints into `runs/*/search/ckpt/`, new
  gas backtest rows in `runs/gas/search/results.tsv` + preds, new checkpoints `x_wm_s3456.pt`, `x_dir_s3456.pt`,
  `repro_*`, one `repro_sv_none_s0` row in the oil results.tsv.
