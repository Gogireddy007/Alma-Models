# Audit: Alma-Fin (crypto perpetuals and ETFs), pre-TEST (2026-11-01)

Auditor: independent auditor with veto, held to the same standard as the oil, power and gas audits.

Scope:
- Prereg: `docs/fin_prereg.md` (36d2232), with A1-A5 (A5 0525126).
- Freeze of record: `runs/fin/FROZEN.json` at f2bbdbc (07:46:20 UTC); bcdbf2e at 07:46:11 differs only in its
  timestamp string.
- Red-team log: `docs/fin_redteam.md`.
- VAL: cd732b7.
- Docs: `docs/fin_search.md`, `docs/fin_results.md`, `docs/fin_leakage.md`.

Checked at HEAD cd732b7. `git status` is clean for `tools/fin`, `docs/fin_prereg.md` and `runs/fin/FROZEN.json`, and
also for the statistics dependencies `tools/power/metrics.py` and `tools/commodity/metrics.py`.

Checks I ran myself:
- **VAL re-derivation:** `tools/audit_fin_val.py`, which imports nothing from `tools/fin`.
- **Strict-mode preflight:** `check_pins()` and `preflight()` on the exact 32-model TEST list.
- **Guard break-tests:** a sandbox git repo holding a copy of `tools/fin`, the pinned data, the prereg and
  `FROZEN.json`, with no checkpoints, so it cannot forecast anything.
- **Checkpoints:** sha256 and parameter counts.
- **Timestamps:** git and file times.

All times below are UTC. DST ended at 06:00 UTC: git dates switch from -0400 to -0500 in the middle of the run.

## Verdict: **GO** for one offline TEST run

There are no blockers.

**Command (run exactly this, once, from the repo root, with no other heavy jobs on the machine):**

```
cd /Users/srimanarayana/Timeseries && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 10 .venv/bin/python tools/fin/run_eval.py --split test --i-have-auditor-go
```

`--market` defaults to `all`. Do not pass `--models`, `--force` or `--market`; strict mode refuses them.

**Afterwards:**
- Force-add and commit `runs/fin/test/RUN.json`, `hashes.json`, `metrics.json` and `report.md`.
- `FROZEN.json` is already committed.
- Commit or discard the unrelated modified `runs/fin/val/report.md` before or after the run. It is outside the strict
  check and does not affect it.

## 1. Freeze integrity

| item | status | evidence |
|---|---|---|
| Freeze precedes every VAL output of the frozen models | PASS | f2bbdbc was committed at 07:46:20 UTC (02:46:20 -0500). The red team's freeze check (f644abb, 07:48) used DEV days only: 2024-08-01 .. 2025-06-30 (`rt_frozen_models.txt`). The rehearsal ran 07:49:21-08:27:15 (`rehearsal_val/RUN.json`, commit f644abb, clean tree, the full 32-model list); per the docs its metrics were not read. The VAL npz for the frozen models were written 08:30:31-08:32:02, VAL metrics at 08:33:06, and the commit cd732b7 at 08:34:10. Every result row has fold ∈ {E1, E2, E3, EM, F1, refit}; none is VAL. Note: `docs/fin_search.md` §4 gives the rehearsal and VAL as "~08:50" and "~08:55 UTC", but the file times show 08:27 and 08:32. That is a mislabel from the DST switch with no effect on order (correct it with a dated note). |
| bcdbf2e vs f2bbdbc | PASS | The diff is only the `timestamps` string in FROZEN.json and `freeze_spec.py`. Both commits precede VAL. |
| sha pins | PASS | All 6 checkpoints: `shasum` = `FROZEN.json["sha256"]` = the pins in the wrappers (`wm_fin.py:27-28`, `agents/{wm_agent,ppo,dqn}.py:16`, `wm_b_fin.py:32-33`). The VAL npz metas carry the same shas, plus `wm_ckpt_sha256` for the agent. The exact-list preflight passes, and it enforces A1 RT-6 (meta ckpt = FROZEN.json = file). |
| K = 985 | PASS (conservative) | `results.tsv` has 116 rows, minus 7 `ref`, giving 109. `results_b.tsv` has 923 rows, minus 47 `ref`, giving 876 (it still counts 20 `ref_*` rows and 2 annotations). K therefore counts **rows** (configuration × fold), not distinct configurations, and includes some reference and annotation rows. That overstates the number of trials and makes the DSR more conservative. Report it as "985 evaluated rows". |
| var_sharpe_trials | PASS | It equals the sample variance (ddof 1) of the 74 listed per-fold agent rows: recomputed 0.0021806, stored 0.0021806. |
| claim_models and pillar tests pre-declared | PASS | FROZEN.json at f2bbdbc lists `claim_models` [wm_fin, wm_fin_cl, wm_agent] and `pillar_tests_preregistered`: rollout vs direct, continual vs frozen, WM agent vs PPO/DQN, portability vs lgbm_asset/har_roll, and beats-baselines including the A5 weekday-aware baselines. These match prereg §7. The `val_go_no_go` rule is in the same file. |
| Twin matching | PASS | wm_fin and wm_direct: 64,285 params per member, seeds 0-2, same recipe, `abl` none/direct. wm_agent, ppo: actor 5,378 + critic 5,377, 500 steps, seeds 0-2. dqn: Q 11,269 params (+4.8% vs 10,755; inside ±5%). wm_fin_cl and wm_fin_cl:frozen: the same weights with observe on or off. |
| go/no-go applied correctly | PASS | The rule: NO-GO only on a crash, > 5% non-finite outputs, or a failed leak test. VAL completed; the non-finite fraction is 0.0000 for every frozen model in both markets (my check); the leak tests in FROZEN.json are OK (the 2 dqn sensitivity "FAIL"s are a discrete argmax not moving, not a leak). Hence GO, and nothing was changed after VAL. |

## 2. VAL re-derivation (cd732b7): 144 of 144 numbers match, 0 differences

From the open `crypto_daily.parquet` I rebuilt:
- labels `y(D) = 1[log RV(D) > mean log RV(D-7..D-1)]`, with ties and NaN excluded;
- the prediction rule (p_up > 0.5, else median > ref; a missing prediction counts as the DEV majority "down");
- the groups from `universe.json`.

What matches to 1e-9:
- **Forecasters (all of them):** P1 accuracy for all / held-out, and P3 QS (all coins).
- **Daily tests:** the day-level DM vs 0.5.
- **Pairwise:** the P1 pairwise mean differences and DM p for wm_fin vs lgbm_dow, har_intra_dow, har_dow, lgbm, har,
  tirex, chronos2, wm_direct and wm_fin_cl.
- **Agents:** P2 net Sharpe for every agent (held-out and all), from my own fill-to-fill P&L with funding and
  7 / 10 bp costs, marked at the 2025-10-31 close. `last_bar` is false for all coins on that day, so no closing cost
  is charged.

VAL headline (no claims, 123 days):
- **P1, all coins:** wm_fin 0.737, lgbm_dow 0.736, wm_direct 0.735, wm_b_moe4 0.731, har_intra_dow 0.729,
  har_dow 0.714, tirex 0.704, chronos2 0.699, har 0.662.
- **P1 vs the best baseline:** wm_fin vs lgbm_dow +0.14 pp, p 0.92. **On VAL the candidate is level with the best
  weekday-aware baseline.**
- **P2, held-out coins:** wm_agent Sharpe -2.39, ppo -2.23, dqn -0.13, buy_hold +0.57.

## 3. TEST readiness

| item | status | evidence |
|---|---|---|
| Strict-mode refusals | PASS (sandbox, cannot forecast) | Each of these was refused before RUN.json was written: no `--i-have-auditor-go`; FROZEN.json missing; FROZEN.json untracked; `--market crypto`; FROZEN.json edited after commit (it is in the dirty check here, unlike in gas); modified `tools/fin/metrics.py`; untracked `tools/fin/wm_evil.py`; modified prereg; data pin mismatch (`universe.json`); a cached npz in `test/`; RUN.json already present. In the clean case, preflight aborted with no checkpoints and no `test/` directory was created. HF offline is checked before RUN.json (`run_eval.py:466`). |
| Real repo preflight | PASS | `check_pins()` is OK on all 7 pins, including the 3 SEALED files. `preflight()` on the 32 models (23 section-5 + 9 frozen) is OK, with no import errors. |
| SEALED files not loaded outside TEST | PASS | The only reader is `data.load(..., allow_test=True)` (`data.py:233-236`). It raises `PermissionError` unless `ALMA_FIN_TEST_GO=1`, which only `run_eval.py --split test --i-have-auditor-go` sets (`run_eval.py:436`; verified: a direct call is refused). No other file in `tools/` references `_SEALED` except `check_pins` (hashing, not parsing) and `build`. The red-team code never uses `allow_test=True`. Access times are not conclusive on this filesystem: they record the first read after a write, here the build-time pin hashing at 23:39 on 10-31. |
| TEST window | PASS (by code) | `TEST_START` 2025-11-01 and `TEST_END` 2026-10-30 (`data.py:50`) cover P1 and P3. `P2_LAST["test"] = "2026-09-29"` (`run_eval.py:60`). The crypto panel uses a complete UTC calendar (`harness.py:104-105`), so the missing 2026-10-28 archive day is a NaN row: its labels are NaN and the 7 days whose reference window contains it are excluded by the generic rule (prereg §1). This was verified by reading the code, not the SEALED data. |
| Statistics dependency | NOTE | `tools/fin/metrics.py` loads `tools/power/metrics.py`, which loads `tools/commodity/metrics.py`. Those two files are outside the strict dirty check. Both are committed and clean now, and RUN.json records the commit. Keep them untouched until the run is done. |

## 4. Claim-validity flags (must appear in the TEST report)

1. **The weekday knob was added after the fact (Am2, 06:15 UTC).** It came after the red team found the weekday
   effect (RT-15), and the extension rows are in K. The P1 edge over non-calendar baselines is mostly a weekday
   effect; the only meaningful P1 comparison is against `lgbm_dow`, `har_intra_dow` and `har_dow` (A5).
2. **The A5 / Am2 gate was set with the competing EM values already known.** The gate is "beat the maximum
   registered weekday-aware baseline on E1-E3", and lgbm_dow's 0.6930 and moe4's 0.6934 were known when it was
   written. wm_fin passes (0.7077); moe4 ties and is a non-claim model. The gate does not prove robustness.
3. **F1 is post-selection.** Each chosen recipe is the maximum over up to 6 F1 looks (RT-11). F1 numbers and F1 pillar
   differences are descriptive and carry a winner's curse. K = 985 rows.
4. **The universe was ranked on VAL-period quote volume** (2025-10-02..31) and requires the coin to be alive in
   2025-10 (A1 RT-2). DEV and VAL are not free of selection or survivorship; TEST is free of both.
5. **Contamination.**
   - Kronos-small was pretrained on K-lines to June 2024, crypto included, so its DEV numbers are flattering; it is
     clean for VAL and TEST.
   - Chronos-2 (released 2025-10-17) has an undisclosed corpus cut-off that may overlap VAL; it is clean for TEST by
     release date.
   - Timestral is contaminated for TEST and excluded from every "beats" claim.
6. **The held-out coins were not zero-shot during selection.** Fitting used the 16 training coins only
   (`train_view`), but E-fold P1 selection included the held-out coins' labels. The portability pillar on held-out
   coins is therefore "unseen in fitting, seen in selection". Say so.
7. **What the pillars can and cannot show.**
   - On DEV/F1, rollout vs direct is undistinguished on P1, with a small QS edge.
   - Continual learning showed no gain on the E-folds.
   - The RL agents lost money on VAL.
   - A pillar claim needs Holm < 0.05 and a CI that excludes 0 on TEST. With an MDE of about 5.4 pp (P1) and an
     SE(Sharpe) of about 1 (P2), expect "not significant" unless the effects are large.
8. **Record keeping.**
   - The VAL/rehearsal times in `docs/fin_search.md` §4 are mislabelled by about 20 minutes (DST).
   - `runs/fin/val/report.md` has an uncommitted change.
   - K counts rows, not distinct configurations.

---

# TEST results audit (2026-11-01, after e505cbd)

Inputs: `runs/fin/test/{RUN.json, hashes.json, metrics.json, report.md, test_run.log}` and 62 forecast npz files.
The independent re-derivation is `tools/audit_fin_test.py`. It reads the open and SEALED raw tables (allowed now,
after the single run) and the npz files, and re-implements labels, P1/P3, the day-level DM, 7-day block bootstraps,
the P2 P&L, Sharpe, PSR, DSR, the paired Sharpe bootstrap and Holm. It imports nothing from `tools/fin`.

## Integrity: PASS

- **One complete run.** RUN.json: status complete, 03:42:29 to 05:31:51 -0500 (08:42 to 10:31 UTC), commit 90a86c2
  (my GO commit), `dirty == ""`, argv exactly the GO command, env `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  32 models.
- **Pins match.** `frozen_sha256` = FROZEN.json (f2bbdbc content), and all 7 data pins match, including SEALED.
- **Hashes match.** `hashes.json` (63 entries) lists exactly the 62 npz plus metrics.json, and every sha matches.
- **No file predates the run.** Every npz has `created >= started` and `split == "test"`.

## Re-derivation: 2,154 numbers checked, 2,146 exact, 8 differences that are a convention only

- **What matches.**
  - P1 accuracy and day-level DM p vs 0.5 for every forecaster: crypto all, train and held-out, plus ETF.
  - P3 QS.
  - Every P1/P3 pairwise mean difference, DM p, bootstrap bound and own-Holm p for wm_fin, wm_fin_cl and wm_direct.
  - P2 Sharpe (base and ×2 costs), PSR(0) and DSR at K = 985 for every agent and group.
  - The paired Sharpe-difference bootstrap for wm_agent: the difference exactly, the p to ±0.01.
- **The 8 differences** are all `wm_fin | wm_fin_cl:frozen`. Those two are the same weights, so their forecasts are
  identical and the difference series is 0. My t-test returns NaN where the harness returns p = 1.0. Holm results are
  unaffected.

## DSR sensitivity (K = 985 vs K = 1,038)

Every agent's DSR is ≤ 0.013 at both K. Examples:

| agent (crypto) | Sharpe | PSR(0) | DSR at K = 985 | DSR at K = 1,038 |
|---|---|---|---|---|
| dqn held-out | +1.05 | 0.90 | 0.0124 | 0.0119 |
| tsmom_1 all | | | 0.0124 | 0.0120 |
| wm_agent held-out | -0.52 | 0.32 | 0.0008 | 0.0008 |

No single strategy is significant (it would need PSR > 0.95 and DSR > 0.95), and the choice of K does not matter.

## RuntimeWarning "Mean of empty slice" (run_eval.py:298-299): no effect on any reported number

The warning comes from the P3 pairwise day means. On 2026-10-28 (the missing archive day) no coin has a finite log RV,
so the common-coin-day mean is NaN for that day. The `ok` mask drops it before the DM and bootstrap (they use 363
days). `np.errstate` does not silence `nanmean`'s Python warning, which is why it was printed. My script handles that
day the same way and reproduces every P3 number exactly. P1 has no labels on 2026-10-28..30, because the trailing
7-day reference includes the NaN day, so P1 uses 361 days (prereg §1).

## Uncommitted `runs/fin/val/report.md`: it is the VAL run's own report, never committed

- It was written at 03:33:06 -0500, the same second as `runs/fin/val/metrics.json`.
- The working-tree `metrics.json` is byte-identical to the one committed in cd732b7.
- Regenerating the report from cd732b7's `metrics.json` with `run_eval.report()` gives a file byte-identical to the
  working copy.
- cd732b7 committed `metrics.json` but not `report.md`, so HEAD still holds the older pre-freeze report (0525126).

It is safe to commit; it adds nothing new.

## TEST headline (crypto, 361 days, all 24 coins; Timestral is contaminated)

**P1 accuracy:**

| model | acc | 7-day block CI |
|---|---|---|
| **har_intra_dow** | **0.7310** | [0.706, 0.759] |
| wm_fin_cl | 0.7273 | |
| wm_fin (and wm_fin_cl:frozen) | 0.7269 | [0.699, 0.754] |
| wm_direct | 0.7243 | |
| wm_b_moe4 | 0.7205 | |
| lgbm_dow | 0.7198 | |
| har_dow | 0.7190 | |
| chronos2 | 0.7173 | |
| tirex | 0.7098 | |
| chronos_bolt_base | 0.7079 | |
| lgbm | 0.664 | |
| har | 0.641 | |
| har_roll | 0.638 | |
| lgbm_asset | 0.645 | |
| kronos_small | 0.555 | |
| majority | 0.521 | |

**P3 QS (lower is better):**

| model | QS |
|---|---|
| wm_direct | **0.1465** |
| wm_fin_cl | 0.1470 |
| wm_fin | 0.1471 |
| wm_b_moe4 | 0.1507 |
| lgbm_dow | 0.1520 |
| har_intra_dow | 0.1541 |
| chronos2 | 0.1558 |
| tirex | 0.1576 |
| har_dow | 0.1589 |
| chronos_bolt_base | 0.1600 |

## Claim verdicts (prereg §7 rule: better AND own-Holm p < 0.05 AND bootstrap CI excluding 0)

1. **Beats every strong baseline on P1: NOT MET.** wm_fin 0.7269 is below har_intra_dow 0.7310 (-0.40 pp, p 0.70).
   It also does not significantly beat lgbm_dow (+0.72 pp, Holm 1.0), har_dow (+0.80 pp, Holm 1.0) or wm_b_moe4.
   Against the non-calendar baselines it wins clearly, all Holm < 1e-4: har +8.6 pp, har_roll +8.9 pp, lgbm +6.3 pp,
   lgbm_asset +8.1 pp, harx, rv_persist, garch, kronos_small, majority. Per A5, those wins mostly reflect the weekday
   input.
2. **Versus the foundation models.**
   - **P1, not claimable.** The differences are numerical only:

     | vs | P1 difference | DM p | own-Holm |
     |---|---|---|---|
     | chronos2 | +0.96 pp | 0.32 | 1.0 |
     | tirex | +1.71 pp | 0.087 | 0.78 |
     | chronos_bolt_base | +1.90 pp | 0.063 | 0.63 |

   - **P3 QS (secondary target), claimable.** wm_fin beats all three. Kronos (-0.202) is also beaten, with Holm
     < 1e-4.

     | vs | QS difference | own-Holm | CI |
     |---|---|---|---|
     | chronos2 | -0.0077 | 0.0235 | [-0.0146, -0.0022] |
     | tirex | -0.0096 | 0.0016 | |
     | chronos_bolt_base | -0.0120 | 0.0002 | |

   - **P3 vs baselines.** wm_fin also beats har_intra_dow (-0.0068, Holm 0.039) and har_dow (Holm < 1e-4) on P3,
     but **not** lgbm_dow (-0.0039, raw p 0.020, Holm 0.081) or wm_b_moe4 (Holm 0.057).
   - **Allowed wording.** "Better probabilistic volatility forecasts (QS) than Chronos-2, TiRex, Chronos-Bolt and
     Kronos on TEST; direction accuracy level with the foundation models and with the best calendar-aware baseline."
3. **Rollout vs direct (world-model pillar): NOT MET.** P1 +0.27 pp (p 0.30, CI [-0.23, +0.75] pp). On P3 the rollout
   is slightly **worse**: QS +0.00065, p 0.27, CI [-0.0003, +0.0017]. Holm over the 2 endpoints is far from 0.05.
4. **Continual vs frozen: NOT MET.** P1 +0.035 pp (p 0.75); QS -0.00009 (p 0.51). There were 3 updates per day; it is
   indistinguishable.
5. **WM agent vs model-free RL: NOT MET; the sign is against the WM.** Held-out Sharpe: wm_agent -0.52, ppo -0.27,
   dqn +1.05. wm_agent minus ppo is -0.25 (p 0.18); minus dqn is -1.57 (p 0.025, CI [-2.83, -0.25], i.e. DQN better).
   On all 24 coins: -0.18 (p 0.29) and -1.28 (p 0.058). Holm over {held-out, all} × {ppo, dqn} gives no significant
   WM advantage. DQN's own Sharpe is not significant either (PSR 0.90, DSR 0.012).
6. **Portability.**
   - **Crypto held-out coins: MET as pre-registered.** wm_fin on the 8 held-out coins beats lgbm_asset (+8.0 pp) and
     har_roll (+9.2 pp), both own-Holm < 1e-4, CIs well clear of 0.
   - **Interpret it narrowly.**
     - The per-asset twins have no calendar input; a pooled har_dow reaches 0.7226 on the held-out coins.
     - The held-out coins were never used for fitting, but their DEV labels were used in E-fold selection (pre-TEST
       flag 6).

     So the claim is "a pooled model fitted on 16 coins forecasts unseen-in-training coins better than per-asset
     models fitted on each coin's own history"; it is not zero-shot-clean.
   - **ETFs (descriptive, E1): NOT MET.** wm_fin scores 0.5825, against 0.6031 for lgbm_asset and 0.6018 for
     har_roll, with tirex 0.5985 and chronos2 0.5963. It is below the per-asset models and the foundation models on
     ETFs. The crypto-fitted weekday baselines do badly on ETFs (har_dow 0.535, lgbm_dow 0.522).
7. **P2 overall.** No agent has PSR > 0.95. ETF buy_hold Sharpe is 1.44 (PSR 0.91) and wm_agent's ETF Sharpe is 1.00
   (PSR 0.83, DSR 0.089). There is no trading claim of any kind.

## Disclosures to carry with every TEST number

- The weekday knob was added after the fact (Am2), and the A5 gate was set with the competing values known.
- F1 is post-selection; K = 985 counts rows (DSR insensitive at 1,038).
- The universe was ranked on VAL-period volume.
- Contamination: Kronos (DEV), Chronos-2 (VAL possible), Timestral (TEST, excluded).
- The held-out coins were used in selection.
- The 2026-10-28 archive gap removes 1 P3 day and 3 P1 days.
- P2 ends 2026-09-29.
- The P1 minimum detectable effect is about 5.4 pp, and SE(Sharpe) is about 1 for one year.
