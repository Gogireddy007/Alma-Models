# Alma-Fin red-team leak audit (running log)

Auditor: red-team leak auditor (agent), started 2026-11-01 00:05 EDT. Scope: harness (`tools/fin/**`, lead),
architect A (`tools/fin/wm_*.py`, `agents/**`, `search/**`), architect B (`search_b/**`, `wm_b_*.py`). DEV only:
the auditor ran no training and read no VAL/TEST value. Tests: `tools/fin/redteam/` (auditor's files only).

## Findings

| id | severity | finding | evidence | owner | status |
|---|---|---|---|---|---|
| RT-1 | HIGH (operational) | `models.py` auto-imports `wm_fin` (torch) before any LightGBM fit; torch-then-LightGBM aborts (`OMP: Error #179` / exit 139 / hang). lgbm, lgbm_asset(_etf), lgbm_policy would kill run_eval preflight (TEST impossible) and the architects' refs runs (no row logged) | standalone repro; rt_train_invariance first run died at lgbm (exit 139) | lead; A (refs.py) | **verified** (ba31784, 7329618, a523750: LightGBM n_jobs=1, outputs bit-identical, my R1-R3 re-run on lgbm/lgbm_asset/lgbm_policy still ALL OK; superseded by 644f4f3/A3: torch first, LightGBM imported lazily in factories with n_jobs=1 -> re-verified: R1-R3 ALL OK, and lgbm -> WMContinual -> lgbm_policy in one process completes): a run_eval-style sequence lgbm -> WM forecaster -> lgbm_policy in one process completes. Residual: torch work *followed by* a fresh multi-threaded LightGBM fit in a bare script still hangs; the VAL rehearsal with the full frozen list must be run to confirm |
| RT-2 | MEDIUM (disclosure) | Universe ranked on quote volume 2025-10-02..10-31 (VAL) and "alive in 2025-10": VAL, the go/no-go gate, is selection/survivorship-biased like DEV; only DEV was disclosed | universe.py docstring, prereg 1b | lead | **fixed** (prereg A1.2, fin_leakage section 2); TEST unaffected |
| RT-3 | MEDIUM (selection) | A's search metric was the 2024-07..2025-06 "selection holdout" itself (no inner folds) | fin_search.md section 0, results.tsv F1 rows | A | **fixed** (Am1 1cb4162): loop on E1..E3 <= 2024-06-30, F1 one-shot for <= 3 finalists/pillar; 4 F1 rows pre-Am1 counted in K |
| RT-4 | MEDIUM (cross-asset) | B's training inputs contain held-out coins' DEV data: loop cache contexts hold all 24 coins and wm_core market-factor channels (X[...,5] F_MLV, X[...,6] F_MR) average over them | `rt_arch.py --only B1`: randomising every held-out row changes training-coin X channels 5,6 (hz/har/aux/A/ref unchanged) | B | **fixed + verified (code)**: Xtr from 16-coin contexts for all training inputs (cache loop3); pre-fix rows kept in K, excluded from selection; residual (policy_b pt/qt from 24-coin X) fixed: LB.predict(train=True) from Xtr |
| RT-5 | LOW | data guard covered only data/fin/**; runs/fin/** (label caches, VAL npz) readable during predict/act/observe | harness._audit | lead | **fixed** (ba31784: runs/fin/** guarded except */ckpt/; test 6). Note: */ckpt/ remains readable, keep data files out of ckpt dirs |
| RT-6 | MEDIUM | strict preflight's checkpoint check could pass vacuously (FROZEN "sha256" map optional; wrapper FROZEN_CKPT sha = None) | run_eval.preflight, wm_fin.FROZEN_CKPT | lead, A | **fixed** in preflight (ba31784, requires ckpt + sha for every FROZEN model, wm_ckpt for agents); **verified at freeze**: 6 pins match files and wrapper constants; preflight aborts on a corrupted pin and passes on the committed FROZEN.json |
| RT-7 | MEDIUM (stats) | DSR K must cover both architects' logs (results.tsv + results_b.tsv) and every VAL look | run_eval reads K from FROZEN.json only | lead, A, B | **verified at freeze**: K 985 = A 116-7 refs + B 923-47 refs (ref_b and annot rows counted: conservative) |
| RT-8 | LOW (mis-spec) | A's RL reward used close-to-close ret(D) (from decision time) and funding(D-1), not the harness fill-to-fill P&L | _rl.real_states | A | **fixed** (common.add_fill_rewards; subset() blanks the fold's last label day) |
| RT-9 | LOW (purge) | B policy_b trains on r(D)=fill(D+1)/fill(D)-1 for D <= fit_end: the last sample ends at the 00:05 bar of the fold's first scored day | policy_b.run (trw) | B | **fixed + verified (code)**: trw days < fit_end |
| RT-10 | INFO | funding settled at D 00:00 is in row D-1 / hour ending D 00:00 (calc_time a few ms after 00:00; known before the 00:05 fill); label window 00:05-24:00 overlaps the fill bar (both after the decision); Yahoo adjclose levels embed later dividends but only ratios are used | data.py | lead | accepted |
| RT-11 | MEDIUM (stats) | Am1 picks the pooled finalist as the best F1 of up to 6 (A+B): F1 becomes a selection set; the winner's F1 is max-of-6 biased and its twin is not selected alike, so F1 candidate-vs-twin differences are biased. Also WM KEEP decisions use all 24 coins, so DEV held-out (portability) numbers are selection-contaminated | fin_search.md Am1 'Finalists' | A, lead | **accepted**: F1 labelled post-selection, F1 twin diffs descriptive only, K counts F1 looks, held-out DEV disclosure; verify in FROZEN.json at freeze |
| RT-12 | LOW-MEDIUM (stats) | make_frozen.py var_sharpe_trials mixes B's per-fold agent rows with their EM (fold-mean) rows -> variance biased down -> DSR inflated | make_frozen.sr_period_of fallback, results_b agent_b EM rows | A | **fixed** (3a48f34: per-fold rows only, (row, fold) list recorded) -> **verified at freeze** (74 rows = A loop_rl per-fold 30 + E3 screens 4 + B agent_b folds 30 + F1 looks 10; B EM rows excluded; var recomputed 0.0021806 = recorded) |
| RT-13 | MEDIUM (scoring) | P1 accuracy / daily accuracy (pairwise DM, bootstrap) and P3 QS are computed on each model's OWN finite predictions: a model abstaining (NaN) on hard coin-days is scored only where it answers, and pairwise tests compare different coin-day sets (selective-prediction cheat). Not exploited now (A F1 preds 8760/8760 coverage; B fold preds 284 full, 35 all-NaN diverged and scored 0, 0 partial) | run_eval.score/pairwise, common.p1_p3, loop_b.score | lead (A, B for loop scores) | **fixed + verified (code)** c8adbad/A4: P1 scored on every labelled coin-day (missing -> DEV-majority direction), P3 pairwise on the common set, coverage in metrics.json, claim void below 99% coverage. Residual: loop scores (common.p1_p3, loop_b.score) still own-coverage, harmless now (0 partial-coverage fold preds) |
| RT-14 | INFO (provenance) | DST ended 02:00 EDT -> clocks now EST; A's loop `--until` uses local wall clock (extra hour) and amendment times mix EDT/EST | `date` 01:03 EST after 01:59 EDT log lines | A, lead | open (use UTC/explicit offsets in freeze/VAL ordering evidence) |
| RT-15 | HIGH (claim validity; not a leak) | Pre-registered P1 baselines have no calendar term, but the label has a strong weekday effect (DEV-early up-rate Sat .243, Sun .255, other days .505-.579). F1: weekday-only rule 0.609 (majority 0.523), HAR 0.664 (own OLS = harness 0.6638), **HAR + weekday dummies 0.693** (held-out 0.703), har_roll 0.668, lgbm 0.669; B's weekday-input finalists 0.703/0.706/0.715. A 'beats every strong baseline' claim would largely be the calendar | rt_f1_refs.py + HAR+dow OLS (pass 6) | lead (A, B disclose) | open, accepted by coordinator (freeze held): add har_dow / lgbm_dow **and har_intra_dow** by prereg amendment BEFORE freeze. Follow-up rt_gap.py: 13-parameter OLS (HAR + ref + weekday + US holiday + last-6h and last-hour log-RV of D-1) = **0.7154** F1 (held-out 0.727) = B's best 0.7152; weekday ~+3 pp, holiday ~+0.5 pp, intraday recency ~+1.5 pp; funding settlement times are fixed daily (no day-specific info). A5 baselines (lead working tree) verified: info set + R1-R3 pass; F1 P1 har_dow 0.6968, lgbm_dow 0.7087, **har_intra_dow 0.7167** (held-out 0.7288) >= B best 0.7152, > A RSSM 0.678-0.685 -> **fixed** (0525126 = tested code; fold_refs.json F1 equals auditor's) |
| RT-16 | MEDIUM (selection gate) | Am2's P1 finalist gate names har_intra_dow as the best calendar-aware E-fold reference, but on E1-E3 lgbm_dow is stronger (EM 0.6930 vs 0.6815; har_dow 0.6751) | runs/fin/fold_refs.json | A (B, lead) | **fixed** (7ce49da: gate = max registered dow-aware baseline, EM + per fold) |

## Tests run (all DEV; outputs next to the scripts)

- `rt_train_invariance.py` (lead factories; positive control R0 bites): R1 every open row > 2025-06-30 randomised
  before the fit; R2 TRAIN_END patched to 2024-06-30 and rows > 2024-06-30 randomised; R3 held-out coins
  randomised. majority, rv_persist, har, harx, lgbm, lgbm_asset (training-coin columns), voltgt_tsmom, lgbm_policy:
  **all bit-identical**.
- `rt_arch.py`:
  - I1: every architect ckpt on disk (A: wm_d64/d128 none/direct F1, loop_b9d2692d92 E1), crypto + ETF, 4 DEV days,
    future daily/hourly values *replaced* by random values (NaN flips included): **identical**; D-1 randomised: changed.
  - C1: WMContinual on a fold ckpt, 10 days, randomised from day j: decisions <= j identical, later differ.
  - P1: fold P1 recomputed from the raw parquet (own labels): all 4 F1 rows match results.tsv (e.g. d128 none 0.6853,
    held-out 0.6928).
  - B1: see RT-4 (FAIL for B; A's datasets are built on training-coin-only contexts).
  - A1: A's cached training sets rebuilt from scratch clean / with every row > 2024-06-30 randomised / with every
    held-out row randomised: legacy `ds_2024-06-30.npz` and loop `ds_w28_2024-06-30.npz` are **bit-identical** in all
    three cases (no future or held-out data; caches not stale). The fold-last-day purge of the fill-based RL rewards
    (`common.subset`) was read and is correct.
  - P1b: all 22 B fold scores in results_b.tsv (7 configs x E1..E3 + smoke) re-derived from raw labels: match.
- Code review checks passed: labels (ref = D-7..D-1, harness-side), fill = 00:05 bar, P&L fill(D)->fill(D+1) with
  fee + slippage on |delta pos| and funding (D 00:00, D+1 00:00], ETF decision at D-1 close / open fill, A's dataset
  (16 coins, max label day 2024-06-30), B's loop cache cut at 2024-06-30 and fits on train_mask with label day <=
  fit_end, cl_b updates after the decision, no architect code references VAL/TEST/SEALED. VAL baseline results are
  committed (7bf8637) and readable by the architects (disclosed in fin_results section 5; looks count toward K).

- `rt_pnl.py`: P2 of tsmom_30 and buy_hold on F1 recomputed from the raw parquet with own fill/funding/cost code: identical to harness.pnl (max diff 0).

- Pass 2 code review (00:35-00:45): A finalize.py / make_frozen.py, B holdout_b.py, B RT-4 fix (Xtr) and RT-9 fix.

- Pass 3 (00:45): I1 re-run on the 3 newest A loop fold ckpts (mutated configs): 24/24 OK. Loop accounting: 13 loop rows vs fold ckpts consistent (EM rows = 3 ckpts, E3 screens = 1).

- Pass 4 (01:10): B results_b.tsv 470 rows (422 wm_b, 47 ref, 1 annot): all 320 B fold scores with saved preds re-derived from raw labels (exact); B selection summary (summ_b.py) averages over seeds (no seed cherry-picking) and uses post-RT-4 rows only. A loop 21 rows.

- Pass 5 (01:40): B rounds 2-5 (weekday one-hot of decision day D: known at decision, OK), B cl2_b shadow-learner meta-rule (predict D -> observe y(D) -> update; learner choice uses only past days' BCE: OK; note it fine-tunes on held-out coins' revealed labels, so a continual model is not zero-shot, consistent with prereg 'zero-shot scored with observe disabled'). A after_wm.sh chain (cl -> rl loops -> refs on F1). Counts: A 33 rows, B 769 rows.

- Pass 6 (01:05 EST): B F1 ckpts (3) pass randomised-future through BForecaster/BVolTgt/BContinual (I1b, crypto + ETF); meta fit_end 2024-06-30, 16 coins, loop3. F1 P1 re-scored from preds (A4 rule): B 0.7031/0.7056/0.7152, A F1 rows 0.678-0.685; refs refit at 2024-06-30: har 0.6638, har_roll 0.6676, lgbm 0.6686, rv_persist 0.6603 -> RT-15.

- Pass 7 (06:10 UTC): gap decomposition (rt_gap.py); B's 4th F1 ckpt + all B ckpts re-run through I1b.

- Pass 7b: B freeze refit `wm_b_moe4_refit.pt` (fit_end 2025-06-30, 16 coins, cache refit4): rt_b_refit.py rebuilt the refit cache with every row > 2025-06-30 AND every held-out row randomised -> training-coin Xtr/targets/hz/har/aux identical (no VAL, no held-out data in the refit). Same cfg/steps/seeds as its F1 version. Its F1 choice among 3 B finalists is post-selection (RT-11 applies to B too).

- Pass 8 (~06:20 UTC): A5 baselines tested (rt_new_baselines.py, rt_train_invariance.py --models har_dow,lgbm_dow,har_intra_dow): ALL OK.

- Pass 9 (~06:25 UTC): all 10 B ckpts (3 F1 finalists, freeze refit wm_b_moe4, 6 family leak ckpts) x {BForecaster, BVolTgt} x {crypto, ETF} x 4 days + BContinual walk: 170/170 OK. wm_b_fin.B_CKPT sha for wm_b_moe4_refit.pt matches the file. B F1 looks: 3 forecaster + 2 agent (count in K).

- Pass 10 (~06:28 UTC): A Am2 read (post-hoc dow knob, UTC stops, har_intra_dow finalist gate, B disclosures): dow is the decision day's weekday in training (day_str) and inference (ctx['day']); continual backfill uses each sample's own label day. Note: the dow extension was triggered by F1 information (RT-15), so F1 of dow configs is post hoc (disclosed in Am2). Newest 3 A loop ckpts: 24/24 OK.

- Pass 11 (06:33 UTC): first dow RSSM fold ckpts (loop_0dbccd3e62 E1/E3, dow=True): randomised-future I1 (crypto + ETF, 4 days) and WMContinual C1 on the dow ckpt: ALL OK.

- Pass 12 (07:35 UTC): A freeze refit started. frozen_wm_fin.pt (recipe 50b221e074, dow=True, 3 seeds, fit_end 2025-06-30): member sha256s match the fin_*_2025-06-30 files and weights are identical; members' last_label_day 2025-06-30. rt_a_refit.py rebuilt ds_w28_2025-06-30 with every row > 2025-06-30 AND every held-out row randomised: bit-identical (no VAL, no held-out data in the refit). I1 on frozen_wm_fin + refit members: 48/48 OK. FROZEN.json not yet written.

- Pass 13 FREEZE AUDIT (07:50-08:05 UTC, f2bbdbc): rt_frozen.py (K, var_sharpe_trials, 6 sha pins on disk and in wrapper
  constants, go/no-go, post-selection labels, commit present): all OK. rt_frozen_models.py: every FROZEN model through
  the registry (wm_fin, wm_direct, wm_fin_cl:frozen, wm_agent, ppo, dqn, wm_voltgt, wm_b_moe4) x crypto/ETF x 4 DEV days:
  randomised future -> identical, D-1 randomised -> changed (dqn included, so its prev_hour insensitivity in A's test is
  argmax granularity, not a leak); wm_fin_cl continual walk identical up to the perturbation day, 3 updates/day;
  LOAD_ERRORS empty; preflight negative test aborts. Frozen refits (A: rt_a_refit.py; B: rt_b_refit.py) contain no VAL or
  held-out data. Pillars declared (rollout vs direct, continual vs frozen, wm_agent vs ppo/dqn, portability, beats incl.
  A5 baselines); claim models wm_fin / wm_fin_cl / wm_agent match them. Note: wm_fin's F1 P1 0.7077 is below
  har_intra_dow 0.7167 (post-selection F1, descriptive). **Verdict: GO for VAL --rehearse + VAL.**

## Passes

- Pass 1 (00:05-00:40): harness, lead factories, A round 1 + loop (wm/cl/meta/rl scoring: fold WM and fold-subset
  start states for imagination, meta-rule compares losses on the just-revealed day only), B loop/policy/cl/wrapper.
  Run accounting at 00:28: A results.tsv 6 rows (4 F1, 1 loop crash, 1 loop START) vs 7 ckpts (loop START = 3 fold
  ckpts in one row): consistent. B results_b.tsv 48 rows (20 ref, 28 wm_b = 7 configs x 4), 39 pred files, no B
  ckpts saved during the loop (scores reproducible from seeds only).
