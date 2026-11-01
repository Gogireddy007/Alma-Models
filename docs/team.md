# Alma research team

**Success:** Alma, as a genuine world model for time series forecasting, beats every baseline on accuracy (scores, lower is better), interval calibration, efficiency and zero-shot generalisation, with no leakage. The baselines are Chronos-2, Timestral 5M/28M/209M, TiRex, Chronos-Bolt, AutoETS, AutoTheta and Seasonal Naive.

| Researcher | Area | Works in |
|---|---|---|
| Latent-objective engineer | JEPA/EMA latent-prediction loss, so the dynamics are trained as a world model | train_latent.py |
| Dr. Ines Varga | Transition dynamics: long-rollout accuracy, RSSM/attention transitions | train_dyn.py |
| Dr. Kofi Mensah | Probabilistic head and calibration (cov80, sharpness) | train_calib.py |
| Dr. Mei Lin | Training efficiency on M3 8GB and the scaling recipe for the overnight run | train_fast.py |
| Dr. Sofia Andrade | Independent audit of leakage, fairness and the claim protocol (red team) | docs/audit_v2.md |
| Dr. Aarav Iyer | Literature scout: ranks techniques from Chronos-2/TiRex/TimesFM/world-model papers by gain per unit of compute | docs/lit_review.md |
| Dr. Lena Hoffmann | Covariates and multivariate targets as world-model "actions" | train_cov.py |
| Dr. Tomás Rivera | Inference-time scaling: test-time augmentation, origin fusion, sampled rollouts, test-time training on the series' own context | train_tta.py |
| Dr. Yuki Tanaka | Training-data mix, synthetic prior v4, leak-checked candidate corpora (proposal only, no downloads) | train_data.py, tools/synth4.py |
| Dr. Priya Raman | Evaluation infrastructure: fixes the critical and high issues from audit_v2 (claim gate, frozen-checkpoint test eval, resume and extra-data guards) | tools/, train.py (atomic install) |
| Dr. Nadia Okafor | Multi-scale / hierarchical world model for long horizons (h 96-288, hourly and sub-hourly) | train_ms.py, runs/jobs/ms.txt |
| Dr. Arjun Mehta | Transition architectures: Koopman, diagonal SSM, RSSM (stochastic), attention | train_arch.py, runs/jobs/arch.txt |
| Dr. Elif Demir | Integration: merges the winning flags into train_v2.py, bit-identical at defaults | train_v2.py |
| Dr. Hana Kim | Compute and memory steward: jobq concurrency, the Chronos-Bolt OOM, training memory audit | tools/jobq.py, tools/bench.py, docs/compute.md |
| Dr. Leo Brandt | Autoresearch automation: leaderboard and automatic rung promotion | tools/leaderboard.py, tools/promote.py |
| Dr. Sara Lindqvist | Stochastic world model: flow-matching latent transition, sampled futures (EpicWorldModel) | train_flow.py, runs/jobs/flow.txt |
| Dr. Kenji Watanabe | Synthetic world prior v5: multi-seasonal, regime-switching, chaotic and intermittent series | tools/synth5.py, train_syn5.py |
| Dr. Aarav Iyer | Literature v2: WorldTS, latent-action TS world model, flow-matching forecasting, TEM | docs/lit_review_v2.md |
| Dr. Sofia Andrade | Integrity audit v3 (read-only): leakage, harness tricks, statistics | docs/audit_v3.md |
| Dr. Marco Rossi | Independent evaluator (read-only): re-derives every reported number with his own code | tools/verify_claims.py, runs/verified.md |
| Dr. Lena Park | Failure analyst: gap decomposition vs foundation models | docs/gap_analysis.md |
| Dr. Ingrid Solberg | Forecast-evaluation statistician: pre-registration, power, multiplicity (plan v2 Phase 3, Gates 3-5) | docs/preregistration.md |
| Dr. Tomás Ferreira | Evaluation infrastructure: prediction store, independent re-scorer, official-harness adapters (plan v2 Phases 1 and 7) | tools/predstore.py, tools/rescore.py, docs/eval_pipeline.md |
| Dr. Yara Haddad | World models and dynamical systems: hypothesis tests, ablation matrix, pattern taxonomy (plan v2 Phase 6) | docs/wm_hypothesis_tests.md |
| Lead (Claude) | Queue scheduling, merges into train.py, comparisons, final claim | train.py, runs/ |

## How we work sustainably

- **One heavy job at a time.** The machine is an M3 with 8 GB. Researchers write code and run only smoke tests; the lead runs ablations through the queue.
- **Never edit train.py directly.** Running queues read it. Each idea lives in its own copy behind an env flag, with defaults unchanged, and the lead merges it.
- **Fair ablations.** Every ablation uses the same fixed-step budget: TSAR_BENCH=1, TSAR_STEPS=2500. Each is compared with paired CIs using tools/compare.py.
- **What gets merged:** only what wins on bench val and doesn't break calibration.
- **Test split:** never touched during iteration. There is one pre-registered confirmatory run at the end (see docs/audit_v2.md).
- **Honest reporting:** final checkpoints, multiple seeds, and failures reported plainly.

## Experiment ladder (small first)

Every idea climbs the same ladder, and only winners spend more compute.

| Rung | Scale | Cost per run | Who gets in |
|---|---|---|---|
| 1. Screen | 0.37M params, 2500 steps, 1 seed, bench val | ~5-8 min | every idea |
| 2. Confirm | same size, 3 seeds, paired family CIs with compare.py | ~20 min per idea | ideas that beat b_base256 on rung 1 |
| 3. Combine | ~1-2M params, ~10k steps, the confirmed winners stacked, 2 seeds | ~30-45 min | the combined recipe, plus leave-one-out of each component |
| 4. Final | overnight (Lin's recipe) | hours | only after rung 3 beats the Timestral and Chronos-2 val numbers |

The test split is touched once, after rung 4, under Gate G.

## Queue log

Run order: superseded at 23:53. See research_log and runs/jobs/rung2.txt (seeds, then arch, then ms, then scaling probe, then borderline seeds, then the Bolt retry).


- queue8: dynamics ablations (Varga): b_dyn_j2d3, b_dyn_j2, b_dyn_d3. The control is b_base256.
- queue13 (runs after queue8, before queue9): world-model objective (latent engineer): b_jepa_vic, b_jepa_sig, b_jepa_sg, the audit ablations b_a1_copy / b_a1_jepa, and A3 (states zeroed at eval).
- queue9: inference-time sweep (Rivera) on the big2 checkpoint, bench val only, 21 configs.
- queue10: covariates as actions (Hoffmann): b_cov1, then the same checkpoint evaluated with covariates withheld.
- queue12: calibration (Mensah): train c_base once, re-score it with chunk-scale / mix / mix+scale / carry; c_aux (lower-layer quantile head) trains on its own.
- queue14 (after queue12): efficiency and schedule (Lin): b_fast (end-to-end check of the fast flags), b_wsdl, b_pcs, Muon at 6e-3 and 1.2e-2.
- queue11: data mix (Tanaka): a1 (mix temperature), a2 (stretch), c and a3 (synth4), a6 (combined), a6p (combined + prefix).

## Protocol decision: covariates

The bench gives every model univariate contexts only; covariates are dropped and multivariate targets are split. The primary claim stays on that identical-input protocol.

Covariate use is a secondary track. It is reported only against a covariate-aware Chronos-2 (chronos2_cov), never against the univariate baselines. TEST has 20 known-future covariate tasks across 10 families, while VAL has only 3 tasks from 1 family, so any covariate claim is weakly selected and must be labelled that way.

## Compute rule (added 15:30, after the memory thrash)

The Mac hit 13 GB of swap and the queued baseline stalled. From now on:
- Researchers run only smoke tests that take under 60 s and stay under 1 GB of memory.
- Anything bigger, such as eval sweeps or 2500-step runs, goes to the lead as a command to queue.
- Only the queue runs heavy jobs.

## Job runner (from 23:20; replaces runs/queueN.sh)

The Mac rebooted at 20:14 and killed the shell queue mid-run. Every heavy job now goes through `tools/jobq.py`:
- It is resumable: a job is done once its log has `val_wql_rel:`. Relaunching skips finished jobs; a crashed job leaves `runs/<name>.failed`.
- Measured 23:29: two jobs at once ran about 5x slower each (GPU memory contention plus swap), so run `--max 1`. It still admits a job only while the kernel's available memory minus the job's estimate stays above `--reserve`.
- Researchers deliver screens as a jobs file, `runs/jobs/<area>.txt`, in the format given in the jobq.py docstring. The lead runs it.

## Memory-efficient working rules for every researcher agent

- **Read narrowly.** Grep for the symbols you need, then read only those line ranges. Never read a whole train*.py; each is about 900 lines.
- **Check memory before a smoke test.** If `sysctl -n kern.memorystatus_level` is below 25, wait 60 s and check again. Training jobs share this machine.
- **Smoke tests:** always run through `uv run tools/smoke.py -- <cmd>`, which holds a team-wide lock (one smoke test at a time) and waits for memory. Keep them under 60 s; peak RSS under 1 GB, or under 2 GB when the smoke test must load TSAR_BENCH data (that load alone is about 1.3 GB). Use `TSAR_STEPS=30` and a small eval (`TSAR_EVAL_MIN` if supported), never TSAR_BENCH full eval. Never run more than one at a time.
- **Never score test or holdout.** Never edit prepare.py, baselines.py, train.py, or another researcher's file.
- **Flags only.** Every new idea is an env flag in your own copy, and defaults must reproduce train.py exactly.
- **Report briefly:** under 300 words, covering the flags, what the smoke test showed, the jobs file path, and the risks.

## Claim pipeline (from 23:58)

No number reaches the user as a result until all five steps are done:
1. The run finishes under jobq, with provenance (git sha, script sha256, env, checkpoint sha) in `runs/<name>.prov.json`.
2. The leaderboard (Brandt) and `results_bench.tsv` (autoresearch loop) record it as keep, discard or ablation.
3. Rossi's `tools/verify_claims.py`, an independent re-implementation, reproduces it. A MISMATCH blocks the claim.
4. Andrade's audit checklist (docs/audit_v3.md) passes, with no open BLOCKER.
5. The claim is reported with seeds and CIs, and is labelled val-selected. Test is touched once, at the end, under Gate G.

## Multiplicity and selection disclosure (audit_v3 H-4)

- **Screens so far:** about 60 on the same 30 VAL tasks (13 families): 38 leaderboard configs, 20 TTA configs, and rung 2 plus the arch/ms/flow/syn5/dualdec screens queued.
- **VAL is a selection set.** Every VAL number is val-selected and optimistic, and is reported as such.
- **The only confirmation is TEST.** It is touched once, at the end, with one pre-registered frozen checkpoint (train_v2.py, TSAR_EVAL_ONLY + sha256). Gate G runs against the full baseline set with Holm across claims.
- **Merge rule (R7):** merging single-seed winners into train_v2.py is allowed only behind default-off flags. The rung-3 recipe uses only rung-2-confirmed flags (3 seeds, seed-matched CI upper < 1).
- **Pre-test checklist:** the 10-point claim checklist in docs/audit_v3.md must pass before the test run.

## Autoresearch discipline for every agent (from 2026-10-30; user request)

This repo is an autoresearch fork: `program.md` is our adaptation and `vendor/autoresearch/` is Karpathy's upstream reference. Every agent that runs, designs or judges experiments works the autoresearch way:
1. **Read before choosing.** Read `research_log.md`, `runs/ledger.tsv` and `docs/lessons.md` before picking an idea. Never repeat a failed idea without a new, stated hypothesis.
2. **One hypothesis, one change, one commit.** Write the hypothesis and its research track (T1-T11 in program.md) in the log before the run, and the outcome after.
3. **Keep rule.** Keep only above the 3-seed noise floor, with mase not worse by more than 1%, and memory and calibration within limits. Ties go to the simpler, smaller or faster model.
4. **Log rows.** Log every run as keep / discard / crash / ablation.
5. **Ablation duty.** Every 10 experiments, ablate a world-model part of the incumbent.
6. **Grep the summary.** Read the metric lines with grep and never flood your context with logs.

Use the loop for productivity, not as a licence. Gates 1-5 of the user's research plan (RESEARCH_PLAN v2, Phase 9) override "never stop": while Gate 1 holds, nobody starts a training run.

## Decision rules from 2026-10-30 (docs/preregistration.md, Solberg; supersedes the 0.008 keep rule)

- **Reference baselines:** Chronos-2 and Timestral-209M, both fixed before results; every target must hold against each.
- **Gate 3 (early signal):** a gain of at least 0.020 wql_rel on 3 *fresh* seeds; a screening seed never counts as confirmation. A gain of 0.008-0.020 goes to 5 seeds and needs at least 0.013.
- **Task ties:** ±2% (3-seed mean log ratio). "Wins on over 60% of tasks" also needs a family-bootstrap lower bound above 0.50 and a majority of families.
- **Selection disclosure:** about 60 VAL screens give a winner's-curse inflation of about 0.021, so every VAL-selected gain must be re-confirmed on a locked Tier-D split. Under the GIFT-clean regime, TIME (minus EPF) and GIFT-Eval are never used for selection.
- **Power:** our VAL can detect about 11-14% vs a baseline at 80% power. A 10% target is a point target, not a provable lower bound.
- **Failed runs:** wql_rel > 2 or non-finite is labelled failed, never ranked (lessons L6; enforced in 974980f).
