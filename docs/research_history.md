# Alma research history (Phase 0 archaeology, 2026-10-30)

Author: Marco Rossi (results verifier). Sources: `runs/*.log`, `*.prov.json`, `*.time`, `runs/jobs/*.txt`, `runs/queue*.sh`,
`data/ckpt/*.val.json`, `runs/verified.md`, `runs/leaderboard.md`, `results_bench.tsv`, `runs/search/`, `research_log.md`,
`docs/audit_v2.md`, `docs/audit_v3.md`, `docs/gap_analysis.md`, git log. The ledger is `runs/ledger.tsv`, with one row per log
(269 rows: 112 bench-val, 143 bench-dev, 14 old-suite). No training or evaluation was run, and no test or holdout data was read.

## How to read the statuses
- **verified**: prov.json present, script sha256 equal to the script at the recorded commit, headline recomputed from the
  per-task table, and part of a 3-seed group. Only 12 rows qualify, and **no 3-seed group is verified on all three seeds.**
- **preliminary**: the same checks pass, but the run is a single-seed screen or a DEV screen (125 rows).
- **reported-unverified**: there is no prov.json. That covers every run before `abe9f26` (00:01 Oct 30), every baseline, and the old suite (66 rows).
- **failed**: 62 rows: 56 search R0 crashes (`runs/failed_r0`), ctx1024 and ctx2048_b64 (eval OOM), dualdec_05 and dualdec_05_jepa (blow-up), and big1 (no final metrics).
- **unresolved**: 4 rows: syn5_only_s2, ctx512, big2, and search_r0_c0003 (still running).
- Protocols: `old-suite` (prepare.evaluate, 15 monash-style datasets) and `bench-val` (30 VAL tasks, 13 families) are **not
  comparable**. `bench-dev` (9 DEV tasks from TRAIN families, used by the search) is a third scale; a DEV score near 1.0 is normal there.
- Recompute: the per-task wql_rel geomean matches the printed headline to within 2.4e-4 (max abs) on every bench and old-suite log. The tables are
  printed to 3 decimals, which accounts for the difference.
  The two dualdec blow-ups differ by up to 4e1 in absolute terms because of the same rounding.
- Provenance limits: prov.json hashes only the train script, bench.py, prepare.py and jobq.py, not `tools/synth*.py`
  or other imports. 165 of 201 prov rows are `+dirty` (23 of them bench-val). In all but syn5_only_s2 and ctx512, the train script still matches the commit.

## Headline numbers, reconciled

| claim | source runs (seeds) | protocol | value | status |
|---|---|---|---|---|
| "0.695 three-seed" | four separate recipes, each over seeds 0,1,2: syn5_only 0.6945±0.0082; b_d_a6p 0.6950±0.0026; b_curr 0.6953±0.0035; b_syn3 0.6959±0.0030 | bench-val | ~0.695 | b_curr/b_d_a6p/b_syn3: s0,s1 reported-unverified, s2 verified. syn5_only: s0,s1 verified, s2 unresolved |
| "0.681" | syn5_10k, seed 0 only (s1/s2 dropped per research_log 07:50) | bench-val | 0.680626 | preliminary (5737cef+dirty, script matches) |
| "Chronos-2 0.559" | bench_chronos2.log, one inference pass | bench-val | 0.5586 | reported-unverified (metrics recomputed OK) |
| c_mix calibration | eval-only re-score of c_base, c_base_s1, c_base_s2 ckpts | bench-val | cov80_err 0.0999→0.0812 (s0); 3-seed 0.1097→0.0869, wql 0.7322→0.7267 | s0,s1 reported-unverified; s2 verified |
| A3 / A1 ablations | b_jepa_vic_A3 (s0), b_a1_copy (s0), b_a1_jepa (s0) | bench-val | 0.9322 / 0.7447 / 0.7459 | reported-unverified, single seed |

- **"0.695" is four claims, not one.** The four recipe means are within 0.0014 of each other, and their seed sd is 0.003–0.008, so they are tied.
  - The quoted b_curr seeds "0.6933/0.6934" are really 0.6932/0.6933 (verified.md).
  - syn5_only's seed 0 (0.6853) was lucky. Its other seeds were 0.6971 and 0.7010.
- **syn5_only_s2 and ctx512 are unresolved** because the prov records HEAD `cce8070+dirty`, but the train_syn5.py sha256 (`4b56ebf…`) does not match cce8070.
  - Those exact bytes were committed later, at `0db6bda` (07:47), with the synth6 flags default-off. The code is recoverable.
  - Strictly, though, the run predates its commit.
- **0.681 syn5_10k** is a single seed. Its matched control is sc_d96_l2_10k: the same curr recipe at the same size and 10k steps, without SYN5, also seed 0, clean `cce8070`. That control scored 0.685076.
  - The difference is −0.65%, which is inside the seed sd of 0.003–0.008. The "best single run" label is accurate, but the run is not evidence for synth5.
- **Chronos-2 0.559 was produced by local inference**, not imported. `bench_chronos2.log` shows HF weights loading,
  `baselines.py:_chronos("amazon/chronos-2")` via `tools/bench.py baselines`, a predict time of 541.5 s, and a result dict with git `6c3c6c2` (2026-10-29 14:28).
  - Limitations: there is no prov.json, and the HF revision is not pinned (`from_pretrained` has no `revision`).
  - The per-task CSVs under `baselines/bench/` are untracked in git.
  - verified.md recomputes 0.55860 from the CSV, and the task identity matches.
  - The research_log table "Chronos-2 0.621" is the old suite. Do not mix the two.
- **c_mix** is a free inference-time calibration change.
  - Paired 3-seed ratio 0.992 [0.986, 0.998] (leaderboard).
  - Seed-0 ratio vs c_base 0.9936 [0.9874, 1.0003] (verified.md).
  - c_base seed 0 equals b_base256 and v2_base to 6 dp. This is deterministic MPS on the same code path, and v2_base (clean prov at cce8070) independently reproduces it.
  - c_mix_s2's prov ckpt sha (`c92d3a…`) is the file *after* the re-score rewrote it (R-2). The parent `c_base_s2.pt` now matches its own prov (`e9516a…`).
- **A1** state copy, no learned transition: +2.6% vs b_base256, CI [0.993, 1.071]. That is not significant.
- **A3** states zeroed at eval: +28.4% vs b_base256, +28.6% vs its parent b_jepa_vic. A3 shows that the decoder reads the state. It does
  **not** show that learned dynamics help. That question belongs to A1, and A1 is inconclusive.

## What works (with evidence)
- **Synthetic prior + curriculum + augmentation (the "curr" recipe)**: the b_curr 3-seed paired ratio vs b_base256 is 0.950 [0.905, 0.989]. That is
  about −5%, and three independent recipes in the same family reproduce it (b_syn3, b_d_a6p, syn5_only, all ~0.695).
  - The evidence is the strongest in the project, but it is still 1/3 or 2/3 provenanced per group.
- **More steps on the curr recipe**: sc_d96_l2_10k 0.6851 and sc_d128_l3_10k 0.6917 vs 2500-step ~0.695. Width and depth do not help:
  sc_d192_l4_10k scored 0.7050. All of these are preliminary (single seed, clean cce8070).
- **c_mix calibration**: cov80 error −21% (3-seed), with no accuracy cost.
- **Older wins (old suite only, not re-tested on bench):** multi-origin training, exp23 normalisation, exp24 seasonal memory.
  These are now baked into the base, so bench numbers cannot isolate them.

## What failed
- **dualdec_05 / dualdec_05_jepa**: wql_rel 16.6 and 2.27e6 (failed).
  - Cause: the consistency term collapsed the encoder state, and the tail quantiles exploded through sinh.
  - Fix in `8a035d0` (CONS default 0). The same change also exists as dangling commit `d3ff8a6`, which research_log cites but which is on no branch.
  - After the fix: dualdec_05_c0 0.7222 and +jepa 0.7406 (preliminary). The method is now sane but neutral or worse vs v2_base 0.7258.
- **Flow transitions**:
  - MeanFlow flow_sig 0.849, flow_s16_vinc 0.854, flow_det_infer 0.878, flow_s16_emp 0.961 (cov err 0.759).
  - DV-Flow dvf_s16 0.735, dvf_med 0.730.
  - All are single seed and preliminary. Every one is worse than the base 0.726 at this budget.
- **Multi-scale coarse/ext** 0.79–0.81. **Muon** 0.83–0.85 (LR not tuned). **TTT/self-calibration** hurts on both protocols.
- **Search R0**: 56 of 122 jobs crashed (20 N_LAYERS=1 index errors, 6 from the missing `memory()` in train_arch2, 30 MPS OOM). Fixed in `96ee856`.
- **ctx1024 / ctx2048_b64**: training finished, then VAL eval OOMed. The ckpts were scored by the *_eval re-scores.

## Dynamics, world-model and architecture lines
- **b_dyn_j2** 0.7192±0.0081 and **b_dyn_d3** 0.7255±0.0118 (3 seeds each; 1/3 and 2/3 provenanced).
  - The seed-0 wins (0.7099, 0.7125) did not hold. Neither config was kept.
  - arch_koopman_j2 0.7131 (single seed) is the only transition swap below the base. ssm, attn and koopman are neutral, and rssm is worse.
- **Mehta arch2 screens** (5737cef+dirty, script matches, seed 0, preliminary): a2_drive 0.6868, a2_slots4 0.6918, a2_dread 0.6950,
  a2_deep_hz 0.6967, a2_mres 0.7037. The parent syn5_only s0 scored 0.6853, so none of them is a gain.
- **Search c0099 / c0110** (train_arch2, R0, 1200 steps, DEV, seed 0): DEV 0.9182 and 0.9206, i.e. 0.885x and 0.888x the incumbent c0000 (1.0372).
  - Incumbent DEV noise sd is 0.005 over 3 seeds, so these gaps are about 20 sd.
  - They are still preliminary: one seed, a short-step DEV proxy (Spearman 0.867 at n=10), and no VAL score yet.
  - A2_DRIVE and DYN_IN=2 give bit-identical DEV results (c0046 = c0020). Treat them as one axis.
- **JEPA objectives**: neutral (0.7246–0.7322, single seed). They keep the world-model loss at no accuracy cost, and show no gain.

## Synthetic v5 and v6
- **synth5**: 3-seed syn5_only 0.6945 vs b_curr 0.6953. No measurable gain.
  - syn5_curr_25 0.6894, syn5_curr_50 0.6983, syn5_frac50/70 0.700/0.704 (single seed).
- **synth6**: gp30 0.6978, gpmix 0.7000, gpmix_f50 0.6980, mix30 0.7039 (single seed, preliminary). No gain vs the syn5 recipe.
- Audit S-2: the synth5 period weights favour sub-hourly data, which matches the VAL/TEST frequency mix. Disclose this as data-prior targeting.

## Context
- ctx512 0.6856 (unresolved: script sha) and ctx1024_eval 0.6844 vs syn5_only s0 0.6853: no gain at batch 128.
- ctx2048_eval 0.6892 vs ctx256_b64 0.7104 at batch 64 (−3.0%). This is one seed, and the batch-64 control is itself poor. The result is uncertain.

## What remains uncertain
- Whether any ~0.695 recipe beats another. They are tied. Rung-2 groups have provenance on only 1–2 of 3 seeds (audit_v3 B-2).
- The size of the synth5 effect at 10k steps (one seed each).
- Whether the context-length gain at 2048 is real.
- Whether the search's DEV gains (c0099/c0110) transfer to VAL.
- Whether learned dynamics matter: A1 is inconclusive, and A3 tests something else.
- big2:
  - Its log holds the ensemble re-score (0.6497), not the training eval (0.6298, no surviving log).
  - Its "0.7355 on bench val" comes only from the `tta/sweep1.log` base line, where an old-suite-trained ckpt was scored on bench.
  - Status: unresolved.
- Baselines: no prov, unpinned HF revisions, untracked CSVs. They are reproducible in principle but not traced.

## Strongest VERIFIED incumbent vs best single run
- **Best single run**: syn5_10k, 0.6806 (seed 0, preliminary).
- **Strongest verified incumbent**: **syn5_only**, the curr recipe + SYN5 at 2500 steps.
  - 3-seed 0.6945±0.0082, paired ratio vs b_base256 0.948 [0.915, 0.981].
  - Seeds 0 and 1 are fully verified (clean `cce8070`, script matches, recomputed). Seed 2 is unresolved: its script was recoverable at `0db6bda`.
  - It is the only ~0.695 group with two clean seeds.
  - It is statistically tied with b_curr, so the operative incumbent is "the curr recipe at ~0.695".
- The reference b_base256 has no prov. Its seed-0 numbers are independently reproduced to 6 dp by v2_base (clean prov).
- Before any rung-3 decision, re-run the rung-2 seeds under jobq provenance (audit_v3 B-2).

## Worth continuing (after Gate 1 lifts)
1. Re-run b_curr and syn5_only s0–s2 under clean provenance, to give one fully verified incumbent.
2. Run 10k steps × 3 seeds on the curr recipe, with and without SYN5 (syn5_10k vs sc_d96_l2_10k). Steps are the only scaling lever that has shown a gain.
3. Score search finalists c0099/c0110 on VAL at 2500 steps × 3 seeds. Merge A2_DRIVE and DYN_IN=2 into one axis.
4. Confirm ctx2048 at batch 128 with 3 seeds. Hourly long-history tasks carry about 70% of the Chronos-2 gap (gap_analysis).
5. Merge c_mix as a default-off flag in train_v2.
6. Drop flow/MeanFlow, multi-scale coarse, dualdec and Muon at this budget.
