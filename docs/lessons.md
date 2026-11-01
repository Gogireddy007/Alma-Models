# Lessons learned (2026-10-29/30), and the process fixes they imply

Written by the lead at the start of Phase 0 (user research plan v2). Each item is a mistake we made, its cost, and the rule that prevents it now.

| # | What happened | Cost | Rule from now on |
|---|---|---|---|
| L1 | syn5_only (0.685, seed 0) was reported as the best result and its "gain" was used to plan wave 3. Over 3 seeds it was 0.6945 vs 0.6953: noise. | 3 queued long runs built on a non-result; one correction to the user | A single-seed result is "preliminary", never "best" or "winner". The autoresearch keep rule applies: the gain must exceed the 3-seed noise floor (about 0.008 wql_rel at 2500 steps). |
| L2 | The queue finished at 04:51 and the Mac sat idle until 07:20. | 2.5 h of compute | Every jobs file ends with a standby block of low-risk informative jobs (seeds of preliminary results, ablations). The lead checks the queue at every turn. |
| L3 | 56 of 122 search configs crashed: N_LAYERS=1 bug, a method dropped from train_arch2, MPS cap too tight. | about 5 h of queue, half the round | Before any batch of more than 10 jobs: preflight one config per unique value of each axis (a few steps, real batch) plus the largest-memory config. Never queue an axis value no smoke has run. |
| L4 | ctx1024 and ctx2048 trained fully and then OOMed at eval. | about 35 min of training, recovered by eval-only re-scores | Smoke tests must exercise the eval path at the job's context length, not just training. |
| L5 | A2_DRIVE turned out to be the same math as DYN_IN=2. | a duplicate search axis | New flags list the existing flags they overlap; reviewers grep READ_KEYS before adding axes. |
| L6 | The dualdec blow-up (wql 16.6, 2.3e6) sat in the leaderboard as a number. | a misleading row | Any wql_rel > 2 or non-finite is auto-labelled failed and investigated, not ranked. |
| L7 | Results were compared across protocols (old suite vs bench) and across seeds without a single ledger. | the reconciliation the user asked for in Phase 0 | runs/ledger.tsv is canonical; every claim cites ledger rows and their status label. |
| L8 | We chased the gap with many architecture tweaks before checking metric comparability and baseline provenance. | unknown, which is the point | Phase 0/1 before Phase 5: no superiority claim without a reproduced baseline under an identical protocol (Gate 2). |

## L9 (2026-10-30): screen a family's main version before its ablations
wm_flow_s0 (samp) scored 0.860 after 49 min; its direct/copy ablations would have cost another ~100 min to test the world-model
hypothesis on a model that is not competitive. Rule: run every family's abl=none first; queue ablations only for families whose
none score is within 0.02 of the best none. Ablations of a losing model answer no question we care about.
