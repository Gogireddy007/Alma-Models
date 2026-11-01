# SSM world-model test over 5 seeds (audit, 2026-11-01)

Question: does the SSM's learned rollout beat its matched direct twin (same modules, params, data, steps)?
`docs/almats_research_program.md` said the SSM family "is the first to pass the WM test" from seed 0 alone, which
was also the seed that selected the SSM family. Direct twins for seeds 1-4 had never been trained.

Method: trained `abl=none` and `abl=direct` for seeds 1-4 with the exact stored recipe of
`data/ckpt/wm_ssm_none_s0.pt` (TSAR_STEPS 2500, SYN3/SYN5, SYN_CURR 0.9, SYN_FRAC 0.3, AUG 0.5, GIFT_CLEAN 1);
re-scored the published seed-0 pair. Bench VAL, 30 tasks, wql_rel (lower is better), geometric mean over tasks.

| seed | rollout (published) | rollout (re-run) | direct (re-run) | direct − rollout | tasks where rollout better |
|---|---|---|---|---|---|
| 0 | 0.6707 | 0.6706 | 0.6926 | +0.0220 | 24 / 30 |
| 1 | 0.6936 | 0.6947* | 0.6789* | −0.0158 | 17 / 30 |
| 2 | 0.6804 | 0.6804 | 0.6746 | −0.0059 | 16 / 30 |
| 3 | 0.6808 | 0.6808 | 0.6951 | +0.0143 | 16 / 30 |
| 4 | 0.6765 | 0.6765 | 0.6861 | +0.0096 | 17 / 30 |

\* seed 1 runs were interrupted at ~1,500 steps and resumed from their checkpoints.

- The published rollout seed scores reproduce (exactly for seeds 0, 2, 3, 4).
- Rollout better on 3 of 5 seeds. Mean per-task log(direct / rollout) over 5 seeds: +0.0071 (≈0.7%),
  family-cluster bootstrap 95% CI [−0.0003, +0.0130]: **not significant**.
- Excluding the selection seed 0: +0.0008 over seeds 1-4, i.e. no effect.

**Verdict: the SSM does not pass the world-model test.** The seed-0 margin was selection luck, as the repo's own
audit_sprint §3 warned. This matches the four market projects, where the direct twin was equal or better.
Script: `audit/ssm_analysis.py` (outside the repo); logs `audit/logs/aud_ssm_*.log`.
