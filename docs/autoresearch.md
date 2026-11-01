# Autoresearch: how the team works (2026-10-31)

Goal: docs/goal.md. One loop: **small experiments → keep the best → build on the best → scale the best.**

## The loop (a tournament)
| Rung | What | Cost | Who gets in |
|---|---|---|---|
| R1 screen | 0.37M params, 2500 steps, 1 seed, bench VAL, paths + TTA inference | 10-30 min | every proposal (one factor vs the current best) |
| R2 confirm | the same with 2 more seeds | 2× R1 | beats the current best's seed-mean by ≥ 0.008 (seed noise) on seed 0 |
| R3 stack | confirmed winners combined, plus a leave-one-out of each, 2 seeds | 2-3× R1 | ≥ 2 confirmed winners, or 1 winner after 2 rounds |
| R4 scale | the stacked best at about 1M, then 4-5M params, 30-60 min each | hours | the R3 winner. Scale stops when a step stops paying |
| Final | best R4 model frozen; TEST scored once, plus the finance tier | one run | after R4 beats Timestral-5M on VAL |

- The **current best** is the reference every proposal must beat. It changes only through R2/R3 evidence.
- A round is about 6-8 R1 screens. The queue must never be idle: the manager keeps the next round queued.
- Every candidate must still pass the world-model check (rollout vs `direct`) before R4.

## The team
| Role | Owns | Proposes per round |
|---|---|---|
| **Manager (lead Claude)** | the queue, promotions, the leaderboard, merges, reports to the user, stopping bad lines | picks the round from proposals |
| World-model architect | transition and state design in `wm/ssm.py` and `wm/tswm.py` (AlmaTS), multi-scale state | ≤ 3 one-factor flags |
| Training and scaling engineer | recipe (LR, steps, batch, schedule, data mix), memory fit of 1M / 4M / 5M on 8 GB, the R4 configs | ≤ 3 |
| Calibration and inference engineer | quantile head, paths, TTA, recalibration on held-out training data, `cov80_err` | ≤ 3 (eval-only preferred) |
| Finance lead | the finance evaluation tier (leak-checked), finance-relevant ideas (volatility, regime, heavy tails) | ≤ 2 |
| Auditor | independent re-check of every promotion: leakage, fairness, re-derived numbers | veto power |

## Rules
- One heavy job at a time (M3, 8 GB). Agents run smoke tests only, through `tools/smoke.py`, which takes one lock across all agents.
- Every idea goes behind a default-off env flag, bit-identical at defaults, committed with its smoke result.
- Proposals go to the manager as jobq lines with a hypothesis and an expected effect. Only the manager queues.
- Report scores, not plumbing. A change counts only if it beats noise.
- No downloads without the user's approval. No TEST scoring before the final. No foundation-model weights or outputs.
