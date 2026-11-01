# Alma: the goal (set by the user, 2026-10-31)

**Build the best time-series forecaster there is, and make it a genuine world model.** Alma must:

- **Beat every benchmark and every model** with a **significant** win. That covers Chronos-2, Timestral 5M/28M/209M, TiRex, Chronos-Bolt, AutoETS, AutoTheta and Seasonal Naive.
- Win on all four axes: accuracy (`wql_rel`, `mase_rel`), interval calibration (`cov80_err`), efficiency (parameters, predict time, memory) and zero-shot generalisation. It should be **super efficient** as well as best-performing.
- Be **way better for finance** and similar use cases. Treasury and cash forecasting, banking, FX, macro and business-finance series get their own evaluation tier.
- Be a real world model: it simulates the future step by step from a learned state, and the simulation is what makes it accurate. The check is that the rollout beats the same model with the simulation removed (`direct`).
- Have **no leakage.** No foundation-model weights or outputs, no training on VAL or TEST series, and `prepare.py`/`baselines.py` stay untouched. TEST is scored once, at the end, with a frozen model.

## Scoreboard (bench VAL, lower is better)
| model | wql_rel | cov80_err | params |
|---|---|---|---|
| Chronos-2 | 0.559 | 0.068 | 120M |
| Timestral-209M | 0.564 | 0.034 | 209M |
| Timestral-28M | 0.572 | 0.036 | 28M |
| Chronos-Bolt base | 0.582 | 0.058 | 205M |
| TiRex | 0.582 | 0.048 | 35M |
| Timestral-5M | 0.585 | 0.047 | 5M |
| Alma: ssm world model, 5-seed mean (raw) | 0.680 | 0.13 | 0.37M |
| Alma: ssm, seeds 0-2, raw → +TTA (same seeds) | 0.6815 → 0.6712 | 0.134 → 0.130 | 0.37M |
| Alma: AlmaTS (Timestral backbone + WM, FRAME=origin), 1 seed, WM test pending | 0.681 | 0.13 | 0.37M |
| AutoTheta / Seasonal naive / AutoETS | 0.959 / 0.988 / 2.68 | | |

The milestones, in order:
1. Beat the statistical baselines. **Done.**
2. Beat Timestral-5M at ≤5M parameters.
3. Beat Chronos-2 and Timestral-209M on every axis.
4. A significant win: beat the best baseline by more than seed noise and family-level CIs, on both VAL and the one-shot TEST, plus the finance tier.

How we get there is in docs/autoresearch.md.
