# Alma: Goal (draft v1, 2026-10-31)

## 1. Mission (your words)

> "Building the World's Most Accurate, Efficient, and Generalizable World Models for Time-Series Forecasting." (Oct 29)

> "Success means Alma, as a genuine world model for time series forecasting, beats every baseline on accuracy, interval calibration, efficiency and zero-shot generalisation, with no leakage." (Oct 29)

> "The size is flexible." (Oct 31)

> "...super efficient and super performance model of all with a significant win and is way better for finance and such usecases." (Oct 31)

**One-line goal:** build a time-series forecasting model that is a genuine world model and wins significantly on the public benchmarks against Timestral and every other leading model, while being more efficient for the accuracy it delivers.

### What the earlier sessions added that you did not ask for (removed here)
- A **0.37M-parameter cap.** You said size doesn't matter, so parameter count is no longer a goal at all; it is only reported.
- **Millisecond latency** as a goal. Latency is reported but is not a target.
- **Specific finance subdomains** (treasury cash, FX, ATM). The finance domain is open; it's decision D2 in section 8.
- **An internal 30-task bench as the scoreboard.** The scoreboard is now the public benchmarks (section 3).

## 2. What "world model" means here, and the test it must pass

**Definition:** Alma encodes the history into a latent state. A learned transition advances that state step by step (or chunk by chunk), and forecasts are read out from the simulated states.

**The test (mandatory):** compare Alma against an otherwise-identical **direct** model. It has the same encoder, parameters, data and compute, but decodes the whole horizon in one shot with no state rollout.
- Alma must win on the primary metric over at least 3 seeds.
- The paired bootstrap CI of the difference must exclude 0.
- If the direct model wins or ties, the result does not count as a world-model result.

*Context:* recent evidence (2610.34409) found that fixed rollout schedules did not beat one-shot decoding for Chronos-2. This test is the core scientific risk, so it runs early (section 7, P1).

## 3. Scoreboard: benchmarks and baselines

**Primary benchmarks (official protocols, no subsets):**
1. **fev-bench** (100 tasks). Headline: SQL skill vs Seasonal Naive. Also report WQL skill, MASE skill and win rate, with paired-bootstrap 95% CIs.
2. **GIFT-Eval** (97 configs). Headline: relative CRPS (WQL) and relative MASE, both geometric mean vs Seasonal Naive. Use the zero-shot label.

**Baselines to beat** (leaderboard values recomputed 2026-10-31):

| Model | fev SQL skill | GIFT rel. CRPS | Size |
|---|---|---|---|
| TimeRSI-1.5 (current fev #1) | 51.8 | — | ? |
| TimeH2O-v1 | 50.8 | 0.441 | ? |
| **Timestral-209M** (named target) | 50.4 | 0.446 | 209M |
| Timestral-28M | 48.4 | 0.461 | 28M |
| Timestral-5M | 46.2 | 0.479 | 5M |
| Chronos-2 | 47.3 | 0.485 | 120M |
| TiRex / TiRex-2 | 42.6 / 45.5 | 0.488 / 0.478 | 35M |
| TimesFM-3, Toto-2.0, Chronos-Bolt | see report | | |
| AutoETS, AutoTheta, Seasonal Naive | reference only (weak probabilistic baselines) | | |

## 4. Success criteria

**Parameter count is not a goal or a constraint (decided 2026-10-31).** Success is judged on accuracy, calibration, generalisation and the world-model test. Size is only reported.

**Compute is one 8 GB Mac (D1).** That is a practical limit, not a target: training much beyond about 5M params is slow on this machine. The tiers below are set by accuracy alone:

| | Accuracy | World model | Calibration |
|---|---|---|---|
| **Full goal** | Significantly beats every single model on fev-bench SQL skill **and** GIFT-Eval rel. CRPS, including Timestral-209M and the current #1 | Passes the section 2 test | Empirical 80%/95% coverage within ±3 pts; MSIS better than Timestral |
| **Strong result** | Significantly beats Chronos-2 (47.3 / 0.485) and Timestral-28M (48.4 / 0.461) on both benchmarks | Passes | Same |
| **Minimum result** | Significantly beats Timestral-5M (46.2 / 0.479) and TiRex on both benchmarks | Passes | Reported honestly |

"Significant" means the paired bootstrap 95% CI of the skill difference over tasks excludes 0, after Holm correction across the named baselines. "By a lot" is a stretch: ≥2 SQL-skill points over the model being beaten.

**What the Mac means in practice:** Timestral-209M leads Timestral-5M by 4.2 SQL-skill points, and most of that gap comes from scale. On Mac compute, the full goal needs Alma to be far better per parameter than Timestral, not just bigger. The strong result is the realistic ceiling unless D1 is reopened.

**Finance tier (secondary claim, reported separately):**
- Data: fev-bench Econ tasks, plus post-cutoff data (after 2025-11-01) for the domain chosen in D2.
- Baselines add a random walk.
- Claim: significantly better than Timestral and Chronos-2 on that tier, plus calibrated coverage.

## 5. Rules (no cheating)
- No foundation-model weights, outputs or distillation from competitors.
- No training on any fev-bench or GIFT-Eval test series. Real pretraining data must come from leakage-checked corpora (e.g. GIFT-Eval-Pretrain or the Chronos corpus minus eval overlap). Publish a leakage manifest.
- Model selection uses dev and validation only. The frozen final model is scored on test once.
- Every claimed result goes through the auditor: 3 seeds, raw and TTA numbers on the same seeds, and a ledger entry.
- Report failures and crashes using the leaderboard's own imputation policy.

## 6. Efficiency reporting (always reported, not a pass/fail criterion)
Report params, training compute, inference latency on CPU, and peak memory for Alma and every baseline.

## 7. Most feasible path

The order is by the evidence behind each step and its cost.

- **P0: honest scoreboard (1–2 days).**
  - Switch evaluation to the official fev-bench and GIFT-Eval harnesses.
  - Validate the harness by reproducing Timestral-5M's published numbers locally (small enough for the Mac). For larger baselines, use the official leaderboard result files rather than rerunning them.
  - Score the current Alma SSM on both benchmarks.
- **P1: world-model test (1 day).** Current SSM vs its direct twin, 3 seeds. Go/no-go for the framing.
- **P2: R1 levers, the highest-evidence ones first** (each on 1 seed, then 3 seeds to confirm):
  1. Computed periodicity: FFT period detector, phase channels over future steps, phase-binned template. Evidence: TinyCast, Fracast-0.
  2. Observer SSM: one transition across context and horizon, with observation correction. Evidence: 2609.33566.
  3. Data: a mixture of synthetic generators (including CauKer and OU regime-switching), plus 25–50% leakage-checked real data. Evidence: "Mix, Don't Pick", TempoPFN.
  4. Long context (2048+) with multi-scale inputs. Rollout in large chunks (≥32–48 steps) with scheduled sampling.
- **P3: scale.** Grow the model as far as the Mac allows (0.37M → 1M → 5M → more if it fits), stopping when a step stops paying. Mac rules for this step:
  - One training job at a time.
  - Mixed precision on MPS, with gradient accumulation instead of big batches.
  - Stream data from disk.
  - Checkpoint so runs can resume overnight.
  - Budget about 1–3 nights per 5M run.
- **P4: final.** Freeze the model, score test once, run the finance tier, write up.

## 8. Open decisions (yours)
- **D1, compute:** decided: **Mac only** (8 GB). Revisit if the stretch tier is hit.
- **D2, finance domain:** which use case counts as "finance"? Options: macro/FRED, FX, banking/cash, corporate treasury, or equities volatility.
- **D3, base architecture:** Alma (our SSM world model) or AlmaTS (world model inside Timestral's structure)? AlmaTS currently ties its direct control. The default is Alma, with AlmaTS ideas allowed as R1 entries.
