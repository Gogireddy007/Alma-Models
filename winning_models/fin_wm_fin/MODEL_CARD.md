# wm_fin: Alma-Fin crypto-volatility world model

## Architecture
- **Core model:** an RSSM (recurrent state-space model with a stochastic latent).
  - d = 64, z = 16;
  - **four 6-hour prior sub-steps per day**;
  - window W = 14 days of daily features plus hourly features;
  - a weekday-of-D input.
- **Training:**
  - lr 2e-3, 800 steps;
  - 3-seed ensemble refit on labels ≤ 2025-06-30 for the 16 training coins.
- **Matched twin `wm_direct`:** identical, except the heads read the D-1 filtered state, with no rollout.
- **Other frozen pillars, in the main repo, not in this folder:**
  - `wm_fin_cl`: continual learning; SGD with 1 step per day, lr 1e-4, L2-to-init 1.0, 60-day replay.
  - `wm_agent`: a Dreamer-style actor-critic trained in imagination.
  - PPO and DQN twins, plus a vol-target control.
- **Code:**
  - `tools/fin/wm_fin.py`;
  - `tools/fin/wm_core.py`;
  - agents in `tools/fin/agents/`;
  - interface in `tools/fin/README.md`.
- **Checkpoints:** `checkpoints/frozen_wm_fin.pt` and `frozen_wm_direct.pt`.

## How it was tested
- **Pre-registration:** `docs/fin_prereg.md`, amendments A1–A5. A5 added calendar-aware baselines after a red-team finding.
- **Universe:**
  - Binance USDT-M perpetuals: 16 training coins and 8 held-out coins (SOL, XRP, 1000PEPE, AVAX, TAO, DOT, BCH, 1000BONK);
  - 13 US ETFs for zero-shot evaluation.
- **Execution timing:** decide at 00:00 UTC, fill at 00:05–00:10.
- **Targets:**
  - **P1:** will next-day 5-minute realized volatility be above its trailing 7-day log mean?
  - **P3:** quantile score.
  - **P2:** net-of-cost Sharpe of the policy on held-out coins, with PSR and DSR at K = 985 evaluated rows.
- **Nested selection:** the autoresearch loop optimised on DEV-early folds E1–E3 only. The DEV holdout F1 was looked at once per finalist. VAL was run once after the freeze. TEST was run once, offline, after auditor GO.
- **Competitors:**
  - HAR, HARX, HAR-roll, GARCH, rv_persist;
  - LightGBM (pooled and per-asset);
  - **calendar-aware har_dow, lgbm_dow and har_intra_dow**;
  - Chronos-2, Chronos-Bolt, TiRex, Kronos-small;
  - Timestral (contaminated);
  - for trading: PPO, DQN, buy-hold, time-series momentum and mean-reversion rules, and an LGBM policy.
- **Leak and integrity checks:**
  - **Red-team** (`docs/fin_redteam.md`): 16 findings, with no leak found.
  - **Audit** (`docs/audit_fin.md`): 2,146 of 2,154 TEST numbers re-derived exactly; the other 8 are identical-model p-value conventions.

## Audited TEST results (24 coins)
| Pillar | Result | Verdict |
|---|---|---|
| P3 quantile score vs foundation models | 0.147 vs Chronos-2 0.156, TiRex 0.158, Chronos-Bolt 0.160 | **Significant** (Holm 0.023 / 0.002 / 0.0002), and vs Kronos |
| Portability to held-out coins | +8.0 pp vs lgbm_asset, +9.2 pp vs har_roll (Holm < 1e-4); but 0.728 vs har_intra_dow 0.731 on the same held-out coins | **Not met against the best baseline**: the 8–9 pp gap is against baselines without a weekday input |
| P1 vs best registered baseline | 0.727 vs har_intra_dow 0.731 | Not met (p 0.70) |
| P1 vs foundation models | +1.0 to +1.9 pp | Not significant |
| Rollout vs direct twin | +0.27 pp (p 0.30) | Not met |
| Continual vs frozen | p 0.75 | Not met |
| WM-RL agent vs PPO/DQN (held-out Sharpe) | −0.52 vs PPO −0.27, DQN +1.05 | Not met; DQN is better (p 0.025) |
| ETF zero-shot | 0.583 vs ~0.60 | Not met |

## Caveats
- The weekday input was added after the red team found a large weekday effect.
- The freeze gate was set while the competing DEV scores were already known.
- F1 numbers are post-selection.
- The universe was ranked on VAL-period volume, so DEV and VAL carry selection bias. TEST does not.
- Held-out coins were used in selection, so the portability result is not strictly zero-shot.
