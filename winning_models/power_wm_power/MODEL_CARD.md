# wm_power: German day-ahead electricity world model

## Architecture
- **Physics-informed world model.** A latent power-system state is rolled forward hour by hour for 48 h, starting from the end of D-2.
- **Heads:**
  - a residual-load head;
  - a **monotone merit-order readout**, in which price rises monotonically with residual load;
  - 3-regime offsets.
- **Training:**
  - architecture: d = 64, 14 days of context, auxiliary loss weight 0.3;
  - optimiser: lr 1e-3 OneCycle, batch 64, 1,500 steps;
  - ensemble: seeds 0, 1, 2, averaged by quantile.
- **Matched twin `wm_power_direct`:** same modules, params, data, steps and seeds, but every hour is a single step of the same GRUCell from the origin state, with no carried state.
- **Code:** `tools/power/wm_power.py`, with `wm_models.py` and `wm_data.py`.
- **Checkpoints:** `checkpoints/frz_phys_none_s{0,1,2}.pt` and `frz_phys_direct_s{0,1,2}.pt`.

## How it was tested
- **Pre-registration:** `docs/power_prereg.md`. TEST covered 365 days from 2025-11-01 and was run once offline after auditor GO.
- **Endpoints:**

  | Endpoint | Type | What is predicted |
  |---|---|---|
  | P1 | primary | daily base-price direction |
  | P2 | primary | hourly direction, every slot |
  | P3 | secondary | peak/off-peak spread direction |
  | P4 | primary | quantile score of the 24 prices, measured as skill vs naive_epf |

- **Competitors:**
  - EPF baselines: LEAR, LEAR without covariates, ARX;
  - naive baselines: naive_d, naive_w, naive_epf, majority, dow_majority;
  - foundation models: Chronos-2, Chronos-2 with covariates, Chronos-Bolt, TiRex, Kronos-small;
  - zero-shot Alma (alma_s0 and its direct twin);
  - Timestral, which is contaminated.
- **Audit:** `docs/audit_power.md`, with 243/243 numbers re-derived.

## Audited TEST results
| Endpoint | wm_power | Best competitor | Verdict |
|---|---|---|---|
| P2 hourly direction | **0.747 (#1)** | LEAR / ARX close behind | No significant win over LEAR, ARX or the foundation models |
| P1 daily direction | 0.732 | lear_nocov 0.742, arx 0.740, lear 0.737 | Behind three models |
| P3 spread direction | 0.732 (#2) | n/a | n/a |
| P4 quantile score | 9.02 (#4) | Chronos-2 8.67 | Behind |
| Significant wins | n/a | n/a | Beats the naive baselines, Kronos and zero-shot Alma |
| World-model test (P2 vs twin) | raw p 0.0168 | n/a | Fails after Holm correction |

## Caveats
- **Intervals under-cover:** cov80 is 0.752.
- **Regime shift:** prices doubled and 15-minute products were introduced during the test period. TEST-month means were seen.
- **Search size:** 49 runs over 33 recipes.
- **Timestamp correction:** the freeze-time labels needed a dated correction (`docs/power_search.md` §6). The auditor ruled the freeze genuine.
