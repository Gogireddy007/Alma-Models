# wm_sv: crude-oil stochastic-volatility world model

## Architecture
- **Latent state:** a latent state-space model with a stochastic-volatility block. The latent level and log-volatility state are advanced one day at a time by a learned transition, and quantiles are read from the simulated paths.
- **Input:** 128 days of WTI history (WTI only).
- **Size:** d = 64, about 25,191 params per member.
- **Training:** lr 2e-3 cosine, batch 64, 600 steps. Training windows end before 2023-01-01. Ensemble of seeds 0–2; the output is the mean of the log-level quantiles.
- **Matched twin `wm_sv_direct`:** identical modules, params, data, steps and seeds, but it decodes the horizon directly with no state rollout.
- **Code:** `tools/commodity/wm_commodity.py` (registry name `wm_sv`; checkpoint shas asserted at load).
- **Checkpoints:** `checkpoints/f_sv_none_s{0,1,2}.pt` (world model) and `f_sv_direct_s{0,1,2}.pt` (twin).

## How it was tested
- **Pre-registration:** `docs/commodity_prereg.md` (A1–A3 amendments). DEV ≤ 2025-06-30, VAL to 2025-10-31, TEST from 2025-11-01, run once.
- **Targets:**
  - T1: direction at h = 1/5/20;
  - T2: volatility direction;
  - T4: probabilistic quantile score at h = 1/5/20.
- **Statistics:** Diebold–Mariano with HLN correction and overlap-robust p-values; Holm correction over competitors.
- **Competitors:**
  - rw_gauss (random walk), auto_arima, auto_ets, garch, momentum, seasonal, vol_persist, vol_meanrev, majority;
  - Chronos-2, Chronos-Bolt, TiRex, Kronos-small (on 50 origins at reduced settings);
  - zero-shot Alma (alma_s0 and its direct twin);
  - Timestral, which is contaminated and excluded from claims.
- **Audit:** `docs/audit_commodity.md`; 157/158 numbers re-derived.

## Audited TEST results
| Claim | Result |
|---|---|
| 1-day probabilistic forecast (T4 h=1) | **Rank 1 of 9** clean models; skill vs RW +0.45% |
| Significantly better than | auto_arima (Holm 0.001), chronos_bolt_base (0.0001), kronos_small (0.002) |
| h = 5 | Significantly better than kronos_small only (Holm 0.0009) |
| vs random walk | Not significant (p 0.48); negative skill at h = 5 and 20 |
| vs Chronos-2 / TiRex | Unadjusted p 0.039 / 0.045, Holm 0.19 (not significant) |
| Direction | No edge for any model |
| World-model test vs twin | Not passed; the one unadjusted signal favours the twin |

## Caveats
- Trained on WTI only. Brent, HH and inventory results are secondary and not re-derived.
- Search size: 34 search rows plus 5 references plus 6 final retrains, and the winner's curse applies.
