# wm_power_blend: German day-ahead power blend (audit follow-up, 2026-11-01)

## What it is
- Equal quantile blend of the frozen `wm_power` (`winning_models/power_wm_power/`) and the pre-registered LEAR
  competitor: `q = sort(0.5 q_wm_power + 0.5 q_lear)` per hourly slot.
- Both members see only the harness information set (prices <= D-1, fundamentals <= D-2).
- No new checkpoints. **Code:** `tools/power/wm_power_blend.py` (registers `wm_power_blend`). LEAR refits a lasso
  per slot and day, so a year of forecasts takes about 12 minutes.

## How the weight was chosen
On VAL (2025-07-01 .. 2025-10-31) only, weights 0..1 in steps of 0.1, criterion `P2 − QS/100` fixed before the TEST
look: 0.5 (VAL P2 0.732 → 0.756, QS 7.32 → 6.84).

## TEST (delivery days 2025-11-01 .. 2026-11-01)
| model | P1 daily | P2 hourly | QS (EUR/MWh) | MAE |
|---|---|---|---|---|
| wm_power | 0.732 | 0.746 | 9.03 | 22.44 |
| lear | 0.737 | 0.730 | 9.22 | 22.89 |
| **wm_power_blend** | **0.751** | **0.755** | **8.60** | **21.47** |

7-day block bootstrap over days:
- QS vs wm_power −0.43, CI [−0.67, −0.18]; vs LEAR −0.61, CI [−0.89, −0.36]: significant against both.
- P2 vs wm_power +0.9 pp, CI [−0.3, +2.0]: **not significant**; vs LEAR +2.5 pp, CI [+1.3, +3.6]: significant.

## Caveats
- TEST is a second look (already scored by the original project and in the audit).
- Not compared here against Chronos-2/TiRex (not installed in the audit environment); Chronos-2's TEST P2 was 0.740.
