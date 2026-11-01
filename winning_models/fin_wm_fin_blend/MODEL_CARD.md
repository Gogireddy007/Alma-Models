# wm_fin_blend: crypto-volatility blend (audit follow-up, 2026-11-01)

## What it is
- Quantile blend of the frozen `wm_fin` (RSSM world model, `winning_models/fin_wm_fin/`) and the pre-registered
  `har_intra_dow` baseline: `q = sort(0.6 q_wm_fin + 0.4 q_har_intra_dow)` over the 13 log-RV quantiles.
- `p_up` = probability that log RV(D) exceeds the label reference (mean log RV of D-7..D-1), read off the blended
  quantile function, so `p_up > 0.5` exactly when the blended median is above the reference.
- No new weights or checkpoints. **Code:** `tools/fin/wm_fin_blend.py` (registers `wm_fin_blend`). Crypto only.

## How the weight was chosen
On VAL (2025-07-01 .. 2025-10-31) only, weights 0..1 in steps of 0.1, criterion `acc − 2·QS` fixed before the TEST
look: 0.4 (VAL accuracy 0.737 → 0.752, QS 0.1543 → 0.1540).

## TEST (24 perps, 2025-11-01 .. 2026-10-30)
| model | P1 acc all / train / held-out | P3 QS all / train / held-out |
|---|---|---|
| wm_fin | 0.728 / 0.728 / 0.728 | 0.1471 / 0.1459 / 0.1496 |
| har_intra_dow (median rule) | 0.731 / 0.731 / 0.731 | 0.1541 / 0.1526 / 0.1569 |
| **wm_fin_blend** | **0.740 / 0.739 / 0.741** | **0.1459 / 0.1446 / 0.1483** |

7-day block bootstrap over days (24 coins pooled per day):
- vs wm_fin (median rule 0.724): +1.7 pp, CI [+0.5, +2.9], significant; QS −0.0014, CI [−0.0034, +0.0007].
- vs har_intra_dow: +0.9 pp, CI [−0.6, +2.3], **not significant**; QS −0.008, CI [−0.011, −0.006], significant.
- Robust to the weight: 0.3 / 0.4 / 0.5 give +0.8 / +0.9 / +1.1 pp over har_intra_dow (all reported).

## Caveats
- TEST is a second look (already scored by the original project and in the audit).
- VAL is only 4 months; the weight is coarse.
- The direction edge over the best simple baseline is not established.
