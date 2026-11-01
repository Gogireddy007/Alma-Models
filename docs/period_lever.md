# Period lever: detected periods as coordinates, plus phase binning (2026-10-31)

Why: most of the remaining gap to Chronos-2 and Timestral comes from hourly and sub-hourly seasonal series
(docs/gap_analysis.md: H and sub-hourly tasks carry 61% of the gap, and seasonal tasks carry 94%). The evidence is
from TinyCast (arXiv 2608.15767) and Fracast-0 (2609.32209):
- A parameter-free period detector, feeding phase coordinates at both context and future positions, helps.
- Phase binning into a cycle template gave the largest single ablation gain.
- A wrong period hurt more than no period.
- Feeding seasonal-naive values as inputs hurt: periods help as coordinates, not as values.

Both flags are off by default. With both off the code is bit-identical (verification below).

## Design

**Detector.** The detector is `wm/ssm.py: detect_periods`, a pure function called by the harness. It runs on a
window of 256 points ending at every origin, padded on the left with NaN. At inference that window is exactly the
context, and in training it is the matching slice, so the two see the same thing. The steps, per window:
1. Centre, scale, apply a weighted linear detrend, standardise, clip at ±5 sd, and zero the missing points.
2. Compute the periodogram with one DFT matmul. Whiten it by an AR(1) spectrum fitted from the lag-1
   autocorrelation, so the null hypothesis is red noise, not white noise.
3. Run Fisher's g-test sequentially on the whitened periodogram. The p-value is m(1-g)^(m-1), with m the effective
   number of bins. A peak is accepted if p < 0.01/4 (Bonferroni over 4 slots), and the first peak that fails stops
   the search.
4. Only periods with **at least 3 observed cycles** are candidates (the minimum-cycle null).
5. Pick the integer period with a **comb autocorrelation** C(P), the mean ACF at P, 2P, 3P and so on up to lag 128.
   It looks inside the peak bin's period range, and long lags make it sharp to about ±1 point.
6. Upgrade to a multiple mP only if C(mP) > C(P) + 0.2, C(mP) > 2.5·C(P) and C(P) ≥ 0.05. This handles a profile
   whose strongest line is a harmonic.
7. **Coherence:** require C(P) ≥ 0.2. A fixed-phase cycle keeps its ACF at every multiple, and a drifting
   quasi-cycle does not.
8. Reject a period that is still rising at the longest admissible period. This is the case where the true period is
   beyond the 3-cycle limit, for example 96 in a 256-point window.
9. Duplicates and harmonics of an accepted period take no slot.

The output is up to K=4 integer periods per origin.

**TSAR_PERIOD=1 (`train_wm.add_period_features`).** The harness attaches three things to the Context:
- `ctx.periods [B,n,4]`: the detected periods.
- `ctx.tok_phase [B,n,16]`: phase features at each context patch's start, using that patch's own detection, so it
  stays causal.
- `ctx.fut_phase [B,n,K,16]`: the same features at every lead patch start of origin j, using origin j's periods.

Each period slot has 4 features: sin, cos, an active flag, and log p / log 256. In ssm, one shared zero-init
`Linear(16, D)` (`ph_emb`) embeds both the context and the future features. They are added to the token embedding in
encode and to every lead input u_k in decode, so the rollout has a clock that continues the context's. The inputs
under every ablation (none, direct, copy and zero) include it, so the direct twin gets the same clock and the
world-model test stays fair. Cost: +1,632 params.

**TSAR_PERIOD_BIN=1 (needs PERIOD; `SSMFamily.template`).** This builds a cycle template per origin:
- Take the encoder outputs o_t of the last 32 patches t ≤ j, weighted by each patch's observed share.
- Average them into 16 phase bins for each detected period. The phase is taken at the patch start.
- Each lead patch reads the bin at its own phase, and the reads are summed over the active periods.
- Two zero-init projections carry the result: `tmpl_in` into u_k and `tmpl_out` into the decoder-head input, under
  every ablation.

It bins encoder states, not raw values, following the "coordinates, not values" finding. Cost: +18,624 params (total
386,193 vs 365,937). The direct and none variants have identical parameter counts and MACs.

**Not implemented: TSAR_OBS (observer SSM).** The ssm encoder already uses the same mixers for the context and the
rollout. An observation-as-correction rewrite would serialise the layer stack per patch, roughly doubling encode
time. I left it out to keep tonight's queue cheap and the screen clean.

## Unit check (`uv run tools/period_check.py`, CPU, about 3 s, 400 series of 256 points each)

"wrong" means a period fired that is neither the true one nor a multiple of it.

| series | hit | fired | wrong |
|---|---|---|---|
| sine p=24, amp/noise 1 | 0.985 | 0.985 | 0 |
| sine p=12 / p=7, amp/noise 1 | 0.998 / 0.993 | | 0 |
| sine p=48 / p=60, amp/noise 1 | 0.735 / 0.585 | 0.99 / 1.0 | 0.145 / 0.320 (mostly off by 1-2 points at 4-5 cycles) |
| sine p=96 (2.7 cycles, below the 3-cycle limit) | 0 | 0 | 0 (not detectable at CONTEXT 256, by design) |
| sine p=24, amp/noise 0.5 / 0.3 | 0.02 / 0 | | 0 (conservative) |
| random 24-point profile + AR(0.7) | 0.573 | 0.688 | 0.058 |
| sine 24 + AR(0.9) + trend | 0.542 | 0.547 | 0 |
| sine 24 + sine 7: both found | 0.975 | | 0.025 |
| sine 24, 20% missing | 1.0 | | 0 |
| sine 24, only the last 100 points observed (4 cycles, ACF too short) | 0 | 0 | 0 |

False-fire rate on series with no period:

| series | false-fire rate |
|---|---|
| white noise | 0 |
| AR(1), phi = 0.5, 0.9 and 0.97 | 0 / 0 / 0 |
| random walk | 0 |
| linear trend | 0 |
| exponential growth | 0 |
| Poisson spikes | 0 |
| level shift | 0 |
| AR(2) quasi-cycle with phi = (1.6, -0.8) | 0.245 (a stochastic cycle with a 13-14 point period; partly accepted) |

**Causality probe** (audit_sprint item 1, at the end of `tools/period_check.py`). I perturbed every point after
patch c (c = 10, 20, 33 and 39) in 64 training-length windows of 320 points. Eight of them had 120 points of left
padding. The periods, context phases and lead phases at origins j < c did not change at all (max change 0.0, with
3,518 detected slots in the unchanged origins), while origins ≥ c did change. The harness `wm_causal` check also
passes with the flags on: max change 0.0 at 40 early origins for none, direct, copy, and FB.

## Bit-identity and smoke tests

- **`tools/audit_sprint_probe.py ssmbit`.** The default-env SSM in the working tree has identical init and
  max|dq| = 0 versus 9c572d5 and HEAD, for none, direct, copy and zero.
- **`TSAR_WM_INFO=1`, flags off.** HEAD (from `git archive`) and the working tree both print params=365937,
  macs=13765632 and wm_causal 0.0.
- **3-step CPU training smoke** (sprint recipe, `TSAR_BENCH_MAXLEN=512`, `TSAR_WM_EVAL_TASKS=ETT_1H`):

| run | step-0 loss | smoke wql_rel / mase_rel / cov80_err | peak RSS |
|---|---|---|---|
| HEAD, flags off | 0.7147 | 1.256466 / 1.407831 / 0.716709 | |
| working tree, flags off | 0.7147 | identical to all 6 digits | 1.09 GB |
| PERIOD + BIN, none | 0.7147 (zero-init) | 1.249034 / 1.394957 / 0.715519 | 1.30 GB |
| PERIOD + BIN, direct | | 1.246213 / 1.400 / 0.704 | 1.38 GB |

The RSS is mostly the TRAIN data load, and the smoke was short. Detection costs about 0.13 s on CPU for one training
step's 5,120 windows. On MPS it is one DFT matmul, two rFFTs and small elementwise loops.

## Queued (runs/jobs/sprint.txt, after wm_ssm_direct_s1 and s2)

These use the recipe of wm_ssm_none_s0 plus the flags:
- `wm_ssm_per_s0` with TSAR_PERIOD=1.
- `wm_ssm_perbin_s0` with TSAR_PERIOD=1 TSAR_PERIOD_BIN=1.

The baseline is family_wql_rel 0.6787 (seed 0). Promotion needs an improvement of at least 0.008, then seeds 3 and 4.
If the lever is promoted, the world-model claim also needs the matching direct twin (`TSAR_WM_ABL=direct` with the
same flags). The direct twin gets the same features by construction.
