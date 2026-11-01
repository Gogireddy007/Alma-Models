# Training and scaling engineer: round-2 recipe screens (R1) and the R4 scaling plan (2026-10-31)

Base (current best, runs/jobs/ar_r1.txt): ssm, 0.37M, 2500 steps, B=128, VAL wql_rel 5-seed mean 0.6804 (raw).
No new flags: everything below uses existing knobs (`TSAR_LR`, `TSAR_BATCH`, `TSAR_SYN_FRAC`, `TSAR_WM_WIDTH`,
`TSAR_WM_ORIGINS`, `TSAR_STEPS`). Measurement tool: `tools/memprobe_ssm.py` (emulated train step, no data load).

## Evidence used (all from runs/*.log + prov.json; older archs unless marked ssm)
| fact | numbers | source |
|---|---|---|
| more steps at 0.37M does ~nothing | 10k vs 2.5k steps: 0.6806 vs 0.6853 (syn5), 0.6851 (sc_d96_l2_10k) | syn5_10k, sc_d96_l2_10k |
| halving batch hurts a lot | B=64: 0.7104 vs B=128: 0.6853 (seed 0, same steps) | ctx256_b64 vs syn5_only |
| more synthetic hurts, monotone | SYN_FRAC 0.3 / 0.5 / 0.7: 0.6853 / 0.6997 / 0.7036 | syn5_only, syn5_frac50, syn5_frac70 |
| lower LR helped, higher LR neutral | DEV, 1200 steps: LR 1.5e-3 1.023, LR 6e-3 1.035, base 3e-3 1.041 (3-seed mean, sd ~0.005), wsd 1.041 | search_r0_c0000*/c0032/c0033/c0034 |
| bigger at LR 3e-3 did worse | 0.85M (d128 L3) 10k: 0.6917; 2.3M (d192 L4) 10k: 0.7050; 0.37M 10k: 0.6851 | sc_* |
| schedule/averaging tweaks null | wsdl+avg 0.7284, pcs+avg 0.7260 vs 0.7258 | b_wsdl, b_pcs, b_base256 |
| ssm wall time | 2500 steps = 1840-2080 s (0.74-0.83 s/step), MPS driver 3.3-3.5 GB, RSS 2.2-2.6 GB; eval with paths+TTA ~30 s | wm_ssm_none_s0/s1 (ssm) |

Reading: at 0.37M the recipe is not step-limited; it is sensitive to gradient noise (batch) and to how much
capacity goes to synthetic data, and the LR used for every size so far was tuned for none of them. The size
ladder failing at LR 3e-3 is the main warning for R4.

## Task 1: round-2 R1 proposals (one factor each)

### r2_lr15: TSAR_LR=1.5e-3 (from 3e-3)
- Hypothesis: 3e-3 is too hot for the selective-SSM recurrence (dt/A parameters are LR-sensitive; Mamba uses lower
  LR on them). Lower LR helped the closest earlier proxy by -0.018 DEV (>3 sd); higher LR did not.
- Expected: -0.003 to -0.012 wql_rel; cov80 unchanged. Risk: under-training at 2500 steps (cosine floor 5%).
- Cost: same as base, ~32 min, mem 3000. Also the first point of the LR-vs-width rule used in R4.

### r2_b256: TSAR_BATCH=256 (same 2500 steps)
- Hypothesis: gradient noise, not samples, is the limit: B=64 cost +0.025 while 4x steps gained ~0.005.
  Doubling B halves the noise at fixed steps.
- Expected: -0.005 to -0.015. If null, batch is saturated and R4 can keep B=128.
- Cost: ~2x compute, est. 50-60 min. Memory: emulated activations scale linearly with origins (~10k at B=256):
  est. 5.0-5.5 GB -> mem=5500. Only run with the queue otherwise idle (8 GB box). Fallback if memory is tight:
  same line with `TSAR_WM_ORIGINS=5000` (keeps origins/step of the base; then it is a "more series per step" test).

### r2_syn15: TSAR_SYN_FRAC=0.15 (curriculum SYN_CURR=0.9 kept)
- Hypothesis: the SYN_FRAC curve is monotone (0.7 > 0.5 > 0.3 worse to better); at 0.37M, capacity spent on
  synthetic windows costs real-data fit. The front-loaded curriculum keeps the prior's early benefit.
- Expected: -0.003 to -0.010 overall; watch weekly tasks (solar_1W, covid 1W), which the prior helps, for a
  regression. Not expected to scale: at >= 1M the prior may pay again (E8), so re-check at R4.
- Cost: as base, ~32 min, mem 3000.

Not proposed: EMA (exp21 failed; no stale-weight-free variant without code), wsd/pcs (null), more steps (null).

```
r2_lr15_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_LR=0.0015 | mem=3000
r2_b256_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_BATCH=256 | mem=5500
r2_syn15_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.15 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 | mem=3000
```
Priority if only one: r2_lr15 (cheap, strongest prior evidence, and it de-risks R4).

## Task 2: R4 scaling plan

### Parameter ladder (2 layers, width knob; counted by instantiating wm/ssm.py)
| target | TSAR_WM_WIDTH | d_model | E | params |
|---|---|---|---|---|
| 0.37M | 1.0 | 96 | 216 | 365,937 |
| 1M | 1.6667 | 160 | 360 | 998,961 |
| 2M | 2.3333 | 224 | 504 | 1,943,281 |
| 4M | 3.3333 | 320 | 720 | 3,943,441 |
| 5M | 3.75 | 360 | 816 | 4,972,017 |
| alt 4.7M deeper | 3.0, N_LAYERS=3 | 288 | 648 | 4,697,793 |

Width over depth: each extra layer adds a sequential step to the 40-patch scan and every rollout step, and the
deeper 4.7M shape used 14% more activation memory and ran 23% slower than the 5M 2-layer shape in the probe.

### Memory: measured (smoke, MPS, emulated step with fwd+bwd+AdamW, reduced batch)
| width | B / origins | alloc peak | driver |
|---|---|---|---|
| 1.0 | 16 / 500 | 153 MB | 196 MB |
| 1.0 | 32 / 1000 | 304 MB | 380 MB |
| 1.0 | 32 / 256 | 102 MB | 172 MB |
| 1.6667 | 32 / 1000 | 505 MB | 596 MB |
| 2.3333 | 32 / 1000 | 719 MB | 860 MB |
| 3.3333 | 16 / 500 | 554 MB | 652 MB |
| 3.75 | 16 / 500 | 639 MB | 744 MB |
| 3.0, L3 | 16 / 500 | 728 MB | 808 MB |
| 3.3333, MAX_H=320 | 16 / 128 | 707 MB | 1512 MB |
(smoke: exit 0, 50 s, peak RSS 413 MB.)

Activations are almost all in the K-step rollout: ~0.27 MB per origin at width 1 vs ~1 MB per batch row for the
encoder, so memory is linear in **origins**, not in batch. B=128 has ~5000 valid origins (wave7 measurement).

Extrapolation rule, calibrated on the real width-1 run (emulated 1.5 GB of activations -> 3.3-3.5 GB driver):
`job MB ~= 1300 + 1.3 x activations + 200 x params(M)` (eval and caches grow with width).

| size | origins/step at B=128 | est. activations | est. job | mem= |
|---|---|---|---|---|
| 1M | all (~5000) | 2.6 GB | 4.9 GB | 5000 |
| 1M | 2048 | 1.07 GB | 2.9 GB | 3200 |
| 2M | 2048 | 1.53 GB | 3.7 GB | 4000 |
| 2M | 1024 | 0.76 GB | 2.7 GB | 3000 |
| 4M | 1024 | 1.13 GB | 3.6 GB | 4000 |
| 4M | 2048 | 2.27 GB | 5.0 GB | 5500 (too tight with swap) |
| 5M | 1024 | 1.31 GB | 4.0 GB | 4500 |
| 4M, MAX_H=320 | 384 | ~2.1 GB | ~5 GB | 5500 (MAX_H=320 at 4M needs ORIGINS<=384) |

**Flag for the manager:** the queued `r1_ssm_w1m_s0` (width 1.67, all origins) has mem=3500, but my estimate is
~4.9 GB. Raise it to mem=5000, or expect admission into swap.

### Wall time per 1000 steps (estimate, +-40%)
The probe ran next to a live training job, so its timings are contended and used only for ratios. Model:
step = 0.4 s fixed (data + kernel launches; the scan is launch-bound) + 0.4 s x (origins/5000) x (d/96)^2,
anchored on the real 0.8 s/step at width 1.
| size | origins | s/step | min / 1000 steps |
|---|---|---|---|
| 0.37M | all | 0.8 | 13-14 (measured) |
| 1M | all | 1.5 | 25 |
| 1M | 2048 | 0.86 | 14 |
| 2M | 2048 | 1.3 | 21 |
| 2M | 1024 | 0.85 | 14 |
| 4M | 1024 | 1.3 | 22 |
| 5M | 1024 | 1.55 | 26 |
Check the `step 100 ... Ns` log line of the first job at each size and rescale the rest.

### Step budget and LR per size
- Compute-optimal guess (20 tokens/param, token = patch, 128 x 40 = 5120 patches/step): 0.37M -> ~1.5k steps (we
  run 2.5k, consistent with steps not paying), 1M -> 4k, 2M -> 8k, 4M -> 15k, 5M -> 19k.
- Over-fitting risk: the Oct-29 0.85M run peaked at ~30 min, then drifted and calibration loosened. Real TRAIN is
  finite; synthetic is infinite but at SYN_FRAC 0.3. So cap the budget below compute-optimal at >= 2M and watch the
  curve with `TSAR_EVAL_MIN` (a diagnostic curve only; the final checkpoint is what is reported, never best-of-curve).
- LR: the earlier size ladder got worse at LR 3e-3, so scale LR down with width (Adam ~ 1/d for hidden matrices,
  below the uP floor so not exact): 1M 2e-3, 2M 1.5e-3, 4-5M 1e-3. If r2_lr15 wins at 0.37M, halve each.
  Before 4M, run the 3-point LR check at 1M (1500 steps each, ~22 min each).

| size | steps | origins | LR | est. train wall |
|---|---|---|---|---|
| 1M | 5000 | 2048 | 2e-3 | ~70 min |
| 2M | 6000 | 1024 | 1.5e-3 | ~85 min |
| 4M | 8000 | 1024 | 1e-3 | ~3 h |
| 5M | 8000 | 1024 | 1e-3 | ~3.5 h |
The `r1_ssm_orig1024_s0` result says whether the origin cap costs accuracy at width 1. If it costs >= 0.008, use
2048 at 1-2M and accept B=96 at 4-5M (origins ~3800 x 96/128 -> cap 2048 fits at ~4.5 GB) instead.

### jobq lines (base recipe; the manager substitutes the R3 winner's flags)
```
# R4 LR check at 1M (1500 steps each; pick the LR, then run r4_1m)
r4_lr_1m_3e3_s0 | train_wm.py | TSAR_STEPS=1500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=1.6667 TSAR_WM_ORIGINS=2048 TSAR_LR=0.003 | mem=3200
r4_lr_1m_2e3_s0 | train_wm.py | TSAR_STEPS=1500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=1.6667 TSAR_WM_ORIGINS=2048 TSAR_LR=0.002 | mem=3200
r4_lr_1m_1e3_s0 | train_wm.py | TSAR_STEPS=1500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=1.6667 TSAR_WM_ORIGINS=2048 TSAR_LR=0.001 | mem=3200
# R4 ladder (one size at a time; stop when a step stops paying >= 0.008)
r4_1m_s0 | train_wm.py | TSAR_STEPS=5000 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=1.6667 TSAR_WM_ORIGINS=2048 TSAR_LR=0.002 TSAR_EVAL_MIN=20 | mem=3200
r4_2m_s0 | train_wm.py | TSAR_STEPS=6000 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=2.3333 TSAR_WM_ORIGINS=1024 TSAR_LR=0.0015 TSAR_EVAL_MIN=20 | mem=3000
r4_4m_s0 | train_wm.py | TSAR_STEPS=8000 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=3.3333 TSAR_WM_ORIGINS=1024 TSAR_LR=0.001 TSAR_EVAL_MIN=30 | mem=4000
r4_5m_s0 | train_wm.py | TSAR_STEPS=8000 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=3.75 TSAR_WM_ORIGINS=1024 TSAR_LR=0.001 TSAR_EVAL_MIN=30 | mem=4500
# World-model check at the winning size (required before Final): same line with TSAR_WM_ABL=direct, e.g.
r4_4m_direct_s0 | train_wm.py | TSAR_STEPS=8000 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=direct TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_WIDTH=3.3333 TSAR_WM_ORIGINS=1024 TSAR_LR=0.001 TSAR_EVAL_MIN=30 | mem=4000
```
Notes:
- `TSAR_EVAL_MIN` pauses the training clock during each eval, so the eval cost (est. 1-4 min at 4-5M) adds to wall
  time. It is a training-only key and does not change the weights. Drop it if the manager prefers.
- The R4 lines change size, LR and steps together on purpose (the ladder is a recipe per size, not a one-factor
  screen). The 1M LR check isolates the LR part.
- Kill rule: if 1M at its best LR is not >= 0.008 better than the 0.37M seed mean, do not run 4-5M with this recipe;
  look at origins/batch first.
