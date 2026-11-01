# World-model architect: round-2 R1 proposals (2026-10-31)

Reference: `ssm` none, 0.37M params, 2500 steps, scored with paths + TTA (`ref_ssm_s0` in runs/jobs/ar_r1.txt).
VAL wql_rel is 0.6804 raw as a 5-seed mean, and about 0.671 with TTA. On seed 0 the rollout passes the
world-model test: none 0.6707, direct 0.6926, copy 0.7083.

All three changes are in `wm/ssm.py`, behind default-off flags (`TSAR_WM_SSM_SPEC`, `TSAR_WM_SSM_FB`,
`TSAR_WM_SSM_LOOP`). Each is one factor against the reference and changes about 2% of the parameters or fewer.
New modules are built after every existing one, and the zero-initialised heads make SPEC and FB start as the
reference function. Each targets a different one of our measured gaps:

| proposal | gap targeted | mechanism |
|---|---|---|
| SPEC | hourly and sub-hourly seasonal tasks (M_DENSE, LOOP, bizitobs, jena, ETT: 71% of the gap) | the origin state sets the rollout's per-series spectrum |
| FB | horizons beyond 64 and the rollout's consistency with path feedback | closed-loop imagination: the next imagined step sees the model's own forecast |
| LOOP | capacity (the largest row in the gap decomposition) | weight-tied depth: 2× compute and 2× world state at +96 params |

Not proposed (overlaps): AlmaTS TRANS=ssm / READ=first already exist (docs/timestral_anatomy.md §8) and wait on the
running AlmaTS direct control. Trading state size against inner width at matched params needs no code
(`TSAR_WM_SSM_STATE=192 TSAR_WM_SSM_EXPAND=1.4`, E=136). It is a standby line, not one of the three.

---

## 1. r2_ssm_spec: system identification for imagination (`TSAR_WM_SSM_SPEC=1`)

**Hypothesis.** The rollout's dynamics are shared across all series. During imagination, layer 1's selective dt
sees only `step_emb[k] + drive(ref_k)` and nothing the encoder inferred about this series. A small head on the
origin state o_j (D→16→n_layers·2P, last layer zero-init) shifts each layer's damping and frequency for that
origin's rollout:
- decay rate: exp(a_log + dlog);
- angular rate: a_im · exp(0.5 tanh(dang)), a factor of 0.61–1.65.

The world model first identifies the system, then simulates it. The encoder scan is unchanged and |λ| < 1 still
holds. `none`, `zero` and `direct` (its one step) use it. `copy` has no transition and does not.

**Why (evidence).**
- Koopman's per-series spectrum modulation, the same idea in another family, was worth 0.016 on s0 (nomod 0.7006
  vs 0.6849 [M], docs/almats_research_program.md §3.2 item 5). It is the largest single architecture gain we have
  measured inside a world-model family.
- The gap is concentrated where per-series periods and damping differ most: hourly and sub-hourly tasks with
  daily/weekly cycles at 3, 12, 18 and 21 patches. M_DENSE, bizitobs, LOOP, jena and ETT carry 71% of the gap
  (docs/gap_analysis.md §1). The ssm's mode periods are fixed at init (2–64 patches) and shared by every series.
- It acts on the carried state, so it should widen none − direct rather than bypass the rollout.

**Expected effect.** wql_rel −0.004 to −0.012, mostly on the H / sub-hourly tasks and on chunk 1 of the long tasks.
cov80 roughly unchanged.

**Param delta.** +8,080 (365,937 → 374,017, +2.2%). MACs +0.06%.

**Cost.** About the reference: ~32 min, MPS activations +3% (probe: 757 vs 738 MB at B=64). mem=3000.

```
r2_ssm_spec_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_SPEC=1 | mem=3000
```

## 2. r2_ssm_fb: closed-loop imagination (`TSAR_WM_SSM_FB=1`)

**Hypothesis.** The rollout is open-loop. Lead k's input is `step_emb[k] + drive(ref_k)`, so the state never sees
what the model just forecast for lead k−1, including that forecast's spread. Feeding back an embedding of the
model's own quantile forecast for lead k−1 turns imagination into a proper generative simulation. The feedback is
`fb = Linear(PATCH·NQ, D)`, zero-init, input clamped to ±20, with gradients through it (BPTT over the imagined
trajectory, as in Dreamer-style imagination). The decoder head runs inside the loop. Only `none` and `zero` use it.

**Why (evidence).**
- Beyond 64 steps the harness already continues by feeding forecast paths back as observations, and that is
  where we lose most: h > 64 wql 0.76 / cov80 0.57 vs Chronos-2 0.57 / 0.74.
- The latent rollout is never trained to consume its own predictions, so the two halves of a long forecast run
  under different regimes. The literature review (§1.1) says imagination pays when the rollout is trained on its
  own predictions.
- The previous lead's quantile spread is a direct signal for how fast uncertainty should grow, which is the
  calibration gap (cov80_err 0.13).

**Expected effect.** wql_rel −0.003 to −0.010, larger on the 8 long-horizon tasks. cov80_err down 0.005–0.02.
Risk: exposure-bias feedback could destabilise early training. The zero init, clamp and grad clip 1.0 should
contain it.

**Param delta.** +7,008 (→ 372,945, +1.9%). MACs +0.4%. Direct and copy build the same parameters but do not use
them, so they stay param-matched.

**Cost.** Same MACs, with 8 small sequential head calls per rollout. Step time is within noise of the reference in
the probe. ~32–35 min, mem=3000.

```
r2_ssm_fb_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_FB=1 | mem=3000
```

## 3. r2_ssm_loop2: weight-tied depth (`TSAR_WM_SSM_LOOP=2`)

**Hypothesis.** At 0.37M the model is capacity-bound, not step-bound: 4× steps gave about 0
(docs/gap_analysis.md §4, scaling_r2 evidence table). Applying the same 2-layer mixer stack twice, with a
zero-init pass embedding, doubles depth and compute. It also doubles the world state (4 SSM state stacks, each
with its own conv buffer) at +96 params. It separates "more compute and state" from "more parameters", which is
what `r1_ssm_w1m` (1M params) measures. The two together tell R4 whether to buy depth or width.

**Why (evidence).**
- Capacity is the largest row of the gap decomposition: 0.025–0.045 (almats_research_program §2).
- Weight-tied and looped depth is the standard way to buy capacity at fixed params (Universal and looped
  Transformers).
- Counter-evidence: bigger untied models at LR 3e-3 did worse (d128 L3 0.6917, d192 L4 0.7050 at 10k steps,
  older arch). Looping keeps the parameter count, and so the LR regime, of the tuned reference.

**Expected effect.** wql_rel −0.003 to −0.012. Neutral to slightly positive on cov80.

**Param delta.** +96 (→ 366,033, +0.03%). MACs 2.0× (27.0M vs 13.8M per forecast).

**Cost.** Probe step time 2–2.5× the reference (contended machine), so ~65–80 min. MPS activations 1.93×
(1424 vs 738 MB at B=64), so about 4.8–5.0 GB driver at B=128. mem=5000. Do not run it alongside another heavy
job.

```
r2_ssm_loop2_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_LOOP=2 | mem=5000
```

Priority if only one runs: **SPEC**. It has the strongest in-house evidence, costs the same as the reference, and
targets the families that carry most of the gap. Then FB, then LOOP, which is the most expensive.

## World-model duty (conditional, only for a proposal that passes R1)

Run the `direct` ablation at the same setting. The none − direct gap (+0.022 on the reference) must not shrink.
SPEC and LOOP apply to direct as well, so the comparison is param- and MAC-matched. FB is param-matched but
unused in direct, by construction.

```
r2_ssm_spec_direct_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=direct TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_SPEC=1 | mem=3000
r2_ssm_fb_direct_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=direct TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_FB=1 | mem=3000
r2_ssm_loop2_direct_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=direct TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_SSM_LOOP=2 | mem=5000
```

## Smoke results (all through tools/smoke.py)

**Bit-identity with flags off.**
- Harness run, CPU, seed 0, 20 steps, B=16, paths + TTA, smoke tasks ETT_1H and solar_1W:
  - HEAD and the working tree give an identical loss trace and identical metrics: smoke_val_wql_rel 1.014036,
    mase_rel 1.114122, cov80_err 0.633440, with identical per-task rows. Only timings differ.
  - 34 s, 1.1–1.4 GB RSS.
- Module-level check (scratch script) on the HEAD copy vs the new file, for all four ablations:
  - identical parameter names and order;
  - identical init;
  - max output diff 0 and max weight diff 0 after 3 Adam steps.
  - Result: BIT-IDENTICAL.

**Causality.**
- Harness `wm_causal: ok` (max change 0.0) for every flag × {none, direct, copy, zero} (`TSAR_WM_INFO=1`).
- The harness check runs at init, where the SPEC and FB heads and the pass embedding are exactly zero. So a
  separate check randomised the new parameters (and drive / step_emb) and perturbed tokens and refs after patch
  20. The max change at earlier origins was 0.0 for every flag × {none, direct, zero}, and all outputs were finite.

**Training plus eval, flag on.** MPS, 10 steps, B=8, paths + TTA on ETT_1H (h=168, so path feedback runs):

| flag | params | finite | smoke wql_rel | wall | peak RSS |
|---|---|---|---|---|---|
| SPEC=1 | 374,017 | yes | 1.1528 | 14 s | 2.2 GB |
| FB=1 | 372,945 | yes | 1.1494 | 18 s | 1.3 GB |
| LOOP=2 | 366,033 | yes | 1.1621 | 24 s | 1.8 GB |

- The earlier CPU attempts at B=16 timed out at 60 s, while the machine was at load 15–27 with memory pressure
  below 25.
- The SPEC peak RSS of 2.2 GB is the MPS-mapped process. Its MPS allocation was 1.2 GB.

**Memory and step probe.** `tools/memprobe_ssm.py 1:2:64:2500:64`, MPS, two repeats:

| config | alloc peak | step time |
|---|---|---|
| off | 738 MB | 225–281 ms |
| SPEC | 757 MB | 188–276 ms |
| FB | 751 MB | 167–490 ms |
| LOOP=2 | 1424 MB | 597–619 ms |

Step times are noisy because the machine was shared. Treat LOOP as about 2–2.5× the reference wall time.
