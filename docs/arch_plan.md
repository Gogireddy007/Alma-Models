# Architecture and training plan: closing the gap to Timestral

Date: 2026-10-29. Branch `autoresearch/oct29`. Author: modeling research (planning only. No training was
run and `train.py` was not edited while this was written).

## 0. Where we stand

| model | val wql_rel | note |
|---|---|---|
| ours, 4 min (exp24, 0.37M) | 0.680 | in-domain trained |
| ours, big2 60 min (0.85M) | 0.614 peak at 30 min / **0.630 final** | overfits after ~30 min, cov err 0.05 -> 0.073 |
| Timestral-5M / 28M / 209M | 0.591 / 0.577 / 0.569 | zero-shot |
| TiRex / Chronos-2 | 0.603 / 0.621 | zero-shot |

To beat Timestral-209M we need about 11% lower wql than big2 final. Per-dataset ratios (big2 eval / Timestral-209M;
the per-dataset rows available are from the big2 *ensemble* eval, so treat them as approximate):

| dataset | ours | T-209M | ratio | likely cause |
|---|---|---|---|---|
| covid_deaths | 0.315 | 0.186 | 1.70 | trend extrapolation, only ~200 points per series |
| ercot | 0.70 (0.46 single fwd) | 0.473 | ~1.0-1.5 | ensemble sign-flip broke it (cov80 0.255) |
| traffic (H, m=24) | 0.789 | 0.613 | 1.29 | weekly cycle (168) barely fits in a 256 context |
| nn5 (D, m=7, h=56) | 0.760 | 0.631 | 1.21 | long horizon, holiday/annual effects, short context |
| tourism_monthly / quarterly | 0.783 / 0.768 | 0.652 / 0.663 | 1.20 / 1.16 | noisy single-season copy, short series |
| hospital (counts, m=12) | 0.856 | 0.741 | 1.16 | count noise, seasonal averaging |
| weather (D) | 0.567 | 0.501 | 1.13 | annual cycle outside context |
| m3/m4 monthly+quarterly | 0.62-0.67 | 0.57-0.61 | 1.08-1.09 | generic prior quality |
| cif, m1_monthly | 0.459 / 0.726 | 0.560 / 0.803 | 0.82 / 0.90 | we already win |

On the four ZS sets, Timestral-209M scores a geometric mean of 0.627. We score 0.748 even though we **train on them**.
So the gap is not caused by in-domain luck. It comes from the prior (data diversity), from context length and from
how the decoder reads the context.

Three facts from our own log shape the ranking:
1. Compute is not the limiter at this size (P1: 4x compute bought 1.6%). Scale helped only after the normalisation
   fix (big1 -> big2). The model then **memorises** dev windows after ~30 min. Data diversity and regularisation come first.
2. The 4-minute val loop is biased against generalisation changes. KernelSynth 20% (exp20), 50% synthetic (exp13) and
   TSMixup (exp18) all lost at 4 min. That is the regime where fitting the in-domain windows fastest wins. These ideas
   must be judged at 30-60 min and on the `TSAR_ZS=1` family holdout, not in the 4-min loop.
3. A1 showed that rollout and a direct head are within noise. The world model's single latent vector z_t is a
   bottleneck: everything the decoder knows about the context must pass through 96-128 numbers.

## 1. What the leaders do (evidence base)

- **Timestral** (Mistral, arXiv 2610.37585). Causal patch Transformer (P=32), RMSNorm, SwiGLU, sigmoid-gated
  attention, RoPE on half of each head's dims, no dropout, 21 quantiles with a monotone head, pinball loss only.
  **Decoupled Horizon Forecasting (DHF)**: separate prediction tokens that attend to all context tokens before
  their block. Many origins and horizons per pass, no context masking. **Pretrained exclusively on synthetic data**
  (8k context), then post-trained on real+synthetic at a 32k context. Timestral-5M is d=192, 8 blocks. The report
  has no ablations.
- **TiRex** (2505.23719). sLSTM recurrent state beats a Transformer and mLSTM at equal data. CPM is needed for
  multi-patch forecasting in a recurrent decoder. Augmentations (amplitude modulation, censoring, spike injection)
  each help, and removing all of them hurts most. About 1/3 of training series come from an extended KernelSynth.
  Context 2048.
- **Toto 2.0** (2605.20119). Pretraining uses *no public data*: 57.5% TempoPFN-style synthetic plus internal metrics.
  An HP sweep at proxy scale chose to exclude public data from pretraining, and fine-tuning then uses 45% public.
  WSD schedule with a short linear decay (1.7-2.6% of steps). Toto-4M saw 3.4T points.
- **TempoPFN** (2510.25502). Synthetic only, GatedDeltaProduct linear RNN. Leave-one-out generator ablation
  (GIFT CRPS 0.577 base): removing the **OU/regime-switching SDE** generator gives +26% (0.729). Removing CauKer
  gives +14%, Spike +12%, Step +11%, ForecastPFN +7%, KernelSynth +6%. Augmentation pipeline: -5.4% CRPS.
- **Reverso** (2602.17634). 0.2-2.6M params, long-conv + DeltaNet hybrid, MASE 0.706 on GIFT with 2.6M params
  (TiRex 0.716). Ablations: no synthetic data 0.786 vs 0.738 (+6.5%), no augmentation 0.755 (+2.3%), **multi-scale
  downsampled input channels** 0.711 -> 0.706 overall and 0.749 -> 0.741 on long horizons, **attention decoder head**
  0.740 -> 0.725 (-2%), DeltaNet or long-conv beat attention (0.732 vs 0.750-0.762). Training took 10-40 H100-hours.
- **Fracast-0 / TinyCast** (2609.32209, 2608.15767). 85K/147K params, GIFT WQL 0.564/0.545. A parameter-free
  periodogram detector, sin/cos channels for the detected periods, an explicit *seasonal future state* in the
  decoder, and a "commit" loss that penalises a median worse than the seasonal copy. The weak spots are medium and
  long horizons.
- **EIDOS** (2602.14024). A latent-prediction (JEPA-style) loss plus a grounding loss plus the forecast loss gives
  -2.3% CRPS and -1.4% MASE on GIFT, mostly at medium and long horizons. Latent loss *without* grounding collapses
  and does not help.
- **Chronos-2** (2510.15821). A synthetic-only variant is only slightly worse than the full model. Group attention is
  for multivariate data and is irrelevant here (all our tasks are univariate).
- **TTM** (2401.03955). About 1M params beats much larger models through multi-resolution *downsampling
  augmentation* of real series plus adaptive patching. GIFT MASE 0.756 (fine-tuned).

## 2. Ranked plan

Ranking = expected gain on val *and* on the holdout, divided by cost and risk. Each item is one env-flagged change
in `train.py`, default off, so the current best is reproducible. Judge items 1, 3 and 5 at 30-60 min and with
`TSAR_ZS=1` as well as on plain val.

---

### 1. Synthetic-first curriculum with a TempoPFN-class GPU generator bank plus real-series augmentation (T5/T6)

**Mechanism.** In PFN terms the training distribution *is* the prior. Our prior is 70% of about 15x2000 short dev
series, reused thousands of times, plus one hand-written generator. Both leaders rely on synthetic data: Timestral
pretrains only on synthetic data, and Toto 2.0 pretrains with no public data. This change does two things:
(a) it widens the prior with the generators that ablate as most important, and (b) it turns the late phase from
"memorise dev windows" into "fine-tune a synthetic prior on real data".

**Evidence.** TempoPFN generator ablation: SDE +26%, CauKer +14%, Step/Spike +11-12% when removed, and the
augmentations are worth -5.4%. Reverso: no synthetic +6.5% MASE, no augmentation +2.3%. TiRex: each augmentation
helps. Chronos-2-synth comes close to Chronos-2. Toto's sweep removed public data from pretraining.

**Expected effect.**
- Arrests the post-30-min drift (the main lever to keep 0.614-level accuracy, about -2.5% vs final).
- Helps covid (logistic/SDE trends), weather and hospital (heteroscedastic OU, counts), cif, and above all the holdout
  domains (finance = SDE/GBM, intermittent = count generators, macro = trends with breaks).
- Calibration: cov err should return toward 0.04-0.05.

**Implementation sketch.**
- Replace the per-series Python loop in `synthetic2` (lines 143-198) with a vectorised torch `synth_bank(n,
  device)` that generates directly on MPS. The current `synthetic2` loops per series in Python (line 151) and would
  dominate step time at a high synthetic share. Generators, as columns of one [n, WINDOW] tensor with a random
  generator id per row:
  1. Regime-switching OU SDE: Euler step, 2-3 regimes with Markov switching of mean, reversion speed and
     volatility, solved with `cumsum` per regime segment or a 320-step loop over the batch.
  2. GP / KernelSynth: precompute Cholesky factors of about 64 composite kernels (RBF, periodic with periods
     {4,7,12,24,52,168,...}/320, linear, RQ, Matérn) at startup. Each factor is 320x320, about 26 MB total in
     fp32. Sample with `L @ eps`.
  3. CauKer-lite: a random 2-3 node DAG of GP draws through random activations (tanh, relu, x², sin) with
     additive mixing.
  4. ForecastPFN-style multiplicative trend x calendar seasonality (weekly within daily, yearly within monthly).
  5. Sawtooth / asymmetric period, step changes, spikes (top-hat, RBF), stochastic rhythm (random-amplitude periodic
     bursts).
  6. Intermittent and count (Poisson / negative binomial with a seasonal rate).
  7. Keep the existing `synthetic` (lines 104-140) as one more generator.
- Add `augment(win)` for **real** rows in `sample_batch` (lines 229-247):
  - amplitude modulation (multiply by a piecewise-linear positive curve)
  - censoring (clip at a random quantile)
  - spike injection
  - **temporal aggregation**: for series with ≥ 2x WINDOW points, take a long slice and sum or mean over k ∈
    {2,3,4,7,12,24} before cropping. TTM/Reverso "downsample" creates new frequencies from the same data with no
    leakage.
  - random gap masks of variable length (Timestral gap augmentation)
- Curriculum: `SYN_FRAC(t) = 1.0` for the first 50-60% of the budget, then 0.3-0.5. Implement it as a function of
  `elapsed / BUDGET` inside the training loop (around line 584), passed into `sample_batch`.

**Cost on the M3.** Generation on the GPU is a few ms per batch. The Cholesky bank is built once in under 5 s. No
change to model FLOPs. Memory +30 MB.

**Risk.**
- It will look *worse* in the 4-min loop (exp13/exp20 precedent).
- Too noisy a prior widens intervals; too clean a prior makes them overconfident. Watch cov80 and tune each
  generator's noise.
- Gate on `TSAR_ZS=1` at 30-60 min and on the big2-config val curve.

---

### 2. Attentive world-model decoder: the rolled-out state reads the context memory (T8/T1, the DHF analogue)

**Mechanism.** Our multi-origin rollout is already DHF in spirit. Each origin's forecast pathway is separate from the
causal context, so no context is masked. The difference is the read path. A Timestral prediction token attends to
every context token before its block. Our imagined state z_{t+k} sees only z_t. Add one cross-attention per rollout
step:
- the query is the imagined state (after the GRU or looped transition)
- the keys and values are the encoder outputs h_1..h_j up to the origin j (causal)
- a learned relative-lag bias b[head, (j+k) - i] indexes "distance from the target patch to the key patch", so the
  model can lock onto the patch one season or one week before the *target*

The explicit world model stays: the state is still carried, rolled forward and decoded. The attention is a memory
read conditioned on that state, as in retrieval-augmented world models. The ablation is just "memory read off".

**Evidence.**
- Reverso attention decoder: -2% MASE (0.740 -> 0.725), with the largest effect on long-range dependencies.
- Timestral's DHF prediction tokens attend to the full context.
- Our A1 ablation is consistent with a z-bottleneck.
- Fracast/TinyCast's decoder gathers context states plus a seasonal future state.

**Expected effect.** m3/m4 monthly and quarterly (pattern matching against older cycles), tourism, nn5 (h=56 =
7 patches: the state must remember the weekly shape for 8 steps), traffic, and long-horizon steps in general.
Estimate 1-3%.

**Implementation sketch.**
- `states()` (lines 371-383) also returns the upper-layer outputs `mem = h` [B, n, D].
- `rollout()` (lines 385-403) gains `mem` and `origin_idx` arguments. To keep the multi-origin batching, call it
  in `train_loss` (lines 429-443) *before* the `[keep]` flattening, with z shaped [B, n, D]. At step k compute
  `q = z_k` [B, n, D] and `attn = SDPA(q, mem, mem, attn_mask=causal_n×n + relbias[k])`. Origin j may only read
  keys i ≤ j, which is exactly `self.causal`. Then `z_read = z_k + W_o attn`. Decode `z_read`, but carry `z_k`
  (not `z_read`) into the next transition so the state stays a pure world model. Optionally let `z_read` feed back
  instead (variant B).
- Flatten with `keep` only for the pinball loss.
- Inference in `forecast` (lines 540-550) uses the last origin's row of the mask.
- Relative bias table: `nn.Parameter(torch.zeros(heads, N_PATCH + K_STEPS))`, indexed by `(j + k + 1 - i)`.

**Cost.** K=8 SDPA calls on [B, n=40, n=40] are small. Expect +15-25% step time and +~0.04M params at D=128.

**Risk.** Moderate. The decoder could bypass the dynamics, so log the A1-style ablation. Patch 8 is not aligned to
seasons of 12 or 7, so exact alignment still relies on `seasonal_ref` (item 4). These two items complement each
other.

---

### 3. Long context via coarse multi-scale prefix tokens (T1/T3)

**Mechanism.** Context 256 at patch 8 cannot see 2+ weeks of hourly traffic or ercot data, a year of nn5 or weather
data, or 20+ years of monthly tourism data. exp14 (context 512) lost because it halved the steps, not because long
context is useless. Reverso extends context 8x with downsampled channels at almost no cost. Timestral post-trains
at 32k and TiRex uses 2k. Add a *coarse prefix*: the history *before* the 256-step fine window is summarised in
blocks of 32 steps (8x downsampled to 4 values, or mean/min/max/last per block). That gives up to 2048 extra steps
as 64 extra tokens. The prefix comes earlier in time than the fine tokens, so causality is preserved.

**Evidence.**
- Reverso multi-scale input: -0.7% overall and -1.1% on long horizons at 2.6M params (larger at small sizes:
  long-horizon MASE 0.783 -> 0.766 for Nano).
- TiRex 1.1 "long-period normalisation".
- Every leader uses ≥ 2k context.

**Expected effect.** traffic, ercot (weekly), nn5 (annual/holiday), weather (annual). These are 4 of our 6 largest
gaps. Little effect on the M-competition sets.

**Implementation sketch.**
- `sample_batch` slices up to `CTX_LONG = 2048 + WINDOW` points (most dev series are short, so the prefix is often
  empty and fully masked).
- `predict` (lines 523-537) builds `ctx` of `CTX_LONG` instead of `CONTEXT`.
- New `self.coarse_embed` (same 2-layer MLP form as `self.embed`, line 352) with a learned scale token (FiLM, as in
  Fracast) and the observation mask.
- Normalise coarse blocks with the loc/scale at the *first fine patch* so they share a frame with the fine tokens.
- Prepend to `h` before `self.lower` (line 380). Extend `self.causal` so every fine token sees all coarse tokens
  and coarse tokens see only earlier coarse tokens. No losses on coarse positions.
- Only fine patches are origins, so `train_loss` is unchanged apart from the index offset.
- Two-stage use, as in Timestral: turn it on only for the last 30-40% of the budget (post-training).

**Cost.** Tokens go from 40 to up to 104: about 1.6-2x attention cost per step in that phase. Peak memory about
+300 MB.

**Risk.** Medium. Short series gain nothing. Step-time loss in the 4-min loop, so test it as phase 2 of a 30-60 min
run.

---

### 4. Robust seasonal memory: multi-season averaged reference plus a weekly lag channel (T7)

**Mechanism.** exp24's seasonal skip was our largest single per-dataset win (tourism_monthly 0.98 -> 0.81).
exp8 showed the failure mode: copying *one* season copies its noise. Replace the single lag-m reference in
`seasonal_ref` (lines 296-312) with:
- (a) the mean, or median for counts, of the last S ∈ {2,3,4} same-phase values that are available causally
- (b) a "seasonal naive + drift" channel: (a) plus the level change over the last season
- (c) for hourly data with m=24, a second-period channel at lag 168

Feed them as extra channels to `ref_proj` (`nn.Linear(n_ch*PATCH, D)`) and give each its own learned skip weight
(`ref_skip` [n_ch, NQ]). Optionally add the Fracast "commit" penalty: hinge on |median - y| exceeding
|seasonal ref - y|, active only on windows where the reference wins, λ ≈ 0.1-0.3.

**Evidence.**
- exp24 itself.
- Fracast/TinyCast seasonal future state plus commit loss (they win at short horizons at 85-150K params).
- Classical: seasonal averaging is why Theta and ETS are hard to beat on tourism and hospital.

**Expected effect.** tourism monthly/quarterly, hospital, m3/m4 monthly, nn5, traffic. 1-2% overall, cheaply.

**Implementation sketch.** In `seasonal_ref`, compute `idx_s = idx - (s-1)*m` for s=1..S, gather and mask as now,
then take a masked mean over s. The second-period lag uses `per2` passed from `sample_batch` (168 when per==24,
else 1). Synthetic rows also need a `per2`; `synthetic2` already has period pairs (line 150).

**Cost.** Negligible (gathers).

**Risk.** Low. It can only add channels the skip can ignore.

---

### 5. Overfitting control and selection protocol for long runs (T3)

**Mechanism.** big2 peaked at 30 min and then drifted, while cov err loosened. Even before item 1 lands, the
optimisation side can lock in the peak:
- **WSD schedule** (warmup, constant LR, short linear decay over the last 10-20%), as used by Toto 2.0 and Reverso.
  A cheap decayed checkpoint can then be branched at any time, so we evaluate "decay now" at 20, 30 and 40 min
  instead of committing to one cosine length.
- **Checkpoint/weight averaging over the decay phase only.** Our EMA failure (exp21) came from averaging stale
  high-LR weights in a 4-min cosine run. That does not apply to averaging within a low-LR tail.
- **Weight decay 0.05-0.1** instead of 0.01 (Reverso uses 0.1). Optional token dropout or stochastic depth only in
  the real-data phase. Timestral uses none, but it is not data-starved.
- **Select on `TSAR_ZS=1` val plus plain val at equal wall clock** and report the final checkpoint, never the
  best-of-curve (audit caveat a).

**Evidence.**
- Toto 2.0 and Reverso both use WSD with linear decay.
- Our own curve: 0.614 at 30 min vs 0.630 final is 2.5% left on the table.

**Implementation sketch.**
- LR line 584: `lr = LR * warm * (1 if frac < 1-DECAY else (1-frac)/DECAY)` with `TSAR_DECAY=0.15`.
- Keep a running average of `model.parameters()` when `frac > 1-DECAY`, reusing the `ema` machinery (lines 452,
  592-596) with a uniform average instead of exponential.
- `WEIGHT_DECAY` env override at line 38.
- To branch, run with `TSAR_MINUTES` smaller from a copied checkpoint: resume already works (lines 559-567).

**Cost.** Zero.

**Risk.** Low.

---

### 6. Modern block internals (T3)

RMSNorm, SwiGLU FFN, sigmoid output gate on attention, and RoPE on half of each head's dimensions instead of the
learned absolute `self.pos` (line 353).

Why RoPE: in multi-origin training, absolute position j conflates "distance from window start" with "amount of
history". Relative position generalises across origins and context lengths, and it is required for item 3.

**Evidence.** Reverso: RoPE attention 0.750 vs sin-cos 0.762 (-1.6%). Timestral uses all of these (no ablations).
Gated attention removes attention-sink artefacts (Qiu et al. 2025).

**Implementation.** Replace `nn.TransformerEncoderLayer` (lines 354-358) with a ~40-line custom block that uses
`F.scaled_dot_product_attention`.

**Cost.** Similar FLOPs. MPS SDPA is fine at n ≤ 104.

**Expected gain.** About 1%.

**Risk.** Low, but it touches everything, so land it as its own commit and re-baseline 3 seeds.

---

### 7. Richer world state: delta-rule matrix memory instead of the GRU state (T8/T1/T11)

**Mechanism.** Replace `state_rnn` (`nn.GRU`, line 357) with a DeltaNet-style matrix state S ∈ R^{d_k×d_v} per
head. The update is S ← S(I - β k kᵀ) + β v kᵀ for each patch, and the readout is S q. During rollout the same
delta rule runs on imagined inputs (`step_emb` + `cond`), so the transition operates on a matrix memory instead of
a 128-vector.

**Evidence.**
- TempoPFN GatedDeltaProduct (synthetic-only, competitive).
- Reverso DeltaNet layers beat attention (0.732 vs 0.750-0.762).
- TiRex: true recurrent state-tracking (sLSTM) beats Transformer blocks.
- Our T8 GRU state gave +2%.

**Expected effect.** Long-horizon coherence (nn5 h=56, traffic) and in-context periodic recall.

**Implementation.** A Python loop over n=40 patches with batched matmuls (no fused kernel on MPS). Head dim 32, 4
heads.

**Cost.** Probably 1.3-1.6x step time.

**Risk.** Medium-high (optimisation stability, speed). Try it after 1-4 have landed.

---

### 8. Then scale (T3)

Only once items 1, 4 and 5 stop the overfit: D=192, L=4-6 (about 2-4M params), 3-6 h runs on the synthetic-heavy
mix. Toto 2.0 shows smooth gains from 4M upward *when data is not repeated*. big1 and big2 show that scale without
fresh data plateaus or overfits.

### Lower priority or not recommended now

- **CPM (TiRex/Toto).** It removes context to create horizons. Our rollout already decouples them, which is
  Timestral's argument for DHF. Skip it.
- **Latent-consistency loss (TSAR_LATENT).** Keep it only in the EIDOS form: a stop-gradient target from an
  aggregated future encoder state, weight about 0.1, *together with* the pinball grounding. EIDOS gains are 1-2%,
  mostly at long horizons. The `q_latent` result is still pending. Do not re-run it at 4 min only.
- **21 training quantiles** (0.01-0.99) while evaluating 9. Cheap, small gain on calibration tails. The monotone
  head already lost (exp12).
- **Sign-flip / crop inference ensembles.** They hurt here (exp4, ensemble 0.650 vs 0.630, ercot cov 0.255).
  Positive-valued data makes -x out of distribution.
- **T9/T10 test-time adaptation.** It can only be judged on `TSAR_ZS=1` or the holdout. On val the context is
  in-sample.

## 3. Suggested sequencing (one heavy job at a time)

1. Item 5 (schedule and averaging) plus item 4 (seasonal memory): cheap, tested in the 4-min loop plus one 30-min
   run.
2. Item 1 generator bank and augmentation, judged on 30-min runs: `TSAR_ZS=1` and plain, against the item-5
   baseline.
3. Items 2 and 6 in the 4-min loop (architecture, so the 4-min loop is an acceptable filter), confirmed at 30 min.
4. Item 3 as a post-training phase in a 60-min run.
5. Combined 3-6 h run at 2-4M params (item 8). Then a confirmatory run by the human on test plus holdout, with the
   baselines scored on test and holdout by the same harness (audit to-do).

Realistic combined expectation on val at 60 min: 0.57-0.60 (items 1+4+5: about 4-5%, items 2+3: 2-4%). Each
estimate is uncertain to ±1-2%.

## 4. Honest assessment: can a 1-10M laptop model beat Timestral on a fair zero-shot holdout?

**On the current val protocol: plausible, but it does not prove the claim.** We train on the evaluated series'
own history, and Timestral does not. Even so, we are 11% behind overall and 19% behind on the four hard sets
(0.748 vs 0.627, trained-on). Beating 0.569 on val with items 1-5 is plausible. It would still be an in-domain vs
zero-shot comparison.

**On a fair zero-shot holdout: unlikely within 4-60 min runs; conceivable only with multi-day runs.** The numbers:
- **Data exposure.** We process ≈ 128x320 points per step at ~12-15 steps/s ≈ 0.5M points/s, about 2e9 points per
  hour or 4e10 per day for a sub-1M model. TiRex saw about 2.6e11 points (500k steps x 256 x 2048). Toto-4M saw
  3.4e12. Timestral's corpus is undisclosed but large and synthetic. A one-week run of a ~1M model reaches TiRex's
  exposure. A 5M model would be roughly 4-5x slower.
- **Small-model evidence.**
  - Reverso-2.6M (40 H100-hours) reaches GIFT MASE 0.706. Timestral-5M scores 0.705 and Timestral-209M 0.658.
  - Fracast-0 and TinyCast (85-150K params) sit 15-20% behind the frontier on WQL.
  - Toto-2.0-4M (3.4T points) is behind Timestral-5M on every benchmark.
  - So at ≤ 5M params the best published zero-shot results land near **Timestral-5M**, not Timestral-209M, even
    with 100-1000x our compute.
- **What it would take.**
  1. A Timestral/TempoPFN-quality synthetic prior (item 1) that we build ourselves and audit for leakage.
  2. A week-scale resumable pretraining run of a 2-5M model, which the checkpoint/resume infra already supports.
  3. Post-training on real data with long context (item 3) on *non-evaluated* corpora. Right now
     `load_train_series` is the only real data allowed, and the holdout families are excluded. That is fair but
     thin.
  4. Architectural edges that are cheap in compute: items 2, 4 and 7, where small models are known to be
     parameter-efficient.

  A realistic target for a fair zero-shot holdout is **TiRex / Timestral-5M level (about 0.59-0.60 equivalent)**.
  Beating Timestral-209M "by a significant margin" zero-shot is not supported by any published small-model result.
  Claim it only if the confirmatory holdout shows it, with baselines re-scored on the same split.

## 5. References

- Timestral, arXiv 2610.37585
- TiRex, arXiv 2505.23719; TiRex-2, arXiv 2607.01204
- Toto 2.0, arXiv 2605.20119
- TempoPFN, arXiv 2510.25502
- Reverso, arXiv 2602.17634
- Fracast-0, arXiv 2609.32209; TinyCast, arXiv 2608.15767
- EIDOS, arXiv 2602.14024
- Chronos-2, arXiv 2510.15821
- TTM, arXiv 2401.03955
- CauKer, arXiv 2508.02879
- Sundial, arXiv 2502.00816
- LoopWM, arXiv 2606.18208; JEPA-TTT, arXiv 2610.00722 (already in program.md)
