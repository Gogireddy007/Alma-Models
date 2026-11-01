# AlmaTS research program: literature, gap diagnosis, ranked experiments, kill criteria

Date: 2026-10-31. Author: research lead (literature and strategy). This is research only: no code was edited and no job
was queued. No dataset or model was downloaded, and TEST was not touched.

Every claim carries a label:
- **[R: source]** is reported by that source. Numbers were read from the paper this session unless marked "(memory)",
  which means recalled from the paper and not re-fetched today.
- **[M]** is measured here: it comes from our logs or ledger, with the run named. Unless a seed count is given, it is one
  seed and therefore preliminary.
- **[S]** is speculative: my estimate or inference.

Scores are VAL `wql_rel` (geometric mean vs Seasonal Naive, lower is better) unless stated otherwise. The bar:
- Chronos-2: 0.559.
- Timestral: 0.564 (209M), 0.572 (28M), 0.585 (5M).
- AlmaTS's current line is about 0.68 at 0.37M parameters.

---

## 0. Summary

1. **At ≤ 5M parameters, recipe and architecture beat raw data scale.** The small-model frontier on GIFT-Eval sits at
   MASE ≈ 0.705:
   - Timestral-5M scores 0.7049 [R: 2610.37585 Table 3].
   - Reverso-2.6M scores 0.706, after 10–40 H100-hours [R: 2602.17634].
   - Toto-2.0-4M saw 3.4T points, about 30,000× one of our runs, and still scores 0.7565 [R: 2605.20119; 2610.37585
     Table 3].

   For AlmaTS this means the right goal is the Timestral-5M point. Getting there needs model capacity, context and
   long-horizon decoding, not a bigger corpus.
2. **On every horizon we evaluate, the strongest baseline is a direct model, not a world model.** Timestral decodes up
   to 1024 steps in one parallel pass from identical query tokens [R: 2610.37585 §4.3; docs/timestral_anatomy.md §7].
   So a world model has to *earn* accuracy against a strong direct readout. The world-model literature says when that
   happens: the rollout is the only path to the future, it is trained on its own predictions, and it is stochastic
   where the future is uncertain (§1.1).
3. **Gap diagnosis (§2).** The 0.68 → 0.56 gap is mostly four overlapping things:
   - model capacity, about 0.025–0.045;
   - context length, about 0.015–0.035;
   - long-horizon decoding, about 0.015–0.03, which overlaps with context;
   - inference and calibration details, about 0.01–0.02.

   Training steps at 0.37M are worth about 0 [M]. The synthetic prior is the most uncertain term.
4. **Realistic ceiling [S].** Within 8 GB, with a 1–5M model and every lever below, the plausible endpoint is VAL
   0.59–0.62, which is Timestral-5M territory. The pre-registered target of "10% better than B\*" means ≤ 0.503. No
   published model of ≤ 10M parameters supports that (§4). The realistic wins are efficiency and calibration at
   near-B\* accuracy.
5. **First moves (§3, ranked by gain per M3-hour):**
   - eval-only inference fixes: quantile-path feedback, symmetric averaging and clamps;
   - context 1024 on the SSM-class model;
   - native long rollout (queued);
   - recalibration on TRAIN-held-out data;
   - multi-scale input.

   Then an iso-wall-time width ladder decides between 1–2M and 4–5M.

---

## 1. State of the art that matters for AlmaTS (2024–2026)

### 1.1 Transformer and latent world models: what makes imagination useful rather than decorative

| Model (arXiv) | Mechanism | Reported numbers | Lesson for AlmaTS |
|---|---|---|---|
| IRIS (2209.00588) | VQ autoencoder (16 tokens/frame) + autoregressive GPT over tokens; agent trained in imagination | Atari-100k mean HNS 1.046, 30M params, 4.1 GPU-days [R: 2405.12399 Tab. 1, 4] | Discrete tokens limit compounding error but lose detail. Our `vqtok` family had the best calibration (cov80_err 0.098) but did not pass the WM test [M: wave 5] |
| TWM (2303.07109) | Transformer-XL over latent, action and reward tokens | mean HNS 0.956 [R: 2310.09615 Tab. 2] | Mixing token types in one sequence is slower and no better |
| STORM (2310.09615) | Categorical-VAE stochastic latent + 2-layer Transformer, KV-cache imagination | mean HNS 1.267, median 0.584, 4.3 h on one RTX 3090. 4 or 6 layers did **not** help. The agent needs both h_t and z_t; using only h_t gives catastrophic-forgetting-like failures [R: 2310.09615 §5] | (a) Stochastic latents reduce compounding error. (b) A small world model saturates early: depth beyond 2 layers did nothing. (c) The deterministic memory and the sampled state carry different information |
| DreamerV3 (2301.04104) | RSSM with GRU deterministic state + categorical stochastic state; symlog, free bits, KL balancing, 1% unimix, fixed hyperparameters across domains | Atari-100k mean HNS 1.097 [R: 2405.12399 Tab. 1]; larger models are more data-efficient [R (memory)] | symlog ≈ our asinh frame. Our `rssm` failed here (0.7496, ablations FAIL [M: wave 5]); at 2500 steps the KL machinery is a cost without a payoff |
| DIAMOND (2405.12399) | EDM diffusion world model in observation space, frame-stacked conditioning, 3 denoising steps | mean HNS 1.459, IQM 0.641, 13M params, 2.9 days on a 4090. DDPM drifts quickly at ≤10 steps while 1-step EDM stays stable over 1000 autoregressive steps. 3 steps vs 1 step: mean HNS 3.05 vs 1.96 on its top-10 games. Frame stacking beats cross-attention for conditioning [R: 2405.12399 §5, App. K–M] | **Rollout stability is a property of the training objective, not of model size.** Our `flow` samp failed (0.860, samples too narrow [M]); DIAMOND shows the generative parametrisation decides whether long rollouts drift |
| Genie (2402.15391); Genie 2/3 (blog posts) | Latent action model + spatiotemporal Transformer over VQ tokens, 11B params, trained from unlabeled video | controllable worlds from a single image; consistency up to about a minute (Genie 2) [R (memory)] | Not applicable at our scale. The transferable idea: a learned latent "action" (here a regime) that is inferred, not given |
| V-JEPA 2 / V-JEPA 2-AC (2506.09985) | Predict representations, not pixels; action-conditioned predictor post-trained on < 62 h of robot video; plan by MPC in latent space | zero-shot pick-and-place on real robots; strong SSv2 and Epic-Kitchens anticipation [R (memory)] | Latent prediction pays off when the latent is grounded and the downstream use reads the latent. EIDOS (2602.14024) is the TS analogue: latent loss + grounding gives −2.3% CRPS and −4.7% CRPS at long horizons, and latent loss alone collapses [R: via docs/lit_review.md §4.1]. Our `jepa` family did not pass the WM test [M] |
| TD-MPC2 (2310.16828) | Decoder-free latent consistency with ρ^k-discounted multi-step loss | [R: via lit_review §4.6] | If we add a latent loss, discount it across leads |
| Parallel Predictive WMs (2610.08627) | Predict many future states in parallel instead of autoregressively, against horizon-linear error growth | title and abstract only; not read [R: abstract] | The planning literature is converging on the same direct-vs-rollout tension we measure |

**Synthesis: when is imagination useful? [S, grounded in the rows above and our wave 4–6 data]**
1. **The state must be the only path to the future.**
   - The GRU rollout tied a direct head because the decoder could read everything it needed from the origin state
     (gc_roll 0.6958 vs gc_direct 0.6978, 3 seeds [M]).
   - The SSM family is the first to pass the WM test: none 0.6707, direct 0.6926 (+0.022), copy 0.7083 (+0.038) on s0
     [M]. There the *same* system encodes and imagines, so there is no bypass.
   - In AlmaTS (`wm/tswm.py`) every lead token also attends to the full context. The carried state `T(z_k)` can
     therefore be ignored, the same failure mode as GRU. The design has to remove the bypass (experiment E7).
2. **Train on the distribution you will roll out on.**
   - DIAMOND's EDM vs DDPM result and Dreamer's imagination training both say that drift is set by train/inference
     mismatch.
   - AlmaTS's `TSAR_WM_TSWM_FRAME=loc` already uses detached *own* medians in training.
   - Our horizon chunking beyond 64 steps still feeds back medians the model never trained on. The measured long-horizon
     collapse (cov80 0.573 for h > 64 vs 0.717 for h ≤ 64, ssm s0 [M]) is exactly this.
3. **Stochasticity where the future is uncertain.**
   - STORM's stochastic latent and Dreamer's categorical state limit compounding error.
   - For forecasting, the payoff is calibrated multi-step *paths*: Joubaud's Timestral study (via research_log) and our
     `TSAR_WM_LONG=paths`.
   - Per-lead flow sampling was not enough: samples were too narrow [M]. A path-level objective is needed (E13).
4. **Small world models saturate early in depth.** In STORM, 2 layers beat 4 and 6 [R]. This supports scaling AlmaTS in
   width and data before depth at our budget [S].

### 1.2 Time-series foundation models: architecture, recipe and ablation evidence

Public numbers on GIFT-Eval (MASE / CRPS) and fev-bench (MASE skill / SQL skill, %), from the Timestral report
[R: 2610.37585 Table 3]:

| Model | Params | GIFT MASE | GIFT CRPS | TIME MASE | fev MASE sk | fev SQL sk |
|---|---|---|---|---|---|---|
| Timestral-209M | 209M | 0.6575 | 0.4460 | 0.6262 | 39.49 | 50.39 |
| Timestral-28M | 28M | 0.6780 | 0.4610 | 0.6382 | 37.27 | 48.42 |
| Timestral-5M | 5.2M | 0.7049 | 0.4789 | 0.6518 | 34.54 | 46.19 |
| TimesFM-3 | 331M | 0.6668 | 0.4557 | 0.6398 | 37.82 | 48.99 |
| t0-gamma | 256M | 0.6608 | 0.4520 | 0.6370 | 37.99 | 49.42 |
| TiRex-2 (PT / ZS) | 38M active | 0.6777 / 0.6973 | 0.4669 / 0.4781 | – | 33.74 | 45.50 |
| Chronos-2 | 120M | 0.6978 | 0.4854 | 0.6620 | 35.50 | 47.28 |
| TimesFM-2.5 | 200M | 0.7050 | 0.4903 | 0.6686 | 30.20 | 42.20 |
| TiRex | 35M | 0.7158 | 0.4885 | 0.6831 | 30.01 | 42.58 |
| Toto-2.0-22M / 4M | 22M / 4M | 0.7188 / 0.7565 | 0.4963 / 0.5242 | 0.6703 / 0.6910 | 30.61 / 28.04 | 42.91 / 40.71 |
| Moirai-2.0 | 11M | 0.7281 | 0.5164 | 0.7030 | 27.22 | 39.31 |
| Sundial-base | 128M | 0.7502 | 0.5590 | 0.7576 | 24.75 | 33.42 |
| Reverso | 2.6M | 0.706 (own report) | – | – | – | – |

(Reverso's figure is from [R: 2602.17634 Table 11].) I found no TimesFM-3 technical report (web search, 2026-10-31).
Its numbers above come from the Timestral table.

**Per-model notes, restricted to what transfers to a ≤ 10M model trainable in 8 GB.**

- **Timestral (2610.37585)**
  - Architecture: pre-norm RMSNorm, sigmoid-gated attention, RoPE on half the head dimensions, SwiGLU, no
    dropout or stochastic depth.
  - Head: a two-layer residual MLP producing 21 quantiles through a monotone cumsum head.
  - Normalisation: per-patch causal standardisation with asinh. Within a block, earlier patches use predicted
    medians, the same at train and inference.
  - Data: pretraining is **synthetic only** at an 8k context, followed by a shorter post-training on real +
    synthetic at 32k.
  - Loss: pinball only.
  - Decoding: up to 1024 steps in one forward pass. Inference uses symmetric (sign-flip) averaging, two passes.
  - Latency: Timestral-5M takes 17 ms on CPU at a 2048 context.
  - Gaps in the report: no ablations, and the corpus size is undisclosed. Scaling still improves from 123M to 209M.
- **Chronos-2 (2510.15821)**
  - 28M "small" vs 120M base: GIFT skill 50.4 vs 51.4, fev 45.3 vs 47.3, Chronos-II 44.1 vs 46.6.
  - Synthetic-only training is close behind (GIFT 50.4 vs 51.4, fev 45.9 vs 47.3).
  - **Long-context post-training from 2k to 8k: GIFT skill 50.1 → 51.4, fev 46.9 → 47.3.** The gain is largest on
    high-frequency, long-season data.
- **TiRex-2 (2607.01204)**
  - Model: 12 xLSTM blocks (alternating mLSTM/sLSTM), d = 512, patch 32, 99 quantiles, dropout 0.1.
  - Training: 700k steps × batch 64 on 2 H100s (about 50 h), then 100k steps at an 8k context.
  - Ablations over 6 seeds on fev: **8k → 2k context costs only +0.002 ± 0.004 MASE**, and sLSTM-only costs +0.008.
  - Streaming: it streams to 32M steps at constant cost.
  - On long context, this conflicts with Chronos-2 (see §2).
- **Toto 2.0 (2605.20119)**
  - Scale: patch 32, context 4096. 4M = d256/L4 and 22M = d512/L6, both trained 400k steps at global batch 64 on
    3.40T points; the authors say these sizes "converged at 400k steps".
  - Data: the proxy sweep removed public data from pretraining. The mix is 57.5% TempoPFN synthetic + 42.5%
    observability.
  - Optimiser: NorMuon + cautious weight decay, with u-µP transfer from a 10M proxy (d_model ≥ 256 is the floor for
    transfer). WSD with a **linear decay over 1.7–2.6% of steps** won.
  - Decoding: CPM with single-pass decoding.
  - **Long horizons: the 4M and 22M models collapse to flat or noisy forecasts past the training context, while 1B and
    2.5B stay coherent at 8192** (Pearson r 0.17 for 4M vs 0.82 for 2.5B at h = 8192 on a synthetic multi-scale signal).
- **Reverso (2602.17634)**
  - Model: 0.2 / 0.55 / 2.6M parameters, long convolutions + DeltaNet, attention decoder head.
  - Training: AdamW with WSD, weight decay 0.1, about 1M points per step, 10/20/40 H100-hours.
  - GIFT MASE by size: 0.757 / 0.721 / 0.706.
  - Ablations (smaller training set):
    - no synthetic data: 0.786 vs 0.738;
    - no augmentation: 0.755;
    - no multi-scale input: long MASE 0.749 → 0.741 at 2.6M and 0.783 → 0.766 at 0.2M;
    - no attention decoder: 0.740 vs 0.725;
    - full attention with RoPE: 0.750, vs DeltaNet 0.732 and the hybrid 0.725.
  - Inference: no flip averaging 0.713 vs 0.706. Inference-time downsampling: long-term 0.778 → 0.741.
  - Seeds: SD 0.0013 across 4 full runs. The 20% run vs the full run: 0.721 vs 0.7115.
- **TempoPFN (2510.25502)**
  - Model: 40M GatedDeltaProduct linear RNN, synthetic only, 3M iterations at batch 40 per GPU on 8–16 GPUs.
  - Leave-one-out generator ablation (500k iterations, GIFT CRPS, base 0.577):
    - −SDE (regime-switching, time-inhomogeneous OU, optional fBm): 0.729 (+26%);
    - −CauKer: 0.656;
    - −Spike: 0.645;
    - −StochRhythm: 0.642;
    - −Step: 0.640;
    - −ForecastPFN: 0.617;
    - −KernelSynth: 0.611;
    - −GP: 0.591.
  - Single-generator training is far worse than the mix (best single generator, CauKer, 0.600), so **diversity
    dominates**.
  - Augmentation: −5.4% CRPS.
- **Moirai 2.0 (2511.11698)**: quantile loss replacing mixture NLL is its largest single gain. The small model beat its
  8× and 30× variants [R: via lit_review §2.6].
- **Sundial (2502.00816)**: a flow-matching head beats diffusion and MSE heads (GIFT CRPS 0.505 vs 0.534 vs 0.642), but
  needs 20 samples at inference [R: via lit_review §2.8]. It is not on the frontier in 2026 (table above).
- **Fracast-0 / TinyCast (2609.32209 / 2608.15767)**: 85–147K parameters, GIFT WQL 0.545–0.564. They use periodogram
  period detection plus a seasonal future state. Weak at medium and long horizons [R: via arch_plan §1].

**Synthetic generators, summarised.**
- KernelSynth (Chronos, 2403.07815): composite-kernel GP.
- ForecastPFN (2311.01933): multiplicative trend × calendar seasonality with Weibull noise.
- CauKer (2508.02879): GP nodes on a random DAG with nonlinear edges.
- TempoPFN SDE: Markov-regime OU with time-varying θ, μ, σ, seasonal mean and volatility, and fBm.
- Ours [M: code read]:
  - `tools/synth3.py`: gp, season, arima, ets, regime, trend, rw, vol, count, pattern and causal (CauKer-like).
  - `tools/synth5.py`: structural, regime-AR, chaos, count and hetero.
  - `tools/synth6.py`: KernelSynth and TSMixup.
- What is still missing relative to TempoPFN [S]:
  - the full time-inhomogeneous OU SDE with fBm;
  - audio-style stochastic rhythms;
  - a synthetic-*first* schedule. Timestral and Toto pretrain mostly or entirely on synthetic data; we use a constant
    `TSAR_SYN_FRAC=0.3`.
- What the data shows [M]: SYN5 vs no SYN5 was a 3-seed null (0.6945 vs 0.6953), and the per-task effects traded weekly
  gains for daily losses (docs/gap_analysis.md §4).

**Context, patch and decoding: what the leaders use.**
- Context: Timestral 32k after 8k pretraining; Chronos-2 8k after 2k; TiRex-2 8k after 2k; Toto 4k; TempoPFN ≤ 2048.
- Patch: 32 for Timestral, Toto 2.0 (down from 64), TiRex and TiRex-2. We use 8 with a 256 context (32 tokens).
- Decoding beyond the native horizon is median feedback in Timestral and Toto. Chronos-2-style quantile-path feedback
  takes far-block 80% coverage from 2% to 57% for Timestral without retraining [R: Joubaud's study via research_log, not
  peer-reviewed]. TiRex-2 streams its recurrent state instead.

### 1.3 Calibration methods for quantile forecasters

- **Monotone heads.** Timestral uses a median plus cumsum of softplus increments, with knot spacing respected and a width
  factor of 3.0 [R: 2610.37585; anatomy §4]. AlmaTS already has it. Toto and Timestral clamp the output to the context
  range ± 10⁴ × scale [R: 2605.20119].
- **Number of quantiles.** Timestral uses 21 and TiRex-2 uses 99 with soft sample-impact capping [R: 2607.01204 App. D].
  The generic-Transformer ablation finds 9, 21 and 99 roughly equal [R: 2602.06909 via lit_review]. Expect only small
  effects, mostly in the tails [S].
- **Symmetric or flip averaging.** Timestral (by default), TimesFM 2.5 and Reverso all use it. Reverso's no-flip ablation
  costs +1.0% GIFT MASE, mostly on short series and horizons [R: 2602.17634 Table 17]. Our only try (exp4, untrained flip)
  hurt [M: lit_review §1], so it may need flip augmentation in training.
- **Conformal on held-out TRAIN.**
  - Split-conformal CQR (1905.03222) and adaptive conformal inference (ACI, 2106.00170) are the standard tools.
  - 2026 time-series variants add retrieval-corrected and regime-aware calibration (2608.10553, 2605.08857) and
    locally adaptive volatility correction (2610.31621) [R: abstracts].
  - For a zero-shot model the defensible form is **calibration fitted once on TRAIN-held-out (DEV) windows**, stratified
    by frequency group and lead bucket, and frozen before VAL [S]. Per-series online ACI on the context is also
    leakage-free, but it is transductive per series.
- **CRPS / path losses.**
  - Pinball over many quantiles approximates CRPS, which is what every leader trains on.
  - Sample-path calibration needs a path-level score: an energy score or sample CRPS over multi-step paths. Per-lead
    marginal training (our `flow` samp) gave paths that were too narrow [M].
- **Our calibration deficit is long-rollout and hourly** [M: gap_analysis §3]:
  - mean |cov80 error| on H/sub-hourly tasks is 0.175 for us vs 0.063 for C2;
  - on D and W/M we are within 0.01;
  - per-task coverage error is uncorrelated with the accuracy gap (r = −0.08).

  So recalibration fixes the cov80 claim but buys little wql [S].

### 1.4 Efficient training on small hardware

- **Precision on MPS.** torch 2.15 is installed [M]. Reports on bf16/fp16 autocast for MPS training conflict: bf16 has
  historically fallen back to fp32 for some ops, and PyTorch 2.15 adds Metal BF16 kernels with about a 1.23× geometric-
  mean speedup on linear shapes [R: secondary web sources, unverified]. The gain on our M3 is therefore **unknown and
  must be measured** before anything relies on it (E14). The Timestral report times inference in bf16, and the
  difference from fp32 is < 0.2% [R: 2610.37585 §4.3].
- **Gradient checkpointing.** It trades about 30% more step time for activation memory roughly proportional to depth
  [S, standard]. It is the lever that makes a 10M AlmaTS at B = 128 fit: the 5M shape already needs
  `TSAR_WM_ORIGINS=1024` to fit in about 2.9 GB of activations [M: runs/jobs/wave7_tswm.txt].
- **Sequence packing.** About 47% of positions in our windows are NaN left-padding [M: docs/compute.md §3]. Packing
  short series end to end, with a block-diagonal causal mask, or length-bucketed batching would recover that compute
  [S]. It changes the mask, so it needs its own parity test.
- **Schedules.**
  - WSD with a short linear decay: Toto decays over 1.7–2.6% of steps, Reverso also uses WSD [R].
  - Decaying from any checkpoint makes iso-wall-time comparisons cheap.
  - We have `TSAR_SCHED=wsd` with a cosine tail [M: harness doc].
- **µP / u-µP (2203.03466, 2407.17465).** Toto transfers hyperparameters from a 10M proxy and notes that d_model ≥ 256
  is the floor for stable transfer [R: 2605.20119 §4.1]. Our widths are 96–192, below that floor, so do not rely on µP
  transfer. Run a 3-point LR sweep (500 steps) at each rung instead [S].
- **Muon / NorMuon.** Toto chose it by sweep [R]. There is no TS ablation at our size. Low priority [S].

---

## 2. Gap diagnosis: where 0.68 → 0.56 comes from

The gap to Chronos-2 is ln(0.68 / 0.559) = 0.196 log, about 0.12 wql_rel points. The rows **overlap**: context,
frequency, horizon and multi-seasonality are collinear in our VAL (gap_analysis §2). Each estimate is "what fixing this
alone would recover from 0.68", so the rows do not add up.

| Source | Est. recoverable (wql_rel points) | Evidence | Confidence |
|---|---|---|---|
| **Model size** (0.37M → 4–5M) | 0.025–0.045 | Reverso 0.2 → 2.6M: GIFT MASE −6.7%; 0.55 → 2.6M: −2.1% [R]. Toto 4M → 22M: MASE −5.0%, CRPS −5.3% [R]. Timestral 5M → 209M on our VAL: −3.7% [M: baselines]. 4× steps at 0.37M gave 0.000 [M: gap_analysis §4], which looks like capacity-bound. Untested here: the width ladder is on standby | medium |
| **Training compute / tokens** (beyond 2500 steps at a fixed size) | ~0 at 0.37M [M]; 0.005–0.02 at ≥ 1M [S] | We see 1.0e8 points per run (2500 × 128 × 320) [M], against 1.1e11 for TiRex-2 [R: 700k × 64 × ~2.4k] and 3.4e12 for Toto-4M [R]. Yet Toto-4M is worse than Reverso-2.6M, so tokens do not substitute for recipe. Reverso 20% run → full run: −1.3% [R] | medium-low |
| **Synthetic pretraining** (prior quality + synthetic-first schedule) | 0.005–0.025 | Chronos-2-synth ≈ full (−1 skill point) [R]. TempoPFN LOO: SDE +26% in a synthetic-only regime [R]. Our SYN5 was a 3-seed null [M]. Our 4-minute-loop failures of synthetic-heavy mixes predate GIFT-clean and were at 0.37M [M: arch_plan §0]. Thin: no clean test at ≥ 1M | low |
| **Context length** (256 → 1024–2048) | 0.015–0.035 | Per-task gap vs median context: Spearman +0.70. Tasks with context > 1000 hold 70% of the gap; multi-seasonal tasks 43% [M: gap_analysis §2]. ctx1024 gave +0.02 on 1 seed [M, preliminary]. Chronos-2 2k → 8k: −2.6% relative GIFT error [R]. **Counter-evidence:** TiRex-2 8k → 2k costs +0.002 on fev [R]; most of the effect may be 256 → 2k, not beyond | medium |
| **Long-horizon decoding** (h > 64 via median feedback) | 0.015–0.03 (overlaps context) | Gap on h > 64 tasks is ~33% vs ~14% for h ≤ 64; ssm cov80 0.573 for h > 64 [M]. Arithmetic: bringing the 8 long-h tasks down to the short-h gap moves 0.68 → ~0.653. `paths` eval-only: ETT_1H −5%, ETT_15T −3%, jena_10T −4% [M]. Toto small models collapse past their training horizon [R] | medium-high (exists); medium (size) |
| **Patch size** (8 vs 32) | 0–0.01 | No clean ablation in the literature. Toto 64 → 32 was not isolated [R]. Mostly matters as the price of long context: ctx 2048 at patch 8 = 256 tokens [S] | low (speculative) |
| **Calibration and inference details** (cov80 0.12 vs 0.034; no flip averaging or clamps; 9 vs 21 knots) | 0.01–0.02 | Reverso flip TTA: −1.0% MASE [R]. Our bench gives Timestral its clampsym TTA, but not AlmaTS [M: anatomy §3]. Coverage error vs accuracy gap r = −0.08 [M] | medium |
| **Block architecture** (GRU → SSM / Timestral block) | 0.01–0.02 | ssm − gru = −0.014 (3 paired seeds); s3 gives 0.6808 vs gru 0.6934 [M, in the 5-seed band]. Reverso sequence-mixer ablation: 2–4% spread [R]. AlmaTS width-1 is queued | medium |

**Reading [S]:**
- Capacity and context/long-horizon each explain roughly a third of the gap. The remaining third is spread over
  inference details, the prior and architecture.
- Because the rows overlap, stacking the mid estimates (~0.13) overstates what is achievable. A realistic combined
  endpoint at 4–5M is **0.59–0.62**.
- The evidence is thinnest for:
  - the synthetic-prior term: no test exists at a size where the prior could matter;
  - the context term above 2k: Chronos-2 and TiRex-2 disagree.

---

## 3. Ranked research program for AlmaTS

**Conventions.**
- Every item is one factor, flag-gated, default off.
- **Family:** run on AlmaTS (`TSAR_WM=tswm`) if the queued width-1 screen (E5) puts AlmaTS within 0.01 of `ssm`.
  Otherwise run on `ssm` and port to AlmaTS later.
- **Base recipe:** wave 7's (`TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5
  TSAR_SYN5=1 TSAR_GIFT_CLEAN=1`).
- **Gate 3:** ≥ 0.020 wql_rel, mean over 3 fresh paired seeds. A gain of 0.008–0.020 goes to 5 seeds and needs ≥ 0.013.
  Eval-only changes are judged on 5 paired existing checkpoints.
- **World-model ablation duty:** `TSAR_WM_ABL=direct` (plus `copy`/`zero` where cheap) at the same setting. The
  none − direct gap must not shrink.
- **Costs:** measured wall times are gru ~10 min, ssm ~25 min (1500 s, 2.5 GB peak MPS, 1.6 GB RSS) per 2500 steps [M].
  AlmaTS-5M is estimated at 1.7 h (direct) and 4 h (none) [M: wave7 estimate]. Other costs are [S].
- **New flags:** marked *(new)*. Each needs a smoke run through `tools/smoke.py` that exercises eval at the job's
  context (lesson L4).

Ranked by expected gain ÷ M3-hours for the 3-seed gate:

| Rank | Exp | Expected Δ wql | M3-h (3 seeds) | Gain per h |
|---|---|---|---|---|
| 1 | E1 inference pack (eval-only) | 0.008–0.015 | ~1 | ~0.012 |
| 2 | E3 context 1024 | 0.015–0.03 | ~2 | ~0.011 |
| 3 | E4 native long rollout (queued) | 0.01–0.025 | ~2.5 | ~0.007 |
| 4 | E2 recalibration on DEV (eval-only) | 0.003–0.012 (cov80 ↓↓) | ~1 | ~0.007 |
| 5 | E10 multi-scale input | 0.005–0.015 | ~1.5 | ~0.007 |
| 6 | E6 iso-time width ladder, 1–2M rung | 0.01–0.02 | ~3 | ~0.005 |
| 7 | E7 AlmaTS bottlenecked SSM transition | 0–0.015 (WM test) | ~2 | ~0.004 |
| 8 | E8 synthetic-first schedule (≥ 1M) | 0.005–0.02 | ~3 | ~0.004 |
| 9 | E9 TempoPFN SDE generator | 0.003–0.015 | ~2 + dev | ~0.004 |
| 10 | E12 grounded latent target | 0–0.01 | ~2 | ~0.003 |
| 11 | E6b width ladder, 4–5M rung | 0.02–0.04 | ~9 | ~0.003 |
| 12 | E11 21 quantile knots | 0.002–0.008 | ~1.5 | ~0.003 |
| 13 | E13 stochastic path imagination | 0–0.01 (cov80 h > 64 ↓) | ~3 | ~0.002 |
| 14 | E15 overnight scaling run | 0.02–0.05 | ~30 | ~0.0012 |
| — | E5 AlmaTS width-1 (queued) and E14 throughput | prerequisites | ~1.5 / ~0.5 | — |

### E1. Inference pack (eval-only): quantile-path feedback + symmetric averaging + clamps
- **Hypothesis.** Part of the gap is inference that Timestral gets on our bench and AlmaTS does not:
  - median feedback beyond 64 steps;
  - no sign-flip averaging;
  - no non-negativity or range clamp.
- **Change.** Three separate eval-only factors, re-scoring existing checkpoints with `TSAR_EVAL_ONLY` (ssm s0–s4, then
  AlmaTS):
  - (a) `TSAR_WM_LONG=paths` (exists);
  - (b) *(new)* `TSAR_WM_TTA=sym`, which forecasts x and −x and averages q(x) with −q₁₋τ(−x), as in
    `forecaster.py:220-231`;
  - (c) *(new)* `TSAR_WM_CLAMP=1`: floor at 0 when the context is non-negative, plus the context range ± 10⁴·scale.
- **Expected effect.**
  - (a) 0.005–0.010 overall: 3–5% on the long tasks [M], 8 of 30 tasks.
  - (b) 0.003–0.010 (Reverso −1.0% [R]), with a real risk of nothing or harm, since exp4's untrained flip hurt [M].
  - (c) 0–0.003 (TimesFM/Toto defaults; not ablated [R]).
- **Cost.** About 1.5–2× eval time. A few minutes per checkpoint; under 1 h for all.
- **Control and world-model ablation.** Control is the same checkpoint with default eval. Apply the same flags to the
  `direct` checkpoints and confirm the none − direct gap is unchanged.
- **Gate.** 5 paired checkpoints; adopt (a), (b) and (c) independently under the 0.008–0.020 → ≥ 0.013 rule. If (b)
  fails untrained, retry once with flip augmentation in training *(new)* `TSAR_FLIP_TRAIN=0.5` as its own factor.

### E2. Recalibration fitted on TRAIN-held-out (DEV) windows (eval-only)
- **Hypothesis.** Under-coverage is systematic by frequency group and lead. A frozen map fitted on DEV removes it without
  hurting accuracy.
- **Change.** *(new)* `tools/recal.py` fits, per (frequency group × lead bucket {≤ 64, 65–160, > 160}), either:
  - an isotonic map of nominal → empirical quantile level (PIT recalibration), or
  - a CQR-style additive widening on the asinh-frame residuals,

  on DEV origins of TRAIN series only (`TSAR_DEV`). It is applied at eval by *(new)* `TSAR_WM_RECAL=<file>`.
- **Expected effect.**
  - cov80_err 0.12 → ≤ 0.06;
  - wql 0.003–0.012, because sharpness costs some of the gain [S];
  - CQR coverage guarantees hold only under exchangeability [R: 1905.03222]. DEV → VAL is a family shift, so expect
    partial transfer.
- **Cost.** Under 1 h.
- **Control and world-model ablation.** Control is the uncalibrated checkpoint. Apply to the `direct` checkpoints too and
  report the none − direct gap.
- **Gate.** cov80_err must improve on ≥ 10 of 13 VAL families, and wql must not worsen by more than 0.005. This is a
  calibration claim, not a Gate-3 accuracy claim.
- **Leakage.** Fit only on DEV/TRAIN and never read VAL.

### E3. Context 1024 at matched batch
- **Hypothesis.** A 256-step context hides weekly and longer seasons on hourly and sub-hourly tasks; those hold 70% of
  the gap.
- **Change.** `TSAR_CONTEXT=1024 TSAR_BATCH=32` vs control `TSAR_CONTEXT=256 TSAR_BATCH=32` (same steps). Both flags
  exist.
- **Expected effect.** 0.015–0.03 overall, ≥ 0.05 log on the context > 1000 subset [M: preliminary +0.02;
  gap_analysis §5 #1; R: Chronos-2 2k → 8k].
- **Cost.** Per row, ssm is about 4× the tokens of 256 (linear); at B = 32 the per-step tokens match the B = 128 runs.
  Roughly 30–40 min per run and 2.5–3.5 GB. AlmaTS at 128 tokens costs about 4× attention per row [S].
- **Pre-check.** Preflight eval at 1024 (L4: ctx1024 OOMed at eval once).
- **World-model ablation.** `direct` at ctx 1024.
- **Gate.** Gate 3, plus the long-context subset must improve ≥ 0.05 log.

### E4. Native long rollout (queued): `TSAR_MAX_H=320` with `TSAR_WM_ORIGINS=1024`
- **Hypothesis.** Imagining 320 steps natively beats chunked median or path feedback for h > 64. This is the
  world-model fix.
- **Change.** As queued in wave 6. The `ORIGINS=1024` control at `MAX_H=64` isolates the origin cap; LEADW=0.25 is on
  standby.
- **Expected effect.**
  - 0.01–0.025 overall, concentrated on the 8 long-h tasks;
  - target: halve their 33% excess;
  - cov80 for h > 64 rises from 0.57 toward ≥ 0.70.
- **Cost.** About 1.5–2× ssm wall time (~45 min) and ~3 GB.
- **World-model ablation.** `direct` at MAX_H=320. **The world-model signature to look for: (direct − none) grows with
  lead bucket.** Report it per bucket.
- **Gate.** Gate 3 vs the ORIGINS control, and short-h tasks must not regress by more than 0.01.

### E5. AlmaTS width-1: none vs direct (queued, prerequisite)
- **Hypothesis.** The Timestral block is at least as good as `ssm` at matched size, and the carried state beats DHF's
  independent queries.
- **Change.** `wm_tswm_none_s0` / `wm_tswm_direct_s0` (queued), then seeds 1–2, and copy/zero on standby.
- **Expected effect [S].**
  - AlmaTS-direct ≈ 0.68–0.70: a Timestral block at 0.37M with 32 context tokens has little to attend over.
  - none − direct ≈ 0: the bypass risk from §1.1.
- **Cost.** About 2 × 30 min per seed [S].
- **Gate.** World-model gate: none better than direct by ≥ 0.008 over 5 seeds. Accuracy gate: AlmaTS-none ≤ ssm + 0.01.
  If AlmaTS-none ≈ direct, go to E7 before any scaling.

### E6. Iso-wall-time width ladder (1–2M rung first, then 4–5M as E6b)
- **Hypothesis.** At our budget AlmaTS is capacity-bound (4× steps gave 0.000 at 0.37M). The compute-optimal size on
  the M3 is 1–2M, not 5M [S].
- **Change.** `TSAR_WM_WIDTH` set to the 1.0M point of the existing ladder (0.37 / 1.0 / 3.96 / 10.06M, research_log
  2026-10-30). Two arms:
  - (a) fixed 2500 steps;
  - (b) fixed wall time, equal to the 4M run's, with `TSAR_SCHED=wsd`.
- **LR.** A 3-point LR sweep at 500 steps per rung, since the widths are below the u-µP floor.
- **Expected effect.**
  - 1M: 0.01–0.02;
  - 4M: 0.02–0.04 [R: Reverso and Toto size steps; M: Timestral 5M → 209M on our VAL];
  - at iso-time, 1–2M may match 4M [S].
- **Cost.** 1M: about 45–60 min per run. 4M: about 2.5–4 h per run, B = 128, about 3 GB with `ORIGINS=1024`.
- **World-model ablation.** `direct` at the same width. Also check whether the WM gap grows or shrinks with scale: STORM
  saw no benefit from depth [R].
- **Gate.** Gate 3 per rung. The decision rule is in §4.

### E7. AlmaTS bottlenecked transition: make the state the only path past lead 0
- **Hypothesis.** AlmaTS's carried state can be bypassed because every lead re-reads the context. A selective-SSM
  transition, with leads ≥ 1 reading the context only through the state, turns the rollout into a real world model.
  This mirrors why `ssm` passes the test (+0.022 / +0.038) and `gru`, `jepa` and `rssm` do not [M].
- **Change (two one-factor steps, in order).**
  - *(new)* `TSAR_WM_TSWM_T=ssm`: a diagonal selective SSM (Mamba/S5-style, input-dependent Δ, as in `wm/ssm.py`)
    replaces the zero-init linear T.
  - *(new)* `TSAR_WM_TSWM_READ=first`: only lead 0 attends to context keys; leads ≥ 1 attend to the imagined tokens
    only.
- **Expected effect.** 0 to 0.015 in accuracy; **+0.01–0.03 on the none − direct gap** [S].
- **Cost.** Width 1: about 30–40 min per run.
- **Control and world-model ablation.** E5's AlmaTS-none is the control; `direct`, `copy` and `zero` are the ablations.
- **Gate.** World-model gate (≥ 0.008, 5 seeds) and no accuracy loss beyond seed noise (0.0065). **This is the
  experiment that makes AlmaTS more of a world model in a way that should help.**

### E8. Synthetic-first schedule (Timestral/Toto two-stage), at ≥ 1M parameters only
- **Hypothesis.** A synthetic-only first phase widens the prior. It helped the leaders, but failed in our earlier
  4-minute loops at 0.37M, a regime that favours fitting the train windows.
- **Change.** *(new)* `TSAR_SYN_PHASE=0.6`: `TSAR_SYN_FRAC=1.0` for the first 60% of steps, then 0.3.
- **Expected effect.** 0.005–0.02 [R: Chronos-2-synth ≈ full; Toto's sweep removed public data; M: SYN5 null at
  0.37M].
- **Cost.** One 1M run per seed, about 1 h.
- **Control and world-model ablation.** Control is a constant 0.3 at the same width. Run `direct` once.
- **Gate.** Gate 3, plus a `TSAR_DEV` zero-shot family check.

### E9. TempoPFN-style SDE generator
- **Hypothesis.** The one generator TempoPFN found indispensable (+26% when removed) is only partly covered by
  synth3's `regime`, which is Markov switching with OU smoothing.
- **Change.** *(new)* family `sde` in `tools/synth3.py`:
  - a 2-state Markov regime OU with time-varying θ, μ, σ (polynomial, sinusoidal, logistic or piecewise drifts);
  - seasonality in both mean and volatility;
  - optional fBm with H ∈ (0.2, 0.8);
  - measurement noise.

  Weight it equal to the other families.
- **Expected effect.** 0.003–0.015. The TempoPFN number is for synthetic-only training; with real data in the mix,
  expect much less [S].
- **Cost.** About half a day of development, then 3 seeds at the current rung.
- **Control and world-model ablation.** Control is synth3 without `sde`. Run `direct` once.
- **Gate.** Gate 3.

### E10. Multi-scale downsampled input channels (Reverso)
- **Hypothesis.** Coarse views (mean-pooled ×2/×4/×8 over a longer lookback) give long-season information at a fraction
  of long-context cost. Reverso found the effect larger at small sizes.
- **Change.** *(new)* `TSAR_WM_MSCALE=1` in `make_context`: add C = 3 causal pooled channels to each patch token,
  aligned to the patch end.
- **Expected effect.** 0.005–0.015, mostly long-h [R: long MASE −1.1% at 2.6M, −2.2% at 0.2M; inference-time
  downsampling −4.8% long-term].
- **Cost.** About +10% step time.
- **Control and world-model ablation.** Control at the same context. Run `direct`. Also run it against E3 as a 2×2 if
  both pass.
- **Gate.** Gate 3.

### E11. 21 training quantile knots, scored on the 9 deciles
- **Change.** *(new)* `TSAR_WM_KNOTS=21` (Timestral's levels; the eval picks the 9 deciles).
- **Expected effect.** 0.002–0.008, mostly cov80 tails [R: 9/21/99 ≈ equal in 2602.06909].
- **Cost.** Same as a base run.
- **Gate.** Gate 3. Low prior.

### E12. Grounded latent prediction target in AlmaTS (EIDOS form)
- **Change.** `TSAR_WM_TSWM_LAT=0.1` (exists), in its grounded form: a stop-gradient aggregated future state plus a
  pinball grounding head, discounted ρ^k across leads (TD-MPC2).
- **Expected effect.** 0 to 0.01, long-h [R: EIDOS −2.3% CRPS and −4.7% at long horizons]. Our `jepa` family failed the
  WM test [M], so the prior is weak.
- **Cost.** About +5–10% step time.
- **World-model ablation.** `direct` with and without the aux.
- **Gate.** Gate 3. The aux must not let `direct` gain more than `none`.

### E13. Stochastic path imagination trained with a path-level score
- **Hypothesis.** Calibrated multi-step paths need a path-level objective. Per-lead flow sampling gave paths that were
  too narrow (0.860) [M]. A sampled transition noise trained with a sample-CRPS / energy score over whole 320-step paths
  gives calibrated paths for h > 64. These replace mixture feedback.
- **Change.** *(new)* `TSAR_WM_SSM_STOCH=8`:
  - 8 noise samples injected in the transition;
  - loss = pinball (unchanged) + 0.1 × energy score of the 8 paths vs the truth;
  - at inference, path quantiles for h > MAX_H.
- **Expected effect.** cov80_err for h > 64 down 0.03–0.08; wql 0 to 0.01 [S, motivated by STORM and DIAMOND].
- **Cost.** About 2× step time.
- **Control.** E4 + `paths`.
- **Gate.** Calibration gate as in E2, plus no wql loss.

### E14. Throughput enabler: MPS autocast and activation checkpointing
- **Change.** *(new)* `TSAR_AMP=bf16|fp16` (autocast; fp32 master weights and loss) and `TSAR_CKPT=1` (checkpoint each
  block).
- **Gate.** Not accuracy. wql within ±0.005 of fp32 (same seed), **and** ≥ 1.3× step speed (AMP) or ≥ 30% lower peak
  memory (CKPT).
- **Cost.** About 0.5 h.
- **Why.** It decides whether E6b and E15 can run at B = 128 with all origins at 5–10M. Run it before E6b.

### E15. Overnight scaling run (only after E3, E4 and E6 resolve)
- **Change.** The best family at the winning E6 size; context 1024 (or 512 + MSCALE); MAX_H=320; WSD with a 10% linear
  decay; 12–15k steps (~8–10 h).
- **Expected effect.** 0.02–0.05 vs the width-1 recipe [S]. This is the run that tests the 0.59–0.62 ceiling.
- **Cost.** About 30 M3-hours for 3 seeds.
- **Gate.** Gate 3. Then Gate 4: held-out families, Tier-D, GIFT/TIME, never used for selection.

### 3.1 Realistic scaling plan within 8 GB
- **Memory.** A 4.5M AlmaTS fits at B = 128 with `ORIGINS=1024`: about 2.9 GB of activations plus 1–2 GB process
  overhead [M: wave7 notes; compute.md]. A 10M model needs B = 64 or checkpointing (E14). One job at a time.
- **Throughput.**
  - 0.37M gru: about 4 steps/s; ssm about 1.7 steps/s [M].
  - 4.5M AlmaTS: about 0.4 steps/s direct and 0.17 none [M: estimate].
  - So one night (~10 h) is about 14k direct or 6k none steps at 5M, about 5.7e8 points. That is 6× a current run and
    1/200 of TiRex-2.
- **Plan.**
  - Run the size question at iso-wall-time, not iso-steps (E6). Prefer the size that wins at a fixed 3 h.
  - My prior is 1–2M at about 30k steps over 4–5M at about 6–14k steps [S]. Reverso-0.55M is only 2.1% behind
    Reverso-2.6M at equal data [R], and Toto 4M/22M converged at 400k steps on far more data [R].
- **The world-model cost trap.** AlmaTS-none decodes leads sequentially: about 6 s/step vs 2.5 s/step for direct at 5M
  [M estimate]. At matched wall time, `direct` gets 2.4× more steps. **The WM test at scale must also be run
  iso-wall-time.** A world model that wins only at iso-steps does not help on an 8 GB machine.

### 3.2 Making AlmaTS more of a world model in ways that should help accuracy
In order of expected payoff [S, grounded in §1.1 and our measurements]:
1. **Remove the bypass** (E7). In wave 5, the state was the only path in the one family that passed the WM test.
2. **Native long imagination** (E4), then **imagination beyond the training horizon**. Train at MAX_H=320 and evaluate
   on DEV long series out to 1024 steps. Compare latent continuation (keep rolling the state) with observation-space
   path feedback. Toto's small models collapse past their training horizon [R], while TiRex-2's recurrent state streams
   cleanly [R]. A world model should extrapolate its *dynamics*, not re-encode its own medians.
3. **Stochastic trajectories for calibration** (E13). Score them at the path level; per-lead marginal sampling already
   failed [M].
4. **Grounded latent targets** (E12). Only with grounding. Ungrounded latent losses collapsed for EIDOS [R] and for our
   `jepa` [M].
5. **Spectral priors in the transition.** Koopman's per-series spectrum modulation was worth +0.016 (nomod 0.7006 vs
   0.6849 on s0 [M]). A damped-oscillator initialisation of the SSM transition's diagonal (periods 2–64 patches) is a
   cheap follow-up to E7 [S].

---

## 4. Risks and kill criteria

| # | Result | Action |
|---|---|---|
| K1 | **AlmaTS world model is decorative.** After E5 + E7 (5 seeds), AlmaTS-none beats AlmaTS-direct by < 0.008 at width 1 **and** at the 4–5M shape | Stop calling the Timestral-shaped rollout a world model. Pivot AlmaTS to "Timestral block as encoder + SSM as the shared encoder/imaginer" (the family that passed), or adopt `ssm` as the main line |
| K2 | **The WM wins only at iso-steps.** At 4–5M, AlmaTS-none loses to AlmaTS-direct at matched wall time | The world model costs more than it returns on this hardware. Keep it only if it wins on calibration (cov80 h > 64) or long-h extrapolation, and report the accuracy cost honestly |
| K3 | **The Timestral backbone does not pay.** AlmaTS (best of none/direct) is not ≥ 0.010 better than `ssm` at matched params and wall time after E5 and E6 (1M rung) | Drop the Timestral backbone and port E1–E4/E10 to `ssm` |
| K4 | **Capacity is not the bottleneck.** In E6, 4M fails to beat 1M by ≥ 0.020 at matched wall time over 3 seeds, and E3/E4 also fail | The gap is in data and recipe we cannot reach on 8 GB. Re-scope the claim to efficiency (params, predict-s, peak MB) and calibration, at stated accuracy |
| K5 | **The ceiling is confirmed.** After E15 the best VAL is above 0.62 | The pre-registered "10% better than B\*" (≤ 0.503) is out of reach. No published ≤ 10M model even matches Timestral-209M, let alone beats it by 10%: Timestral-5M is 7% behind on GIFT MASE and 3.7% behind on our VAL [R, M]. Tell the owner before more compute is spent |
| K6 | **Gains do not generalise.** Any VAL gain from E3/E8/E9/E10 disappears (sign flips or < 50% retained) on `TSAR_DEV` held-out families or Tier-D | Treat it as VAL overfitting (winner's curse at K screens ≈ 0.015–0.021) and revert |
| K7 | **Calibration traded for accuracy.** E2/E13 improve cov80 but cost wql > 0.005 or lose sharpness on > half the families | Keep recalibration as an optional output mode, not the default |
| K8 | **Leakage alarm.** A synthetic or real data change moves VAL-exposed families (solar, kdd, favorita) much more than clean ones | Stop, run `tools/overlap_scan.py`, audit |

**Main risks [S].**
1. Context and long-horizon gains are collinear with the same 10 hourly tasks. A gain there is one gain, so do not count
   it twice in planning.
2. The 4.5M AlmaTS-none step-time estimate is an upper bound measured under load. If the real figure is 2× better, K2
   eases.
3. TiRex-2's null result for 8k → 2k context suggests the context lever may saturate near 1–2k; E3 tests only 1024.
4. bf16 on MPS may not speed anything up, which caps the size at about 5M (E14).

---

## 5. References (arXiv ids)

**Time-series foundation models**
- Timestral 2610.37585
- Chronos-2 2510.15821
- Chronos / KernelSynth 2403.07815
- TiRex 2505.23719; TiRex-2 2607.01204
- Toto 2.0 2605.20119
- Reverso 2602.17634
- TempoPFN 2510.25502
- Moirai 2.0 2511.11698
- Sundial 2502.00816
- ForecastPFN 2311.01933
- CauKer 2508.02879
- Mamba4Cast 2410.09385
- EIDOS 2602.14024
- Generic-Transformer revisit 2602.06909
- Fracast-0 2609.32209; TinyCast 2608.15767
- HART 2610.12883 and Timer-M1 2610.11734: newer, not read
- TimesFM-3: no report found; numbers from Timestral Table 3

**World models**
- IRIS 2209.00588
- TWM 2303.07109
- STORM 2310.09615
- DreamerV3 2301.04104
- DIAMOND 2405.12399
- Genie 2402.15391
- V-JEPA 2 2506.09985
- TD-MPC2 2310.16828
- Parallel Predictive WMs 2610.08627: abstract only

**Calibration**
- CQR 1905.03222
- ACI 2106.00170
- Retrieval-corrected CP 2608.10553
- RareCP 2605.08857
- Volatility-corrected PIs 2610.31621
- TSFM reliability evaluation 2609.25788: not read

**Training**
- µP 2203.03466
- u-µP 2407.17465
- WSD (MiniCPM) 2404.06395
- Gated attention 2505.06708
