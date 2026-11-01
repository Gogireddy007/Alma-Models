# Literature review: what to borrow for Alma

Date: 2026-10-29. Author: Dr. Aarav Iyer (literature scout). Read-only on the repo apart from this file. I ran no training.

Scope: 2024 to 2026 work on (a) why the current leaders are strong, (b) world-model and latent-dynamics methods
that apply to forecasting, and (c) techniques that help small models. Every claim below comes from the paper as
cited. "Claimed gain" is the paper's own number on its own benchmark. It is not a prediction for bench val.
"Expected on Alma" is my estimate, and each one is uncertain to about ±1%.

Bench-val bar (wql_rel): Chronos-2 0.559, Timestral-209M 0.564, Timestral-28M 0.572, TiRex 0.582, Timestral-5M 0.585.

## 0. Bottom line

1. **Data and training recipe beat architecture in every ablation that separates them.** That holds in the
   Generic-Transformer, ORBIT/Falcon-2.0, Moirai 2.0 and Tabby ablations and in Chronos-2-Synth. Alma already
   has most of the data levers behind flags: synth3, AUG, SYN_CURR, PREFIX, balanced sampling. Their outcome
   will matter more than anything below. My top 8 are the cheap levers that remain after those.
2. **A latent world-model loss only helps when it is grounded.** In EIDOS, the latent loss on its own drifts
   toward collapse and adds nothing at short and medium horizons. Latent loss plus a grounding loss gives
   −2.3% CRPS. That matches our LATENT=0.3 result (0.844 vs 0.679). The fix is in the loss form, not in the idea.
3. **Small models are compute-efficient at a fixed FLOP budget.** Financial World Modeling found ViT-Tiny beat
   ViT-Base at 10× the FLOPs. Moirai 2.0's base and large versions did not beat small, and EIDOS-12.7M had the
   best CRPS of its family. A 2-5M Alma is the right size for the M3, provided it gets fresh, non-repeated data.
4. **The free inference-side tricks the leaders use** are positivity clamps, context-range clamps and
   flip-invariant averaging, but the last one only works if the model is trained for it. We have tried only
   the untrained version (exp4), and it failed.

## 1. Ranked top 8: expected gain per unit of compute

Ranking = (expected bench-val gain × confidence) ÷ (extra train/inference compute on the M3 + engineering risk).
None of these is implemented in `train.py` today, as far as I can see. WSD exists as `TSAR_SCHED=wsd` with a
cosine tail; item 2 changes that tail.

| # | Technique | Source | Claimed gain | Expected on Alma | Extra compute | Fits 2-5M on M3 |
|---|---|---|---|---|---|---|
| 1 | Positivity and context-range output clamps | TimesFM 2.5 (`infer_is_positive`), Toto 2.0 | not ablated (default in both) | −0.3 to −1% wql, better cov80 on count/sales/solar | ~0 (inference only) | yes |
| 2 | Short linear decay + uniform averaging over the decay; progressive halving (PCS) for the overnight run | Toto 2.0, Tabby | Toto picked linear decay over 1.7-2.6% of steps by sweep. Tabby PCS beats cosine and WSD-S | −1 to −2.5% (locks in big2's peak, 0.614 vs 0.630 final) | 0 | yes |
| 3 | Grounded latent loss (EIDOS form), or SIGReg anti-collapse | EIDOS; LeWM | −2.3% CRPS, −1.4% MASE; −4.7% CRPS at long horizons | −1 to −2%, mostly at h ≥ 4 patches (nn5, traffic, ETT-H, LOOP-H) | +5-10% step time | yes |
| 4 | Deep quantile supervision on intermediate states | Tabby (DQS); Falcon-2.0 cross-depth alignment | MASE 0.748 → 0.737 (−1.4%), CRPS 0.512 → 0.504 | −0.5 to −1.5% | +3-5% | yes |
| 5 | Synthetic Data Distillation (SDD): pinball against the generator's exact conditional law | Meta, 2609.09586 | 1.6-1.85× faster convergence with CPM; 1.2-1.33× with all-position (teacher-forcing) supervision, which is Alma's regime | −0.5 to −1% at a fixed 2500 steps; more if run time is the binding constraint | 0 on GPU; extra numpy in synth3 | yes |
| 6 | Loss-guided, dataset-stratified window selection (keep the central 80%) | 2609.37255 | beats full data at 80% retention: Chronos-Bolt MASE 0.796 → 0.777, CRPS 0.555 → 0.542 | −1 to −2% on real rows | one scoring pass, or ~0 for the online variant | yes |
| 7 | Muon/NorMuon for hidden matrices | Toto 2.0 | chosen by sweep at proxy scale; no TS ablation | uncertain: faster convergence at fixed steps, maybe −0.5 to −1.5% | ~+5% step (Newton-Schulz on ≤192×768 matrices) | yes |
| 8 | Flip-invariance trained in, then flip-averaged inference | TimesFM 2.5 (`force_flip_invariance`) | default on; not ablated publicly | −0.5 to −1% wql; tighter calibration | 2× inference (Alma infers in seconds) | yes |

Just below the cut: modern block internals (§3.1), multi-head auxiliary state targets (§4.4), a sLSTM-style
state cell (§2.2), and 21 training quantiles (§3.3).

### Implementation sketches for the top 8

Each one goes behind its own env flag in a copy of `train.py`, default off, as the team rules require.

**1. Output clamps (`TSAR_CLAMP=1`).** In `forecast()`, after `from_model_space`:
- If `nanmin(context) >= 0`, apply `q = q.clamp(min=0)` (TimesFM 2.5 `infer_is_positive`).
- Clamp every quantile to `[cmin − R·scale_last, cmax + R·scale_last]` with a large R, for example 50. This
  copies Toto 2.0's guard against blow-ups. They use 10⁴ × the anchor scale.
- Then sort along the quantile axis. `prepare.evaluate` already sorts, but sort anyway so the head is monotone
  by construction.

Optional training-side version: when a training row's context is non-negative, compute the pinball on
`max(q, lower)` in model space, so the head learns that the floor exists.

Leakage: none, since it uses only the context.

**2. Decay-phase recipe (`TSAR_SCHED=wsdl`, `TSAR_DECAY=0.1`, `TSAR_AVG=1`).**
- At line ~709, replace the cosine tail with a linear decay to 0 over the last 10% of steps. Toto 2.0 found
  that linear beats 1-sqrt, and that a short tail is enough.
- During the tail, keep a uniform running average of the weights, reusing the `ema` deepcopy with
  `w_avg += (w − w_avg)/k`, and evaluate it. This is not exp21's EMA: that run averaged stale high-LR weights.
- For the overnight run, use Tabby's PCS. Each stage holds a constant LR at half the previous stage's LR and
  ends with its own short decay, so every stage end is a converged checkpoint you can evaluate or branch from.
  This replaces "pick one cosine length" and works with the resume code (lines ~559-567).

Judge it against `TSAR_SCHED=cos` with paired CIs at a fixed step count, and also check the final checkpoint
of a 30-60 minute run. Never the best point on the curve.

**3. Grounded latent loss (`TSAR_LATENT=0.1 TSAR_LATENT_FORM=eidos`).** Change `latent_consistency()`:
- *Target.* The current target is the raw future encoder state `z.detach()[j+k]`. Replace it with an
  aggregated future target, built by a depthwise Conv1d over `z[j+1 : j+k]` followed by an MLP (EIDOS
  `h_φ`). Its kernel spans the K patches; EIDOS uses kernel = horizon. Keep the stop-gradient on the target,
  as EIDOS does.
- *Distance.* Use negative cosine, not MSE: `-(F.normalize(prior) * F.normalize(sg(target))).sum(-1)`.
  The cosine is invariant to the overall state scale. With MSE the GRU can shrink the state to win the loss.
- *Grounding.* Add a small head `g_ψ(target) → quantiles` for the patches it summarises, trained with pinball.
  This is what keeps the target from collapsing.
- *Weights.* EIDOS's sensitivity sweep chose 0.1 for both λ_latent and λ_gnd; the forecast pinball keeps
  weight 1.

Alternative: LeWM's SIGReg. Keep the MSE prediction loss, drop the stop-gradient, and add `λ·SIGReg(z)` with
λ = 0.1 and M = 256-1024 random directions. SIGReg runs the Epps-Pulley statistic on 1-D projections of the
batch of states, which is about 30 lines and an O(B·D·M) matmul, so it is cheap. Use one of the two, not both.

Report the A1-style ablation (state pathway off) alongside, because this is the evidence for the "genuine
world model" claim.

**4. Deep quantile supervision (`TSAR_DQS=0.3`).**
- Add `self.decode_lo = nn.Linear(D, PATCH*NQ)` to read the **lower** encoder output `h`, taken after `lower`
  and before `state_rnn`. It predicts the next patch only (k = 1) from every origin.
- Add `self.decode_mid`, which applies the same rollout to `z` before `upper` if `N_LAYERS ≥ 3`.
- Loss: `DQS · pinball(aux)` plus Tabby's consistency term, which pulls each intermediate prediction toward
  the line between the detached first-exit and final-exit predictions.
- Falcon's cheaper variant: a cosine alignment from shallow `h` to stop-gradient deep `h`.
- Inference is unchanged. The auxiliary heads are dropped.

**5. SDD on synthetic rows (`TSAR_SDD=1`).**
- Extend `S3.generate(..., return_moments=True)` so that, for tractable families, it returns the conditional
  mean μ[b, j, t] and standard deviation s[b, j, t] of each future point given the history up to origin j. Do
  this only for the origins Alma trains on: 40 fine patches × MAX_H = 64. Closed forms per family:
  - AR/ARIMA/ETS/OU, and regime (given the recorded regime path): a Kalman forward pass, which is cheap and
    vectorised.
  - rw/geometric rw: Gaussian/lognormal increments, closed form.
  - `pattern` and other "deterministic g + iid noise" rows: μ = g(t), s = σ_t.
  - gp: this needs the GP posterior per origin. That is an O(L³) solve per row, which is too slow in numpy,
    so mark gp rows as intractable.
  - causal (CauKer) and count: no closed form. Use the status-quo loss for these rows.
- Post-processing: affine steps (standardisation, lognormal scale, offset, positivity shift) transform (μ, s)
  exactly. Added Gaussian or Student-t noise adds its variance, with a t-CDF for the t case. Censoring,
  quantisation and missing-value masking break tractability, so those rows also fall back.
- Loss: for rows with moments, replace `pinball(pred, tgt)` with the Gaussian distilled pinball
  `s·[φ(z) + z(Φ(z) − τ)]`, where z = (q̂ − μ)/s, after mapping μ and s into model space with the same
  loc/scale as `to_model_space` (affine again, so exact).
- Expected effect is a 1.2-1.3× speed-up, not a better optimum. It is unbiased: the minimiser is still the
  conditional τ-quantile.

**6. Loss-guided window selection (`TSAR_RFL=0.8`).**
- *Offline version (paper).* With Alma's own earlier checkpoint (big2 or the current best), score R random
  windows per dataset from `load_train_series` using normalised pinball. Store, per dataset, the 10th and 90th
  percentile thresholds. In `sample_batch`, reject real windows whose score falls outside the band and redraw.
  Scores are only needed for thresholds, so about 2k windows per dataset is enough, which takes minutes.
- *Online variant (zero extra compute; my extrapolation, not in the paper).* In `train_loss`, compute the
  per-row loss, then mask out real rows in the top and bottom 10% of the per-row loss within their dataset in
  that batch.
- Stratify by dataset: the paper's global ranking was worse by 0.6-3.5%. Do not apply this to synthetic rows,
  where the prior is meant to include the hard tail.
- Leakage: use Alma's own checkpoints as reference only (§6).

**7. Muon/NorMuon (`TSAR_OPT=muon`).**
- Use Muon (Newton-Schulz orthogonalised momentum) on 2-D hidden weights: attention in/out projections, FFN,
  GRU weights and `to_state`. Keep AdamW for embeddings, `decode`, `ref_proj`, biases, norms, `pos`,
  `step_emb` and `ref_skip`.
- NorMuon adds a per-row second-moment normalisation, 3 lines on top of Muon. Use cautious weight decay, as
  Toto does: decay only where the sign agrees with the update.
- Sweep the LR once at the 2500-step budget; Toto found the AdamW-side LR ~0.012 for its proxy.
- MPS: 5 quintic iterations on 96-192-wide matrices are negligible. Use fp32 on MPS; bf16 is unreliable there.

**8. Trained flip-invariance (`TSAR_FLIP_TRAIN=0.5` plus `TSAR_ENS=256+,256-`).**
- In `sample_batch`, negate a random 50% of synthetic rows and 25% of real rows. `seasonal_ref` and
  `causal_stats` are sign-equivariant, so nothing else changes.
- Inference: the existing reflect-and-average (`-q.flip(-1)`), then apply clamp #1.
- exp4 failed because the model had never seen −x. This version teaches it.
- Risk: some capacity goes to a symmetry that positive data never uses. Gate on cov80 and on the counts tasks.

## 2. What makes the leaders strong (a)

### 2.1 Timestral (Mistral) [alphaxiv.org/abs/2610.37585](https://www.alphaxiv.org/abs/2610.37585)

- **Architecture.** Causal patch Transformer, P = 32, with RMSNorm, SwiGLU, sigmoid-gated attention and RoPE
  on half of each head. There are no timestamps or frequency embeddings, and no dropout or stochastic depth.
- **Output head.** 21 quantiles (0.01-0.99) from a monotone head: the median plus cumulative sums of
  softplus increments.
- **DHF.** Prediction tokens are separate from the context, so no context is masked.
- **Normalisation.** Each forecast patch is standardised with statistics up to the preceding patch, and
  within a block the not-yet-observed patches are filled with predicted medians.
- **Data.** Pretrained only on synthetic data at 8k context, then post-trained on real + synthetic at 32k.
- **Loss.** Pinball only, with no auxiliary loss.
- **Ablations.** None in the report. The 5M model is d = 192, 8 blocks.
- **Fit for Alma.** Alma's multi-origin rollout is already DHF in spirit. The only untested piece is the
  per-patch re-standardisation inside the forecast block. Alma decodes everything in the origin frame, which
  may hurt trending series at long h (§3.4).

### 2.2 TiRex (xLSTM) [alphaxiv.org/abs/2505.23719](https://www.alphaxiv.org/abs/2505.23719)

- **Model.** 35M parameters, sLSTM blocks with a true recurrent state. sLSTM beats mLSTM and Transformer
  variants, and switching even one block from mLSTM to sLSTM helps.
- **CPM.** Contiguous Patch Masking feeds future patches as missing inputs, so uncertainty propagates through
  the state. Removing it hurts long horizons.
- **Augmentations.** Amplitude modulation, censoring and spike injection each help. Removing all three hurts
  most.
- **Data.** Chronos data + TSMixup + extended KernelSynth (GPU sampler) + GIFT-pretrain. 500k steps,
  context 2048.
- **Fit for Alma.** The AUG flag already implements the augmentations. CPM is not needed, because the rollout
  is DHF-like. A state-tracking cell (sLSTM-style exponential gating with a normaliser state) in place of
  `nn.GRU`/`GRUCell` is a medium-cost option. It needs a Python loop over 40 patches on MPS, which is slower
  than cuDNN-style fused GRU.

### 2.3 TiRex-2 [alphaxiv.org/abs/2607.01204](https://www.alphaxiv.org/abs/2607.01204)

Extends TiRex to multivariate data and streaming, which is not relevant to our univariate bench.

### 2.4 Chronos-2 [alphaxiv.org/abs/2510.15821](https://www.alphaxiv.org/abs/2510.15821)

- **Synthetic-only variant.** GIFT skill 50.4 vs 51.4 for the full model, fev 45.9 vs 47.3, Chronos-II
  46.4 vs 46.6.
- **Size.** The 28M "small" model is ~1 skill point behind on GIFT and ~2× faster.
- **Context.** Post-training from 2k to 8k context gives +1.3 GIFT skill points, mostly on high-frequency,
  long-season data.
- **Generators.** TSI (trend/seasonality/irregularity) and TCM (random temporal causal graphs → AR).
- **Fit for Alma.** Group attention is multivariate and irrelevant here. Their corpus overlaps GIFT-Eval
  train portions, which is a leakage issue for us (§6).

### 2.5 Toto 2.0 (Datadog) [alphaxiv.org/abs/2605.20119](https://www.alphaxiv.org/abs/2605.20119)

- **Data.** No public data in pretraining: the proxy sweep removed it. The mix is 57.5% TempoPFN synthetic
  and 42.5% internal observability. Fine-tuning then uses 45% public data.
- **Optimiser.** NorMuon with cautious weight decay and u-μP transfer.
- **Schedule.** Linear decay over only 1.7-2.6% of steps, which won the sweep.
- **Architecture.** P = 32, asinh on standardised input, residual-MLP patch in/out projections, PerDimScale
  query scaling, CPM.
- **Decoding.** Each real-space quantile is clamped to the context range ± 10⁴ × the anchor scale.
- **Size.** The 4M model is behind Timestral-5M everywhere and saw 3.4T points.
- **Leakage flag.** Datadog observability data is the same source as BOOM, so our TEST `boomlet` family is
  in-distribution for Toto. Never use Toto weights or outputs.

### 2.6 Moirai 2.0 [alphaxiv.org/abs/2511.11698](https://www.alphaxiv.org/abs/2511.11698)

- **Ablation chain.** The largest single gain is quantile loss replacing mixture-NLL. Decoder-only beats a
  masked encoder, and a residual-block output head beats a linear one. Random masking alone hurts, and only
  helps when combined with multi-token prediction.
- **Size.** The small model beat its own 8× and 30× variants.
- **Leakage flag.** Moirai 2.0's corpus includes the GIFT-Eval train split. Every VAL family carries a
  "Moirai-2" exposure flag in `docs/protocol.md`.

### 2.7 TimesFM 2.5 / 3

- **Release.** 200M parameters. The TimesFM 2.5 repo has flags `force_flip_invariance`, `infer_is_positive`,
  `fix_quantile_crossing` and a continuous quantile head
  ([github.com/google-research/timesfm](https://github.com/google-research/timesfm),
  [HF docs](https://huggingface.co/docs/transformers/model_doc/timesfm2_5)). Defaults differ between
  versions, so check them per release.
- **TimesFM-3.** Uses CPM according to the Timestral report. I could not retrieve a full technical report.
- **Fit for Alma.** Items 1 and 8 of the top list are direct transplants.

### 2.8 Sundial [alphaxiv.org/abs/2502.00816](https://www.alphaxiv.org/abs/2502.00816)

- **Method.** TimeFlow, a flow-matching head on a causal Transformer with RoPE.
- **Results.** GIFT CRPS 0.505 with the flow head vs 0.534 with diffusion vs 0.642 with MSE. It is not
  compared against a quantile head.
- **Data.** 1T points, mostly real data plus ERA5.
- **Fit.** It needs 20 samples × K steps at inference. That is poor value on the M3, and a quantile head
  already captures most of the gain over MSE. Skip.

### 2.9 TabPFN-TS [alphaxiv.org/abs/2501.02945](https://www.alphaxiv.org/abs/2501.02945)

- **Method.** Forecasting as tabular regression on calendar features plus a running index, using synthetic
  priors only.
- **Results.** Strong on short series and on MASE in Chronos-ZS, but ~2000× slower than TiRex.
- **Fit.** The transferable idea is the PFN view: the prior is the training distribution. That is already
  the basis of synth3, so there is nothing further to port.

### 2.10 Open recipes with real ablations

**Falcon-2.0 / ORBIT** [alphaxiv.org/abs/2608.13262](https://www.alphaxiv.org/abs/2608.13262)

- **Sampling.** Bootstrap stochastic sampling (dataset → record → variable → endpoint → context start →
  horizon) beats sliding windows by −11.7% MASE and −13.6% CRPS on GIFT. Joint context and horizon sampling
  adds another 2.9% and 6.1%.
- **Architecture.** Parallel patch prediction is the largest architectural gain (−8% MASE). Output gating is
  <1%, and the residual SwiGLU patch projection gives 0.4-1%.
- **Alma status.** Alma already samples dataset → series → endpoint uniformly, which is the bootstrap scheme.
  Context lengths are covered implicitly by multi-origin training, and horizons by the rollout. Nothing big
  left here.

**Tabby** [alphaxiv.org/abs/2609.13956](https://www.alphaxiv.org/abs/2609.13956)

- **DQS.** Removing it raises MASE from 0.7368 to 0.7476 and CRPS from 0.5038 to 0.5117.
- **CauKerV2.** Removing it costs +2.3% MASE.
- **Schedule.** PCS beats both cosine and WSD-S.
- **Data.** Sub-dataset-balanced sampling.
- **Masking.** Short terminal masks of at most 2 patches; longer ones hurt.

**Revisiting the Generic Transformer** [alphaxiv.org/abs/2602.06909](https://www.alphaxiv.org/abs/2602.06909)

- **Data.** Real + synthetic together beats either alone. KernelSynth on its own is worst.
- **Quantiles.** 9, 21 and 99 quantiles perform about the same.
- **Depth vs width.** Deeper and narrower beats shallow and wide at equal parameters. This matters for
  Alma's scale-up choice: L = 4-6 at D = 128-192 rather than wider.
- **Other.** Longer context helps, and steps are not saturated.
- **Leakage evidence.** The "leaky TSMixup" variant scores better than the clean one, which is direct
  evidence that leakage inflates benchmark numbers.

**Normalisation for causal LTSMs** [alphaxiv.org/abs/2606.09954](https://www.alphaxiv.org/abs/2606.09954)

- RevIN-style global statistics leak, give lower training loss and do not give better forecasts.
- Causal statistics + asinh is best at context > 128, and prefix statistics are best at 128.
- Alma's causal "last" statistics + asinh is the right choice. Keep it.

## 3. Architecture and head details (cheap, mid-value)

### 3.1 Modern block internals

RMSNorm, SwiGLU, sigmoid-gated attention, RoPE on half the head dimensions, and residual-MLP patch in/out
projections.

- **Evidence.** Reverso: RoPE 0.750 vs sin-cos 0.762 (−1.6%). Falcon: gating <1%, residual SwiGLU projection
  0.4-1%. Moirai 2.0: residual output head helps. Toto 2.0 uses residual-MLP projections.
- **Sketch.** Replace `nn.TransformerEncoderLayer` with a custom ~40-line block (arch_plan item 6). Make
  `decode` a 2-layer residual MLP rather than `LayerNorm → Linear`.
- **Expected.** −0.5 to −1.5% combined, at neutral compute.

### 3.2 Per-patch re-standardisation inside the forecast block (Timestral)

- **Sketch.** At rollout step k, re-express the target frame using statistics that include the k−1 predicted
  medians. Decode in that frame and map back. Done properly, the training targets use the same rule.
- **Expected.** Helps trending series at long h (covid-like, uk_covid cumulative). Uncertain, medium effort.

### 3.3 21 training quantiles (0.01-0.99), scored on the 9 deciles

- **Evidence.** Timestral and Tabby (99) use it. The generic-Transformer ablation shows little difference.
- **Expected.** Small gain, mostly in the cov80 tails. Cheap: change `Q` in train.py only, and pick the 9
  evaluation columns in `forecast`.

### 3.4 Multi-scale / long context

PREFIX is implemented.

- **Evidence.** Chronos-2 2k→8k: +1.3 skill points. Reverso multi-scale input: −0.7% overall, −1.1% at long
  horizons.
- **Recommendation.** Keep it as a post-training phase, as Timestral and Chronos-2 do.

## 4. World models and latent dynamics (b)

### 4.1 EIDOS: latent-space predictive learning for TSFMs [alphaxiv.org/abs/2602.14024](https://www.alphaxiv.org/abs/2602.14024)

- **Method.** Causal Transformer predicting a stop-gradient aggregated future embedding (depthwise conv +
  MLP), negative cosine distance, a quantile grounding head on the target, and the forecast loss.
  λ_latent = λ_gnd = 0.1.
- **Ablation vs v1.** Latent loss alone: overall MASE unchanged, long-horizon CRPS −2.2%. Latent + grounding:
  −1.4% MASE, −2.3% CRPS, and −4.7% CRPS at long horizons.
- **Collapse.** Without grounding, the latent loss falls to a lower value, which the authors identify as
  collapse.
- **Size.** EIDOS-Small (12.7M) has the best CRPS of the family.
- **Fit.** Top-8 #3.

### 4.2 LeWorldModel / SIGReg [alphaxiv.org/abs/2603.19312](https://www.alphaxiv.org/abs/2603.19312)

- **Method.** End-to-end JEPA with only two terms: MSE prediction plus SIGReg (isotropic-Gaussian matching
  through random 1-D projections, Epps-Pulley statistic). No EMA and no stop-gradient. One effective
  hyperparameter, λ ≈ 0.1. 15M parameters, trained on one GPU in hours.
- **Findings.** Adding a reconstruction decoder did not help control. Dropout of 0.1 in the predictor helped.
- **Fit.** The alternative to EIDOS for #3. Caveat: SIGReg is weak when the data has low intrinsic dimension
  (their Two-Room case). Alma's per-series states may sit in that regime, so EIDOS-style grounding is the
  safer first try.

### 4.3 JEPA-TTT [alphaxiv.org/abs/2610.00722](https://www.alphaxiv.org/abs/2610.00722)

- **Method.** Adapt only the dynamics predictor at test time, with the encoder and reward head frozen and
  the loss in latent MSE. Persistent across episodes with dense replay.
- **Results.** Planning score 0.267 → 0.678 and latent error −83% under dynamics shift (control tasks, not
  forecasting). Persistence was critical: 0.729 vs 0.289 with per-episode reset.
- **Fit.** This is `TSAR_TTA_*`. The literature suggests two changes:
  - adapt only `dynamics` + `step_emb` (+ `decode`), never `lower`/`upper`;
  - a per-dataset persistent variant: adapt across the series of one task in order, using only contexts.
- Judge only on `TSAR_ZS=1` / TEST-like protocol, and log `predict_seconds`. Leakage notes in §6.

### 4.4 Towards Financial World Modeling [alphaxiv.org/abs/2610.09048](https://www.alphaxiv.org/abs/2610.09048)

- **Results.** 18 encoder strategies over 32 months. The supervised multi-head encoder (return + volatility
  + spread) was best: return IC 0.0311 vs 0.0272 single-task, volatility 0.0941 vs 0.0883.
- **SSL.** Mostly at or near a random-ViT floor. LeJEPA with time-warp views was the best SSL method.
- **Scaling.** ViT-Tiny beat ViT-Base at 10× FLOPs, with WSD and annealed checkpoints.
- **Latent organisation.** Forecast skill and latent organisation are uncorrelated (ρ = −0.19).
- **Variance.** Evaluation period explains 97-99.9% of variance, against seeds.
- **Fit.** Two cheap ideas and one evaluation lesson:
  - **(a) Multi-head auxiliary targets.** From each imagined state z_{t+k}, predict log future scale and
    the future-vs-current level change in addition to quantiles. Add a `nn.Linear(D, 2)` and an L1 loss
    with weight 0.1. Cost ~0. Expected gain is small and uncertain.
  - **(b) Time-warp views.** If any SSL is used, time-warp / re-aggregation views were the best, which is
    synth3's `reagg` augmentation.
  - **(c) Evaluation.** Variance across datasets dominates variance across seeds, so keep paired per-task CIs
    (tools/compare.py) rather than more seeds.

### 4.5 Looped / recurrent depth

**LoopWM** [alphaxiv.org/abs/2606.18208](https://www.alphaxiv.org/abs/2606.18208)

- **Method.** Contractive diagonal state retention, Poisson-sampled loop count, truncated BPTT, adaptive exit.
- **Evidence.** On text world-model tasks only (ScienceWorld/ALFWorld), with no forecasting evidence. Our
  LOOPS = 3 run lost under machine load and should be re-run at fixed steps before any conclusion.

**Test-Time Forecasting** [alphaxiv.org/abs/2610.19552](https://www.alphaxiv.org/abs/2610.19552)

- **Method.** Looped linear attention ≈ DMD/Koopman propagator estimation.
- **Results.** Loop scaling improves out-of-distribution ODE/PDE rollouts.
- **Concrete tip.** Inject the input by **concatenation** at each loop, not by addition alone.

**Recommendation.** If LOOPS is retried, use concatenation injection `[h, e]` → Linear inside
`LoopedDynamics`, and evaluate loop-count scaling at test time (train with mean 3, test with 6).
Low priority.

### 4.6 RSSM / Dreamer, TD-MPC2 [alphaxiv.org/abs/2310.16828](https://www.alphaxiv.org/abs/2310.16828)

- **Ideas.** Stochastic latent with a KL balance (RSSM), and a decoder-free latent consistency loss with
  rollout-discounted weights (TD-MPC2).
- **Transferable piece.** TD-MPC2 weights the k-step latent loss by ρ^k (ρ ≈ 0.5), which is cheap to add to
  `latent_consistency`. Use it with #3 so that far-future latent errors do not dominate.
- **Stochastic latent.** A Dreamer-style stochastic latent with sampled trajectories would give
  trajectory-coherent intervals, but it needs many samples at inference. Not worth it on the M3 while the
  quantile head is calibrated.

### 4.7 Other 2026 TS "world model" papers

- WorldTS [2609.31162](https://www.alphaxiv.org/abs/2609.31162), TimeWorld
  [2610.34797](https://www.alphaxiv.org/abs/2610.34797), Pythia
  [2610.05240](https://www.alphaxiv.org/abs/2610.05240) and BeliefGraph-JEPA
  [2610.05409](https://www.alphaxiv.org/abs/2610.05409) are covariate-, action- or multimodal-conditioned.
  Our bench is univariate, so they are not applicable now.
- Phys-JEPA [2606.16076](https://www.alphaxiv.org/abs/2606.16076) is multivariate physical systems. I did not
  find an ablation transferable to univariate zero-shot.

## 5. Small-model techniques (c)

| Paper | Size | Key result | Relevance |
|---|---|---|---|
| Reverso [2602.17634](https://www.alphaxiv.org/abs/2602.17634) | 0.2-2.6M | GIFT MASE 0.706 (≈ Timestral-5M). No synthetic data +6.5%, no augmentation +2.3%, attention decoder head −2%, DeltaNet/long-conv beat attention | Already in arch_plan (CTX_READ = attention decoder; synth3; AUG) |
| Fracast-0 [2609.32209](https://www.alphaxiv.org/abs/2609.32209) / TinyCast [2608.15767](https://www.alphaxiv.org/abs/2608.15767) | 85-147K | WQL 0.545-0.564 on GIFT; periodogram-detected periods as sin/cos channels; "commit" loss vs seasonal copy | Our seasonal_ref covers most of it; a periodogram period detector would replace the given `m_season` when it is wrong (e.g. ETT-15T: daily = 96, weekly = 672) |
| TempoPFN [2510.25502](https://www.alphaxiv.org/abs/2510.25502) | ~small | Generator leave-one-out: OU/regime SDE +26%, CauKer +14% | synth3 already includes both |
| Mamba4Cast [2410.09385](https://www.alphaxiv.org/abs/2410.09385) | small | synthetic-only SSM; competitive zero-shot | supports synthetic-first |
| Tiny-TSM [2511.19272](https://www.alphaxiv.org/abs/2511.19272) | 23M | one A100, strong GIFT | recipe-over-scale evidence |
| Synthetic Data Distillation [2609.09586](https://www.alphaxiv.org/abs/2609.09586) | 4M-2.5B | 1.2-1.85× faster convergence | top-8 #5 |
| Loss-guided selection [2609.37255](https://www.alphaxiv.org/abs/2609.37255) | any | central-80% stratified beats full data | top-8 #6 |
| Portfolios of small models [2510.06419](https://www.alphaxiv.org/abs/2510.06419) | small | a portfolio of small pretrained specialists can rival one large model | possible later: 2-3 frequency-group specialists of 1-2M each, with routing by `freq`. Costs extra training runs, so not now |

## 6. Leakage flags

These are concrete risks, in order of severity.

1. **No pretrained FM weights, outputs or embeddings** for distillation, initialisation, RFL scoring or
   ensembling.
   - Moirai 2.0 trained on the GIFT-Eval train split, and all 13 VAL families are GIFT-Eval families.
   - Chronos-2's corpus partially overlaps GIFT-Eval training portions, and its solar data shares the NREL
     source with VAL `solar`.
   - Timestral's real post-training corpus is undisclosed.
   - Toto 2.0 trained on Datadog observability, the BOOM source, so it overlaps TEST `boomlet`.
   - Any teacher signal therefore imports knowledge of VAL and/or TEST families.
   - Top-8 #6 must use **Alma's own checkpoints** as the reference.
2. **GiftEvalPretrain / LOTSA / Chronos corpora.** These are "non-leaking" only relative to GIFT-Eval
   *test* windows.
   - They contain `favorita_sales` (same Favorita source as VAL `favorita_stores`); the 2609.37255 figure
     shows it in the LOTSA/Chronos pool.
   - They contain NREL solar.
   - They may contain other retail or energy sources that relate to TEST rossmann/entsoe.
   - Before adding any of them, run `bench.py overlap --split val` and `--split test`, and drop whole source
     families, as was done for fred_md, exchange_rate and covid_deaths.
3. **TSMixup on real data** must mix only TRAIN series. The Generic-Transformer paper's "leaky" variant
   shows that leakage inflates scores.
4. **Test-time adaptation (JEPA-TTT style).**
   - Adapt on contexts only, never targets.
   - Per-series is clean.
   - Persistent per-dataset adaptation is transductive across series. It is allowed only if declared up
     front and run identically on TEST, and only on series whose forecast windows do not overlap in
     calendar time with another series' context. Cross-series temporal leakage is possible in panels such as
     LOOP_SEATTLE and entsoe.
5. **Selection on VAL.** Every top-8 item adds a knob, and tuning several on bench val (30 tasks) overfits
   it. Pre-register the flags tested, use the fixed 2500 steps and paired CIs, and keep TEST for the single
   confirmatory run.
6. **SDD, clamps, schedules and Muon** introduce no leakage.

## 7. Suggested order

One heavy job at a time.

1. #1 clamps and #2 decay/averaging. These are inference-only or schedule-only, so they can be checked on
   the existing best checkpoint plus one 2500-step pair.
2. #3 grounded latent loss, with the state-off ablation. This is the world-model evidence and should be
   prioritised for the claim.
3. #4 DQS and #6 RFL-online, as a 2500-step batch.
4. #5 SDD, once synth3 rows dominate the batch, so SDD has rows to act on.
5. #7 Muon, then #8 trained flip-invariance, judged on cov80 as well as wql.
6. Then scale to 2-5M, choosing deeper rather than wider.

## Sources (web, beyond alphaXiv)

- [google-research/timesfm (GitHub)](https://github.com/google-research/timesfm)
- [TimesFM 2.5, Hugging Face Transformers docs](https://huggingface.co/docs/transformers/model_doc/timesfm2_5)
