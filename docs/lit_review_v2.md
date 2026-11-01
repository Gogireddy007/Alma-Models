# Literature review v2: generative and world-model papers for Alma's long-horizon and interval gaps

Date: 2026-10-29. Author: Dr. Aarav Iyer. Papers 1-3 read in full; papers 4-5 from the lead's summary. No compute run.

Our runs so far:

| Run | val wql_rel | Note |
|---|---|---|
| b_base256 | 0.7258 | control |
| JEPA + VICReg | 0.7246 | neutral |
| JEPA + SIGReg | 0.7322 | worse |
| c_mix | 0.7212 | |
| big2 | | cov80_error 0.088 |

**None of the five papers is zero-shot, and none compares probabilistically with Chronos or TimesFM.** Every reported gain comes from per-dataset training.

## 1. DV-Flow (2610.18621)

**Mechanism.** The velocity is the sum of a context-only deterministic term and a local flow term:

v = f_θ(x) + g_φ(y_τ, τ, x)

L = E‖(y−ε) − f_θ − g_φ‖² + λ·MAE(y, f_θ), with y_τ = (1−τ)ε + τy and λ = 0.3.

Theorem 1: in residual coordinates the flow transports noise to y − f_θ, which has median 0 when f_θ is MAE-trained. f_θ never enters g_φ's inputs, so the target is stationary and there are no gradient spikes. The other ingredients are:
- OU-kernel noise, exp(−|d|/l);
- SATN normalisation: remove the seasonal mean, centre on the last value, and divide by a σ_t that grows linearly with lead time.

**Numbers.** CRPS_sum on 6 multivariate GluonTS sets, with horizon 24 or 30:
- Electricity: 0.0197, against 0.0206 for TimeGrad.
- KDDCup: 0.0578, against 0.111.
- Clean ablation against vanilla FM with SATN and the GP prior: Electricity 0.0247 → 0.0197, Traffic 0.0461 → 0.0386.
- The GP prior alone takes KDD from 0.171 to 0.141.
- MAE beats MSE as the anchor loss.
- Chronos and TimesFM are not compared.

**Cost.** 1.3× parameters, 1.4× training time and 1.1× inference relative to a single network; 150 steps with Dopri5.

**Skepticism.** These are summed multivariate scores at short horizons, and each model is trained per dataset. Nothing here covers h > 64.

**For Alma.** Our q50 is already an MAE-anchored f_θ. What is missing is coherent sample paths for the chunk feedback (#1).

## 2. WorldTS (2609.31162)

**Mechanism.** One causal-patch encoder is shared across history and future.
- Stage 1 trains 10‖Ẑ − sg Z‖₁ + 15(1 − cos) + 0.1·VICReg. Raising the VICReg weight to 0.5 hurts.
- Stage 2 freezes everything except the decoder and trains ‖x − D(Ẑ)‖₁ + α‖x − D(Z)‖₁, decoding both the predicted and the encoded true future states.
- Covariates enter through MLP fusion and low-rank adapters.

**Numbers.** Point MSE and MAE only.
- It is best on all 12 EPF and wind sets; NP scores 0.297 against 0.325 for KITE.
- On MMSP (h = 24) it scores 0.061, against 0.067 for TFT, 0.090 for Chronos-Bolt+UniCA and 0.098 for TimesFM+UniCA.
- MMSP ablations: one-stage observation-space training scores 0.085, dropping the latent loss 0.078, and dropping covariates 0.101.

**Cost.** Small: D is 128-256.

**Skepticism.** Covariates (satellite imagery and NWP) carry most of the gain. The models are per-dataset and produce no intervals.

**For Alma.** Dual decoding is free, because our causal encoder already yields z_{t+8k} (#2). Covariates belong only to Hoffmann's chronos2_cov track, and VAL has just 3 such tasks.

## 3. ACTS-WM (2610.37963)

**Mechanism.**
- An inverse dynamics model (IDM) computes z = MLP[F(X₁); F(X₂); e_τ].
- z is quantised by VQ with K = 16, d = 16 and M = 2 slots.
- A forward model (a 1-D U-Net) reconstructs X₂, with loss ‖X̂₂ − X₂‖² + 0.05‖z − sg a‖².
- A policy π(X₁) regresses z at evaluation.
- A V-JEPA-2 encoder with an action-conditioned predictor feeds frozen linear, ridge or NLH probes.

**Numbers.** Linear-probe MSE at H = 48:
- ETTh1: 0.100 → 0.084.
- ETTh2: 0.478 → 0.351.
- WikiTraffic: 1.167 → 1.063.

Against that:
- With the NLH head the method loses on every set.
- It degrades at H = 168 and 336.
- The baselines are JEPA probes only; multivariate ETTh1 at 1.218 is far from SOTA.
- Pretraining includes the evaluation sets' training splits.
- Training used two GB10 nodes for 200 epochs.

**Skepticism.** At evaluation the action is a deterministic function of the context, so it adds capacity rather than information or uncertainty.

**For Alma.** Use a discrete regime latent that is *sampled*, not regressed (#4).

**Leakage.** The IDM sees the future, so any evaluation path that uses the posterior leaks targets.

## 4. EpicWorldModel (2610.05996)

A flow-matching latent predictor, SIGReg, one-step MeanFlow sampling, and dispersion used as uncertainty. Its +20 points are on robotics under partial observability. Our SIGReg run lost, so an isotropic latent is not free for us.

**For Alma.** A stochastic flow transition (#3).

## 5. TEM (2610.31889)

Energy heads trained with Langevin negatives, giving inference-time sharpening. For us it could only re-score sampled paths. Contrastive MCMC is fragile at 2500 steps, and the DINOv2 alignment is forbidden; the only allowed substitute is our own EMA encoder. **Not in the top 4.** Revisit only once #1 or #3 gives us a sampler worth sharpening.

## Ranked top 4 (gain per unit compute)

Screen each with `TSAR_BENCH=1 TSAR_STEPS=2500` at 0.37M parameters, with the flag default off. Report wql and cov80 separately for h ≤ 64 and h > 64.

**1. Flow sample-path head: `TSAR_DVF=1` (train_calib.py).**
- **Head:** g_φ is an MLP of about 40k parameters taking (y_τ ∈ ℝ⁶⁴, Fourier(τ), z_t). Its global velocity is the decoded q50 path, which is not detached.
- **Loss:** add `TSAR_DVF_W`=0.3 × ‖(y − ε) − (q50 + g_φ)‖², computed in model space.
- **Noise:** OU noise with `TSAR_DVF_L`=8, scaled by a σ_k that is linear in lead and fitted on training rows only.
- **Inference:** `TSAR_TTA_LONG=flow` draws S = 16 paths with 8 Euler steps each. Each path feeds back through the chunks on its own, and the forecast takes the empirical quantiles. The quantile head stays in use for h ≤ 64.
- **Expectation:** cov80 error for h > 64 falls by about a third. wql improves by 1-2% on h ≥ 96 hourly and sub-hourly data, about 0.5-1% overall.
- **Cost:** +15%, about 8 min per screen.
- **Leakage:** none.

**2. Dual decoding plus the L1-cosine latent loss: `TSAR_DUALDEC=0.5`, `TSAR_JEPA_FORM=l1cos` (train_latent.py).**
- **Dual decoding:** decode the encoder's own z at origin t+8k through the same quantile head as the rolled ẑ_k, and add α × pinball. A sub-flag decides whether that z is detached.
- **Latent loss:** λ‖ẑ − sg z‖₁ + λ_sim(1 − cos), with λ ∈ {1, 3} and VICReg at 0.1.
- **Expectation:** −0.3 to −0.8% wql, mostly at short horizons.
- **Cost:** +5%, the cheapest of the four.
- **Leakage:** none.

**3. Stochastic flow transition: `TSAR_TRANS=flow` (train_arch.py).**
- **Transition:** z_{k+1} = GRU(z_k) + g_φ(n_τ, τ, z_k), with the GRU as the global velocity.
- **Training target:** sg(z_{t+8(k+1)}) − GRU(z_k).
- **Sampling:** `TSAR_FLOW_MF=1` uses MeanFlow for one-step sampling of 16 paths.
- **Gating:** run it only after Mehta's rssm screen, and only if the Gaussian prior collapses or helps cov80 just a little.
- **Expectation:** it fixes collapse at the source. The cov80 gain is likely; the wql gain is ±1%.
- **Cost:** +25-40%.
- **Leakage:** none.

**4. Discrete regime latent: `TSAR_TRANS=rssm TSAR_RSSM_CAT=16x2` (train_arch.py).**
- **Posterior:** 2×16 categorical codes that see z_{t+64}, in training only.
- **Prior:** a head π(z_t) trained with KL, using the 5:1 split and free nats.
- **Inference:** codes are sampled from the prior, never the argmax or the posterior.
- **Expectation:** about −0.3% wql with modest cov80 gains. The evidence is weak.
- **Cost:** +10%.
- **Leakage:** run an A3-style audit to confirm that evaluation never calls the posterior.
