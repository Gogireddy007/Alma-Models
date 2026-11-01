# Timestral anatomy, and where a world model fits (2026-10-31)

Source: the Apache-2.0 inference package at `.venv/lib/python3.12/site-packages/timestral/` (paths below are
relative to it), the three released `config.json` files in the local Hugging Face cache (configs only; no weights
were loaded), `docs/lit_review.md` §2.1 and `docs/arch_plan.md` §1. The package has no training code: statements
about training are either from the report (via our lit review) or inferred from the inference code, and are marked
as such. Our re-implementation is `wm/tswm.py` (family `tswm`).

## 1. Patching and causal scaling

| Item | What the code does | Where |
|---|---|---|
| Patch | Non-overlapping patches of `patch_size` = 32 steps; the context is left-padded to a multiple of 32 with unobserved zeros | `config.py:42`, `forecaster.py:70-77` |
| Missing values | Forward-filled before the model; the observation mask travels with the values | `forecaster.py:60-67`, `forecaster.py:346` |
| Causal statistics | Running mean and Bessel-corrected std over every observed value up to `t`, in float64 (Welford/cumsum form); std floored at `minimum_scale` = 1e-6 | `scaler.py:14-59` |
| Per-patch frame | Each patch uses the statistics at its **last** step (everything observed up to the end of that patch), broadcast to the patch | `scaler.py:62-79` |
| Squash | `asinh((x - loc) / scale)` on observed steps, 0 on unobserved ones | `scaler.py:88-101`, `model.py:122-133` |
| Global stabiliser | Off, and forbidden with the advancing frame | `config.py:59`, `config.py:77-78` |
| Output frame | The network predicts the whole block in the frame of the **last context patch** (the anchor), inverted with `sinh(.) * scale + loc` | `model.py:227-229`, `scaler.py:104-110` |
| Advancing refine frame | After the network, horizon patch h (h >= 1) is re-read in the frame of "context + the block's medians of patches 0..h-1": the medians are folded into float64 Welford states one patch at a time, and the quantiles are mapped `sinh(pred) * scale_h + loc_h`. Patch 0 keeps the anchor frame. With perfect medians this equals causal per-patch normalisation of the true series | `scaler.py:165-208`, `forecaster.py:458-483` |

Reading of the advancing frame: training (report, lit_review §2.1) normalises each forecast patch with statistics up
to the preceding patch, i.e. with the **true** earlier horizon values (teacher forcing). Inference substitutes the
predicted medians. Because the statistics are a running mean/std over a context of up to 32k steps, the frame moves
little within a block; it is a correction, not a re-centring.

## 2. Block design

| Item | Detail | Where |
|---|---|---|
| Patch embedding | `ResidualBlock(2P, 4D, D)` with bias on `[values, missing]` per step: `out(silu(hidden(x))) + residual(x)` | `model.py:35-51`, `layers.py:30-42` |
| Norm | Pre-norm RMSNorm, eps 1e-6, computed in fp32 | `layers.py:18-27`, `config.py:52` |
| Attention | Fused QKV, no bias. The weight layout is `(qkv, head_dim, n_heads)`, i.e. heads interleaved (a fixed permutation, irrelevant when training from scratch) | `layers.py:157-166`, `config.py:53` |
| Sigmoid gate | The SDPA output is multiplied by `sigmoid(x W_g)` (x = the normed input), then `W_O` ("sdpa_sigmoid") | `layers.py:159`, `layers.py:182-188`, `config.py:55-56` |
| Partial RoPE | Time attention only. `rope_dim = head_dim * 0.5`; the inverse-frequency table is zero-padded past `rope_dim/2`, so with the concatenated-halves form only dims `i` and `i + head_dim/2` for `i < rope_dim/2` rotate; theta 1e4 | `layers.py:58-113`, `config.py:50-51`, `config.py:84-87` |
| MLP | SwiGLU with one fused input projection, `w3(gate * silu(up))`, no bias; hidden = 8/3 D in every release | `layers.py:45-55`, `config.py:54` |
| Layer kinds | `time`, `variate`, `time_variate`, `variate_time`; each runs its attention(s) in order, each a pre-norm residual, then one MLP. Released schedules: `time, time, time, time_variate` repeated | `config.py:14-23`, `layers.py:191-223` |
| Variate attention | Over the rows (variates) of the same series per patch, no positions. With a univariate series there is one key, so it reduces to `W_O(W_V x * sigmoid(W_g x))` | `layers.py:170-188`, `masks.py:121-146` |
| Positions | `0..L+h-1` minus the row's first observed patch. Under RoPE only differences matter, so the anchoring changes nothing within a row; it matters only for aligning rows of one series | `masks.py:50-53`, `masks.py:149-153` |
| Final norm | RMSNorm after the stack | `layers.py:259`, `layers.py:269` |
| Regularisation | None: no dropout, no stochastic depth (report) | lit_review §2.1 |

## 3. Decoupled Horizon Forecasting (DHF)

- **Prediction tokens.** Every horizon patch gets the **same** learned vector `forecast_query` (zero-initialised),
  appended after the context tokens: the sequence is `[Enc ; Dec]`. Tokens differ only by their RoPE position
  `L..L+h-1` (`model.py:42`, `model.py:53-54`, `model.py:215-216`, `masks.py:50-53`).
- **Shared stack.** Encoder and prediction tokens run through the same layers (`model.py:153-162`). The head reads
  only the prediction tokens (`model.py:162`).
- **Masks (no covariates, `masks.py:34-47`).** Enc to Enc: causal over valid keys, diagonal always open
  (`masks.py:25-31`). Enc to Dec: never (`masks.py:42`). Dec to Enc: every valid context key (`masks.py:43`).
  Dec to Dec: **causal** (`tril`, `masks.py:44`). A key is valid once the series has started
  (`attend_missing_patches`, `masks.py:12-22`). The attention runs two query passes so the empty Enc-to-Dec quadrant
  is skipped (`layers.py:116-126`, `layers.py:174-179`).
- **Block size K.** `block_size` = 32 patches in every release (= 1024 steps). One `predict_block` call decodes at
  most `per_shot_horizon_patches` = min(32, K) patches (`forecaster.py:108`, `forecaster.py:116`,
  `forecaster.py:427`).
- **More than K patches.** Block-autoregressive in observation space: after each block the **median** is appended
  to the context (extreme values zeroed), the context is trimmed to `max_context`, and the next block is decoded
  (`forecaster.py:396-456`, feedback at `forecaster.py:438-453`). Every block is refined into the advancing frame
  first (`forecaster.py:432`).
- **Training layout (inferred from the covariate path).** `_predict_window` runs `[Enc ; Dec]` over a whole window
  with `n` prediction tokens: the token at patch p belongs to the block anchored at `s + K floor((p - s) / K)`, reads
  `Enc[j < anchor]`, and is causal within its block (`masks.py:56-101`, `model.py:231-273`). So one pass trains many
  origins and horizons with no context masking, but each future patch is predicted from **one** origin, its block
  anchor. Our harness instead trains every origin x K leads.
- **Inference extras.** "Clampsym" TTA by default: forecast `x` and `-x`, reflect and average the second
  (`-Q(1-p)`), and clip at 0 for series whose context is non-negative (`forecaster.py:220-231`); plus a cap at
  `[min - 1e4 s, max + 1e4 s]` of the context (`forecaster.py:323-330`). Our bench runs Timestral with these on.

## 4. Monotone quantile head

- `ResidualBlock(D, 4D, P * Q)` without bias, per prediction token, then reshape to `[P, Q]` (`model.py:79-98`).
- `monotone_cumsum` (`model.py:57-76`): the median channel is free; the others fan out as
  `median -/+ cumsum(width_factor * dp * softplus(raw))`, where `dp` is the gap to the next level inward, so unequal
  knot spacing is respected. No crossing by construction.
- Knots: 21 levels {0.01, 0.05, 0.10, ..., 0.95, 0.99}, `quantile_width_factor` 3.0 in every release (the package
  default is 2.0, `config.py:63`). The loss is pinball only (report).

## 5. Released configurations (local `config.json`, configs only)

| | Timestral-5M | Timestral-28M | Timestral-209M |
|---|---|---|---|
| d_model / heads / head_dim | 192 / 3 / 64 | 384 / 6 / 64 | 768 / 12 / 64 |
| MLP hidden | 512 | 1024 | 2048 |
| Layers | 8 (`time` x3 + `time_variate`) x2 | 12 (x3) | 24 (x6) |
| patch / block_size / context | 32 / 32 / 32768 | same | same |
| rope theta / partial | 1e4 / 0.5 | same | same |
| quantile knots / width factor | 21 / 3.0 | same | same |
| norm eps / biases / gates | 1e-6 / none / sdpa_sigmoid (time and variate) | same | same |
| asinh / min scale / attend_missing / covariates | on / 1e-6 / on / on | same | same |
| Our VAL (wql_rel / mase_rel / cov80_err) | 0.5846 / 0.7299 / 0.047 | 0.5718 / 0.7124 / 0.036 | 0.5639 / 0.7062 / 0.034 |

Parameter count of the 5M by hand: 6 time layers x 479.6k + 2 time_variate layers x 663.9k = 4.21M, patch embedding
0.21M, head 0.79M: 5.2M. Scores are from `docs/research_brief.md:53-57`.

## 6. Training recipe (report, via docs/lit_review.md:154-168 and docs/arch_plan.md:45-50)

- Pretrained **only on synthetic data** at an 8k context, then post-trained on real + synthetic at a 32k context.
- Pinball loss only, no auxiliary loss; 21 quantiles; no dropout or stochastic depth.
- Gap augmentation (random missing spans) (`docs/arch_plan.md:127`).
- No ablations are published, and the corpus size is undisclosed.

## 7. Which parts are already a world model, and which are not

| Part | World-model role | Verdict |
|---|---|---|
| Causal encoder over patches | A filter: it assimilates observations causally and its per-layer K/V is a memory of the past | **Yes**: the observation model / belief |
| Advancing refine frame | Feeds the median path back, but only into normalisation (output space), after the network | Not a latent dynamic |
| Prediction tokens (DHF) | Identical learned queries, separated only by position, decoded **in parallel**. Each reads the context; the causal Dec-to-Dec mask lets lead k read the hidden states of leads < k *within the same pass*, but nothing is fed back: no input depends on an earlier lead's output, and no state is carried or rolled | **No**: a direct multi-horizon readout |
| Block feedback beyond K patches | Observation-space autoregression (medians appended to the context) | A data-space rollout, not a latent one. It never triggers on our bench (max h = 288 < 1024) |

So on every horizon we evaluate, Timestral is a **direct** multi-horizon model: a learned memory read by K
independent-in-input queries, with no transition function, no imagined state and no dynamics prior. Its strengths
(the gated RMSNorm/SwiGLU/RoPE block, monotone head, asinh scaling, DHF's dense supervision) are all on the
representation and readout side.

## 8. The world model we embed (`wm/tswm.py`)

- `direct` = the Timestral replica: the backbone above at our harness's PATCH = 8, the DHF masks, `forecast_query`
  tokens (plus a zero-initialised drive from the harness's seasonal-naive patch, which every family gets), and the
  monotone head producing the harness's 9 deciles.
- `none` = the world model: lead k+1's token is `forecast_query + drive(ref) + T(z_k)`, where `z_k` is lead k's
  final latent state after the whole stack and `z_{-1}` is the encoder state at the origin. Leads are decoded
  sequentially with a KV cache of the imagined tokens: the stack plus `T` is the transition, the state is carried
  and rolled with no observation feedback, and every lead still reads the context through attention (memory read).
  This is the Transformer-world-model pattern (latent autoregressive imagination), built inside DHF's own
  attention pattern, so `none` and `direct` have identical masks, identical per-token MACs and +2.5% params (only
  `T`). With `T = 0`, `none` reproduces `direct` exactly (tested to fp32 noise), so any difference is learned use of
  the carried state.
- `copy` gives every lead `T(z_{-1})` (no rolling); `zero` zeroes the rolled states before the head.
- Frame: `TSAR_WM_TSWM_FRAME=loc` (default) is the advancing refine frame translated to our harness's NORM=last
  frame (location = the last imagined median of the previous lead), computed from the model's own detached medians
  in training as well, so there is no teacher-forcing gap. `adv` also advances the scale as Timestral does.
- Patch: kept at 8. Timestral's 32 is tied to its 32k context (1024 tokens); at CONTEXT = 256 it would leave 8
  context tokens and K = 2 leads.
- AlmaTS options (2026-10-31, all default-off and bit-identical; details in the `wm/tswm.py` docstring):
  - `TSAR_WM_TSWM_TRANS=ssm` replaces `T` with a selective diagonal complex SSM carried across leads (input-dependent
    step from `z_k`, `wm/ssm.py`'s S6/S5 step; `C` and `Dskip` zero-initialised, so it starts as exactly `direct`).
    +1.7% params at width 1 (374,665 vs 368,361).
  - `TSAR_WM_TSWM_READ=first` lets only lead 0 read the context (E7 in docs/almats_research_program.md): leads >= 1
    see the earlier lead tokens only, so the per-lead bypass of the carried state is closed. `direct` + first is
    still the matched control (lead 0's context read reaches later leads through Dec-to-Dec attention, nothing is
    fed back).
  - Harness: `TSAR_WM_TTA=1` (Timestral's clampsym for every family, eval-only capable) and `TSAR_CONTEXT=512|1024`.
