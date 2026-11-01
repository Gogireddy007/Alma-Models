# ts-autoresearch

You are an autonomous researcher building a time-series **world model** forecaster that beats
strong industry baselines on accuracy, calibration and compute. The full mission is in
`MISSION.md`; the pre-registered protocol and success gates are in `RESEARCH_PLAN.md`.
This file tells you how to run the experiment loop.

Adapted from karpathy/autoresearch for an Apple-silicon laptop (M3, 8 GB, PyTorch MPS).

## The goal

Establish world models as a new frontier architecture for time-series forecasting: the most
accurate, efficient and generalizable forecaster, with a clear world model embedded inside a
Transformer (an explicit latent state that is carried, rolled forward and decoded). It must beat
Mistral's Timestral family by a significant margin (val wql_rel 0.569 for 209M; target well
below that), and beat the other industry baselines (TiRex, Chronos-2, Chronos-Bolt, AutoETS,
AutoTheta, Seasonal Naive) on accuracy, calibration, efficiency and zero-shot generalization,
under pre-registered gates with no leakage.

## Distillation and RL: when they are allowed

- **Distillation from Timestral (or any baseline)** is a diagnostic only, never the headline
  model. Timestral's training data is undisclosed and may include these benchmarks, so a
  distilled student inherits possible leakage, and a student cannot claim a new architecture
  beats its teacher. Allowed use: distill on our own synthetic series to test whether our
  architecture can represent a Timestral-quality function (an architecture vs. data question).
  Keep such runs out of `results.tsv` keep rows.
- **Self-distillation** (our own EMA or larger model as teacher) is fair game (T10).
- **RL** is low priority: the pinball loss is dense and differentiable, so RL adds variance
  without new signal. Its fair uses are choosing curricula or hyperparameters (the autoresearch
  loop already does this) and optimising non-differentiable targets such as interval coverage.

## Setup

Work with the user to:

1. **Agree on a run tag** based on today's date (e.g. `oct29`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**: `README.md`, `MISSION.md`, `RESEARCH_PLAN.md`, `prepare.py`
   (read-only: data, splits, metric), `baselines.py` (read-only), `train.py` (the file you edit),
   and `research_log.md` if it exists.
4. **Verify data exists**: `data/suite.pkl`. If missing, tell the human to run `uv run prepare.py`.
5. **Read the bar to beat**: `baselines/results.tsv` (val split rows). Note the best `wql_rel`
   and `mase_rel` and which model holds them.
6. **Initialize `results.tsv`** with the header row below, and `research_log.md` if missing.
7. **Confirm and go.**

## The metric

`uv run train.py` trains for a fixed 4-minute budget (`prepare.TIME_BUDGET`), then scores the
model on the **val** split of the 15 development datasets. It prints:

```
---
val_wql_rel:      0.912345   <- PRIMARY. Probabilistic loss relative to Seasonal Naive. Lower is better.
val_mase_rel:     0.934567   <- point accuracy relative to Seasonal Naive. Lower is better.
val_cov80_error:  0.041200   <- |coverage of 10-90% interval - 0.80|. Lower is better.
training_seconds: 240.0
predict_seconds:  2.3        <- inference cost for all val forecasts
total_seconds:    246.1
peak_mem_mb:      1123.9
num_params_M:     0.540
num_steps:        1100
```

Extract with: `grep "^val_\|^predict_seconds\|^peak_mem_mb\|^num_params_M" run.log`

**Keep a change only if all of these hold:**
- `val_wql_rel` improves by more than the noise floor (see "First runs").
- `val_mase_rel` does not get worse by more than 1%.
- `val_cov80_error` stays below 0.10.
- `peak_mem_mb` stays below 3000 (this laptop has 8 GB shared memory).

Ties and near-ties go to the **simpler, smaller or faster** model. Deleting code for equal
results is a win. A model with half the parameters or half the `predict_seconds` at equal
accuracy is a win: log it as `keep` with an efficiency note.

## What you CAN do

- Edit `train.py`. Everything is fair game: architecture, latent state, dynamics, losses,
  normalization, sampling, synthetic data generated in code, optimizer, schedule, ensembling.

## What you CANNOT do

- Edit `prepare.py` or `baselines.py`, or change the evaluation in any way.
- Add dependencies. Only what is in `pyproject.toml` main dependencies (numpy, pandas, torch).
  `statsforecast` and `chronos` are for baselines only; do not import them in `train.py`.
- Train on anything except `prepare.load_train_series()` and synthetic data you generate.
  Never read `data/raw/` or `data/suite.pkl` directly.
- Set `TSAR_SPLIT=test`, or score the `test` / `holdout` splits. Those are for the human's
  confirmatory runs only. Looking at them would invalidate the study.

## First runs

1. Run `train.py` unchanged: this is the baseline.
2. Run it twice more with `SEED = 1` and `SEED = 2` (revert the seed afterwards). The spread of
   `val_wql_rel` across the three runs is the **noise floor**. Record it at the top of
   `research_log.md`. An "improvement" smaller than the noise floor is not an improvement.

## Logging

`results.tsv` is tab-separated, untracked by git, one row per experiment:

```
commit	val_wql_rel	val_mase_rel	cov80_error	params_M	predict_s	peak_mb	status	description
```

`status` is `keep`, `discard`, `crash` or `ablation`. Use 0 for metrics of crashed runs.

`research_log.md` is your lab notebook and long-term memory, also untracked. Before each
experiment write one line: the hypothesis and which research track it belongs to. After it,
write one line: what happened and what you learned. Every 10 experiments, write a short summary
of what is working, what is not, and what you will try next. Re-read it before choosing ideas,
so you never repeat a failed idea by accident.

## The experiment loop

LOOP FOREVER:

1. Look at the git state and `research_log.md`.
2. Pick one idea. Write the hypothesis in the log. Edit `train.py`.
3. `git commit -am "<short description>"`
4. `uv run train.py > run.log 2>&1` (never let the output flood your context)
5. Read the summary with the grep above. If it is empty, `tail -n 50 run.log`, fix trivial
   bugs and retry; if the idea is broken, log `crash` and move on.
6. Log the row in `results.tsv` and the outcome in `research_log.md`.
7. If it passes the keep rule, keep the commit. Otherwise `git reset --hard HEAD~1`.

**Timeout**: a run should take about 5 minutes. Kill anything over 10 minutes and treat it as a crash.

**Ablation duty**: the mission requires proof that the world-model parts matter. Every 10
experiments, run one ablation of the current best model (e.g. `DIRECT = True`, no latent
rollout, no stochasticity, smaller latent state) as a separate commit with status `ablation`,
then reset. Record what the ablation costs or saves.

**NEVER STOP**: once the loop starts, do not ask the human whether to continue. Run until you
are interrupted. If you run out of ideas, re-read `MISSION.md`, the research tracks below and
your log, combine near-misses, or try something more radical.

## Phase 2: sustainable scale-up runs

The 4-minute loop finds what works per unit of compute. Scale-up runs test whether the best
design keeps improving with size and time. They use the same `train.py` with environment
overrides and still only score the val split:

```bash
nohup caffeinate -is nice -n 10 env TSAR_RUN=big1 TSAR_MINUTES=180 \
  TSAR_D_MODEL=192 TSAR_N_LAYERS=4 TSAR_N_HEADS=6 TSAR_LR=2e-3 TSAR_EVAL_MIN=15 \
  uv run train.py > runs/big1.log 2>&1 &
grep EVAL runs/big1.log          # the scaling curve
```

- One heavy job at a time: this laptop has 8 GB, and parallel runs distort timing.
- Checkpoints go to `data/ckpt/<run>.pt` every 10 minutes. Re-running the same command
  resumes where it stopped (after a crash, a reboot or the lid closing).
- Log each run in `research_log.md` with its curve, and compare at equal wall-clock minutes.

## Research tracks to draw ideas from

Spread effort across tracks; tag each experiment with its track in the log.

- **T1 latent dynamics**: rollout cell (GRU, SSM, linear/Koopman, MLP), latent size,
  rollout patch size, multi-step latent consistency losses, teacher forcing in latent space.
- **T2 self-supervised objectives**: auxiliary masked-patch reconstruction, JEPA-style
  latent prediction with an EMA target encoder, contrastive views. Keep only what helps the metric.
- **T3 efficiency**: weight-shared/looped encoder blocks, adaptive depth, smaller models,
  faster data sampling (the Python batch sampler is likely a bottleneck: more steps in the
  same 4 minutes is free accuracy).
- **T4 probabilistic**: quantile head vs. distribution head vs. sampled latent trajectories,
  calibration-aware losses, quantile crossing fixes.
- **T5 generalization**: synthetic data (random seasonal/trend/AR/regime-switch generators,
  KernelSynth-style GP mixtures) mixed into training, dataset balancing, frequency handling,
  robustness to unseen domains (the holdout datasets are finance, macro, energy, retail and
  intermittent demand, so avoid tricks that only fit the dev datasets).
- **T6 robustness**: training-time masking of context values, outlier-robust scaling,
  handling short contexts.
- **T8 world model embedded inside a Transformer**: the Transformer carries an explicit
  latent world state as part of its computation, not just a latent vector at the end.
  Examples: a dedicated state token (or a few memory tokens) that every layer reads and
  writes, updated patch by patch by a learned transition block inside the stack
  (recurrent-memory style); future prediction tokens that attend only to the rolled-out
  state, not to raw history; a JEPA-style loss making the in-Transformer state at patch t+k
  predictable from the state at patch t. The ablation is the same Transformer with the
  state pathway removed. This track is the user's priority alongside T1.
- **T9 continual / test-time learning**: the model keeps learning after pretraining. Inside
  `predict`, adapt the dynamics (transition cell + decoder) on the context itself: origins
  inside the context have known futures, so the multi-origin loss is computable without peeking.
  Keep the encoder frozen (JEPA-TTT, arXiv 2610.00722, shows large gains from adapting only the
  dynamics predictor). Variants: per-dataset vs per-series adaptation, persistent adaptation
  across series of one dataset, adapter-only updates (LoRA/bias), replay of pretraining data to
  stop forgetting. Log the extra `predict_seconds` honestly.
- **T10 self-improving architectures**: the model or the loop improves itself. Examples:
  error-feedback correction (a small head reads the model's own in-context errors and corrects
  the forecast), iterative refinement (feed the forecast back as input for a second pass),
  self-distillation from an EMA teacher, and the autoresearch loop itself evolving `train.py`.
- **T11 looped world models**: LoopWM (arXiv 2606.18208): a weight-shared dynamics block looped
  T times per latent step with a contractive, spectrally constrained residual
  (h <- A h + (1-A) e + R(h, e)). Training samples T, and test time can raise T. Also looped
  (weight-tied) encoder layers as cheap depth. Test-Time Forecasting (arXiv 2610.19552) shows
  recurrent depth learns in-context dynamics propagators that generalise to unseen systems.
  `TSAR_LOOPS=<mean loops>` switches the transition from GRUCell to the looped core.
- **T7 hybrids**: blend the network with a simple statistical forecast computed in-code
  (e.g. seasonal naive or a damped trend), or learn a residual on top of it.

## The bar

Beat **Timestral** by a significant margin: val wql_rel 0.569 (209M), 0.577 (28M), 0.591 (5M).
Matching it is not success. Also beat TiRex (0.603), Chronos-2 (0.621) and AutoETS (0.781).

## Notes from "Towards Financial World Modeling" (arXiv 2610.09048)

- A supervised **multihead** encoder (return + volatility + spread) beat single-task models
  and most SSL objectives. Auxiliary targets that share a representation help (idea: predict
  future scale/volatility as extra heads, T2).
- SSL was mixed: the best (LeJEPA with time-warp views) only slightly beat a random encoder.
  Any SSL loss here must beat the pinball-only control.
- Smaller models were more compute-efficient at equal FLOPs, which matches our big1 result.
- Forecast accuracy and latent organization are nearly uncorrelated (rank −0.19): a good
  world-model state needs more than one target.
- Performance varies far more across periods than across seeds: evaluate across many datasets.

## Ideas from Timestral (Mistral, arXiv 2610.37585), the current leaderboard leader

Timestral-5M is the efficiency bar: 5.2M params (d=192, 8 blocks, 3 heads, SwiGLU 512).
Its report has no ablations, so each of these is an untested hypothesis here:

1. **Multi-origin rollout loss (T1)**: encode causally once, keep a latent state at every
   patch, and roll the dynamics forward from many origins per sequence, scoring every step.
   Many horizons and origins per forward pass, and no context is masked.
2. **Monotone quantile head (T4)**: predict the median; build other quantiles as
   median ± cumsum(softplus(increments)) so they cannot cross.
3. **Symmetric averaging at inference (T6)**: forecast x and −x, reflect the second
   (q_τ ← −q_{1−τ}), average. Training-free; costs 2× inference, so log `predict_seconds`.
4. **asinh after standardization (T6)**: mean/std over observed values, then asinh;
   invert on output. Helps heavy tails and counts.
5. **Gap augmentation (T6)**: hide variable-length gaps (zero + mask) in the context during
   training. Rate and length are yours to set.
6. **Synthetic-first curriculum (T5)**: spend most of the budget on generated series
   (trend, multi-resolution seasonality, noise, regime changes, spikes, level shifts,
   count-valued), then the rest on real + synthetic. Watch calibration: too noisy a prior
   widens intervals, too clean a prior makes the model overconfident.
7. **Cheap blocks (T3)**: RMSNorm, SwiGLU, a sigmoid gate on attention output, RoPE on half
   of each head's dimensions, no dropout.
8. **Pinball only (T2 control)**: Timestral uses no auxiliary loss. Any auxiliary loss you
   add must beat this control.

Useful references: EvoCast (arXiv 2610.04517) and TimeRSI (arXiv 2610.19725) are agents
doing similar searches; "AutoResearch at Production Scale" (arXiv 2609.30541) lists failure
modes such as search stagnation and memory decay. Use the log to avoid them.
