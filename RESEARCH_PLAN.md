# Research plan and pre-registered protocol

Written on 2026-10-29, **before any test or holdout result was looked at**. Changes to the
gates below must be dated and justified in this file; they may never be made after looking at
test or holdout numbers.

## What counts as a world model here

A forecaster qualifies as a world model in this project only if it has all three:

1. an **encoder** that compresses observed history into an explicit latent state;
2. a learned **transition** that advances that latent state over future time without new
   observations (a rollout in latent space), shared across horizons;
3. a **decoder** that maps any rolled-out latent state to a forecast distribution.

A model that maps history directly to all horizons does not qualify, however large. Whether
the world-model structure *helps* is an empirical question answered by the ablations
(`DIRECT = True` in `train.py` is the first one).

## Benchmark

- **Data**: 20 public datasets from the Chronos zero-shot benchmark collection
  (`autogluon/chronos_datasets`), spanning tourism, M1/M3/M4 competitions, healthcare, banking
  (NN5), COVID, weather, retail (M5, Dominick), energy (ERCOT, Australian electricity),
  traffic, macro (FRED-MD), FX and intermittent spare parts. Frequencies from 30-minute to
  quarterly; horizons from 8 to 56 steps.
- **Development datasets (15)**: used for training and the val split.
- **Holdout datasets (5)**: `exchange_rate`, `monash_fred_md`, `monash_australian_electricity`,
  `dominick`, `monash_car_parts`. Never trained on, never seen during iteration.
- **Splits** (chronological, per series): train = everything before the last two horizons;
  val = second-to-last horizon; test = last horizon. Context is capped at 1,024 points.
- **Sampling**: up to 300 series per dataset are scored (fixed seed 1234).
- **Metrics**: WQL over 9 quantiles (≈ CRPS), MASE of the median, and 10–90% interval
  coverage. Headline numbers are geometric means across datasets of each model's ratio to
  Seasonal Naive.
- **Compute**: parameters, peak memory, training seconds and total inference seconds, all on
  the same laptop (Apple M3, 8 GB).

## Baselines

Run with `baselines.py` on exactly the same windows:

| Family | Models | Status |
|---|---|---|
| Statistical | Naive, Seasonal Naive, AutoETS, AutoTheta, AutoARIMA (statsforecast; 512-point input window) | included |
| Foundation models, zero-shot | Chronos-Bolt small/base, Chronos-2, TiRex, TimesFM-2.5, Timestral 5M/28M/209M | included where the package installs on macOS |
| Global ML | LightGBM via mlforecast | phase 2 |
| Supervised neural | N-HiTS, PatchTST, DeepAR, iTransformer (neuralforecast), tuned with an equal budget | phase 2 |
| Other foundation models | Moirai-2, Toto-2, Timestral, MOMENT | phase 2, where they run in 8 GB |

**Caveat to report**: foundation models are zero-shot but may have seen some of these
datasets in pretraining (e.g. M4 in several pretraining corpora). Our model trains on the
development datasets' histories. Holdout results are the fair zero-shot comparison.

## Research tracks

Tracks 1–7 are defined in `MISSION.md` §5. Added 2026-10-29 at the user's request:

**Track 8 — World model embedded inside a Transformer.** Test whether a Transformer that
carries an explicit, persistent latent world state through its layers (state or memory
tokens updated by a learned transition at each patch, with forecasts decoded from the
rolled-out state) beats the same Transformer without that pathway, at equal parameters and
equal training compute.

## Ambition vs. gates

The goal is to beat every baseline here, including the current leaderboard leaders
(Timestral, TimesFM-3, Chronos-2, TiRex), by a large margin. The 10% gate below is the
minimum to call a result a success, not the target.

## Phases

1. **Exploration (now)**: the autoresearch loop iterates on `train.py` against the val split
   only, following `program.md`. Covers tracks T1–T7 at small scale.
2. **Confirmation**: pick at most 3 candidate models from the log. Retrain each with 3 seeds.
   Run `TSAR_SPLIT=test` on test and holdout for candidates and baselines, once.
3. **Extension**: phase-2 baselines, multiple rolling-origin windows, missing-data and
   distribution-shift stress tests, multivariate and irregular datasets, covariates.

## Pre-registered success gates (evaluated on test and holdout, phase 2 onward)

Let B be the best baseline on the val split by `wql_rel`, chosen before the test run.

| Gate | Criterion |
|---|---|
| Accuracy | `wql_rel` at least 10% lower than B on the test split, and lower than B on at least 10 of 15 development datasets |
| Generalization | `wql_rel` lower than B on the holdout split, and lower on at least 3 of 5 holdout datasets |
| Calibration | mean coverage error no worse than B's + 0.02 |
| Efficiency | at least 2× lower total inference seconds than B on the same hardware, or ≥ 2× fewer parameters, at accuracy within 2% of B |
| Robustness | phase 3: retains a larger share of its clean accuracy than B under 20% random missing context values |
| Reproducibility | each gate holds in the mean over 3 seeds, with the seed standard deviation reported |

Report, for every gate: fraction of datasets improved, median relative improvement, the worst
regression, and results broken down by frequency and domain. A result that fails a gate is
reported as a failure, not dropped.

## Leakage audit checklist

- [x] Normalization uses context values only (`train.py: normalize`).
- [x] Training data ends before the val window (`prepare.load_train_series`).
- [x] Holdout datasets excluded from training (`prepare.HOLDOUT`).
- [x] Test and holdout scoring needs an explicit `TSAR_SPLIT=test`.
- [ ] Training windows of non-scored series overlap the val period in calendar time for
      datasets whose series end on different dates. Audit in phase 2.
- [ ] Pretraining overlap of each foundation model with our datasets: document per model.
      Timestral's report does not list its real post-training datasets, so its overlap is
      unknown; Chronos models were trained without the Chronos Benchmark II datasets.

## Efficiency target

Timestral-5M (5.2M parameters, 17 ms per forecast on an 8-thread server CPU, report §4.3) is
the reference for "comparable accuracy at much lower cost". Our efficiency gate compares
against it on this laptop, not against the report's hardware.
