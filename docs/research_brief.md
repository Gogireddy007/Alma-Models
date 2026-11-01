# Alma — Research Brief for an Independent Research Agent

Written 2026-10-31 by the project lead. This brief stands on its own: read it fully before doing anything.
Repository: `/Users/srimanarayana/Timeseries` (branch `autoresearch/oct29`).

---

## 1. The goal in one paragraph

Build **Alma**, a time-series forecasting model that is a **genuine world model**. It should beat every major
forecasting model on four things at once:
- **accuracy;**
- **interval calibration;**
- **efficiency;**
- **zero-shot generalisation.**

There must be **no data leakage**. Every win must be large, repeatable and independently verifiable. Owner's own words:
*"Success means Alma, as a genuine world model for time series forecasting, beats every baseline on accuracy (scores,
lower is better), interval calibration, efficiency and zero-shot generalisation, with no leakage."* The owner also said
the model **must be a world model**. A model without a world-model design that happens to win does not meet the goal.

## 2. What "world model" means here

A forecaster qualifies only if all three of these hold:

1. **Latent state.** It compresses the observed history into a state z_t, a representation of "what the system looks like now".
2. **Learned transition.** A learned transition rolls that state forward into the future without new observations, so the model
   imagines the future step by step. Forecasts are read from the imagined states.
3. **The dynamics must earn their keep.** This is the falsifiable part. In the same harness, with params matched within 3%,
   the model must be *measurably better* than its own ablations:
   - `direct`: no state is carried between leads; each lead is read from the origin state.
   - `copy`: no transition; z_k = z_0 + the lead's input embedding.
   - `zero`: imagined states are zeroed before the decoder.

   If `direct` ties `none`, the "world model" is decoration. This happened to our first design (see §6).

Optional world-model ingredients count as hypotheses, not requirements. Each must show a measurable benefit or it is removed:
- stochastic latents;
- self-supervised latent prediction (JEPA);
- probabilistic trajectory sampling;
- cross-series transfer;
- continual adaptation;
- covariates.

## 3. Baselines to beat

All scores are on our VAL bench (§4), lower is better. `wql_rel` and `mase_rel` are geometric means relative to Seasonal
Naive; `cov80_error` = |empirical coverage of the 80% interval − 0.80|.

| Model | wql_rel | mase_rel | cov80_error | predict s (30 tasks, M3) |
|---|---|---|---|---|
| **Chronos-2** (amazon/chronos-2) | **0.5586** | 0.7195 | 0.068 | 542 |
| **Timestral-209M** (mistralai) | **0.5639** | **0.7062** | **0.034** | 1549 |
| Timestral-28M | 0.5718 | 0.7124 | 0.036 | 432 |
| Chronos-Bolt-base | 0.5817 | 0.7391 | 0.058 | 1862 |
| TiRex | 0.5818 | 0.7267 | 0.048 | 1294 |
| Timestral-5M | 0.5846 | 0.7299 | 0.047 | 59 |
| AutoTheta | 0.9594 | 0.8943 | 0.130 | 633 |
| Seasonal Naive (with PI) | 0.9876 | 1.0 | 0.142 | — |
| AutoETS | 2.68 (blow-ups) | 0.915 | 0.093 | 1668 |

- **Strongest reproducible baseline B\***: {Chronos-2, Timestral-209M}. They are within 1% of each other on wql, and
  Timestral wins on mase and calibration. Every target must hold against **both**.
- **Newer models the plan also requires** (not yet run locally): TimesFM-3, TiRex-2, Toto-2.0, t0-gamma.
- **Protocol notes:**
  - Our bench caps context at 2048 and runs Chronos-2 univariate (no cross-learning, no covariates). Both choices
    *handicap* the baselines relative to their published setups (`docs/eval_contract.md`).
  - Some baselines likely trained on our VAL families solar, kdd_cup_2022 and favorita. This is disclosed in
    `bench.VAL_EXPOSURE`.

**Where Alma stands:** about **0.67–0.70** wql_rel at **0.37M params**. The gap to B\* is about 17–20%. Calibration is
worse (cov80_error ≈ 0.11–0.13 vs 0.034).

## 4. How results are measured

- **Bench** (`tools/bench.py`):
  - 30 VAL tasks in 13 families, taken from fev-bench task definitions.
  - Metrics are pooled over windows, as a geometric mean vs Seasonal Naive.
  - Separately there are a TEST split (48 tasks, 18 families) and a TRAIN corpus.
  - Official fev aggregation is per window, and its numbers differ (e.g. 0.2687 official vs 0.1890 pooled for one model).
    Our pooled numbers are for internal selection only.
- **Official suites** (mandatory for any claim): GIFT-Eval, TIME and fev-bench, run with their *official* harnesses.
  - The harnesses are vendored at `vendor/eval/` and run from `.venv-eval`.
  - Re-scorer agreement is verified to 1e-6 (`tools/rescore.py`, `tools/predstore.py`).
  - fev-bench contains our TEST tasks, so it can only be the final confirmatory run.
  - TIME's EPF dataset is our TEST data, so it is excluded.
- **Zero-shot rule (GIFT):** no GIFT test dataset in training. Training therefore uses the **GIFT-clean corpus**
  (`TSAR_GIFT_CLEAN=1`, which drops 13 datasets).
- **Pre-registered win rules** (`docs/preregistration.md`, binding):
  - **Ours vs ours (Gate 3).** Need a gain of **≥ 0.020 wql_rel, mean over 3 fresh seeds**. Gains of 0.008–0.020 go to
    5 seeds and need ≥ 0.013. Ties are within ±2%. Seed SD ≈ 0.0065.
  - **Ours vs a baseline.** The target is a **10% improvement** in the point estimate over B\*. The family-clustered CI
    upper bound must be < 1, and Alma must win > 60% of tasks.
  - **Power.** The minimum detectable effect vs a baseline is 11–14% at 80% power.
  - **Winner's curse.** The best of K single-seed screens is inflated by about 0.017 (K=20) to 0.021 (K=60). Never call
    a single-seed result "best".
- **Five dimensions to win:** accuracy (wql, mase), calibration (cov80 error, sharpness), efficiency (predict
  s/1000 forecasts, peak MB, params), zero-shot generalisation (GIFT/TIME/fev official), and robustness. Efficiency is
  where a 0.4M–10M model can win outright.

## 5. Hard rules (non-negotiable)

1. **No leakage.**
   - Never score TEST during research. TEST is touched once, at the end, by a frozen checkpoint (Gate 5).
   - Never train on VAL/TEST series or their sources.
   - Every new training corpus is overlap-scanned (exact hash + z-norm + lag correlation) against VAL and TEST, and the
     loader refuses it until a clean stamp exists (`tools/overlap_scan.py`, `bench.require_*_stamp`).
2. **No foundation-model weights or outputs in Alma.** Distillation is diagnostic only.
3. **Do not edit** `prepare.py` or `baselines.py`.
4. **Ask the owner before downloading** any external dataset or model.
5. **Compute:** a MacBook M3 with 8 GB of unified memory.
   - One heavy job at a time, through `tools/jobq.py`.
   - Smoke tests go through `tools/smoke.py` (< 60 s, < 2 GB) and must exercise the eval path.
6. **Autoresearch discipline** (`program.md`, adapted from karpathy/autoresearch):
   - one hypothesis, one change, one commit;
   - log keep/discard/crash;
   - respect the noise floor;
   - do ablation duty;
   - read the log before choosing.
7. **Honest labels:** every result is *verified* (≥ 3 seeds, re-derived), *preliminary* (1 seed), *failed* or
   *unresolved*. These go in `runs/ledger.tsv` and `research_log.md`.

## 6. What we know so far (evidence, as of 2026-10-31)

**Architecture of the incumbent ("gru" family)**
- Patch-8 tokens feed a causal Transformer, which produces the latent z_t.
- A GRUCell rolls z forward K=8 patches (64 steps). Horizons beyond 64 feed the median back.
- A 9-quantile decoder reads the imagined states.
- A learned seasonal-naive skip is added.
- Inputs use causal standardisation + asinh.
- Training mixes synthetic curricula (syn3/syn5) with real data.
- Size is 0.37M params, about 7 min per 2500-step run on the M3.

**Verified (3 seeds)**
- Four recipe variants tie at about **0.695**: curriculum, synthetic mixes and calibration tweaks.
- **The world-model test failed for the GRU design.**
  - GRU rollout scores 0.6958 ± 0.0070 and a matched no-dynamics `direct` head scores 0.6978 ± 0.0039. Paired, the
    difference is +0.002, a tie.
  - Removing or freezing the transition does hurt (+0.023 / +0.031). So the model needs per-lead capacity, but recurrence
    as such adds nothing.
- **More of the same data does not help.** The 21 extra GIFT-clean corpora gave −0.004, below the bar.
- Making the corpus GIFT-clean cost about 0.01.
- Longer context (1024/2048) helped about 0.02 on one seed (preliminary). Larger models have not been tried at all.

**Wave 5: a screen of radically different world-model families**
- All families run in one harness (`train_wm.py` + `wm/<family>.py`, interface in `docs/wm_families.md`).
- Conditions are identical for all: same data, recipe and eval, ~0.37M params, 2500 steps.
- Scores are seed 0 only, so all are **preliminary**:

| Family | Idea | wql_rel | mase_rel | cov80_err |
|---|---|---|---|---|
| **ssm** | Mamba/S5-style selective SSM; the *same* system encodes the past and imagines the future | **0.6707** | 0.832 | 0.125 |
| **koopman** | linear latent dynamics, learned stable complex spectrum (damped oscillators locked to cycles) | **0.6849** | 0.841 | 0.113 |
| gru (reference) | Transformer + GRUCell rollout | 0.6935 | 0.844 | 0.129 |
| gru, direct ablation | no carried state | 0.6985 | 0.853 | 0.121 |
| flow (head) | flow-matching patch sampler; quantile head on one imagined path | 0.7049 | 0.860 | 0.124 |
| jepa | predictor trained to hit an EMA target encoder's future embeddings | 0.7086 | 0.864 | 0.116 |
| rssm | Dreamer-style deterministic + categorical stochastic state, KL | 0.7496 | 0.897 | 0.142 |
| flow (samp) | quantiles from 16 sampled imagined paths | 0.8600 | 0.977 | 0.248 (samples too narrow) |
| ctime | closed-form continuous-time oscillator ODE-RNN | 0.8818 | 0.996 | 0.360 (failed) |
| vqtok | discrete VQ tokens + autoregressive code transformer | running | | |

**What the screen suggests (preliminary)**
- The two families with *structured linear/selective state dynamics* (SSM, Koopman) are the first to beat the incumbent.
- SSM is ahead by 0.023 on one seed.
- The winner's-curse inflation for about 9 screens is about 0.015, so this is not yet a result.
- Seeds 1–2 for SSM, Koopman and the gru reference are queued, followed by the direct/copy/zero ablations that decide
  whether their dynamics earn their keep.

**Data in hand**
- The GIFT-clean corpus has 51.7M points.
- A 1.58 GB GiftEvalPretrain subset has been downloaded: 25 folders of mostly 5–30 min traffic, energy, cloud and air data,
  +33.9M points (+66%).
  - It is **locked until its overlap scan (4–9 h) is clean**.
  - It targets the measured gap: 61% of our gap is in hourly/sub-hourly tasks and 43% in multi-seasonal ones
    (`docs/gap_analysis.md`).

## 7. The research questions that matter most

Ranked by expected value:

1. **Scale.** The gap to 120M–209M-param baselines is about 17–20%. Small-model tweaks move scores by under 1%.
   - Test a model-size ladder (0.37M → 1M → 4M → 10M, `TSAR_WM_WIDTH`) for the best 1–2 families.
   - Test a data ladder (GIFT-clean → + GEP once stamped).
   - Find where the scaling curve bends on 8 GB.
2. **Which world-model family wins, and does its dynamics pass the ablation test?** Confirm SSM and Koopman over 3 seeds.
   Also check whether the SSM/Koopman advantage grows with lead time; that is the signature a world model should leave.
3. **Calibration.** cov80_error is about 0.12 vs 0.034 for Timestral. Candidates:
   - conformal / quantile recalibration on TRAIN-held-out data;
   - trajectory sampling that is actually calibrated;
   - variance-aware heads.
4. **Long horizons.** Horizons beyond 64 steps currently feed the median back. Options:
   - a longer native rollout;
   - multi-rate states;
   - feeding back sampled paths.
5. **Context length.** 1024–2048 looked +0.02 on one seed; confirm it, and pair it with SSM, which handles long
   context cheaply.
6. **Multivariate / cross-series and covariates.** These are world-model-native advantages that our univariate bench
   currently ignores.
7. **Efficiency.** Report params, MACs, predict seconds and peak MB for every candidate. A 10M model that matches
   Chronos-2 at 1/20th the cost is already a strong result on one of the four goals.

## 8. Decision gates

| Gate | Requirement |
|---|---|
| 1. Evidence integrity | Reconcile the ledger before major training. |
| 2. Benchmark reproduction | Reproduce the official harnesses before any superiority claim. |
| 3. Early signal | ≥ 0.020 on 3 fresh seeds before spending more compute. |
| 4. Generalisation | Hold on held-out families, a Tier-D locked split, and GIFT/TIME (never used for selection). |
| 5. Frozen promotion | Freeze one checkpoint, score TEST and the official suites once, publish everything. |

## 9. What a useful contribution from you looks like

- **Concrete, falsifiable proposals,** each with:
  - the hypothesis;
  - the exact change (family file / flag);
  - the expected effect size and why;
  - the cost on an 8 GB M3;
  - the control and ablation that would refute it.
- **Literature grounding.** Bring the relevant state of the art:
  - SSM/Koopman/world-model forecasting;
  - scaling laws for TS foundation models;
  - the training recipes of Chronos-2, TimesFM, TiRex and Toto;
  - calibration methods.

  Say what transfers to a ≤ 10M-param, 8 GB-trainable model.
- **Respect §5:** no leakage, no FM weights, no TEST peeking, one change at a time. Label every claim
  verified / preliminary / speculative.
- **Do not** propose "train a 200M model" as the answer. Either show how to win at small scale, or show how to scale
  within the 8 GB budget (e.g. gradient checkpointing, bf16, curriculum, data selection).

## 10. Key files

| File | What it is |
|---|---|
| `MISSION.md` | full mission statement |
| `docs/preregistration.md` | binding win rules and power analysis |
| `docs/eval_contract.md` | protocol differences vs the published baselines |
| `docs/benchmark_contract.md` | benchmark contract |
| `docs/gap_analysis.md` | where the gap to Chronos-2 is (by frequency and pattern) |
| `docs/wm_hypothesis_tests.md` | the world-model falsification tests |
| `docs/wm_families.md` | harness interface and fairness rules |
| `docs/gift_clean_corpus.md` | training-data regime and GEP plan |
| `docs/lessons.md` | process lessons L1–L9 |
| `docs/research_history.md` | research history |
| `runs/ledger.tsv` | canonical ledger |
| `research_log.md` | chronological log |
| `baselines/bench/results.tsv` | baseline scores |
| `train_wm.py`, `wm/*.py` | current harness and the eight families |
| `tools/bench.py`, `tools/jobq.py`, `tools/smoke.py`, `tools/overlap_scan.py` | evaluation, queue, smoke and leakage tooling |
