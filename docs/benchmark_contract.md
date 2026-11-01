# Benchmark contract: GIFT-Eval, TIME, fev-bench (Phase 0, research only)

Written 2026-10-30 by Arjun Iyer (literature and benchmark lead). Nothing was downloaded or installed, and the TEST split was not touched.
Sources: the Timestral report [arXiv:2610.37585](https://arxiv.org/abs/2610.37585) (HTML v1, Table 3, which reflects the 2026-10-15 release) and
[mistralai/timestral](https://github.com/mistralai/timestral) @ `a534bdb`; TIME, [PMLR v306 qiao26g](https://proceedings.mlr.press/v306/qiao26g.html) (PDF read in full);
[gift-eval](https://github.com/SalesforceAIResearch/gift-eval) @ `82f89c3` and [arXiv:2410.10393](https://arxiv.org/abs/2410.10393); [fev](https://github.com/autogluon/fev) @ `193e4fc`,
the `benchmarks/fev_bench/tasks.yaml` and `tasks_mini.yaml` text files, and fev-bench [arXiv:2509.26468v2](https://arxiv.org/abs/2509.26468). Sizes and licences come from the HF API (2026-10-30).
Labels: **E** means an estimate made by extrapolation, not a measurement. "not found" means I could not verify it.

## 0. Headline findings

1. **Official fev-bench is mostly our TEST split.** Of the 100 official tasks, 48 are our TEST tasks, 30 are our VAL tasks and 22 are missing (section 2).
   Of the 20 fev-bench-mini tasks, 13 are TEST. Any official fev-bench run, mini included, therefore consumes TEST. It can only be the single confirmatory run.
2. **Our TRAIN contaminates GIFT-Eval.** TRAIN_LOCAL holds three GIFT test datasets as whole series: `m4_quarterly`, `monash_hospital` and `monash_car_parts`.
   Under GIFT rules our current checkpoints must be flagged `testdata_leakage: Yes`. A zero-shot GIFT claim needs a retrain on a GIFT-clean TRAIN.
3. **TIME contains a TEST source.** TIME's "EPF Electricity Price" is the Lago et al. 2021 data, the same source as our TEST `epf_*` tasks.
   Running TIME during iteration would expose TEST content. Until the confirmatory run, either exclude that one dataset or defer TIME.
4. **The paper has no ablation numbers for its mechanisms.** Timestral describes DHF, asinh normalisation and its synthetic-first recipe, but reports no ablations for them.
   Every mechanism in section 5 is a hypothesis to test, not a proven gain.

## 1. Per-suite contract

### GIFT-Eval
- **Harness:** `SalesforceAIResearch/gift-eval` has no tags; pin `82f89c3` (2026-10-30). Scoring uses GluonTS `evaluate_model(axis=None, mask_invalid_label=True, allow_nan_forecast=False)`.
  There are 97 configs (dataset/freq/term, short/medium/long) built from 23 datasets and 28 HF folders, with 144k series and 177M points.
- **Data:** `Salesforce/GiftEval` @ `3084173`, 1.59 GB. The single largest folder is `electricity` at 677 MB. Licence: Apache-2.0, "for research purposes only".
- **Metrics:** MASE and CRPS per config, each normalised by Seasonal Naive and aggregated by geometric mean (lower is better). The leaderboard also ranks by CRPS_Rank and MASE_Rank.
- **Context and horizon:** the term fixes the horizon and the harness fixes the windows. Context is the model's choice. Timestral uses the full history up to 32,768 points; we cap at 2048. Disclose our cap.
- **Leakage:** a model is `zero-shot` only if it never saw a test dataset and never trained on any GIFT train split. `pretrained` may include the train splits.
  The non-leaking pretraining corpus is `Salesforce/GiftEvalPretrain` (230B points).
- **Submission:** open a PR adding `results/<model>/all_results.csv` (98 lines, 15 columns) plus `config.json` (model_type, testdata_leakage, replication_code_available, ...).
  Validate locally with `scripts/validate_results.py`.
- **Wall time on the 8 GB M3 (E):** I scaled from our measured VAL runs (30 tasks, at most 1000 forecasts per task, CPU), assuming GIFT is 5 to 10 times the VAL work.
  - Alma (0.4M, VAL about 50 s on CPU): roughly 5 to 15 min on CPU. MPS is unnecessary.
  - 200M model (Timestral-209M took 1549 s on VAL on CPU): roughly 2 to 5 h on CPU at context 2048. Full 32k-point histories are not feasible within a day.
  - MPS is untested for Timestral; its README lists CPU and CUDA only. Weights are 0.84 GB in fp32, so memory is fine.

### TIME (ICML 2026)
- **Harness:** `zqiao11/TIME` has no tags; pin `c633e23` (2026-10-16). It covers 98 tasks over 50 "fresh" datasets.
  Short, medium and long horizons were set per dataset by an LLM-plus-human process; low-frequency datasets get a single horizon.
- **Data:** `Real-TSF/TIME` @ `83e3d0b`, 220 MB. The three `azure2019_*` folders take 155 MB of it. The repo licence is CC BY-NC 4.0, and the per-dataset licences are mixed:
  CC BY 4.0, SG ODL 1.0, public domain, CC0, Apache-2.0, and "Terms of Service" for PVOutput, Realtor.com, Port/IMF and others. **Non-commercial: research use only.**
- **Metrics:** MASE and CRPS per task, averaged over windows, variates and series. Each is normalised by the Seasonal Naive task score, then aggregated by geometric mean over all 98 tasks.
  Distribution-type models are scored from 100 samples.
- **Windows:** the last L_test points of each series form the test period, which is split into non-overlapping windows with stride H, giving W = floor(L_test / H).
  Every variate is a target and no covariates are used. Context is the model's choice: the paper ran 2048 to 8192, and Timestral used the full history.
- **Leakage:** freshness is by construction; the data comes from 2021 to 2025 sources. There is no self-flag mechanism.
  Conflict of interest: two of the authors work at Datadog, the maker of Toto.
- **Scoring:** write `experiments/<model>.py` that produces quantiles of shape (instances, quantiles, H), then call `save_window_predictions`.
  `scripts/compute_local_leaderboard.py` scores locally. To submit, open a PR on `Real-TSF/TIME-Output` with `results/<model>`.
- **Wall time (E):** about 3 to 6 times VAL. Alma under 10 min on CPU. A 200M model 1.5 to 3 h on CPU, with the Azure 5-minute series dominating.

### fev-bench
- **Harness:** `pip install fev`. The latest release is 0.10.0 (2026-08-31). Our task definitions are pinned to `f323f6c`; main is now `193e4fc`.
  The suite has 100 tasks over 96 datasets, 46 of them with covariates. `fev-bench-mini` has 20 tasks.
- **Data:** `autogluon/fev_datasets` @ `d3e7e72`, 658 MB in total. We already hold 199 MB locally. The HF licence is "other": a licence per dataset. The fev code is Apache-2.0.
- **Metrics:** SQL (quantiles 0.1 to 0.9) and MASE. WQL and WAPE are secondary.
  - Skill score: S = 1 - geomean(clip(E / E_SeasonalNaive, 0.01, 100)).
  - Win rate: ties count as half.
  - 1000 paired task bootstraps give the confidence intervals.
- **Windows:** up to 20 windows when a dataset has fewer than 10 series, up to 10 for 10 to 2000 series, and 1 window above that. At least 2H + 1 history points are required.
  The horizon comes from the competition or from the frequency. Context is the model's choice; the paper's caps run from 2048 to 16000.
- **Leakage:** each model self-flags every task whose dataset it trained on at the same frequency. Flagged tasks are imputed with Chronos-Bolt (Base) scores.
  Published flag shares: Moirai-2 28%, TimesFM-2.5 10%, Toto-1 8%, TiRex 1%.
- **Submission:** add a wrapper under `models/` and open a PR with the results. The leaderboard is the HF space `autogluon/fev-bench`.
- **Wall time (E):** official tasks are uncapped (all series, for example 30k M5 series in one window), so about 4 to 8 times VAL.
  Alma under 15 min on CPU. A 200M model 2 to 4 h on CPU.

## 2. Overlap with our repo (tools/bench.py)

**fev-bench tasks missing from VAL and TEST (22).** All but three are already excluded on purpose (EXCLUDED_FAMILIES):
- `ercot_1H/1D/1W/1M`, `m5_1D/1W/1M` and `hospital`: these are in our TRAIN.
- `fred_md_2025` x2 and `fred_qd_2025` x2.
- `favorita_transactions_1D/1W/1M` and `proenfo_gfc12/14/17`.
- `australian_tourism` and `LOOP_SEATTLE_5T`.
- The three not covered by any EXCLUDED_FAMILIES note: `kdd_cup_2022_10T` and `kdd_cup_2022_30T` (TiRex, TimesFM-2.5, Toto-1 and Moirai-2 flag the 10T variant; we hold 1D in VAL), and `proenfo_gfc14`, which is in mini.

**Contamination from TRAIN into each suite** (exact means the same dataset; related means the same source but different series or period):

| TRAIN dataset | GIFT-Eval test | fev-bench | TIME |
|---|---|---|---|
| m4_quarterly (LOCAL) | **exact** m4_quarterly | - | - |
| monash_hospital (LOCAL) | **exact** hospital | **exact** `hospital` (excluded) | - |
| monash_car_parts (LOCAL) | **exact** car_parts | - | - |
| m5, ercot (LOCAL) | - | **exact**: 3 m5 tasks and 4 ercot tasks (excluded) | - |
| monash_tourism_* (LOCAL) | - | possible `australian_tourism` (excluded) | - |
| monash_australian_electricity (LOCAL) | - | - | related: OpenElectricity NEM (AEMO demand, later period) |
| m4_daily/hourly/monthly/weekly/yearly (EXTRA) | **exact**, 5 configs | - | - |
| monash_kdd_cup_2018, saugeenday, temperature_rain (EXTRA) | **exact** | - | - |
| monash_electricity_hourly/weekly (EXTRA) | **related**: electricity (same UCI source) | - | - |

Verdicts:
- **GIFT-Eval.** The current TRAIN gives `testdata_leakage: Yes`. A clean run must drop 3 LOCAL and all 10 GIFT-overlapping EXTRA sets; all other TRAIN sets appear to be GIFT-free.
  Separately, 9 of our VAL families are GIFT test families: ETT, LOOP_SEATTLE, M_DENSE, SZ_TAXI, bizitobs_l2c, hierarchical_sales, jena_weather, solar and restaurant.
  Model selection on them is not training leakage under GIFT rules, but it is selection exposure and must be disclosed next to any GIFT number.
- **fev-bench.** If we submit, we self-flag the 8 exact tasks (m5 x3, ercot x4, hospital), and `australian_tourism` if the tourism source is confirmed. Those tasks are then imputed with Chronos-Bolt.
- **TIME.** None of its datasets is in TRAIN, so TIME is the cleanest of the three suites for us. The one caveat is related: same-market NEM demand.
  The problem runs in the other direction: TIME contains TEST-equivalent data. EPF = our TEST `epf_*`. ECDC COVID and WHO FluNet are related to our TEST `ecdc_ili`.

## 3. Comparison models (HF API, 2026-10-30)

Scores come from Timestral Table 3. GIFT and TIME: MASE / CRPS geometric mean relative to Seasonal Naive, lower is better. fev: MASE / SQL skill in %, higher is better.
Memory column: fp32 weights against 8 GB of RAM.

| Model (checkpoint @ rev) | Params | Licence | Fits in 8 GB? | GIFT MASE/CRPS | TIME MASE/CRPS | fev MASE/SQL skill |
|---|---|---|---|---|---|---|
| Timestral-209M `mistralai/Timestral-209M` @9874d25 | 209M, 0.84 GB | Modified MIT ("other") | yes on CPU (ran VAL, 1549 s) | 0.6575/0.4460 | 0.6262/0.5223 | 39.49/50.39 |
| Timestral-123M @5a14728 | 123M | same | yes | 0.6589/0.4476 | 0.6277/0.5253 | 39.19/50.18 |
| Timestral-28M @49b2640 | 28M | same | yes (VAL 432 s) | 0.6780/0.4610 | 0.6382/0.5344 | 37.27/48.42 |
| Timestral-5M @bdd20cf | 5.2M | same | yes (VAL 59 s) | 0.7049/0.4789 | 0.6518/0.5445 | 34.54/46.19 |
| TimesFM-3 `google/timesfm-3.0-pytorch` @43046b8 | 331M, 1.3 GB | **TimesFM Non-Commercial v1.0** | likely on CPU; MPS untested | 0.6668/0.4557 | 0.6398/0.5363 | 37.82/48.99 |
| Chronos-2 `amazon/chronos-2` @29ec376 | 119M | Apache-2.0 | yes (VAL 542 s) | 0.6978/0.4854 | 0.6620/0.5563 | 35.50/47.28 |
| TiRex-2 `NX-AI/TiRex-2` @30788a3 | 82.5M (38.4M univariate) | Apache-2.0 | yes | 0.6777/0.4669 (`-gifteval-pretrain`) | not reported | 33.74/45.50 (`-fevbench` ckpt) |
| Toto-2.0 `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}` | 4M to 2.45B | Apache-2.0 | 4m to 313m yes; 1B (4.2 GB) marginal; 2.5B (9.8 GB) **no** | 2.5B 0.6956/0.4759; 4m 0.7565/0.5242 | 2.5B 0.6419/0.5394 | 2.5B 32.54/44.42 |
| t0-gamma `theforecastingcompany/t0-gamma` @e08aee0 | 256M, 1.0 GB | Apache-2.0 (code: `tfc-t0`, PyTorch and MLX) | likely; MLX build suits the M3 | 0.6608/0.4520 | 0.6370/0.5340 | 37.99/49.42 |
| PatchTST, DLinear (light neural) | - | - | trivially | **not found** in a verified source (GIFT leaderboard has full-shot entries; I did not extract them) | not found | not found |
| Seasonal Naive | - | - | - | 1.0/1.0 by definition | 1.0/1.0 | 0/0 |

Notes:
- Every named model exists; none is unverifiable. "TiRex-2" has suite-specific checkpoints (`-gifteval-zs`, `-gifteval-pretrain`, `-fevbench`), so quote the checkpoint with every score.
- TimesFM-3 trains on GiftEvalPretrain minus the fev-overlapping sets. Its non-commercial licence matters if Alma is ever productised.
- Mistral's README says Timestral-209M is "#1" on all three suites. That claim is self-reported, and the leaderboards were not re-read here.

## 4. Minimal download list for approval (smallest first; nothing fetched yet)

| # | Item | Size | Licence | Purpose |
|---|---|---|---|---|
| 1 | `fev==0.10.0` (pip, code only) | under 1 MB | Apache-2.0 | Score one VAL task (ETT_1D, data already local) with official fev and check it against `tools/bench.py`. 0 MB of data. |
| 2 | gift-eval repo @ `82f89c3` plus `gluonts` (pip) | about 5 MB of code | Apache-2.0 | Harness |
| 3 | GIFT smoke data: `us_births` 0.04 MB, `covid_deaths` 0.23 MB, `hierarchical_sales` 1.0 MB | 1.3 MB | Apache-2.0 | Three-config smoke. Avoid `saugeenday`, `hospital` and `car_parts`, which are TRAIN-contaminated. |
| 4 | TIME repo @ `c633e23` plus `Real-TSF/TIME` subset `Job_Claims`, `Housing_Inventory`, `JOLTS` | under 0.1 MB of data | CC BY-NC 4.0 / public domain | Three-task smoke, with `epf_electricity_price` excluded |
| 5 | Result files for the comparison table: `gift-eval/results/*/all_results.csv`, fev `results/*.csv` | under 20 MB | Apache-2.0 | Published scores to compare against, with no model weights |
| 6 | Full `Real-TSF/TIME` minus EPF | 219 MB | mixed, non-commercial | Full TIME run for Alma |
| 7 | Full `Salesforce/GiftEval` | 1.59 GB (918 MB without `electricity`) | Apache-2.0 | Full GIFT run, only after a GIFT-clean retrain |

Out of scope for Phase 0:
- **Comparison weights:** published scores are enough. If we must re-run one, fetch the smallest first: Toto-2.0-4m (17 MB), Toto-2.0-22m (88 MB), chronos-2-small (112 MB).
- **Official fev-bench:** the confirmatory run only, because it overlaps TEST.

## 5. Mechanism ideas from the leaders (8)

| # | Idea | Evidence | Our repo status | Cost | Smallest falsifying experiment |
|---|---|---|---|---|---|
| 1 | **Decoupled horizon forecasting (DHF):** learned prediction tokens at each context position, random block lengths, 1-to-k-step direct targets, context computed once | Timestral §DHF. No ablation numbers (it also uses about 2x the tokens). | GRU latent rollout, MAX_H 64, PATCH 8; TSAR_DUALDEC is the nearest analogue | Medium: new head, about 2x tokens per step | 2500-step VAL A/B of direct prediction tokens against the GRU rollout, scored on h ≥ 96 tasks (ETT_15T, bizitobs_5T, jena_10T, SZ_TAXI_15T). Kill if the paired CI includes 0. |
| 2 | **Causal per-patch standardisation plus asinh;** each forecast patch standardised with statistics up to the previous patch | Timestral §5 | `causal_stats` NORM last/mean. No asinh anywhere in the repo. | Low: one flag | TSAR_NORM=mean_asinh against the default on VAL, plus a heavy-tail subset (uk_covid, favorita). Kill if WQL is not better by more than 1% (paired CI). |
| 3 | **Synthetic-only stage 1, then real-plus-synthetic stage 2 at longer context** | Timestral: 8k synthetic, then 32k mixed; Chronos-2 and TiRex both lean on synthetic data | TSAR_SYN_CURR does the opposite (synthetic share decays); syn5 generators exist | Low | Same 2500 steps: 1250 syn5-only, then 1250 mixed (ctx 256 to 512), against the incumbent mix. Kill if VAL WQL is no better. |
| 4 | **Non-crossing quantiles by cumulative softplus, 21 levels with 0.01/0.99** | Timestral head | 9 levels, sorted after the fact | Low | Swap the head; compare cov80 and WQL on VAL. Kill if cov80 error does not drop. |
| 5 | **Longer output patch than input patch** | TimesFM-3 card (32 in, 64 out) | in = out = 8 | Low | Output patch 16 or 32 with in = 8; long-h VAL subset |
| 6 | **Missing-span augmentation with hidden values still scored** | Timestral §DHF | obs mask already in tokens; no augmentation | Low | Random masking of 0 to 20% of spans; score on VAL tasks with NaNs |
| 7 | **Variate attention every fourth block** | Timestral, Chronos-2, TiRex-2, TimesFM-3 | Hoffmann's `train_cov.py` track | Medium | Multivariate VAL tasks only (jena, bizitobs, ETT, uk_covid); kill if no gain over channel-independent |
| 8 | **Long context (2048 to 8k) via a coarse prefix** | Timestral uses 32k histories on all suites | TSAR_PREFIX exists; MAX_CONTEXT 2048 in bench | Medium (memory) | PREFIX on/off at a fixed budget; VAL hourly and sub-hourly tasks |

Recommended order by expected gain per unit of compute: 2, then 3, then 1.
