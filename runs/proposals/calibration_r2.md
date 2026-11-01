# Calibration and inference engineer: round-2 proposals (2026-10-31)

Target: cov80_err of the ssm main line from ~0.09-0.14 (paths or TTA eval-only logs, s0-s2) toward <= 0.05, with
wql_rel no worse. Baselines: 0.034 (Timestral-209M) to 0.068.

## What the smoke tests showed (this changes the plan)

1. **Where the VAL error is** (ssm s0, paths + TTA, runs/wm_ssm_none_s0_tta.log): 27 of 30 VAL tasks cover < 0.80.
   The intervals are too narrow almost everywhere, not only at long leads. The worst are bizitobs_l2c_5T 0.22 (h 288),
   uk_covid_utla_1W/cumulative 0.26, jena_10T 0.53, solar_1W 0.54 and ETT_1H 0.56 (h 168).
2. **On TRAIN the same model is calibrated or over-covered.** On the tails of its own training series (in-sample),
   ssm s0 covers 0.79-0.85 at leads <= 64 and 0.86-0.93 at sub-daily leads 65-288 (smoke, 60 series per dataset, 14
   datasets). A map fitted there therefore narrows the long-lead upper quantiles (c = 0.5, the grid floor), and on the
   4-task smoke subset it makes VAL worse: cov80_err 0.275 -> 0.305 and wql 0.736 -> 0.745.
   **So the VAL deficit is a TRAIN -> VAL family shift, and only out-of-family data can measure it.** The same
   conclusion rules out the alternatives:
   - in-sample recalibration;
   - holding out the last window of each training series (it is still the same family);
   - a training-side coverage or interval loss (pinball is already proper, and in-sample there is nothing to fix);
   - per-series self-calibration from the context (`TSAR_SELFCAL`), which was already tried and hurt
     (research_history.md).
3. **TSAR_WM_TTA as a 50/50 mixture instead of quantile averaging** (tried, then removed): on the smoke subset
   cov80_err went 0.275 -> 0.284 and wql 0.7359 -> 0.7367. Not proposed, and the code was taken out.

The proposals below therefore fit the map **out of family**. That is still a TRAIN-only fit: the DEV split
(tools/devsplit.py) holds 5 TRAIN source families out of training, so a model trained with `TSAR_DEV=1` meets them
zero-shot, as the main model meets VAL.

## Code (default-off, bit-identical at defaults)

- `tools/wmrecal.py` (new): the map, the pinball grid fit, and the json save/load.
- `train_wm.py`: the keys `TSAR_WM_RECAL=0|lead|freqlead|file`, `TSAR_WM_RECAL_N` and `TSAR_WM_RECAL_MAP`. They are
  inference keys, so they are allowed with `TSAR_EVAL_ONLY`. The docstring section is "Recalibration".
  - `lead|freqlead` with `TSAR_DEV=1`: fit on the held-out DEV families, then write the map json. The map is not
    applied to the DEV score.
  - `lead|freqlead` without DEV: fit in-sample on the training pool. This is the negative control from finding 2.
  - `file`: load a map json and apply it to VAL.

**The map.** For the final forecast q[n, 9, h] (after paths and TTA), with median m:
`q'_k[t] = m[t] + c[g, b(t), k] * (q_k[t] - m[t])`. The result is sorted and the TTA non-negative clip is re-applied.
- k is the quantile level: 8 scales. The median never moves, so mase_rel and the point forecast are unchanged.
- b is the lead bucket: 1-8, 9-16, 17-32, 33-64, 65-128, 129-256, 257+.
- g is the frequency group (sub-daily / daily / weekly and coarser) under `freqlead`, and pooled under `lead`.
  `freqlead` also fits the pooled cells as its fallback.
- Each c minimises the weighted pinball loss at level k over 57 log-spaced scales in [0.5, 4]. Every dataset has
  equal weight per cell. Each window is scaled by the mean absolute first difference of its context. Cells with
  fewer than 30 windows fall back to the pooled cell. Leads past the fitted range use the longest fitted pooled bucket.

## Leakage argument: exactly which data each fitted parameter sees

The scales c (8 levels x 7 buckets, x 4 cells with groups) are the only fitted parameters. In the proposed runs they
see only the following.
- **Series.** The TRAIN series of the DEV families: `devsplit.DEV_DATASETS` under `TSAR_GIFT_CLEAN=1`. These are
  monash_traffic (H), monash_australian_electricity (30min), nn5 (D), monash_nn5_weekly (W), monash_tourism_monthly
  (M), monash_tourism_quarterly (Q) and dominick (W).
  - They are taken from `bench.load_train` after TRAIN_DROP, so no series that overlaps VAL or TEST is included.
    The subsample is the same as for training.
  - train_wm captures them just before `devsplit.holdout` removes them from the training pool. devsplit asserts that
    none of them is a VAL, TEST or excluded family, nor a TRAIN_DROP set.
- **Windows.** Up to `TSAR_WM_RECAL_N` = 150 series per dataset are drawn with a fixed rng (seed 20261031),
  independent of TSAR_SEED and of VAL. Each is split into a context y[:-H] (its last 2048 points) and a target
  y[-H:]. H is 288 for sub-daily, 56 for daily and 13 for coarser data.
- **Forecasts.** They come from the DEV-trained checkpoint, through the same `predict` (paths + TTA) that the bench
  calls. That model never trained on these families.
- **What VAL and TEST contribute: nothing.** The map is fitted and written inside the DEV run, which never loads VAL
  (devsplit.evaluate scores DEV tasks only), and it is frozen as a json before any VAL eval. The `file` re-scores read
  only the json, and the bench then scores VAL once with it. TEST is never touched, since train_wm refuses
  TSAR_SPLIT=test.
- **Overlap with the DEV score.** The DEV tasks score the last windows of the same DEV series, so the map's
  calibration targets overlap the DEV scoring windows. That is why the map is not applied to the DEV score. The DEV
  number of the run stays the raw, unrecalibrated one.
- **Transfer caveat (not leakage).** The map is fitted on the DEV-trained model and applied to the main checkpoints,
  which did train on the DEV families. It transfers the recipe's out-of-family under-dispersion and assumes the main
  model under-disperses on new families in the same way. Items 1 and 2 test exactly that on VAL.
- **Selection on VAL.** Choosing `freqlead` or `lead` is one binary choice made on VAL, the same exposure as any R1
  screen. No fitted number ever sees VAL.

## 1. r2_devmap: out-of-family per-lead, per-frequency map (1 DEV training run + 5 eval-only re-scores)

- **Hypothesis.** The VAL under-coverage is the recipe's systematic under-dispersion on unseen families, and it grows
  with lead and is worst at sub-daily frequencies. A model trained without the DEV families shows the same
  under-dispersion on them, and a map fitted there transfers to VAL.
- **Expected effect.**
  - cov80_err goes from 0.11-0.13 to 0.05-0.08 on the 5-seed mean.
  - wql_rel changes by -0.003 to +0.003. Wider q0.1/q0.9 on under-covered tasks lowers their pinball loss, while the
    over-covered hierarchical_sales and solar_1D lose a little. The median does not move.
  - mase_rel is unchanged by construction.
- **Risk.** The smoke map (a 5-step model) shows that sparse cells can hit the grid caps of 0.5 and 4. With the real
  model and 150 series per dataset there are about 300 sub-daily windows. If the json still has caps in the
  sub-daily long-lead cells, item 2 (pooled) is the safer choice.
- **Cost.**
  - One R1 training run of about 35 min. Its DEV score (raw) is a free side result for the WM screen, as the DEV
    proxy of the ssm recipe.
  - Five eval-only re-scores of 1-2 min each.
- **Gate** (almats_research_program E2). cov80_err must improve on at least 10 of 13 VAL families, and the 5-seed
  mean wql must be no worse than ref_ssm_s* by more than 0.005. This is a calibration claim, not an accuracy claim.
- **World-model check.** Re-scoring `wm_ssm_direct_s0` with the same json must not change the none-vs-direct wql
  ranking.

```
r2_devmap_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_DEV=1 TSAR_WM_RECAL=freqlead TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json | mem=3000
r2_devmap_fl_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s0.pt | after=r2_devmap_s0 | run=wm_ssm_none_s0 | mem=2500
r2_devmap_fl_s1 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=1 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s1.pt | after=r2_devmap_s0 | run=wm_ssm_none_s1 | mem=2500
r2_devmap_fl_s2 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=2 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s2.pt | after=r2_devmap_s0 | run=wm_ssm_none_s2 | mem=2500
r2_devmap_fl_s3 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=3 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s3.pt | after=r2_devmap_s0 | run=wm_ssm_none_s3 | mem=2500
r2_devmap_fl_s4 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=4 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_freqlead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s4.pt | after=r2_devmap_s0 | run=wm_ssm_none_s4 | mem=2500
```
Compare these against ref_ssm_s0..s4 from ar_r1.txt (same inference, no map). The `run=` keys set only TSAR_RUN.
An eval-only run writes no checkpoint.

## 2. r2_devmap_lead: the same, pooled over frequencies (eval-only on the item-1 checkpoint, then 5 re-scores)

- **Hypothesis.** Most of the error is a shared lead-growing under-dispersion. Pooling all seven DEV datasets in every
  bucket gives steadier scales than the 2-dataset sub-daily cells of `freqlead`.
- **Expected effect.** About as large as item 1, and it may lose 0.005-0.01 of cov80_err on the sub-daily tasks.
- **Decision rule.** Keep `freqlead` only if it beats `lead` on the 5-seed mean cov80_err by more than 0.005 without
  costing wql. Otherwise the simpler `lead` wins.
- **Cost.**
  - One eval-only DEV re-fit: it re-runs the DEV eval, about 2-3 min, and writes the second json.
  - Five re-scores of 1-2 min each.

```
r2_devmap_lead_fit | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_DEV=1 TSAR_WM_RECAL=lead TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/r2_devmap_s0.pt | after=r2_devmap_s0 | run=r2_devmap_s0 | mem=2500
r2_devmap_ld_s0 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=0 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s0.pt | after=r2_devmap_lead_fit | run=wm_ssm_none_s0 | mem=2500
r2_devmap_ld_s1 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=1 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s1.pt | after=r2_devmap_lead_fit | run=wm_ssm_none_s1 | mem=2500
r2_devmap_ld_s2 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=2 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s2.pt | after=r2_devmap_lead_fit | run=wm_ssm_none_s2 | mem=2500
r2_devmap_ld_s3 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=3 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s3.pt | after=r2_devmap_lead_fit | run=wm_ssm_none_s3 | mem=2500
r2_devmap_ld_s4 | train_wm.py | TSAR_STEPS=2500 TSAR_BENCH=1 TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 TSAR_GIFT_CLEAN=1 TSAR_WM=ssm TSAR_WM_ABL=none TSAR_SEED=4 TSAR_WM_LONG=paths TSAR_WM_TTA=1 TSAR_WM_RECAL=file TSAR_WM_RECAL_MAP=runs/recal/r2_devmap_lead.json TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s4.pt | after=r2_devmap_lead_fit | run=wm_ssm_none_s4 | mem=2500
```
The eval-only DEV re-fit is done when its log has `dev_wql_rel:` (jobq's DEV rule). The checkpoint
data/ckpt/r2_devmap_s0.pt is written by item 1 (TSAR_RUN = job name).

## 3. Not proposed this round

- **In-sample `lead` on the main checkpoints.** The smoke shows it hurts (finding 2). It stays in the code only as
  a documented negative control. If the auditor wants it on record, one optional line costs 1-2 min:
  `r2_recal_insample_s0 | train_wm.py | <ref_ssm_s0 env> TSAR_WM_RECAL=lead TSAR_EVAL_ONLY=data/ckpt/wm_ssm_none_s0.pt | after=wm_ssm_none_s0 | run=wm_ssm_none_s0 | mem=2500`
- **Quantile averaging across origins.** The older origins lose the newest observations, so the result is wider but
  also biased, at 3x predict time. The out-of-family map does the widening for free.
- **A training-side calibration loss.** See finding 2: in-sample there is no mis-coverage for the loss to fix.

## Smoke results (tools/smoke.py, 60 s, ssm s0, 4 VAL tasks: ETT_1H, ETT_1W, bizitobs_l2c_5T, uk_covid_utla_1W/cumulative)

All runs used paths + TTA=1 and printed smoke_val_* only.

| run | smoke wql_rel | cov80_err | peak RSS |
|---|---|---|---|
| HEAD train_wm.py, defaults | 0.735948 | 0.274662 | 1.05 GB |
| new train_wm.py, defaults (bit-identity check, run twice) | 0.735948 | 0.274662 | 1.24-1.74 GB |
| RECAL=lead, in-sample, N=60 | 0.744673 | 0.304662 | 1.08 GB |
| RECAL=freqlead, in-sample, N=60 | 0.747242 | 0.309840 | 1.22 GB |
| TSAR_WM_TTA=mix (code then removed) | 0.736711 | 0.284002 | 1.96 GB |
| DEV fit + save (5 training steps, N=40, freqlead), 0.4 s fit | dev raw 1.790 (untrained) | | 1.98 GB |
| RECAL=file with that 5-step map on ssm s0 (plumbing only) | 0.834575 | 0.113774 | 1.46 GB |

- The 5-step map is junk: most scales sit at the grid caps. It shows that the json round trip and the `file` path
  work, and that the map moves VAL coverage strongly. It is not an effect estimate.
- The in-sample fit tables (cov80 on TRAIN before -> after, per bucket) are printed in every RECAL log.
- mase_rel was 0.884753 in every recal run, which confirms that the median is untouched.
