# Phase 6: testing the world-model hypothesis (Dr. Yara Haddad, 2026-10-30)

Read-only design document. Gate 1 holds: nothing here has been run, and nothing touches TEST. Tracks: T1 (latent
dynamics), T8 (state inside the Transformer), T11 (looped dynamics). Every number cited is from
`research_log.md`, `runs/search/leaderboard.md` or `baselines/bench/results.tsv`. There is no `runs/ledger.tsv`
yet (lesson L7), so the citations are to log lines. Treat all of them as preliminary until the ledger exists.

## 0. The hypothesis, stated so that it can fail

**H-WM:** Alma forecasts well because a learned transition `z_{k+1} = f(z_k, u_k)` carries information forward
in latent space, information that a model of the same size and compute without a carried state cannot represent.

A per-lead readout of the Transformer's last token would also produce a good forecast, and it is not a world
model. An ablation is decisive only if it separates those two explanations. Three facts about the code limit what
any ablation can show:

1. **The latent rollout covers at most 8 patches (64 steps).** `MAX_H=64` and `K_STEPS = MAX_H // PATCH = 8`
   (`train_arch2.py:83,252`). For H > 64, `predict` re-encodes the median forecast through the Transformer in
   chunks of 64 (`train_arch2.py:962-969`). On the 8 long-horizon VAL tasks, most lead times therefore come
   from observation-space autoregression, not latent dynamics. Section 3 counts the long-horizon tasks
   separately for this reason.
2. **The seasonal reference bypasses the state.** The decoder receives `ref_proj(r_k)` plus a learned per-quantile
   copy `r_k * ref_skip` (`train_arch2.py`, `rollout`). A2_DRIVE (which is the same as DYN_IN=2) also feeds
   `r_k` into the transition input. A "no dynamics" ablation has to keep these exogenous inputs, or it is
   ablating the seasonal input rather than the dynamics.
3. **A2_DREAD lets each rolled state cross-attend to the context tokens.** The state then works as a query, and it
   does not need to compress the context. A top configuration with DREAD on is partly a Transformer decoder.
   That is acceptable, but it has to be ablated before anyone calls it a world model.

## 1. Ablation matrix

Control arm: **R** = the train_syn5 recipe with `TSAR_A2_DRIVE=1` (the best single axis on DEV and VAL), run in
`train_arch2.py`. The second-stage control is **R+** = c0099-like (`A2_SLOTS=4 A2_DREAD=1 A2_MRES=8,16
A2_ZDIM=128 D_MODEL=64 N_LAYERS=3`). All arms use TSAR_BENCH=1, TSAR_STEPS=2500, seeds 0/1/2 and VAL, with paired
family CIs from tools/compare.py. The keep threshold is the 3-seed floor (about 0.008 wql_rel).

**Matching rule.** Parameters are counted with `sum(p.numel())`. FLOPs are counted as MACs per origin per forecast.
A GRUCell of width d costs 6d² MACs per step, so K steps cost 6d²K. A replacement is matched when both counts are
within ±3%. If it is too small, widen its hidden layer. If it is too large, shrink it. No arm adds encoder depth.

| ID | Ablation | Flag or minimal code change | Matching | Prediction if H-WM holds | Existing evidence (strength) |
|---|---|---|---|---|---|
| W0 | Control R / R+ | `TSAR_A2_DRIVE=1` (R+: c0099 flags) | reference | n/a | VAL a2_drive 0.6868 vs the syn5 incumbent at 0.6945/0.6953 over 3 seeds: -1.1%, **1 seed, about at the floor**. DEV c0099 0.885x, **1 seed, 1200 steps, best of 120 (selection bias)** |
| W1 | **Matched direct decoder (DIRECT)**: no carried state | new `TSAR_A2_DIRECT=1` (section 2) | params and MACs matched to the transition | worse than W0, and the gap grows with lead k | early exp11 "direct linear head" 0.709 vs 0.701, with better mase: **noise, 1 seed, old recipe, not matched** |
| W2 | State copy (A1): `z_k = z_0 + e_k`, drive kept | train_v2: `TSAR_STATE_OFF=copy`. **Caveat: it drops DRIVE as well** (`train_v2.py:1136`). train_arch2: add `STATE_OFF=copy` with `z_k = z_0 + step_emb[k] + drive(r_k)` | unmatched (removes the transition's params). W1 is the matched version | worse than W0 | A1 +2.6%, CI [0.993, 1.071], **not significant, 1 seed, old base** |
| W3 | Freeze the transition at random init | train_arch2: `TSAR_A2_FREEZE_TRANS=1` → `model.dynamics.requires_grad_(False)` after init | params and FLOPs identical (trainable count reported) | worse than W0. If equal, the decoder has learned to read a random reservoir | none |
| W4 | Zero the states at eval (A3) | train_v2: `TSAR_STATE_OFF=zero` (eval-only, an INFER_KEY). train_arch2: zero `S` before `dread_read` and the decode loop | identical weights | large loss | A3 +28.4%: **significant, but it only shows the path is used.** The state is out of distribution here, so this is not a dynamics test |
| W5a | Linear (Koopman) transition | train_v2: `TSAR_TRANS=koopman` (DYN_IN=2 for the drive) | match d. The Koopman R-net is smaller, so report params | if nonlinear dynamics matter, worse | koopman 0.733, ssm 0.724, attn 0.728, rssm 0.745 vs gru 0.726: **1 seed each, but consistent: the form of the transition hardly matters** |
| W5b | Residual MLP transition | `TSAR_A2_TRANS=res TSAR_A2_DEPTH=1` | width set to match 6d² | about equal to W0 | DEV c0039 0.986x (noise) |
| W6 | No synthetic data | `TSAR_SYN_FRAC=0` (and `TSAR_SYN5=0`). First check that `n_syn=0` takes a clean path in a 30-step smoke | same model | worse, and most on multi-seasonal and regime tasks (the synthetic prior encodes those dynamics) | syn_frac 0.5 + synth3: CI [0.929, 0.989] **win**. synth5 adds nothing: syn5_only 0.6945 vs 0.6953 (3 seeds) |
| W7 | No cross-series interaction | **None exists to remove.** The bench protocol is univariate (covariates dropped, multivariate series split), and slot mixing happens within a series. The nearest test is covariates withheld | n/a | n/a | b_cov1 0.7252 with covariates, 0.7133 without: no cross-variable gain. Report as out of scope, not as a result |
| W8a | Shorter context at inference | eval-only `TSAR_ENS=128+` or `64+` (a crop on an existing checkpoint) | same weights | a world model that summarises history should degrade gracefully | none |
| W8b | Shorter context in training | `TSAR_CONTEXT=128` or `64` | same params, fewer FLOPs | worse on multi-seasonal tasks | ctx1024 0.6844 vs ctx256 0.6853: **longer context gives nothing** |
| W9 | Remove the decoder bypass (R+ only) | R+ with `TSAR_A2_DREAD=0`, then also with `TSAR_A2_SLOTS=1` | report params | if R+'s gain lives in the dynamics, it survives without DREAD | DEV ofat dread 0.987x, slots4 0.977x: **1 seed each** |
| W10 | Rollout length vs feedback | `TSAR_MAX_H=32` / `128` (K=4 / 16) | params differ only by step_emb | a longer latent rollout should help long horizons | DEV max_h=32 1.037x (worse), max_h=128 0.976x: **1 seed** |
| W11 | Statistical baselines | already scored: `baselines/bench/per_task/*_val.csv` | n/a | Alma beats them on every pattern tag | AutoTheta 0.959, SeasonalNaive 0.988 (AutoETS 2.68 has blow-ups; per lesson L6 it is not ranked) vs Alma about 0.69. Strong, but these baselines are weak |

## 2. The matched sequence-model control (W1, DIRECT)

There is no DIRECT path in `train.py`, `train_v2.py` or `train_arch2.py`. `grep -i direct` finds only the
TSAR_JUMP comments, and JUMP is a gated *addition* to the rollout, not a replacement for it. The DIRECT that
program.md mentions in its ablation duty was never implemented in the current scripts.

**The minimal change** goes in `train_arch2.py`, behind `TSAR_A2_DIRECT` (0|1, default 0, built only when on, so
defaults stay bit-identical). Add it to the axis checks and to the module docstring.

```python
# __init__ (after the other A2 modules)
if A2_DIRECT:   # per-lead map from the origin state; no state carried between leads
    self.direct = nn.Sequential(nn.LayerNorm(2 * ZD), nn.Linear(2 * ZD, H_DIR), nn.GELU(), nn.Linear(H_DIR, ZD))
    nn.init.zeros_(self.direct[-1].weight); nn.init.zeros_(self.direct[-1].bias)
    # self.dynamics is NOT built (it would otherwise count as unused parameters)
# rollout(): replace `z = self.dynamics(u, z)` with
if A2_DIRECT:
    zk = z0 + self.direct(torch.cat([z0, u.expand_as(z0)], -1))   # u = step_emb[k] (+ drive(r_k)), the same inputs
else:
    z = self.dynamics(u, z); zk = z
```

- **Param match.** The GRUCell has 6d² + 6d parameters. The MLP has 2d·H_DIR + H_DIR·d + 4d + H_DIR + d.
  Setting **H_DIR = 2d** gives 6d² + 8d, which is within 0.5%. With SLOTS > 1 the MLP is applied per slot, and the
  slot-mixing attention is kept, but it mixes slots of the same lead and never across leads.
- **FLOP match.** The MLP runs once per lead, K times, at 6d² MACs each, which is equal to the GRU.
- **What it keeps.** The Transformer, to_state, step_emb, drive, decoder, seasonal skip, DREAD, the loss and the data.
  The only change is that no latent state flows from lead k to lead k+1.
- **Guards.** Refuse it with LOOPS, LATENT > 0 (there is no prior trajectory to make consistent), JUMP, TRANS other
  than gru, and PREFIX.
- **Smoke test** (when Gate 1 lifts): 30 steps through tools/smoke.py. Print the trainable-parameter count for
  both arms, and require the difference to be at most 3%.

## 3. Pattern taxonomy of the VAL tasks

The tags were computed from VAL contexts and targets only, through `bench._build_task`, with no model and no TEST
data. The script took 34 s; it was a scratch file and is not kept in the repo. Each statistic is a mean over the
task's windows.

- **trend change**: on a share ≥ 0.5 of windows, |slope(target) − slope(last max(H, 2s) context points)| · H / sd(context) > 1.
- **regime shift**: on a share ≥ 0.3 of windows, either |median(target) − median(last max(s, H) of context)| / IQR(context) > 1, or the sd ratio of target to that tail is > 3 or < 1/3.
- **intermittent**: zero share of the context ≥ 0.15.
- **multi-seasonal**: ACF ≥ 0.2 at the primary season s and also at the slower season (168 for H, 365 for D), after removing a moving average. For sub-hourly tasks, 2048 points cannot show the weekly cycle, so they are tagged "daily+intraday (weekly invisible)".
- **long horizon**: H ≥ 96, which means beyond the 64-step latent rollout, so median feedback is in play.

| Tag | VAL tasks |
|---|---|
| Long horizon (8) | ETT_15T, ETT_1H, LOOP_SEATTLE_1H, M_DENSE_1H, SZ_TAXI_15T, SZ_TAXI_1H, bizitobs_l2c_5T, jena_weather_10T |
| Multi-seasonal (5 + 4) | M_DENSE_1H (ACF 0.78 / 0.86), LOOP_SEATTLE_1H (0.55 / 0.60), ETT_1H (0.39 / 0.31), SZ_TAXI_1H (0.22 / 0.22), jena_weather_1H (0.22 / 0.22, weak). Weekly invisible: ETT_15T, SZ_TAXI_15T, bizitobs_l2c_5T, jena_weather_10T |
| Intermittent (6 + 2) | bizitobs_l2c_1H (0.25), bizitobs_l2c_5T (0.27), favorita_stores_1D (0.28), favorita_stores_1M (0.23), favorita_stores_1W (0.21), hierarchical_sales_1D (0.25). Partly, from zero-valued channels: jena_weather_10T (0.19), jena_weather_1H (0.17) |
| Regime shift (10) | solar_1W (0.82), uk_covid_utla_1W/cumulative (0.86), uk_covid_utla_1D/new (0.50), bizitobs_l2c_1H and 5T (0.50 / 0.49), uk_covid_nation_1W/new (0.42), uk_covid_nation_1D/cumulative (0.39), uk_covid_nation_1D/new (0.34), ETT_1W (0.31), favorita_stores_1M (0.30) |
| Trend change (9) | uk_covid_utla_1W/cumulative (0.74), ETT_1W (0.71), uk_covid_utla_1D/new (0.69), kdd_cup_2022_1D (0.67), jena_weather_1D (0.55), ETT_15T (0.54), bizitobs_l2c_1H and 5T (0.54 / 0.53), favorita_stores_1M (0.53) |
| None of the above (6) | LOOP_SEATTLE_1D, M_DENSE_1D, restaurant, solar_1D, hierarchical_sales_1W, uk_covid_nation_1W/cumulative (smooth cumulative series, which Alma already wins) |

**Caveats.** n=30 tasks, and the tags are collinear. As gap_analysis.md found, long horizon, multi-seasonal and
sub-hourly frequency are almost the same 8-9 tasks. The trend-change and regime tags on short weekly tasks
(n ≤ 5 windows) are noisy. Report each tag as a slice of the paired family CI, never as a per-task win count.

## 4. Run order (cheapest decisive first) and compute

Timings: the incumbent is about 147 s per 1200 steps (DEV log), so 2500 steps plus VAL eval is about 6 min. R+
(c0099: slots4 + dread) is 489 s per 1200 steps, about 3.3x slower, so roughly 18 min. Eval-only re-scores take
about 1-2 min.

| Step | Arms × seeds | Runs | Time |
|---|---|---|---|
| 0. Free: re-slice every existing VAL dump and the baseline per_task CSVs by the section 3 tags. Also W8a (crop 128/64) and W4 on the existing train_v2 checkpoints | eval-only | about 4 | about 8 min |
| 1. **W1 DIRECT vs W0 R** (the decisive one) | 2 × 3 (a2_drive seed 0 exists, so 5 new) | 5 | about 30 min |
| 2. W2 copy+drive, W3 frozen | 2 × 3 | 6 | about 36 min |
| 3. W6 no-synthetic | 1 × 3 | 3 | about 18 min |
| 4. W5a Koopman+drive, W10 MAX_H=128 | 2 × 3 | 6 | about 36 min |
| 5. W8b CONTEXT=128 | 1 × 3 | 3 | about 15 min |
| 6. R+, R+ DREAD=0, R+ DIRECT | 3 × 3 | 9 | about 2.7 h |

The total is about 5.5 h of queue time, with one job at a time. Steps 1 and 2 (about 1 h) answer the main
question. Step 6 should run only if step 1 is ambiguous, or if R+ is to be promoted.

## 5. Judgement

- **Where the Transformer does the work.** Almost everywhere. Under the old base, the state is used (A3), but a
  rollout without a learned transition costs only +2.6%, and the CI crosses 1 (A1). Five transition families land
  within about 1% of each other, apart from rssm. JEPA is neutral, and longer context gives nothing. Taken
  together, the GRU behaves like a lead-indexed readout of z_0. The largest gains (synthetic data, norm=mean,
  drive) are data, normalisation and exogenous-input effects, not effects of the dynamics.
- **Where the latent dynamics might do real work.** First, **mid leads, k = 3..8, on multi-seasonal tasks** (M_DENSE_1H,
  LOOP_1H, ETT_1H), where the drive gives the rollout a phase signal to integrate. Second, on **regime-shift and
  trend-change tasks**, where a state that evolves could beat a static readout. Long-horizon tasks cannot test this,
  because past 64 steps the model runs median feedback, not latent rollout.
- **The single most decisive experiment.** W1: a parameter- and FLOP-matched DIRECT decoder against the R rollout,
  3 seeds, at 2500 steps, sliced by lead k and by tag. If DIRECT is within the noise floor overall and on the
  multi-seasonal slice, the claim has to be scoped as follows: "Alma is a Transformer forecaster with a latent
  bottleneck; its recurrent dynamics are not shown to add accuracy." If the rollout wins, and its advantage grows
  with k, H-WM survives its first falsification test.
