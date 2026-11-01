# World-model families: shared harness and fairness rules (Haddad, 2026-10-30)

Wave 4 found that the GRU latent rollout ties a no-dynamics direct head (gc_roll vs gc_direct, +0.002 wql_rel), so
small edits to the rollout are used up. Wave 5 screens world-model families that are *radically different*, and
compares them in one harness so that only the family differs between them.

- Harness: `train_wm.py`. It copies, and does not import, the data, recipe, normalisation, loss and eval from
  `train_arch2.py`. `train_arch2.py` is unchanged.
- Families: package `wm/`. The interface is in `wm/base.py`, the registry in `wm/__init__.py`, and the reference
  family in `wm/gru.py`.
- Selection: `TSAR_WM=<family>` (default `gru`), `TSAR_WM_ABL=none|direct|copy|zero`, `TSAR_WM_WIDTH=<float>`.

## 1. What the harness owns (identical for every family)

| Part | What it does |
|---|---|
| Data | bench TRAIN (`tools/bench.load_train`), `TSAR_GIFT_CLEAN`, `TSAR_BENCH_EXTRA`, `TSAR_DEV` holdout. Only TSAR_BENCH=1 is supported. Never `data/bench/gep/`. |
| Synthetic mix | `TSAR_SYN_FRAC`, `TSAR_SYN3`, `TSAR_SYN5`, `TSAR_SYN2`, `TSAR_SYN_CURR`, `TSAR_AUG`, as in train_arch2 |
| Windows | CONTEXT=256, MAX_H=64, PATCH=8, so K=8 lead patches; BATCH=128. Every patch in a window with an observed future is an origin. |
| Normalisation | per-patch causal running frame (`TSAR_NORM=last`: last value and mean absolute step), asinh, frame features |
| Seasonal reference | seasonal-naive forecast from each origin, in that origin's frame. Given to the family as `ctx.ref`. It is also added through a learned per-quantile skip (`ref_skip`, 9 params) that every family gets. |
| Loss | pinball over 9 quantiles in model space, plus the family's declared aux losses |
| Optimiser | AdamW at lr 3e-3 and wd 0.01, 100-step warmup, cosine to 5% (`TSAR_SCHED=wsd` optional), grad clip 1.0, optional `TSAR_EMA`, fixed `TSAR_STEPS` |
| Eval | `bench.evaluate` on VAL (or DEV); horizons above MAX_H run in MAX_H-step chunks. `TSAR_WM_LONG=median` (default) feeds the median back; `TSAR_WM_LONG=paths` rolls quantile paths and mixes them (section 7). TEST is refused outright. |
| Guards | strict TSAR_* key check (the keys train_wm.py reads, the bench tool keys, and the selected family's `env_keys`); the causality check; checkpoint/resume with a config fingerprint (jobq sets `TSAR_RUN`); `TSAR_EVAL_ONLY` (VAL/DEV) |

## 2. The interface (wm/base.py)

```python
class MyFamily(WMFamily):                       # an nn.Module
    name = "myfam"
    aux_weights = {"kl": 0.1}                   # every aux key must be declared; {} if none
    env_keys = ("TSAR_WM_MYFAM_SLOTS",)         # every env key it reads; prefix TSAR_WM_<NAME>_

    def __init__(self, cfg: WMConfig): ...      # read sizes from cfg; refuse meaningless ablations here
    def encode(self, ctx: Context) -> Any: ...  # once per batch; must be causal in the patch index
    def decode(self, enc, ctx: Context, b: LongTensor[N], j: LongTensor[N])
        -> tuple[Tensor[N, K, PATCH, NQ], dict[str, Tensor]]: ...
    def param_groups(self) -> dict[str, nn.Module]: ...   # optional, e.g. {"transition": self.dyn}
```

- `WMConfig`: `patch, context, max_h, k_steps, n_patch` (tokens per training window), `nq, tok_dim, d_model`
  (after width scaling), `n_layers, n_heads, width, abl`.
- `Context`: `tok [B,n,2*PATCH+3]` (patch values in the patch's own asinh frame, the observed mask, and the
  frame features), `x [B,L]` (raw values with NaN), `loc, scale, cnt [B,n]`, and `ref [B,n,MAX_H]` (clamped to
  ±20). Index j only uses points up to the end of patch j.
- Output: quantiles in the frame of origin (b, j), for lead patches k = 0..K-1. The harness adds
  `ref_skip * ref`, scores the pinball loss, and maps back to raw space at eval.
- Origins: training passes every valid (b, j) of a window, so N is large. Eval passes one origin per series,
  j = n-1. A family that cannot share work across origins can recompute per origin in `decode`; that is legal
  but slow, and it pays for it in MACs and wall time.

### The four ablation modes (TSAR_WM_ABL)

| Mode | Meaning | Params |
|---|---|---|
| `none` | the family as designed: a state carried and rolled from lead k to lead k+1 | reference |
| `direct` | no carried state. Lead k comes from the origin state and lead k's inputs only, through a replacement matched to the transition. | within 3% of `none` |
| `copy` | no transition: z_k = z_0 + the lead's input embedding | fewer (report them) |
| `zero` | rolled states zeroed before the decoder | identical to `none`; also usable eval-only on a `none` checkpoint |

A family may refuse a mode that is meaningless for it with `wm.base.refuse(name, abl, why)`, a SystemExit. The
refusal and its reason go in the family's docstring and in the screen report. Example: a family with no explicit
per-lead state cannot define `copy`.

## 3. Adding a family

1. Create `wm/<name>.py` with a `WMFamily` subclass. Write a module docstring that covers the design, the four
   ablations (or the refusals), and the env keys it reads.
2. Register it in `wm/__init__.py`: `REGISTRY["<name>"] = "wm.<name>:<Class>"`.
3. Family-specific knobs are `TSAR_WM_<NAME>_<KNOB>`, listed in `env_keys`, read in the family module and
   defaulted. The strict-key check accepts only the harness keys and the selected family's `env_keys`, so another
   family's key (or a train_arch2 `TSAR_A2_*` flag) is refused.
4. Check size: `TSAR_WM_INFO=1 TSAR_WM=<name> uv run python train_wm.py` prints params, MACs and the causality
   result, then exits without loading data (a few seconds). Do this for all four ablations.
5. Smoke test (team rules, one at a time, under 60 s and under 2 GB, eval path exercised, lesson L4):
   ```
   uv run tools/smoke.py -- env TSAR_WM=<name> TSAR_BENCH=1 TSAR_GIFT_CLEAN=1 TSAR_STEPS=20 TSAR_BATCH=32 \
     TSAR_SYN_FRAC=0.3 TSAR_SYN3=1 TSAR_SYN_CURR=0.9 TSAR_AUG=0.5 TSAR_SYN5=1 \
     TSAR_WM_EVAL_TASKS=ETT_1W,ETT_1H uv run python train_wm.py
   ```
   ETT_1H (h=168) exercises the chunked long-horizon path. A subset prints `smoke_val_*` lines and never
   `val_*`.
6. Add jobs to `runs/jobs/wave5.txt` (format below). Do not run them; the lead queues them.

## 4. Fairness rules for a screen

1. **Same everything else.** Same harness, data (`TSAR_GIFT_CLEAN=1`), recipe flags, `TSAR_STEPS=2500`,
   `TSAR_BENCH=1`, seeds 0/1/2, eval. A family may not change the loss, normalisation, data, optimiser or eval. If
   it needs one of these, the change goes into the harness as a flag that every family then runs.
2. **Matched size.** At `TSAR_WM_WIDTH=1`, total params (including `ref_skip`) must be within ±15% of the `gru`
   reference, 369,489. Report MACs per forecast as well. A family above 2x the reference's MACs reports that, and
   it is compared at matched MACs too (by lowering its width).
3. **Four modes.** Every family reports none/direct/copy/zero, or a written refusal. A family is called a
   *world model* only if `none` beats its own `direct` beyond the noise floor (Gate 3 of docs/preregistration.md:
   at least 0.020 wql_rel on 3 fresh seeds, or 0.013 on 5 seeds for gains of 0.008-0.020). Otherwise it is a
   forecaster with a latent bottleneck, which may still be worth keeping, but under that name.
4. **Causality.** `wm_causal: ok` must appear in the log, and a failure stops the run. No family may read future
   patches of a training window.
5. **Honest seeds.** A single seed is a screen. Keep or discard follows docs/team.md and docs/preregistration.md.
   Any wql_rel above 2, or non-finite, is labelled failed (lesson L6).
6. **No external models.** No foundation-model weights, outputs or distillation targets. Only data reached
   through `ctx`.

## 5. Width ladder (for later scale runs; params only, nothing trained)

`d_model = 8 * round(96 * TSAR_WM_WIDTH / 8)`. Depth stays at `TSAR_N_LAYERS` (2); it can be raised
independently. Family `gru`, `TSAR_WM_ABL=none`, measured with `TSAR_WM_INFO=1`:

| TSAR_WM_WIDTH | d_model | params | MACs per forecast | target |
|---|---|---|---|---|
| 1.0 | 96 | 369,489 | 10.4M | 0.4M (screening default) |
| 1.6667 | 160 | 1,004,881 | 28.3M | 1M |
| 3.3333 | 320 | 3,955,281 | 111.5M | 4M |
| 5.3333 | 512 | 10,063,953 | 283.8M | 10M |

A family whose parameters do not scale like d² (for example, many small experts) must give its own ladder in
its docstring, with widths that reach the same four targets within ±15%.

## 6. Jobs file format (runs/jobs/wave5.txt)

`name | train_wm.py | KEY=VAL ... | mem=<MB>`, one job per line; tools/jobq.py runs them in order. The reference
parity check for `gru` against gc_roll (VAL 0.6958 ± 0.0070 over 3 seeds) is at the top of the file. The runs/
directory is gitignored, so the jobs file lives only on this machine.

## 7. Long horizons (2026-10-31): `TSAR_WM_LONG` and `TSAR_MAX_H`

Both are off by default, and the defaults are bit-identical to c06e0a4 (smoke diff in the commit message).

**`TSAR_WM_LONG=paths`** is an inference key: it may be set with `TSAR_EVAL_ONLY` on a checkpoint trained with
median, like `zero`. For h > MAX_H, chunk 1 is one rollout. Its sorted quantiles at the levels in
`TSAR_WM_LONG_PATHS` (default all 9) become |S| context paths. Each path is rolled forward one chunk, which gives |S|
predictive quantile sets. These are combined as a *mixture distribution*, `train_wm.mixture_quantiles`:
- each component's CDF interpolates its 9 knots linearly, with linear tails that carry the 0.1 mass on each side;
- the mixture CDF, `sum_c w_c F_c`, is then exactly piecewise linear, so it is evaluated at every breakpoint and
  inverted exactly;
- the weights `w_c` are a midpoint-rule quadrature over the latent level u ~ U(0,1): 0.15, 0.1 x 7, 0.15 for the 9
  levels.

Path p then appends the mixture's level-tau_p quantile, so each path stays a comonotone level-tau_p trajectory. The
cost is 1 + |S| x (chunks - 1) rollouts instead of `chunks`. With |S| = 9 at h = 168 that is 19 vs 3, and at
h = 288 it is 37 vs 5. Checks: with C = 1 the method returns the component's own quantiles exactly, and it matches a
2M-sample Monte Carlo of a two-component mixture to 3 decimals. Unlike Chronos-Bolt's pooling of the 81 values as
samples, it respects the weights and the mass between and beyond the knots.

**`TSAR_MAX_H=<int>`** (a multiple of PATCH) sets the native rollout. The family imagines K = MAX_H / PATCH lead
patches from its latent state with no observation feedback. Training windows are CONTEXT + MAX_H long, and origins
j <= CONTEXT/PATCH - 1 see the full future, so the far leads are learned from contexts no longer than eval's. The
`step_emb` and `pos` tables grow with K: at MAX_H = 320 that is +3,072 to +9,216 params, so report it. A family that
cannot roll K patches sets `max_k` / `max_k_why` in its class, and the harness refuses with that reason.
- `TSAR_WM_ORIGINS=<int>` caps the training origins per batch with a uniform random subset drawn from the data rng.
  With all origins, MAX_H = 320 has about 9x the lead states of MAX_H = 64. 1024 origins x 40 leads matches the
  default's ~5k x 8.
- `TSAR_WM_LEADW=<float>` sets the pinball weight of steps beyond 64 relative to steps 1..64. The default of 1 is
  flat, and then steps beyond 64 carry 80% of the loss at MAX_H = 320. At 0.25 the two halves weigh the same.
- CONTEXT stays 256. A 288-step hourly forecast is longer than the context's 1.5 weeks; whether 512 helps is a
  separate screen.
