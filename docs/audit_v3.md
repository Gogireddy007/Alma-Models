# Audit v3: new model scripts, autoresearch harness, multiplicity, TEST integrity

Auditor: Dr. Sofia Andrade (independent evaluation, red team). Date: 2026-10-30.
Branch `autoresearch/oct29`, HEAD `abe9f26` plus uncommitted `train_v2.py` (DUALDEC, JEPA_FORM),
`train_flow.py`, `tools/synth5.py` and `tools/verify_claims.py`.

**Scope.** Everything added since `22d58cb` (audit_v2 fixes):

- the scripts `train_v2.py`, `train_arch.py`, `train_ms.py`, `train_data.py`, `train_tta.py` and
  `train_flow.py` (work in progress);
- the tools `tools/bench.py` (Chronos-Bolt `_mem_guard`), `tools/jobq.py`, `tools/leaderboard.py`,
  `tools/promote.py` and the Gate G code in `tools/compare.py`;
- the artefacts `runs/leaderboard.md`, `runs/jobs/*.txt`, `runs/tta/sweep1.*`,
  `data/bench/test_access.log` and `baselines/bench/`.

**Method.** The audit was read-only. I used grep, `diff` against the parent script, targeted line
reads, and pandas on the existing per-task CSVs. No model was trained, loaded or scored, and no
TEST file was opened. I only `stat`-ed TEST files for their access times.

**Bottom line.**

- **Every inference path in the new scripts is causal.** The `rssm` posterior, the `attn` keys,
  the `flow` posterior target, the `ms` far-horizon loss and its seasonal reference, `DUALDEC`
  and path TTA all touch future values only inside the training loss, and only on TRAIN data.
- **TEST is still untouched.**
- **There are two blockers, both about process.**
  - Nine of the twelve training scripts carry none of the audit_v2 guards. A single environment
    variable makes them retrain on TEST and score it.
  - No number on the leaderboard can be traced to a committed script and config.
- **There is one real leak path through a flag.** `TSAR_TTA_STEPS > 0` adapts shared weights on a
  batch that contains later windows of the same series. Those windows' contexts hold the earlier
  windows' targets.
- **The autoresearch tools inflate or misdescribe results in three ways:**
  - the reference drifts as it gains seeds;
  - rung 3 silently drops flags that `train_v2.py` does not know;
  - about 60 VAL screens have no multiplicity accounting.

Severity scale:

- **BLOCKER**: a number reported now would be invalid, or the single-use TEST set can be burned by
  one plausible action.
- **HIGH**: a leak or a misdescribed result is reachable through a plausible action.
- **MEDIUM**: it biases the inference or weakens the evidence.
- **LOW**: tidy it up or disclose it.

---

## 0. Verified clean (no action needed)

| Item | Where | Finding |
|---|---|---|
| `TSAR_TRANS=rssm` posterior | `train_arch.py` `RSSMTransition.begin/forward`; `arch_loss` | `post` is set only when `ctx` holds `"post"` **and** `self.training`. `arch_loss` builds it from `z.detach()` at patches j+1..j+K of the **training** window. At inference `forecast` passes `ctx=None` for rssm, and the open-loop state uses the prior mean `mp`. There is no inference path to future data. |
| `TSAR_TRANS=attn` causal mask | `AttnTransition.begin` | `lag = j - i` (query j, key i), and `masked_fill(lag < 0, -inf)` blocks keys after the origin. At inference, `j = n-1` over context states only. The encoder is causal (audit_v2 §1). |
| `koopman`, `ssm` | `train_arch.py` | Pure functions of z and the step input. SSM state is reset by `begin()` on every rollout. |
| `TSAR_LONG_H` far-origin loss | `train_ms.py` `train_loss_long` | The encoder sees `win[:, :TWIN]`. Far targets `win[:, TWIN:]` enter only the pinball target. The sampled origins `j` lie inside the window. |
| `seasonal_ref(H=LONG_H, j=…)` | `train_ms.py:378-411` | `idx = end + k - m*ceil(k/m) <= end`, and `ffill` uses a causal `cummax`. At inference, `x` is the context only (`:847-852`). |
| `TSAR_FLOW*`, `TSAR_DVF` | `train_flow.py` `FlowTransition`, `DVFHead`, `forecast` | The posterior target and the running `ms` buffers are updated only in train mode, from the training window. `forecast` never passes `post`. EMA copies the buffers (`:1252`). |
| `TSAR_DUALDEC` (uncommitted) | `train_v2.py` `_dualdec_term` | Called only from the training loss. Decoding the encoder state that has seen the target patch is a teacher signal on TRAIN data. Inference never calls it. |
| Path TTA, clamp, crop/flip ensembles, SELFCAL, TTT | `train_tta.py:804-916`, `train_v2.py:1180-1374` | Every quantity is computed from the row's own context or the model's own quantiles. `ctx_scale` uses the context only. TTT and SELFCAL are per row. (Batch-wide `TTA_STEPS` is **not** clean: see H-1.) |
| synth4 / stretch / KernelSynth | `tools/synth4.py`, `train_data.py:148-175, 278-290` | No benchmark file is read (`synth4.py:519` reads TRAIN `monash_traffic` in `__main__` only). Stretch copies TRAIN rows. Gap weights come from TRAIN frequency shares. |
| VAL/TEST disjointness, TEST guard | `bench.py:248-264` | `_guard` plus the new `_guard_config` inside `_build_task` cover every bench path into a TEST parquet. |
| **TEST untouched** | `data/bench/test_access.log`, `runs/`, `data/bench/fev/` | The log has the same 6 target-free entries as at audit_v2 (last at 13:43). There is no `TSAR_SPLIT` in any file under `runs/`. No log has a test or holdout result line (only the old-suite `zs4_wql_rel` print). The latest access time over the 48 TEST parquets is 13:27:33, before the last logged access. That is weak evidence, but consistent. |
| Bolt and baseline parity on VAL | `baselines/bench/per_task/*_val.csv` | `sn_wql` and `n_forecasts` are identical between chronos2 and auto_ets on all 30 tasks. The audit_v2 parity still holds for the new auto_ets and auto_theta CSVs. |

---

## 1. BLOCKERS

**B-1. Nine training scripts bypass every audit_v2 guard. `TSAR_SPLIT=test` makes them retrain and score TEST.**

- **Where:** these scripts set `split = "test" if os.environ.get("TSAR_SPLIT") == "test"` with no
  `TSAR_EVAL_ONLY`, no `cfg_mismatch` and no `ZS and BENCH` check:

  | Script | Line |
  |---|---|
  | `train_arch.py` | 1007 |
  | `train_ms.py` | 955 |
  | `train_data.py` | 932 |
  | `train_tta.py` | 1058 |
  | `train_flow.py` | 1139 |
  | `train_dyn.py` | 788 |
  | `train_calib.py` | 891 |
  | `train_cov.py` | 961 |
  | `train_latent.py` | 890 |

  Resume checks only `zs` (for example `train_arch.py:944`, `train_flow.py:1208`). Only `train.py`,
  `train_fast.py` and `train_v2.py` carry P3, P4 and P7.
- **Failure scenarios:**
  1. `TSAR_SPLIT=test` is left exported in the lead's shell. `jobq.py:216` passes `os.environ`
     through to every job, so every queued arch, ms, data or flow job trains a fresh model and
     scores TEST. `bench._guard` accepts this, because it requires only `TSAR_SPLIT=test`. The
     single-use TEST set is burned for configurations nobody pre-registered.
  2. A rung-3 or rung-4 winner trained with one of these scripts has no `cfg` in its checkpoint.
     `TSAR_EVAL_ONLY` in `train.py` and `train_v2.py` refuses it, so the winner cannot be frozen.
     The temptation is then to score TEST by "just running the script".
  3. `TSAR_RUN=q_base2 TSAR_BENCH=1 uv run train_data.py` silently resumes the old-suite
     `data/ckpt/q_base2.pt`. Fourteen non-bench checkpoints are still on disk (`big1/2`, `exp2x`,
     `q_*`). They were trained on `monash_fred_md`, `exchange_rate` and `monash_covid_deaths`, so
     this reopens audit_v2 L-1.
- **Fixes:**
  - Move the TEST precondition into `bench._guard` itself, so every script inherits it. For
    split `test`, require `TSAR_EVAL_ONLY` and `TSAR_CKPT_SHA256` in the environment as well as
    `TSAR_SPLIT=test`, and refuse otherwise.
  - Make `jobq.py` strip or refuse `TSAR_SPLIT`, `TSAR_EVAL_ONLY` and `TSAR_CKPT_SHA256` from the
    inherited environment.
  - Rule: a frozen candidate must be trained by a guarded script (today, `train_v2.py`). Port any
    winning flag there first.
  - Move the old-suite checkpoints out of `data/ckpt/`.

**B-2. No reported number is traceable to code plus config.**

- **Where:** all 38 rows of `runs/leaderboard.md` are `UNVERIFIED (no prov)`. There are 0
  `runs/*.prov.json` files. Provenance started at `abe9f26`, after every finished run.
- **The logs carry no config.** They begin at `step 0` and print no env and no git hash, so a log
  cannot say which flags produced it.
- **Grouping is by name only.** `jobq.done()` (`jobq.py:52`) skips any job whose log contains
  `val_wql_rel:`, whatever the config. The leaderboard groups `X` with `X_s1` by name alone
  (`leaderboard.py` `split_seed`).
- **Failure scenario:** a jobs file is edited (flags changed, same name). The stale log is kept as
  `done` and reported under the new description. Or `X_s1` was run with different flags from `X`
  and averaged in as a seed.
- **Fixes:**
  - Only runs with a `prov.json` that is clean (`git_dirty=false`, script sha equal to the commit)
    may feed a promotion decision or any number given to the user.
  - Before any rung-3 decision, re-run the rung-2 confirmations under jobq provenance. This is
    cheap compared with mis-merging a recipe.
  - Make `done()` also require that the prov `env` equals the jobs-file env. Otherwise mark the run
    stale.

---

## 2. HIGH

**H-1. `TSAR_TTA_STEPS > 0` is a transductive target leak across windows.**

- **Where:**
  - `train.py:682-700`, `:768`;
  - `train_v2.py:1124-1145`, `:1406`;
  - `train_tta.py:710-730`, `:912`.
- **How it leaks:**
  1. `adapt(x)` fine-tunes the shared dynamics and decoder on the futures inside **every**
     context of the chunk (`context_loss`, up to 512 rows).
  2. `bench.evaluate` passes all windows of all series of a task in one call.
  3. `bench.py:693-695` documents that window k+1's context contains window k's target.
  4. So the forecast for window k is made by weights fitted to its own target values, read
     from window k+1's context.
- **Status:** not exercised. `runs/tta/sweep1.txt` has no `TTA_STEPS` config.
- **How it could still be used:**
  - `TSAR_TTA_STEPS` is in `INFER_KEYS` (`train_v2.py:61`), so it can be applied to any VAL
    checkpoint as an "exploratory" re-score.
  - If it were ever in a training env, it would apply on TEST too.
- **Fixes:**
  - Hard-refuse `TSAR_TTA_STEPS > 0` under `TSAR_BENCH=1`.
  - Alternatively, make the adaptation strictly per row (per-row parameters, as TTT does), so that
    nothing is shared across rows.
  - Add a unit test: forecasts must be invariant to which other contexts share the batch. This
    test should apply to every inference option.

**H-2. `promote.py --rung3` silently drops winning flags that `train_v2.py` does not implement.**

- **Where:**
  - `promote.py:163`, which unions the winners' env from their jobs files;
  - `:172-187`, which emits the result on `--script train_v2.py` (the default at `:202`).
- **Which flags are lost:** `train_v2.py` has no `TSAR_TRANS`, `TSAR_LONG_H`, `TSAR_MS_COARSE`,
  `TSAR_HDEC`, `TSAR_FLOW_*`, `TSAR_DVF`, `TSAR_CHUNK_CARRY` or `TSAR_COV` (checked by grep). It
  does not reject unknown `TSAR_*` keys.
- **Failure scenario:** `arch_attn` is confirmed at rung 2. `r3_combo` then lists
  `TSAR_TRANS=attn` but trains a GRU. Its stored config, and any write-up, describe an attention
  transition that is not in the model. The `loo_arch_attn` row is an empty ablation that "shows"
  the component does not matter.
- **Fixes:**
  - `train_v2.py` must refuse any `TSAR_*` key it does not read. Keep an allow-list built at
    import, plus `CTRL_KEYS`.
  - `promote.py` must refuse a winner whose script is not the rung-3 script, unless its flags are
    in that allow-list.

**H-3. Reference drift: the leaderboard comparisons are not seed-matched, and the reference set changes over time.**

- **Where:**
  - `leaderboard.py:154`: `R` is every reference seed present at run time (b_base256 plus
    c_base_s1 now, plus c_base_s2 later);
  - `:168-172`: each row is compared with the seed-mean of `R`, whatever its own seeds.
- **Consequences:**
  - A rung-1 (seed-0) row is compared with a 2-seed and later a 3-seed reference mean.
  - The ratio and the CI of every row move when a reference seed lands, so a promotion decision
    depends on when `promote.py` ran.
  - The paired seed-0 versus seed-0 comparison is thrown away. It is the comparison that shares
    data and synthetic RNG streams with the reference (common random numbers).
- **More problems:**
  - CI_fam ignores training noise (audit_v2 S-5, still open for screening). The reference's own
    two seeds differ by 0.0057 in `val_wql_rel` (0.7258 vs 0.7315), a 0.8% swing. Several rung-1
    `**` rows sit within about 2 to 3% of 1.
  - `c_base_s1` is in both the reference group and the `c_base` group, so those two rows are not
    independent.
- **Fixes:**
  - Freeze the reference log list per rung (record it in the jobs file).
  - Compare seed-k with reference seed-k (paired by seed), and gate rung 2 on per-seed ratios
    being below 1 (G4-style).
  - Print the reference list in the md header. It already does this; also make `promote.py`
    refuse a board whose `ref_logs` differ from the rung's frozen list.

**H-4. Multiplicity and VAL overfitting: about 60 screens on 30 tasks and 13 families, with no accounting and no held-out step before TEST.**

- **Count so far:**
  - 38 leaderboard rows, about 35 of them non-reference configs;
  - 20 inference configs in `runs/tta/sweep1`;
  - 30 rung-2 jobs, and 16 arch and ms lines queued;
  - all against the same 30 VAL tasks.
- **Expected false hits:** under the null, a one-sided 2.5% upper bound below 1 gives about 0.9
  spurious `**` rows out of 35. Nine rows are flagged now, so most are probably real. Their
  **effect sizes**, however, are biased upward (winner's curse). This matters most for:
  - path TTA, chosen best of 20 on one checkpoint;
  - "rung-1 winning flags" merged into `train_v2.py` on single-seed evidence (`train_v2.py:2`).
    That contradicts R7 and team.md ("what gets merged").
- **What holds out, and what does not:** rung 2 re-uses the same VAL, so it controls training
  noise, not task selection. The only held-out evidence is TEST, once. No Holm or other
  correction is planned at screening, which is acceptable, but this is not documented anywhere.
- **The rung-4 entry gate is not a result.** "Rung 3 beats Timestral and Chronos-2 val numbers"
  (team.md:47) compares a VAL-tuned model with zero-shot baselines. It may gate compute. It must
  never be reported.
- **Fixes:**
  - Keep a running count of VAL configurations in `research_log.md`. Report it with the final
    claim, together with the Z4 VAL-to-TEST gap.
  - Pre-register exactly one primary candidate (audit_v2 §5.1).
  - Before freezing, report a leave-one-family-out selection check on VAL: the candidate's rank
    when each family is removed. A winner that relies on one or two families is flagged.
  - No VAL ratio against a baseline appears in any user-facing table except as "VAL (tuned on)".

**H-5. The baseline set is ambiguous. One candidate member is numerically broken.**

- **The two lists disagree:**
  - team.md defines success against Chronos-2, Timestral ×3, TiRex, **Chronos-Bolt, AutoETS,
    AutoTheta** and Seasonal Naive.
  - `compare.py:57` `FINAL_SET` (Gate G) omits Bolt, AutoETS and AutoTheta.
  - audit_v2 §5.1 requires further baselines to be "listed now or never".
- **AutoETS is degenerate on VAL:**
  - `wql_rel` is 2.68 (family 3.35);
  - `SZ_TAXI_1H` has `wql_rel` 2.2e10, from exploding intervals;
  - `M_DENSE_1H` is 246 and `SZ_TAXI_15T` is 16.6.
  - Bolt has no VAL score yet. The log ends in an MPS OOM, and the rerun is queued
    (`runs/bolt_after.sh`).
- **Failure scenario:** "beats every baseline" is computed on `FINAL_SET` but read against team.md's
  longer list. Or "beats AutoETS" is reported as a meaningful win.
- **Fixes:**
  - Commit the final B set in `RESEARCH_PLAN.md` and make `FINAL_SET` match it.
  - If AutoETS stays in the set, report it with a "diverges on 3 VAL tasks" note.
  - Never quote a geometric-mean ratio against AutoETS without the per-task table.

---

## 3. MEDIUM

**M-1. Gate G0 does not fingerprint the targets.**

- **Where:** `compare.py:304-315` `_check_parity` checks only `n_forecasts` and `horizon`.
- **Risk:** two runs with equal counts but different windows (a changed parquet, a subsample seed
  or cutoff change, or a `TSAR_TTA_MAXF`-style subset) pass G0.
- **Fix:** also require exact equality of `sn_wql` and `sn_mase` per task. They are deterministic
  functions of contexts and targets, and both sides' CSVs already carry them. Add the L6 input
  hash on TEST.

**M-2. `TSAR_RFL_CKPT` loads any checkpoint.**

- **Where:** `train_data.py:722-725`, `train_v2.py:1078-1081`.
- **Risk:** the comment says "TSAR_BENCH=1-trained only", but nothing checks it.
  - An old-suite reference (`q_*.pt` loads, because it has the same shape) decides which TRAIN
    rows are dropped. The selection would then be informed by a model that trained on
    `monash_covid_deaths`, `fred_md` and `exchange_rate`, which overlap VAL and TEST.
  - A reference picked as the VAL-best run makes the data selection VAL-informed. That is
    acceptable, but it must be listed under L7.
- **Fix:** require `ck["cfg"]["bench"] == 1` and the same `train_sha256` and `train_drop_sha256`
  as the current run. Store the reference's sha256 in `run_cfg`.

**M-3. synth4 is designed around TEST domains.**

- **Where:** `tools/synth4.py:17-33`, `:45`, `:59-63`.
- **What happened:** no data is read, but several families mirror named TEST or VAL families:
  - `bursty` is described as "observability-like", which matches boomlet and redset;
  - `epidemic` with an annual flu cycle matches ecdc_ili;
  - `solar` with fev's `m = 1` for daily data matches solar_with_weather;
  - `cumulative` matches uk_covid cumulative;
  - `TARGET` is the "fev-bench frequency mix".
  - The TEST family list is in `protocol.md`, so it was visible when synth4 was designed.
- **Effect:** if `TSAR_SYN4 > 0` is in the frozen config, the Z3 far-OOD subset no longer means
  "no analogue in training" for boomlet, redset, ecdc_ili and solar_with_weather.
- **Fix:**
  - Disclose under L-6 and L7.
  - Report Z3 twice: with and without the synth4-targeted families.
  - Freeze synth4 at its current hash. No synth4 or synth5 change may cite TEST families.
  - `tools/synth5.py` (new, untracked) should get the same review before use.

**M-4. Path-TTA selection evidence comes from a contaminated, single checkpoint.**

- **Where:** `runs/tta/sweep1.log`, which ran on `data/ckpt/big2.pt`. That is an old-suite model
  trained on corpora excluded from bench TRAIN.
- **The evidence is thin:**
  - 20 configs, one seed, a 5,383-forecast subsample;
  - the win comes from 8 long-horizon tasks only: W/L 8/0, CI_fam [0.976, 0.996].
- **Fix:** re-confirm `TSAR_TTA_LONG=paths` on bench-trained checkpoints (b_base256 and at least
  2 seeds) before it enters a frozen config. Counting it in H-4 is not enough.

**M-5. The TEST parquets are still in the working tree (audit_v2 P6, partial).**

- **Where:** `data/bench/fev/`, which holds all 48 TEST configs.
- **Risk:** `_guard_config` closes the bench path, but a direct `pd.read_parquet` (from a
  researcher script, a notebook or `tools/synth5.py`) still bypasses both guard and log.
- **Fix:** seal the files (move them out of the tree, or make them `chmod 000` with a
  sha-verified restore) until the confirmatory session.

**M-6. jobq re-scores trust the parent checkpoint blindly.**

- **Where:** `jobq.py:214-216`. A `run=<parent>` job does not delete the checkpoint, which is
  correct. But:
  - if the parent's `.pt` is missing or stale, the unguarded script **trains a new model** under
    the re-score's name and flags;
  - nothing checks that the checkpoint is the parent's finished one.
- **Fix:**
  - For `run != name`, require `data/ckpt/<run>.pt` to exist with `complete=True` and the
    parent's prov `ckpt_sha256`.
  - Pass `TSAR_EVAL_ONLY` to guarded scripts.

**M-7. Gate W3/A3 cannot run on TEST as specified.**

- **Where:** `train_v2.py:64-65`, `:95`, `:1429`. On TEST, any inference key that differs from
  the stored config is refused, including `TSAR_STATE_OFF=zero`.
- **Effect:** the A3 ablation in audit_v2 §5.3 W3 needs exactly that override.
- **Fix:** pre-register the A3 evaluation as its own allowed (checkpoint, override) pair with its
  sha256. Do not loosen the TEST rule generally.

---

## 4. LOW

- **L-1. Bolt `_mem_guard` (`bench.py:799-855`, `:866`).**
  - The inputs and windows are identical to every other model, because the wrapper slices the
    same copied contexts.
  - Chunk size and the CPU fallback for a single series (`:844`) change float paths only.
  - Before Bolt enters the set, verify on one h > 64 task that the batch-64 and batch-7 outputs
    agree to about 1e-4 relative. Log which series ran on CPU.
  - Bolt `predict_seconds` (nice 10, mixed device) cannot support gate E.
- **L-2. `train_flow.py` (uncommitted; audit pending its commit).**
  - Inference is clean (section 0).
  - Sampled paths draw `torch.randn` in batch order, so a series' forecast depends on its batch
    position. That will fail the R5 1e-3 VAL reproduction.
  - Fix: seed a per-task `torch.Generator` and draw noise per row.
  - The script is unguarded (B-1).
- **L-3. `results_bench.tsv` "keep".** `leaderboard.py` `status()` labels single-seed rows
  `keep`. Rename the label to `rung1-pass`, so that nobody reads it as a confirmed result.
- **L-4. n = 2 standard deviations.** The leaderboard shows `± sd` for 2 seeds (for example
  b_curr ± 0.0001). That is not an estimate of training noise. R3 requires at least 3 seeds for
  any reported spread.
- **L-5. Old-suite holdout in the unguarded copies.** `train_data.py:950`: with `TSAR_SPLIT=test`
  and without BENCH, the script also scores the 5 holdout datasets. These are no longer
  zero-shot (audit_v2 §8), but they should not be spent either. B-1's fix covers this.
- **L-6. Bit-identity at defaults is asserted, not verified.** It is verified only for
  `train_calib.py` (c_base equals b_base256). `train_v2`, `train_arch`, `train_ms`, `train_data`
  and `train_flow` claim it in their commits.
  - Fix: run a 30-step `smoke_param_sum` comparison against `train.py` per script (the lead
    queues it; this audit uses no compute).
  - Until then, a drift in a default path would show up as fake reference drift.

---

## 5. Status of the audit_v2 preconditions

| ID | Status |
|---|---|
| P3, P4, P7 (eval-only, config guard, ZS with BENCH) | Done in `train.py`, `train_fast.py` and `train_v2.py` **only**. Missing in 9 copies (B-1). |
| P5 (BENCH_EXTRA stamp) | Enforced in `bench.load_train`, so every copy inherits it. |
| P6 (seal TEST) | Partial (M-5). |
| P1, P2, P8 | Not re-audited here. `compare.py claim` has Gate G G0-G5, Holm and Bonferroni. Add the M-1 fix. |
| §5.1 pre-registration | Not yet written in `RESEARCH_PLAN.md`. The baseline set must be settled first (H-5). |

---

## 6. Claim checklist: all must hold before any number goes to the user as a result

1. **Provenance.**
   - Every number comes from runs with a clean `prov.json`: committed script, `git_dirty=false`,
     env recorded, `ckpt_sha256`.
   - Every baseline was scored at the single frozen commit.
   - Nothing comes from an UNVERIFIED leaderboard row (B-2).
2. **Script.**
   - The candidate was trained by a guarded script (`train_v2.py` or `train.py`).
   - Every flag in its env is read by that script, which refuses unknown `TSAR_*` keys (H-2).
   - Bit-identity at defaults has been verified (L-6).
3. **No transductive inference.**
   - `TSAR_TTA_STEPS = 0`.
   - A batch-invariance test passes for the frozen inference options (H-1).
4. **Reference and seeds.**
   - Rung decisions used a frozen, seed-matched reference (H-3).
   - The final candidate was trained on 5 fresh seeds {101…105} that were never used in
     development, with no seed dropped.
5. **Selection is accounted for.**
   - The number of VAL configurations screened is stated (about 60 today, plus the queues).
   - There is one pre-registered primary candidate.
   - `--candidates K` is used if more than one candidate carries a claim.
   - VAL numbers appear only as "VAL (tuned on)", never against baselines as a result (H-4).
6. **Baseline set.** It is committed in `RESEARCH_PLAN.md`, equals `FINAL_SET`, and includes the
   Bolt, AutoETS and AutoTheta decision. AutoETS divergence is disclosed if it is included (H-5).
7. **Gate G on TEST.**
   - It passes on `wql_rel` **and** `mase_rel`, for every B in the set.
   - It is run once by `compare.py claim --set final --min-seeds 5 --split test`, with G0
     including the `sn_wql`/`sn_mase` equality (M-1).
   - Its output is committed with `test_access.log`.
8. **TEST integrity.**
   - TEST was sealed until the session (M-5).
   - `test_access.log` has only the 6 known target-free entries before the session.
   - Each candidate checkpoint reproduces its recorded VAL score within 1e-3 in the same session.
   - TEST is reached only through `TSAR_EVAL_ONLY` + `TSAR_CKPT_SHA256` (B-1).
9. **Data provenance disclosed.**
   - TRAIN_DROP was derived from TEST contexts.
   - The synth3 and synth4 priors use benchmark conventions and TEST-domain-targeted families
     (M-3).
   - The `RFL_CKPT` reference is recorded and bench-trained (M-2).
   - The path-TTA selection was re-confirmed on bench checkpoints (M-4).
10. **Wording.**
    - "beats B" only per Gate G and Holm;
    - "every baseline" only as an intersection-union test over the committed set;
    - "world model" only if W1 to W3 pass, with A3 pre-registered (M-7);
    - failed gates are printed, not omitted.

---

# v3.1: verification of the fixes, `train_flow.py`, re-score jobs, `synth5.py` (2026-10-30)

**Scope.** I checked these commits: `32d4883` (B-1/H-1), `507b93d` (H-3/H-5), `0eedfd9` (H-4) and
`68132d2` (`train_flow.py`). I also checked `tools/synth5.py` (untracked). As before, this was a
static read only, with no compute.

## V. Verification of the fixes

| Item | Verdict | Evidence and residual |
|---|---|---|
| B-1 | **Fixed for bench.** | `bench._guard` (`bench.py:251-266`) refuses `test` unless `argv[0]` is `train.py`/`train_v2.py` (or `bench.py` for baselines), `TSAR_EVAL_ONLY` exists and its file sha256 equals `TSAR_CKPT_SHA256`. `tools/run.py:134-136` sets `argv[0]` to `train.py`, so the R5 command still works. `jobq.py` strips `TSAR_SPLIT/EVAL_ONLY/CKPT_SHA256/TTA_STEPS` from the shell, forces `TSAR_SPLIT=val`, and refuses non-val jobs. **Residuals (LOW):** (a) the basename check is bypassable on purpose (for example, by copying a script to `train.py`); it is a guard against accidents, not an adversary. (b) The old-suite holdout path in the unguarded copies (`split == "test" and not BENCH` → `P.evaluate(..., "holdout")`, for example `train_flow.py:1342`) goes through `prepare`, not `bench`, so it is still open (old L-5). (c) An unguarded script with `TSAR_SPLIT=test` outside jobq still trains its full budget before the guard fires. Compute is wasted, but TEST is not touched. (d) `tools/overlap_scan.py` on TEST is now refused (its `argv[0]` is `overlap_scan.py`). The L2 lag re-scan before the TEST session needs an explicit allowance for target-free scans. |
| H-1 | **Fixed.** | `_guard` refuses `TSAR_TTA_STEPS>0` on every split. It reads the environment at `evaluate` time, so it also catches the `train_tta` sweep's `os.environ.update` and the stored config that eval-only restores. The batch-invariance unit test was not added; see F-1, which is the same class of defect. |
| H-3 | **Fixed, with a minor residual.** | Seed k is paired with reference seed k (`leaderboard.py` build). Rung 1 is seed 0 against reference seed 0. A missing reference seed is filled by seed 0 and marked `0*`, which understates reference noise but is disclosed. The per-rung frozen reference list, and the `promote.py` refusal when `ref_logs` changes, were not done (LOW). |
| H-5 | **Fixed.** | `FINAL_SET` now holds all nine baselines (`compare.py:57-62`). A claim refuses to run (exit 2), or is FAIL by construction with `--allow-incomplete`, until `chronos_bolt_base` is scored. AutoETS divergent tasks and the median ratio are printed. |
| H-4 | **Disclosed** (`team.md`, "Multiplicity and selection disclosure"). | It covers the screen count, VAL as a selection set, TEST as the sole confirmation, and a rung-3 that uses only rung-2-confirmed flags. Leave-one-family-out selection is not mentioned; it is optional. |
| H-2 | Open (Demir). | `train_v2.py` still has no unknown-key refusal. |

## F. `train_flow.py` (commit `68132d2`)

**F-0. Target use: clean.**

- The flow target `self.post` is set in `FlowTransition.begin` only when `ctx` contains `"post"`
  **and** `self.training`.
- `forecast()` passes `ctx` only for `TRANS == "attn"`. For flow it is `None`, as the docstring
  says.
- `predict_chunk` calls `ema.eval()` first.
- The running `ms` buffers (σ_k) and the `DVFHead.ms` buffers are updated only inside the
  training losses, from TRAIN windows.
- At inference the transition uses only `h = GRU(u, z)`, `c = LN(h) + u`, the noise and the
  training-time σ.
- `DVFHead.sample` uses `q50` and `z0` from the context.
- No inference path reads the stop-gradient future encoder state.

**F-1 (MEDIUM). The sampled-path noise depends on batch position and task order. It is not batch-invariant.**

- **Where:** `train_flow.py:702` (`eps = torch.randn_like(h)` when `self.paths`) and `:732`
  (`DVFHead.noise`, `torch.randn`). Both draw from the global torch stream, which is seeded once
  before `EVALUATE` (`train_flow.py:1325`).
- **What a series' noise depends on:**
  - every forecast drawn before it in the run: all earlier tasks and earlier chunks;
  - its row position in the `PATH_ROWS = 128` chunk (`:120`, `:1154`);
  - the ensemble members;
  - the device generator (MPS or CPU).
- **So the same series gets a different forecast** if the task list or order changes
  (`TSAR_TTA_TASKS`), if forecasts are subsampled (`TSAR_TTA_MAXF`), if `TSAR_PATH_ROWS` or the
  ensemble changes, or if VAL and TEST are scored in one session (R5 scores VAL first).
- **It is not a leak.** The noise is independent of the data. But:
  - Monte-Carlo noise (S=16 paths) is confounded with the configuration. For example,
    `flow_s16_emp` vs `flow_s16_vinc` share noise only by coincidence of identical call order.
  - The R5 "re-score VAL within 1e-3" check is fragile.
- **Minimal fix:**
  1. Generate the noise per row on the CPU from a seed derived from that row's own input:
     `seed = SEED ^ crc32(context_after_crop.astype(float32).tobytes()) ^ hash(chunk_offset,
     path_index, ensemble_member)`.
  2. Precompute `eps[row, path, k, D]` (and `[row, path, H]` for DVF) with
     `np.random.default_rng(seed)`.
  3. Pass it into the rollout. `FlowTransition.forward` reads `self.eps[:, k]` instead of
     `randn_like` when `not self.training`.
  4. For later chunks the fed-back context already differs by path, so the hash separates paths.
  5. Add a unit test: forecast a 2-series batch in both orders and inside a 64-row batch. The
     outputs must be bit-identical per series on the CPU.
  6. Port the fix along with the flow code to `train_v2.py`: `train_flow.py` cannot score TEST
     under B-1.

**F-2 (LOW).** `train_flow.py` itself still has no eval-only or resume guard (the B-1 copies),
so any flow winner must be ported to `train_v2.py` before it is frozen (H-2 applies).

## R. Re-score jobs (`run=<other job>`)

**R-1. Verified: re-scores reuse the checkpoint without retraining.**

- `jobq.py:214-215` deletes the checkpoint only when `run == name`.
- In the unguarded scripts (for example `train_flow.py` resume, then `while … prog >= 1: break`),
  a finished checkpoint resumes at `step = STEPS` and goes straight to evaluation.
- Empirical check: the logs of `c_mix`, `c_scale`, `c_mixscale` and `c_carry` have **0** `step`
  lines and `num_steps 2500`. Their `training_seconds` equal the parent's (314.5; and 267.8 for
  `c_mix_s1` vs `c_base_s1`).
- The flow re-scores (`flow_s16_emp`, `flow_det_infer` → `flow_s16_vinc`; `dvf_med` → `dvf_s16`)
  follow the same path.

**R-2 (MEDIUM). A re-score rewrites the parent's checkpoint.**

- **Where:** `save_ckpt(training_seconds)` runs unconditionally after the loop (`train_flow.py`
  end of training; likewise in the other copies).
- **Effect:** `done` grows by the load time, so the parent's file bytes and sha change after every
  re-score.
  - The parent's `prov.json` `ckpt_sha256` goes stale.
  - In the guarded scripts, the checkpoint's stored config and `.val.json` would be rewritten with
    the re-score's environment and score.
  - That would break the R5 sha binding.
- **Fix:** skip `save_ckpt` when no step was taken in this session.

**R-3 (MEDIUM). `train_v2.py` re-scores are impossible under jobq.**

- Resume refuses any `INFER_KEYS` difference (`cfg_mismatch(ck["cfg"], CTRL_KEYS)`,
  `train_v2.py:1898`), and jobq now refuses any job with `TSAR_EVAL_ONLY`.
- So M-4 (re-confirm path TTA on bench checkpoints) and any rung-3 inference re-score on
  `train_v2.py` cannot run.
- **Fix:** let jobq accept `TSAR_EVAL_ONLY` when the split is `val`. Refuse only `TSAR_SPLIT != val`.

**R-4 (LOW).** A re-score still trains from scratch if the parent's `.pt` is missing (M-6). The
`after=` dependency checks only the parent's log.

## S. `tools/synth5.py` (Watanabe, untracked; used by `train_syn5.py` `TSAR_SYN5`)

**S-1. Clean of any fit to VAL or TEST data.**

- The module imports only numpy and opens no file.
- No benchmark name appears.
- Every parameter is a hand-set prior: Lorenz, pendulum, GARCH and Croston ranges, and calendar
  periods.
- `_check()` computes diagnostics (ACF, spectra, kurtosis) on generated rows only. It does not
  compare them with any benchmark statistic.

**S-2 (LOW, disclose).**

- The period weights favour sub-hourly data (`DAILY_W` puts 0.30 on 288). The docstring says this
  follows TRAIN gaps, but it also matches the VAL and TEST frequency mix.
- The `structural` family (traffic, load and sensor profiles with outages) is generic, not
  targeted at named families.
- List it under L-6/L7 with synth4. It is less TEST-targeted than synth4 (M-3).
- `train_syn5.py` carries the eval-only guards (`:52-60`), unlike the other copies.

## Updated claim checklist additions

- **11.** Stochastic inference (flow or DVF paths) is batch-invariant: noise is seeded per row from
  the row's own input, with a unit test (F-1).
- **12.** No re-score has rewritten a frozen candidate's checkpoint (R-2). The sha256 in
  `prov.json`, `.val.json` and the pre-registration must all agree.
