# Compute and memory (Hana Kim, 2026-10-29)

Machine: M3, 8 GB unified memory, ~4.1 GB free when idle (Claude, two browsers, a 1.2 GB
Virtualization VM and Notion are resident). Swap was 6.4-9.7 GB during these measurements.

## 1. Concurrency verdict: run one job at a time

Paired run at 23:22 (b_d_a6p_s1 + b_dyn_j2_s1, admitted with `--max 2 --reserve 1200`). Neither pair job
finished; the lead killed both at 23:29, so the numbers below are step rates taken from the logs.

| Job | Solo s/100 steps (rung-1 log) | Paired s/100 steps (steps 100-400) | Slowdown |
|---|---|---|---|
| b_d_a6p (train_data.py) | ~19 (476 s / 2500) | ~46 | 2.4x |
| b_dyn_j2 (train_dyn.py) | ~14 (356 s / 2500) | ~34 (lead measured 48 later) | 2.4-5x |

- Pair throughput is 1/2.4 + 1/2.4 ≈ 0.83 solo-equivalents at best, and ≈ 0.6 using the lead's
  later 5x number. Both are a net loss compared with running serially. The first 100 steps took
  113 s (solo: 25 s) while the jobs paged in.
- Cause: memory, not GPU. top showed each paired process at a footprint of about 2.6 GB, so two
  jobs need about 5.2 GB against 4.1 GB free. kern.memorystatus_level fell to 18-22 and swap
  reached 9.7 GB. The jobs file's `mem=800` / `mem=1600` and the logs' `peak_mem_mb`
  (`torch.mps.driver_allocated_memory()` at exit: MPS only, 577-1568 MB) undercount the real
  process footprint by 1-2 GB.
- **Three 0.37M jobs at once: no.** 3 x ~2.0-2.6 GB = 6-8 GB, against 4.1 GB free.
- **Recommendation: `--max 1` (now the default), `--reserve 1800`.** With `--max 1` the reserve
  only matters if someone raises `--max`. In that case, delete the `mem=` fields so that the
  measured footprint in `runs/<name>.time` drives admission. With real estimates (~2.5 GB x 1.2)
  and a 1.8 GB reserve, a second job is never admitted on this box, which is the right outcome.
- A second job could only pay off with about 3 GB more free memory (close the VM and browsers) and
  a measured solo footprint under 1.5 GB. Re-test then, using `.time` peaks.

### jobq changes (tools/jobq.py)
- The default is now `--max 1`. Admission **pauses** (one `PAUSE ...` / `RESUME` line) while swap
  used is over `--swap-max 6144` MB or `kern.memorystatus_level` is under `--min-level 20`. Running
  jobs are never killed.
- Every job runs as `uv run -- /usr/bin/time -l -o runs/<name>.time python <script>`, and the
  DONE/FAIL line reports its `peak` (peak memory footprint, which includes MPS and swapped pages)
  and its `rss`. A rerun uses 1.2x that footprint as its estimate when the job has no `mem=`.
- `--dry` prints the admission plan (`RUNNING` / `START` / `wait slot|memory|after`, or `PAUSED`)
  and exits.
- Caveat: the queue running now (started 23:29) predates the pause guard. The guard takes effect at
  the next relaunch. macOS swap shrinks slowly, so with swap at 8.7 GB the guard will hold
  admission until swap falls under 6 GB. If that stalls the queue, use `--swap-max 9000`
  (memorystatus_level is the more responsive signal).

## 2. Chronos-Bolt OOM (tools/bench.py)

Cause: for h > 64, Chronos-Bolt rolls out each of its 9 quantiles as a separate series
(`context.repeat(1, 9, 1)` in `ChronosBoltPipeline.predict`). baselines.py's `batch=64` therefore
becomes 576 rows x 2048 context (129 tokens) through T5-base. The failing 220 MiB allocation is
exactly one cross-attention value tensor: 576 x 129 x 768 x 4 B. The encoder FFN alone is
576 x 129 x 3072 x 4 B = 0.9 GB per layer, and the MPS cache was never emptied between tasks
of different shapes, so usage climbed to 8.09 GiB.

Fix (bench.py only; baselines.py untouched): `_mem_guard` wraps `chronos_bolt_*`:
- at most `TSAR_BOLT_BATCH` series per call (default 64), and `batch // 9` = 7 when h > 64 (63 rows);
- on MPS OOM it halves the chunk, and a single series that still OOMs runs on CPU;
- `gc.collect(); torch.mps.empty_cache()` after every task.

Predictions are unchanged up to float noise. Each series is independent: left NaN padding is
masked, Bolt's patches align from the right end, and the quantile reduction is per series.
Expected peak: 0.8 GB weights plus ~0.3 GB activations, so about 1.2 GB on MPS and ~2 GB for the
process. The watermark ratios below set a hard MPS cap of ~2.9 GiB (0.55 x the 5.33 GiB
recommended working set). Not smoke-tested, because memorystatus_level stayed under 25.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.55 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.45 TSAR_BOLT_BATCH=64 \
      nice -n 10 uv run -- /usr/bin/time -l -o runs/bench_chronos_bolt_base.time \
      python tools/bench.py baselines --split val chronos_bolt_base > runs/bench_chronos_bolt_base.log 2>&1

## 3. Training memory audit (train.py, read-only)

Where memory actually goes (0.37M params, B=128, CONTEXT=256, MAX_H=64, 40 patches):

| Consumer | Size | Notes |
|---|---|---|
| Train pool `by_dataset` (CPU numpy f32) | ~30-60 MB | From parquet metadata. It is not on the device; only the 128-row batch is copied over each step. `y[-20000:]` views keep full arrays of ercot/aus-electricity alive (~10 MB). |
| Training activations (MPS) | ~150-250 MB | Encoder ~70 MB (2 layers, B·T·D = 0.49 M floats); rollout over kept origins ~100 MB. |
| NaN padding (~47% of rows/positions) | ~35 MB of encoder activations | The rollout already drops padded origins (`[keep]`). The encoder must keep pad tokens: real tokens attend to them under the causal mask, so dropping them changes results. **Not worth changing.** |
| Eval (`predict_chunk`, 512 series per chunk, no_grad) | ~50-120 MB transient | The eval batch size is not the problem. |
| MPS allocator cache | **0.4-1.3 GB** | `driver_allocated_memory` at exit is 577-1568 MB against a working set under 300 MB. `[keep]` gives a different row count every step, and every eval task has its own (B, h), so the cache fills with blocks of sizes it cannot reuse. No `empty_cache` anywhere. |
| torch + MPS runtime, pandas, val tasks | ~0.8-1.0 GB | Fixed for each process. This is why running two processes costs twice. |

Proposed changes, in order of MB saved. All of them are bit-identical, because none changes any
arithmetic:

1. **`torch.mps.empty_cache()` after each eval task and once before the final eval** (in `predict`,
   or by wrapping `EVALUATE`). It releases the training-time cache before the eval's
   variable-shape allocations pile on top. Expected saving: **~0.4-1.2 GB** of peak MPS
   (577-1568 MB at exit down to ~300 MB).
2. **Cap the allocator with env, with no code change:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25
   PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15` (~1.3 GiB hard cap, GC from ~0.8 GiB). The MPS
   allocator frees cached blocks before it raises an OOM, so results do not change. jobq could set
   this for every job. Expected saving: **~0.3-0.8 GB** of peak footprint during training,
   together with (1). Risk: a run whose real working set exceeds 1.3 GiB (big models at rung 3/4)
   would OOM, so raise the ratio for those.
3. **Periodic `torch.mps.empty_cache()` every 250 training steps** (cost ~1 ms each). It stops the
   cache built by variable-size `[keep]` indexing from growing across the run. Expected saving:
   **~0.2-0.5 GB**. It overlaps with (2); pick (2) if env-only is preferred.

Not recommended: smaller eval chunks (saves under 100 MB), stripping padding in the encoder
(changes results), moving the train pool to the device (it is CPU and small already). The real
lever for throughput is serial jobs plus (1)-(2), which should bring one job to about 1.5 GB.
Check that against the first `.time` peaks.

## 4. Architecture search throughput (2026-10-30, tools/search.py)

Measured with smoke runs of `train_search.py` (base recipe + TSAR_SYN5=1, TSAR_DEV=1) while the wave3 MPS queue
ran (syn5_only_s2, then ctx512). Swap was ~9.9 GB, memorystatus_level 34-43 at the start.

| Setup | s/step | Footprint (time -l) | Effect on the running MPS job |
|---|---|---|---|
| MPS solo, ctx 256 (syn5_only_s1 log) | 0.154 | 1.5-2.0 GB | - |
| MPS solo, ctx 512 b128 (ctx512 log) | 0.32 | - | - |
| 1 CPU screen beside MPS, ctx 256 | 0.35 | 2.9 GB peak, 1.1-1.4 GB RSS | 17.5 -> 18-24 s/100 steps |
| 2 CPU screens (4 threads each) beside MPS | 0.54-0.60 each (1.6x) | 2.9 GB each | 25-31 s/100 steps; level fell to 24 |
| CPU, ctx 2048 b64 (attn, HDEC) | 3.9 | - | - |

- DEV eval (9 tasks, tools/devsplit.py): 2-5 s on CPU at ctx 256 (VAL: ~22 s). At ctx 2048 it took ~50 s on CPU.
- Startup + data load ~12-16 s per job, so a 400-step incumbent screen is ~1.5 min on MPS (~40/h); the sampled
  space averages ~3.4 min (~18/h) because long contexts and wide models cost 2-6x per step.
- Verdict: screens run serially on MPS through one jobq runner (`--max 1`). A second runner on CPU adds ~20%
  screens at best, slows the MPS job 1.3-1.8x, and two runners don't see each other's slots (`--max` is per
  runner; only memorystatus_level couples them). With swap at ~10 GB it is not safe: 2.9 GB CPU + ~2.5 GB MPS
  against ~4 GB free. Only reconsider below 6 GB swap and only for ctx <= 512 screens.
