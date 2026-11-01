# Reproducibility (audit, 2026-11-01)

What was checked by re-running training from the recipes on a different Mac than the one that produced the frozen
checkpoints (same torch 2.15.0 / numpy / pandas as `uv.lock`):

| Project | Re-train vs frozen checkpoint | Same machine, run twice |
|---|---|---|
| Oil (`wm_sv`, 1 thread) | weights equal to 7e-7 | identical |
| Gas (`wm_gas`, 2 threads) | weights differ up to 0.11, forecasts corr 0.99996 (mean 0.7 Bcf) | identical (max diff 0.0) |
| Gas DEV backtest | forecasts corr 0.99997 (mean 0.6 Bcf) | identical |

Multi-threaded CPU training is deterministic on one machine but not across machines: a different core layout
changes the order of floating-point sums, and 1,200 steps amplify the last-bit differences. Single-threaded training
(oil) reproduced across machines.

**Rule from now on:** to make a run that anyone can reproduce bit-for-bit, train with `ALMA_THREADS=1`
(read by `tools/{gas,power}/search/run.py` and `tools/fin/search/{loop,train_wm,finalize,train_agents}.py`; it sets
one thread and `torch.use_deterministic_algorithms(True)`). Without it the previous thread counts are kept, so every
existing results row stays reproducible on its original hardware. The research track (`train_wm.py`) trains on MPS,
which is deterministic only on the same machine and macOS version.
