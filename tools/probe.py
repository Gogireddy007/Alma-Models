"""Probe a saved checkpoint on hand-made series (ramp, exponential, seasonal+trend, step).
Usage: TSAR_RUN=<ckpt> TSAR_MINUTES=0 [model env...] uv run tools/probe.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
src = open("train.py").read().split("# Evaluation")[0]
exec(compile(src, "train.py", "exec"))
t = np.arange(200, dtype=np.float32)
cases = {
    "ramp": 100 + 2 * t,
    "exp": 10 * np.exp(t / 60),
    "cumcount": np.cumsum(np.random.default_rng(0).poisson(5 + t / 20)).astype(np.float32),
    "trend+season12": 500 + 3 * t + 80 * np.sin(2 * np.pi * t / 12),
    "flat_noise": 50 + np.random.default_rng(1).normal(0, 2, 200).astype(np.float32),
}
h = 24
for name, y in cases.items():
    q = predict([y], h, 1, "D")[0]                       # [9, h]
    tt = np.arange(200, 200 + h)
    truth = {"ramp": 100 + 2 * tt, "exp": 10 * np.exp(tt / 60), "trend+season12": 500 + 3 * tt + 80 * np.sin(2 * np.pi * tt / 12),
             "flat_noise": np.full(h, 50.0)}.get(name)
    med = q[4]
    print(f"{name:15s} last={y[-1]:9.1f}  med@1={med[0]:9.1f} med@12={med[11]:9.1f} med@24={med[23]:9.1f}"
          + (f" | truth@1={truth[0]:9.1f} @12={truth[11]:9.1f} @24={truth[23]:9.1f}" if truth is not None else
             f" | linear-extrap@24={y[-1] + 24 * (y[-1] - y[-25]) / 24:9.1f}"))
