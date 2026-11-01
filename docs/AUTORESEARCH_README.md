# ts-autoresearch

An autonomous research loop for building a **time-series world model** that beats strong
industry forecasters on accuracy, calibration and compute. It is adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) to run on an Apple-silicon
laptop.

An AI agent such as Claude Code edits `train.py`, trains for 4 minutes, scores the model on a
fixed benchmark, keeps the change if it helps, reverts it if not, and repeats. That gives
about 10 experiments an hour, or about 100 overnight.

- `MISSION.md`: the research mission (goals, tracks, success standard).
- `RESEARCH_PLAN.md`: benchmark design, pre-registered success gates, leakage checklist.
- `program.md`: instructions for the agent running the loop.

## Files

| File | Role | Who edits it |
|---|---|---|
| `prepare.py` | Downloads 20 benchmark datasets, defines the splits and the metric | Nobody during a run |
| `baselines.py` | Industry baselines (statistical and foundation models) on the same windows | Human |
| `train.py` | The world-model forecaster: encoder, latent dynamics, quantile decoder | The agent |
| `program.md` | The agent's instructions | Human |
| `vendor/autoresearch` | Karpathy's original, for reference | Nobody |

## Quick start

```bash
uv sync --group baselines        # Python 3.12 environment
uv run prepare.py                # one-time download and split (~250 MB, ~1 min)
uv run baselines.py              # score the baselines on the val split
uv run train.py                  # one 4-minute experiment
```

Then start the loop. Open Claude Code in this folder and say:

```
Read program.md and let's kick off a new experiment run.
```

## The benchmark in one paragraph

20 public datasets from the Chronos benchmark collection, from 30-minute energy data to
quarterly macro series. 15 are used for development; 5 (FX, FRED-MD macro, Australian
electricity, Dominick retail, car-parts intermittent demand) are held out entirely to test
generalization. Every series is split chronologically into train / val / test windows. Models
are scored by weighted quantile loss (≈ CRPS), MASE and 80% interval coverage, each relative
to Seasonal Naive (lower is better, below 1.0 beats it). The test and holdout splits only run
with `TSAR_SPLIT=test` and are reserved for confirmatory runs.
