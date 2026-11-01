# Alma Models

Alma is a family of time-series **world models**: a latent state is advanced step by step by a learned transition, and forecasts are read out from the simulated states. Alma was built and evaluated on four markets under a pre-registered, leak-audited protocol, and is now running a fifth, live project.

Everything here was evaluated **once** on a sealed, out-of-sample TEST year, then independently re-derived by an auditor agent. Wins and losses are reported side by side, and nothing has been cherry-picked.

- **Best models with checkpoints and model cards:** [`winning_models/`](winning_models/)
- **Full per-project write-ups:** [`docs/`](docs/)

## Projects

| # | Project | Target | Sealed TEST window | Status |
|---|---|---|---|---|
| 1 | Oil | WTI price, 1/5/20-day probabilistic and direction | 2025-11-01 → 2026-10 | Done, audited ([audit](docs/audit_commodity.md)) |
| 2 | Power | German day-ahead electricity, hourly | 365 days from 2025-11-01 | Done, audited ([audit](docs/audit_power.md)) |
| 3 | Natural gas | Weekly EIA storage change (Lower-48 + 5 regions) | 52 weeks from 2025-11 | Done, audited ([audit](docs/audit_gas.md)) |
| 4 | Alma-Fin | Next-day realized-vol direction, 24 crypto perps; RL trading; ETF zero-shot | 2025-11-01 → 2026-10-30 | Done, audited ([audit](docs/audit_fin.md)) |
| 5 | Alma-Live Gas | UNG forecasting and paper trading, with continual learning | Forward test from launch | **In progress**: data, ledger and harness built; model search running ([docs](docs/live_gas_research.md)) |

## Audited headline results

| Project | What Alma wins (significant, Holm-corrected) | What it does not win |
|---|---|---|
| Oil | wm_sv ranks #1 of 9 on the 1-day probabilistic forecast, and beats ARIMA, Chronos-Bolt and Kronos | Does not beat the random walk or Chronos-2 (tie); no directional edge |
| Power | wm_power ranks #1 numerically on hourly direction (0.747), and beats the naive baselines, Kronos and zero-shot Alma | No significant win over LEAR, ARX or the foundation models; 80% intervals under-cover (0.75) |
| Gas | wm_gas beats **every** external model on quantile score, including OLS (Holm 0.030 / 0.039). It beats every clean foundation model on MAE: 11.5 Bcf vs Chronos-2-cov 19.8 (Holm 0.0016) | Not significant on MAE vs OLS (14.4, p 0.13); no accuracy win |
| Alma-Fin | wm_fin significantly beats Chronos-2, TiRex, Chronos-Bolt and Kronos on quantile score (Holm 0.023 / 0.002 / 0.0002). It transfers to unseen crypto coins: +8–9 pp over per-coin baselines, Holm < 1e-4 | Volatility direction 0.727 vs calendar regression 0.731 (tie); RL agent loses to DQN; no ETF transfer |
| **World-model test** (rollout vs matched direct twin) | | **Not passed in any project yet** |

## Integrity protocol

- **Pre-registration before any score:** each project's pre-registration (`docs/*_prereg.md`) and its amendments are dated and committed.
- **Pinned data and checkpoints:** sha256 pins in `data/**/manifest*.json` and in `runs/*/FROZEN.json`.
- **Leak tests:** information-set assertions, plus randomized-future perturbation tests. A model's output must not change when every future value is scrambled.
- **Strict TEST mode:** it refuses to run without a committed freeze, a clean tree and matching pins. The auditor tried to break it in a sandbox with 11 attack cases, and all were refused.
- **Red-team agents:** they log every finding (`docs/*_redteam.md`). Alma-Fin had 16 findings and Alma-Live 20; all HIGH findings were fixed and re-verified.
- **Independent re-derivation:** for example, 2,146 of 2,154 Alma-Fin TEST numbers matched exactly. The other 8 are p-value conventions on comparisons of identical models.
- **Full disclosure of trials:** every experiment counts. K = 985 evaluated rows for Alma-Fin, used in the deflated Sharpe.

## Repository layout

| Path | Contents |
|---|---|
| `winning_models/` | Frozen best model per project, with checkpoints, SHA256SUMS, FROZEN.json and a model card |
| `tools/<project>/` | Data fetchers, harness (`run_eval.py`), models, leak tests, search loops, red-team scripts |
| `runs/<project>/` | VAL/TEST outputs (`metrics.json`, `report.md`, `RUN.json`, `hashes.json`) and search logs (`results.tsv`) |
| `docs/` | Research briefs, pre-registrations, leakage designs, search logs, red-team logs, audits |
| `wm/`, `train*.py`, `prepare.py`, `program.md` | The general Alma world-model research loop, adapted from karpathy/autoresearch ([notes](docs/AUTORESEARCH_README.md)) |

## Data

Raw market data is **not** redistributed because of source licences (Binance, Yahoo, ENTSO-E and others). EIA and NOAA data are public. Rebuild the data with the fetch scripts in `tools/<project>/`. Every processed file has a sha256 pin, so a rebuild can be checked byte for byte.

## Hardware

Everything ran on one 8 GB Apple M3 MacBook.
