# Alma-Live Gas autoresearch program

**The one question.** Does the gas world model beat (a) its matched direct twin and (b) the strongest
feature-matched baselines on the forecasting targets? The baselines are ridge-penalised linear quantile regression
and LightGBM quantile regression on identical inputs, plus the prior frozen wm_gas on storage. The three targets are:
- next-day UNG return QS (r1);
- weekly storage deviation vs the 5-yr average (dsurp);
- 5-day realized vol (lrv5).

## Setup (fixed; never edited by the loop)

- **Data:** `tools/live/wm_store.py` builds `runs/live/search/cache/panel_store.npz` from the data lead's
  point-in-time records. Inputs at 15:30 ET of day d satisfy published_at <= t. Only core data is used: UNG closes
  through d-1, first-print storage, and CPC observed and forecast degree days plus calendar. COT and NG=F are excluded.
- **Harness:** `tools/live/search/run.py` does the following:
  - trains on 4 expanding DEV folds (test years Jul 2020..Jun 2024, train strictly before, embargoed on label dates);
  - runs the randomized-future leak test on every checkpoint;
  - scores the folds;
  - appends a row to `runs/live/search/results.tsv`.
- **Metric (one scalar, higher is better):** `skill = 1 - S_F`, where S_F is the mean over the 4 folds and the 3
  targets {r1, lrv5, dsurp} of QS(model) / QS(ridge-QR). QS is the mean pinball over levels 0.05..0.95. In plain
  terms, skill is the mean of the three skill scores vs the feature-matched ridge.

## The editable file

`tools/live/wm_core.py` (model + training) is the only file the loop edits. `DEFAULT_CFG` at its top is the
recipe. Architecture, losses, transition, heads and optimiser may all change. The data, the harness, the metric and
the leak test may not.

## Budget

Each experiment gets a fixed wall-clock budget of `budget_s` = 45 s per fold. With 4 folds, eval and the leak test,
that is about 4 minutes. The direct twin gets the same per-fold wall-clock budget, with parameters matched to
+-5% (`wm_core.matched_cfg`). Single seed (0) during the loop.

## Loop

1. Read results.tsv and this file, then write a one-line hypothesis.
2. Edit wm_core.py, then `git commit` with the hypothesis.
3. Run `nice -n 10 .venv/bin/python tools/live/search/run.py exp "<hypothesis>"`, then grep `METRIC`.
4. If skill improves on the incumbent by > 0.002, keep it. Then immediately run
   `run.py twin "<commit>"` so the WM-vs-twin gap is tracked for every incumbent.
5. Otherwise, run `git revert --no-edit HEAD`. The row stays in results.tsv.
6. Repeat until the clock stop in docs/live_search.md.

Every row (exp, twin, crash, leakfail) counts in K for the deflated Sharpe.

## Paused until the WM shows a DEV forecasting edge

The following are paused:
- Dreamer and imagination-PPO actors;
- raw PPO/DQN;
- the n-order equity graph as a timing input;
- the continual-rule search.

The only policy is the Kelly vol-target champion, a deterministic function of the forecast distribution. One
continual-vs-frozen check is run at freeze time.
