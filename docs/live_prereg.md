# Alma-Live Gas: forward paper-test pre-registration (DRAFT for manager sign-off)

Data lead, 2026-11-01. This draft is based on the manager's decisions after `docs/live_gas_research.md` (11f197b).
**Paper only; no real orders.** The forward test is the only clean claim test: history ≤ 2025-10-31 is development,
and 2025-11-01 onward is EXPOSED (`docs/live_leakage.md`).

## 1. What is frozen before launch

- **Launch day** L: the first trading day after the manager's GO, expected 2026-11-02 or later.
- Before L, `runs/live/FROZEN.json` (architect) is committed. It names:
  - the model, its checkpoint sha256 and the code commit;
  - its online-update rule (`observe`), which may keep learning but only through `observe(outcome)` and only on
    outcomes published at or before the next decision.
- Data pipeline commit and `data/live/manifest.json` pins as of L. Source rules can only change by appending
  versioned rules, logged in `docs/live_leakage.md`.
- Clock: decide at 15:30 ET with `ctx = Store().asof(decision_time, live=True)`; fill at the 16:00 ET close of d.
- Costs: 10 bp per side; every P&L is also reported at 2x and 3x.
- Comparators, all running as shadow books from L in the same ledger format:
  - rule baselines: buy_hold, short_ung_carry, tsmom_20, seasonal, weather_rev, storage_surprise;
  - forecast baselines: fc_hist (climatology) and fc_ewma (EWMA Gaussian / HAR-like RV / damped persistence);
  - the frozen prior wm_gas, if the architect provides its adapter before L.

## 2. Primary endpoints (forecasting; manager decision 8)

Quantile levels are QS = {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}. The quantile score is the mean pinball loss over
the levels, which is proportional to a discretised CRPS.

| id | target | origin | primary metric | comparator | test |
|---|---|---|---|---|---|
| P1 | Storage deviation dev(W) = first-print printed net change(W) - naive 5-yr same-week mean (L48, Bcf) | last decision (15:30 ET) before W's first print (normally Wednesday) | QS (Bcf) | fc_ewma (damped persistence + empirical errors) and fc_hist | one-sided paired test on per-week QS differences (HAC/DM-HLN), α = 0.05 |
| P2 | UNG log return close(d) -> close(d+1) | 15:30 ET daily | QS | fc_ewma (Gaussian random walk, EWMA vol) | DM-HLN on daily QS differences, one-sided, α = 0.05 |
| P3 | log RV5 of UNG (sum of squared close-to-close log returns d+1..d+5) | 15:30 ET daily (overlapping) | QS on log RV5 | fc_ewma (HAR-like 20-day RV + empirical errors) | DM-HLN with HAC lag 5, one-sided, α = 0.05 |

- The three primary tests are Holm-adjusted (family of 3).
- Secondary forecast metrics: 90% interval coverage; P1 median-sign accuracy (Wilson CI); P2 direction as a
  **negative control** (it is expected to be about 50%).

## 3. Secondary endpoint (trading P&L)

- **Book:** UNG; the weight is the model's `act(ctx)`.
- **Metrics:**
  - net Sharpe at 1x, 2x and 3x costs;
  - PSR(0);
  - DSR, with k = every strategy run in the shadow loop, all counted;
  - max drawdown and turnover.
- **Readouts:** at 12 and 24 months, plus monthly descriptive reports (no stopping on performance; see section 5).
- Paired Sharpe difference vs buy_hold and vs tsmom_20, using a block bootstrap (7-day blocks).

## 4. Duration and power (honest minimum detectable effects)

- **Duration:** 12 months is the first formal readout; 24 months is the final. The test runs the full term unless a
  kill criterion (section 5) fires.
- **Trading** (iid approximation; t ≈ SR_annual × √years; one-sided α = 0.05):

| horizon | MDE Sharpe, 50% power | MDE Sharpe, 80% power |
|---|---|---|
| 12 months (252 days) | 1.65 | 2.49 |
| 24 months | 1.16 | 1.76 |

  Realistic net Sharpe for a single-market daily strategy is 0.5-1.0. **The paper test cannot establish a trading
  edge statistically at either horizon.** P&L is reported, never claimed, unless it clears these bars.

- **P2 direction (negative control):** n = 252, p0 = 0.5. The 80%-power MDE is 0.5 + 2.49 × 0.5/√252 = **57.8%**.
- **P1 storage median sign:** n = 52 at 12 months. The 80%-power MDE is 0.5 + 2.49 × 0.5/√52 = **67.3%** (n = 104
  at 24 months: 62.2%).
- **Forecast QS endpoints:** MDE = 2.49 × sd(per-period QS difference) / √n_eff, expressed as relative QS skill. The
  sd is taken from DEV (`runs/live/eval/dev/forecast_rows.parquet`, fc_ewma vs fc_hist). n_eff = n for P1/P2 and
  n/5 for overlapping P3. See the table in section 4a; those numbers are fixed at launch.

### 4a. Forecast-endpoint MDE (from DEV loss-difference variability)

The minimum detectable **relative QS improvement** over the comparator is computed with one-sided α = 0.05. It uses
the per-period QS difference sd measured on DEV for fc_ewma vs fc_hist (2017-01..2024-06; n = 1,879 days and 390
weeks), so MDE = z × sd / √n_eff / QS(comparator):

| endpoint | DEV comparator QS | sd of per-period QS difference | n_eff at 12 m | MDE 80% power, 12 m | MDE 50%, 12 m | MDE 80%, 24 m |
|---|---|---|---|---|---|---|
| P1 storage dev (Bcf) | 10.02 | 7.57 | 52 | **26%** | 17% | 18% |
| P2 UNG next-day return | 0.00721 | 0.00184 | 252 | **4.0%** | 2.6% | 2.8% |
| P3 log RV5 (overlapping, n_eff = n/5) | 0.187 | 0.160 | 50 | **30%** | 20% | 21% |

Caveats:
- A new model's loss differences can be more or less variable than this baseline pair. The MDE is recomputed on the
  frozen model's VAL rows before launch and logged here; the larger of the two values governs.
- For reference, on DEV fc_ewma beats fc_hist by 3.2% on P2 and 18.9% on P3 (both p < 0.001), and by 0.6% on P1
  (p = 0.89).
- A WM therefore needs about +4% QS on returns, which is plausible only through the volatility forecast. It needs
  about +26% on storage QS vs damped persistence, a large but realistic weather-driven gain given that the
  prior-project storage results were detectable at n = 52. And it needs about +30% on RV5, against a strong HAR-like
  baseline.

## 5. Decision rule and kill criteria

- **Claim rule:** a primary endpoint is "supported" only if its Holm-adjusted one-sided p < 0.05 at the 12- or
  24-month readout vs **both** forecast baselines, with the same sign at both readouts if both are reached. Anything
  else is reported as "not supported", including the effect size and CI.
- **Real-money discussion:** only after (a) at least one primary endpoint is supported, (b) live-vs-backtest parity
  of the paper book (same ctx hashes and decisions when replayed), and (c) a P&L readout clearing the section 4 bar.
  Even then, the decision is the user's.
- **Kill / halt criteria** (operational, not performance fishing):
  1. A ledger fails `verify` or its git anchor: the book is void from that point and the test restarts.
  2. Leakage is found (any red-team or test_infoset failure on production code): stop, fix, restart the clock.
  3. Drawdown ≥ 10%: `halt_review`, flat until a human review note is appended; the time stays in the record.
  4. Daily loss ≥ 2%: the next decision is flat (kill switch).
  5. Stale data: forced flat for that day (counted as flat, not as missing).
  6. More than 10% of decision days missed in any month: report as an operational failure; the readout is delayed
     by the missed days.
- **No early stopping for success.** Monthly reports are descriptive. Interim looks do not change the claim rule;
  they do count toward the DSR k.

## 6. Data and ledger integrity

- Every decision line records the ctx input sha256, the data manifest sha256, the model meta, git HEAD and the
  target.
- The ledger is hash-chained, and git-committed after every decide and settle.
- Replay: for any day, `Store().asof(decision_time, live=True)` built from the committed PIT files must reproduce
  the logged input sha256.
