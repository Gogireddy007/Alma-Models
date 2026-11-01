# Alma-Fin research brief: where a continual, portable world-model RL trader has a real chance

Author: deep-research lead agent, 2026-10-31. Branch `alma/investor-sprint` (local only). Status: v1, sections 1–4 are the
build spec. No training was run, no data was downloaded, and no value after 2025-10-31 was inspected. Every external claim
carries a source tag; the list is in §9. Our own lessons come from `docs/audit_commodity.md`, `docs/audit_power.md` and
`docs/gas_research.md`.

## 1. Decision (read this first)

| item | choice |
|---|---|
| Fragment | **Binance USDT-M perpetual futures, liquid large caps.** The universe is the 24 perps ranked highest by 30-day quote volume on 2025-10-31, each listed on or before 2024-11-01. The universe is fixed once, using only data up to 2025-10-31. |
| Horizon | **1 day.** The decision is made at 00:00 UTC, the fill is at the 00:05–00:10 UTC 5-min bar, and the position is held to the next decision. Every daily sample is non-overlapping. |
| Primary target (P1) | **Next-day realized-vol direction:** `y = 1[log RV(t+1) > mean_{k=0..6} log RV(t-k)]`, where RV is the sum of squared 5-min log returns over 00:05–24:00 UTC. |
| Trading claim (P2) | Net-of-cost daily P&L of the world-model RL policy on the **held-out coins**, measured by Sharpe, PSR, DSR, turnover and max drawdown. It is co-primary but **expected to be under-powered** (see §4.3). |
| Secondary | Log-score / CRPS of the day-ahead RV distribution (P3). Next-day return direction on non-overlapping days (P4, reported, with no claim expected). |
| Transfer | Train on 16 coins. Test zero-shot on 8 held-out coins (every third coin by volume rank) and on 13 US ETFs (SPY, QQQ, the 11 SPDR sector ETFs) at daily frequency with a range-based RV. |
| Splits | **Keep** DEV ≤ 2025-06-30, VAL 2025-07-01..2025-10-31, TEST 2025-11-01..2026-10-30, sealed (see §6). |

Why this fragment. (a) Our own audits found that oil price direction at 1–20 days was noise while volatility was
predictable (`docs/audit_commodity.md`). In power, the WM was top-tier but never significantly beat its direct twin
(`docs/audit_power.md`). Volatility is the one target where structure is documented across all asset classes [HAR, AC24, BTCHAR].
(b) Intraday crypto return signals exist, but they are smaller than fees. The quarter-hour order-imbalance effect on Binance
perps is about **0.51 bp gross per trade against a 5 bp taker / 2 bp maker fee** [QH]. That rules out 5-min/1-h *return*
trading for us. (c) Daily bars make crypto→ETF transfer a like-for-like test. (d) Binance publishes free, reproducible
klines with taker-buy volume and funding rates [BPD]. (e) The RL policy can still earn its keep through **risk sizing**,
where vol forecasts matter, rather than through return alpha. Vol-managed portfolios are the documented channel [MM17].

## 2. What is public about EquiLibre Technologies (claims, not facts)

- It is a Prague startup founded by ex-DeepMind researchers Martin Schmid, Rudolf Kadlec and Matej Moravčík, who worked on
  DeepStack poker. It has about 25 staff, and Rich Sutton is an adviser [TC26].
- It has an exclusive arrangement to supply algorithms to Tower Research Capital. Bloomberg reported that the deal was
  agreed in late 2023 and that Tower's CIO confirmed it [BBG24].
- TechCrunch reports that it "first deployed on crypto markets in 2025", that it then moved to equities, and that its
  algorithms trade "billions in daily volume" across S&P 500/Nasdaq names. It reports a Creandum-led Series A at about
  $500M, size undisclosed [TC26, REC26].
- The company claims "zero negative months since inception". **The claim is unaudited, and no returns, Sharpe, capacity or
  method details are public** [TC26]. Nothing public shows that it uses world models, continual learning or cross-asset
  transfer. "RL trading" is the only stated method. Lesson for us: a credible lab says nothing verifiable, so our edge with
  investors is **audited, pre-registered numbers**, not claims.

## 3. Evidence base per pillar

| pillar in finance | out-of-sample, cost-aware evidence | verdict |
|---|---|---|
| World-model RL (Dreamer [DV3], MuZero [MZ]) | We found **no peer-reviewed Dreamer/MuZero trading study** with net-of-cost OOS results. Trading discussions are secondary and speculative [DTB]. | Hype, so far. This is the novelty, but it must win via its twins. |
| K-line foundation models (Kronos) | Kronos was pretrained on 12B K-lines from 45 exchanges, **up to June 2024**, crypto included. Its test periods start July 2024. It reports RankIC and vol-MAE gains, with no net trading P&L [KR]. | Plausible as a feature source/baseline. Clean for our TEST. |
| Generic TSFMs on returns | Zero-shot Chronos/TimesFM give OOS R² of -1.4% / -2.8% and about 51% direction on daily returns across 94 countries [RN25]. They do better on vol/spreads [MAR25]. | Returns: no. Vol: yes. |
| Model-free deep RL trading | Many crypto DRL papers report OOS gains, often with short windows or costs only at test time. Surveys flag overfitting [ENS23, DRLS]. **We found no study applying DSR/PBO** to DRL crypto results (search gap). | Mostly hype after costs. Needs DSR. |
| Continual / online learning | FSNet → OneNet cut online error under drift by about 50% vs SOTA on generic benchmarks [ONE]. HAR is "HARd to beat" partly *because* of rolling re-estimation [HRW]. | Real, but a rolling-window baseline captures most of it, so the twin must be rolling HAR/LGBM. |
| Cross-asset portability | A universal LOB model trained on pooled stocks works on unseen stocks [SC19]. Pooled crypto models beat per-pair models [CRYU]. Kronos zero-shot across classes [KR]. | **Best-supported pillar.** No asset-ID embeddings. |
| Momentum / vol timing (baselines) | TSMOM works across 58 futures [MOP12]. Crypto momentum is significant but in-sample [LT21]. Vol-managed portfolios raise Sharpe in equities [MM17]. Crypto evidence is anecdotal [VMC]. | Strong, simple baselines to beat. |

Combinations with OOS, cost-aware evidence: **pooled (portable) vol forecasting + rolling re-estimation + vol-scaled
positions** (pieces from [HRW, SC19, MM17, MOP12]). Hype: WM imagination-trained RL alpha, and TSFM return direction.
We pitch the first and *test* the second honestly.

## 4. Candidate fragments, feasibility and metrics

### 4.1 Candidates

| candidate | documented predictability | typical OOS after costs | free data on 8 GB Mac | transfer test |
|---|---|---|---|---|
| Crypto perps 5-min/1-h returns | Real but tiny: 0.51 bp/trade gross [QH]. OFI features [XPC]. | Negative at 5 bp taker [QH]. | Yes [BPD], but large files. | coin→coin |
| **Crypto daily RV direction** (chosen) | High persistence, HAR-type models [BTCHAR, HAR]. | Not a P&L target. Feeds sizing. | Yes, 1-h/5-min klines, funding [BPD]. | coin→held-out coin→ETF |
| Crypto daily return/TSMOM | Momentum at daily/weekly [LT21]. | Thin after ~7–10 bp per turnover. | Yes | yes |
| US equity/sector ETFs daily | TSMOM [MOP12]. Returns about 51% direction [RN25]. | Small. One year has ≈250 obs. | Daily OHLC (Stooq/Yahoo) | transfer target only |
| Intraday equities | Microstructure is universal [SC19]. | — | **No free reproducible data.** | — |
| FX | The random walk is hard to beat at short horizons [MR83]. | ≈0 | Partial | weak |

### 4.2 Where RL "alpha" vanishes after costs

Costs are 5 bp taker fee [QH] plus 2 bp slippage on majors (5 bp on the rest) per side, plus funding every 8 h [BPD].
A policy that turns over its book once a day pays **≈7–10 bp/day ≈ 25–36%/yr**. Any 1-h or 5-min return policy dies
here [QH]. Daily return signals with about 51–53% hit rates [RN25] also die unless turnover is below about 0.2/day. The P2
policy must therefore act on **position size** (vol targeting, de-risking) with a turnover penalty in the reward. That is
where gains survive in theory [MM17].

### 4.3 Metrics that are hard to fake, and the minimum detectable edge on TEST

- P1: hit rate on non-overlapping days, pooled over coins, with a paired test vs each baseline. For inference we use a
  **day-level aggregate** (one number per day), a McNemar test and a 7-day block bootstrap, as in power
  (`docs/audit_power.md` §12).
- P2: net Sharpe, **PSR** (single pre-registered strategy) and **DSR** (deflated by the number K of recipes tried in
  the search) [DSR, PSR]. Also turnover, max drawdown and Calmar.
- P3: log score / CRPS of the RV distribution, with a DM-HLN test.

**Effective sample.** TEST has ≈364 days. Crypto vol is factor-driven. If the correlation of per-coin labels is
ρ≈0.6 (an assumption; to be measured on VAL), the effective N per day is n/(1+(n−1)ρ) = 1.6 for 24 coins and 1.5 for
8 coins. That gives **N_eff ≈ 560–590 coin-days**.
- Hit rate vs 50%, α = 0.05 two-sided, 80% power: MDE ≈ 2.8·0.5/√560 ≈ **±5.9 pp**.
- Paired vs the best baseline with a 25% discordance rate: MDE ≈ 2.8·√(0.25/560) ≈ **5.9 pp**. Typical ML-over-HAR gains are
  1–3 pp [AC24, HRW]. **A significant win over HAR on P1 is possible but not likely. Plan the pitch around that.**
- P2 Sharpe from one year: SE(annualised SR) ≈ √(1+SR²/2) ≈ 1.0–1.2, so the MDE vs 0 is ≈ **2.5 annualised Sharpe**.
  Vs a strategy with 0.7 correlation, SE ≈ 0.77, so the MDE is ≈ 1.9. **No honest one-year test can confirm an ordinary
  trading edge.** Report PSR/DSR with CIs and make no "significant alpha" claim unless the numbers clear these bars.
- ETF transfer arm: about 250 days × 13 ETFs, ρ≈0.7, so N_eff ≈ 350 and MDE ≈ 7.5 pp. It is descriptive unless strong.

### 4.4 Continual-learning protocol (no leakage)

1. **Freeze before TEST.** Freeze the architecture, the online rule (optimizer, lr, steps/day, replay window, anchor
   penalty) and every hyperparameter on VAL. Commit FROZEN.json with hashes and the search count K, as for power.
2. **Clock.** Day t's label RV(t) is complete at 24:00 UTC. At decision time 00:00 of day t+1, features use bars closing
   ≤ 23:55 of day t. **Fill** at the 00:05–00:10 bar's (O+H+L+C)/4, so there is no same-bar execution. The funding settled
   at 00:00 is applied to the position held at 00:00 (the old position).
3. **Online update.** After the day t label matures and before the day t+1 decision, take exactly S gradient steps on a
   replay buffer of days ≤ t (last 90 days + 10% reservoir from DEV). L2-to-anchor (EWC-lite) toward the frozen
   weights. Assert in code: `max(train_label_time) < decision_time`.
4. **Perturbation test** (as in `tools/power/test_infoset.py`). Multiply every value after a decision time by U(0.3,3).
   All decisions up to that time must stay bit-identical.
5. **Universe/survivorship.** The rank uses 30-day quote volume up to 2025-10-31. A coin delisted in TEST is closed at its
   last traded price, and its P&L is kept. No replacement coins. Delisted symbols must be reconstructed from
   the [BPD] archives (unverified that all delisted files are retained; check on download).
6. **Costs:** as in §4.2, charged on |Δposition| at the fill price. A stress run doubles costs.

### 4.5 Matched twins (each claim has exactly one twin, same params ±5%, same data, same seeds)

| claim | candidate | twin |
|---|---|---|
| WM beats model-free RL | Dreamer-lite actor-critic trained in latent imagination | PPO (and DQN with 5 discrete sizes) on the same encoder features, real data only |
| Rollout beats direct | RSSM 24-step hourly latent rollout → day RV / return heads | same encoder, direct day-ahead heads (`abl=direct`, as in power) |
| Continual beats frozen | online rule from §4.4 | the same 2025-10-31 weights, never updated (plus rolling-refit HAR/LGBM as the strong baseline) |
| Portable beats specific | pooled model, zero-shot on held-out coins/ETFs | per-asset HAR-X and LightGBM fitted on that asset's own DEV+VAL history |

## 5. Baselines (all on identical origins, costs and clocks)

- Trading: buy & hold (equal weight), TSMOM 1/7/30-day [MOP12, LT21], 1-day mean reversion, **vol-targeted TSMOM
  using a HAR forecast** (the honest strongest; [MM17]), and LightGBM on hand-crafted features (returns, ranges, taker-buy
  ratio, funding, volume z).
- Vol: HAR-RV, HAR-X (adds funding and taker imbalance), HARQ [HAR, BTCHAR], GARCH(1,1)/GJR, rolling-window
  re-estimation for all of them [HRW].
- FMs, zero-shot on log-RV and on returns: Chronos-2 (released 2025-10-17, before TEST) [C2], Chronos-Bolt, TiRex
  (xLSTM, 35M, 2025) [C2 refs, KDN], **Kronos-small** (pretrained to June 2024, Binance pairs included, clean for TEST; it
  has seen DEV, so DEV/VAL rankings flatter it) [KR]. Timestral is **CONTAMINATED** for any period after its release and
  gets no claims (`docs/audit_power.md` §11).
- RL: PPO, DQN (the twins above).

## 6. Splits

Keep them. They are consistent with three prior audits, they put Chronos-2, Kronos and TiRex weights before TEST, and
VAL (Jul–Oct 2025) gives about 120 days for freezing. Data: Binance perps from 2020-01 for warm-up and DEV. Download
everything once, but **the loader must refuse rows dated after 2025-10-31** until the auditor gives GO (assert on max date,
as in `docs/audit_commodity.md` 1e). TEST ends 2026-10-30 so that the last label is complete.

## 7. Architecture blueprints (Mac: CPU/MPS, one job at a time, < 1M params each)

- **Inputs** per coin per hour, 168-h context: log return, log(high/low), taker-buy ratio − 0.5, volume z-score (trailing
  30d), funding rate (forward-filled), and BTC's same features as a market token. Scale each coin by its trailing-30d vol.
  **No asset-ID embedding** (portability [SC19]). For ETFs, a daily-only adapter uses the same features, minus taker and
  funding, which are masked.
- **Pillar 1, WM:** a GRU-RSSM with 128-d deterministic and 16-d Gaussian stochastic state, a decoder for the next-hour
  feature vector, and KL balancing [DV3]. At 00:00, roll out 24 hourly latents × 32 samples → a distribution of day RV
  and return → P1/P3 heads. Twin: direct heads.
- **Pillar 2, RL:** a daily action a ∈ [−1, 1] per coin (position as a fraction of a vol-target budget). Reward:
  a·r_net − c·|Δa| − funding − λ·(a·r)². The actor-critic is trained on imagined 3-day latent trajectories, Dreamer-style,
  started from real DEV states. Twin: PPO/DQN on real transitions. A **WM-vol-targeted TSMOM rule** is reported alongside
  as a "planner-free" control.
- **Pillar 3, continual:** §4.4 rule. S ≤ 5 steps/day, lr frozen from VAL, anchor λ frozen. Twin: frozen weights.
- **Pillar 4, portability:** a 16-coin pooled fit. Zero-shot on 8 held-out coins and 13 ETFs. Twin: per-asset models.
- Budget: each model trains in under 1 h on CPU (≈ 16 coins × 50k h ≈ 0.8M samples). The 5-min klines for 24 coins
  × 6 years are ≈ 15M rows ≈ 0.5 GB parquet. Process one coin at a time.

## 8. Biggest risks (stated up front for investors)

1. **Power:** P1 needs about +6 pp over HAR to be significant. Realistic gains are 1–3 pp [AC24], so the likely honest
   result is "top-tier, not significant", as in power.
2. **P2 cannot be confirmed in one year** (MDE ≈ 2–2.5 Sharpe). Costs eat any return-timing edge [QH].
3. **The RV labels' common factor** (ρ is unknown; if it exceeds 0.6, N_eff shrinks). Measure it on VAL, never on TEST.
4. **The WM may not beat its direct twin**, as in power. In that case the pitch is "portable + continual vol engine",
   not "imagination".
5. **FM contamination:** Kronos saw DEV. Chronos-2's corpus cut-off is not public, but its weights predate TEST.

## 9. Sources

- [TC26] TechCrunch, EquiLibre, 2026-06-30: https://techcrunch.com/?p=3137627
- [BBG24] Bloomberg via Yahoo, "Ex-DeepMind trio … Tower Research": https://ca.finance.yahoo.com/news/ex-deepmind-trio-bring-algos-121716666.html
- [REC26] The Recursive, Series A: https://therecursive.com/equilibre-technologies-raises-series-a-at-eur438m-valuation-to-scale-ai-trading-agents
- [QH] Kim & Hansen, "The Quarter-Hour Effect" (Binance perps, 2021–2024): https://arxiv.org/abs/2607.09426 ; fee/gross figures via https://cryptoslate.com/the-invisible-opening-bell-inside-cryptos-endless-trading-day/
- [XPC] Bieganowski, explainable crypto microstructure: https://arxiv.org/abs/2602.00776
- [BPD] Binance public data: https://github.com/binance/binance-public-data/ (files at https://data.binance.vision)
- [KR] Kronos (pretraining to June 2024, tests from July 2024): https://arxiv.org/abs/2508.02739
- [RN25] Rahimikia & Ni, Re(Visiting) TSFMs in Finance: https://arxiv.org/abs/2511.18578
- [MAR25] Marconi, TSFMs on yields/vol/spreads: https://arxiv.org/abs/2507.07296
- [C2] Chronos-2 (2025-10-17): https://arxiv.org/abs/2510.15821 ; [KDN] TiRex overview: https://www.kdnuggets.com/5-time-series-foundation-models-you-are-missing-out-on
- [DV3] DreamerV3: https://arxiv.org/abs/2301.04104 ; [MZ] MuZero: https://arxiv.org/abs/1911.08265 ; [DTB] secondary trading note: https://the-algotrading-book-website.vercel.app/chapters/15-reinforcement-learning/251-dreamer-trading
- [ENS23] Wang & Klabjan, DRL ensemble crypto: https://arxiv.org/abs/2309.00626 ; [DRLS] survey summary: https://www.emergentmind.com/topics/deep-reinforcement-learning-for-cryptocurrency-trading
- [ONE] OneNet (NeurIPS 2023): https://arxiv.org/abs/2309.12659
- [HRW] "HARd to Beat", rolling windows: https://arxiv.org/abs/2406.08041 ; [AC24] Audrino & Chassot (1,445 stocks), same as HRW
- [HAR] Corsi (2009) HAR-RV: https://doi.org/10.1093/jjfinec/nbp001 ; [BTCHAR] Shen, Urquhart & Wang, Bitcoin HAR: https://eprints.nottingham.ac.uk/57681
- [SC19] Sirignano & Cont, universal price formation: https://arxiv.org/abs/1803.06917 ; [CRYU] crypto universal LOB model: https://arxiv.org/abs/2003.00803
- [MOP12] Moskowitz, Ooi & Pedersen, TSMOM: https://doi.org/10.1016/j.jfineco.2011.11.003 ; [LT21] Liu & Tsyvinski: https://nber.org/papers/w24877
- [MM17] Moreira & Muir, vol-managed portfolios: https://doi.org/10.1111/jofi.12513 ; [VMC] crypto vol management (blog): https://medium.com/coinmonks/whats-vol-got-to-do-with-it-886b57c2d762
- [MR83] Meese & Rogoff (1983): https://doi.org/10.1016/0022-1996(83)90017-X
- [DSR] Bailey & López de Prado, Deflated Sharpe: https://papers.ssrn.com/abstract=2460551 ; [PSR] PSR / MinTRL slides: https://newyork.qwafafew.org/wp-content/uploads/sites/4/2017/01/Lopez_de_Prado_Sharpe.pdf
