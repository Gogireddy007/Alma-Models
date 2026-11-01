# Pre-registered analysis plan (Phase 3: what counts as a meaningful win)

Author: Dr. Ingrid Solberg. Written 2026-10-30, under Gate 1, **before any final, Tier-D or TEST result exists**.
Scope: how every confirmatory number for Alma is computed, tested and worded. Changes need a dated entry
in §8, written before the affected data are scored. Where this file differs from `docs/eval_rigor.md`, the
stricter rule applies. All numbers in §0 come from existing VAL logs (nine 3-seed triples:
b_curr, b_d_a6p, b_dyn_d3, b_dyn_j2, b_prefixsyn, b_syn3, c_base, c_mix, syn5_only; all 2500 steps) and
`baselines/bench/per_task/*_val.csv`.

## 0. Measured noise and power (the inputs to every threshold below)

| Quantity (log scale; ×0.70 for wql_rel units near our level) | Value |
|---|---|
| Seed SD of the VAL headline, pooled over 9 configs | 0.0093 (≈ 0.0065 wql_rel); range 0.004 to 0.016 by config |
| Per-task seed SD (pooled) | median 0.022, IQR 0.013 to 0.034; 9/30 tasks above 0.03; max 0.128 (uk_covid_nation_1W/cumulative) |
| Ours vs ours, 80% power, two-sided 5%, n seeds each side | n=3: 2.8% relative (≈ 0.020 wql_rel); n=5: 1.9% (≈ 0.013) |
| Ours vs baseline, family-cluster SE of the mean log ratio (13 VAL families) | Chronos-2 0.040, TiRex 0.045, Timestral-28M 0.051, Timestral-209M 0.053 |
| Minimum detectable effect vs baseline, 80% power | Chronos-2 10.7%, TiRex 11.8%, Timestral-209M 13.7% (z); ≈ 1 point more with a t(12) critical value |
| Power to detect a true 10% gain (one interval) | Chronos-2 0.74, TiRex 0.66, Timestral-209M 0.52. Gate G is a conjunction (CI_fam, CI_famw, wql and mase), so its power is lower |
| Observed gain needed to show the gain is *at least* 10% (lower bound > 10%) | about 17% (Chronos-2) to 19% (Timestral-209M) |
| Seed share of the variance of a baseline comparison at n=3 | about 2.5% (seed SE 0.006 vs total 0.040). Task heterogeneity dominates |
| Task-win tests on 30 tasks (one-sided 5%) | H0 share ≤ 0.5 needs ≥ 20/30 (power 0.73 at true 0.7); H0 ≤ 0.6 needs ≥ 23/30 (power 0.28 at 0.7, 0.76 at 0.8) |
| Winner's curse: expected best-of-K noise gain vs incumbent, single seed | K=20: 0.017 wql_rel; K=60: 0.021; K=120: 0.024. With 3-seed means: 0.010 / 0.012 / 0.014 |

**Conclusion on power.** On 30 VAL tasks / 13 families, 10% over the strongest baseline is at the edge
of detectability (52 to 74% power for one interval) and cannot be shown *as a lower bound*. TEST (48 tasks,
18 families) shrinks the SE by about √(13/18) ≈ 0.85 if heterogeneity is similar, which gives roughly 9 to 12%
MDE. The 10% target is therefore pre-registered as a **point-estimate target with CI_fam upper < 1**,
not as "CI excludes 10%". Seeds barely move baseline power; families do. Seeds matter for ours-vs-ours.

## 1. Metrics and the strongest reproducible baseline

**Primary metric:** the probabilistic score of each suite. **Secondary:** point accuracy, calibration,
sharpness, runtime, memory. Official aggregation is computed first and reported first; ours second.

| Suite | Primary (official aggregation) | Secondary (official) | Ours (second) |
|---|---|---|---|
| fev-bench | scaled quantile loss: skill score (1 − gmean ratio to Seasonal Naive) and win rate, from the `fev` package at a pinned version | MASE skill and win rate | Gate G on WQL/MASE ratios, family-weighted ratio |
| GIFT-Eval | CRPS normalised by Seasonal Naive, geometric mean, and mean CRPS rank, from the official scripts at a pinned commit | MASE (same normalisation and rank) | Gate G, by domain, frequency and term (short/medium/long) |
| TIME | the leaderboard's published aggregate and script, pinned by commit before any run | its point metric | Gate G on its tasks |
| Our bench (VAL tuned-on; Tier-D; TEST) | task-weighted gmean `wql_rel` (fev convention) | `mase_rel`, `wql_rel_family`, cov80 error, sharpness | Gate G (`compare.py claim`) |

GIFT-Eval and TIME are not on disk (no downloads under this role). Their rows apply only after the download
is approved, the pin recorded here, and the overlap scan run. Every overlap of a baseline's pretraining
with a suite is listed per model and suite; exposed tasks are reported with and without.

**Sharpness** = mean (q90 − q10) width divided by Seasonal Naive's width on the same windows, reported with
cov80. **Runtime** = predict seconds per 1,000 forecasts, and **memory** = peak MB, both on the M3 at batch 1
and at the default batch, same session as the baselines.

**Strongest reproducible baseline B\*.** Chosen now from VAL bench numbers, before any Tier-D or TEST score:
1. Eligible: public weights; scored by `bench.py` under the identical protocol with provenance; re-derived by
   `tools/verify_claims.py` within 1e-3; finite on every task (AutoETS, VAL gmean 2.68, is excluded until fixed).
2. B\* = lowest task-weighted VAL `wql_rel`. Any eligible baseline within 1% of it, or better than it on
   `mase_rel` or `wql_rel_family`, joins B\*, and every target must hold against each member (intersection-union).
3. **Applied today:** Chronos-2 0.5586 and Timestral-209M 0.5639 are within 1% (ratio 1.0095), and
   Timestral-209M is better on mase (0.706 vs 0.720). **B\* = {Chronos-2, Timestral-209M}**, fixed for all suites.
   The leader of each external leaderboard is reported descriptively and is not gated.

**Relative improvement** is 100·(S_base − S_wm)/S_base per task, and 100·(1 − gmean ratio) for aggregates.

## 2. Task-level win / tie / loss and the "> 60% wins" test

- Δ_t = seed-mean log(S_wm) − log(S_base) on task t. **Tie if |Δ_t| ≤ 0.02** (≈ ±2%, the median 95%
  half-width of a 3-seed task mean, 1.96 × 0.022/√3). Win if Δ_t < −0.02, loss if Δ_t > 0.02.
- Sensitivity, always reported: ±1%, and a per-task CI rule (tie unless |Δ_t| > 1.96·s_t/√n with s_t the
  pooled per-task seed SD from the nine triples above). The current 0.1% band is reported only for continuity.
- Win share = (W + ½T)/N. **"> 60% task wins" holds only if** the point share > 0.60 **and** the one-sided 95%
  family-cluster bootstrap lower bound of the share is > 0.50 **and** a strict majority of families win
  (family Δ below −0.02). A share test against 0.60 itself is underpowered (§0) and is reported, not gated.

## 3. Uncertainty

- **Seeds.** 3 minimum for any reported number. 5 for: the final claim (Gate G `--min-seeds 5`), any
  ours-vs-ours decision with 0.013 ≤ Δ < 0.020 wql_rel, and any config whose 3-seed SD exceeds 0.012.
  **The seed that selected a config never counts in its confirmation** (fresh seeds only).
- **Bootstrap.** Family-cluster bootstrap, 10,000 draws, seed 0, task-weighted (CI_fam) and family-weighted
  (CI_famw). Seeds are averaged per task first; the seed t-interval is a separate gate (G4).
- **Combining seed and task variance:** Var(mean log ratio) ≈ σ²_family/G + σ²_seed,task/(n·N). We report
  both components. Recommended `compare.py` changes: (a) a two-stage bootstrap (resample families, then seed
  indices within) giving one interval; (b) a small-cluster correction (t_{G−1} critical value or BCa), since
  percentile intervals on 13 clusters are anti-conservative; (c) pair seeds by seed id from `prov.json`, not
  by glob position; (d) the ±2% tie band and win-share CI of §2; (e) `--k-configs K` printing the
  curse-adjusted gain; (f) a `retain` command for §4; (g) leave-one-family-out ranks; (h) sharpness columns.

## 4. Multiplicity and selection

- **Count.** About 60 VAL screens and a 122-config DEV search (66 completed in R0). Every config scored on a
  split increments that split's K, kept in `research_log.md` and reported with the claim.
- **Winner's curse.** For K=60 single-seed VAL screens, pure noise produces an expected best gain of about
  0.021 wql_rel (≈ 3%), larger than the R1(a) keep threshold of 0.014. For 3-seed means the curse is about
  0.012. Empirical Bayes on the full VAL and DEV spreads (between-config SD 0.07 to 0.08 vs seed SD 0.005 to
  0.009) says rankings are mostly real; the bias sits in the margins between near-top configs and the incumbent.
- **Re-confirmation on Tier-D.** Tier-D is a locked split, defined (task list and sha256 manifest committed
  here) before any candidate is frozen: source families disjoint from TRAIN, DEV, VAL and TEST, at least 8
  families, scored once per candidate set. A VAL gain g (fresh 3 seeds, vs the incumbent) is **confirmed**
  only if the Tier-D gain is > 0 with a one-sided 95% family-bootstrap bound **and** retains ρ = g_D/g ≥ 0.5.
  If ρ < 0.5 the candidate is reported as "VAL-selected, not confirmed" and is not promoted. If no clean Tier-D
  can be built without downloads, the ρ rule is applied on TEST and reported as part of the claim.
- **Corrections.** Screening is uncorrected and labelled exploratory. Confirmatory: "beats every member of
  the baseline set" is an intersection-union test, each (B, metric) at unadjusted α = 0.05; individual
  "beats B" statements use Holm across all (B, metric) statements; at most 3 candidates, Bonferroni across
  them (`--candidates`); one **primary** candidate named in §8 before unblinding. Pre-registered strata (§6)
  use Holm across strata. Secondary metrics are descriptive except calibration C1.

## 5. Decision rules for Gates 3 to 5

Assumed mapping: Gate 3 = early signal, Gate 4 = generalisation, Gate 5 = promotion to a frozen candidate.
Units are VAL `wql_rel`, Δ = incumbent − candidate, at 2500 steps, seed-matched, fresh seeds.

**Gate 3 (early signal).** Floor f = 0.008 (re-measured per R2).
- Signal: Δ ≥ 0.020 (2.5 f, the 3-seed MDE) on 3 fresh seeds, every seed improves, seed t-interval excludes 0.
- Ambiguous: 0.008 ≤ Δ < 0.020. Run 5 seeds; then signal iff Δ ≥ 0.013 and the interval excludes 0.
- Stop: Δ < 0.008, or `mase_rel` worse by > 1%, cov80 error worse by > 0.02, peak memory ≥ 3000 MB, any
  non-finite task, or losses (beyond the ±2% band) on more than 5 of 13 families.
- A DEV-search winner enters Gate 3 on VAL directly; its DEV margin is not evidence for VAL.

**Gate 4 (generalisation).** All must hold on VAL with the Gate 3 seeds:
- leave-one-family-out: the gain stays > 0 with each family dropped, and no family carries > 40% of the
  total log gain (no single-dataset dependence);
- family-weighted ratio < 1, and a strict majority of families win;
- no frequency, horizon tercile or domain stratum worse than the incumbent by > 5%;
- DEV (zero-shot TRAIN families) not worse than the incumbent beyond its seed noise;
- Tier-D confirmation with ρ ≥ 0.5 (§4).

**Gate 5 (promotion to frozen candidate).** Gate 4 passed; 5 seeds; Gate G on VAL vs B\* used only to gate
compute (never reported as a result); calibration within C1 limits (cov80 error ≤ B\*'s + 0.02, sharpness not
worse with coverage not better); efficiency at least as in `RESEARCH_PLAN.md`; provenance, `verify_claims.py`
and the audit_v3 checklist pass; checkpoint sha256 recorded in §8.

## 6. Reporting rules

- **Wording ladder** (TEST, frozen checkpoints, vs every member of B\*):
  "comparable to B" (any Gate G item fails) → "narrow advantage" (Gate G passes, ratio > 0.95) →
  "clear advantage" (ratio ≤ 0.95, CI_fam upper ≤ 0.98) → "meets the 10% target" (ratio ≤ 0.90 against each
  member of B\*, Gate G passes, > 60% wins per §2, 5 seeds, Tier-D confirmed, no family > 40% of the gain,
  no material calibration or efficiency loss).
- **Scoped claims** ("on long horizons only", "sub-daily only") are allowed only for these pre-registered
  strata: horizon ≥ 96; sub-daily frequency; intermittent (> 30% zeros); the covariate track vs chronos2_cov.
  Each needs ≥ 4 families, Holm across strata, and the overall result printed next to it. Any other stratum
  is labelled "post hoc, exploratory".
- **Full table, always:** every task × every baseline, seed mean ± SD, Δ_t, W/T/L under all three tie rules,
  per-family ratios, CI_fam, CI_famw, the seed interval, cov80, sharpness, runtime and memory, each model's
  effective context, and exposure flags. VAL columns are headed "VAL (tuned on)".
- **Negative results** are printed in the same table: failed gates, ablations that show no world-model
  benefit, the VAL-to-Tier-D retention ρ, the VAL-to-TEST gap, K per split, and crashed or excluded runs.
- **Deviations** go in §8 with a date and reason. Any analysis changed after unblinding is reported both ways,
  the pre-registered one first.

## 7. Current practice that violates this plan (to fix before Gate 5)

1. R1(a) keeps a single-seed gain of 0.014; after K ≈ 10 screens that is the expected best noise gain.
2. Rung-2 triples reuse the screening run as seed 0 (`x.log`, `x_s1`, `x_s2`): selection bias enters the
   confirmation. Rung 2 also re-uses VAL, so it controls seeds, not task selection; no Tier-D exists.
3. The "0.008 noise floor" keep rule is about 1.5 SE for a 3-seed difference (p ≈ 0.2), not a significance bar.
4. `compare.py` ties at 0.1% (task) and −0.001 (family), well inside the 2.2% median per-task seed SD.
5. `program.md` "The bar" quotes old-suite numbers (Timestral 0.569, TiRex 0.603, Chronos-2 0.621); on the
   bench VAL, Chronos-2 is lowest. B\* was not frozen. AutoETS (VAL gmean 2.68) is still in the success list.
6. `runs/ledger.tsv` does not exist although lessons L7 calls it canonical.
7. DEV screens run 1200 steps on 9 tasks / 5 families with seed noise from one triple; DEV-to-VAL
   validity is unchecked.

## 8. Registry and deviations (append only)

- 2026-10-30: plan registered. B\* = {Chronos-2, Timestral-209M}. Tier-D: not yet defined. Primary
  candidate: not yet named. GIFT-Eval/TIME pins: pending download approval.
