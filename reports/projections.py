"""Approximate forward-sample sizes needed to confirm each TEST-year improvement, and the forward-test calendar.

For a paired difference with TEST mean d and bootstrap 95% CI [lo, hi] on n TEST units, SE = (hi - lo) / (2 * 1.96)
and the per-unit SD is SE * sqrt(n). Units needed for a two-sided 5% test with 80% power, if the same effect and
variability persist: n* = ((1.96 + 0.84) * SD / |d|)^2. Inputs are copied by regex from audit/validation/*.out."""
import datetime as dt
import os
import re

AUD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit", "validation")
rd = lambda f: open(os.path.join(AUD, f)).read()
cases = []
g = rd("gas_final.out")
for kind in ("thu", "fri"):
    for other in ("wm_gas \\(frozen\\)", "st_ols_wx"):
        m = re.search(rf"{kind} NN pool \+ 0.20 OLS - {other}: dMAE ([-+\d.]+) CI\[([-+\d.]+),([-+\d.]+)\]", g)
        cases.append((f"gas {kind} MAE vs {other.replace(chr(92), '')}", 52, "weeks", *map(float, m.groups())))
p = rd("power_boot.out")
for other in ("wm_power", "lear"):
    for met in ("QS", "P2"):
        m = re.search(rf"blend vs {other}\s+{met}: diff ([-+\d.]+)\s+95% CI \[([-+\d.]+), ([-+\d.]+)\]", p)
        cases.append((f"power {met} vs {other}", 365, "days", *map(float, m.groups())))
f = rd("fin_boot.out")
for other in ("wm_fin", "har_intra_dow"):
    m = re.search(rf"blend0.4\s+vs {other}\s+acc ([-+\d.]+)pp CI\[([-+\d.]+),([-+\d.]+)\]", f)
    cases.append((f"crypto P1 (pp) vs {other}", 364, "days", *map(float, m.groups())))
    m = re.search(rf"blend0.4\s+vs {other}\s+acc .*?\| QS ([-+\d.]+) CI\[([-+\d.]+),([-+\d.]+)\]", f)
    cases.append((f"crypto QS vs {other}", 364, "days", *map(float, m.groups())))
print(f"{'comparison':38s} {'n':>4s} {'diff':>8s} {'CI lo':>8s} {'CI hi':>8s} {'signif':>6s} {'n* units':>9s} {'~years':>6s}")
for name, n, unit, d, lo, hi in cases:
    se = (hi - lo) / (2 * 1.96); sd = se * n ** 0.5
    nstar = ((1.96 + 0.84) * sd / abs(d)) ** 2 if d != 0 else float("inf")
    yrs = nstar / (52 if unit == "weeks" else 365)
    sig = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"{name:38s} {n:4d} {d:+8.4f} {lo:+8.4f} {hi:+8.4f} {sig:>6s} {nstar:9.0f} {yrs:6.1f}  ({unit})")
# Crypto TEST n = 364 days (L.shape[0] in fin_boot.py, 2025-11-01..2026-10-30).
first = dt.date(2026, 11, 5)
print("\nforward calendar (EIA weekly storage report, normally Thursdays; holiday weeks can shift by a day):")
for k in (1, 4, 13, 20, 52):
    print(f"  gas release #{k:2d}: {first + dt.timedelta(weeks=k - 1)}")
start = dt.date(2026, 11, 2)
for k in (30, 60, 180, 365):
    print(f"  power/crypto day {k:3d}: {start + dt.timedelta(days=k - 1)}")
