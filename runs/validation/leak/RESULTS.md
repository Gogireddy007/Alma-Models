# V3 — Exhaustive leak tests (every Oct-Nov 2026 origin) with positive controls

TEST rows: future corrupted, forecast must NOT change (changed = 0). CONTROL rows: a deliberately buggy context builder lets one future value in, and the same test MUST flag it (changed > 0).

| project | model | corruption | forecasts | changed | kind | result |
|---|---|---|---|---|---|---|
| oil | wm_sv | noise | 24 | 0 | leak test | **as expected** |
| oil | wm_sv | nan | 24 | 0 | leak test | **as expected** |
| oil | wm_sv | extreme | 24 | 0 | leak test | **as expected** |
| oil | wm_sv | shift | 24 | 0 | leak test | **as expected** |
| oil | wm_sv | off-by-one | 23 | 23 | positive control | **as expected** |
| oil | wm_sv_direct | noise | 24 | 0 | leak test | **as expected** |
| oil | wm_sv_direct | nan | 24 | 0 | leak test | **as expected** |
| oil | wm_sv_direct | extreme | 24 | 0 | leak test | **as expected** |
| oil | wm_sv_direct | shift | 24 | 0 | leak test | **as expected** |
| oil | wm_sv_direct | off-by-one | 23 | 23 | positive control | **as expected** |
| gas | wm_gas | noise | 10 | 0 | leak test | **as expected** |
| gas | wm_gas | nan | 10 | 0 | leak test | **as expected** |
| gas | wm_gas | extreme | 10 | 0 | leak test | **as expected** |
| gas | wm_gas | shift | 10 | 0 | leak test | **as expected** |
| gas | wm_gas | observed-wx | 5 | 5 | positive control | **as expected** |
| gas | wm_gas_direct | noise | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_direct | nan | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_direct | extreme | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_direct | shift | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_direct | observed-wx | 5 | 5 | positive control | **as expected** |
| gas | wm_gas_ens | noise | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_ens | nan | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_ens | extreme | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_ens | shift | 10 | 0 | leak test | **as expected** |
| gas | wm_gas_ens | observed-wx | 5 | 5 | positive control | **as expected** |
| power | wm_power | noise | 36 | 0 | leak test | **as expected** |
| power | wm_power | nan | 36 | 0 | leak test | **as expected** |
| power | wm_power | extreme | 36 | 0 | leak test | **as expected** |
| power | wm_power | shift | 36 | 0 | leak test | **as expected** |
| power | wm_power | includes-D | 35 | 35 | positive control | **as expected** |
| power | wm_power_direct | noise | 36 | 0 | leak test | **as expected** |
| power | wm_power_direct | nan | 36 | 0 | leak test | **as expected** |
| power | wm_power_direct | extreme | 36 | 0 | leak test | **as expected** |
| power | wm_power_direct | shift | 36 | 0 | leak test | **as expected** |
| power | wm_power_direct | includes-D | 35 | 35 | positive control | **as expected** |
| fin | wm_fin | noise | 34 | 0 | leak test | **as expected** |
| fin | wm_direct | noise | 34 | 0 | leak test | **as expected** |
| fin | wm_fin | nan | 34 | 0 | leak test | **as expected** |
| fin | wm_direct | nan | 34 | 0 | leak test | **as expected** |
| fin | wm_fin | extreme | 34 | 0 | leak test | **as expected** |
| fin | wm_direct | extreme | 34 | 0 | leak test | **as expected** |
| fin | wm_fin | shift | 34 | 0 | leak test | **as expected** |
| fin | wm_direct | shift | 34 | 0 | leak test | **as expected** |
| fin | wm_fin | includes-D | 11 | 10 | positive control | **as expected** |
| fin | wm_direct | includes-D | 11 | 10 | positive control | **as expected** |

**45/45 rows as expected; 872 corrupted-future forecasts checked.**
