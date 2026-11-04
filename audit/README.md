# Independent audit (1–3 November 2026)

Scripts, logs and outputs of the independent re-audit of Alma Models. The write-up is in [`REPORT.md`](REPORT.md),
and the full status report with figures is in [`../reports/`](../reports/).

## Layout note
These scripts were run with this folder **next to** the repository clone (`<root>/audit/` and `<root>/repo/`), and
they resolve the repo as `../repo`. To re-run them from this copy, place the folder next to a clone of the repository
(or create a `repo` symlink that points at the repository root).

## Contents
| Path | What it is |
|---|---|
| `oil_check.py`, `gas_check.py`, `power_check.py`, `fin_check.py`, `fin_score.py` | Independent re-scores of the frozen models, with leak tests |
| `gas_exp.py`, `run_gas_jobs.py`, `gas_final.py`, `gas_calib.py`, `gas_final2.py`, `gas_ens_leak.py`, `GAS_FREEZE.md` | Gas ensemble and interval experiments (DEV first, a written freeze, one TEST look) |
| `fin_val.py`, `fin_blend.py`, `fin_boot.py` | Crypto blend selection on VAL, TEST score and bootstrap |
| `power_blend.py`, `power_boot.py`, `power_wrap_check.py` | Power blend selection on VAL, TEST score and bootstrap |
| `alma_bench_check.py`, `run_ssm.sh`, `ssm_lanes.sh`, `ssm_analysis.py`, `logs/aud_ssm_*.log` | 5-seed SSM world-model test (research track) |
| `validation/*.out` | Outputs of every script, re-run on 3 November 2026; the reports cite these |
| `*.csv`, `*.json` | Result tables |
| `fin_test/`, `fin_val/`, `power_test/`, `power_val/` | Cached model forecasts (quantiles only, no market data) |

Raw market data is not included (source licences); rebuild it with the repository's fetch scripts.
The 8 audit checkpoints of the SSM test are in `../data/ckpt/aud_ssm_*.pt`.
