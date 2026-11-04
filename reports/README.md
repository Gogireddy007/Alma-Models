# Reports

| File | Contents |
|---|---|
| [`Alma_Models_Status_Report_2026-11-03.pdf`](Alma_Models_Status_Report_2026-11-03.pdf) | Full project status report (17 pages): verified results against all benchmarks, the audit, improvements, what went wrong, targets and time to confirmation. Every number is traced to a source file (Appendix A). |
| [`Alma_Models_Architecture.pdf`](Alma_Models_Architecture.pdf) | Architecture diagrams (8 pages), drawn from the model source code: end-to-end system, world model vs direct twin, wm_gas, wm_gas_ens, wm_power, wm_fin, wm_sv, and the forward test. |
| [`Alma_Models_Audit_Report_2026-11-01.pdf`](Alma_Models_Audit_Report_2026-11-01.pdf) | Short audit summary of 1 November 2026, superseded by the status report. |
| `alma_status_report.tex`, `alma_architecture.tex` | LaTeX sources. |
| `make_figures.py`, `figures/` | Builds every figure from the TEST reports and audit outputs; `facts.json` holds every plotted value. |
| `projections.py` | Forward sample sizes and the forward-test calendar. |
| `check_numbers.py` | Checks that every number in the report appears in a source file. |

The scripts were run with `report/` next to the repository clone (`<root>/report/`, `<root>/repo/`,
`<root>/audit/`); see `../audit/README.md`.
