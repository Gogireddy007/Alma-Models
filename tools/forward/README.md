# Forward test

The TEST year has been scored several times, so the only clean evidence for the models now comes from data published
after they were frozen (2026-11-01). `FREEZE.json` pins the sha256 of every code file and checkpoint the forward
models use; `forward_eval.py score` refuses to run if any of them changed.

    .venv/bin/python tools/forward/forward_eval.py refresh --project gas      # weekly, after the Thursday EIA report
    .venv/bin/python tools/forward/forward_eval.py score   --project gas      # -> runs/forward/forward_gas_<date>.json

Models: gas `wm_gas_ens` vs `wm_gas`, `wm_gas_direct`, `st_ols_wx`; power `wm_power_blend` vs `wm_power`, `lear`;
fin `wm_fin_blend` vs `wm_fin`, `har_intra_dow` (fin data arrives monthly). Each file reports n and bootstrap 95%
intervals of the candidate-minus-competitor differences. Wait for about 20 gas weeks or 60 power/fin days before
reading anything into them. `ALMA_FORWARD_DRYRUN_START=<date>` tests the pipeline on old data and writes to
`runs/forward/DRYRUN_NOT_A_RESULT/` only.
