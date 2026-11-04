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

## Scorer changes

`forward_eval.py` is not sha-pinned in `FREEZE.json` (only model code and checkpoints are), but it scores a running
forward test, so every change to it is recorded here and needs the owner's agreement before merging.

- **2026-11-04: fin P1 label now comes from `harness.Panel.labels`.** `score_fin()` used to build the P1 reference as
  `rolling(7)` over the days present in the parquet, so it counted 7 *rows*, not 7 *calendar days*. The Binance
  archive has no bars at all for 2026-10-28, so for decision days 2026-10-29..2026-11-04 the old code quietly used D-8
  in place of the missing day. The pre-registered label (`Panel.labels`, used by `tools/fin/run_eval.py`) runs on the
  complete UTC calendar and leaves those coin-days unlabelled. `score_fin()` now takes `logrv`/`ref` from
  `P.labels(i)` for each scored day. The P3 quantile score keeps the `run_eval.py` mask (finite log RV and finite
  quantiles, no reference needed), so days with log RV but no P1 label still count for QS. The output also gives
  `n_p1_labelled` and `p1_unlabelled_days`. Found by the deployment audit (`docs/deploy_2026_10_11.md` §5, finding 2).
