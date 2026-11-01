# Alma-Gas harness

Binding protocol: `docs/gas_prereg.md`. Leakage record: `docs/gas_leakage.md`. Results: `docs/gas_results.md`.

    nice -n 10 .venv/bin/python tools/gas/fetch.py                 # raw data -> data/gas/raw, manifests
    nice -n 10 .venv/bin/python tools/gas/data.py build            # daily panel (price task)
    nice -n 10 .venv/bin/python tools/gas/storage.py build         # weekly table + CPC degree days (storage task)
    nice -n 10 .venv/bin/python tools/gas/test_infoset.py          # information-set tests (must pass)
    nice -n 10 .venv/bin/python tools/gas/run_eval.py --task storage --split dev --models st_ols_wx,wm_mymodel

Plugging in a model (architect): create `tools/gas/wm_<name>.py` with `register(REGISTRY)` adding factories that
return an object with
- `storage(ctxs, qs, series) -> [n, len(qs)]` quantiles of dS_W (Bcf) for `series` in
  `storage.SERIES`, ctxs from `storage.build_context(k, kind)` (keys: avg5, ds_hist, dev_hist, lvl_hist,
  hist_weeks, hist_hdd, hist_cdd, wx [7,2], wx_src, wx_normal, obs_daily, hh, kind, target_week, ...), and/or
- `batch(ctxs, h, qs) -> [n, len(qs), h]` price quantiles, ctxs from `data.build_context(i, 1024)` (keys y, covs
  [ctx, len(data.COVS)], ctx_dates, origin_date);
- optional `meta` (dict; put `ckpt` + `ckpt_sha256` for checkpoints), `stride`.
Training data only via `data.train_panel / train_windows` and `storage.train_weeks` (+ `storage.build_context` on
those weeks). Never read `data/gas/*.csv` rows after 2025-06-30 for fitting. Run DEV with `--split dev`; VAL once
for the frozen candidate; TEST only by the manager after the auditor's GO.
