"""
Covariate-aware adapter for tools/bench.py (bench.py itself is not modified).

bench.py hands every model univariate contexts only: multivariate targets are split into independent
series and fev's past / known-future covariate columns are dropped. This adapter rebuilds the SAME
windows (same series subsample, same contexts, same targets; asserted against bench._build_task) and
attaches, per forecast, the covariates fev-bench defines for the task:

    past   [Cp, len(ctx)]       observed up to the forecast origin only: fev past_dynamic_columns,
                                plus (COV_SIBLINGS) the other target variates of the same item
    known  [Ck, len(ctx) + h]   known through the horizon: fev known_dynamic_columns

Scoring is delegated to bench.evaluate, one task at a time, so the metric code is identical; the
aggregates are recomputed exactly as bench.evaluate does. predict_fn gets a 5th argument:
predict_fn(contexts, h, season, freq, covs) with covs a list (one per context) of {"past", "known"}.

Covariate specs come from the public fev-bench task definitions (github.com/autogluon/fev,
benchmarks/fev_bench/tasks.yaml @ f323f6c). The test split stays behind bench._guard.

FAIRNESS: the baselines in baselines/bench/ were scored WITHOUT covariates. A covariate-aware Alma
must be compared against covariate-aware baselines (chronos2_cov below) before any claim.
"""

import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

# fev-bench covariate columns per bench task name: (known_dynamic, past_dynamic). Static columns ignored.
COV_SPEC = {
    # VAL
    "favorita_stores_1D": (["holiday", "onpromotion"], ["oil_price"]),
    "favorita_stores_1W": (["onpromotion"], ["oil_price"]),
    "favorita_stores_1M": (["onpromotion"], ["oil_price"]),
    "kdd_cup_2022_1D": ([], ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv"]),
    "uk_covid_nation_1D/new": ([], ["new_vaccinated_1", "new_vaccinated_2", "new_vaccinated_3",
                                    "hospital_cases", "icu_ventilator_occupancy"]),
    "uk_covid_nation_1W/new": ([], ["new_vaccinated_1", "new_vaccinated_2", "new_vaccinated_3",
                                    "hospital_cases", "icu_ventilator_occupancy"]),
    "uk_covid_nation_1D/cumulative": ([], ["cumulative_vaccinated_1", "cumulative_vaccinated_2",
                                           "cumulative_vaccinated_3", "hospital_cases", "icu_ventilator_occupancy"]),
    "uk_covid_nation_1W/cumulative": ([], ["cumulative_vaccinated_1", "cumulative_vaccinated_2",
                                           "cumulative_vaccinated_3", "hospital_cases", "icu_ventilator_occupancy"]),
    # TEST (only usable under TSAR_SPLIT=test; listed so the confirmatory run needs no code change)
    **{f"entsoe_{f}": (["radiation_diffuse_horizontal", "temperature", "radiation_direct_horizontal"], [])
       for f in ("15T", "30T", "1H")},
    "epf_be": (["Generation forecast", "System load forecast"], []),
    "epf_de": (["PV+Wind Forecast", "Ampirion Load Forecast"], []),
    "epf_fr": (["Generation forecast", "System load forecast"], []),
    "epf_np": (["Grid load forecast", "Wind power forecast"], []),
    "epf_pjm": (["Zonal COMED load foecast", "System load forecast"], []),
    "gvar": ([], ["poil", "pmat", "pmetal"]),
    "hermes": (["external"], []),
    **{f"rohlik_orders_{f}": (["holiday", "shops_closed", "winter_school_holidays", "school_holidays"],
                              ["shutdown", "mini_shutdown", "blackout", "mov_change", "frankfurt_shutdown",
                               "precipitation", "snow", "user_activity_1", "user_activity_2"]) for f in ("1D", "1W")},
    **{f"rohlik_sales_{f}": (["total_orders", "sell_price_main"] + [f"type_{i}_discount" for i in range(7)]
                             + ["holiday", "shops_closed", "winter_school_holidays", "school_holidays"],
                             ["availability"]) for f in ("1D", "1W")},
    "rossmann_1D": (["SchoolHoliday", "Promo", "DayOfWeek", "Open", "StateHoliday"], ["Customers"]),
    "rossmann_1W": (["Open", "SchoolHoliday", "StateHoliday", "Promo"], ["Customers"]),
    **{f"solar_with_weather_{f}": (["wind_speed", "day_length", "humidity", "rain_1h", "snow_1h", "temp", "pressure"],
                                   ["global_horizontal_irradiance", "clouds_all"]) for f in ("15T", "1H")},
    **{f"uci_air_quality_{f}": (["T", "RH", "AH"], []) for f in ("1D", "1H")},
    "walmart": (["MarkDown2", "Unemployment", "MarkDown4", "Fuel_Price", "MarkDown1", "Temperature",
                 "MarkDown5", "MarkDown3", "CPI", "IsHoliday"], []),
}

SIBLINGS = int(os.environ.get("TSAR_COV_SIBLINGS", 1))    # other target variates as past covariates
MAX_SIBLINGS = int(os.environ.get("TSAR_COV_MAX_SIB", 12))  # cap (jena has 21 variates)


def _num(v):
    """A covariate cell list -> float32 array. Strings / categoricals (e.g. favorita 'holiday') become
    0/1 event indicators (1 = a non-empty label); booleans 0/1."""
    a = np.asarray(v)
    if a.dtype.kind in "fiub":
        return a.astype(np.float32)
    s = pd.Series(a)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.to_numpy(np.float32)
    return (s.notna() & (s.astype(str).str.len() > 0) & (s.astype(str) != "0")).to_numpy(np.float32)


def build_task_cov(spec):
    """Mirror of bench._build_task that keeps, for each window, the covariate slices. Returns the bench
    task dict (unchanged fields) plus "covs"."""
    base = B._build_task(spec)
    df = pd.read_parquet(B._fev_file(spec["config"]))
    cols = B._target_columns(df, spec["target"])
    known_cols, past_cols = COV_SPEC.get(spec["name"], ([], []))
    H, W, step, cut = spec["horizon"], spec["num_windows"], spec["step"], spec["cutoff"]
    min_ctx = max(B.MIN_CONTEXT, int(spec["min_context"]))
    series = []   # same order as bench: rows x target columns
    for ri, row in df.iterrows():
        for c in cols:
            series.append((ri, c, np.asarray(row[c], dtype=np.float32), row["timestamp"]))
    rng = np.random.default_rng(B.SEED + zlib.crc32(spec["name"].encode()))
    n_keep = max(1, B.MAX_FORECASTS_PER_TASK // W)
    if len(series) > n_keep:
        keep = np.sort(rng.permutation(len(series))[:n_keep])
        series = [series[i] for i in keep]
    covs, ctxs = [], []
    for ri, col, y, ts in series:
        row = df.loc[ri]
        kn = [_num(row[k]) for k in known_cols]
        pa = [_num(row[k]) for k in past_cols]
        if SIBLINGS and len(cols) > 1:
            pa += [np.asarray(row[c], dtype=np.float32) for c in cols if c != col][:MAX_SIBLINGS]
        if isinstance(cut, str):
            first = int(np.searchsorted(pd.DatetimeIndex(ts).values, np.datetime64(cut), side="right"))
        else:
            first = len(y) + cut
        for k in range(W):
            c = first + k * step
            if c < 0 or c + H > len(y):
                continue
            ctx = y[max(0, c - B.MAX_CONTEXT):c]
            fin = np.flatnonzero(np.isfinite(ctx))
            if len(fin) < min_ctx:
                continue
            s0 = max(0, c - B.MAX_CONTEXT) + fin[0]
            ctxs.append(ctx[fin[0]:])
            covs.append({"past": np.stack([p[s0:c] for p in pa]) if pa else np.zeros((0, c - s0), np.float32),
                         "known": np.stack([q[s0:c + H] for q in kn]) if kn else np.zeros((0, c + H - s0), np.float32)})
    assert len(ctxs) == len(base["contexts"]), f"{spec['name']}: window mismatch"
    for a, b in zip(ctxs, base["contexts"]):
        assert len(a) == len(b) and np.array_equal(a, b, equal_nan=True), f"{spec['name']}: context mismatch"
    base["covs"] = covs
    base["n_known"], base["n_past"] = len(known_cols), (covs[0]["past"].shape[0] if covs else 0)
    return base


def load_tasks_cov(split, names=None):
    B._guard(split, f"bench_cov.load_tasks_cov({split})")
    return [build_task_cov(s) for s in B.SPLIT_TASKS[split] if names is None or s["name"] in names]


def evaluate(predict_fn, split="val", verbose=True, names=None, tasks=None, use_covs=True):
    """bench.evaluate, task by task, with each task's covariates bound to predict_fn's 5th argument.
    use_covs=False passes covs=None (same model, covariates withheld: the clean ablation)."""
    tasks = tasks if tasks is not None else load_tasks_cov(split, names)
    rows, secs = [], 0.0
    for t in tasks:
        covs = t["covs"] if use_covs else None

        def fn(ctxs, h, m, freq, _c=covs):
            return predict_fn(ctxs, h, m, freq, _c)
        fn.__name__ = getattr(predict_fn, "__name__", "predict_cov")
        r = B.evaluate(fn, split, verbose=False, tasks=[t])
        rows.append(r["per_dataset"])
    df = pd.concat(rows, ignore_index=True)
    fam = df.groupby("family")[["wql_rel", "mase_rel"]].agg(B._gmean)
    result = {"split": split, "wql_rel": B._gmean(df["wql_rel"]), "mase_rel": B._gmean(df["mase_rel"]),
              "cov80_error": float((df["cov80"] - 0.8).abs().mean()),
              "wql_rel_family": B._gmean(fam["wql_rel"]), "mase_rel_family": B._gmean(fam["mase_rel"]),
              "win_rate_vs_sn": float((df["wql_rel"] < 1).mean()), "n_tasks": len(df), "n_families": len(fam),
              "predict_seconds": float(df["predict_seconds"].sum()), "per_dataset": df}
    if verbose:
        with pd.option_context("display.width", 160, "display.precision", 3, "display.max_rows", 200):
            print(df.drop(columns=["sn_wql", "sn_mase"]).to_string(index=False))
        print(f"{split}: wql_rel={result['wql_rel']:.4f} mase_rel={result['mase_rel']:.4f} "
              f"cov80_error={result['cov80_error']:.4f} family-gmean wql_rel={result['wql_rel_family']:.4f} "
              f"({result['n_tasks']} tasks, {result['n_families']} families, covariates={'on' if use_covs else 'off'})")
    return result


def chronos2_cov(batch=32):
    """Covariate-aware Chronos-2 for the fair comparison (NOT yet run; heavy). Uses the dict input of
    Chronos2Pipeline.predict_quantiles: past_covariates / future_covariates."""
    state = {}

    def predict(ctxs, h, m, freq, covs=None):
        import torch
        import prepare as P
        from chronos import BaseChronosPipeline
        if "pipe" not in state:
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            state["pipe"] = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=dev, torch_dtype=torch.float32)
        qs, out = P.QUANTILES.tolist(), []
        for s in range(0, len(ctxs), batch):
            inp = []
            for i in range(s, min(s + batch, len(ctxs))):
                d = {"target": ctxs[i]}
                if covs is not None:
                    pc = {f"p{j}": v for j, v in enumerate(covs[i]["past"])}
                    pc.update({f"k{j}": v[:len(ctxs[i])] for j, v in enumerate(covs[i]["known"])})
                    if pc:
                        d["past_covariates"] = pc
                    if len(covs[i]["known"]):
                        d["future_covariates"] = {f"k{j}": v[len(ctxs[i]):] for j, v in enumerate(covs[i]["known"])}
                inp.append(d)
            q = state["pipe"].predict_quantiles(inp, prediction_length=h, quantile_levels=qs)[0]
            out.append(torch.stack([t.reshape(-1, h, len(qs))[0] for t in q]).permute(0, 2, 1).float().cpu().numpy())
        return np.concatenate(out)
    return predict


if __name__ == "__main__":
    # covariate summary of the val split (cheap): python tools/bench_cov.py
    for t in load_tasks_cov("val"):
        if t["covs"] and (t["n_known"] or t["n_past"]):
            c = t["covs"][0]
            print(f"{t['name']:32s} forecasts={len(t['covs']):4d} known={t['n_known']} past={t['n_past']} "
                  f"past_nan={np.isnan(c['past']).mean() if c['past'].size else 0:.2f}")
