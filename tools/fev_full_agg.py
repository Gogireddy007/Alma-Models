"""
Aggregate Alma's fev-bench summaries with the official leaderboard result CSVs using fev.analysis.leaderboard /
pairwise_comparison, with exactly the settings of scripts/generate_fev_bench_figures.py in the fev repo
(baseline "Seasonal Naive", leakage imputation "Chronos-Bolt", missing_strategy="impute", clip [0.01, 100],
1000 bootstrap resamples, seed 123). Every leaderboard model is recomputed on the SAME task subset as Alma.

    HF_DATASETS_OFFLINE=1 .venv-eval/bin/python -I tools/fev_full_agg.py --fev-clone <fev clone> \\
        --models alma_ssm_s0,alma_ssm_s3,alma_ssm_s4 --out runs/fev_full/agg

Subsets: nontest52 (every fev-bench task outside our TEST split), nonval22 (nontest52 minus our 30 VAL tasks,
which Alma was selected on), val30. --check reproduces the published full-100 numbers first.
"""
import os
import re
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE, LEAK_MODEL, NRES, SEED = "Seasonal Naive", "Chronos-Bolt", 1000, 123
FOCUS = ["Timestral-5M", "Timestral-209M", "Chronos-2", "TiRex", "Chronos-Bolt", "AutoETS", "AutoTheta",
         "Seasonal Naive"]
PARAMS = {"Timestral-5M": "5.2M", "Timestral-209M": "209M", "Chronos-2": "120M", "TiRex": "35M",
          "Chronos-Bolt": "205M", "AutoETS": "-", "AutoTheta": "-", "Seasonal Naive": "-"}
DOMAIN = {  # our own grouping (fev-bench ships no domain column)
    "energy": ["proenfo_gfc12", "proenfo_gfc14", "proenfo_gfc17", "solar_1D", "solar_1W", "kdd_cup_2022_10T",
               "kdd_cup_2022_30T", "kdd_cup_2022_1D", "ETT_1W", "ETT_1D", "ETT_1H", "ETT_15T", "ercot_1H",
               "ercot_1D", "ercot_1W", "ercot_1M"],
    "retail": ["restaurant", "m5_1D", "m5_1W", "m5_1M", "hierarchical_sales_1D", "hierarchical_sales_1W",
               "favorita_stores_1D", "favorita_stores_1W", "favorita_stores_1M", "favorita_transactions_1D",
               "favorita_transactions_1W", "favorita_transactions_1M"],
    "mobility": ["LOOP_SEATTLE_5T", "LOOP_SEATTLE_1D", "LOOP_SEATTLE_1H", "M_DENSE_1D", "M_DENSE_1H",
                 "SZ_TAXI_15T", "SZ_TAXI_1H"],
    "weather": ["jena_weather_10T", "jena_weather_1D", "jena_weather_1H"],
    "cloud/IT": ["bizitobs_l2c_1H", "bizitobs_l2c_5T"],
    "health": ["hospital", "uk_covid_utla_1D/new", "uk_covid_utla_1W/cumulative", "uk_covid_nation_1D/new",
               "uk_covid_nation_1D/cumulative", "uk_covid_nation_1W/new", "uk_covid_nation_1W/cumulative"],
    "econ/tourism": ["australian_tourism", "fred_md_2025/cee", "fred_qd_2025/cee", "fred_md_2025/macro",
                     "fred_qd_2025/macro"],
}
DOM = {t: d for d, ts in DOMAIN.items() for t in ts}


def freq_class(name, horizon_s):
    m = re.search(r"_(\d*)(T|H|D|W|M|Q|Y)(/|$)", name)
    special = {"proenfo_gfc12": "H", "proenfo_gfc14": "H", "proenfo_gfc17": "H", "restaurant": "D",
               "hospital": "M", "australian_tourism": "Q", "fred_md_2025/cee": "M", "fred_md_2025/macro": "M",
               "fred_qd_2025/cee": "Q", "fred_qd_2025/macro": "Q"}
    if name in special:
        return special[name]
    if not m:
        return "?"
    return {"T": "sub-hourly"}.get(m.group(2), m.group(2)) if m.group(2) != "T" else "sub-hourly"


def split_sets():
    src = open(os.path.join(ROOT, "tools", "bench.py")).read()
    val = src.split("VAL_TASKS = [")[1].split("TEST_TASKS = [")[0]
    test = src.split("TEST_TASKS = [")[1].split("\n]\n")[0]
    pat = r"^\s+\('([^']+)', '[^']+', '([^']+)'"
    return {c for _, c in re.findall(pat, val, re.M)}, {c for _, c in re.findall(pat, test, re.M)}


def load_lb(clone):
    fs = sorted(glob.glob(os.path.join(clone, "benchmarks", "fev_bench", "results", "*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def load_alma(model, name):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "fev_full", model,
                                                                      "summaries", "*.json")))]
    df = pd.DataFrame(rows)
    df["model_name"] = name
    return df


def lb(df, metric, n_res=NRES):
    import fev
    r = fev.analysis.leaderboard(df, metric_column=metric, missing_strategy="impute", baseline_model=BASE,
                                 leakage_imputation_model=LEAK_MODEL, n_resamples=n_res, seed=SEED)
    r = r.astype("float64")
    for c in [c for c in r.columns if c.startswith(("skill_score", "win_rate"))]:
        r[c] = r[c] * 100
    return r


def pairwise(df, metric, alma, others):
    import fev
    inc = [alma] + [m for m in others if m != alma]
    inc += [m for m in (BASE, LEAK_MODEL) if m not in inc]
    r = fev.analysis.pairwise_comparison(df, metric_column=metric, missing_strategy="impute", baseline_model=BASE,
                                         leakage_imputation_model=LEAK_MODEL, n_resamples=NRES, seed=SEED,
                                         included_models=inc)
    return r.xs(alma, level="model_1").astype("float64") * 100


def rel_table(df, metric):
    """Per-task error relative to Seasonal Naive after leakage and failure imputation, clipped to [0.01, 100]:
    exactly the matrix whose geometric mean the skill score is. Index = task_name."""
    from fev.analysis import pivot_table, _handle_leakage_imputation, _load_summaries, TASK_DEF_COLUMNS
    e = pivot_table(df, metric_column=metric, baseline_model=BASE)
    ov = pivot_table(df, metric_column="trained_on_this_dataset")
    e = _handle_leakage_imputation(e, ov, LEAK_MODEL).clip(0.01, 100).fillna(1.0)
    s = _load_summaries(df.drop_duplicates("task_name"))
    m = {tuple(r[c] for c in TASK_DEF_COLUMNS): r["task_name"] for _, r in s.iterrows()}
    e.index = [m[tuple(i)] for i in e.index]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fev-clone", default=os.environ.get("FEV_CLONE", ""))
    ap.add_argument("--models", default="alma_ssm_s0")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "fev_full", "agg"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = load_lb(a.fev_clone)
    res = {}
    if a.check:   # published full-100 numbers (README / papers): reproduce before trusting anything else
        for metric in ("SQL", "MASE", "WQL"):
            r = lb(L, metric, n_res=None)
            res.setdefault("check_full100", {})[metric] = r.loc[FOCUS, ["skill_score", "win_rate"]].round(2).to_dict()
        print(json.dumps(res["check_full100"]["SQL"], indent=1))
    val, test = split_sets()
    sn = L[L.model_name == BASE].drop_duplicates("task_name")
    tag = {r.task_name: ("VAL" if r.dataset_config in val else "TEST" if r.dataset_config in test else "other")
           for r in sn.itertuples()}
    subsets = {"nontest52": [t for t, g in tag.items() if g != "TEST"],
               "nonval22": [t for t, g in tag.items() if g == "other"],
               "val30": [t for t, g in tag.items() if g == "VAL"]}
    assert len(subsets["nontest52"]) == 52 and len(subsets["nonval22"]) == 22 and len(subsets["val30"]) == 30
    alma = {m: load_alma(m, "Alma" + ("" if len(a.models.split(",")) == 1 else f" ({m.split('_')[-1]})"))
            for m in a.models.split(",")}
    for m, df in alma.items():
        bad = set(df.task_name) - set(subsets["nontest52"])
        if bad:
            sys.exit(f"{m}: summaries for non-subset tasks {sorted(bad)}")
        miss = set(subsets["nontest52"]) - set(df.task_name)
        print(f"{m}: {len(df)} task summaries, missing {sorted(miss)}")
        nf = df[["task_name", "n_nonfinite"]][df.n_nonfinite > 0]
        if len(nf):
            print("  non-finite forecasts replaced by 0:", nf.to_dict("records"))
    for sub, tasks in subsets.items():
        base = L[L.task_name.isin(tasks)]
        for m, df in alma.items():
            name = df.model_name.iloc[0]
            full = pd.concat([base, df[df.task_name.isin(tasks)]], ignore_index=True)
            for metric in ("SQL", "WQL", "MASE"):
                r = lb(full, metric)
                pw = pairwise(full, metric, name, FOCUS)
                res.setdefault(sub, {}).setdefault(m, {})[metric] = {
                    "leaderboard": r.round(3).reset_index().to_dict("records"),
                    "pairwise_vs": pw.round(2).reset_index().to_dict("records")}
            if m == list(alma)[0]:
                rel = {metric: rel_table(full, metric) for metric in ("SQL", "WQL", "MASE")}
                for metric, e in rel.items():
                    e.to_csv(os.path.join(a.out, f"rel_{metric}_{sub}.csv"))
    json.dump(res, open(os.path.join(a.out, "agg.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "agg.json"))


if __name__ == "__main__":
    main()
