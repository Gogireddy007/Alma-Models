"""
Leak-free benchmark protocol: separate TRAIN / VALIDATION / TEST data by *source dataset family*.
See docs/protocol.md for the rationale, citations and the per-model exposure table.

    TRAIN  our model may learn from it (plus synthetic data). Chronos-style public corpora that the
           compared foundation models also trained on, so neither side gets a data advantage.
    VAL    model selection and all tuning. fev-bench tasks from families that are disjoint from
           TRAIN and from TEST. Some of them were seen by some foundation models in pretraining
           (marked in VAL_EXPOSURE), which is acceptable because val numbers are never reported as
           a zero-shot comparison.
    TEST   final, used once. fev-bench tasks whose source data no compared model (Chronos-2,
           Chronos-Bolt, TiRex, TiRex-2, TimesFM-2.5/3, Moirai-2, Toto-1/2, Timestral) reports as
           training data at any frequency, and that share no source with TRAIN or VAL.

Windows follow the fev-bench task definitions exactly (horizon, num_windows, initial_cutoff,
window_step_size, seasonality, target columns; fev commit f323f6c, dataset revision pinned below),
with two documented deviations that keep it laptop-sized: at most MAX_FORECASTS_PER_TASK forecasts
per task (a fixed random subset of series, every window kept) and contexts cropped to the last
MAX_CONTEXT points. Every model is scored on exactly the same contexts and targets.

Metric: prepare.py's own scoring (WQL over 9 quantiles, MASE of the median, 10-90% coverage),
each divided by Seasonal Naive on the same windows, geometric mean over tasks (headline) and
over families (robustness check).

Usage (from the repo root; nothing heavy, CPU only):
    uv run tools/bench.py audit                       # disjointness + exposure table, no download
    uv run tools/bench.py download --split val        # ~42 MB   (train: --split train [--extra])
    uv run tools/bench.py summary --split val         # tasks, series, windows
    uv run tools/bench.py smoke                       # naive/seasonal-naive on 3 small val tasks
    uv run tools/bench.py baselines --split val auto_ets chronos2
    TSAR_SPLIT=test uv run tools/bench.py baselines --split test ...     # confirmatory, once

In code:
    import tools.bench as B          (or sys.path.insert(0, "tools"); import bench as B)
    B.load_train(); B.load_val(); B.load_test()      # load_test needs TSAR_SPLIT=test
    res = B.evaluate(predict_fn, "val")              # predict_fn(contexts, h, season, freq) -> [n, 9, h]
    B.compare(res_a["per_dataset"], res_b["per_dataset"])   # paired bootstrap, sign test
"""

import os
import sys
import json
import math
import time
import zlib
import hashlib
import argparse
import warnings
import subprocess

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import prepare as P  # noqa: E402  (metric functions, QUANTILES, fill_nans; prepare.py is not modified)

# ---------------------------------------------------------------------------
# Constants (fixed; change only with a dated note in docs/protocol.md, never after seeing test)
# ---------------------------------------------------------------------------

SEED = 20261029
MAX_CONTEXT = 2048              # every model sees the same last <= 2048 points
MAX_FORECASTS_PER_TASK = 1000   # series subsampled (fixed seed) so that n_series * windows <= this
MIN_CONTEXT = 2                 # seasonal naive / MASE scale need at least 2 points
BENCH_DIR = os.path.join(ROOT, "data", "bench")
TEST_LOG = os.path.join(BENCH_DIR, "test_access.log")

FEV_REPO, FEV_REV = "autogluon/fev_datasets", "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
CHRONOS_REPO, CHRONOS_REV = "autogluon/chronos_datasets", "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"

# fev families whose configs live in <family>/<suffix>/ on the hub
_SUBDIR_FAMILIES = {
    "ETT", "LOOP_SEATTLE", "M_DENSE", "SZ_TAXI", "bizitobs_l2c", "boomlet", "entsoe",
    "favorita_stores", "hierarchical_sales", "hospital_admissions", "jena_weather", "kdd_cup_2022",
    "redset", "rohlik_orders", "rohlik_sales", "rossmann", "solar", "solar_with_weather",
    "uci_air_quality", "uk_covid_nation", "uk_covid_utla", "us_consumption",
}

# (task, family, fev config, horizon, num_windows, initial_cutoff, step, seasonality, target, min_ctx)
# Generated from the fev-bench result files (github.com/autogluon/fev, benchmarks/fev_bench/results,
# commit f323f6c). target "target_*:N" means the N columns target_0 .. target_{N-1}.
VAL_TASKS = [  # 30 tasks
    ('ETT_15T', 'ETT', 'ETT_15T', 96, 20, -1920, 96, 96, ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT'], 1),
    ('ETT_1D', 'ETT', 'ETT_1D', 28, 20, -560, 28, 7, ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT'], 1),
    ('ETT_1H', 'ETT', 'ETT_1H', 168, 20, -3360, 168, 24, ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT'], 1),
    ('ETT_1W', 'ETT', 'ETT_1W', 13, 5, -65, 13, 1, ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT'], 1),
    ('LOOP_SEATTLE_1D', 'LOOP_SEATTLE', 'LOOP_SEATTLE_1D', 28, 10, -280, 28, 7, 'target', 1),
    ('LOOP_SEATTLE_1H', 'LOOP_SEATTLE', 'LOOP_SEATTLE_1H', 168, 10, -1680, 168, 24, 'target', 1),
    ('M_DENSE_1D', 'M_DENSE', 'M_DENSE_1D', 28, 10, -280, 28, 7, 'target', 1),
    ('M_DENSE_1H', 'M_DENSE', 'M_DENSE_1H', 168, 10, -1680, 168, 24, 'target', 1),
    ('SZ_TAXI_15T', 'SZ_TAXI', 'SZ_TAXI_15T', 96, 10, -960, 96, 96, 'target', 1),
    ('SZ_TAXI_1H', 'SZ_TAXI', 'SZ_TAXI_1H', 168, 2, -336, 168, 24, 'target', 1),
    ('bizitobs_l2c_1H', 'bizitobs_l2c', 'bizitobs_l2c_1H', 24, 20, -480, 24, 24, 'target_*:7', 1),
    ('bizitobs_l2c_5T', 'bizitobs_l2c', 'bizitobs_l2c_5T', 288, 20, -5760, 288, 288, 'target_*:7', 1),
    ('favorita_stores_1D', 'favorita_stores', 'favorita_stores_1D', 28, 10, -280, 28, 7, 'sales', 1),
    ('favorita_stores_1M', 'favorita_stores', 'favorita_stores_1M', 12, 2, -24, 12, 12, 'sales', 1),
    ('favorita_stores_1W', 'favorita_stores', 'favorita_stores_1W', 13, 10, -130, 13, 1, 'sales', 1),
    ('hierarchical_sales_1D', 'hierarchical_sales', 'hierarchical_sales_1D', 28, 10, -280, 28, 7, 'target', 1),
    ('hierarchical_sales_1W', 'hierarchical_sales', 'hierarchical_sales_1W', 13, 10, -130, 13, 1, 'target', 1),
    ('jena_weather_10T', 'jena_weather', 'jena_weather_10T', 144, 20, -2880, 144, 144, 'target_*:21', 1),
    ('jena_weather_1D', 'jena_weather', 'jena_weather_1D', 28, 11, -308, 28, 1, 'target_*:21', 1),
    ('jena_weather_1H', 'jena_weather', 'jena_weather_1H', 24, 20, -480, 24, 24, 'target_*:21', 1),
    ('kdd_cup_2022_1D', 'kdd_cup_2022', 'kdd_cup_2022_1D', 14, 10, -140, 14, 1, 'Patv', 1),
    ('restaurant', 'restaurant', 'restaurant', 28, 8, -224, 28, 7, 'target', 1),
    ('solar_1D', 'solar', 'solar_1D', 28, 10, -280, 28, 1, 'target', 1),
    ('solar_1W', 'solar', 'solar_1W', 13, 1, -13, 13, 1, 'target', 1),
    ('uk_covid_nation_1D/cumulative', 'uk_covid_nation', 'uk_covid_nation_1D', 28, 20, -560, 28, 1, ['cumulative_admissions', 'cumulative_cases', 'cumulative_deaths'], 1),
    ('uk_covid_nation_1D/new', 'uk_covid_nation', 'uk_covid_nation_1D', 28, 20, -560, 28, 7, ['new_admissions', 'new_cases', 'new_deaths'], 1),
    ('uk_covid_nation_1W/cumulative', 'uk_covid_nation', 'uk_covid_nation_1W', 8, 4, -32, 8, 1, ['cumulative_admissions', 'cumulative_cases', 'cumulative_deaths'], 1),
    ('uk_covid_nation_1W/new', 'uk_covid_nation', 'uk_covid_nation_1W', 8, 4, -32, 8, 1, ['new_admissions', 'new_cases', 'new_deaths'], 1),
    ('uk_covid_utla_1D/new', 'uk_covid_utla', 'uk_covid_utla_1D', 28, 10, -280, 28, 7, 'new_cases', 1),
    ('uk_covid_utla_1W/cumulative', 'uk_covid_utla', 'uk_covid_utla_1W', 13, 5, -65, 13, 1, 'cumulative_cases', 1),
]
TEST_TASKS = [  # 48 tasks
    ('boomlet_1062', 'boomlet', 'boomlet_1062', 288, 20, -5760, 288, 288, 'target_*:21', 1),
    ('boomlet_1209', 'boomlet', 'boomlet_1209', 288, 20, -5760, 288, 288, 'target_*:53', 1),
    ('boomlet_1225', 'boomlet', 'boomlet_1225', 60, 20, -1200, 60, 1440, 'target_*:49', 1),
    ('boomlet_1230', 'boomlet', 'boomlet_1230', 288, 20, -5760, 288, 288, 'target_*:23', 1),
    ('boomlet_1282', 'boomlet', 'boomlet_1282', 60, 20, -1200, 60, 1440, 'target_*:35', 1),
    ('boomlet_1487', 'boomlet', 'boomlet_1487', 288, 20, -5760, 288, 288, 'target_*:54', 1),
    ('boomlet_1631', 'boomlet', 'boomlet_1631', 96, 20, -1920, 96, 48, 'target_*:40', 1),
    ('boomlet_1676', 'boomlet', 'boomlet_1676', 96, 20, -1920, 96, 48, 'target_*:100', 1),
    ('boomlet_1855', 'boomlet', 'boomlet_1855', 24, 20, -480, 24, 24, 'target_*:52', 1),
    ('boomlet_1975', 'boomlet', 'boomlet_1975', 24, 20, -480, 24, 24, 'target_*:75', 1),
    ('boomlet_2187', 'boomlet', 'boomlet_2187', 24, 20, -480, 24, 24, 'target_*:100', 1),
    ('boomlet_285', 'boomlet', 'boomlet_285', 60, 20, -1200, 60, 1440, 'target_*:75', 1),
    ('boomlet_619', 'boomlet', 'boomlet_619', 60, 20, -1200, 60, 1440, 'target_*:52', 1),
    ('boomlet_772', 'boomlet', 'boomlet_772', 60, 20, -1200, 60, 1440, 'target_*:67', 1),
    ('boomlet_963', 'boomlet', 'boomlet_963', 60, 20, -1200, 60, 1440, 'target_*:28', 1),
    ('ecdc_ili', 'ecdc_ili', 'ecdc_ili', 13, 10, -130, 13, 1, 'target', 1),
    ('entsoe_15T', 'entsoe', 'entsoe_15T', 96, 20, -1920, 96, 96, 'target', 1),
    ('entsoe_1H', 'entsoe', 'entsoe_1H', 168, 20, -3360, 168, 24, 'target', 1),
    ('entsoe_30T', 'entsoe', 'entsoe_30T', 96, 20, -1920, 96, 48, 'target', 1),
    ('epf_be', 'epf', 'epf_be', 24, 20, -480, 24, 24, 'target', 1),
    ('epf_de', 'epf', 'epf_de', 24, 20, -480, 24, 24, 'target', 1),
    ('epf_fr', 'epf', 'epf_fr', 24, 20, -480, 24, 24, 'target', 1),
    ('epf_np', 'epf', 'epf_np', 24, 20, -480, 24, 24, 'target', 1),
    ('epf_pjm', 'epf', 'epf_pjm', 24, 20, -480, 24, 24, 'target', 1),
    ('gvar', 'gvar', 'gvar', 8, 10, -80, 8, 4, ['Dp', 'ep', 'eq', 'lr', 'r', 'y'], 1),
    ('hermes', 'hermes', 'hermes', 52, 1, -52, 52, 1, 'target', 1),
    ('hospital_admissions_1D', 'hospital_admissions', 'hospital_admissions_1D', 28, 20, -560, 28, 7, 'target', 1),
    ('hospital_admissions_1W', 'hospital_admissions', 'hospital_admissions_1W', 13, 16, -208, 13, 1, 'target', 1),
    ('redset_15T', 'redset', 'redset_15T', 96, 10, -960, 96, 96, 'target', 1),
    ('redset_1H', 'redset', 'redset_1H', 24, 10, -240, 24, 24, 'target', 1),
    ('redset_5T', 'redset', 'redset_5T', 288, 10, -2880, 288, 288, 'target', 1),
    ('rohlik_orders_1D', 'rohlik_orders', 'rohlik_orders_1D', 61, 5, '2023-05-01', 61, 7, 'orders', 1),
    ('rohlik_orders_1W', 'rohlik_orders', 'rohlik_orders_1W', 8, 5, '2023-05-01', 8, 1, 'orders', 1),
    ('rohlik_sales_1D', 'rohlik_sales', 'rohlik_sales_1D', 14, 1, '2023-12-15', 14, 7, 'sales', 14),
    ('rohlik_sales_1W', 'rohlik_sales', 'rohlik_sales_1W', 8, 1, '2023-12-15', 8, 1, 'sales', 8),
    ('rossmann_1D', 'rossmann', 'rossmann_1D', 48, 10, -480, 48, 7, 'Sales', 1),
    ('rossmann_1W', 'rossmann', 'rossmann_1W', 13, 8, -104, 13, 1, 'Sales', 1),
    ('solar_with_weather_15T', 'solar_with_weather', 'solar_with_weather_15T', 96, 20, -1920, 96, 96, 'target', 1),
    ('solar_with_weather_1H', 'solar_with_weather', 'solar_with_weather_1H', 24, 20, -480, 24, 24, 'target', 1),
    ('uci_air_quality_1D', 'uci_air_quality', 'uci_air_quality_1D', 28, 11, -308, 28, 7, ['C6H6(GT)', 'CO(GT)', 'NO2(GT)', 'NOx(GT)'], 1),
    ('uci_air_quality_1H', 'uci_air_quality', 'uci_air_quality_1H', 168, 20, -3360, 168, 24, ['C6H6(GT)', 'CO(GT)', 'NO2(GT)', 'NOx(GT)'], 1),
    ('us_consumption_1M', 'us_consumption', 'us_consumption_1M', 12, 10, -120, 12, 12, 'target', 1),
    ('us_consumption_1Q', 'us_consumption', 'us_consumption_1Q', 8, 10, -80, 8, 4, 'target', 1),
    ('us_consumption_1Y', 'us_consumption', 'us_consumption_1Y', 5, 10, -50, 5, 1, 'target', 1),
    ('walmart', 'walmart', 'walmart', 39, 1, -39, 39, 1, 'target', 1),
    ('world_co2_emissions', 'world_co2_emissions', 'world_co2_emissions', 5, 9, -45, 5, 1, 'target', 1),
    ('world_life_expectancy', 'world_life_expectancy', 'world_life_expectancy', 5, 10, -50, 5, 1, 'target', 1),
    ('world_tourism', 'world_tourism', 'world_tourism', 5, 2, -10, 5, 1, 'target', 1),
]

# Families that may never appear in VAL or TEST because TRAIN (or a compared model) contains the
# same source data, possibly at another frequency. Documented in docs/protocol.md section 3.
EXCLUDED_FAMILIES = {
    "ercot": "in TRAIN (Chronos BII ercot, same neuralprophet source)",
    "m5": "in TRAIN (Chronos BII m5); fev flags Moirai-2/TimesFM-2.5/Toto-1",
    "hospital": "in TRAIN (monash_hospital, same data)",
    "fred_md_2025": "fev flags Moirai-2/TimesFM-2.5/Toto-1; FRED-MD also in prepare.py",
    "fred_qd_2025": "same FRED-McCracken source as fred_md_2025 (leaked at monthly frequency)",
    "favorita_transactions": "fev flags Moirai-2/TimesFM-2.5/Toto-1",
    "proenfo_gfc12": "fev flags Moirai-2/TimesFM-2.5/Toto-1",
    "proenfo_gfc14": "fev flags Moirai-2/TimesFM-2.5/Toto-1",
    "proenfo_gfc17": "fev flags Moirai-2/TimesFM-2.5/Toto-1",
    "australian_tourism": "Tourism Australia data, may overlap monash_tourism_* in TRAIN",
    "LOOP_SEATTLE_5T": "209 MB download; 1D/1H variants kept in VAL",
}

# Known pretraining exposure of VAL families (acceptable for model selection, never reported as
# zero-shot). TEST families have none by construction (see docs/protocol.md).
VAL_EXPOSURE = {
    "ETT": "Moirai-2 (fev flag)", "LOOP_SEATTLE": "Moirai-2 (fev flag)",
    "M_DENSE": "Moirai-2 (fev flag)", "SZ_TAXI": "Moirai-2 (fev flag)",
    "bizitobs_l2c": "Moirai-2 (fev flag)", "jena_weather": "Moirai-2 (fev flag)",
    "hierarchical_sales": "Moirai-2 (fev flag)", "restaurant": "Moirai-2 (fev flag)",
    "kdd_cup_2022": "10T variant: Moirai-2, TiRex, TimesFM-2.5, Toto-1 (fev flags); "
                    "TiRex likely (GIFT pretrain 'kdd2022', 2505.23719 Table 6) [eval_contract.md s4]",
    "favorita_stores": "same Kaggle source as favorita_transactions (flagged for 3 models); "
                       "TiRex possible ('favorita sales', item level, same Kaggle source) [eval_contract.md s4]",
    "solar": "NREL solar, same source as Chronos/Chronos-2 'solar' training data: Chronos-2 likely "
             "(2510.15821 Table 6), Chronos-Bolt likely (Chronos corpus, 2403.07815 App. B), "
             "TiRex likely (2505.23719 Table 5) [eval_contract.md s4]",
    "uk_covid_nation": "none known", "uk_covid_utla": "none known",
}

# --- TRAIN corpus -----------------------------------------------------------
# (name, season, freq, default horizon). Local = already in data/raw from prepare.py (0 download).
TRAIN_LOCAL = [
    ("monash_tourism_monthly", 12, "M", 24), ("monash_tourism_quarterly", 4, "Q", 8),
    ("monash_m1_monthly", 12, "M", 18), ("monash_m3_monthly", 12, "M", 18),
    ("monash_m3_quarterly", 4, "Q", 8), ("m4_quarterly", 4, "Q", 8),
    ("monash_cif_2016", 12, "M", 12), ("monash_hospital", 12, "M", 12),
    ("monash_car_parts", 12, "M", 12), ("nn5", 7, "D", 56), ("monash_nn5_weekly", 1, "W", 8),
    ("monash_weather", 1, "D", 30), ("m5", 7, "D", 28), ("dominick", 1, "W", 8),
    ("ercot", 24, "H", 24), ("monash_traffic", 24, "H", 24),
    ("monash_australian_electricity", 48, "30min", 48),
]
# prepare.py datasets deliberately NOT in TRAIN, with the VAL/TEST family they could leak into.
TRAIN_EXCLUDED = {
    "monash_fred_md": "FRED macro panel; overlaps us_consumption (BEA PCE) and gvar macro series",
    "exchange_rate": "daily FX 1990-2016; overlaps gvar 'ep' (exchange rates) in TEST",
    "monash_covid_deaths": "2020 COVID panel incl. UK; overlaps uk_covid (VAL) / ecdc_ili period",
}
# Optional extension (uv run tools/bench.py download --split train --extra, ~246 MB): the public
# part of the Chronos / Chronos-2 / TiRex pretraining corpus, so we train on what they trained on.
TRAIN_EXTRA = [
    ("m4_daily", 7, "D", 14), ("m4_hourly", 24, "H", 48), ("m4_monthly", 12, "M", 18),
    ("m4_weekly", 1, "W", 13), ("m4_yearly", 1, "Y", 6), ("monash_m1_quarterly", 4, "Q", 8),
    ("monash_m1_yearly", 1, "Y", 6), ("monash_m3_yearly", 1, "Y", 6),
    ("monash_tourism_yearly", 1, "Y", 4), ("monash_electricity_hourly", 24, "H", 48),
    ("monash_electricity_weekly", 1, "W", 8), ("monash_kdd_cup_2018", 24, "H", 48),
    ("monash_pedestrian_counts", 24, "H", 48), ("monash_rideshare", 24, "H", 48),
    ("monash_saugeenday", 1, "D", 30), ("monash_temperature_rain", 1, "D", 30),
    ("taxi_30min", 48, "30min", 48), ("uber_tlc_daily", 7, "D", 14), ("uber_tlc_hourly", 24, "H", 48),
    ("wind_farms_daily", 1, "D", 30), ("wind_farms_hourly", 24, "H", 48),
]

TASK_FIELDS = ("name", "family", "config", "horizon", "num_windows", "cutoff", "step", "season",
               "target", "min_context")
SPLIT_TASKS = {"val": [dict(zip(TASK_FIELDS, t)) for t in VAL_TASKS],
               "test": [dict(zip(TASK_FIELDS, t)) for t in TEST_TASKS]}
_TEST_CONFIGS = {t["config"] for t in SPLIT_TASKS["test"]}

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


TEST_MODEL_SCRIPTS = {"train.py", "train_v2.py"}   # the only scripts with a frozen-checkpoint eval path (audit_v3 B-1)


def _guard(split, what):
    assert split in ("val", "test"), split
    if int(os.environ.get("TSAR_TTA_STEPS", 0) or 0) > 0:     # audit_v3 H-1: shared-weight TTA leaks
        raise PermissionError(f"{what}: TSAR_TTA_STEPS>0 adapts on windows whose contexts hold "
                              "other windows' targets; not allowed in bench evaluation.")
    if split == "test":
        if os.environ.get("TSAR_SPLIT") != "test":
            raise PermissionError(f"{what}: the test split is reserved for the single confirmatory "
                                  "run. Set TSAR_SPLIT=test to use it (see docs/protocol.md).")
        script = os.path.basename(sys.argv[0])
        if script not in ("bench.py", "overlap_scan.py"):     # a model script, not baselines or the lag scan
            ckpt, sha = os.environ.get("TSAR_EVAL_ONLY"), os.environ.get("TSAR_CKPT_SHA256")
            if script not in TEST_MODEL_SCRIPTS or not ckpt or not sha or not os.path.exists(ckpt) \
                    or _file_sha256(ckpt) != sha:
                raise PermissionError(f"{what}: TEST needs {sorted(TEST_MODEL_SCRIPTS)} with TSAR_EVAL_ONLY="
                                      "<frozen ckpt> and a matching TSAR_CKPT_SHA256 (audit_v3 B-1).")
        os.makedirs(BENCH_DIR, exist_ok=True)
        with open(TEST_LOG, "a") as f:   # audit trail: every test access is recorded
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{what}\tgit={_git_head()}\t"
                    f"argv={' '.join(sys.argv)}\n")


def _guard_config(config):
    """Second line of defence (audit_v2 L-3): reading a TEST parquet through _build_task needs
    TSAR_SPLIT=test even if the caller skipped _guard. Check only; _guard does the logging."""
    if config in _TEST_CONFIGS and os.environ.get("TSAR_SPLIT") != "test":
        raise PermissionError(f"_build_task({config}): TEST data; set TSAR_SPLIT=test (see docs/protocol.md)")


# TSAR_BENCH_EXTRA guard (audit_v2 L-2): TRAIN_EXTRA may be trained on only after a content scan
# against VAL and TEST found nothing that TRAIN_DROP does not already remove. tools/overlap_scan.py
# stamp-extra writes this file only when that scan is clean; the key binds it to the exact extra
# files, the TRAIN_DROP list and the scan settings, so any change makes the stamp stale.
EXTRA_STAMP = os.path.join(BENCH_DIR, "overlap_extra.json")


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extra_scan_key():
    """sha256 over TRAIN_LOCAL, TRAIN_EXTRA (with each extra file's sha256) and TRAIN_DROP."""
    files = {}
    for name, *_ in TRAIN_EXTRA:
        p = os.path.join(BENCH_DIR, "train", f"{name}.parquet")
        files[name] = _file_sha256(p) if os.path.exists(p) else None
    blob = json.dumps({"local": TRAIN_LOCAL, "extra": TRAIN_EXTRA, "files": files,
                       "drop": {k: sorted(v) for k, v in TRAIN_DROP.items()}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def require_extra_stamp():
    if not os.path.exists(EXTRA_STAMP):
        raise PermissionError(
            "TRAIN_EXTRA was never overlap-scanned against VAL/TEST (audit_v2 L-2). Download it, then run "
            "`TSAR_SPLIT=test uv run tools/overlap_scan.py stamp-extra` (target-free on TEST, logged); "
            f"it writes {os.path.relpath(EXTRA_STAMP, ROOT)} only if no unremoved overlap remains.")
    st = json.load(open(EXTRA_STAMP))
    if st.get("result") != "clean" or st.get("key") != extra_scan_key():
        raise PermissionError(f"{os.path.relpath(EXTRA_STAMP, ROOT)} is stale or not clean (TRAIN_EXTRA files, "
                              "TRAIN_DROP or the lists changed since the scan). Re-run tools/overlap_scan.py stamp-extra.")

# ---------------------------------------------------------------------------
# Download (pinned revisions, sha256 manifest)
# ---------------------------------------------------------------------------

def _fev_path(config):
    for fam in sorted(_SUBDIR_FAMILIES, key=len, reverse=True):
        if config.startswith(fam + "_"):
            return f"{fam}/{config[len(fam) + 1:]}/train-00000-of-00001.parquet"
    return f"{config}/train-00000-of-00001.parquet"


def _fetch(repo, rev, path, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = HF_FILE.format(repo=repo, rev=rev, path=path)
    for attempt in range(5):
        try:
            with requests.get(url, timeout=120, stream=True) as r:
                r.raise_for_status()
                h = hashlib.sha256()
                with open(dest + ".tmp", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
            os.replace(dest + ".tmp", dest)
            man_path = os.path.join(BENCH_DIR, "manifest.json")
            man = json.load(open(man_path)) if os.path.exists(man_path) else {}
            man[os.path.relpath(dest, BENCH_DIR)] = {"repo": repo, "rev": rev, "path": path,
                                                     "sha256": h.hexdigest(),
                                                     "bytes": os.path.getsize(dest)}
            json.dump(man, open(man_path, "w"), indent=1, sort_keys=True)
            return dest
        except Exception as e:  # network hiccups
            print(f"  download of {path} failed ({e}), retrying")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {repo}/{path}")


def _fev_file(config):
    return _fetch(FEV_REPO, FEV_REV, _fev_path(config), os.path.join(BENCH_DIR, "fev", f"{config}.parquet"))


def _train_file(name, extra):
    if not extra:
        return P._download(name)   # data/raw/<name>.parquet, shared with prepare.py
    # chronos_datasets: some large sets are sharded; all extras here are single files
    return _fetch(CHRONOS_REPO, CHRONOS_REV, f"{name}/train-00000-of-00001.parquet",
                  os.path.join(BENCH_DIR, "train", f"{name}.parquet"))


def download(split, extra=False):
    if split == "train":
        for name, *_ in TRAIN_LOCAL:
            _train_file(name, False)
        if extra:
            for name, *_ in TRAIN_EXTRA:
                _train_file(name, True)
        return
    for cfg in sorted({t["config"] for t in SPLIT_TASKS[split]}):
        _fev_file(cfg)

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def _target_arrays(df):
    """chronos_datasets: one 'target' column, or several float sequence columns (all used)."""
    cols = ["target"] if "target" in df.columns else [
        c for c in df.columns if c not in ("id", "timestamp") and df[c].dtype == object]
    for c in cols:
        for v in df[c]:
            yield np.asarray(v, dtype=np.float32)


# TRAIN series removed because their content duplicates TEST data under another name (found
# 2026-10-29; method and evidence in docs/bench_integrity.md). {dataset: [series key]}, where the key
# is the position of the series in _target_arrays() order, before any subsampling. All are BEA / GVAR
# macro series that M3/M4 contain anonymously (other vintage or units): max correlation of first
# differences >= 0.95 over >= 60 aligned points with a TEST context of us_consumption_1M/1Q or gvar.
TRAIN_DROP = {
    "m4_quarterly": [4591, 4770, 5365, 10293, 13322, 13325, 13328, 13330, 15569, 15597, 15600,
                     18358, 19365],
    "monash_m3_monthly": [820, 825],
    "monash_m3_quarterly": [660, 666, 685],
    # TRAIN_EXTRA residuals of overlap_scan.py stamp-extra (2026-10-30, runs/overlap_extra_stamp.log, 806 hits;
    # review in docs/extra_overlap_review.md). Conservative policy: every flagged key is dropped, including
    # the probable false positives (wind_farms/m4_daily vs boomlet, m3_yearly vs world_life_expectancy).
    # TEST hash (znorm, L=32) and TEST lag (diff corr >= 0.95, n >= 24 yearly / 60 otherwise) hits.
    "m4_daily": [4221, 4224],
    "m4_monthly": [6216, 7530, 7535, 7721, 8431, 8451, 8452, 44134, 44135, 44137],
    "m4_yearly": [15, 51, 53, 54, 55, 188, 251, 292, 363, 364, 440, 468, 488, 499, 509, 512, 521, 534, 544,
                  581, 1097, 1103, 1107, 1116, 1120, 1126, 1128, 1158, 1166, 1257, 1281, 1335, 1350, 1366,
                  1373, 1676, 1699, 1700, 1743, 1857, 2023, 2097, 2100, 2296, 2356, 2450, 2929, 2984, 3054,
                  3065, 3156, 3387, 3563, 3583, 3595, 3611, 3621, 3624, 3644, 3651, 3657, 3665, 3666, 3725,
                  3922, 3961, 3991, 3992, 4040, 4093, 4108, 4120, 4124, 4127, 4132, 4163, 6112, 7544, 7552,
                  7554, 7585, 7607, 7611, 7644, 7667, 7691, 7730, 7756, 7765, 7769, 7781, 8326, 8511, 9053,
                  9106, 9112, 9252, 9261, 9373, 9644, 9650, 10048, 10150, 10156, 10276, 10315, 10408, 10442,
                  10452, 10454, 10458, 10460, 10468, 10473, 10478, 10487, 10501, 10505, 10506, 10508, 10509,
                  10510, 10511, 10527, 10529, 10530, 10715, 10716, 10722, 10772, 10773, 10774, 10775, 10776,
                  10777, 10778, 10779, 10780, 10781, 10782, 10783, 10784, 10785, 10786, 10787, 10788, 10791,
                  10792, 10936, 10937, 10940, 10941, 10944, 10948, 10949, 10963, 10983, 10986, 10988, 10990,
                  10991, 11396, 11397, 11399, 11403, 11530, 11584, 11635, 11666, 11668, 11688, 11738, 11740,
                  11741, 11742, 11745, 11746, 11747, 11748, 11752, 11753, 11837, 11838, 11870, 11871, 11939,
                  12040, 12178, 12282, 12283, 12284, 12285, 12308, 12365, 12384, 12541, 12649, 13111, 13112,
                  13113, 13186, 13239, 13414, 13594, 13779, 13834, 13942, 13985, 14007, 14009, 14011, 14012,
                  14014, 14025, 14066, 14119, 14120, 14121, 14125, 14189, 14194, 14195, 14201, 14270, 14290,
                  14446, 14547, 14549, 14574, 14580, 14584, 14585, 14727, 14918, 15136, 15212, 15259, 15262,
                  15351, 15754, 15755, 15757, 15759, 15774, 15777, 15782, 15787, 15803, 15806, 15813, 15816,
                  15818, 15824, 15825, 15827, 15828, 15833, 15976, 16216, 16358, 16359, 16725, 16726, 16727,
                  16728, 16743, 16746, 16747, 16748, 16762, 16763, 16938, 17100, 17231, 17256, 17306, 17322,
                  17399, 17400, 17409, 17430, 17449, 17451, 17499, 17504, 17523, 17530, 17538, 17545, 17565,
                  17567, 17568, 17657, 17662, 17774, 17793, 17796, 17812, 17826, 17946, 17982, 18018, 18108,
                  18130, 18165, 18169, 18185, 18370, 18487, 18494, 18546, 18563, 18564, 18718, 18741, 18870,
                  18871, 19328, 19351, 19416, 19420, 19422, 19542, 19620, 19691, 19769, 19956, 19959, 19964,
                  20174, 20334, 20723, 20791, 20980, 21379, 21411, 21455, 21506, 21519, 21629, 22376, 22430,
                  22452, 22459, 22512, 22538, 22548],
    "monash_m3_yearly": [348, 472, 473, 474, 475, 490, 530, 535, 559, 588],
    "wind_farms_hourly": [18, 100, 143, 218, 219],
}


def _train_specs(extra=False):
    return [(s, False) for s in TRAIN_LOCAL] + ([(s, True) for s in TRAIN_EXTRA] if extra else [])


# TSAR_GIFT_CLEAN=1 (default off; Tanaka 2026-10-30, docs/gift_clean_corpus.md): TRAIN without any GIFT-Eval
# test dataset, so a GIFT-Eval number can be submitted as zero-shot with testdata_leakage: No. Verified against
# the 28 folders of Salesforce/GiftEval @ 3084173 (HF API) and docs/benchmark_contract.md section 2.
# Applied in load_train only. _train_specs, extra_scan_key and the overlap scans stay on the full lists, so a
# stamp is always made on (and valid for) the full TRAIN_EXTRA and the flag cannot weaken the stamp guard.
GIFT_TEST_OVERLAP = {
    "m4_quarterly": "exact: GIFT m4_quarterly",
    "monash_hospital": "exact: GIFT hospital",
    "monash_car_parts": "exact: GIFT car_parts_with_missing",
    "m4_daily": "exact: GIFT m4_daily", "m4_hourly": "exact: GIFT m4_hourly",
    "m4_monthly": "exact: GIFT m4_monthly", "m4_weekly": "exact: GIFT m4_weekly",
    "m4_yearly": "exact: GIFT m4_yearly",
    "monash_kdd_cup_2018": "exact: GIFT kdd_cup_2018_with_missing",
    "monash_saugeenday": "exact: GIFT saugeenday",
    "monash_temperature_rain": "exact: GIFT temperature_rain_with_missing",
    "monash_electricity_hourly": "related: GIFT electricity (same UCI ElectricityLoadDiagrams clients)",
    "monash_electricity_weekly": "related: GIFT electricity/W (same UCI source, weekly aggregate)",
}
assert set(GIFT_TEST_OVERLAP) <= {n for n, *_ in TRAIN_LOCAL + TRAIN_EXTRA}


def gift_clean():
    return os.environ.get("TSAR_GIFT_CLEAN", "0").strip() not in ("", "0")


# --- TSAR_BENCH_GEP=1: GiftEvalPretrain subset (default off; Tanaka 2026-10-30, docs/gift_clean_corpus.md s5) ---
# 25 folders of Salesforce/GiftEvalPretrain @ GEP_REV, one HF save_to_disk arrow shard each, downloaded by the lead
# to data/bench/gep/ with a sha256 manifest (pinned below). GIFT built this corpus without GIFT-Eval test data; the
# residual risk is against our fev VAL/TEST, so training on it needs a clean `overlap_scan.py stamp-gep`
# (data/bench/overlap_gep.json, keyed by gep_scan_key()). Read memory-mapped with pyarrow; multivariate targets are
# split into univariate series (key = position in row-major (row, variate) order of the whole shard, before any
# cap). GEP series are appended after LOCAL/EXTRA with their own rng, so the LOCAL/EXTRA part is unchanged.
GEP_REPO = "Salesforce/GiftEvalPretrain"
GEP_REV = "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874"
GEP_DIR = os.path.join(BENCH_DIR, "gep")
GEP_MANIFEST = os.path.join(GEP_DIR, "manifest.json")
GEP_MANIFEST_SHA256 = "abe5ed94efcab05d1168a548c4c05cae3fa50c7d48b13785ead05083daaf3f7a"
GEP_BYTES = 1_581_673_939
GEP_STAMP = os.path.join(BENCH_DIR, "overlap_gep.json")
# (folder, training dataset, season, freq, default horizon). Folders of one domain and frequency share a training
# dataset name, so the 25 folders enter the dataset-uniform samplers as 9 datasets, not 25.
TRAIN_GEP = [
    ("PEMS03", "gep_traffic_5min", 288, "5min", 288), ("PEMS04", "gep_traffic_5min", 288, "5min", 288),
    ("PEMS07", "gep_traffic_5min", 288, "5min", 288), ("PEMS08", "gep_traffic_5min", 288, "5min", 288),
    ("PEMS_BAY", "gep_traffic_5min", 288, "5min", 288), ("LOS_LOOP", "gep_traffic_5min", 288, "5min", 288),
    ("Q-TRAFFIC", "gep_qtraffic_15min", 96, "15min", 96),
    ("SHMETRO", "gep_metro_15min", 96, "15min", 96), ("HZMETRO", "gep_metro_15min", 96, "15min", 96),
    ("BEIJING_SUBWAY_30MIN", "gep_metro_30min", 48, "30min", 48),
    ("london_smart_meters_with_missing", "gep_london_30min", 48, "30min", 48),
    ("alibaba_cluster_trace_2018", "gep_cloud_5min", 288, "5min", 288),
    ("bdg-2_bear", "gep_buildings_H", 24, "H", 48), ("bdg-2_fox", "gep_buildings_H", 24, "H", 48),
    ("bdg-2_panther", "gep_buildings_H", 24, "H", 48), ("bdg-2_rat", "gep_buildings_H", 24, "H", 48),
    ("hog", "gep_buildings_H", 24, "H", 48), ("bull", "gep_buildings_H", 24, "H", 48),
    ("cockatoo", "gep_buildings_H", 24, "H", 48), ("ideal", "gep_buildings_H", 24, "H", 48),
    ("smart", "gep_buildings_H", 24, "H", 48), ("borealis", "gep_buildings_H", 24, "H", 48),
    ("sceaux", "gep_buildings_H", 24, "H", 48),
    ("china_air_quality", "gep_air_H", 24, "H", 48), ("oikolab_weather", "gep_weather_H", 24, "H", 48),
]
# Caps for training (the overlap scan always reads every full series): a random contiguous crop of at most
# GEP_MAX_LEN points per univariate series (also <= load_train's max_len), and per folder at most GEP_MAX_SERIES
# series and GEP_MAX_POINTS points, chosen in a fixed per-folder random order.
GEP_MAX_LEN = 4096
GEP_MAX_SERIES = 2000
GEP_MAX_POINTS = 4_000_000
# GEP series removed after review of stamp-gep hits: {folder: [key]}. Kept apart from TRAIN_DROP so that GEP drops
# never make the TRAIN_EXTRA stamp (extra_scan_key hashes TRAIN_DROP) stale.
GEP_DROP = {
    # stamp-gep 2026-10-31 (docs/gep_overlap_review.md): the only 3 of 1272 lag hits whose moving steps still agree
    # (winsorised r >= 0.8 over >= min_overlap moving steps). All 3 are at n = 64-65 vs boomlet H tasks and look like
    # shared diurnal shape, not copies; they fail the strict re-test, and are dropped anyway for margin.
    "bull": [33],                 # ~ boomlet_1975/target_3:  n=65, r_wins 0.961, but level r 0.07
    "ideal": [40],                # ~ boomlet_1855/target_28: n=64, r_wins 0.904, level r 0.962
    "china_air_quality": [1377],  # ~ boomlet_1855/target_34: n=64, r_wins 0.815, level r 0.870
}
assert len({f for f, *_ in TRAIN_GEP}) == len(TRAIN_GEP) and set(GEP_DROP) <= {f for f, *_ in TRAIN_GEP}
assert not {d for _, d, *_ in TRAIN_GEP} & {n for n, *_ in TRAIN_LOCAL + TRAIN_EXTRA}


def gep_on():
    return os.environ.get("TSAR_BENCH_GEP", "0").strip() not in ("", "0")


_GEP_VERIFIED = None


def gep_verify():
    """Check the pinned manifest and every file in it (sha256 and size). Returns {relpath: sha256}. Once per process."""
    global _GEP_VERIFIED
    if _GEP_VERIFIED is None:
        if not os.path.exists(GEP_MANIFEST):
            raise FileNotFoundError(f"{os.path.relpath(GEP_MANIFEST, ROOT)} missing: download the GEP subset first "
                                    "(docs/gift_clean_corpus.md section 5).")
        if _file_sha256(GEP_MANIFEST) != GEP_MANIFEST_SHA256:
            raise PermissionError(f"{os.path.relpath(GEP_MANIFEST, ROOT)} does not match GEP_MANIFEST_SHA256.")
        m = json.load(open(GEP_MANIFEST))
        assert m["repo"] == GEP_REPO and m["revision"] == GEP_REV, (m["repo"], m["revision"])
        assert sum(f["bytes"] for f in m["files"].values()) == GEP_BYTES
        for f, _, *_ in TRAIN_GEP:
            if len(_gep_shards(m, f)) != 1:
                raise PermissionError(f"GEP folder {f}: expected exactly one arrow shard in the manifest")
        for rel, meta in m["files"].items():
            p = os.path.join(GEP_DIR, rel)
            if not os.path.exists(p) or os.path.getsize(p) != meta["bytes"] or _file_sha256(p) != meta["sha256"]:
                raise PermissionError(f"GEP file {rel} is missing or does not match the manifest sha256.")
        _GEP_VERIFIED = {rel: meta["sha256"] for rel, meta in m["files"].items()}
    return _GEP_VERIFIED


def _gep_shards(manifest, folder):
    return sorted(r for r in manifest["files"] if r.startswith(folder + "/") and r.endswith(".arrow"))


def gep_scan_key():
    """sha256 over GEP_REV, TRAIN_GEP, the verified file sha256s and GEP_DROP (binds the stamp to the exact bytes)."""
    blob = json.dumps({"rev": GEP_REV, "gep": TRAIN_GEP, "files": gep_verify(),
                       "drop": {k: sorted(v) for k, v in GEP_DROP.items()}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def require_gep_stamp():
    if not os.path.exists(GEP_STAMP):
        raise PermissionError(
            "The GiftEvalPretrain subset (TSAR_BENCH_GEP) was never overlap-scanned against VAL/TEST. Run "
            "`TSAR_SPLIT=test uv run tools/overlap_scan.py stamp-gep` through jobq (runs/jobs/gep_scan.txt); "
            f"it writes {os.path.relpath(GEP_STAMP, ROOT)} only if no unremoved overlap remains.")
    st = json.load(open(GEP_STAMP))
    if st.get("result") != "clean" or st.get("key") != gep_scan_key():
        raise PermissionError(f"{os.path.relpath(GEP_STAMP, ROOT)} is stale or not clean (GEP files, TRAIN_GEP or "
                              "GEP_DROP changed since the scan). Re-run tools/overlap_scan.py stamp-gep.")


class _GepShard:
    """One memory-mapped shard: univariate series as float32 views into the map (copy what you keep).
    keys run over (row, variate) in row-major order; lengths[k] is the length of series k."""

    def __init__(self, folder):
        import pyarrow as pa
        import pyarrow.compute as pc
        (rel,) = _gep_shards(json.load(open(GEP_MANIFEST)), folder)
        self.folder, self._map = folder, pa.memory_map(os.path.join(GEP_DIR, rel), "r")
        try:                                                           # HF save_to_disk writes the IPC stream format
            tab = pa.ipc.open_stream(self._map).read_all()             # zero-copy: buffers point into the map
        except pa.ArrowInvalid:
            tab = pa.ipc.open_file(self._map).read_all()
        self.freq = tab.column("freq")[0].as_py() if tab.num_rows else None
        self.rows = tab.num_rows
        self._chunks, lens, nvar, cid = [], [], [], []
        for j, col in enumerate(tab.column("target").chunks):         # per record batch: never concatenated (copy)
            nested = not pa.types.is_floating(col.type.value_type)
            inner = col.flatten() if nested else col                  # one list<float> entry per univariate series
            nvar.append(pc.list_value_length(col).to_numpy(zero_copy_only=False) if nested
                        else np.ones(len(col), np.int64))
            ln = pc.list_value_length(inner).to_numpy(zero_copy_only=False).astype(np.int64)
            vals = inner.flatten()
            dense = vals.null_count == 0
            self._chunks.append((np.concatenate([[0], np.cumsum(ln)[:-1]]).astype(np.int64), vals,
                                 vals.to_numpy(zero_copy_only=True) if dense else None))
            lens.append(ln); cid.append(np.full(len(ln), j, np.int64))
        self.lengths = np.concatenate(lens) if lens else np.zeros(0, np.int64)
        self.n_var = np.concatenate(nvar) if nvar else np.zeros(0, np.int64)
        self._cid = np.concatenate(cid) if cid else np.zeros(0, np.int64)
        self._local = np.arange(len(self.lengths)) - np.concatenate(
            [[0], np.cumsum([len(x) for x in lens])[:-1]]).repeat([len(x) for x in lens]) if lens else self._cid

    def __len__(self):
        return len(self.lengths)

    def get(self, k, a=0, b=None):
        """Series k, points [a, b) (float32; NaN for missing). A view into the map when its chunk has no nulls."""
        starts, vals, dense = self._chunks[int(self._cid[k])]
        s, n = int(starts[int(self._local[k])]), int(self.lengths[k])
        b = n if b is None else min(b, n)
        if dense is not None:
            return dense[s + a:s + b]
        return vals.slice(s + a, b - a).to_numpy(zero_copy_only=False).astype(np.float32)

    def __iter__(self):
        for k in range(len(self)):
            yield k, self.get(k)


def gep_series(folders=None):
    """Every full univariate GEP series, before caps and GEP_DROP (for the overlap scan):
    yields (folder, key, float32 array view)."""
    gep_verify()
    for f, *_ in TRAIN_GEP:
        if folders and f not in folders:
            continue
        sh = _GepShard(f)
        for k, y in sh:
            yield f, k, y
        del sh


def load_gep(max_len=None, seed=SEED, drop_overlap=True):
    """The capped GEP training series as load_train dicts (dataset = the TRAIN_GEP group name, plus "gep_folder").
    Per folder, in a fixed random order (rng seeded by seed and the folder name): keep series with >= MIN_CONTEXT
    finite points until GEP_MAX_SERIES series or GEP_MAX_POINTS points; each kept series is a random contiguous crop
    of at most min(GEP_MAX_LEN, max_len) points. GEP_DROP is applied after selection, so it changes nothing else."""
    require_gep_stamp()
    L = min(GEP_MAX_LEN, max_len) if max_len else GEP_MAX_LEN
    out = []
    for folder, ds, season, freq, h in TRAIN_GEP:
        sh = _GepShard(folder)
        rng = np.random.default_rng([seed, zlib.crc32(folder.encode())])
        kept, pts = [], 0
        for k in rng.permutation(len(sh)):
            if len(kept) >= GEP_MAX_SERIES or pts >= GEP_MAX_POINTS:
                break
            n = int(sh.lengths[k])
            a = int(rng.integers(0, n - L + 1)) if n > L else 0
            y = np.array(sh.get(int(k), a, a + L), dtype=np.float32)
            if np.isfinite(y).sum() < MIN_CONTEXT:
                continue
            kept.append((int(k), y))
            pts += len(y)
        drop = set(GEP_DROP.get(folder, ())) if drop_overlap else set()
        out += [{"dataset": ds, "target": y, "horizon": h, "season": season, "freq": freq, "key": k,
                 "gep_folder": folder} for k, y in sorted(kept, key=lambda r: r[0]) if k not in drop]
        del sh
    return out


def load_train(extra=False, max_series_per_dataset=2000, max_len=None, seed=SEED, drop_overlap=True):
    """
    The only real data our model may learn from (synthetic data generated in code is also allowed).
    Whole series: no window of any TRAIN series is ever scored, so nothing has to be held back.
    extra=True adds the public Chronos/Chronos-2/TiRex pretraining sets in TRAIN_EXTRA.
    drop_overlap=True removes the series listed in TRAIN_DROP (content shared with VAL/TEST).
    Returns a list of dicts {dataset, target, horizon, season, freq} (same shape as
    prepare.load_train_series()); each also carries "key" (position within its dataset).
    TSAR_GIFT_CLEAN=1 skips every GIFT_TEST_OVERLAP dataset (LOCAL and EXTRA); the stamp guard still applies.
    The subsampling rng is shared across datasets, so the kept datasets' subsamples differ from the flag-off run.
    TSAR_BENCH_GEP=1 appends the capped GiftEvalPretrain subset (load_gep; needs a clean stamp-gep stamp) after
    everything else; the LOCAL/EXTRA part of the list is unchanged by it.
    """
    if extra:
        require_extra_stamp()
    if gep_on():
        require_gep_stamp()            # fail before the slow LOCAL/EXTRA reads
    specs = _train_specs(extra)
    if gift_clean():
        specs = [sp for sp in specs if sp[0][0] not in GIFT_TEST_OVERLAP]
    rng = np.random.default_rng(seed)
    out = []
    for (name, season, freq, h), is_extra in specs:
        df = pd.read_parquet(_train_file(name, is_extra))
        ys = [(k, y[-max_len:] if max_len else y) for k, y in enumerate(_target_arrays(df))]
        ys = [(k, y) for k, y in ys if np.isfinite(y).sum() >= MIN_CONTEXT]
        if max_series_per_dataset and len(ys) > max_series_per_dataset:
            ys = [ys[i] for i in rng.permutation(len(ys))[:max_series_per_dataset]]
        if drop_overlap:   # after subsampling, so every other series is unchanged by the filter
            drop = set(TRAIN_DROP.get(name, ()))
            ys = [(k, y) for k, y in ys if k not in drop]
        out += [{"dataset": name, "target": y, "horizon": h, "season": season, "freq": freq, "key": k}
                for k, y in ys]
    if gep_on():
        out += load_gep(max_len=max_len, seed=seed, drop_overlap=drop_overlap)
    return out

# ---------------------------------------------------------------------------
# VAL / TEST windows
# ---------------------------------------------------------------------------

def _norm_freq(ts):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = pd.infer_freq(pd.DatetimeIndex(ts[: min(len(ts), 64)]))
        f = getattr(f, "freqstr", f)
    except Exception:
        f = None
    if not f:
        return "D"
    f = f.split("-")[0]
    for k, v in (("YE", "Y"), ("YS", "Y"), ("QE", "Q"), ("QS", "Q"), ("ME", "M"), ("MS", "M"),
                 ("W", "W"), ("B", "B"), ("D", "D"), ("h", "H"), ("H", "H")):
        if f == k:
            return v
    if f in ("min", "T"):
        return "1min"
    return f.replace("T", "min")   # 5min, 10min, 15min, 30min


def _target_columns(df, target):
    if isinstance(target, list):
        return target
    if target.startswith("target_*:"):
        n = int(target.split(":")[1])
        cols = [f"target_{i}" for i in range(n)]
        assert all(c in df.columns for c in cols), f"missing target columns in {list(df.columns)[:10]}"
        return cols
    return [target]


def _build_task(spec):
    _guard_config(spec["config"])
    df = pd.read_parquet(_fev_file(spec["config"]))
    cols = _target_columns(df, spec["target"])
    H, W, step, cut = spec["horizon"], spec["num_windows"], spec["step"], spec["cutoff"]
    min_ctx = max(MIN_CONTEXT, int(spec["min_context"]))
    series = []   # (series_id, values, timestamps)
    for _, row in df.iterrows():
        for c in cols:
            series.append((f"{row['id']}/{c}" if len(cols) > 1 else str(row["id"]),
                           np.asarray(row[c], dtype=np.float32), row["timestamp"]))
    freq = _norm_freq(series[0][2])
    # fixed per-task subsample of series so that n_series * W <= MAX_FORECASTS_PER_TASK
    rng = np.random.default_rng(SEED + zlib.crc32(spec["name"].encode()))
    n_keep = max(1, MAX_FORECASTS_PER_TASK // W)
    if len(series) > n_keep:
        keep = np.sort(rng.permutation(len(series))[:n_keep])
        series = [series[i] for i in keep]
    ctxs, tgts, sids, wins = [], [], [], []
    for sid, y, ts in series:
        if isinstance(cut, str):     # datetime cutoff: past = timestamps <= cutoff
            first = int(np.searchsorted(pd.DatetimeIndex(ts).values, np.datetime64(cut), side="right"))
        else:                        # negative int: relative to the end of each series
            first = len(y) + cut
        for k in range(W):
            c = first + k * step
            if c < 0 or c + H > len(y):
                continue
            ctx = y[max(0, c - MAX_CONTEXT):c]
            fin = np.flatnonzero(np.isfinite(ctx))
            if len(fin) < min_ctx:
                continue
            ctxs.append(ctx[fin[0]:])          # drop leading all-NaN padding
            tgts.append(y[c:c + H])
            sids.append(sid)
            wins.append(k)
    return {"name": spec["name"], "family": spec["family"], "config": spec["config"],
            "horizon": H, "season": spec["season"], "freq": freq, "contexts": ctxs,
            "targets": np.stack(tgts) if tgts else np.zeros((0, H), np.float32),
            "series_id": sids, "window": np.asarray(wins)}


def load_tasks(split, names=None):
    """Windows for every task of a split (optionally a subset by task name)."""
    _guard(split, f"load_tasks({split})")
    specs = [s for s in SPLIT_TASKS[split] if names is None or s["name"] in names]
    return [_build_task(s) for s in specs]


def load_val(names=None):
    return load_tasks("val", names)


def load_test(names=None):
    return load_tasks("test", names)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _gmean(s):
    s = np.asarray(s, dtype=np.float64)
    return float(np.exp(np.log(np.clip(s, 1e-6, None)).mean()))


def evaluate(predict_fn, split="val", verbose=True, names=None, tasks=None):
    """
    Score predict_fn(contexts, horizon, season, freq) -> [n, 9, horizon] on every task of a split.
    contexts: list of 1-D float32 arrays (may contain NaN), oldest first, at most MAX_CONTEXT long.

    Returns {"split", "wql_rel", "mase_rel", "cov80_error", "wql_rel_family", "mase_rel_family",
             "win_rate_vs_sn", "predict_seconds", "per_dataset"}. per_dataset has one row per task
    (column "dataset" holds the task name, so tools/compare.py can read the CSV).
    """
    _guard(split, f"evaluate({getattr(predict_fn, '__name__', type(predict_fn).__name__)})")
    tasks = tasks if tasks is not None else load_tasks(split, names)
    Qn = len(P.QUANTILES)
    rows = []
    for t in tasks:
        h, m, ctxs, tgts = t["horizon"], t["season"], t["contexts"], t["targets"]
        if not ctxs:
            print(f"warning: {t['name']} has no valid windows, skipped")
            continue
        t0 = time.time()
        # copies: a predict_fn that edits its inputs in place cannot change the contexts used
        # for scoring (MASE scale, seasonal naive) or seen by the next model on shared tasks
        q = np.asarray(predict_fn([c.copy() for c in ctxs], h, m, t["freq"]), dtype=np.float32)
        secs = time.time() - t0
        assert q.shape == (len(ctxs), Qn, h), f"{t['name']}: bad shape {q.shape}"
        n_bad = int((~np.isfinite(q)).any(axis=(1, 2)).sum())   # NaN/inf are scored as 0: never silent
        if n_bad:
            print(f"warning: {t['name']}: {n_bad}/{len(ctxs)} forecasts contain NaN/inf (set to 0)")
        q = np.sort(np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
        wql, mase, cov = P._score(q, tgts, ctxs, m)
        sn = np.stack([P._seasonal_naive(c, h, m) for c in ctxs])
        sn_wql, sn_mase, _ = P._score(np.repeat(sn[:, None, :], Qn, axis=1), tgts, ctxs, m)
        rows.append({"dataset": t["name"], "family": t["family"], "n_forecasts": len(ctxs),
                     "n_series": len(set(t["series_id"])), "horizon": h, "freq": t["freq"],
                     "wql": wql, "mase": mase,
                     "wql_rel": wql / max(sn_wql, 1e-8), "mase_rel": mase / max(sn_mase, 1e-8),
                     "cov80": cov, "n_nonfinite": n_bad, "sn_wql": sn_wql, "sn_mase": sn_mase, "predict_seconds": secs})
    df = pd.DataFrame(rows)
    fam = df.groupby("family")[["wql_rel", "mase_rel"]].agg(_gmean)
    result = {
        "split": split,
        "wql_rel": _gmean(df["wql_rel"]),
        "mase_rel": _gmean(df["mase_rel"]),
        "cov80_error": float((df["cov80"] - 0.8).abs().mean()),
        "wql_rel_family": _gmean(fam["wql_rel"]),
        "mase_rel_family": _gmean(fam["mase_rel"]),
        "win_rate_vs_sn": float((df["wql_rel"] < 1).mean()),
        "n_tasks": len(df), "n_families": len(fam),
        "predict_seconds": float(df["predict_seconds"].sum()),
        "per_dataset": df,
    }
    if verbose:
        with pd.option_context("display.width", 160, "display.precision", 3, "display.max_rows", 200):
            print(df.drop(columns=["sn_wql", "sn_mase"]).to_string(index=False))
        print(f"{split}: wql_rel={result['wql_rel']:.4f} mase_rel={result['mase_rel']:.4f} "
              f"cov80_error={result['cov80_error']:.4f} family-gmean wql_rel={result['wql_rel_family']:.4f} "
              f"({result['n_tasks']} tasks, {result['n_families']} families)")
    return result

# ---------------------------------------------------------------------------
# Statistics: paired comparison of two models on the same tasks
# ---------------------------------------------------------------------------

def _as_runs(x):
    return [x] if isinstance(x, pd.DataFrame) else list(x)


def compare(a, b, metric="wql", n_boot=10000, seed=0):
    """
    Paired comparison of model A against model B on the same split.
    a, b: a per_dataset DataFrame from evaluate(), or a list of them (one per training seed).
    The per-task score is the mean over seeds of log(metric); ratio = exp(mean_task(logA - logB)),
    i.e. the geometric-mean ratio A/B (< 1 means A is better). Reports
      ci95_task    - percentile bootstrap over tasks
      ci95_family  - cluster bootstrap over source families (tasks of one family move together;
                     the honest interval, since tasks from one source are correlated)
      wins/losses  - tasks where A is better/worse, with an exact two-sided sign-test p-value
      seed_ratios  - the ratio computed per seed of A (spread = training noise), when several
    """
    ra, rb = _as_runs(a), _as_runs(b)
    key = "dataset"
    la = pd.concat([np.log(r.set_index(key)[metric].clip(lower=1e-12)) for r in ra], axis=1).mean(axis=1)
    lb = pd.concat([np.log(r.set_index(key)[metric].clip(lower=1e-12)) for r in rb], axis=1).mean(axis=1)
    common = la.index.intersection(lb.index)
    assert len(common) == len(la) == len(lb), "A and B were not scored on the same tasks"
    d = (la[common] - lb[common]).to_numpy()
    fams = ra[0].set_index(key).loc[common, "family"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(d)
    boot_t = np.exp(d[rng.integers(0, n, (n_boot, n))].mean(axis=1))
    ufam = np.unique(fams)
    idx_by_fam = [np.flatnonzero(fams == f) for f in ufam]
    boot_f = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([idx_by_fam[j] for j in rng.integers(0, len(ufam), len(ufam))])
        boot_f[i] = np.exp(d[pick].mean())
    wins, losses = int((d < 0).sum()), int((d > 0).sum())
    k, nn = min(wins, losses), wins + losses
    p_sign = min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) / 2 ** nn) if nn else 1.0
    seed_ratios = []
    for r in ra:
        ls = np.log(r.set_index(key).loc[common, metric].clip(lower=1e-12))
        seed_ratios.append(float(np.exp((ls - lb[common]).mean())))
    per_fam = pd.Series(d, index=common).groupby(fams).mean().apply(np.exp)
    return {"metric": metric, "ratio": float(np.exp(d.mean())), "n_tasks": n, "n_families": len(ufam),
            "ci95_task": tuple(float(v) for v in np.percentile(boot_t, [2.5, 97.5]).round(4)),
            "ci95_family": tuple(float(v) for v in np.percentile(boot_f, [2.5, 97.5]).round(4)),
            "wins": wins, "losses": losses, "p_sign": p_sign,
            "seed_ratios": seed_ratios,
            "seed_sd": float(np.std(seed_ratios, ddof=1)) if len(seed_ratios) > 1 else None,
            "per_family_ratio": per_fam.round(4).to_dict()}

# ---------------------------------------------------------------------------
# Leakage audits
# ---------------------------------------------------------------------------

def audit():
    """Static checks: VAL and TEST families disjoint, no excluded family used, exposure table."""
    fv = {t["family"] for t in SPLIT_TASKS["val"]}
    ft = {t["family"] for t in SPLIT_TASKS["test"]}
    assert not fv & ft, f"VAL/TEST share families: {fv & ft}"
    bad = {t["name"] for s in SPLIT_TASKS for t in SPLIT_TASKS[s]
           if t["family"] in EXCLUDED_FAMILIES or t["name"] in EXCLUDED_FAMILIES}
    assert not bad, f"excluded tasks present: {bad}"
    train_names = {n for n, *_ in TRAIN_LOCAL + TRAIN_EXTRA}
    assert not train_names & set(TRAIN_EXCLUDED), "TRAIN contains an excluded dataset"
    print(f"TRAIN: {len(TRAIN_LOCAL)} local datasets (+{len(TRAIN_EXTRA)} optional extra)")
    for k, v in TRAIN_EXCLUDED.items():
        print(f"  not in TRAIN: {k:24s} {v}")
    print(f"VAL:   {len(SPLIT_TASKS['val'])} tasks, {len(fv)} families")
    for f in sorted(fv):
        print(f"  {f:22s} exposure: {VAL_EXPOSURE.get(f, '?')}")
    print(f"TEST:  {len(SPLIT_TASKS['test'])} tasks, {len(ft)} families: {', '.join(sorted(ft))}")
    print("  exposure: none reported by any compared model at any frequency (fev-bench flags)")
    for k, v in EXCLUDED_FAMILIES.items():
        print(f"  excluded: {k:22s} {v}")
    print("audit OK")


def _chunk_hashes(y, L, stride, znorm=False):
    """
    Hashes of the length-L runs of y (NaNs removed) taken every `stride` points.
    exact (znorm=False): the values rounded to 1e-3.
    znorm=True: each run z-normalised and rounded to 0.01, so copies that were rescaled or shifted
    (other units, an added offset) also match: the near-duplicate check.
    Low-information runs are skipped so zero padding, flat lines and ramps do not match: fewer than
    L/4 distinct values, and for znorm also fewer than L/4 distinct first differences.
    """
    y = y[np.isfinite(y)].astype(np.float64)
    if len(y) < L:
        return np.zeros(0, np.int64)
    w = np.lib.stride_tricks.sliding_window_view(y, L)[::stride]
    if len(w):
        srt = np.sort(w, axis=1)
        w = w[1 + (np.diff(srt, axis=1) != 0).sum(axis=1) >= L // 4]
    if znorm and len(w):
        d = np.sort(np.round(np.diff(w, axis=1), 6), axis=1)
        w = w[1 + (np.diff(d, axis=1) != 0).sum(axis=1) >= L // 4]
    if not len(w):
        return np.zeros(0, np.int64)
    if znorm:
        sd = w.std(axis=1, keepdims=True)
        w = (w - w.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0) * 100.0
    else:
        w = w * 1000.0
    qv = np.clip(np.round(w), -4e15, 4e15).astype(np.int64)
    coef = np.random.default_rng(7).integers(1, 2 ** 61, L, dtype=np.int64)
    return (qv * coef).sum(axis=1)   # wrap-around int64 hash of the exact rounded values


def _train_source(extra, train_datasets=None):
    """(name, key, y) for every series of every TRAIN dataset (no subsampling, before TRAIN_DROP)."""
    for (name, *_), is_extra in _train_specs(extra):
        if train_datasets and name not in train_datasets:
            continue
        df = pd.read_parquet(_train_file(name, is_extra))
        for k, y in enumerate(_target_arrays(df)):
            yield name, k, y
        del df


def _series_spans(t, include_target):
    """
    Per series of a built task, the stretch of data that the windows expose, rebuilt from the exact
    arrays models receive: the union of all contexts (window k+1's context contains window k's
    target), plus the last window's target only when include_target is True.
    """
    by = {}
    for i, sid in enumerate(t["series_id"]):
        by.setdefault(sid, []).append(i)
    for sid, idx in by.items():
        idx = sorted(idx, key=lambda i: t["window"][i])
        parts = [t["contexts"][idx[0]]]
        for prev, cur in zip(idx, idx[1:]):   # every fev-bench task here has step == horizon
            n_new = min(int(t["window"][cur] - t["window"][prev]) * t["horizon"], len(t["contexts"][cur]))
            parts.append(t["contexts"][cur][len(t["contexts"][cur]) - n_new:])
        if include_target:
            parts.append(t["targets"][idx[-1]])
        yield sid, np.concatenate(parts)


def overlap(split="val", L=32, train_datasets=None, extra=False, max_series_per_dataset=None, source=None,
            tasks=None):
    """
    Content check for duplicated data under another name, against the FULL train corpus (every
    series of every TRAIN dataset, no subsampling, before TRAIN_DROP).
    Train bank: runs of L points at stride L (any shared stretch of >= 2L-1 points is found).
    Probe: every run of L points at stride 1 of the data each task exposes (_series_spans), i.e.
      val:  the union of all contexts plus the targets;
      test: the union of contexts only, never a target of the final window (target-free; no
            scoring happens and no forecast is made).
    Two matchers: exact (values rounded to 1e-3) and znorm (scale/offset invariant near-duplicates).
    Prints per task the number of series sharing a run with TRAIN and which TRAIN datasets/series
    matched; returns {task: {"exact": [...], "znorm": [...]}} with (dataset, key, n_runs) tuples.
    Expected result: 0 everywhere. Any TRAIN hit belongs in TRAIN_DROP.
    source: optional iterable of (name, key, y) that replaces the TRAIN bank (stamp-gep: gep_series());
    tasks: optional task names to probe (default: every task of the split).
    """
    _guard(split, f"overlap({split}, target_free={split == 'test'})")
    include_target = split != "test"
    names, keys, bank_e, bank_z, src_e, src_z = [], [], [], [], [], []
    for name, k, y in (source if source is not None else _train_source(extra, train_datasets)):
        j = len(names)
        names.append(name); keys.append(k)
        he, hz = _chunk_hashes(y, L, L), _chunk_hashes(y, L, L, znorm=True)
        bank_e.append(he); src_e.append(np.full(len(he), j, np.int32))
        bank_z.append(hz); src_z.append(np.full(len(hz), j, np.int32))
    banks = {}
    for tag, bk, sr in (("exact", bank_e, src_e), ("znorm", bank_z, src_z)):
        h, s_ = np.concatenate(bk), np.concatenate(sr)
        o = np.argsort(h, kind="stable")
        banks[tag] = (h[o], s_[o])
    print(f"train: {len(names)} series from {len(set(names))} datasets; chunk bank exact="
          f"{len(banks['exact'][0])} znorm={len(banks['znorm'][0])} hashes; "
          f"probe={'contexts+targets' if include_target else 'contexts only (target-free)'}")
    report = {}
    for spec in SPLIT_TASKS[split]:
        if tasks is not None and spec["name"] not in tasks:
            continue
        t = _build_task(spec)
        res = {"exact": {}, "znorm": {}}
        hit_series = {"exact": set(), "znorm": set()}
        n_series = 0
        for sid, y in _series_spans(t, include_target):
            n_series += 1
            for tag in ("exact", "znorm"):
                q = _chunk_hashes(y, L, 1, znorm=(tag == "znorm"))
                hb, sb = banks[tag]
                lo = np.searchsorted(hb, q, "left"); hi = np.searchsorted(hb, q, "right")
                m = hi > lo
                if not m.any():
                    continue
                hit_series[tag].add(sid)
                for a_, b_ in zip(lo[m], hi[m]):
                    for j in sb[a_:b_]:
                        res[tag][int(j)] = res[tag].get(int(j), 0) + 1
        report[t["name"]] = {tag: sorted(((names[j], keys[j], n) for j, n in res[tag].items()),
                                         key=lambda r: -r[2]) for tag in res}
        ne, nz = len(hit_series["exact"]), len(hit_series["znorm"])
        print(f"  {t['name']:32s} series={n_series:5d} windows={len(t['contexts']):5d} "
              f"series_sharing_{L}pt_run exact={ne} znorm={nz}" + ("   <-- INVESTIGATE" if ne or nz else ""))
        for tag in ("exact", "znorm"):
            for nm, k, n in report[t["name"]][tag][:5]:
                print(f"      {tag}: train {nm}[{k}] shares {n} runs")
    return report

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _naive(ctxs, h, m, freq):
    last = np.array([P.fill_nans(c)[-1] for c in ctxs], dtype=np.float32)
    return np.repeat(np.repeat(last[:, None, None], len(P.QUANTILES), 1), h, 2)


def _snaive(ctxs, h, m, freq):
    sn = np.stack([P._seasonal_naive(c, h, m) for c in ctxs])
    return np.repeat(sn[:, None, :], len(P.QUANTILES), axis=1)


def summary(split):
    for t in load_tasks(split):
        lens = [len(c) for c in t["contexts"]]
        print(f"{t['name']:32s} fam={t['family']:20s} freq={t['freq']:6s} H={t['horizon']:4d} "
              f"m={t['season']:5d} series={len(set(t['series_id'])):5d} forecasts={len(lens):5d} "
              f"ctx_median={int(np.median(lens)) if lens else 0:5d} "
              f"target_nan={float(np.isnan(t['targets']).mean()) if len(lens) else 0:.3f}")


BOLT_BATCH = int(os.environ.get("TSAR_BOLT_BATCH", "64"))   # series per Chronos-Bolt call (h <= 64)


def _mps_free():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _mem_guard(predict_fn, batch):
    """
    Chronos-Bolt under a memory cap (M3 8 GB). For h > 64 Bolt rolls out each of its 9 quantiles as a
    separate series, so a 64-series call becomes 576 rows x 2048 context and needed > 8 GiB on MPS.
    Here each call gets at most `batch` series (batch // 9 when h > 64); on MPS OOM the chunk is halved,
    and a single series that still OOMs runs on CPU. The MPS cache is emptied after every task.
    Every series is forecast independently (left NaN padding is masked, patches align from the right),
    so this changes predictions only by float noise. baselines.py is not modified.
    """
    def predict(ctxs, h, m, freq):
        st = predict_fn.state
        pl = getattr(st["pipe"], "model_prediction_length", 64) if "pipe" in st else 64
        b = batch if h <= pl else max(1, batch // len(P.QUANTILES))
        out, s = [], 0
        try:
            while s < len(ctxs):
                oom = False
                try:
                    out.append(predict_fn(ctxs[s:s + b], h, m, freq))
                    s += b
                except RuntimeError as e:
                    if "out of memory" not in str(e):
                        raise
                    oom = True                         # handled outside: the traceback pins MPS tensors
                if not oom:
                    continue
                _mps_free()
                if b > 1:
                    b = max(1, b // 2)
                    print(f"  bolt OOM (h={h}): batch -> {b}", flush=True)
                    continue
                print(f"  bolt OOM (h={h}) at batch 1: series {s} on CPU", flush=True)
                model = st["pipe"].model
                dev = model.device
                model.to("cpu")
                try:
                    out.append(predict_fn(ctxs[s:s + 1], h, m, freq))
                    s += 1
                finally:
                    model.to(dev)
        finally:
            _mps_free()
        return np.concatenate(out)

    predict.state = predict_fn.state
    return predict


def run_baselines(models, split):
    import baselines as BL   # read-only import; baselines.py is not modified
    reg = BL.registry()
    out_dir = os.path.join(ROOT, "baselines", "bench")
    os.makedirs(os.path.join(out_dir, "per_task"), exist_ok=True)
    tasks = load_tasks(split)
    for name in models or list(reg):
        t0 = time.time()
        fn = _mem_guard(reg[name], BOLT_BATCH) if name.startswith("chronos_bolt") else reg[name]
        try:
            r = evaluate(fn, split, verbose=False, tasks=tasks)
        except ImportError as e:
            print(f"{name:20s} skipped (not installed: {e.name})")
            continue
        r["per_dataset"].to_csv(os.path.join(out_dir, "per_task", f"{name}_{split}.csv"), index=False)
        row = {"model": name, "split": split, **{k: round(r[k], 4) for k in
               ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family", "win_rate_vs_sn")},
               "predict_seconds": round(r["predict_seconds"], 1), "total_seconds": round(time.time() - t0, 1),
               "date": time.strftime("%Y-%m-%d"), "git": _git_head()}
        path = os.path.join(out_dir, "results.tsv")
        pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not os.path.exists(path))
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "download", "summary", "smoke", "baselines", "overlap"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--extra", action="store_true", help="train: include TRAIN_EXTRA")
    ap.add_argument("--L", type=int, default=32, help="overlap: run length in points (default 32)")
    args = ap.parse_args()
    if args.cmd == "audit":
        audit()
    elif args.cmd == "download":
        download(args.split, extra=args.extra)
    elif args.cmd == "summary":
        summary(args.split)
    elif args.cmd == "smoke":
        names = ["ETT_1W", "uk_covid_nation_1W/new", "solar_1W", "hierarchical_sales_1W"]
        tasks = load_val(names)
        for fn in (_snaive, _naive):
            evaluate(fn, "val", tasks=tasks)
    elif args.cmd == "baselines":
        run_baselines(args.models, args.split)
    elif args.cmd == "overlap":
        overlap(args.split, L=args.L, train_datasets=args.models or None, extra=args.extra)
