"""
Unit tests for the bench claim tooling (tools/compare.py, tools/run.py dump hook). CSV only:
no model, no data download, no TEST access. Runs in a few seconds:

    uv run python tools/test_compare_bench.py        (or: uv run pytest tools/test_compare_bench.py)
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import compare as C  # noqa: E402

PT = os.path.join(ROOT, "baselines", "bench", "per_task")


def _bench_log(csv, path, split="val"):
    """A train.py-style log, printed exactly as bench.evaluate(verbose=True) + train.py print it."""
    df = pd.read_csv(csv)
    with open(path, "w") as f:
        f.write("step     0  loss 1.0\n")
        with pd.option_context("display.width", 160, "display.precision", 3, "display.max_rows", 200):
            f.write(df.drop(columns=["sn_wql", "sn_mase"]).to_string(index=False) + "\n")
        f.write(f"{split}: wql_rel=0.5 mase_rel=0.7 cov80_error=0.05 family-gmean wql_rel=0.5 (30 tasks, 13 families)\n")
        f.write(f"---\n{split}_wql_rel:      0.5\n")


def test_parse_bench_log():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "x.log")
        _bench_log(os.path.join(PT, "chronos2_val.csv"), log)
        assert C.spec_kind(log, "val") is True
        tab = C.parse_log(log)["val"]
        ref = pd.read_csv(os.path.join(PT, "chronos2_val.csv"))
        assert list(tab["dataset"]) == list(ref["dataset"]) and list(tab["family"]) == list(ref["family"])
        assert np.allclose(tab["wql_rel"], ref["wql_rel"], atol=6e-4)
        assert (tab["n_nonfinite"] == 0).all()


def test_resolve_bench_names():
    r = C.Run("chronos2", "val", bench=True)
    assert len(r.df) == 30 and "family" in r.df.columns
    assert C.detect_bench(["chronos2", "tirex"], "val", flag=True) is True
    assert C.detect_bench([os.path.join(PT, "tirex_val.csv")], "val") is True


def test_refuse_mixing():
    old = os.path.join(ROOT, "baselines", "per_dataset", "chronos2_val.csv")
    for call in (lambda: C.detect_bench([old, os.path.join(PT, "tirex_val.csv")], "val"),
                 lambda: C.Run(old, "val", bench=True)):
        try:
            call()
        except SystemExit:
            continue
        raise AssertionError("mixing old-suite and bench tables was not refused")


def test_family_ci_matches_bench_compare():
    import bench as BN
    a, b = os.path.join(PT, "timestral_209m_val.csv"), os.path.join(PT, "chronos2_val.csv")
    out = C.compare([C.Run(a, "val", True)], [C.Run(b, "val", True)], "val", "wql_rel")
    ref = BN.compare(pd.read_csv(a), pd.read_csv(b), metric="wql_rel")
    assert abs(out["ratio"] - ref["ratio"]) < 1e-12
    assert np.allclose(out["ci_fam"], ref["ci95_family"], atol=6e-5), (out["ci_fam"], ref["ci95_family"])
    assert np.allclose(out["ci_ds"], ref["ci95_task"], atol=6e-5)
    # numbers quoted in docs/audit_v2.md section 3 (power table)
    assert round(out["ratio"], 3) == 1.009
    assert np.allclose(out["ci_fam"], (0.972, 1.063), atol=6e-4), out["ci_fam"]


def test_holm():
    adj = C.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02]), adj
    assert np.allclose(C.holm([0.5]), [0.5])


def test_gate():
    sn = C.Run("seasonal_naive_pi", "val", True)
    good = C.compare([C.Run("chronos2", "val", True)], [sn], "val", "wql_rel")
    g = dict((k, ok) for k, ok, _ in C.bench_gate(good, min_seeds=1))
    assert all(g.values()), g                                        # chronos2 clearly beats SN-PI
    bad = C.compare([C.Run("timestral_209m", "val", True)], [C.Run("chronos2", "val", True)], "val", "wql_rel")
    g = dict((k, ok) for k, ok, _ in C.bench_gate(bad, min_seeds=1))
    assert not g["G1 CI_fam upper < 1"]                              # 1.009, CI crosses 1
    g = dict((k, ok) for k, ok, _ in C.bench_gate(good, min_seeds=3))
    assert not g["G4 seed condition"]                                # one seed < 3 required


def test_run_bench_dump():
    """tools/run.py patches bench.evaluate and writes per_task/preds/meta (seasonal naive, 1 val task)."""
    import run as R
    with tempfile.TemporaryDirectory() as d:
        os.environ["TSAR_DUMP"] = d
        try:
            BN = R.bench_module()
            res = R._bench_evaluate_and_dump(BN._snaive, "val", verbose=False, names=["solar_1W"])
        finally:
            del os.environ["TSAR_DUMP"]
        tab = pd.read_csv(os.path.join(d, "per_task_val.csv"))
        assert list(tab["dataset"]) == ["solar_1W"] and np.allclose(tab["wql_rel"], 1.0)
        preds = np.load(os.path.join(d, "preds_val.npz"))
        assert preds["solar_1W"].shape[1:] == (9, 13)
        assert C.spec_kind(d, "val") is True and len(C.Run(d, "val", True).df) == 1
        assert res["wql_rel"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
