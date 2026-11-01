"""
Re-score a baseline from baselines.py and keep its raw quantile forecasts, so tools/compare.py
can run series-level paired tests against it and the multi-level calibration check can use them.
Does not touch baselines/results.tsv or baselines/bench/results.tsv. Heavy (same cost as the
baseline itself) - run when the machine is idle.

    uv run tools/dump_baseline.py chronos2 [--split val]            # old suite -> baselines/dumps/chronos2/
    uv run tools/dump_baseline.py --bench chronos2 [--split val]    # bench     -> baselines/bench/dumps/chronos2/
    (TSAR_BENCH=1 in the environment implies --bench.) The bench test split still needs TSAR_SPLIT=test
    and is logged by bench's guard.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="val", choices=["val", "test", "holdout"])
    ap.add_argument("--bench", action="store_true", help="score on tools/bench.py tasks (leak-audited protocol)")
    args = ap.parse_args()
    bench = args.bench or bool(int(os.environ.get("TSAR_BENCH", 0) or 0))
    if bench and args.split == "holdout":
        sys.exit("the bench has no holdout split (val / test only)")

    import baselines as B  # noqa: E402
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import run as R  # noqa: E402  (tools/run.py: the dump wrapper; does not run train.py on import)

    reg = B.registry()
    tasks = None
    for name in args.models:
        if name not in reg:
            sys.exit(f"unknown baseline {name}; choose from {list(reg)}")
    for name in args.models:
        if bench:
            os.environ["TSAR_DUMP"] = os.path.join(ROOT, "baselines", "bench", "dumps", name)
            if tasks is None:                 # build windows once, shared by every model (like run_baselines)
                tasks = R.bench_module().load_tasks(args.split)
            res = R._bench_evaluate_and_dump(reg[name], split=args.split, verbose=False, tasks=tasks)
        else:
            os.environ["TSAR_DUMP"] = os.path.join(ROOT, "baselines", "dumps", name)
            res = R._evaluate_and_dump(reg[name], split=args.split, verbose=False)
        print(f"{name:20s} wql_rel={res['wql_rel']:.4f} -> {os.environ['TSAR_DUMP']}")


if __name__ == "__main__":     # statsforecast spawns worker processes: the guard is required
    main()
