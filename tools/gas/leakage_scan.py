"""
Gas leakage scan: does Alma's real training data (the 14 TRAIN datasets in the s0 checkpoint cfg) contain the EIA
storage series (levels or weekly changes, L48 + regions) or Henry Hub spot? Same 6-gram method as
tools/commodity/leakage_scan.py (levels rounded; scale-free log-ratios for positive series; for the weekly changes,
which change sign, exact integer-Bcf 6-grams of the changes). Storage references use weeks released <= 2025-10-31
only (no TEST value is read).

    nice -n 10 .venv/bin/python tools/gas/leakage_scan.py > runs/gas/leakage_scan.txt
"""
import os
import sys
import json
import importlib.util

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import data as D       # noqa: E402
import storage as S    # noqa: E402

_spec = importlib.util.spec_from_file_location("cls", os.path.join(ROOT, "tools", "commodity", "leakage_scan.py"))
CL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CL)
grams, N = CL.grams, CL.N


def _seq(r):
    """ordered list of 6-gram byte keys (None for constant grams)."""
    if len(r) < N:
        return []
    w = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(r, N))
    keep = w.max(1) != w.min(1)
    return [bytes(x.view(np.uint8)) if k else None for x, k in zip(w, keep)]


def seq(y, kind):
    y = np.asarray(y, np.float64)
    y = y[np.isfinite(y)]
    if kind == "level":
        return _seq(np.round(y, 2))
    if kind == "change":
        return _seq(np.round(y, 0))
    y = y[y > 0]
    return _seq(np.round(np.diff(np.log(y)), 3)) if len(y) > N else []


def change_grams(y):
    return {g for g in seq(y, "change") if g is not None}


def longest_run(ref_seq, train_set):
    best = cur = 0
    for g in ref_seq:
        cur = cur + 1 if (g is not None and g in train_set) else 0
        best = max(best, cur)
    return best


def refs_seq():
    w = S.load()[0]
    w = w[w.avail <= D.VAL_END]
    out = {}
    for s in S.SERIES:
        out[f"stor_lvl_{s}"] = {"level": seq(w[f"lvl_{s}"].values, "level"), "ratio": seq(w[f"lvl_{s}"].values, "ratio")}
        out[f"stor_ds_{s}"] = {"change": seq(w[f"ds_{s}"].values, "change")}
    dates, hh, _, _ = D.panel()
    m = dates <= np.datetime64(D.VAL_END.date(), "D")
    out["hh_eia_daily"] = {"level": seq(hh[m], "level"), "ratio": seq(hh[m], "ratio")}
    return out


def refs():
    w = S.load()[0]
    w = w[w.avail <= D.VAL_END]
    out = {}
    for s in S.SERIES:
        out[f"stor_lvl_{s}"] = {"level": grams(w[f"lvl_{s}"].values, "level"), "ratio": grams(w[f"lvl_{s}"].values, "ratio")}
        out[f"stor_ds_{s}"] = {"level": change_grams(w[f"ds_{s}"].values), "ratio": set()}
    dates, hh, _, _ = D.panel()
    m = dates <= np.datetime64(D.VAL_END.date(), "D")
    out["hh_eia_daily"] = {"level": grams(hh[m], "level"), "ratio": grams(hh[m], "ratio")}
    return out


def main():
    ck = torch.load(os.path.join(ROOT, "data/ckpt/wm_ssm_none_s0.pt"), map_location="cpu", weights_only=False)["cfg"]
    names = ck["train_datasets"]
    print(f"TRAIN datasets ({len(names)}): {names}")
    rf = refs()
    rs = refs_seq()
    report = {}
    for name in names:
        path = os.path.join(ROOT, "data", "raw", f"{name}.parquet")
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        df = pd.read_parquet(path)
        cols = ["target"] if "target" in df.columns else [c for c in df.columns
                                                           if c not in ("id", "timestamp") and df[c].dtype == object]
        hits = {k: {"level": 0, "ratio": 0} for k in rf}
        runs = {}
        n = 0
        for c in cols:
            for v in df[c]:
                y = np.asarray(v, np.float64)
                n += 1
                gl, gr, gc = grams(y, "level"), grams(y, "ratio"), change_grams(y)
                tr = {"level": gl, "ratio": gr, "change": gc}
                for k, r in rf.items():
                    if k.startswith("stor_ds_"):
                        h = len(gc & r["level"])
                        hits[k]["level"] += h
                    else:
                        h = len(gl & r["level"]) + len(gr & r["ratio"])
                        hits[k]["level"] += len(gl & r["level"])
                        hits[k]["ratio"] += len(gr & r["ratio"])
                    if h:
                        for kind, sq in rs[k].items():
                            L = longest_run(sq, tr[kind])
                            if L > runs.get(f"{k}/{kind}", 0):
                                runs[f"{k}/{kind}"] = L
        tot = {k: v for k, v in hits.items() if v["level"] or v["ratio"]}
        report[name] = {"series": n, "gram_hits": tot, "longest_consecutive_run": runs}
        print(f"{name:32s} series={n:6d} hits={tot or 'none'}\n{'':32s} longest consecutive matching grams: "
              f"{runs or '-'}", flush=True)
        del df
    json.dump(report, open(os.path.join(ROOT, "runs", "gas", "leakage_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
