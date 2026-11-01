"""Build runs/fin/FROZEN.json (docs/fin_search.md Am1): pooled K over results.tsv + results_b.tsv, var_sharpe_trials,
checkpoint sha256 (keyed by the same relative paths as the wrappers' meta["ckpt"]), recipes, twins, online rule,
go/no-go rule. Also writes the sha256 into the wrappers' FROZEN_CKPT constants.

  .venv/bin/python tools/fin/search/make_frozen.py --spec runs/fin/search/freeze_spec.json
"""
import os
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402

ROOT = C.ROOT
TSV_B = os.path.join(ROOT, "runs", "fin", "search_b", "results_b.tsv")


def read_tsv(p):
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="warn")


def sr_period_of(row):
    try:
        n = json.loads(row.get("note") or "{}")
    except Exception:
        n = {}
    vals = []
    if isinstance(n, dict):
        if "sr_period_heldout" in n:
            vals.append(n["sr_period_heldout"])
        for v in (n.get("folds") or {}).values():
            if isinstance(v, list) and len(v) > 1 and isinstance(v[1], dict) and "sr_period_heldout" in v[1]:
                vals.append(v[1]["sr_period_heldout"])
    if not vals and row.get("SR_heldout") not in ("", None) and row.get("fold", "") != "EM":
        vals.append(float(row["SR_heldout"]) / np.sqrt(365))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    A, B = read_tsv(C.TSV), read_tsv(TSV_B)
    ref = lambda df: df["kind"].isin(["ref"]) if len(df) else []
    nA, nB = len(A), len(B)
    kA = int((~ref(A)).sum()) if nA else 0
    kB = int((~ref(B)).sum()) if nB and "kind" in B else nB
    agent_kinds = ("agent", "loop_rl", "holdout_rl", "f1ens_rl")
    srs, used = [], []
    for df in (A, B):
        if not len(df):
            continue
        for _, r in df.iterrows():
            kd = r.get("kind", "")
            if kd in agent_kinds or (kd not in ("ref",) and r.get("SR_heldout", "") not in ("", None)
                                     and kd.startswith(("agent", "rl", "loop_rl", "b_rl", "holdout", "f1ens"))):
                if r.get("status", "") in ("ok",):
                    v = [float(x) for x in sr_period_of(r) if x is not None and np.isfinite(float(x))]
                    srs += v
                    used += [(r.get("name", ""), r.get("fold", ""), x) for x in v]
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else None
    sha = {}
    for name, rel in spec["ckpts"].items():
        sha[rel] = C.sha(os.path.join(ROOT, rel))
    # write sha into wrapper constants
    wf = os.path.join(C.FIN, "wm_fin.py")
    s = open(wf).read()
    for nm in ("wm_fin", "wm_direct"):
        rel = spec["ckpts"][nm]
        s = re.sub(rf'"{nm}": \("{re.escape(rel)}", [^)]*\)', f'"{nm}": ("{rel}", "{sha[rel]}")', s)
    s = re.sub(r'ONLINE = \{.*\}', "ONLINE = " + json.dumps(spec["online_rule"]).replace("null", "None"), s)
    open(wf, "w").write(s)
    for nm in ("wm_agent", "ppo", "dqn"):
        p = os.path.join(C.FIN, "agents", f"{nm}.py")
        rel = spec["ckpts"][nm]
        t = open(p).read()
        t = re.sub(r'FROZEN_CKPT = \("([^"]+)", [^)]*\)', f'FROZEN_CKPT = ("{rel}", "{sha[rel]}")', t)
        open(p, "w").write(t)
    fz = dict(spec["frozen"])
    fz.update({"models": spec["models"], "claim_models": spec["claim_models"], "sha256": sha,
               "K": kA + kB, "K_detail": {"architect_A_rows_excl_refs": kA, "architect_B_rows_excl_refs": kB,
                                          "results_tsv_rows": nA, "results_b_tsv_rows": nB},
               "var_sharpe_trials": var_sr, "var_sharpe_trials_n": len(srs), "var_sharpe_trials_rows": used,
               "var_sharpe_trials_note": "variance of the per-period (daily) net Sharpe on held-out coins over every "
                                         "agent row (all folds, both logs); DSR uses per-period units"})
    out = os.path.join(ROOT, "runs", "fin", "FROZEN.json")
    json.dump(fz, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: fz[k] for k in ("K", "K_detail", "var_sharpe_trials", "var_sharpe_trials_n", "claim_models")},
                     indent=1))


if __name__ == "__main__":
    main()
