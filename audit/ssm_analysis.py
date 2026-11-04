"""World-model test for the research-track SSM over 5 seeds: rollout (abl=none) vs matched direct twin.

Seed 0: per-task tables of the published checkpoints (alma_bench_check.py). Seeds 1-4: per-task tables printed in
audit/logs/aud_ssm_{none,direct}_s{seed}.log by train_wm.py's VAL evaluation (same bench.evaluate).
Headline = geometric mean of per-task wql_rel (as in the repo). Paired test: per-task log(direct / rollout), averaged
over seeds, family-cluster bootstrap (13 families) and sign test over seeds.
"""
import io
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMED_NONE = {0: 0.6707, 1: 0.6936, 2: 0.6804, 3: 0.6808, 4: 0.6765}   # docs/audit_r1.md (5-seed list)


def from_log(path):
    lines = open(path).read().splitlines()
    hdr = [j for j, l in enumerate(lines) if re.match(r"^\s*dataset\s+family\s", l)]
    if not hdr:
        return None
    txt = "\n" + "\n".join(lines[hdr[-1]:])
    i = 0
    block = []
    for line in txt[i + 1:].splitlines():
        if re.match(r"^\s*(val|dev): wql_rel=", line) or not line.strip():
            break
        block.append(line)
    df = pd.read_fwf(io.StringIO("\n".join(block)))
    return df[["dataset", "family", "wql_rel"]]


def table(abl, seed):
    if seed == 0:
        return pd.read_csv(os.path.join(HERE, f"ssm_pertask_wm_ssm_{abl}_s0.csv"))[["dataset", "family", "wql_rel"]]
    return from_log(os.path.join(HERE, "logs", f"aud_ssm_{abl}_s{seed}.log"))


def gm(x):
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-6)))))


def main():
    rows, diffs = [], []
    for s in range(5):
        a, b = table("none", s), table("direct", s)
        if a is None or b is None:
            rows.append({"seed": s, "status": "missing"})
            continue
        m = a.merge(b, on=["dataset", "family"], suffixes=("_none", "_direct"))
        assert len(m) == 30, len(m)
        d = np.log(m.wql_rel_direct) - np.log(m.wql_rel_none)          # > 0: rollout better
        diffs.append(pd.Series(d.values, index=m.dataset))
        rows.append({"seed": s, "none": gm(m.wql_rel_none), "claimed_none": CLAIMED_NONE[s], "direct": gm(m.wql_rel_direct),
                     "direct_minus_none": gm(m.wql_rel_direct) - gm(m.wql_rel_none), "tasks_rollout_better": int((d > 0).sum())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(diffs) >= 2:
        D = pd.concat(diffs, axis=1).mean(1)
        fam = table("none", 0).set_index("dataset").family.reindex(D.index)
        rng = np.random.default_rng(0)
        fams = fam.unique()
        bs = []
        for _ in range(5000):
            pick = rng.choice(fams, len(fams))
            bs.append(np.concatenate([D[fam == f].values for f in pick]).mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        n_seeds = len(diffs)
        pos = int((t.direct_minus_none > 0).sum())
        print(f"\nseeds used: {n_seeds}; mean per-task log(direct/rollout) {D.mean():+.4f} "
              f"(family-cluster 95% CI [{lo:+.4f}, {hi:+.4f}]); seeds with rollout better: {pos}/{n_seeds}")
        fresh = [d for s, d in zip([r['seed'] for r in rows if 'none' in r], diffs) if s != 0]
        if fresh:
            Df = pd.concat(fresh, axis=1).mean(1)
            print(f"excluding the selection seed 0: mean {Df.mean():+.4f} over {len(fresh)} seeds")
    t.to_csv(os.path.join(HERE, "ssm_wm_test.csv"), index=False)


if __name__ == "__main__":
    main()
