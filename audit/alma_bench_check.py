"""Re-score the published Alma research checkpoints on the bench VAL split (claimed: wm_ssm_none_s0 wql_rel 0.6707)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "repo")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "tools", "commodity"))
import models as CM
import bench as BN
out = {}
for ck in sys.argv[1:]:
    p = CM._alma(ck)
    p.batch([[1.0] * 300], 5, [0.5])                 # triggers load() and captures predict_fn
    fn = [c.cell_contents for c in p.batch.__closure__ if isinstance(c.cell_contents, dict) and "fn" in c.cell_contents][0]["fn"]
    r = BN.evaluate(fn, "val", verbose=False)
    out[ck] = {k: r[k] for k in ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family") if k in r}
    r["per_dataset"].to_csv(os.path.join(HERE, "ssm_pertask_" + os.path.basename(ck).replace(".pt", ".csv")), index=False)
    print(ck, out[ck], flush=True)
json.dump(out, open(os.path.join(HERE, "alma_bench_check.json"), "w"), indent=1)
