"""Auditor re-derivation (audit r1, 2026-10-31). Text-only: parses the LAST per-task table of runs/<name>.log and
baselines/bench/per_task/<name>_val.csv; imports nothing from the repo.
  headline wql_rel / mase_rel = geometric mean over tasks of the per-task ratio; cov80_err = mean |cov80 - 0.8|.
Log tables print 3 decimals, so re-derived log means carry ~+-3e-4 rounding; CSVs are full precision.
Also checks that every run is scored on the same 30 VAL tasks with the same n_forecasts as the baselines."""
import csv, glob, json, math, os, re, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gm = lambda xs: math.exp(sum(math.log(max(x, 1e-6)) for x in xs) / len(xs))


def table(path):
    lines = open(path, errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith("dataset") and "wql_rel" in l]
    if not starts:
        return None, None
    hdr = lines[starts[-1]].split()
    rows = {}
    for l in lines[starts[-1] + 1:]:
        p = l.split()
        if len(p) != len(hdr):
            break
        rows[p[0]] = dict(zip(hdr, p))
    printed = {}
    for l in lines:
        m = re.match(r"val_(wql_rel|mase_rel|cov80_error):\s+([\d.]+)", l)
        if m:
            printed[m.group(1)] = float(m.group(2))
    return rows, printed


def derive(rows):
    w = [float(r["wql_rel"]) for r in rows.values()]; m = [float(r["mase_rel"]) for r in rows.values()]
    c = [abs(float(r["cov80"]) - 0.8) for r in rows.values()]
    return gm(w), gm(m), sum(c) / len(c)


ref_n = {}
for r in csv.DictReader(open(os.path.join(ROOT, "baselines/bench/per_task/seasonal_naive_pi_val.csv"))):
    ref_n[r["dataset"]] = (int(r["n_forecasts"]), float(r["sn_wql"]))
print("== baselines (full-precision CSV; wql_rel recomputed as wql / sn_wql)")
tsv = {r["model"]: r for r in csv.DictReader(open(os.path.join(ROOT, "baselines/bench/results.tsv")), delimiter="\t")}
for f in sorted(glob.glob(os.path.join(ROOT, "baselines/bench/per_task/*_val.csv"))):
    name = os.path.basename(f)[:-8]
    rs = list(csv.DictReader(open(f)))
    same = all(ref_n.get(r["dataset"], (None,))[0] == int(r["n_forecasts"]) and
               abs(ref_n[r["dataset"]][1] - float(r["sn_wql"])) < 1e-9 for r in rs) and len(rs) == len(ref_n)
    w = gm([float(r["wql"]) / float(r["sn_wql"]) for r in rs]); m = gm([float(r["mase_rel"]) for r in rs])
    c = sum(abs(float(r["cov80"]) - 0.8) for r in rs) / len(rs)
    t = tsv.get(name, {})
    print(f"  {name:20s} wql_rel {w:.4f} (tsv {t.get('wql_rel')})  mase_rel {m:.4f} (tsv {t.get('mase_rel')})  "
          f"cov80_err {c:.4f} (tsv {t.get('cov80_error')})  tasks={len(rs)} same_tasks_and_sn={same}")

groups = {
    "ssm none (raw)": [f"wm_ssm_none_s{i}" for i in range(5)],
    "gru none (raw)": [f"wm_gru_s{i}" for i in range(5)],
    "ssm none +TTA (eval-only)": [f"wm_ssm_none_s{i}_tta" for i in range(3)],
    "ssm none raw, seeds 0-2": [f"wm_ssm_none_s{i}" for i in range(3)],
    "gru +TTA (eval-only)": [f"wm_gru_s{i}_tta" for i in range(3)],
    "gru raw, seeds 0-2": [f"wm_gru_s{i}" for i in range(3)],
    "ssm none +paths": [f"wm_ssm_none_s{i}_paths" for i in range(3)],
    "gru +paths": [f"wm_gru_s{i}_paths" for i in range(3)],
    "tswm": ["wm_tswm_origin_s0", "wm_tswm_none_s0", "wm_tswm_direct_s0", "wm_ssm_direct_s0", "wm_ssm_copy_s0"],
}
for g, names in groups.items():
    print(f"== {g}")
    vals = []
    for n in names:
        p = os.path.join(ROOT, "runs", n + ".log")
        rows, printed = table(p) if os.path.exists(p) else (None, None)
        if not rows or "wql_rel" not in printed:
            print(f"  {n:28s} no final result"); continue
        w, m, c = derive(rows)
        tasks_ok = len(rows) == 30 and all(ref_n[k][0] == int(r["n_forecasts"]) for k, r in rows.items())
        prov = os.path.join(ROOT, "runs", n + ".prov.json")
        env = json.load(open(prov))["env"] if os.path.exists(prov) else {}
        print(f"  {n:28s} derived wql {w:.4f} mase {m:.4f} cov {c:.4f} | printed wql {printed['wql_rel']:.4f} "
              f"mase {printed.get('mase_rel', float('nan')):.4f} cov {printed.get('cov80_error', float('nan')):.4f} "
              f"| 30 tasks same n as baselines: {tasks_ok} | steps={env.get('TSAR_STEPS')} seed={env.get('TSAR_SEED')} "
              f"clean={env.get('TSAR_GIFT_CLEAN')} gep={env.get('TSAR_BENCH_GEP', 'unset')} extra={env.get('TSAR_BENCH_EXTRA', 'unset')}")
        vals.append((printed["wql_rel"], printed.get("mase_rel"), printed.get("cov80_error"), w))
    if len(vals) > 1:
        ws = [v[0] for v in vals]
        print(f"  -> mean printed wql {st.mean(ws):.4f} (sd {st.stdev(ws):.4f}, n={len(ws)}), "
              f"mase {st.mean(v[1] for v in vals):.4f}, cov {st.mean(v[2] for v in vals):.4f}; "
              f"mean derived wql {st.mean(v[3] for v in vals):.4f}")
