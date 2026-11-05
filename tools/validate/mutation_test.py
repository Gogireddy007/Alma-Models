"""V1 — mutation testing of the deployment audit: plant one specific fake per copy of the sealed snapshot and require
tools/deploy/audit_deploy.py to FAIL on it (and an unmutated copy to PASS, so there are no false alarms).

    .venv/bin/python tools/validate/mutation_test.py   # -> runs/validation/mutation/RESULTS.json + .md

Each mutant is a copy of runs/deploy_2026_10_11 (forecast paths rewritten to the copy) with one change, audited in a
fresh process with ALMA_DEPLOY_OUT pointing at the copy. A mutant is "caught" when at least one check of the expected
kind fails. A mutant that passes the audit is a hole in the harness and is reported as such.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(ROOT, "runs", "deploy_2026_10_11")
OUT = os.path.join(ROOT, "runs", "validation", "mutation")
LV = np.round(np.arange(0.1, 0.91, 0.1), 2)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fresh_copy(name):
    d = os.path.join(OUT, "work", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(SNAP, d, ignore=shutil.ignore_patterns("data_manifests", "figures", "AUDIT*", "RESULTS.md"))
    rel_old, rel_new = os.path.relpath(SNAP, ROOT), os.path.relpath(d, ROOT)
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.endswith(".json"):
                p = os.path.join(root, fn)
                txt = open(p).read()
                if rel_old in txt:
                    open(p, "w").write(txt.replace(rel_old, rel_new))
    return d


def jload(d, rel):
    return json.load(open(os.path.join(d, rel)))


def jsave(d, rel, obj):
    json.dump(obj, open(os.path.join(d, rel), "w"), indent=1)


def npz_edit(path, fn):
    z = dict(np.load(path))
    fn(z)
    np.savez_compressed(path[:-4] if path.endswith(".npz") else path, **z)


# --------------------------------------------------------------------------- mutations
def m_control(d):
    pass


def m_oil_bluff_metric(d):
    """Report a better score than the model achieved (wm_sv October 1-day QS improved by 5%)."""
    r = jload(d, "oil/results.json")
    r["windows"]["oct_2026"]["h1"]["models"]["wm_sv"]["qs"] *= 0.95
    jsave(d, "oil/results.json", r)


def m_oil_edit_sealed_forecast(d):
    """Edit a sealed forecast after the fact without updating its seal."""
    r = jload(d, "oil/results.json")
    npz_edit(os.path.join(ROOT, r["preds_file"]), lambda z: z.__setitem__("q_wm_sv", z["q_wm_sv"] * 1.01))


def m_oil_fabricated_consistent(d):
    """Sophisticated fake: change the forecasts, re-seal them AND recompute every reported QS consistently."""
    r = jload(d, "oil/results.json")
    pf = os.path.join(ROOT, r["preds_file"])
    npz_edit(pf, lambda z: z.__setitem__("q_wm_sv", z["q_wm_sv"] * np.exp(0.002)))
    z = np.load(pf)
    import pandas as pd
    raw = pd.read_csv(os.path.join(ROOT, "data/commodity/raw/DCOILWTICO.csv"))
    raw.columns = ["date", "v"]
    raw["v"] = pd.to_numeric(raw["v"], errors="coerce")
    raw = raw.dropna()
    dates, v = pd.to_datetime(raw.date).to_numpy(), raw.v.to_numpy(float)
    oi = z["origins"]
    for wn, w in r["windows"].items():
        lo, hi = {"oct_2026": ("2026-10-01", "2026-10-31"), "nov_2026": ("2026-11-01", "2026-11-30")}[wn]
        for hk, x in w.items():
            if not x.get("n"):
                continue
            h = int(hk[1:])
            t = oi + h - 1
            ok = t < len(v)
            td = np.where(ok, dates[np.minimum(t, len(v) - 1)], np.datetime64("NaT"))
            sel = ok & (td >= np.datetime64(lo)) & (td <= np.datetime64(hi))
            y = np.log(v[t[sel]])
            q = np.log(z["q_wm_sv"][sel][:, 2:11, h - 1])
            e = y[:, None] - q
            x["models"]["wm_sv"]["qs"] = float(np.where(e >= 0, LV * e, (LV - 1) * e).mean(1).mean())
    r["preds_sha256"] = sha(pf)
    jsave(d, "oil/results.json", r)


def m_gas_truth_tamper(d):
    """Change a realised value (the EIA storage change of the first October week) to flatter the model."""
    r = jload(d, "gas/results.json")
    r["windows"]["oct_2026"]["actual_bcf"][0] += 5.0
    jsave(d, "gas/results.json", r)


def m_gas_overclaim(d):
    """Claim significance on 5 weeks of data."""
    r = jload(d, "gas/results.json")
    s = r["windows"]["oct_2026"]["kinds"]["thu"]["by_candidate"]["wm_gas_ens"]["qs"]["st_ols_wx"]
    s["verdict_qs"], s["n_eff"] = "cand_better", 5.0
    jsave(d, "gas/results.json", r)


def m_gas_fabricated_consistent(d):
    """Replace the sealed wm_gas_ens October forecasts with better ones (closer to the truth), re-seal, and
    recompute MAE and QS consistently."""
    r = jload(d, "gas/results.json")
    pf = os.path.join(ROOT, r["preds_file"])
    y = np.array(r["windows"]["oct_2026"]["actual_bcf"])

    def f(z):
        for kind in ("thu", "fri"):
            k = f"oct_2026__{kind}__wm_gas_ens"
            q = z[k].copy()
            q += (y - q[:, 6])[:, None] * 0.5                    # halve every error
            z[k] = q
    npz_edit(pf, f)
    z = np.load(pf)
    for kind in ("thu", "fri"):
        q = z[f"oct_2026__{kind}__wm_gas_ens"]
        e = y[:, None] - q[:, 2:11]
        for blk in r["windows"]["oct_2026"]["kinds"][kind]["by_candidate"].values():
            blk["qs"]["wm_gas_ens"]["qs"] = float(np.where(e >= 0, LV * e, (LV - 1) * e).mean(1).mean())
            blk["abs_err"]["wm_gas_ens"]["abs_err"] = float(np.abs(q[:, 6] - y).mean())
    r["preds_sha256"] = sha(pf)
    jsave(d, "gas/results.json", r)


def m_power_label_tamper(d):
    """Change realised prices (the labels) for one delivery day."""
    p = os.path.join(d, "power/preds/power_LABELS_local.npz")
    npz_edit(p, lambda z: z["actual"].__setitem__(3, z["actual"][3] + 25.0))


def m_power_weights_changed(d):
    """Report that a network's parameters changed during deployment (i.e. it was trained)."""
    r = jload(d, "power/results.json")
    r["param_watch"][0]["unchanged"] = False
    r["param_watch"][0]["sha_end"] = "0" * 64
    jsave(d, "power/results.json", r)


def m_fin_label_tamper(d):
    """Shift realised log-RV labels of three days."""
    p = os.path.join(d, "fin/preds/fin_LABELS_local_corrected.npz")
    npz_edit(p, lambda z: z["log_rv"].__setitem__(slice(5, 8), z["log_rv"][5:8] + 0.3))


def m_fin_fabricated_resealed(d):
    """Edit the wm_fin forecasts and re-seal them, but leave the metrics."""
    r = jload(d, "fin/results_corrected.json")
    pf = os.path.join(ROOT, r["preds_file"])
    npz_edit(pf, lambda z: z.__setitem__("q_wm_fin", z["q_wm_fin"] - 0.05))
    r["preds_sha256"] = sha(pf)
    jsave(d, "fin/results_corrected.json", r)


def m_fin_online_update(d):
    """Report a continual-learning update during deployment."""
    r = jload(d, "fin/results_corrected.json")
    r["models"]["wm_fin"]["updates"] = 3
    jsave(d, "fin/results_corrected.json", r)


MUTANTS = [  # name, function, audit sections to run, check kinds expected to FAIL ([] = must pass)
    ("control_unmutated", m_control, ["nofit", "oil", "gas"], []),
    ("oil_bluff_metric", m_oil_bluff_metric, ["oil"], ["D"]),
    ("oil_edit_sealed_forecast", m_oil_edit_sealed_forecast, ["oil"], ["G"]),
    ("oil_fabricated_consistent", m_oil_fabricated_consistent, ["oil"], ["R"]),
    ("gas_truth_tamper", m_gas_truth_tamper, ["gas"], ["C"]),
    ("gas_overclaim_significance", m_gas_overclaim, ["gas"], ["F"]),
    ("gas_fabricated_consistent", m_gas_fabricated_consistent, ["gas"], ["R"]),
    ("power_label_tamper", m_power_label_tamper, ["power"], ["C"]),
    ("power_weights_changed", m_power_weights_changed, ["nofit"], ["B"]),
    ("fin_label_tamper", m_fin_label_tamper, ["fin"], ["C"]),
    ("fin_fabricated_resealed", m_fin_fabricated_resealed, ["fin"], ["D", "R"]),
    ("fin_online_update", m_fin_online_update, ["nofit"], ["B"]),
]


def audit(d, sections):
    checks = []
    env = dict(os.environ, ALMA_DEPLOY_OUT=d)
    for sec in sections:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools/deploy/audit_deploy.py"), "--section", sec],
                       cwd=ROOT, env=env, capture_output=True, text=True)
        p = os.path.join(d, f".audit_{sec}.json")
        checks += json.load(open(p)) if os.path.exists(p) else [{"section": sec, "check": "section output",
                                                                  "status": "FAIL", "detail": "no output"}]
    return checks


def main():
    only = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, "RESULTS.json")
    res = json.load(open(rp)) if (only and os.path.exists(rp)) else []
    res = [x for x in res if x["mutant"] not in only]
    for name, fn, sections, expect in MUTANTS:
        if only and name not in only:
            continue
        d = fresh_copy(name)
        fn(d)
        checks = audit(d, sections)
        fails = [c for c in checks if c["status"] == "FAIL"]
        kinds = sorted({c["section"] for c in fails})
        if expect:
            ok = any(k in kinds for k in expect)
            verdict = "CAUGHT" if ok else "MISSED (harness hole)"
        else:
            ok = not fails
            verdict = "PASS (no false alarm)" if ok else "FALSE ALARM"
        row = {"mutant": name, "what": (fn.__doc__ or "no change").strip(), "sections": sections,
               "expected_fail_kinds": expect, "failed_kinds": kinds, "n_checks": len(checks), "n_fail": len(fails),
               "failed_checks": [f"{c['section']} {c['check']}" for c in fails][:6], "ok": ok, "verdict": verdict}
        res.append(row)
        print(f"{name:30s} {verdict:22s} failed kinds {kinds} ({len(fails)}/{len(checks)})", flush=True)
        shutil.rmtree(d)
    json.dump(res, open(rp, "w"), indent=1)
    L = ["# V1 — Mutation tests of the deployment audit\n",
         "Each row plants one fake in a copy of the sealed 4 Nov snapshot; the audit must fail on it. "
         "The control copy must pass.\n", "| mutant | planted fake | expected to trip | tripped | result |",
         "|---|---|---|---|---|"]
    for x in res:
        L.append(f"| {x['mutant']} | {x['what']} | {', '.join(x['expected_fail_kinds']) or '— (must pass)'} | "
                 f"{', '.join(x['failed_kinds']) or '—'} | **{x['verdict']}** |")
    n_ok = sum(x["ok"] for x in res)
    L.append(f"\n**{n_ok}/{len(res)} as expected.**")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_ok}/{len(res)} as expected")
    sys.exit(0 if n_ok == len(res) else 1)


if __name__ == "__main__":
    main()
