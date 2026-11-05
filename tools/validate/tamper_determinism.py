"""V2 — tamper tests and determinism tests.

    .venv/bin/python tools/validate/tamper_determinism.py   # -> runs/validation/tamper/RESULTS.json + .md

Tamper (negative controls; every one must be REFUSED, and its unflipped control must load):
  T1 a one-byte flip in a frozen checkpoint (each project) -> the model wrapper refuses to load it;
  T2 the same flip -> tools/deploy/common.verify_checkpoints refuses the deployment;
  T3 a one-character edit of frozen model code -> verify_checkpoints refuses (FREEZE.json code pins);
  T4 a pinned raw data file whose bytes no longer match its pin -> fetch_from_manifest exits 1;
  T5 the no-training lock: an optimizer step and a backward() call raise inside a deployment process.
All tampering happens in a throw-away copy of the repository (runs/validation/tamper/tree), never in the real files.

Determinism: each frozen world model is run twice on identical inputs in separate processes; forecasts must be
bit-identical. Also reported: the largest difference between 1 and 4 CPU threads (floating-point order only).
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs", "validation", "tamper")
TREE = os.path.join(OUT, "tree")
PY = os.path.join(ROOT, ".venv", "bin", "python")
RES = []


def rec(test, project, expected, ok, detail):
    RES.append({"test": test, "project": project, "expected": expected, "ok": bool(ok), "detail": detail[-400:]})
    print(f"[{'OK ' if ok else 'BAD'}] {test:4s} {project:6s} {expected:12s} {detail[-160:]}", flush=True)


def make_tree():
    if os.path.exists(TREE):
        shutil.rmtree(TREE)
    os.makedirs(TREE)
    for d in ("tools", "winning_models"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(TREE, d), ignore=shutil.ignore_patterns("__pycache__"))
    for proj in ("gas", "power", "commodity"):
        shutil.copytree(os.path.join(ROOT, "runs", proj, "search", "ckpt"), os.path.join(TREE, "runs", proj, "search", "ckpt"))
    os.makedirs(os.path.join(TREE, "runs", "fin", "search", "ckpt"))
    for fn in ("frozen_wm_fin.pt", "frozen_wm_direct.pt"):                 # the files the fin wrapper loads
        shutil.copy(os.path.join(ROOT, "runs", "fin", "search", "ckpt", fn), os.path.join(TREE, "runs", "fin", "search", "ckpt", fn))
    for d in ("data", "wm"):
        os.symlink(os.path.join(ROOT, d), os.path.join(TREE, d))
    subprocess.run(["git", "init", "-q"], cwd=TREE)


def run_py(code, cwd=TREE, env=None):
    p = subprocess.run([PY, "-c", textwrap.dedent(code)], cwd=cwd, capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    return p.returncode, (p.stdout + p.stderr).strip()


LOADERS = {   # project -> (checkpoint path in the tree, code that loads it through the real wrapper)
    "oil": ("runs/commodity/search/ckpt/f_sv_none_s0.pt",
            "import sys; sys.path.insert(0,'tools/commodity'); import models as M; M.get('wm_sv'); print('LOADED')"),
    "power": ("runs/power/search/ckpt/frz_phys_none_s0.pt",
              "import sys; sys.path.insert(0,'tools/power'); import models as M; p=M.get('wm_power'); "
              "import numpy as np, data as D; d,P,C,f=D.panel(); c=D.build_context(len(d)-1,736,d,P,C,f); "
              "p.batch([c['prices']],[c['covs']],24,[0.5],dates=[{'ctx_dates':c['ctx_dates'],'target_day':c['target_day']}]); print('LOADED')"),
    "gas": ("winning_models/gas_wm_gas_ens/checkpoints/x_wm_s3456.pt",
            "import sys; sys.path.insert(0,'tools/gas'); import models as M; M.get('wm_gas_ens'); print('LOADED')"),
    "fin": ("runs/fin/search/ckpt/frozen_wm_fin.pt",
            "import sys; sys.path.insert(0,'tools/fin'); import models as M; M.get('wm_fin'); print('LOADED')"),
}


def flip(path):
    b = bytearray(open(path, "rb").read())
    b[len(b) // 2] ^= 0x01
    open(path, "wb").write(bytes(b))


def tamper_tests():
    make_tree()
    for proj, (rel, code) in LOADERS.items():
        p = os.path.join(TREE, rel)
        rc, out = run_py(code)
        rec("T1c", proj, "loads", rc == 0 and "LOADED" in out, "control (untouched copy): " + out.splitlines()[-1] if out else "")
        orig = open(p, "rb").read()
        flip(p)
        rc, out = run_py(code)
        rec("T1", proj, "refused", rc != 0 and "LOADED" not in out and "sha" in out.lower(),
            "1-byte flip in " + rel + " -> " + (out.splitlines()[-1] if out else ""))
        proj2 = {"oil": "oil", "power": "power", "gas": "gas", "fin": "fin"}[proj]
        rc, out = run_py(f"import sys; sys.path.insert(0,'tools/deploy'); import common as C; C.verify_checkpoints('{proj2}'); print('ACCEPTED')")
        rec("T2", proj, "refused", rc != 0 and "ACCEPTED" not in out,
            "verify_checkpoints after the flip -> " + (out.splitlines()[-1] if out else ""))
        open(p, "wb").write(orig)
    # T3 frozen code edit
    for proj, rel in (("power", "tools/power/wm_power.py"), ("gas", "tools/gas/wm_gas.py"), ("fin", "tools/fin/wm_fin.py")):
        p = os.path.join(TREE, rel)
        orig = open(p).read()
        open(p, "w").write(orig + "\n# one extra comment\n")
        rc, out = run_py(f"import sys; sys.path.insert(0,'tools/deploy'); import common as C; C.verify_checkpoints('{proj}'); print('ACCEPTED')")
        rec("T3", proj, "refused", rc != 0 and "ACCEPTED" not in out, f"edit {rel} -> " + (out.splitlines()[-1] if out else ""))
        open(p, "w").write(orig)
    # T4 data pin mismatch: a manifest entry whose bytes on disk no longer match the pin
    man = json.load(open(os.path.join(ROOT, "data/fin/manifest.json")))
    k = next(x for x in man["files"] if "/um/klines_5m/BTCUSDT/" in x and x.endswith("2024-01.zip"))
    fake = {"files": {k: dict(man["files"][k], sha256="0" * 64)}}
    mp = os.path.join(OUT, "manifest_with_wrong_pin.json")
    json.dump(fake, open(mp, "w"))
    p = subprocess.run([PY, "tools/deploy/fetch_from_manifest.py", os.path.relpath(mp, ROOT)], cwd=ROOT,
                       capture_output=True, text=True)
    rec("T4", "fin", "refused", p.returncode == 1, "pinned file whose bytes do not match -> exit "
        f"{p.returncode}: {p.stdout.strip().splitlines()[0]}")
    good = {"files": {k: man["files"][k]}}
    json.dump(good, open(mp, "w"))
    p = subprocess.run([PY, "tools/deploy/fetch_from_manifest.py", os.path.relpath(mp, ROOT)], cwd=ROOT,
                       capture_output=True, text=True)
    rec("T4c", "fin", "accepted", p.returncode == 0, "control, correct pin -> exit " + str(p.returncode))
    os.remove(mp)
    # T5 no-training lock
    for what, code in (("optimizer step", "import torch; m=torch.nn.Linear(2,1); o=torch.optim.SGD(m.parameters(),0.1); o.step()"),
                       ("backward()", "import torch; x=torch.ones(2,requires_grad=True); (x*2).sum().backward()")):
        rc, out = run_py("import sys; sys.path.insert(0,'tools/deploy'); import common as C; C.no_training(); " + code +
                         "; print('TRAINED')", cwd=ROOT)
        rec("T5", "all", "refused", rc != 0 and "TrainingAttempted" in out and "TRAINED" not in out,
            f"{what} under no_training() -> " + (out.splitlines()[-1] if out else ""))
    shutil.rmtree(TREE)


DET = {   # project -> code printing a sha256 of the forecasts for fixed inputs
    "oil": """
        import sys, hashlib, numpy as np, pandas as pd; sys.path.insert(0,'tools/commodity'); import models as M
        df = pd.read_csv('data/commodity/raw/DCOILWTICO.csv'); v = pd.to_numeric(df.iloc[:,1], errors='coerce').dropna().to_numpy(float)
        q = M.get('wm_sv').batch([v[i-1024:i] for i in range(len(v)-40, len(v), 4)], 20, [0.1,0.5,0.9])
        print('H', hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(), repr(q.ravel()[:3].tolist()))""",
    "power": """
        import sys, hashlib, numpy as np; sys.path.insert(0,'tools/power'); import models as M, data as D
        d,P,C,f = D.panel(); cs = [D.build_context(i,736,d,P,C,f) for i in range(len(d)-20, len(d), 4)]
        q = M.get('wm_power').batch([c['prices'] for c in cs],[c['covs'] for c in cs],24,[0.1,0.5,0.9],dates=[{'ctx_dates':c['ctx_dates'],'target_day':c['target_day']} for c in cs])
        print('H', hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(), repr(np.asarray(q).ravel()[:3].tolist()))""",
    "gas": """
        import sys, hashlib, numpy as np; sys.path.insert(0,'tools/gas'); import storage as S, models as M
        w,o,f = S.load(); ks = S.target_weeks('test', w)[-8:]
        cs = [S.build_context(int(k),'thu',w,o,f) for k in ks]
        q = np.asarray(M.get('wm_gas_ens').storage(cs,[0.1,0.5,0.9],'l48'), float)
        print('H', hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(), repr(q.ravel()[:3].tolist()))""",
    "fin": """
        import os, sys, hashlib, numpy as np; os.environ['ALMA_FIN_TEST_GO']='1'; sys.path.insert(0,'tools/fin'); import harness as H, models as M
        P = H.Panel('crypto', allow_test=True); q = H.run(P, M.get('wm_fin'), list(range(len(P.days)-12, len(P.days)-2, 3)))['q']
        print('H', hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(), repr(q.ravel()[:3].tolist()))""",
}


def determinism_tests():
    for proj, code in DET.items():
        outs = {}
        for tag, thr in (("run1_t1", "1"), ("run2_t1", "1"), ("run3_t4", "4")):
            rc, out = run_py(code, cwd=ROOT, env={"OMP_NUM_THREADS": thr, "MKL_NUM_THREADS": thr, "ALMA_THREADS": thr})
            line = [l for l in out.splitlines() if l.startswith("H ")]
            outs[tag] = line[-1].split(" ", 2) if line else ["H", f"ERROR rc={rc}: {out[-200:]}", ""]
        same = outs["run1_t1"][1] == outs["run2_t1"][1] and not outs["run1_t1"][1].startswith("ERROR")
        rec("D1", proj, "bit-identical", same, f"two runs, 1 thread: sha {outs['run1_t1'][1][:16]} vs {outs['run2_t1'][1][:16]}")
        same4 = outs["run1_t1"][1] == outs["run3_t4"][1]
        RES.append({"test": "D2", "project": proj, "expected": "report", "ok": True,
                    "detail": f"1 vs 4 threads: {'bit-identical' if same4 else 'differ in the last bits: ' + outs['run1_t1'][2] + ' vs ' + outs['run3_t4'][2]}"})
        print(f"[INFO] D2   {proj:6s} 1 vs 4 threads bit-identical: {same4}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    tamper_tests()
    determinism_tests()
    json.dump(RES, open(os.path.join(OUT, "RESULTS.json"), "w"), indent=1)
    L = ["# V2 — Tamper and determinism tests\n", "| test | project | expected | result | detail |", "|---|---|---|---|---|"]
    for r in RES:
        L.append(f"| {r['test']} | {r['project']} | {r['expected']} | {'**as expected**' if r['ok'] else '**NOT as expected**'} | "
                 f"{r['detail'].replace('|', '/')[:220]} |")
    n_ok = sum(r["ok"] for r in RES)
    L.append(f"\n**{n_ok}/{len(RES)} as expected.**")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_ok}/{len(RES)} as expected")


if __name__ == "__main__":
    main()
