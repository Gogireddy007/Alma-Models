"""Auditor probe for docs/audit_sprint.md (investor sprint, 2026-10-31). Read-only, CPU, small.

  wm      world-model test: rollout (wm_ssm_none_s*) vs direct twin (wm_ssm_direct_s*).
          (a) checkpoint configs: train_sha256 / train_series / datasets / drop hash / env diff / state_dict shapes;
          (b) per-task tables re-derived from the logs (gmean wql_rel, family gmean = val/family_wql_rel);
          (c) paired per-seed deltas, seed-set sensitivity, per-task paired log ratio with a family-cluster bootstrap.
  ssmbit  bit-identity of wm/ssm.py versions (default env) by building the family with the same torch seed and running
          encode/decode for every ablation on random tokens: old committed (b464a5de, 9c572d5) vs HEAD vs working tree.

    nice -n 10 .venv/bin/python tools/audit_sprint_probe.py wm
    nice -n 10 .venv/bin/python tools/audit_sprint_probe.py ssmbit
"""
import glob, json, math, os, re, subprocess, sys, tempfile, importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def table(path):
    lines = open(path, errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith("dataset") and "wql_rel" in l]
    if not starts:
        return None, {}
    hdr = lines[starts[-1]].split()
    rows = {}
    for l in lines[starts[-1] + 1:]:
        p = l.split()
        if len(p) != len(hdr):
            break
        rows[p[0]] = dict(zip(hdr, p))
    printed = {}
    for l in lines:
        m = re.match(r"(val|family)_(wql_rel|mase_rel|cov80_error):\s+([\d.]+)", l)
        if m:
            printed[f"{m.group(1)}_{m.group(2)}"] = float(m.group(3))
    return rows, printed


gm = lambda xs: float(np.exp(np.mean(np.log(np.clip(np.asarray(xs, float), 1e-6, None)))))


def derive(rows):
    w = [float(r["wql_rel"]) for r in rows.values()]
    fam = {}
    for r in rows.values():
        fam.setdefault(r["family"], []).append(float(r["wql_rel"]))
    return gm(w), gm([gm(v) for v in fam.values()]), len(rows), len(fam)


def wm():
    import torch
    runs = {f"{a}_s{s}": f"wm_ssm_{a}_s{s}" for a in ("none", "direct") for s in range(5)}
    print("== (a) checkpoint configs")
    cfgs = {}
    for k, r in runs.items():
        p = os.path.join(ROOT, "data", "ckpt", f"{r}.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        shapes = {n: tuple(t.shape) for n, t in ck["model"].items()}
        cfgs[k] = (c, shapes, ck.get("complete"), ck.get("step"))
        print(f"  {r:20s} complete={ck.get('complete')} step={ck.get('step')} train_sha={c['train_sha256'][:12]} "
              f"series={c['train_series']} n_ds={len(c['train_datasets'])} drop={c['train_drop_sha256'][:12]} "
              f"params={sum(int(np.prod(s)) for s in shapes.values())} git={c.get('git')}")
        del ck
    if cfgs:
        ref = cfgs[next(iter(cfgs))]
        for k, (c, sh, *_ ) in cfgs.items():
            env_d = {e for e in set(c["env"]) | set(ref[0]["env"]) if c["env"].get(e) != ref[0]["env"].get(e)}
            other = [x for x in c if x not in ("env", "git") and c[x] != ref[0][x]]
            print(f"  {k:10s} env keys differing from {next(iter(cfgs))}: {sorted(env_d)}; other cfg diffs: {other}; "
                  f"same state_dict shapes: {sh == ref[1]}")
        print(f"  train datasets: {sorted(ref[0]['train_datasets'])}")

    print("== (b) per-task re-derivation from logs")
    tabs = {}
    for k, r in runs.items():
        p = os.path.join(ROOT, "runs", f"{r}.log")
        if not os.path.exists(p):
            continue
        rows, pr = table(p)
        if not rows or len(rows) != 30:
            print(f"  {r}: no complete 30-task table yet")
            continue
        tabs[k] = rows
        w, f, nt, nf = derive(rows)
        print(f"  {r:20s} wql_rel {w:.4f} (printed {pr.get('val_wql_rel')})  family {f:.4f} "
              f"(printed {pr.get('family_wql_rel')})  tasks={nt} fams={nf}")
    nf = {k: {t: v["n_forecasts"] for t, v in r.items()} for k, r in tabs.items()}
    if nf:
        print(f"  identical task list and n_forecasts across runs: {len({json.dumps(v, sort_keys=True) for v in nf.values()}) == 1}")

    print("== (c) paired comparison")
    val = {k: derive(v)[0] for k, v in tabs.items()}
    for s in range(5):
        n, d = val.get(f"none_s{s}"), val.get(f"direct_s{s}")
        print(f"  seed {s}: none {n}  direct {d}  delta(direct-none) {None if n is None or d is None else round(d - n, 4)}")
    for name, seeds in (("{0,3,4}", (0, 3, 4)), ("{1,2,3,4} (fresh, excl. selection seed 0)", (1, 2, 3, 4)),
                        ("{0..4}", range(5)), ("{1,2}", (1, 2))):
        v = [val[f"none_s{s}"] for s in seeds if f"none_s{s}" in val]
        print(f"  rollout mean over seeds {name}: {np.mean(v):.4f} (n={len(v)})")
    paired = [s for s in range(5) if f"none_s{s}" in tabs and f"direct_s{s}" in tabs]
    if paired:
        tasks = sorted(tabs[f"none_s{paired[0]}"])
        fam = {t: tabs[f"none_s{paired[0]}"][t]["family"] for t in tasks}
        lr = np.array([[math.log(float(tabs[f"direct_s{s}"][t]["wql_rel"]) / float(tabs[f"none_s{s}"][t]["wql_rel"]))
                        for t in tasks] for s in paired]).mean(0)          # seed-mean log ratio per task (>0: rollout better)
        fams = sorted(set(fam.values()))
        rng = np.random.default_rng(0)
        idx = {f: [i for i, t in enumerate(tasks) if fam[t] == f] for f in fams}
        boots = []
        for _ in range(10000):
            pick = rng.integers(0, len(fams), len(fams))
            ii = [i for p in pick for i in idx[fams[p]]]
            boots.append(lr[ii].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wins = int((lr > 0.02).sum()); loss = int((lr < -0.02).sum())
        print(f"  paired seeds {paired}: mean log(direct/none) over tasks {lr.mean():+.4f} "
              f"(family-cluster 95% CI {lo:+.4f}..{hi:+.4f}); rollout wins/ties/losses (+-2%) {wins}/{30 - wins - loss}/{loss}")
        fl = {f: lr[idx[f]].mean() for f in fams}
        print("  per family log(direct/none): " + ", ".join(f"{f} {v:+.3f}" for f, v in sorted(fl.items(), key=lambda x: -x[1])))


def ssmbit():
    import torch
    sys.path.insert(0, ROOT)
    for k in [k for k in os.environ if k.startswith("TSAR_")]:
        del os.environ[k]
    from wm.base import WMConfig, Context
    srcs = {"9c572d5 (b464a5de)": subprocess.check_output(["git", "-C", ROOT, "show", "9c572d5:wm/ssm.py"]),
            "HEAD": subprocess.check_output(["git", "-C", ROOT, "show", "HEAD:wm/ssm.py"]),
            "worktree": open(os.path.join(ROOT, "wm", "ssm.py"), "rb").read()}
    tmp = tempfile.mkdtemp()
    out = {}
    for abl in ("none", "direct", "copy", "zero"):
        cfg = WMConfig(patch=8, context=256, max_h=64, k_steps=8, n_patch=40, nq=9, tok_dim=19, d_model=96,
                       n_layers=2, n_heads=4, width=1.0, abl=abl)
        g = torch.Generator().manual_seed(7)
        B, n = 3, 40
        ctx = Context(tok=torch.randn(B, n, 19, generator=g), x=torch.randn(B, n * 8, generator=g),
                      loc=torch.randn(B, n, generator=g), scale=torch.rand(B, n, generator=g) + 0.5,
                      cnt=torch.arange(1, n + 1).float().repeat(B, 1) * 8,
                      ref=torch.randn(B, n, 64, generator=g))
        b = torch.arange(B).repeat_interleave(5); j = torch.tensor([3, 10, 20, 30, 39] * B)
        res = {}
        for tag, src in srcs.items():
            p = os.path.join(tmp, f"ssm_{abs(hash(tag))}.py")
            open(p, "wb").write(src)
            spec = importlib.util.spec_from_file_location(f"ssm_{abs(hash(tag))}", p)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            torch.manual_seed(0)
            fam = mod.SSMFamily(cfg)
            with torch.no_grad():
                q, _ = fam.decode(fam.encode(ctx), ctx, b, j)
            res[tag] = ({k: v.clone() for k, v in fam.state_dict().items()}, q)
        base = res["9c572d5 (b464a5de)"]
        for tag, (sd, q) in res.items():
            same_sd = sd.keys() == base[0].keys() and all(torch.equal(sd[k], base[0][k]) for k in sd)
            print(f"  abl={abl:6s} {tag:20s} params={sum(v.numel() for v in sd.values())} identical_init={same_sd} "
                  f"max|dq| vs 9c572d5={float((q - base[1]).abs().max()):.1e}")


if __name__ == "__main__":
    torch_threads = os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"wm": wm, "ssmbit": ssmbit}[sys.argv[1] if len(sys.argv) > 1 else "wm"]()
