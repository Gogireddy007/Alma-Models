"""
Lag-searching growth-correlation scan: TRAIN vs VAL/TEST, and the TSAR_BENCH_EXTRA stamp.

Reconstruction (committed, per audit_v2 L-3) of the scratch scan that justified bench.TRAIN_DROP
(docs/bench_integrity.md section 1). Exact/znorm hashing (bench.overlap) misses re-releases of the
same series with another vintage, deflator or rounding. This scan searches EVERY alignment of each
probe series against each TRAIN series of the same frequency and reports pairs whose first
differences have Pearson r >= --thr over >= --min-overlap aligned points.

Probe = the data each task exposes (bench._series_spans): VAL contexts + targets; TEST contexts
only (target-free: no forecast, no score). TRAIN = the full corpus before TRAIN_DROP, like
bench.overlap. A hit already in TRAIN_DROP is reported as "dropped"; anything else is a residual
hit that must be reviewed and, if real, added to TRAIN_DROP.

    uv run tools/overlap_scan.py lag --split val --tasks solar_1W              # any VAL task
    TSAR_SPLIT=test uv run tools/overlap_scan.py lag --split test               # macro + world_* tasks
    uv run tools/overlap_scan.py selftest                                       # positive/negative controls
    TSAR_SPLIT=test uv run tools/overlap_scan.py stamp-extra                    # gate for TSAR_BENCH_EXTRA
    TSAR_SPLIT=test uv run tools/overlap_scan.py stamp-gep --out runs/overlap_gep_hits.json   # gate for TSAR_BENCH_GEP
    uv run tools/overlap_scan.py stamp-gep --splits val --folders HZMETRO --tasks ETT_15T   # partial: never stamps

stamp-extra runs bench.overlap (exact + znorm, L=32) on VAL and TEST and this lag scan on the TEST
macro/world tasks, all with TRAIN_EXTRA included, and writes data/bench/overlap_extra.json only if
no residual hit remains. bench.load_train(extra=True) refuses to run without a current stamp.
Heavy (hashes the full corpus): queue it, do not run it on a loaded machine.

stamp-gep does the same for the GiftEvalPretrain subset (bench.TRAIN_GEP, every full univariate series of the 25
verified shards, before caps and GEP_DROP): bench.overlap (exact + znorm, L=32) against VAL contexts + targets and
TEST contexts (target-free), and this lag scan of every VAL and TEST task whose frequency class is a GEP class
(5min, 15min, 30min, H) against the GEP series of that class. Long series are cut into overlapping segments
(overlap >= the longest probe, so every alignment of a probe lies inside one segment; Pearson r is unchanged by the
per-segment standardisation) and scanned in batches, so memory stays bounded. Sub-daily classes need
n >= max(--min-overlap, --min-overlap-periods x season) aligned points: smooth diurnal series (temperature, load)
correlate by phase over a day or two, while a re-released copy matches over its whole span. Writes
data/bench/overlap_gep.json only if no residual hit remains (bench.GEP_DROP-aware). Every lag hit is then re-tested
over all lags of that pair (confirm_pair; docs/gep_overlap_review.md): the aligned stretch must also have >= min_overlap
steps that bend (change slope) in both series, r >= --thr on the moving steps winsorised at 5 robust SDs, and level r >= --thr. Hits that
fail are kept in the --out file with their diagnostics as "filtered", not residual (--no-confirm turns this off).
`confirm-gep` re-tests a saved --out file; `selftest --known-tp` checks the re-test keeps the stamp-extra true positives.

Method per (probe p, train t), both as first differences (or log differences, --transform logdiff),
each standardised over the whole series (Pearson r on a sub-window is unaffected): for every lag,
masked FFT cross-correlations give n, sum p, sum t, sum p^2, sum t^2, sum pt over the overlap, hence
r(lag). The maximum over lags with n >= min_overlap is kept. Train series are processed in chunks so
memory stays bounded (~chunk x 2^ceil(log2(Lmax+Lp)) x 16 bytes x 6).
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import bench as BN  # noqa: E402

# Default TEST probe set: macro / demographic families with anonymised M-competition analogues
# (docs/audit_v2.md L-2).
TEST_MACRO = ["us_consumption_1M", "us_consumption_1Q", "us_consumption_1Y", "gvar",
              "world_co2_emissions", "world_life_expectancy", "world_tourism"]
# Documented false positives that do not block the stamp (docs/bench_integrity.md section 1).
KNOWN_FP = {("boomlet_772", "dominick")}
FREQ_CLASS = {"Y": "Y", "A": "Y", "Q": "Q", "M": "M", "W": "W", "D": "D", "B": "D", "H": "H"}


def _fclass(freq):
    return FREQ_CLASS.get(freq, freq)


def _prep(y, transform):
    y = np.asarray(y, np.float64)
    if transform == "logdiff":
        y = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
    d = np.diff(y)
    m = np.isfinite(d)
    if m.sum() < 3:
        return None
    mu, sd = d[m].mean(), d[m].std()
    d = np.where(m, (d - mu) / (sd if sd > 0 else 1.0), 0.0)
    return d, m.astype(np.float64)


def _xcorr(TF, PF, nfft, workers=None):
    """irfft(T * conj(P)): entry [i, k] = sum_j t_i[j + k] p[j] with circular wrap (nfft is large
    enough that no wrap overlaps). Lags k in [-(Lp-1), Lt-1] map to index k mod nfft."""
    if workers:                                   # scipy.fft, multithreaded (stamp-gep); same math, other rounding
        import scipy.fft as sf
        return sf.irfft(TF * np.conj(PF)[None, :], nfft, axis=1, workers=workers)
    return np.fft.irfft(TF * np.conj(PF)[None, :], nfft, axis=1)


def _rfft(a, nfft, workers=None, axis=-1):
    if workers:
        import scipy.fft as sf
        return sf.rfft(a, nfft, axis=axis, workers=workers)
    return np.fft.rfft(a, nfft, axis=axis)


def scan_pairs(probes, train, min_overlap=60, thr=0.95, chunk=512, transform="diff", workers=None):
    """probes: [(probe_id, y)], train: [(train_id, y)]. Returns [(probe_id, train_id, r, n, lag)]
    for the best lag of each pair with r >= thr. lag = start of the probe within the train series.
    workers: None = numpy FFT (the stamp-extra path, unchanged); k = scipy.fft with k threads."""
    P = [(pid, _prep(y, transform)) for pid, y in probes]
    P = [(pid, x) for pid, x in P if x is not None and x[1].sum() >= min_overlap]
    T = [(tid, _prep(y, transform)) for tid, y in train]
    T = [(tid, x) for tid, x in T if x is not None and x[1].sum() >= min_overlap]
    if not P or not T:
        return []
    Lp = max(len(x[0]) for _, x in P)
    hits = []
    order = np.argsort([len(x[0]) for _, x in T])         # similar lengths per chunk: less padding
    for s in range(0, len(T), chunk):
        part = [T[i] for i in order[s:s + chunk]]
        Lt = max(len(x[0]) for _, x in part)
        nfft = 1 << int(np.ceil(np.log2(Lt + Lp)))
        tv = np.zeros((len(part), Lt)); tm = np.zeros((len(part), Lt))
        for i, (_, (d, m)) in enumerate(part):
            tv[i, :len(d)], tm[i, :len(m)] = d, m
        F_m, F_v, F_v2 = (_rfft(a, nfft, workers, axis=1) for a in (tm, tv, tv * tv))
        lens = np.array([len(x[0]) for _, x in part])
        for pid, (pd_, pm) in P:
            G_m, G_v, G_v2 = (_rfft(a, nfft, workers) for a in (pm, pd_, pd_ * pd_))
            n = np.rint(_xcorr(F_m, G_m, nfft, workers))
            st, st2 = _xcorr(F_v, G_m, nfft, workers), _xcorr(F_v2, G_m, nfft, workers)
            sp, sp2 = _xcorr(F_m, G_v, nfft, workers), _xcorr(F_m, G_v2, nfft, workers)
            spt = _xcorr(F_v, G_v, nfft, workers)
            with np.errstate(invalid="ignore", divide="ignore"):
                vt, vp = n * st2 - st * st, n * sp2 - sp * sp
                r = (n * spt - st * sp) / np.sqrt(vt * vp)
            r[(n < min_overlap) | (vt <= 1e-9 * n * n) | (vp <= 1e-9 * n * n) | ~np.isfinite(r)] = -np.inf
            best = r.argmax(1)
            rb = r[np.arange(len(part)), best]
            for i in np.flatnonzero(rb >= thr):
                k = int(best[i])
                lag = k if k < nfft - len(pd_) else k - nfft
                if lag < lens[i]:
                    hits.append((pid, part[i][0], float(rb[i]), int(n[i, k]), lag))
    return hits


def _train_by_class(extra, only=None):
    """{freq class: [((dataset, key), y)]} over the FULL corpus (no subsample, before TRAIN_DROP)."""
    out = {}
    for (name, season, freq, h), is_extra in BN._train_specs(extra):
        if only and name not in only:
            continue
        df = pd.read_parquet(BN._train_file(name, is_extra))
        out.setdefault(_fclass(freq), []).extend(((name, k), y) for k, y in enumerate(BN._target_arrays(df)))
        del df
    return out


def lag_scan(split, tasks=None, extra=False, train_only=None, min_overlap=60, min_overlap_yearly=24,
             thr=0.95, transform="diff", any_freq=False, verbose=True):
    """Returns {"hits": [...], "residual": [...]} with dict rows. TEST: contexts only."""
    BN._guard(split, f"overlap_scan.lag({split}, target_free={split == 'test'})")
    if tasks is None:
        tasks = TEST_MACRO if split == "test" else [t["name"] for t in BN.SPLIT_TASKS["val"]]
    specs = [s for s in BN.SPLIT_TASKS[split] if s["name"] in tasks]
    missing = set(tasks) - {s["name"] for s in specs}
    if missing:
        sys.exit(f"unknown {split} tasks: {sorted(missing)}")
    bank = _train_by_class(extra, train_only)
    rows = []
    for spec in specs:
        t = BN._build_task(spec)
        fc = _fclass(t["freq"])
        mo = min_overlap_yearly if fc == "Y" else min_overlap
        train = [x for c, v in bank.items() if any_freq or c == fc for x in v]
        probes = list(BN._series_spans(t, include_target=split != "test"))
        t0 = time.time()
        hits = scan_pairs(probes, train, mo, thr, transform=transform)
        for sid, (ds, key), r, n, lag in hits:
            dropped = key in set(BN.TRAIN_DROP.get(ds, ()))
            rows.append({"task": t["name"], "series": sid, "train": ds, "key": key, "r": round(r, 4),
                         "n": n, "lag": lag, "dropped": dropped})
        if verbose:
            nres = sum(1 for h in rows if h["task"] == t["name"] and not h["dropped"])
            print(f"  {t['name']:28s} freq={fc} probes={len(probes):4d} train={len(train):6d} "
                  f"min_overlap={mo} hits={sum(1 for h in rows if h['task'] == t['name']):3d} residual={nres}"
                  f"  ({time.time() - t0:.0f}s)" + ("   <-- INVESTIGATE" if nres else ""), flush=True)
    for h in rows:
        if verbose:
            print(f"      {h['task']}/{h['series']} ~ {h['train']}[{h['key']}] r={h['r']} n={h['n']} lag={h['lag']}"
                  + ("  (in TRAIN_DROP)" if h["dropped"] else ""))
    return {"hits": rows, "residual": [h for h in rows if not h["dropped"]]}


def selftest():
    """Controls on synthetic series only (no benchmark data): an affinely rescaled, noisy, shifted copy
    must be found at the right lag; an independent random walk must not."""
    rng = np.random.default_rng(0)
    train = [(("syn", k), np.cumsum(rng.normal(0, 1, 300 + 7 * k)) + 50) for k in range(40)]
    src = train[17][1]
    copy = 7.3 * src[40:200] + 100 + rng.normal(0, 0.05, 160)
    copy[[5, 90]] = np.nan                                         # NaNs are masked, not fatal
    walk = np.cumsum(rng.normal(0, 1, 160))
    hits = scan_pairs([("copy", copy), ("walk", walk)], train, min_overlap=60, thr=0.95, chunk=16)
    found = [(p, t, round(r, 3), n, lag) for p, t, r, n, lag in hits]
    print("selftest hits:", found)
    assert any(p == "copy" and t == ("syn", 17) and lag == 40 for p, t, _, _, lag in hits), "copy not found"
    assert not any(p == "walk" for p, *_ in hits), "random walk matched"
    assert all(t == ("syn", 17) for p, t, *_ in hits if p == "copy"), "copy matched the wrong series"
    # strict re-test (confirm_pair, used by stamp-gep): the copy must survive; the two classic growth-correlation
    # false positives must be flagged by the scan and then rejected
    c = confirm_pair(copy, src, 60, 24)
    assert c["confirmed"] and c.get("lag_confirmed") == 40, ("copy rejected by the re-test", c)
    flat_t = np.full(1000, 5.0); flat_t[100:400] = 7.0                  # provisioned capacity: two steps, else flat
    flat_p = np.zeros(600); flat_p[50:350] = 1.0                       # same two steps 50 points into the probe
    assert scan_pairs([("flat", flat_p)], [(("flat", 0), flat_t)], 60, 0.95), "flat control not flagged"
    assert not confirm_pair(flat_p, flat_t, 60, 24)["confirmed"], "flat/step control confirmed"
    sp_t, sp_p = rng.normal(0, 1, 1000), rng.normal(0, 1, 600)         # independent noise sharing ONE big jump
    sp_t[300:] += 200.0; sp_p[200:] += 200.0
    assert scan_pairs([("spike", sp_p)], [(("spike", 0), sp_t)], 60, 0.95), "single-jump control not flagged"
    assert not confirm_pair(sp_p, sp_t, 60, 24)["confirmed"], "single-jump control confirmed"
    q = np.round(copy / 3.0) * 3.0                                   # a coarsely re-quantised copy still passes
    assert confirm_pair(q, src, 60, 24)["confirmed"], "re-quantised copy rejected"
    ramp_t = np.cumsum(rng.normal(0, 1, 1000))                         # linearly interpolated gaps in both series,
    ramp_t[200:300] = ramp_t[199] + np.cumsum(np.r_[np.full(50, 1.0), np.full(50, 3.0)])   # kinks aligned: r = 1
    ramp_p = np.cumsum(rng.normal(0, 1, 600))                         # over the scan's 99-step sub-window
    ramp_p[:100] = np.cumsum(np.r_[np.full(50, 0.1), np.full(50, 0.3)])
    c = confirm_pair(ramp_p, ramp_t, 60, 24, lag=200, n_scan=99, r_scan=1.0)
    assert c["r_diff"] > 0.99 and c["n"] == 99 and not c["confirmed"], ("interpolated-ramp control", c)
    long_t = np.cumsum(rng.normal(0, 1, 2000))                         # a copy of only the first 200 of 600 aligned
    part = np.concatenate([2.0 * long_t[500:700] + 3, np.cumsum(rng.normal(0, 1, 400)) + 9e3])  # steps (segment edge)
    assert confirm_pair(part, long_t, 60, 24, lag=500, n_scan=199, r_scan=1.0)["confirmed"], "sub-window copy rejected"
    print("selftest OK (scan + strict re-test controls)")


# Real duplicates found by stamp-extra (docs/extra_overlap_review.md): TRAIN_EXTRA re-releases of BEA series in TEST
# us_consumption (task, series, train dataset, key). Any stricter re-test of lag hits must still confirm all of them.
KNOWN_TP_EXTRA = [("us_consumption_1Y", "gasoline_and_other_energy_goods", "m4_yearly", k)
                  for k in (3644, 3665, 3666, 3563, 14189, 14194, 14195, 14201, 14007, 14009, 14011, 14119, 14120,
                            12384, 13113)] + [
    ("us_consumption_1Y", "personal_consumption_expenditures", "m4_yearly", 9261),
    ("us_consumption_1Y", "motor_vehicles_and_parts", "m4_yearly", 13112),
    ("us_consumption_1M", "personal_consumption_expenditures", "m4_monthly", 6216)]


def selftest_known_tp():
    """confirm_pair must keep every KNOWN_TP_EXTRA pair (reads the TEST us_consumption contexts, target-free, logged,
    exactly as stamp-extra does; needs TSAR_SPLIT=test and the TRAIN_EXTRA m4 files)."""
    BN._guard("test", "overlap_scan.selftest_known_tp(target_free=True)")
    tasks = sorted({t for t, *_ in KNOWN_TP_EXTRA})
    probes = {}
    for spec in BN.SPLIT_TASKS["test"]:
        if spec["name"] in tasks:
            t = BN._build_task(spec)
            for sid, y in BN._series_spans(t, include_target=False):
                probes[(t["name"], sid)] = (y, _fclass(t["freq"]))
    bank = _train_by_class(True, sorted({d for _, _, d, _ in KNOWN_TP_EXTRA}))
    bank = {k: y for v in bank.values() for k, y in v}
    bad = []
    for task, sid, ds, key in KNOWN_TP_EXTRA:
        y, fc = probes[(task, sid)]
        c = confirm_pair(y, bank[(ds, key)], 24 if fc == "Y" else 60, {"Y": 1, "M": 12}.get(fc, 1))
        print(f"  known TP {task}/{sid} ~ {ds}[{key}]: confirmed={c['confirmed']} r={c['r_diff']:.3f} n={c['n']} "
              f"n_active={c['n_active']} n_bends={c['n_bends']} r_active_wins={c['r_active_wins']:.3f} r_level={c['r_level']:.3f} "
              f"lag={c['lag0']}", flush=True)
        if not c["confirmed"]:
            bad.append((task, sid, ds, key))
    assert not bad, f"strict re-test hides known true positives: {bad}"
    print(f"known-TP selftest OK ({len(KNOWN_TP_EXTRA)} pairs confirmed)")


def _overlap_residual(report, drop=None, known_fp=None):
    drop = BN.TRAIN_DROP if drop is None else drop
    known_fp = KNOWN_FP if known_fp is None else known_fp
    res = []
    for task, r in report.items():
        for tag in ("exact", "znorm"):
            for ds, key, n in r[tag]:
                if key in set(drop.get(ds, ())) or (task, ds) in known_fp:
                    continue
                res.append({"task": task, "tag": tag, "train": ds, "key": key, "runs": n})
    return res


def stamp_extra(args):
    if os.environ.get("TSAR_SPLIT") != "test":
        sys.exit("stamp-extra scans TEST contexts (target-free) and needs TSAR_SPLIT=test; it is logged.")
    for name, *_ in BN.TRAIN_EXTRA:                     # never stamp a partial download
        if not os.path.exists(os.path.join(BN.BENCH_DIR, "train", f"{name}.parquet")):
            sys.exit(f"TRAIN_EXTRA file missing: {name}. Run `uv run tools/bench.py download --split train --extra`.")
    key = BN.extra_scan_key()
    print("== exact/znorm hashes, VAL (contexts + targets), TRAIN incl. extra")
    rv = _overlap_residual(BN.overlap("val", L=32, extra=True))
    print("== exact/znorm hashes, TEST (contexts only), TRAIN incl. extra")
    rt = _overlap_residual(BN.overlap("test", L=32, extra=True))
    print("== lag scan, TEST macro/world tasks, TRAIN incl. extra")
    rl = lag_scan("test", extra=True, min_overlap=args.min_overlap, min_overlap_yearly=args.min_overlap_yearly,
                  thr=args.thr)["residual"]
    residual = {"val_hash": rv, "test_hash": rt, "test_lag": rl}
    n = sum(len(v) for v in residual.values())
    if n:
        print(json.dumps(residual, indent=1, default=str))
        sys.exit(f"{n} residual hit(s): review them, add real duplicates to bench.TRAIN_DROP, re-run. No stamp written.")
    if BN.extra_scan_key() != key:
        sys.exit("TRAIN_EXTRA / TRAIN_DROP changed during the scan; re-run. No stamp written.")
    stamp = {"result": "clean", "key": key, "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": BN._git_head(),
             "scans": {"hash": {"L": 32, "splits": ["val", "test"]},
                       "lag": {"tasks": TEST_MACRO, "thr": args.thr, "min_overlap": args.min_overlap,
                               "min_overlap_yearly": args.min_overlap_yearly, "transform": "diff"}},
             "known_false_positives": sorted(map(list, KNOWN_FP))}
    json.dump(stamp, open(BN.EXTRA_STAMP, "w"), indent=1)
    print(f"clean: wrote {BN.EXTRA_STAMP}")


# --- stamp-gep (TSAR_BENCH_GEP) ------------------------------------------------------------------------------
GEP_KNOWN_FP = set()          # (task, folder) pairs reviewed as false positives; none yet
GEP_SEASON = {"5min": 288, "15min": 96, "30min": 48, "H": 24}


def _segments(train, Lp_raw, nfft):
    """Cut each (tid, y) into raw segments of S = nfft - Lp_raw points starting every S - Lp_raw points: any stretch
    of <= Lp_raw points (every probe alignment) lies inside one segment, and (S - 1) + (Lp_raw - 1) <= nfft keeps
    scan_pairs at this nfft. Yields ((tid, offset), segment view)."""
    S = nfft - Lp_raw
    step = S - Lp_raw
    assert step > 0, (nfft, Lp_raw)
    for tid, y in train:
        n = len(y)
        if n <= S:
            yield (tid, 0), y
            continue
        for a in range(0, n, step):
            yield (tid, a), y[a:a + S]
            if a + S >= n:
                break


def _gep_probes(splits, tasks=None):
    """{class: [((split, task, sid), y)]}: VAL contexts + targets, TEST contexts only, for the GEP classes."""
    probes = {}
    for split in splits:
        BN._guard(split, f"overlap_scan.gep_lag({split}, target_free={split == 'test'})")
        for spec in BN.SPLIT_TASKS[split]:
            if tasks is not None and spec["name"] not in tasks:
                continue
            t = BN._build_task(spec)
            fc = _fclass(t["freq"])
            if fc in GEP_SEASON:
                for sid, y in BN._series_spans(t, include_target=split != "test"):
                    probes.setdefault(fc, []).append(((split, t["name"], sid), np.asarray(y, np.float32)))
            del t
    return probes


def gep_lag_class(fc, P, folders=None, min_overlap=60, min_overlap_periods=2, thr=0.95, chunk=128, batch=512,
                  workers=4, verbose=True):
    """Lag scan of the probes P of class fc against every GEP series of that class (segmented, in batches).
    Returns (hit rows, stats); one row per (probe, GEP series): its best alignment."""
    flds = [f for f, _, _, fq, _ in BN.TRAIN_GEP if fq == fc and (not folders or f in folders)]
    mo = max(min_overlap, min_overlap_periods * GEP_SEASON[fc])
    Lp_raw = max(len(y) for _, y in P)
    # probes grouped by length: each group gets its own segment length, so short probes (most H probes are
    # ~2.5k points) are not padded to the FFT size of the longest probe of the class
    groups = {}
    for pid, y in P:
        groups.setdefault(max(4096, 1 << int(np.ceil(np.log2(4 * len(y))))), []).append((pid, y))
    nfft = max(groups)
    t0, n_seg, n_pts, pairs, best = time.time(), 0, 0, 0, {}
    for g_nfft, G in sorted(groups.items()):
        g_lp = max(len(y) for _, y in G)
        for f in flds:
            seg_iter = _segments(((k, y) for _, k, y in BN.gep_series([f])), g_lp, g_nfft)
            while True:
                part = [s for _, s in zip(range(batch), seg_iter)]
                if not part:
                    break
                n_seg += len(part); n_pts += sum(len(y) for _, y in part); pairs += len(part) * len(G)
                for pid, (k, a), r, n, lag in scan_pairs(G, part, mo, thr, chunk=chunk, workers=workers):
                    key = (pid, f, k)
                    if key not in best or r > best[key][0]:
                        best[key] = (r, n, lag + a)
            if verbose:
                print(f"    {fc} group nfft={g_nfft} ({len(G)} probes) {f}: pairs so far {pairs} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    rows = [{"split": pid[0], "task": pid[1], "series": pid[2], "train": f, "key": int(k), "r": round(r, 4), "n": n,
             "lag": lag, "class": fc, "dropped": k in set(BN.GEP_DROP.get(f, ())) or (pid[1], f) in GEP_KNOWN_FP}
            for (pid, f, k), (r, n, lag) in best.items()]
    sec = time.time() - t0
    stats = {"probes": len(P), "probe_max_len": Lp_raw, "nfft": nfft, "min_overlap": mo, "folders": flds,
             "groups": {str(k): len(v) for k, v in sorted(groups.items())}, "segments": n_seg, "segment_points": n_pts,
             "pairs": pairs, "seconds": round(sec, 1),
             "hits": len(best)}
    if verbose:
        print(f"  lag class={fc:5s} probes={len(P):5d} (max len {Lp_raw}) folders={len(flds)} segments={n_seg} "
              f"points={n_pts / 1e6:.1f}M pairs={stats['pairs']} nfft<={nfft} min_overlap={mo} hits={len(best)} "
              f"({sec:.0f}s, {stats['pairs'] / max(sec, 1e-9):.0f} pairs/s)", flush=True)
    return rows, stats


def _lag_r(t, p, nfft):
    """Pearson r and n for every lag of probe p against train t (both float64 with NaN = missing), as in scan_pairs:
    index k mod nfft <-> probe start k within t. Returns (r, n) arrays of length nfft (r = -inf where undefined)."""
    mt, mp = np.isfinite(t), np.isfinite(p)
    tv, pv = np.where(mt, t - np.nanmean(t) if mt.any() else 0.0, 0.0), np.where(mp, p - np.nanmean(p) if mp.any() else 0.0, 0.0)
    sd_t, sd_p = (tv[mt].std() if mt.any() else 0.0) or 1.0, (pv[mp].std() if mp.any() else 0.0) or 1.0
    tv, pv, mt, mp = tv / sd_t, pv / sd_p, mt.astype(np.float64), mp.astype(np.float64)
    F = [np.fft.rfft(a, nfft)[None, :] for a in (mt, tv, tv * tv)]
    G = [np.fft.rfft(a, nfft) for a in (mp, pv, pv * pv)]
    n = np.rint(_xcorr(F[0], G[0], nfft))[0]
    st, st2 = _xcorr(F[1], G[0], nfft)[0], _xcorr(F[2], G[0], nfft)[0]
    sp, sp2 = _xcorr(F[0], G[1], nfft)[0], _xcorr(F[0], G[2], nfft)[0]
    spt = _xcorr(F[1], G[1], nfft)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        vt, vp = n * st2 - st * st, n * sp2 - sp * sp
        r = (n * spt - st * sp) / np.sqrt(vt * vp)
    r[(n < 3) | (vt <= 1e-9 * n * n) | (vp <= 1e-9 * n * n) | ~np.isfinite(r)] = -np.inf
    return r, n


def _active(d):
    """Finite, non-zero steps of a difference series (a repeated value gives an exact 0 in float64 of float32 data)."""
    return np.isfinite(d) & (np.abs(d) > 1e-9 * (np.nanstd(d) if np.isfinite(d).any() else 1.0))


def _bends(d, act):
    """Steps j where the series changes slope (|d[j] - d[j-1]| > 1e-3 robust SDs of the moving steps). Flat stretches,
    single jumps and linearly interpolated gaps (constant d) have none; natural data bends at almost every step."""
    v = d[act]
    sc = (1.4826 * np.median(np.abs(v - np.median(v))) or v.std() or 1.0) if len(v) else 1.0
    b = np.zeros(len(d), bool)
    with np.errstate(invalid="ignore"):
        b[1:] = np.abs(np.diff(d)) > 1e-3 * sc
    return b & np.isfinite(d)


def _winsor(d, act, c=5.0):
    """Moving steps of d (NaN elsewhere), clipped to median +- c robust SDs (1.4826 MAD) of those steps. Monotone, so an
    affine copy keeps r ~ 1; one or a few huge shared jumps no longer dominate the correlation."""
    x = np.where(act, d, np.nan)
    v = x[np.isfinite(x)]
    if len(v) < 3:
        return x
    m = np.median(v)
    sc = 1.4826 * np.median(np.abs(v - m)) or v.std() or 1.0
    return np.clip(x, m - c * sc, m + c * sc)


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or a.std() <= 0 or b.std() <= 0:
        return -np.inf, int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def confirm_pair(p, t, mo, season, thr=0.95, lag=None, n_scan=None, r_scan=None):
    """Stricter re-test of one (probe, train) pair that the growth-correlation scan flagged.

    A lag counts as overlap only if, over the aligned stretch:
      1. r of first differences >= thr over n >= mo points (the scan's own criterion, unchanged);
      2. >= mo of those aligned steps BEND in both series (the difference changes: non-zero second difference).
         Flat, idle or piecewise-constant stretches (zeros, provisioned capacity, quantised idle) and linearly
         interpolated gaps (constant slope) give r ~ 1 from a few shared jumps or kinks; this removes them;
      3. r over those moving steps of the differences winsorised at median +- 5 robust SDs >= thr: the bulk of the
         steps must agree, not one or a few huge shared jumps (spiky cloud metrics coincide that way);
      4. r of levels >= thr (a copy, affine or not, matches in level too).
    Every condition is implied by an affine copy with small noise, so a real re-release keeps passing (selftest).
    The conditions are checked (a) directly in the time domain at the scan's own lag `lag` (lag = probe start within
    t, as in the hit rows) over the full overlap and over the scan's own sub-window of n_scan steps, and (b) at every lag via whole-series FFT cross-correlations. (b) is unreliable only where
    the train stretch is near-constant relative to the whole series, which condition 2 rejects anyway. Each condition
    only removes alignments, so a pair the scan did not flag cannot pass. Returns a dict: confirmed flag, the confirmed
    lag, and diagnostics at the scan's lag (the FFT argmax when lag is None) for the review."""
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)
    dp, dt = np.diff(p), np.diff(t)
    ap, at = _active(dp), _active(dt)
    wp, wt = _winsor(dp, ap), _winsor(dt, at)
    nfft = 1 << int(np.ceil(np.log2(len(t) + len(p))))
    r_d, n_d = _lag_r(dt, dp, nfft)
    r_a, n_a = _lag_r(wt, wp, nfft)
    r_l, _ = _lag_r(t, p, nfft)
    bp, bt = _bends(dp, ap), _bends(dt, at)
    n_b = np.rint(_xcorr(np.fft.rfft(bt.astype(np.float64), nfft)[None, :], np.fft.rfft(bp.astype(np.float64), nfft),
                         nfft))[0]
    ok = (r_d >= thr) & (n_d >= mo) & (n_b >= mo) & (r_a >= thr) & (r_l >= thr)
    if lag is None:
        k0 = int(np.argmax(np.where(n_d >= mo, r_d, -np.inf)))
        lag = k0 if k0 < nfft - len(dp) else k0 - nfft
    lag = int(lag)
    # direct check and diagnostics at `lag`: probe diff j <-> train diff j + lag, over the full overlap and, when the
    # scan's n is given, over its first and last n jointly finite steps (stamp-gep scans segments, so a hit near a
    # segment edge is a correlation over a sub-window of the alignment; that exact sub-window is re-tested too)
    i0, i1 = max(0, -lag), max(max(0, -lag), min(len(dp), len(dt) - lag))
    fin_all = np.isfinite(dp[i0:i1]) & np.isfinite(dt[i0 + lag:i1 + lag])
    idx = np.flatnonzero(fin_all)
    wins = [(i0, i1)]
    if n_scan and 3 <= n_scan < len(idx):
        wins += [(i0, i0 + idx[n_scan - 1] + 1), (i0 + idx[-n_scan], i1)]
    best = None
    for w0, w1 in wins:
        rd, n = _pearson(dp[w0:w1], dt[w0 + lag:w1 + lag])
        ra, na = _pearson(wp[w0:w1], wt[w0 + lag:w1 + lag])
        rl, _ = _pearson(p[w0:w1 + 1], t[w0 + lag:w1 + 1 + lag])
        nb = int((bp[w0:w1] & bt[w0 + lag:w1 + lag]).sum())
        ok_w = rd >= thr and n >= mo and nb >= mo and ra >= thr and rl >= thr
        # report the window that passes, else the one closest to the scan's own (r, n)
        score = (ok_w, -abs(n - (n_scan or n)) - abs(rd - (r_scan if r_scan is not None else rd)))
        if best is None or score > best[0]:
            best = (score, (w0, w1, rd, n, ra, na, nb, rl, ok_w))
    w0, w1, rd, n, ra, na, nb, rl, direct_ok = best[1]
    s = season
    rs, _ = _pearson(p[w0 + s:w1 + 1] - p[w0:w1 + 1 - s], t[w0 + lag + s:w1 + 1 + lag] - t[w0 + lag:w1 + 1 + lag - s]) \
        if w1 - w0 > s + 3 else (-np.inf, 0)
    sl_p, sl_t = slice(w0, w1), slice(w0 + lag, w1 + lag)
    sp_, st_ = dp[sl_p], dt[sl_t]
    fin = np.isfinite(sp_) & np.isfinite(st_)
    lp, lt = p[w0:w1 + 1], t[w0 + lag:w1 + 1 + lag]
    out = {"confirmed": bool(direct_ok or ok.any()), "lag0": lag, "window": [int(w0), int(w1)], "r_diff": rd, "n": n,
           "zero_p": float(1 - (ap[sl_p] & fin).sum() / max(fin.sum(), 1)),
           "zero_t": float(1 - (at[sl_t] & fin).sum() / max(fin.sum(), 1)),
           "n_active": na, "n_bends": nb, "r_active_wins": ra, "r_level": rl, "r_seasdiff": rs,
           "uniq_p": int(len(np.unique(lp[np.isfinite(lp)]))), "uniq_t": int(len(np.unique(lt[np.isfinite(lt)])))}
    a_, b_ = sp_[fin], st_[fin]
    if len(a_) > 10 and a_.std() > 0 and b_.std() > 0:
        c = (a_ - a_.mean()) / a_.std() * (b_ - b_.mean()) / b_.std()
        out["top1pct_share"] = float(np.sort(c)[::-1][:max(1, len(c) // 100)].sum() / max(c.sum(), 1e-12))
    if direct_ok:
        out.update(lag_confirmed=lag, r_confirmed=rd)
    elif ok.any():
        kc = int(np.argmax(np.where(ok, r_d, -np.inf)))
        out.update(lag_confirmed=kc if kc < nfft - len(dp) else kc - nfft, r_confirmed=float(r_d[kc]))
    return out


CONFIRM_RULE = ("r_diff>=thr & n>=min_overlap & n_bends(both change slope, |d2|>1e-3 robust SD)>=min_overlap & r(winsorised 5 MAD moving "
                "diffs)>=thr & r_level>=thr, at one lag (direct at the scan lag over the full overlap or the scan's "
                "own n-step sub-window, or any lag via FFT)")


def gep_confirm(rows, splits, tasks=None, min_overlap=60, min_overlap_periods=2, thr=0.95, verbose=True, detail=False):
    """Re-test every lag-scan hit row with confirm_pair (reads the same probes as the scan: VAL contexts + targets,
    TEST contexts only). Adds 'confirm' (dict) to each row in place; returns the number confirmed."""
    if not rows:
        return 0
    need = sorted({h["task"] for h in rows} & set(tasks or {h["task"] for h in rows}))
    probes = {}
    for fc, P in _gep_probes(splits, need).items():
        for pid, y in P:
            probes[pid] = y
    by_f = {}
    for h in rows:
        by_f.setdefault(h["train"], []).append(h)
    BN.gep_verify()
    n_ok, t0 = 0, time.time()
    for f, hs in by_f.items():
        sh = BN._GepShard(f)
        for h in hs:
            pid = (h["split"], h["task"], h["series"])
            if pid not in probes:                         # e.g. a VAL-only rerun of a cached VAL+TEST part
                continue
            fc = h["class"]
            mo = max(min_overlap, min_overlap_periods * GEP_SEASON[fc])
            h["confirm"] = confirm_pair(probes[pid], np.array(sh.get(h["key"]), np.float64), mo, GEP_SEASON[fc], thr,
                                        lag=h["lag"], n_scan=h["n"], r_scan=h["r"])
            if detail:                                    # review aid (confirm-gep --keys): the aligned stretch itself
                c = h["confirm"]
                w0, w1, lg = c["window"][0], c["window"][1], h["lag"]
                yp, yt = np.asarray(probes[pid], np.float64)[w0:w1 + 1], np.array(sh.get(h["key"]), np.float64)[w0 + lg:w1 + 1 + lg]
                print(f"  {h['task']}/{h['series']} ~ {f}[{h['key']}] window {w0}-{w1}: NaN p/t {np.isnan(yp).sum()}/"
                      f"{np.isnan(yt).sum()}, unique p/t {len(np.unique(yp[np.isfinite(yp)]))}/{len(np.unique(yt[np.isfinite(yt)]))}"
                      f", range p {np.nanmin(yp):.4g}..{np.nanmax(yp):.4g} t {np.nanmin(yt):.4g}..{np.nanmax(yt):.4g}")
                print("    p:", np.array2string(yp[:48], precision=4, max_line_width=200))
                print("    t:", np.array2string(yt[:48], precision=4, max_line_width=200))
            n_ok += h["confirm"]["confirmed"]
        del sh
    if verbose:
        print(f"  confirm: {n_ok}/{len(rows)} lag hits pass the strict re-test ({time.time() - t0:.0f}s)", flush=True)
    return n_ok


GEP_PARTS_DIR = os.path.join(ROOT, "runs", "gep_scan")      # resumable full scan: one cached result per part


def _part(name, settings, fn, cache):
    """Run fn() or reuse runs/gep_scan/<name>.json when its key and settings match (full scans only)."""
    path = os.path.join(GEP_PARTS_DIR, f"{name}.json")
    if cache and os.path.exists(path):
        c = json.load(open(path))
        if c.get("settings") == settings:
            print(f"  [{name}] reused {os.path.relpath(path, ROOT)} ({c['date']})", flush=True)
            return c["result"]
    res = fn()
    if cache:
        os.makedirs(GEP_PARTS_DIR, exist_ok=True)
        json.dump({"settings": settings, "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "result": res},
                  open(path + ".tmp", "w"), default=str)
        os.replace(path + ".tmp", path)
    return res


def stamp_gep(args):
    splits = args.splits.split(",")
    folders = args.folders.split(",") if args.folders else None
    tasks = args.tasks.split(",") if args.tasks else None
    partial = bool(folders or tasks or set(splits) != {"val", "test"})
    if "test" in splits and os.environ.get("TSAR_SPLIT") != "test":
        sys.exit("stamp-gep scans TEST contexts (target-free) and needs TSAR_SPLIT=test; it is logged.")
    t0 = time.time()
    BN.gep_verify()                                        # sha256 of every file vs the pinned manifest
    key = BN.gep_scan_key()
    print(f"GEP verified ({BN.GEP_BYTES} bytes, rev {BN.GEP_REV[:8]}); key {key[:16]}; "
          f"{'PARTIAL (no stamp, no cache)' if partial else 'full (resumable: ' + os.path.relpath(GEP_PARTS_DIR, ROOT) + ')'}"
          f" scan, splits={splits}", flush=True)
    # cache key: the data only (not GEP_DROP / known FPs, which are applied to cached hits after loading), so adding
    # drops after review re-uses the parts. Delete runs/gep_scan/ after any change to the scan code.
    base = {"data": hashlib.sha256(json.dumps({"rev": BN.GEP_REV, "gep": BN.TRAIN_GEP, "files": BN.gep_verify()},
                                              sort_keys=True).encode()).hexdigest()}
    cache = not partial
    residual = {}
    for split in splits:
        print(f"== exact/znorm hashes, {split.upper()} ({'contexts + targets' if split == 'val' else 'contexts only'})"
              f", GEP bank", flush=True)
        rep = _part(f"hash_{split}", {**base, "L": 32},
                    lambda: BN.overlap(split, L=32, source=BN.gep_series(folders), tasks=tasks), cache)
        rep = {t: {tag: [tuple(x) for x in v] for tag, v in r.items()} for t, r in rep.items()}
        residual[f"{split}_hash"] = _overlap_residual(rep, drop=BN.GEP_DROP, known_fp=GEP_KNOWN_FP)
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
    print("== lag scan, same-frequency classes (5min/15min/30min/H)", flush=True)
    lag_set = {**base, "thr": args.thr, "min_overlap": args.min_overlap,
               "min_overlap_periods": args.min_overlap_periods, "splits": sorted(splits)}
    probes = None
    rows, stats = [], {}
    for fc in GEP_SEASON:
        def run(fc=fc):
            nonlocal probes
            if probes is None:
                probes = _gep_probes(splits, tasks)
            if not probes.get(fc):
                return {"rows": [], "stats": {"probes": 0}}
            r, st = gep_lag_class(fc, probes[fc], folders, args.min_overlap, args.min_overlap_periods, args.thr,
                                  chunk=args.chunk, workers=args.workers)
            return {"rows": r, "stats": st}
        res = _part(f"lag_{fc}", lag_set, run, cache)
        for h in res["rows"]:                              # GEP_DROP may have changed since a cached part
            h["dropped"] = h["key"] in set(BN.GEP_DROP.get(h["train"], ())) or (h["task"], h["train"]) in GEP_KNOWN_FP
        rows += res["rows"]; stats[fc] = res["stats"]
    # strict re-test of every flagged pair (docs/gep_overlap_review.md): not cached, cheap (one FFT pass per hit)
    if args.confirm:
        print("== strict re-test of lag hits (moving steps, winsorised and level r)", flush=True)
        gep_confirm([h for h in rows if not h["dropped"]], splits, tasks, args.min_overlap, args.min_overlap_periods,
                    args.thr)
    for h in rows:
        h["filtered"] = bool(args.confirm and not h["dropped"] and not h.get("confirm", {}).get("confirmed", True))
        c = h.get("confirm") or {}
        print(f"      {h['split']}:{h['task']}/{h['series']} ~ {h['train']}[{h['key']}] r={h['r']} n={h['n']} "
              f"lag={h['lag']}" + ("  (dropped/known FP)" if h["dropped"] else "")
              + (f"  (filtered: n_bends={c.get('n_bends')} r_active_wins={c.get('r_active_wins', 0):.3f} r_level="
                 f"{c.get('r_level', 0):.3f})" if h["filtered"] else ""))
    residual["lag"] = [h for h in rows if not h["dropped"] and not h["filtered"]]
    n = sum(len(v) for v in residual.values())
    print(f"stamp-gep: residual hits {n} ({', '.join(f'{k}={len(v)}' for k, v in residual.items())}); "
          f"{time.time() - t0:.0f}s total", flush=True)
    if args.out:
        json.dump({"key": key, "partial": partial, "residual": residual, "lag_hits": rows, "lag_stats": stats},
                  open(args.out, "w"), indent=1, default=str)
        print(f"hits written to {args.out}")
    if n:
        print(json.dumps(residual, indent=1, default=str)[:20000])
        sys.exit(f"{n} residual hit(s): review them, add real duplicates to bench.GEP_DROP (or GEP_KNOWN_FP with a "
                 "note), re-run. No stamp written.")
    if partial:
        print("partial scan clean; no stamp written (a stamp needs every folder, every task, VAL and TEST).")
        return
    if BN.gep_scan_key() != key:
        sys.exit("GEP files / TRAIN_GEP / GEP_DROP changed during the scan; re-run. No stamp written.")
    stamp = {"result": "clean", "key": key, "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": BN._git_head(),
             "rev": BN.GEP_REV,
             "scans": {"hash": {"L": 32, "splits": ["val", "test"]},
                       "lag": {"classes": stats, "thr": args.thr, "min_overlap": args.min_overlap,
                               "min_overlap_periods": args.min_overlap_periods, "transform": "diff",
                               "confirm": CONFIRM_RULE if args.confirm else None,
                               "filtered_hits": sum(h["filtered"] for h in rows)}},
             "known_false_positives": sorted(map(list, GEP_KNOWN_FP))}
    json.dump(stamp, open(BN.GEP_STAMP, "w"), indent=1)
    print(f"clean: wrote {BN.GEP_STAMP}")


def confirm_gep_file(args):
    """Re-test the lag hits of a stamp-gep --out file (--hits) with confirm_pair; write them with diagnostics to --out.
    Review tool only: never stamps. --tasks restricts it (e.g. to fit a smoke)."""
    if os.environ.get("TSAR_SPLIT") != "test":
        sys.exit("confirm-gep re-reads TEST contexts (target-free) and needs TSAR_SPLIT=test; it is logged.")
    rows = json.load(open(args.hits))["lag_hits"]
    if args.tasks:
        rows = [h for h in rows if h["task"] in args.tasks.split(",")]
    if args.folders:
        rows = [h for h in rows if h["train"] in args.folders.split(",")]
    if args.keys:
        rows = [h for h in rows if f"{h['train']}:{h['key']}" in args.keys.split(",")]
    gep_confirm(rows, sorted({h["split"] for h in rows}), None, args.min_overlap, args.min_overlap_periods, args.thr,
                detail=bool(args.keys))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {len(rows)} rows to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["lag", "selftest", "stamp-extra", "stamp-gep", "confirm-gep"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tasks", help="comma-separated task names (default: TEST macro/world, or all VAL)")
    ap.add_argument("--train", help="comma-separated TRAIN datasets to scan against (default: all)")
    ap.add_argument("--extra", action="store_true", help="include TRAIN_EXTRA in the train bank")
    ap.add_argument("--any-freq", action="store_true", help="scan across frequencies too")
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--min-overlap", type=int, default=60)
    ap.add_argument("--min-overlap-yearly", type=int, default=24)
    ap.add_argument("--transform", default="diff", choices=["diff", "logdiff"])
    ap.add_argument("--out", help="write hits as JSON here")
    ap.add_argument("--no-confirm", dest="confirm", action="store_false",
                    help="stamp-gep: skip the strict re-test of lag hits (every growth-correlation hit is residual)")
    ap.add_argument("--known-tp", action="store_true",
                    help="selftest: re-test the known stamp-extra true positives (TEST contexts; TSAR_SPLIT=test)")
    ap.add_argument("--keys", help="confirm-gep: only these folder:key hits, and print their aligned stretch")
    ap.add_argument("--hits", default=os.path.join(ROOT, "runs", "overlap_gep_hits.json"),
                    help="confirm-gep: a stamp-gep --out file")
    ap.add_argument("--splits", default="val,test", help="stamp-gep: splits to probe (both for a stamp)")
    ap.add_argument("--folders", help="stamp-gep: comma-separated GEP folders (partial scan, never stamps)")
    ap.add_argument("--min-overlap-periods", type=int, default=2,
                    help="stamp-gep lag: sub-daily classes need >= this many seasons of aligned points")
    ap.add_argument("--chunk", type=int, default=128, help="stamp-gep lag: train segments per FFT chunk")
    ap.add_argument("--workers", type=int, default=4, help="stamp-gep lag: scipy.fft threads (0 = numpy)")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest_known_tp() if args.known_tp else selftest()
    if args.cmd == "stamp-extra":
        return stamp_extra(args)
    if args.cmd == "stamp-gep":
        return stamp_gep(args)
    if args.cmd == "confirm-gep":
        return confirm_gep_file(args)
    res = lag_scan(args.split, args.tasks.split(",") if args.tasks else None, args.extra,
                   args.train.split(",") if args.train else None, args.min_overlap, args.min_overlap_yearly,
                   args.thr, args.transform, args.any_freq)
    print(f"hits={len(res['hits'])} residual={len(res['residual'])}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    sys.exit(1 if res["residual"] else 0)


if __name__ == "__main__":
    main()
