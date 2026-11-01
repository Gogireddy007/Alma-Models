"""Finance evaluation tier (default off; Finance lead, 2026-10-31). Design and leakage argument:
runs/proposals/finance_r2.md. Built only from data already on disk; nothing is downloaded and the bench TEST
split is never read.

Tasks (bench metrics: wql_rel / mase_rel vs seasonal naive on the same windows, cov80_error; plus the same
numbers vs the random walk, `*_rw`, the finance bar. For season-1 tasks seasonal naive IS the random walk.)

  task             source             what                         freq  h   m   sel / final
  fx_1B_h5         exchange_rate      8 USD FX rates, 1990-2019     B    5   1   75 / 25 windows per series (time)
  fx_1B_h30        exchange_rate      same, ~6 weeks ahead          B    30  1   20 / 8 windows (time)
  macro_1M_h12     monash_fred_md     107 FRED-MD macro series      M    12  1   6 / 2 windows (time)
  m4fin_1D_h14     m4_daily           M4 'Finance' category         D    14  1   600 / 400 series (1 window each)
  m4fin_1W_h13     m4_weekly          M4 'Finance' category         W    13  1   all 164 series, 98 / 66
  m4fin_1M_h18     m4_monthly         M4 'Finance' category         M    18  12  600 / 400 series
  m4fin_1Q_h8      m4_quarterly       M4 'Finance' category         Q    8   4   600 / 400 series
  conditional (only for a checkpoint that did NOT train on the source; every current recipe does, so these are
  skipped until a training run holds them out):
  cash_nn5_1D_h14  nn5                111 ATM cash withdrawals      D    14  7   6 / 2 windows (time)
  cash_nn5_1W_h13  monash_nn5_weekly  same ATMs, weekly sums        W    13  1   2 / 1 windows (time)
  bank_cif_1M_h12  monash_cif_2016    72 banking-risk series        M    12  12  43 / 29 series

`sel` (default) is for model selection. `final` is a reserved holdout (later windows / disjoint series) scored once
with the frozen final model, next to TEST; it needs TSAR_FIN_FINAL=1 and is logged.

Leak gate (evaluate(train_datasets=...), train_wm.py passes run_cfg()["train_datasets"]):
  * a task is skipped when its source FAMILY is in the checkpoint's training set (any M4 frequency trained ->
    every m4fin task skipped; nn5 or nn5_weekly -> both cash tasks);
  * a task is skipped when some trained dataset has not been content-scanned against its source (SCANNED below);
  * series in FIN_DROP (content hits of the scan against any TRAIN dataset) and in bench.TRAIN_DROP (M4 series that
    duplicate TEST macro content) are removed for every model, so baselines and checkpoints see identical windows.

Usage (tools/smoke.py for agents):
    uv run tools/finance_tier.py summary                       # tasks, series, windows, exclusions
    uv run tools/finance_tier.py baselines [models] [--tasks a,b]   # default: snaive/naive points + statsforecast
    uv run tools/finance_tier.py scan --kind hash|lag --train d1,d2 [--sources s1,s2]   # content scan, cached
    uv run tools/finance_tier.py status                        # scan coverage + residual hits
    TSAR_FIN_TIER=1|only ... train_wm.py                       # score a model (only = skip VAL)
"""
import os, sys, json, time, zlib, hashlib, argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as BN  # noqa: E402  (read-only use)

P = BN.P
ROOT = BN.ROOT
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "runs", "finance_tier")
FINAL_LOG = os.path.join(BN.BENCH_DIR, "fin_final_access.log")
SEED = 20261031
L_HASH = 32

# (task, family, source, category, horizon, season, freq, n_sel, n_final, split_by)
TASK_SPECS = [
    ("fx_1B_h5", "fx", "exchange_rate", None, 5, 1, "B", 75, 25, "time"),
    ("fx_1B_h30", "fx", "exchange_rate", None, 30, 1, "B", 20, 8, "time"),
    ("macro_1M_h12", "fred_md", "monash_fred_md", None, 12, 1, "M", 6, 2, "time"),
    ("m4fin_1D_h14", "m4fin", "m4_daily", "Finance", 14, 1, "D", 600, 400, "series"),
    ("m4fin_1W_h13", "m4fin", "m4_weekly", "Finance", 13, 1, "W", 98, 66, "series"),
    ("m4fin_1M_h18", "m4fin", "m4_monthly", "Finance", 18, 12, "M", 600, 400, "series"),
    ("m4fin_1Q_h8", "m4fin", "m4_quarterly", "Finance", 8, 4, "Q", 600, 400, "series"),
    ("cash_nn5_1D_h14", "nn5", "nn5", None, 14, 7, "D", 6, 2, "time"),
    ("cash_nn5_1W_h13", "nn5", "monash_nn5_weekly", None, 13, 1, "W", 2, 1, "time"),
    ("bank_cif_1M_h12", "cif", "monash_cif_2016", None, 12, 12, "M", 43, 29, "series"),
]
FIELDS = ("name", "family", "source", "category", "horizon", "season", "freq", "n_sel", "n_final", "split_by")
TASKS = [dict(zip(FIELDS, t)) for t in TASK_SPECS]
SOURCES = sorted({t["source"] for t in TASKS})
# a task is gated off when ANY dataset of its family's sources was trained on (other frequencies of the same
# source can be aggregates of the same series)
FAMILY_SOURCES = {
    "fx": {"exchange_rate"}, "fred_md": {"monash_fred_md"},
    "m4fin": {"m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_hourly", "m4_yearly"},
    "nn5": {"nn5", "monash_nn5_weekly"}, "cif": {"monash_cif_2016"},
}
_FREQ_CLASS = {"B": "D", "D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y", "H": "H"}
# TRAIN datasets that hold economic / business series: a re-released copy (other vintage or units) of a finance
# series can only hide there, so the growth-correlation (lag) scan is required against these in the same freq class.
# Every other TRAIN dataset (weather, traffic, retail, energy, sensors) needs the exact + znorm hash scan only.
ECON_TRAIN = {"monash_m1_monthly", "monash_m1_quarterly", "monash_m1_yearly", "monash_m3_monthly",
              "monash_m3_quarterly", "monash_m3_yearly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly",
              "m4_hourly", "m4_yearly", "monash_tourism_monthly", "monash_tourism_quarterly",
              "monash_tourism_yearly", "monash_cif_2016", "monash_hospital", "monash_car_parts", "nn5",
              "monash_nn5_weekly"}
# Not ECON_TRAIN: m5 and dominick are store x SKU unit sales (a finance/macro series cannot be another vintage of
# one); they, like every other TRAIN set, get the exact + znorm hash scan.

# --- content scan results (filled from `scan`; raw evidence in runs/finance_tier/scan/*.json) ------------------
# SCANNED[kind] = TRAIN datasets whose FULL series set (no subsample, before TRAIN_DROP) was scanned against every
# tier source outside its own family. FIN_DROP = {source: [series key]} of every residual hit (none expected).
# Scanned 2026-10-31 (tools/smoke.py, runs/finance_tier/scan/). Not yet scanned: the 9 GEP datasets (hash; GEP
# training is itself blocked until stamp-gep) and m4_daily / m4_monthly (lag vs FX, nn5, FRED-MD, CIF; only a
# non-GIFT-clean TSAR_BENCH_EXTRA checkpoint trains on them). Checkpoints that trained on these get the affected
# tasks skipped by eligibility(), not scored.
SCANNED = {
    "hash": {n for n, *_ in BN.TRAIN_LOCAL + BN.TRAIN_EXTRA},          # all 38 LOCAL + EXTRA sets, full series
    "lag": ECON_TRAIN - {"m4_daily", "m4_monthly"},
}
# Every residual hit is dropped (conservative, like bench.TRAIN_DROP). FRED-MD: 23 of 107 series are the same US
# macro series that M1 / M3 / M4 monthly carry anonymously (exact, znorm and growth-correlation hits, r up to 1.0).
# m4_monthly / m4_quarterly: growth-correlation hits (r 0.95-0.97, n 60-110) vs M1 / M3 / tourism, mostly smooth
# seasonal false positives, dropped anyway. exchange_rate, nn5, nn5_weekly, cif and the M4 daily / weekly probes: 0.
FIN_DROP = {
    "monash_fred_md": [3, 18, 19, 21, 24, 28, 29, 30, 32, 33, 44, 45, 47, 48, 50, 51, 52, 55, 59, 60, 79, 87, 106],
    "m4_monthly": [38593, 38869, 39252, 39297, 39340, 39825, 39955, 40177, 40733, 40928, 41056, 41886, 42061, 44233],
    "m4_quarterly": [18729, 19175, 19866, 19869, 20096, 20101, 20114, 20118, 20161, 20613, 20858, 21158, 21249,
                     21259, 21272, 22347, 22354, 22375],
}


def _src_file(name):
    p = os.path.join(RAW, f"{name}.parquet")
    return p if os.path.exists(p) else BN._train_file(name, True)


def _dataset_freq_class(name):
    for n, _s, f, _h in BN.TRAIN_LOCAL + BN.TRAIN_EXTRA:
        if n == name:
            return _FREQ_CLASS.get(f, f)
    if name in ("exchange_rate",):
        return "D"
    if name in ("monash_fred_md",):
        return "M"
    return None


_SRC_CACHE = {}


def _source_series(name):
    """[(key, y float32, category)] in _target_arrays order (key = position, as TRAIN_DROP)."""
    if name not in _SRC_CACHE:
        import pyarrow.parquet as pq
        df = pd.read_parquet(_src_file(name), columns=[c for c in pq.read_schema(_src_file(name)).names
                                                        if c != "timestamp"])
        cats = df["category"].tolist() if "category" in df.columns else [None] * len(df)
        ys = list(BN._target_arrays(df.drop(columns=[c for c in ("category",) if c in df.columns])))
        assert len(ys) == len(cats)
        _SRC_CACHE[name] = [(k, y, c) for k, (y, c) in enumerate(zip(ys, cats))]
    return _SRC_CACHE[name]


def _selected_keys(t):
    """Fixed series selection per task -> (sel keys, final keys), before FIN_DROP."""
    ser = _source_series(t["source"])
    drop = set(BN.TRAIN_DROP.get(t["source"], ()))
    h, W = t["horizon"], t["n_sel"] + t["n_final"]
    need = h + BN.MIN_CONTEXT + (h * (W - 1) if t["split_by"] == "time" else 0)
    ok = [k for k, y, c in ser if (t["category"] is None or c == t["category"]) and k not in drop
          and np.isfinite(y).sum() >= need]
    if t["split_by"] == "time":
        return ok, ok
    rng = np.random.default_rng(SEED + zlib.crc32(t["name"].encode()))
    perm = [ok[i] for i in rng.permutation(len(ok))]
    n_s = min(t["n_sel"], len(perm))
    return sorted(perm[:n_s]), sorted(perm[n_s:n_s + t["n_final"]])


def probe_keys(source):
    """Every series key of `source` that any task exposes (sel and final), the scan's probe set."""
    out = set()
    for t in TASKS:
        if t["source"] == source:
            a, b = _selected_keys(t)
            out |= set(a) | set(b)
    return sorted(out)


def build_task(t, part="sel"):
    assert part in ("sel", "final")
    sel, fin = _selected_keys(t)
    keys = [k for k in (sel if part == "sel" else fin) if k not in set(FIN_DROP.get(t["source"], ()))]
    ser = {k: y for k, y, _ in _source_series(t["source"])}
    h, W = t["horizon"], t["n_sel"] + t["n_final"]
    ctxs, tgts, sids, wins = [], [], [], []
    for k in keys:
        y = ser[k]
        last = int(np.flatnonzero(np.isfinite(y))[-1]) + 1      # drop trailing NaN padding
        y = y[:last]
        if t["split_by"] == "time":
            ks = range(t["n_sel"]) if part == "sel" else range(t["n_sel"], W)
            cuts = [(w, len(y) - (W - w) * h) for w in ks]
        else:
            cuts = [(0, len(y) - h)]
        for w, c in cuts:
            if c < 0:
                continue
            ctx = y[max(0, c - BN.MAX_CONTEXT):c]
            fi = np.flatnonzero(np.isfinite(ctx))
            if len(fi) < BN.MIN_CONTEXT:
                continue
            ctxs.append(ctx[fi[0]:].copy())
            tgts.append(y[c:c + h].copy())
            sids.append(str(k))
            wins.append(w)
    return {"name": t["name"], "family": t["family"], "config": f"fin:{t['source']}", "horizon": h,
            "season": t["season"], "freq": t["freq"], "contexts": ctxs,
            "targets": np.stack(tgts) if tgts else np.zeros((0, h), np.float32),
            "series_id": sids, "window": np.asarray(wins), "source": t["source"]}


def _final_guard(part, what):
    if part != "final":
        return
    if os.environ.get("TSAR_FIN_FINAL") != "1":
        raise PermissionError("the finance tier 'final' part is a reserved holdout: set TSAR_FIN_FINAL=1 for the "
                              "one frozen-model run (logged)")
    os.makedirs(os.path.dirname(FINAL_LOG), exist_ok=True)
    with open(FINAL_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{what}\tgit={BN._git_head()}\targv={' '.join(sys.argv)}\n")


def eligibility(train_datasets):
    """{task: None (eligible) or reason}. train_datasets=None: no model training to check (baselines)."""
    out = {}
    for t in TASKS:
        if train_datasets is None:
            out[t["name"]] = None
            continue
        tr = set(train_datasets)
        fam = FAMILY_SOURCES[t["family"]] & tr
        if fam:
            out[t["name"]] = f"trained on {sorted(fam)}"
            continue
        cls = _FREQ_CLASS[t["freq"]]
        miss = sorted(d for d in tr if d not in SCANNED["hash"]
                      or (d in ECON_TRAIN and _dataset_freq_class(d) == cls and d not in SCANNED["lag"]))
        out[t["name"]] = f"not content-scanned vs {miss}" if miss else None
    return out


def load_tasks(part="sel", train_datasets=None, names=None):
    _final_guard(part, f"finance_tier.load_tasks({part})")
    el = eligibility(train_datasets)
    tasks, skipped = [], {}
    for t in TASKS:
        if names and t["name"] not in names:
            continue
        if el[t["name"]]:
            skipped[t["name"]] = el[t["name"]]
            continue
        tasks.append(build_task(t, part))
    return tasks, skipped


def _rw_scores(t):
    """wql / mase of the point random walk (last finite value) on the task's windows."""
    Qn = len(P.QUANTILES)
    rw = np.stack([np.full(t["horizon"], P.fill_nans(c)[-1], np.float32) for c in t["contexts"]])
    w, m, _ = P._score(np.repeat(rw[:, None, :], Qn, axis=1), t["targets"], t["contexts"], t["season"])
    return w, m


def evaluate(predict_fn, part="sel", train_datasets=None, verbose=True, names=None, tasks=None):
    """bench.evaluate on the finance tier (identical scoring code) + random-walk-relative columns.
    train_datasets: the checkpoint's training datasets (run_cfg()["train_datasets"]); tasks whose source family was
    trained on, or that are not content-scanned against a trained dataset, are skipped and listed."""
    skipped = {}
    if tasks is None:
        tasks, skipped = load_tasks(part, train_datasets, names)
    if not tasks:
        raise SystemExit(f"finance tier: no eligible task ({skipped})")
    t0 = time.time()
    r = BN.evaluate(predict_fn, split="val", verbose=False, tasks=tasks)   # tasks= given: no split is loaded
    r["split"] = f"fin_{part}"
    df = r["per_dataset"]
    rw = {t["name"]: _rw_scores(t) for t in tasks}
    df["wql_rel_rw"] = [w / max(rw[n][0], 1e-8) for n, w in zip(df["dataset"], df["wql"])]
    df["mase_rel_rw"] = [m / max(rw[n][1], 1e-8) for n, m in zip(df["dataset"], df["mase"])]
    r["wql_rel_rw"] = BN._gmean(df["wql_rel_rw"])
    r["mase_rel_rw"] = BN._gmean(df["mase_rel_rw"])
    r["wql_rel_rw_family"] = BN._gmean(df.groupby("family")["wql_rel_rw"].agg(BN._gmean))
    r["skipped"] = skipped
    r["tasks"] = list(df["dataset"])
    if verbose:
        with pd.option_context("display.width", 200, "display.precision", 3, "display.max_rows", 200):
            print(df.drop(columns=["sn_wql", "sn_mase", "n_nonfinite"]).to_string(index=False))
        for k, v in skipped.items():
            print(f"  skipped {k}: {v}")
        print(f"fin_{part}: wql_rel={r['wql_rel']:.4f} mase_rel={r['mase_rel']:.4f} cov80_error={r['cov80_error']:.4f} "
              f"family-gmean wql_rel={r['wql_rel_family']:.4f} | vs random walk: wql={r['wql_rel_rw']:.4f} "
              f"mase={r['mase_rel_rw']:.4f} family wql={r['wql_rel_rw_family']:.4f} "
              f"({r['n_tasks']} tasks, {r['n_families']} families, {time.time() - t0:.1f}s)", flush=True)
    return r


def print_metrics(r, prefix):
    """Key: value lines in train_wm's output style."""
    for k in ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family", "wql_rel_rw", "mase_rel_rw",
              "wql_rel_rw_family"):
        print(f"{prefix}_{k}: {r[k]:.6f}")
    print(f"{prefix}_tasks: {','.join(r['tasks'])}")
    for k, v in r["skipped"].items():
        print(f"{prefix}_skipped: {k} ({v})")


# --------------------------------------------------------------------------------------------------------------
# Content scan: tier source series (full length, every probe key) vs every series of a TRAIN dataset
# --------------------------------------------------------------------------------------------------------------

def _scan_sources(train_name, sources):
    fam_of = {s: f for f, ss in FAMILY_SOURCES.items() for s in ss}
    tier_fams = {t["source"]: t["family"] for t in TASKS}
    # the tier's own family is gated, not scanned (an M4 tier task is never scored for an M4-trained model)
    return [s for s in sources if fam_of.get(train_name) != tier_fams[s]]


def _probe_key_hash(sources):
    h = hashlib.sha256()
    for s in sources:
        h.update(json.dumps([s, probe_keys(s), BN._file_sha256(_src_file(s))]).encode())
    return h.hexdigest()


def _train_series(name, shard=(0, 1)):
    i, n = shard
    if name.startswith("gep_"):
        folders = [f for f, d, *_ in BN.TRAIN_GEP if d == name]
        for j, (f, k, y) in enumerate(BN.gep_series(folders)):
            if j % n == i:
                yield (f, k), y
        return
    extra = name in {n for n, *_ in BN.TRAIN_EXTRA}
    import pyarrow.parquet as pq
    f = BN._train_file(name, extra)
    cols = [c for c in pq.read_schema(f).names if c not in ("timestamp", "category")]
    df = pd.read_parquet(f, columns=cols)              # same columns _target_arrays uses, minus timestamps
    for k, y in enumerate(BN._target_arrays(df)):
        if k % n == i:
            yield k, y


def _chunk_hashes_many(ys, L, stride, znorm=False):
    """bench._chunk_hashes over many series at once (same windows, filters, rounding and hash; checked equal by
    `selftest`). Returns (hashes, owner index into ys)."""
    wins, own = [], []
    for i, y in enumerate(ys):
        y = np.asarray(y)[np.isfinite(y)].astype(np.float64)
        if len(y) < L:
            continue
        w = np.lib.stride_tricks.sliding_window_view(y, L)[::stride]
        wins.append(w); own.append(np.full(len(w), i, np.int64))
    if not wins:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    w, own = np.concatenate(wins), np.concatenate(own)
    srt = np.sort(w, axis=1)
    k = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1) >= L // 4
    w, own = w[k], own[k]
    if znorm and len(w):
        d = np.sort(np.round(np.diff(w, axis=1), 6), axis=1)
        k = 1 + (np.diff(d, axis=1) != 0).sum(axis=1) >= L // 4
        w, own = w[k], own[k]
    if not len(w):
        return np.zeros(0, np.int64), own
    if znorm:
        sd = w.std(axis=1, keepdims=True)
        w = (w - w.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0) * 100.0
    else:
        w = w * 1000.0
    qv = np.clip(np.round(w), -4e15, 4e15).astype(np.int64)
    coef = np.random.default_rng(7).integers(1, 2 ** 61, L, dtype=np.int64)
    return (qv * coef).sum(axis=1), own


def selftest(_a=None):
    """_chunk_hashes_many == bench._chunk_hashes on real TRAIN series (dominick, m1) and synthetic edge cases."""
    rng = np.random.default_rng(0)
    ys = [y for _, (_k, y) in zip(range(300), _train_series("dominick"))] + \
         [y for _, (_k, y) in zip(range(100), _train_series("monash_m1_monthly"))] + \
         [np.r_[rng.normal(size=50), np.nan, rng.normal(size=70)], np.zeros(80), np.arange(100.0), np.ones(5)]
    for zn in (False, True):
        for stride in (1, L_HASH):
            h, own = _chunk_hashes_many(ys, L_HASH, stride, zn)
            ref = [BN._chunk_hashes(y, L_HASH, stride, znorm=zn) for y in ys]
            assert np.array_equal(h, np.concatenate(ref)), (zn, stride)
            assert np.array_equal(own, np.concatenate([np.full(len(r), i) for i, r in enumerate(ref)]))
    print(f"selftest OK: {len(ys)} series, exact/znorm, stride 1/{L_HASH}")


_BANKS = {}


def _probe_bank(s, zn):
    """stride-1 L-run hashes of every probe series of source s, with ids (source index << 32 | key); cached."""
    if (s, zn) not in _BANKS:
        ser = {k: y for k, y, _ in _source_series(s)}
        keys = np.asarray(probe_keys(s), np.int64)
        hs, ids = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        for i in range(0, len(keys), 25):          # stride-1 windows: chunked to bound memory
            h, own = _chunk_hashes_many([ser[k] for k in keys[i:i + 25]], L_HASH, 1, zn)
            hs.append(h); ids.append((SOURCES.index(s) << 32) | keys[i:i + 25][own])
        _BANKS[(s, zn)] = (np.concatenate(hs), np.concatenate(ids))
    return _BANKS[(s, zn)]


def scan_hash(train_name, sources, shard=(0, 1)):
    """Exact (1e-3) and znorm (affine) L=32 runs: probe bank = tier series at stride 1, query = TRAIN at stride L
    (any shared stretch >= 2L-1 points is found, as bench.overlap with the roles swapped)."""
    banks = {}
    for tag, zn in (("exact", False), ("znorm", True)):
        hs, ids = zip(*[_probe_bank(s, zn) for s in sources])
        h, idv = np.concatenate(hs), np.concatenate(ids)
        o = np.argsort(h, kind="stable")
        banks[tag] = (h[o], idv[o])
    hits = {}
    n = 0
    it = _train_series(train_name, shard)
    while True:
        batch, pts = [], 0
        for x in it:                               # <= 5000 series and ~4M points per batch (bounded memory)
            batch.append(x); pts += len(x[1])
            if len(batch) >= 5000 or pts >= 4_000_000:
                break
        if not batch:
            break
        n += len(batch)
        tks = [tk for tk, _ in batch]
        for tag, zn in (("exact", False), ("znorm", True)):
            q, own = _chunk_hashes_many([y for _, y in batch], L_HASH, L_HASH, zn)
            hb, ib = banks[tag]
            lo, hi = np.searchsorted(hb, q, "left"), np.searchsorted(hb, q, "right")
            m = hi > lo
            for a, b, o in zip(lo[m], hi[m], own[m]):
                for code in ib[a:b]:
                    s, k = SOURCES[int(code) >> 32], int(code) & 0xFFFFFFFF
                    key = (tag, s, int(k), str(tks[o]))
                    hits[key] = hits.get(key, 0) + 1
    rows = [{"kind": tag, "source": s, "key": k, "train": train_name, "train_key": tk, "n_runs": c}
            for (tag, s, k, tk), c in sorted(hits.items())]
    return rows, n


def scan_lag(train_name, sources, shard=(0, 1), thr=0.95, min_overlap=60):
    """Growth-correlation scan (overlap_scan.scan_pairs: corr of standardised first differences >= thr over
    >= min_overlap aligned points, any lag), same frequency class only. Probe = the exposed stretch of each tier
    series (the hash scan covers the full series). Probes grouped by length (smaller FFTs)."""
    import overlap_scan as OS
    cls = _dataset_freq_class(train_name)
    train = [(tk, y) for tk, y in _train_series(train_name, shard)]
    rows = []
    for s in sources:
        if _dataset_freq_class(s) != cls:
            continue
        ser = {k: y for k, y, _ in _source_series(s)}
        span = max(BN.MAX_CONTEXT + t["horizon"] * (t["n_sel"] + t["n_final"] if t["split_by"] == "time" else 1)
                   for t in TASKS if t["source"] == s)
        # the exposed stretch (every context any window can see, plus all targets): trailing NaN cut first
        probes = []
        for k in probe_keys(s):
            y = ser[k][:int(np.flatnonzero(np.isfinite(ser[k]))[-1]) + 1]
            probes.append(((s, k), y[-span:]))
        groups = {}
        for p in probes:
            groups.setdefault(int(np.ceil(np.log2(max(len(p[1]), 2)))), []).append(p)
        for _, grp in sorted(groups.items()):
            for (ps, pk), tk, r, nn, lag in OS.scan_pairs(grp, train, min_overlap, thr):
                rows.append({"kind": "lag", "source": ps, "key": int(pk), "train": train_name, "train_key": str(tk),
                             "r": round(r, 4), "n": nn, "lag": lag})
    return rows, len(train)


def cmd_scan(a):
    os.makedirs(os.path.join(OUT, "scan"), exist_ok=True)
    srcs_all = a.sources.split(",") if a.sources else SOURCES
    for d in a.train.split(","):
        sources = _scan_sources(d, srcs_all)
        if a.kind == "lag":
            sources = [s for s in sources if _dataset_freq_class(s) == _dataset_freq_class(d)]
        if not sources:
            print(f"{a.kind} {d}: no tier source to scan (own family or other frequency class)")
            continue
        t0 = time.time()
        shard = tuple(int(x) for x in a.shard.split("/"))
        rows, n = (scan_hash if a.kind == "hash" else scan_lag)(d, sources, shard)
        res = {"kind": a.kind, "train": d, "sources": sources, "shard": list(shard), "probe_key": _probe_key_hash(sources),
               "n_train_series": n, "hits": rows, "seconds": round(time.time() - t0, 1), "git": BN._git_head(),
               "date": time.strftime("%Y-%m-%d %H:%M")}
        sh = "" if shard == (0, 1) else f"__shard{shard[0]}of{shard[1]}"
        path = os.path.join(OUT, "scan", f"{a.kind}__{d}__{'+'.join(sources)}{sh}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{a.kind} {d}: {n} train series vs {sources}: {len(rows)} hits ({res['seconds']}s) -> "
              f"{os.path.relpath(path, ROOT)}", flush=True)
        for r in rows[:20]:
            print("   ", r)


def cmd_status(_a):
    d = os.path.join(OUT, "scan")
    cov = {"hash": {}, "lag": {}}
    hits, shards = [], {}
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        r = json.load(open(os.path.join(d, fn)))
        ok = r["probe_key"] == _probe_key_hash(r["sources"])
        i, n = r.get("shard", [0, 1])
        if ok:
            shards.setdefault((r["kind"], r["train"], tuple(r["sources"]), n), set()).add(i)
        hits += [h for h in r["hits"]] if ok else []
        if not ok:
            print(f"stale (probe set or source file changed): {fn}")
    for (kind, tr, srcs, n), got in shards.items():
        if got == set(range(n)):
            cov[kind].setdefault(tr, set()).update(srcs)
    all_train = [n for n, *_ in BN.TRAIN_LOCAL + BN.TRAIN_EXTRA] + sorted({d_ for _, d_, *_ in BN.TRAIN_GEP})
    for kind in ("hash", "lag"):
        full = []
        for t in all_train:
            need = _scan_sources(t, SOURCES)
            if kind == "lag":
                if t not in ECON_TRAIN:
                    continue
                need = [s for s in need if _dataset_freq_class(s) == _dataset_freq_class(t)]
            have = cov[kind].get(t, set())
            state = "complete" if set(need) <= have else f"missing {sorted(set(need) - have)}"
            if set(need) <= have:
                full.append(t)
            print(f"{kind:4s} {t:32s} {state}")
        print(f"SCANNED[{kind!r}] = {sorted(full)}")
    by = {}
    for h in hits:
        by.setdefault(h["source"], set()).add(h["key"])
    print(f"residual hits: {len(hits)}; FIN_DROP = { {k: sorted(v) for k, v in by.items()} }")


# --------------------------------------------------------------------------------------------------------------
# Baselines and summary
# --------------------------------------------------------------------------------------------------------------

def _snaive_point(ctxs, h, m, freq):
    sn = np.stack([P._seasonal_naive(c, h, m) for c in ctxs])
    return np.repeat(sn[:, None, :], len(P.QUANTILES), axis=1)


def _rw_point(ctxs, h, m, freq):
    return _snaive_point(ctxs, h, 1, freq)


def cmd_baselines(a):
    import baselines as BL   # read-only import
    reg = {"snaive_point": _snaive_point, "rw_point": _rw_point}
    want = a.models or ["snaive_point", "rw_point", "naive", "seasonal_naive_pi", "auto_ets", "auto_theta"]
    if any(m not in reg for m in want):
        reg.update(BL.registry())
    names = set(a.tasks.split(",")) if a.tasks else None
    tasks, _ = load_tasks(a.part, None, names)
    os.makedirs(os.path.join(OUT, "per_task"), exist_ok=True)
    for name in want:
        fn = reg[name]
        if name.startswith("chronos_bolt"):
            fn = BN._mem_guard(fn, BN.BOLT_BATCH)
        t0 = time.time()
        try:
            r = evaluate(fn, part=a.part, verbose=False, tasks=tasks)
        except ImportError as e:
            print(f"{name:20s} skipped (not installed: {e.name})")
            continue
        tag = f"{name}_{a.part}" + (f"_{'+'.join(sorted(names))}" if names else "")
        r["per_dataset"].to_csv(os.path.join(OUT, "per_task", f"{tag}.csv"), index=False)
        row = {"model": name, "part": a.part, "tasks": len(tasks), **{k: round(r[k], 4) for k in
               ("wql_rel", "mase_rel", "cov80_error", "wql_rel_family", "wql_rel_rw", "mase_rel_rw")},
               "seconds": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d"), "git": BN._git_head(),
               "subset": ",".join(sorted(names)) if names else "all"}
        path = os.path.join(OUT, "baselines.tsv")
        pd.DataFrame([row]).to_csv(path, sep="\t", index=False, mode="a", header=not os.path.exists(path))
        with pd.option_context("display.width", 200, "display.precision", 3):
            print(r["per_dataset"][["dataset", "n_forecasts", "wql_rel", "mase_rel", "cov80", "wql_rel_rw"]]
                  .to_string(index=False))
        print(row, flush=True)


def cmd_model(a):
    """Score a frozen train_wm.py checkpoint on the tier without editing train_wm.py: run it in eval-only mode
    (its stored config is restored and verified by train_wm itself) with bench.evaluate wrapped, so the call that
    would score VAL scores the finance tier. The gate reads the checkpoint's own run_cfg()["train_datasets"]."""
    import runpy
    if any(k.startswith("TSAR_") and k not in ("TSAR_DEVICE", "TSAR_WM_TTA", "TSAR_WM_LONG", "TSAR_WM_LONG_PATHS")
           for k in os.environ):
        raise SystemExit("finance_tier model: unset TSAR_* keys (the checkpoint's config is restored by train_wm)")
    os.environ["TSAR_EVAL_ONLY"] = a.ckpt
    orig = BN.evaluate

    def wrapped(predict_fn, split="val", verbose=True, names=None, tasks=None):
        val = orig(predict_fn, split, verbose, names, tasks) if a.with_val else None
        g = predict_fn.__globals__
        if g.get("DEV") or g.get("EVAL_TASKS"):
            raise SystemExit("finance_tier model: DEV / smoke-subset checkpoints are not scored on the tier")
        cfg = g["run_cfg"]()
        r = evaluate(predict_fn, part=a.part, train_datasets=cfg["train_datasets"], verbose=True)
        print("---")
        print(f"fin_ckpt: {a.ckpt}  family={cfg['family']}  train_datasets={len(cfg['train_datasets'])}")
        print_metrics(r, f"fin_{a.part}")
        os.makedirs(os.path.join(OUT, "per_task"), exist_ok=True)
        r["per_dataset"].to_csv(os.path.join(OUT, "per_task", f"ckpt_{os.path.basename(a.ckpt)[:-3]}_{a.part}.csv"),
                                index=False)
        if val is None:
            raise SystemExit(0)          # stop before train_wm prints VAL-labelled lines
        return val

    BN.evaluate = wrapped
    sys.argv = ["train_wm.py"]
    runpy.run_path(os.path.join(ROOT, "train_wm.py"), run_name="__main__")


def cmd_summary(a):
    for t in TASKS:
        sel, fin = _selected_keys(t)
        b = build_task(t, "sel")
        L = [len(c) for c in b["contexts"]]
        tg = b["targets"]
        print(f"{t['name']:16s} src={t['source']:18s} freq={t['freq']} h={t['horizon']:3d} m={t['season']:2d} "
              f"sel: series={len(set(b['series_id'])):4d} forecasts={len(L):4d} ctx_med={int(np.median(L)) if L else 0:5d} "
              f"target_nan={float(np.isnan(tg).mean()) if len(L) else 0:.3f} | final series={len(fin)} "
              f"| TRAIN_DROP removed={len(set(BN.TRAIN_DROP.get(t['source'], ())))} FIN_DROP={len(FIN_DROP.get(t['source'], ()))}")
    print("eligibility for the GIFT-clean LOCAL recipe (current main line):")
    tr = [n for n, *_ in BN.TRAIN_LOCAL if n not in BN.GIFT_TEST_OVERLAP]
    for k, v in eligibility(tr).items():
        print(f"  {k:16s} {'eligible' if v is None else 'skipped: ' + v}")


assert not set(SOURCES) & set(BN.EXCLUDED_FAMILIES)
assert {"exchange_rate", "monash_fred_md"} <= set(BN.TRAIN_EXCLUDED), "FX / FRED-MD must stay out of TRAIN"
assert not {"exchange_rate", "monash_fred_md"} & {n for n, *_ in BN.TRAIN_LOCAL + BN.TRAIN_EXTRA}
assert not {d for _, d, *_ in BN.TRAIN_GEP} & set(SOURCES)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["summary", "baselines", "scan", "status", "model", "selftest"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--part", default="sel", choices=["sel", "final"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--kind", default="hash", choices=["hash", "lag"])
    ap.add_argument("--train", default="")
    ap.add_argument("--sources", default="")
    ap.add_argument("--ckpt", default="", help="model: a train_wm.py checkpoint (data/ckpt/<run>.pt)")
    ap.add_argument("--with-val", action="store_true", help="model: also score bench VAL first")
    ap.add_argument("--shard", default="0/1", help="scan: k/n = every n-th TRAIN series from k (big datasets)")
    a = ap.parse_args()
    {"summary": cmd_summary, "baselines": cmd_baselines, "scan": cmd_scan, "status": cmd_status,
     "model": cmd_model, "selftest": selftest}[a.cmd](a)
