"""
Alma-Live Gas PAPER broker. No real orders, ever: there is no broker API in this file or anywhere in tools/live.

Clock (common.py, single source of truth): decide at 15:30 ET on trading day d with ctx = Store().asof(decision_time,
live=True) (published_at <= t AND fetched_at <= t); fill at the 16:00 ET close of d plus costs; the close is known
from the d 18:00 ET bar, so fills/marks are written by `settle` in the evening. Positions are held close-to-close.

Instruments (weights in [-1, 1] of equity; gross |w| sum capped at MAX_GROSS):
  UNG  primary   ETF; cost = COST_BP (10 bp) per side on traded notional (spread + slippage + commission)
  BOIL secondary 2x ETF; same cost per side, |w| <= 0.5
  NG   research  NYMEX front month by the calendar roll rule (common.ng_active_contract), 10,000 MMBtu, tick 0.001 =
                 $10; cost per contract side = SLIP_TICKS ticks + $2.50 commission; a roll = close old + open new
                 (both charged). Disabled unless --instruments includes NG (contract-level history is forward only).
Risk: daily loss >= 2% of start-of-day equity -> 'halt_day' (next decision forced flat = kill switch flattens at the
next close); drawdown >= 10% from peak -> 'halt_review' (flat until a human appends `resume --note ...`); stale data
at decision time -> decision forced flat with reason.

Ledger (runs/live/paper/<book>/ledger.jsonl): append-only JSON lines, each with seq, kind, wall-clock written_at,
prev (hash of the previous line) and hash = sha256(prev + canonical JSON of the line without 'hash'). Decision lines
carry the model id/meta, ctx input sha256, data manifest sha256, git HEAD, the forecast and target weights. In live
mode a decision is accepted only inside [decision_time, decision_time + LATE_MIN] of wall-clock time, once per day;
settle is accepted only after the fill bar is published. `verify` re-checks the chain. Backfilling a past day is
refused in live mode (the wall clock is not under the caller's control).

    .venv/bin/python tools/live/paper.py decide --book shadow --model tsmom_20       # at 15:30-15:50 ET
    .venv/bin/python tools/live/paper.py settle --book shadow                        # after 18:00 ET
    .venv/bin/python tools/live/paper.py status|verify --book shadow
    .venv/bin/python tools/live/paper.py resume --book shadow --note "reviewed by <name>: ..."
"""
import os
import sys
import json
import argparse
import hashlib
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PAPER_DIR = os.path.join(C.RUNS, "paper")
COST_BP = 10.0
LIMITS = {"UNG": 1.0, "BOIL": 0.5, "NG": 1.0}
MAX_GROSS = 1.0
DAILY_LOSS_HALT = 0.02
DRAWDOWN_HALT = 0.10
LATE_MIN = 20
NG_MULT, NG_TICK, NG_COMM, SLIP_TICKS = 10000.0, 0.001, 2.50, 1
CAPITAL = 1_000_000.0
GENESIS = "0" * 64


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


class Ledger:
    def __init__(self, path):
        self.path = path
        self.lines = []
        if os.path.exists(path):
            with open(path) as f:
                self.lines = [json.loads(x) for x in f if x.strip()]

    @property
    def head(self):
        return self.lines[-1]["hash"] if self.lines else GENESIS

    def append(self, kind, body, written_at=None):
        line = {"seq": len(self.lines), "kind": kind, "written_at": C.iso(written_at or C.utcnow()),
                "prev": self.head, **body}
        line["hash"] = hashlib.sha256((line["prev"] + canon({k: v for k, v in line.items() if k != "hash"})).encode()).hexdigest()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(canon(line) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.lines.append(line)
        return line

    def verify(self):
        prev, last_w = GENESIS, None
        for i, ln in enumerate(self.lines):
            if ln["seq"] != i:
                return False, f"seq {ln['seq']} at position {i}"
            if ln["prev"] != prev:
                return False, f"broken link at seq {i}"
            h = hashlib.sha256((ln["prev"] + canon({k: v for k, v in ln.items() if k != "hash"})).encode()).hexdigest()
            if h != ln["hash"]:
                return False, f"hash mismatch at seq {i} (line edited)"
            if last_w is not None and i > 1 and ln["written_at"] < last_w:
                return False, f"written_at goes backwards at seq {i}"
            prev, last_w = ln["hash"], ln["written_at"]
        return True, f"ok ({len(self.lines)} lines, head {prev[:12]})"

    def verify_anchor(self, ref="HEAD"):
        """External anchor (red-team LRT-19a/b): the committed version of this ledger at `ref` must be a byte prefix of
        the current file, so truncation or a recomputed rewrite of committed history is detected."""
        relp = os.path.relpath(self.path, C.ROOT)
        try:
            old = subprocess.check_output(["git", "-C", C.ROOT, "show", f"{ref}:{relp}"], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return True, "no committed version yet (nothing to anchor against)"
        cur = open(self.path, "rb").read()
        if not cur.startswith(old):
            return False, f"committed ledger at {ref} is not a prefix of the current file (truncated or rewritten)"
        n_old = old.count(b"\n")
        return True, f"anchored: {n_old} committed lines are an exact prefix"

    def anchor(self, msg, tries=6):
        """Commit only this ledger file (git commit -- <path>); the commit is the external anchor of the head hash.
        Retries on a git index-lock race (other agents committing); every attempt's head is also appended to
        runs/live/paper/anchors.log (outside the ledger) for the manager."""
        import time
        relp = os.path.relpath(self.path, C.ROOT)
        ok = False
        for k in range(tries):
            a = subprocess.run(["git", "-C", C.ROOT, "add", "-f", relp], capture_output=True, text=True)
            r = subprocess.run(["git", "-C", C.ROOT, "commit", "-q", "-m",
                                f"paper ledger {relp}: {msg} head {self.head[:16]}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
                                "--", relp], capture_output=True, text=True)
            if a.returncode == 0 and r.returncode == 0:
                ok = True
                break
            if "index.lock" not in (a.stderr + r.stderr):
                break
            time.sleep(2 * (k + 1))
        os.makedirs(os.path.dirname(os.path.dirname(self.path)), exist_ok=True)
        with open(os.path.join(os.path.dirname(os.path.dirname(self.path)), "anchors.log"), "a") as f:
            f.write(canon({"ledger": relp, "head": self.head, "seq": len(self.lines) - 1, "msg": msg,
                           "committed": ok, "at": C.iso(C.utcnow())}) + "\n")
        return ok

    def of(self, kind):
        return [x for x in self.lines if x["kind"] == kind]


# --------------------------------------------------------------------------- book state
class Book:
    """Paper account. mode='live' enforces the wall clock; mode='sim' takes the clock from the caller (backtests,
    shadow replays: never mixed with a live ledger)."""

    def __init__(self, name, mode="live", root=PAPER_DIR, instruments=("UNG", "BOIL"), cost_bp=COST_BP,
                 capital=CAPITAL, start=None):
        self.name, self.mode = name, mode
        self.L = Ledger(os.path.join(root, name, "ledger.jsonl"))
        if not self.L.lines:
            self.L.append("genesis", {"book": name, "mode": mode, "instruments": list(instruments), "cost_bp": cost_bp,
                                      "capital": capital, "limits": LIMITS, "max_gross": MAX_GROSS,
                                      "daily_loss_halt": DAILY_LOSS_HALT, "drawdown_halt": DRAWDOWN_HALT,
                                      "clock": {"decision_et": f"{C.DECISION_HH:02d}:{C.DECISION_MM:02d}",
                                                "fill_et": f"{C.FILL_HH:02d}:{C.FILL_MM:02d} close"},
                                      "ng": {"mult": NG_MULT, "tick": NG_TICK, "comm": NG_COMM, "slip_ticks": SLIP_TICKS,
                                             "roll_bdays": C.ROLL_BDAYS},
                                      "git_head": git_head(), "real_orders": False},
                          written_at=None if mode == "live" else (start or pd.Timestamp("2000-01-01", tz="UTC")))
        g = self.L.lines[0]
        if g["mode"] != mode:
            raise SystemExit(f"book {name} is a {g['mode']} ledger; refusing to open it in {mode} mode (red-team LRT-19c)")
        self.instruments, self.cost_bp, self.capital = g["instruments"], g["cost_bp"], g["capital"]

    # state from the ledger (single source of truth)
    def state(self):
        st = {"equity": self.capital, "peak": self.capital, "pos": {}, "contract": None, "halt": None,
              "halt_day": None, "last_decision_day": None, "last_settled_day": None}
        for ln in self.L.lines:
            k = ln["kind"]
            if k == "decision":
                st["last_decision_day"] = ln["day"]
            elif k == "mark":
                st["equity"], st["peak"], st["pos"] = ln["equity"], ln["peak"], ln["pos"]
                st["contract"] = ln.get("contract")
                st["last_settled_day"] = ln["day"]
            elif k == "halt":
                if ln["reason"].startswith("halt_review"):
                    st["halt"] = ln["reason"]
                else:
                    st["halt_day"] = ln["day"]
            elif k == "resume":
                st["halt"] = None
        return st

    # ---------------------------------------------------------------- decide
    def decide(self, day, model, ctx, now=None, stale=None):
        day = pd.Timestamp(day).normalize()
        dt_ = C.decision_time(day)
        now = now or C.utcnow()
        if self.mode == "live":
            if not C.is_bday(day):
                raise SystemExit(f"{day.date()} is not a trading day")
            if not (dt_ <= now <= dt_ + pd.Timedelta(minutes=LATE_MIN)):
                raise SystemExit(f"live decision for {day.date()} refused: wall clock {C.iso(now)} outside "
                                 f"[{C.iso(dt_)}, +{LATE_MIN} min] (no backfill)")
            assert ctx["live"] and ctx["t"] == dt_, "live decisions need ctx = asof(decision_time, live=True)"
        st = self.state()
        if st["last_decision_day"] is not None and pd.Timestamp(st["last_decision_day"]) >= day:
            raise SystemExit(f"decision for {day.date()} refused: already decided {st['last_decision_day']}")
        if st["last_decision_day"] is not None and st["last_settled_day"] != st["last_decision_day"]:
            raise SystemExit(f"decision for {day.date()} refused: {st['last_decision_day']} not settled yet")
        reason = None
        pred, w = {}, {}
        with C.no_data_access():
            if hasattr(model, "predict"):
                pred = model.predict(ctx) or {}
            w = model.act(ctx) if hasattr(model, "act") else {}
        w = {k: float(np.clip(np.nan_to_num(float(v)), -LIMITS.get(k, 1), LIMITS.get(k, 1)))
             for k, v in (w or {}).items() if k in self.instruments}
        g = sum(abs(v) for v in w.values())
        if g > MAX_GROSS:
            w = {k: v * MAX_GROSS / g for k, v in w.items()}
        if stale:
            reason = "stale_data:" + ",".join(stale)
        elif st["halt"]:
            reason = st["halt"]
        elif st["halt_day"] is not None:
            prev_settled = st["last_settled_day"]
            if prev_settled == st["halt_day"]:
                reason = f"halt_day after {st['halt_day']}"
        if reason:
            w = {k: 0.0 for k in self.instruments}
        man_sha = C.sha256_file(C.MANIFEST) if os.path.exists(C.MANIFEST) else None
        body = {"day": str(day.date()), "decision_time": C.iso(dt_), "model": getattr(model, "name", type(model).__name__),
                "model_meta": getattr(model, "meta", {}), "input_sha256": ctx["meta"]["input_sha256"],
                "ctx_max_published_at": C.iso(ctx["meta"]["max_published_at"]) if ctx["meta"]["max_published_at"] is not None else None,
                "ctx_max_fetched_at": C.iso(ctx["meta"]["max_fetched_at"]) if ctx["meta"]["max_fetched_at"] is not None else None,
                "manifest_sha256": man_sha, "git_head": git_head() if self.mode == "live" else None,
                "forecast": {k: (float(v) if np.isscalar(v) else v) for k, v in pred.items()},
                "target": w, "forced_flat": reason, "ng_active": ctx["ng_roll"]["active"]}
        return self.L.append("decision", body, written_at=now if self.mode == "live" else max(now, dt_))

    # ---------------------------------------------------------------- settle (fill at the close + mark)
    def settle(self, day, closes, now=None, splits=None):
        """closes: {instrument: close price of `day`} (NG: {'NG': (contract_symbol, close)} for the active contract,
        and the previous contract's close on roll days via closes['NG_prev']). Fills the day's decision at these
        closes with costs and marks the book. Must be called after the fill bar is published (d 18:00 ET)."""
        day = pd.Timestamp(day).normalize()
        now = now or C.utcnow()
        if self.mode == "live" and now < C.et(day, 18, 0):
            raise SystemExit("settle refused: the 16:00 close bar is published at 18:00 ET")
        dec = [x for x in self.L.of("decision") if x["day"] == str(day.date())]
        if not dec:
            raise SystemExit(f"no decision for {day.date()}")
        if any(x["day"] == str(day.date()) for x in self.L.of("mark")):
            raise SystemExit(f"{day.date()} already settled")
        dec = dec[-1]
        st = self.state()
        eq0 = st["equity"]
        pos = dict(st["pos"])                # instrument -> quantity (shares / contracts)
        # 1) P&L of the position held since the previous close, valued at today's close
        prev_marks = self.L.of("mark")
        prev_close = dict(prev_marks[-1]["closes"]) if prev_marks else {}
        # splits with ex-date in (previous mark day, day]: shares x factor, previous close / factor (red-team LRT-20)
        if prev_marks and splits:
            pd0 = pd.Timestamp(prev_marks[-1]["day"])
            for k, evs in splits.items():
                for sd, fac in evs:
                    sd = pd.Timestamp(sd)
                    if pd0 < sd <= day and k in prev_close:
                        if pos.get(k):
                            pos[k] = pos[k] * fac
                        prev_close[k] = prev_close[k] / fac
                        self.L.append("split", {"day": str(day.date()), "instrument": k, "ex_date": str(sd.date()),
                                                "factor": fac}, written_at=now)
        pnl = 0.0
        for k, q in pos.items():
            if not q:
                continue
            if k == "NG":
                sym_prev = st["contract"]
                c_now = closes.get("NG_prev", closes["NG"])[1] if closes["NG"][0] != sym_prev else closes["NG"][1]
                pnl += q * (c_now - prev_close["NG"][1]) * NG_MULT
            else:
                pnl += q * (closes[k] - prev_close[k])
        eq_pre = eq0 + pnl
        # 2) trade to the target at today's close
        cost, fills, newpos = 0.0, [], {}
        contract = closes["NG"][0] if "NG" in closes else None
        for k in self.instruments:
            w = dec["target"].get(k, 0.0)
            if k == "NG":
                px = closes["NG"][1]
                tgt = int(round(w * eq_pre / (px * NG_MULT)))
                cur = pos.get("NG", 0)
                rolled = st["contract"] is not None and st["contract"] != contract and cur != 0
                per = SLIP_TICKS * NG_TICK * NG_MULT + NG_COMM
                n_traded = (abs(cur) + abs(tgt)) if rolled else abs(tgt - cur)
                c = n_traded * per
                if tgt != cur or rolled:
                    fills.append({"instrument": "NG", "contract": contract, "from": cur, "to": tgt, "price": px,
                                  "rolled": rolled, "cost": c})
                newpos["NG"] = tgt
            else:
                px = closes[k]
                tgt = int(np.floor(abs(w) * eq_pre / px)) * int(np.sign(w))
                cur = pos.get(k, 0)
                c = abs(tgt - cur) * px * self.cost_bp * 1e-4
                if tgt != cur:
                    fills.append({"instrument": k, "from": cur, "to": tgt, "price": px, "cost": c,
                                  "cost_bp": self.cost_bp})
                newpos[k] = tgt
            cost += c
        eq = eq_pre - cost
        peak = max(st["peak"], eq)
        dd = 1 - eq / peak
        for f in fills:
            self.L.append("fill", {"day": str(day.date()), "fill_time": C.iso(C.fill_time(day)),
                                   "decision_seq": dec["seq"], **f}, written_at=now)
        ret = eq / eq0 - 1
        mark = self.L.append("mark", {"day": str(day.date()), "closes": closes, "pos": newpos, "contract": contract,
                                      "pnl_gross": pnl, "cost": cost, "equity": eq, "peak": peak, "drawdown": dd,
                                      "ret": ret, "decision_seq": dec["seq"]}, written_at=now)
        if (eq_pre / eq0 - 1) <= -DAILY_LOSS_HALT:
            self.L.append("halt", {"day": str(day.date()), "reason": f"halt_day: daily loss {eq_pre / eq0 - 1:.2%}"},
                          written_at=now)
        if dd >= DRAWDOWN_HALT and not st["halt"]:
            self.L.append("halt", {"day": str(day.date()), "reason": f"halt_review: drawdown {dd:.2%} >= 10%"},
                          written_at=now)
        return mark

    def resume(self, note, now=None):
        if not note or len(note) < 10:
            raise SystemExit("resume needs a human review note")
        return self.L.append("resume", {"note": note}, written_at=now)


# --------------------------------------------------------------------------- live helpers
def stale_checks(ctx, day):
    """Data freshness at decision time: names of stale inputs (-> forced flat)."""
    out = []
    prev = C.add_bdays(day, -1)
    u = ctx["px"].get("UNG")
    if u is None or not len(u) or u.index[-1] < prev:
        out.append("UNG_bar")
    st = ctx["storage"]
    if not len(st) or (ctx["t"] - st["released_at"].iloc[-1]) > pd.Timedelta(days=9):
        out.append("storage")
    f = ctx["cpc_fcst"]
    if not len(f) or (ctx["t"] - f["published_at"].max()) > pd.Timedelta(days=3):
        out.append("cpc_fcst")
    return out


def splits_from_store(S, now=None):
    """{instrument: [(ex_date, factor)]} from px split records published by now (factor = new shares per old)."""
    v = S.visible("px", now or C.utcnow())
    out = {}
    for k in ("UNG", "BOIL"):
        r = v[v["series"] == f"{k}.split"]
        out[k] = [(x, float(f)) for x, f in zip(r["obs"], r["value"])]
    return out


def closes_from_store(S, day):
    """Closes of `day` for the paper instruments from the latest store snapshot (published by now)."""
    v = S.visible("px", C.utcnow())
    out = {}
    ds = str(pd.Timestamp(day).date())
    for k in ("UNG", "BOIL"):
        r = v[(v["series"] == f"{k}.close") & (v["obs"] == ds)]
        if len(r):
            out[k] = float(r["value"].iloc[-1])
    act = C.ng_active_contract(day)["symbol"]
    r = v[(v["series"] == f"{act}.close") & (v["obs"] == ds)]
    if len(r):
        out["NG"] = (act, float(r["value"].iloc[-1]))
    return out


def main(argv=None):
    import models as M
    from store import Store
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "decide", "settle", "status", "verify", "resume"])
    ap.add_argument("--book", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--day", default=None, help="trading day (default: today ET)")
    ap.add_argument("--note", default="")
    ap.add_argument("--instruments", default="UNG,BOIL")
    ap.add_argument("--sim", action="store_true", help="DRY RUN: sim-mode book under runs/live/paper_sim, clock "
                                                         "taken from --day (never counts; separate root)")
    a = ap.parse_args(argv)
    day = pd.Timestamp(a.day) if a.day else C.utcnow().tz_convert(C.ET).normalize().tz_localize(None)
    mode = "sim" if a.sim else "live"
    root = os.path.join(C.RUNS, "paper_sim") if a.sim else PAPER_DIR
    B = Book(a.book, mode=mode, root=root, instruments=tuple(a.instruments.split(",")),
             start=C.decision_time(day) - pd.Timedelta(days=1) if a.sim else None)
    if a.cmd == "init":
        if not a.sim and len(B.L.lines) == 1:
            print("genesis", B.L.head, "anchored" if B.L.anchor("genesis") else "anchor commit failed")
        else:
            print("exists", B.L.head)
        return 0
    if a.cmd == "verify":
        ok, msg = B.L.verify()
        ok2, msg2 = B.L.verify_anchor()
        print(msg)
        print(msg2)
        return 0 if ok and ok2 else 1
    if a.cmd == "status":
        print(json.dumps(B.state(), indent=1, default=str))
        print(B.L.verify()[1])
        return 0
    if a.cmd == "resume":
        print(B.resume(a.note)["hash"])
        return 0
    S = Store()
    if a.cmd == "decide":
        model = M.load(a.model or "flat")
        now = C.decision_time(day) if a.sim else None
        ctx = S.asof(C.decision_time(day), live=not a.sim)
        ln = B.decide(day, model, ctx, now=now, stale=stale_checks(ctx, day))
        print(json.dumps({k: ln[k] for k in ("seq", "day", "model", "target", "forced_flat", "hash")}))
        if not a.sim:
            print("anchored" if B.L.anchor(f"decide {day.date()}") else "anchor commit failed (commit manually)")
        return 0
    if a.cmd == "settle":
        cl = closes_from_store(S, day)
        if not all(k in cl for k in B.instruments):
            raise SystemExit(f"closes missing for {day.date()}: {cl} (run update.py --only px after 18:00 ET)")
        now = C.et(day, 18, 5) if a.sim else None
        ln = B.settle(day, cl, now=now, splits=splits_from_store(S, now))
        print(json.dumps({k: ln[k] for k in ("seq", "day", "equity", "drawdown", "ret", "hash")}))
        if not a.sim:
            print("anchored" if B.L.anchor(f"settle {day.date()}") else "anchor commit failed (commit manually)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
