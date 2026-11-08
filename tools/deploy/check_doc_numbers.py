"""Every decimal number in docs/deploy_2026_10_11.md must come from the deployment outputs (no hand-typed numbers).

    .venv/bin/python tools/deploy/check_doc_numbers.py        # exit 1 if any number has no source

Sources: every numeric value in runs/deploy_2026_10_11/**/*.json and every number printed in RESULTS.md / AUDIT.md.
A document number matches a source value when the value, formatted with the same number of decimals (or the same
scientific mantissa), gives exactly the same string.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "runs", "deploy_2026_10_11")
DOC = os.path.join(ROOT, "docs", "deploy_2026_10_11.md")
NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?(?![\w.])")


def walk(x, out):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            walk(v, out)
    elif isinstance(x, list):
        for v in x:
            walk(v, out)
    elif isinstance(x, str):
        out += [float(t) for t in NUM.findall(x)]


vals, text = [], ""
for p in glob.glob(os.path.join(RUN, "**", "*.json"), recursive=True) + [
        os.path.join(ROOT, "runs", "deploy_daily", "EXANTE_LEDGER.json")]:
    walk(json.load(open(p)), vals)
for p in ("RESULTS.md", "AUDIT.md"):
    t = open(os.path.join(RUN, p)).read()
    text += t
    vals += [float(x) for x in NUM.findall(t)]
vals = set(vals)
bad, n = [], 0
for tok in NUM.findall(open(DOC).read()):
    n += 1
    if tok in text:
        continue
    if "e" in tok:
        d = len(tok.split("e")[0].split(".")[1])
        fmt = lambda v: f"{v:.{d}e}"          # noqa: E731
    else:
        d = len(tok.split(".")[1])
        fmt = lambda v: f"{v:.{d}f}"          # noqa: E731
    if not any(fmt(v) == tok or fmt(-v) == tok for v in vals):
        bad.append(tok)
print(f"{n - len(bad)}/{n} numbers in {os.path.relpath(DOC, ROOT)} traced to deployment outputs")
if bad:
    print("UNTRACED:", bad)
sys.exit(1 if bad else 0)
