"""Cross-check: every decimal number in the LaTeX report must appear in a source file (after rounding), else it is listed."""
import glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "report/alma_status_report.tex")).read()
tex = re.sub(r"%.*", "", tex)
srcs = (glob.glob(f"{ROOT}/repo/runs/*/test/report*.md") + glob.glob(f"{ROOT}/repo/winning_models/*/MODEL_CARD.md")
        + glob.glob(f"{ROOT}/audit/*.csv") + glob.glob(f"{ROOT}/audit/validation/*.out") + glob.glob(f"{ROOT}/audit/*.json")
        + [f"{ROOT}/audit/GAS_FREEZE.md", f"{ROOT}/audit/REPORT.md", f"{ROOT}/report/facts.json", f"{ROOT}/repo/README.md",
           f"{ROOT}/repo/alma_goal.md", f"{ROOT}/repo/docs/fev_full_results.md", f"{ROOT}/repo/docs/almats_research_program.md",
           f"{ROOT}/repo/docs/live_gas_research.md", f"{ROOT}/repo/docs/live_search.md", f"{ROOT}/repo/runs/live/eval/val/report.md",
           f"{ROOT}/repo/docs/research_history.md", f"{ROOT}/repo/docs/ssm_wm_test_5seed.md", f"{ROOT}/repo/docs/reproducibility.md",
           f"{ROOT}/repo/tools/gas/wm_gas_ens.py", f"{ROOT}/repo/tools/forward/forward_eval.py"])
corpus = "\n".join(open(p, errors="ignore").read() for p in srcs)
vals = set()
for m in re.finditer(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?", corpus):
    try: vals.add(float(m.group()))
    except ValueError: pass
def found(x, nd):
    if x in vals or -x in vals: return True
    tol = 0.5 * 10 ** (-nd) + 1e-12
    return any(abs(abs(v) - abs(x)) <= tol for v in vals)
miss = []
for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", tex):
    s = m.group(1); x = float(s); nd = len(s.split(".")[1])
    if not found(x, nd) and not found(x / 100, nd + 2) and not found(x * 100, max(nd - 2, 0)):
        ctx = tex[max(0, m.start() - 50):m.end() + 20].replace("\n", " ")
        miss.append((s, ctx))
print(f"checked numbers; {len(miss)} not found directly in sources:")
for s, c in miss: print(f"  {s:>9s} | ...{c}...")
