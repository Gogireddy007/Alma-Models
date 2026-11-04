"""Build every figure of the project report from the source files (no hand-typed numbers).

Sources: repo/runs/*/test/report*.md (official TEST reports), audit/*.csv and audit/validation/*.out (audit re-runs),
repo/docs/almats_research_program.md (bench-VAL bars). Writes figures/*.pdf and facts.json (every plotted value).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO, AUD = os.path.join(ROOT, "repo"), os.path.join(ROOT, "audit")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
FACTS = {}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
C = {"alma": "#1d4e89", "new": "#c2410c", "base": "#9aa5b1", "fm": "#4b8b3b", "comp": "#7c5295", "grey": "#52606d"}


def md_table(path, heading, nth=0):
    """Parse the nth markdown table that follows the line starting with `heading`."""
    lines = open(path).read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith(heading)]
    i = starts[nth] + 1
    while not lines[i].startswith("|"):
        i += 1
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append([c.strip() for c in lines[i].strip("|").split("|")])
        i += 1
    return pd.DataFrame(rows[2:], columns=rows[0])


def num(x):
    m = re.match(r"^\s*([-+]?\d*\.?\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def clean(name):
    return name.replace(" (CONTAMINATED)", "").replace(" C", "").replace(" S", "").strip()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def kind_color(kind, model):
    if model.startswith("wm_") or kind in ("candidate",):
        return C["alma"]
    if kind == "alma":
        return "#6b8fbf"
    if kind == "fm":
        return C["fm"]
    if kind == "competitor":
        return C["comp"]
    return C["base"]


def hbar(ax, names, vals, colors, fmt, xlabel, title, highlight=None):
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.68)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(vals) * 1.18)
    if highlight:
        for t in ax.get_yticklabels():
            if t.get_text() in highlight:
                t.set_fontweight("bold")


# ---------------------------------------------------------------- 1. power P4 quantile score
pw = md_table(os.path.join(REPO, "runs/power/test/report.md"), "## Probabilistic / point accuracy")
pw["model"] = pw["model"].map(clean)
pw = pw[~pw["model"].isin(["kronos_small", "timestral_5m"])]          # subsample / contaminated: not ranked
pw["QS"] = pw["QS (0.1..0.9)"].map(num)
pb = open(os.path.join(AUD, "validation/power_blend.out")).read()
blend_qs = float(re.search(r"TEST blend w=0.5\s+P1 [\d.]+ P2 [\d.]+ QS ([\d.]+)", pb).group(1))
rows = list(zip(pw["model"], pw["QS"], pw["kind"])) + [("wm_power_blend (audit)", blend_qs, "new")]
rows.sort(key=lambda r: r[1])
FACTS["power_qs"] = {r[0]: r[1] for r in rows}
fig, ax = plt.subplots(figsize=(5.6, 3.7))
hbar(ax, [r[0] for r in rows], [r[1] for r in rows],
     [C["new"] if r[2] == "new" else kind_color(r[2], r[0]) for r in rows], "{:.2f}",
     "Quantile score, EUR/MWh (lower is better)", "German day-ahead power, TEST 2025-11-01 to 2026-10-31 (365 days)",
     {"wm_power", "wm_power_blend (audit)"})
save(fig, "power_qs")

# ---------------------------------------------------------------- 2. power P2 hourly direction
pd2 = md_table(os.path.join(REPO, "runs/power/test/report.md"), "## Direction")
pd2["model"] = pd2["model"].map(clean)
pd2 = pd2[~pd2["model"].isin(["kronos_small", "timestral_5m", "majority", "naive_d"])]
pd2["P2"] = pd2["P2 acc [day-block 95%]"].map(num)
blend_p2 = float(re.search(r"TEST blend w=0.5\s+P1 [\d.]+ P2 ([\d.]+)", pb).group(1))
rows = list(zip(pd2["model"], pd2["P2"], pd2["kind"])) + [("wm_power_blend (audit)", blend_p2, "new")]
rows.sort(key=lambda r: -r[1])
FACTS["power_p2"] = {r[0]: r[1] for r in rows}
fig, ax = plt.subplots(figsize=(5.6, 3.5))
hbar(ax, [r[0] for r in rows], [r[1] for r in rows],
     [C["new"] if r[2] == "new" else kind_color(r[2], r[0]) for r in rows], "{:.3f}",
     "Hourly direction accuracy P2 (higher is better)", "German power, hourly direction (P2), TEST",
     {"wm_power", "wm_power_blend (audit)"})
ax.set_xlim(0.5, 0.8)
save(fig, "power_p2")

# ---------------------------------------------------------------- 3. gas MAE + QS (Thursday origin)
gs = md_table(os.path.join(REPO, "runs/gas/test/report.md"), "### origin thu, Lower 48")
gs["model"] = gs["model"].map(clean)
gs = gs[~gs["model"].isin(["majority", "timestral_5m"])]
gs["MAE"] = gs["MAE Bcf"].map(num)
gs["QSv"] = gs["QS"].map(num)
g2 = open(os.path.join(AUD, "validation/gas_final2.out")).read()
ens = re.search(r"thu candidate \+ adaptive intervals\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", g2)
ens_mae, ens_qs = float(ens.group(1)), float(ens.group(2))
rows = list(zip(gs["model"], gs["MAE"], gs["QSv"], gs["kind"])) + [("wm_gas_ens (audit)", ens_mae, ens_qs, "new")]
FACTS["gas_thu"] = {r[0]: {"MAE": r[1], "QS": r[2]} for r in rows}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=False)
for ax, j, lab in ((axs[0], 1, "MAE, Bcf (lower is better)"), (axs[1], 2, "Quantile score, Bcf (lower is better)")):
    rr = sorted(rows, key=lambda r: r[j])
    hbar(ax, [r[0] for r in rr], [r[j] for r in rr], [C["new"] if r[3] == "new" else kind_color(r[3], r[0]) for r in rr],
         "{:.1f}" if j == 1 else "{:.2f}", lab, "MAE" if j == 1 else "Quantile score",
         {"wm_gas", "wm_gas_ens (audit)"})
fig.suptitle("US gas storage (EIA weekly change, Lower 48), Thursday origin, TEST 52 weeks", x=0.02, ha="left", fontsize=9.5)
fig.tight_layout()
save(fig, "gas_mae_qs")

# ---------------------------------------------------------------- 4. gas DEV scaling: per-fold MAE
ge = pd.read_csv(os.path.join(AUD, "gas_exp_scaled.csv")).set_index("exp")
sel = {"OLS + weather (ref_ols_wx)": "ref_ols_wx", "Frozen recipe, 7 seeds (E0)": "E0 frozen recipe: wm x7",
       "Rollout + direct pool, 14 seeds (E2)": "E2 wm+direct x14", "E2 + OLS, LOFO blend": "E2+ref_ols_wx LOFO blend"}
fig, ax = plt.subplots(figsize=(5.6, 2.9))
folds = np.arange(1, 7)
marks = ["s", "o", "^", "D"]
for (lab, key), mk, col in zip(sel.items(), marks, [C["base"], C["alma"], "#6b8fbf", C["new"]]):
    v = [float(x) for x in ge.loc[key, "fold_mae_thu"].split()]
    FACTS.setdefault("gas_dev_folds", {})[lab] = v
    ax.plot(folds, v, marker=mk, color=col, label=f"{lab}: mean {ge.loc[key, 'mae_thu']:.2f}")
ax.set_xticks(folds, [f"F{i}" for i in folds])
ax.set_ylabel("DEV MAE, Bcf (Thursday)")
ax.set_xlabel("Rolling DEV fold (2019-07 to 2025-06)")
ax.set_title("Gas: small experiment scaled up on the DEV backtest (no TEST data used)", loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
save(fig, "gas_dev_folds")

# ---------------------------------------------------------------- 5. crypto accuracy vs QS
fs = md_table(os.path.join(REPO, "runs/fin/test/report.md"), "### Forecasters, group all (24 assets)")
fs["model"] = fs["model"].map(clean)
fs = fs[~fs["model"].isin(["majority", "kronos_small", "wm_fin_cl:frozen", "timestral_5m"])]
fs["P1"] = fs["P1 acc"].map(num)
fs["QSv"] = fs["P3 QS"].map(num)
fb = open(os.path.join(AUD, "validation/fin_blend.out")).read()
m = re.search(r"TEST all\s+wm_fin acc [\d.]+ qs [\d.]+ \| har_intra_dow acc [\d.]+ qs [\d.]+ \| blend acc ([\d.]+) qs ([\d.]+)", fb)
b_acc, b_qs = float(m.group(1)), float(m.group(2))
FACTS["fin_all"] = {r.model: {"P1": r.P1, "QS": r.QSv} for r in fs.itertuples()}
FACTS["fin_all"]["wm_fin_blend (audit)"] = {"P1": b_acc, "QS": b_qs}
fig, (ax, az) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
OFF = {"wm_fin": (-38, 5), "wm_fin_cl": None, "wm_direct": (4, -9), "har_intra_dow": (4, 3), "wm_b_moe4": (-46, -8),
       "lgbm_dow": (4, 2), "har_dow": (4, 3), "chronos2": (4, -7), "tirex": (4, 2), "chronos_bolt_base": (4, -8)}
for r in fs.itertuples():
    col = kind_color(r.kind, r.model)
    ax.scatter(r.QSv, r.P1, color=col, s=20, zorder=3)
    if r.model in ("lgbm", "rv_persist", "garch"):
        ax.annotate(r.model, (r.QSv, r.P1), fontsize=6.3, xytext=(3, 2), textcoords="offset points")
    if r.P1 >= 0.70:
        az.scatter(r.QSv, r.P1, color=col, s=28, zorder=3)
        if OFF.get(r.model):
            lab = "wm_fin, wm_fin_cl" if r.model == "wm_fin" else r.model
            az.annotate(lab, (r.QSv, r.P1), fontsize=6.6, xytext=OFF[r.model], textcoords="offset points")
for a in (ax, az):
    a.scatter(b_qs, b_acc, color=C["new"], s=60, marker="*", zorder=4)
ax.annotate("har, harx, har_roll,\nlgbm_asset", (0.1775, 0.638), fontsize=6.3, xytext=(4, -16), textcoords="offset points")
az.annotate("wm_fin_blend (audit)", (b_qs, b_acc), fontsize=6.8, color=C["new"], xytext=(5, 2), textcoords="offset points")
ax.add_patch(plt.Rectangle((0.143, 0.703), 0.019, 0.042, fill=False, ls=":", color=C["grey"]))
ax.set_xlabel("Quantile score (lower is better)")
ax.set_ylabel("Volatility-direction accuracy P1")
ax.set_title("All models", loc="left")
az.set_xlabel("Quantile score (lower is better)")
az.set_title("Zoom: the leading group", loc="left")
fig.suptitle("Crypto volatility, 24 Binance perpetuals, TEST 2025-11-01 to 2026-10-30", x=0.02, ha="left", fontsize=9.5)
fig.tight_layout()
save(fig, "fin_scatter")

# ---------------------------------------------------------------- 6. oil skill vs random walk
oc = md_table(os.path.join(REPO, "runs/commodity/test/report_wm_sv.md"), "### wti T4 log-price quantile score, h=1")
oc["model"] = oc["model"].map(clean)
oc = oc[~oc["model"].isin(["kronos_small", "timestral_5m"]) & ~oc["model"].str.startswith("alma_s")
        & ~oc["model"].str.startswith("alma_direct_s0_tta")]
oc["skill"] = oc["skill vs RW"].map(num) * 100
ocsv = pd.read_csv(os.path.join(AUD, "oil_check.csv"))
ew = float(ocsv[(ocsv.h == 1) & (ocsv.model == "ewma94")].skill_vs_rw.iloc[0]) * 100
rows = list(zip(oc["model"], oc["skill"])) + [("EWMA(0.94) random walk (audit)", ew)]
rows.sort(key=lambda r: -r[1])
FACTS["oil_skill_h1_pct"] = dict(rows)
fig, ax = plt.subplots(figsize=(5.6, 2.9))
y = np.arange(len(rows))
cols = [C["new"] if "audit" in r[0] else (C["alma"] if r[0].startswith("wm_") else C["base"]) for r in rows]
ax.barh(y, [r[1] for r in rows], color=cols, height=0.65)
ax.set_yticks(y, [r[0] for r in rows])
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.6)
for yi, r in zip(y, rows):
    ax.text(r[1], yi, f" {r[1]:+.2f}%" if r[1] >= 0 else f"{r[1]:+.2f}% ", va="center", ha="left" if r[1] >= 0 else "right", fontsize=7)
ax.set_xlabel("Quantile-score skill vs Gaussian random walk, % (higher is better)")
ax.set_title("WTI crude oil, 1-day probabilistic forecast, TEST 246 origins", loc="left")
ax.set_xlim(-14, 4)
save(fig, "oil_skill")

# ---------------------------------------------------------------- 7. SSM 5-seed world-model test
ss = pd.read_csv(os.path.join(AUD, "ssm_wm_test.csv"))
FACTS["ssm"] = ss.to_dict(orient="list")
fig, ax = plt.subplots(figsize=(5.0, 2.8))
for r in ss.itertuples():
    ax.plot([r.seed, r.seed], [r.none, r.direct], color=C["grey"], lw=1, zorder=1)
ax.scatter(ss.seed, ss.none, color=C["alma"], label="Rollout (world model)", zorder=3, s=30)
ax.scatter(ss.seed, ss.direct, color=C["new"], marker="s", label="Direct twin (no rollout)", zorder=3, s=26)
ax.axhline(0.559, color=C["fm"], ls="--", lw=1)
ax.text(4.45, 0.5615, "Chronos-2 0.559", color=C["fm"], fontsize=7, ha="right")
ax.set_xticks(ss.seed, ["0\n(selection)", "1", "2", "3", "4"])
ax.set_xlabel("Seed")
ax.set_ylabel("Bench-VAL wql_rel (lower is better)")
ax.set_ylim(0.54, 0.71)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
ax.set_title("Research-track SSM: rollout vs matched direct twin, 30 VAL tasks", loc="left")
save(fig, "ssm_seeds")

# ---------------------------------------------------------------- 8. research track vs leaderboard models (bench VAL)
rp = open(os.path.join(REPO, "docs/almats_research_program.md")).read()
chron = float(re.search(r"Chronos-2: (0\.\d+)", rp).group(1))
tim = re.search(r"Timestral: (0\.\d+) \(209M\), (0\.\d+) \(28M\), (0\.\d+) \(5M\)", rp)
bars = [("Chronos-2 (120M)", chron, "fm"), ("Timestral-209M", float(tim.group(1)), "fm"),
        ("Timestral-28M", float(tim.group(2)), "fm"), ("Timestral-5M", float(tim.group(3)), "fm"),
        ("Alma SSM, best seed (s0)", float(ss.none.min()), "alma"), ("Alma SSM, 5-seed mean", float(ss.none.mean()), "alma"),
        ("Direct twin, 5-seed mean", float(ss.direct.mean()), "new")]
FACTS["bench_val"] = {b[0]: b[1] for b in bars}
fig, ax = plt.subplots(figsize=(5.6, 2.6))
hbar(ax, [b[0] for b in bars], [b[1] for b in bars], [C[b[2]] for b in bars], "{:.4f}",
     "Bench-VAL wql_rel vs Seasonal Naive (lower is better)", "Zero-shot research track vs leading foundation models")
ax.set_xlim(0.5, 0.75)
save(fig, "bench_val")

# ---------------------------------------------------------------- 9. VAL weight selection curves (fin, power)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
fw = re.search(r"VAL wm_fin\+har_intra_dow: (.*)", fb).group(1)
pts = [(float(a), float(b), float(c)) for a, b, c in re.findall(r"w=([\d.]+):acc([\d.]+)/qs([\d.]+)", fw)]
FACTS["fin_val_curve"] = pts
w, acc, qs = map(np.array, zip(*pts))
ax = axs[0]
ax.plot(w, qs, marker="o", color=C["alma"], ms=3)
ax.axvline(0.4, color=C["new"], ls="--", lw=1)
ax.set_xlabel("Weight on har_intra_dow")
ax.set_ylabel("VAL quantile score", color=C["alma"])
ax2 = ax.twinx()
ax2.plot(w, acc, marker="s", color=C["grey"], ms=3)
ax2.set_ylabel("VAL accuracy P1", color=C["grey"])
ax2.spines["right"].set_visible(True)
ax.set_title("Crypto: blend weight chosen on VAL (w = 0.4)", loc="left")
pv = [(float(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in
      re.findall(r"VAL w=([\d.]+) P1 ([\d.]+) P2 ([\d.]+) QS ([\d.]+) MAE ([\d.]+)", pb)]
FACTS["power_val_curve"] = pv
w, p1, p2, q, mae = map(np.array, zip(*pv))
ax = axs[1]
ax.plot(w, q, marker="o", color=C["alma"], ms=3)
ax.axvline(0.5, color=C["new"], ls="--", lw=1)
ax.set_xlabel("Weight on LEAR")
ax.set_ylabel("VAL quantile score", color=C["alma"])
ax2 = ax.twinx()
ax2.plot(w, p2, marker="s", color=C["grey"], ms=3)
ax2.set_ylabel("VAL hourly accuracy P2", color=C["grey"])
ax2.spines["right"].set_visible(True)
ax.set_title("Power: blend weight chosen on VAL (w = 0.5)", loc="left")
fig.tight_layout()
save(fig, "val_weights")

# ---------------------------------------------------------------- 10. world-model test across projects
wm = [  # (project, metric, rollout, direct, lower_is_better, source)
    ("Oil, QS h=1", float(oc[oc.model == "wm_sv"]["mean QS"].iloc[0]), float(oc[oc.model == "wm_sv_direct"]["mean QS"].iloc[0]), True),
    ("Gas, MAE thu", *[float(re.search(rf"thu\s+{n} \(frozen\)\s+([\d.]+)", open(os.path.join(AUD, "validation/gas_final.out")).read()).group(1))
                       for n in ("wm_gas", "wm_gas_direct")], True),   # 3-decimal values from the audit re-run
    ("Power, QS", float(pw[pw.model == "wm_power"].QS.iloc[0]), float(pw[pw.model == "wm_power_direct"].QS.iloc[0]), True),
    ("Power, P2 acc", float(pd2[pd2.model == "wm_power"].P2.iloc[0]), float(pd2[pd2.model == "wm_power_direct"].P2.iloc[0]), False),
    ("Crypto, QS", float(fs[fs.model == "wm_fin"].QSv.iloc[0]), float(fs[fs.model == "wm_direct"].QSv.iloc[0]), True),
    ("Crypto, P1 acc", float(fs[fs.model == "wm_fin"].P1.iloc[0]), float(fs[fs.model == "wm_direct"].P1.iloc[0]), False),
    ("SSM, wql_rel (5-seed mean)", float(ss.none.mean()), float(ss.direct.mean()), True),
]
FACTS["wm_test"] = {a: {"rollout": b, "direct": c} for a, b, c, _ in wm}
fig, ax = plt.subplots(figsize=(5.6, 2.7))
labels, gains = [], []
for name, r, d, low in wm:
    g = (d - r) / d * 100 if low else (r - d) / d * 100       # > 0: rollout better
    labels.append(name)
    gains.append(g)
FACTS["wm_test_gain_pct"] = dict(zip(labels, gains))
y = np.arange(len(labels))
ax.barh(y, gains, color=[C["alma"] if g > 0 else C["new"] for g in gains], height=0.6)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.6)
for yi, g in zip(y, gains):
    ax.text(g, yi, f" {g:+.2f}%" if g >= 0 else f"{g:+.2f}% ", va="center", ha="left" if g >= 0 else "right", fontsize=7)
ax.set_xlabel("Relative advantage of rollout over its direct twin, % (> 0: world model better)")
ax.set_title("World-model test on TEST: no significant advantage for the rollout", loc="left")
ax.set_xlim(-3.5, 3.8)
save(fig, "wm_test")

# ---------------------------------------------------------------- 11. gas adaptive intervals before/after
g2rows = re.findall(r"(thu|fri)\s+candidate (\(frozen intervals\)|\+ adaptive intervals)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", g2)
FACTS["gas_intervals"] = [list(r) for r in g2rows]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, j, lab, ref in ((axs[0], 3, "Quantile score (Bcf)", None), (axs[1], 4, "80% coverage", 0.8), (axs[2], 6, "80% interval width (Bcf)", None)):
    for k, kind in enumerate(("thu", "fri")):
        vv = [float(r[j]) for r in g2rows if r[0] == kind]
        ax.bar([k - 0.18, k + 0.18], vv, width=0.34, color=[C["base"], C["new"]])
        for x, v in zip([k - 0.18, k + 0.18], vv):
            ax.text(x, v, f"{v:.3f}" if j == 4 else f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    if ref:
        ax.axhline(ref, color="k", ls=":", lw=0.8)
    ax.set_xticks([0, 1], ["Thursday", "Friday"])
    ax.set_title(lab, loc="left")
axs[1].set_ylim(0.7, 1.0)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["base"]), plt.Rectangle((0, 0), 1, 1, color=C["new"])],
           labels=["fixed intervals (14 seeds + 0.2 OLS)", "adaptive intervals (wm_gas_ens)"], frameon=False, fontsize=7,
           loc="lower center", ncol=2)
save(fig, "gas_intervals")

json.dump(FACTS, open(os.path.join(HERE, "facts.json"), "w"), indent=1, default=float)
print("figures:", sorted(os.listdir(FIG)))
