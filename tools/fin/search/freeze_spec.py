"""Write runs/fin/search/freeze_spec.json from the mechanical finalist decisions (input to make_frozen.py)."""
import json

O = "runs/fin/search/"
wm, cl, rl = (json.load(open(O + f"{p}_final.json")) for p in ("wm", "cl", "rl"))
lw, lc, lr = (json.load(open(O + f"loop_{p}.json")) for p in ("wm", "cl", "rl"))


def f(st, k):
    return next(h["cfg"] for h in st["hist"] if h["key"] == k)


cmp = json.load(open(O + "cmp_f1.json"))
spec = {
    "models": ["wm_fin", "wm_direct", "wm_fin_cl", "wm_fin_cl:frozen", "wm_agent", "ppo", "dqn", "wm_voltgt",
               "wm_b_moe4"],
    "claim_models": ["wm_fin", "wm_fin_cl", "wm_agent"],
    "ckpts": {"wm_fin": "runs/fin/search/ckpt/frozen_wm_fin.pt", "wm_direct": "runs/fin/search/ckpt/frozen_wm_direct.pt",
              "wm_agent": "runs/fin/search/ckpt/frozen_wm_agent.pt", "ppo": "runs/fin/search/ckpt/frozen_ppo.pt",
              "dqn": "runs/fin/search/ckpt/frozen_dqn.pt", "wm_b_moe4": "runs/fin/search_b/ckpt/wm_b_moe4_refit.pt"},
    "online_rule": {"steps": 1, "lr": 1e-4, "l2_init": 1.0, "replay_days": 60, "params": "all", "meta": None},
    "frozen": {
        "protocol": "docs/fin_search.md section 0 + Am0, Am1, Am2 (+ correction); prereg docs/fin_prereg.md with lead "
                    "amendments through A5 (0525126)",
        "timestamps": "UTC. FROZEN.json written 2026-11-01 ~07:45 UTC; first freeze commit bcdbf2e at 07:46:11 UTC; this correction commit is the freeze of record (commit time authoritative)",
        "recipes": {
            "wm_fin": {"arch": "RSSM (GRU h + Gaussian z), 4 six-hour prior sub-steps per day, weekday-of-D head input; "
                               "day heads read latents rolled through day D",
                       "cfg": f(lw, wm["key"]), "loop_key": wm["key"], "seeds": [0, 1, 2], "fit_end": "2025-06-30",
                       "train_coins": 16},
            "wm_direct": {"twin_of": "wm_fin", "difference": "day heads read the filtered state at D-1 (no rollout); "
                                                             "identical modules, params, losses, data, steps, seeds"},
            "wm_fin_cl": {"base": "wm_fin weights", "online_rule": f(lc, cl["key"]),
                          "twin": "wm_fin_cl:frozen (identical weights, observe disabled)",
                          "self_improving_meta_rule": "lr_adapt tested on E-folds (S_C -0.0003), not kept"},
            "wm_agent": {"algo": "Dreamer-style actor-critic in imagination of frozen wm_fin (dynamics backprop of "
                                 "lambda-returns, EMA critic)", "cfg": f(lr, rl["key"]), "loop_key": rl["key"],
                         "seeds": [0, 1, 2], "note": "one policy per wm_fin member; position = mean over members"},
            "ppo": {"twin_of": "wm_agent", "difference": "PPO on REAL DEV transitions (fill-to-fill returns, realised "
                                                         "funding); same obs, nets, horizon, batch, steps, seeds, lam"},
            "dqn": {"twin_of": "wm_agent", "difference": "DQN, 5 sizes, all-action targets, REAL transitions; same obs, "
                                                         "horizon, batch, steps, seeds; Q hidden 128 (params within 5%)"},
            "wm_voltgt": {"rule": "sign(30d return) x min(1, sigma*/sigma_wm_fin); non-RL control"},
            "wm_b_moe4": {"owner": "architect B", "commit": "aec7e7e",
                          "status": "non-claim; the pooled P1 finalist is A's wm_fin by best F1 (0.7194 vs 0.7152)"}},
        "twin_matching": {"wm_fin_vs_wm_direct": "identical module list and params per member, same seeds/steps/data",
                          "agents": "dreamer actor+critic vs ppo actor+critic vs dqn Q(hidden 128): params within 5%"},
        "selection": {
            "loop_folds": "E1 fit<=2022-12-31/test 2023H1; E2 fit<=2023-06-30/test 2023H2; E3 fit<=2023-12-31/test 2024H1",
            "holdout": "F1 fit<=2024-06-30 / test 2024-07-01..2025-06-30 (one look per finalist)",
            "wm_final": wm, "cl_final": cl, "rl_final": rl,
            "P1_gate": "beat max registered dow-aware baseline on E1-E3 (lgbm_dow EM 0.6930; per fold .666/.696/.717): "
                       "wm_fin recipe EM 0.7077 (.691/.714/.718) passes; wm_b_moe4 EM 0.6934 passes by 0.0004 (tie). "
                       "moe4's and lgbm_dow's EMs were known when the gate was committed (gate not independent of them).",
            "F1_post_selection_note": "F1 numbers of chosen models are the max over <=6 F1 looks (winner's curse, "
                                      "RT-11); F1 pillar differences are descriptive only"},
        "dev_F1_3seed_pillar_comparisons_descriptive": cmp,
        "pillar_tests_preregistered": {
            "rollout_vs_direct": "wm_fin vs wm_direct: P1 (DM daily) and P3 QS, all 24 coins, Holm over 2 endpoints",
            "continual_vs_frozen": "wm_fin_cl vs wm_fin_cl:frozen: P1 and P3, all 24 coins",
            "wm_vs_model_free": "wm_agent vs ppo, dqn: P2 net Sharpe, held-out (primary) and all 24; Holm over "
                                "{held-out, all} x {ppo, dqn}",
            "portability": "wm_fin (observe disabled) on held-out coins vs lgbm_asset and har_roll; ETFs descriptive",
            "beats_baselines": "wm_fin on P1 vs every non-contaminated section-5 model incl. the A5 dow-aware baselines"},
        "val_go_no_go": "Written before VAL. VAL is run once, confirmation/debug only. NO-GO for TEST only if a frozen "
                        "model crashes, returns non-finite outputs for >5% of scored coin-days, or the leak test fails; "
                        "otherwise GO. No model, rule or threshold changes after VAL.",
        "val_numbers_not_consulted": "Architects A and B did not read the lead's VAL numbers (docs/fin_results.md, "
                                     "runs/fin/val/) before this freeze.",
        "leak_tests": {
            "frozen": "runs/fin/search/test_leak.txt: every invariance check OK (future, fill-bar, determinism, "
                      "continual walk); 2 sensitivity checks FAIL for dqn on prev_hour (discrete argmax unchanged by a "
                      "one-hour perturbation; prev_day changes it). Not a leak.",
            "families": "runs/fin/search/test_leak_{rssm,cl,meta,agents,rssm_dow}.txt",
            "B": "runs/fin/search_b/test_leak_b_final.txt ALL OK"},
        "disclosures": [
            "4 round-1 rows were scored on F1 before Am1 (counted in K)",
            "DST: the WM loop's clock stop overran by 5 min; the aborted proposal is logged",
            "dow knob added post hoc (Am2) after the red team found the weekday effect; extension rows in K",
            "the cl loop ran on the pre-dow WM incumbent, the rl loop on the dow incumbent",
            "RL loop stopped early: one-knob neighbourhood exhausted (13 proposals)",
            "E-fold P1 includes held-out coins: DEV held-out numbers are not zero-shot-clean",
            "RT-8: PPO/DQN rewards fill-to-fill with realised funding; Dreamer rewards are WM-imagined day returns",
            "B: 6 NaN-diverged rows logged as P1 0 count as crashes; har_intra_dow specified after the red team saw F1; "
            "B saw A's round-1 F1 rows; cache label loop3 -> loop4",
            "Continual: no gain on E-folds (all S_C <= 0.00001); frozen rule = starting rule; F1 S_C +0.0021 (one look)",
            "Self-improving meta rule (online LR adaptation from past errors) tested, not selected"]}}
json.dump(spec, open(O + "freeze_spec.json", "w"), indent=1, default=str)
print("ok")
