# V2 — Tamper and determinism tests

| test | project | expected | result | detail |
|---|---|---|---|---|
| T1c | oil | loads | **as expected** | control (untouched copy): LOADED |
| T1 | oil | refused | **as expected** | 1-byte flip in runs/commodity/search/ckpt/f_sv_none_s0.pt -> AssertionError: checkpoint /home/user/Alma-Models/runs/validation/tamper/tree/runs/commodity/search/ckpt/f_sv_none_s0.pt: sha256 29f5f517f7d9c7b5eb453ba058127b |
| T2 | oil | refused | **as expected** | verify_checkpoints after the flip -> refused: loaded checkpoint copy differs from its pin {'file': 'runs/commodity/search/ckpt/f_sv_none_s0.pt', 'sha256': '29f5f517f7d9c7b5eb453ba058127b760115b168850637f8fa7cdbe46419a869 |
| T1c | power | loads | **as expected** | control (untouched copy): LOADED |
| T1 | power | refused | **as expected** | 1-byte flip in runs/power/search/ckpt/frz_phys_none_s0.pt -> AssertionError: sha256 mismatch for runs/power/search/ckpt/frz_phys_none_s0.pt: 9b3d4d16da1ab365d43b126a0618344e9acf8c32373e4ff8e9bde1f7db1c10bf |
| T2 | power | refused | **as expected** | verify_checkpoints after the flip -> refused: loaded checkpoint copy differs from its pin {'file': 'runs/power/search/ckpt/frz_phys_none_s0.pt', 'sha256': '9b3d4d16da1ab365d43b126a0618344e9acf8c32373e4ff8e9bde1f7db1c10bf |
| T1c | gas | loads | **as expected** | control (untouched copy): LOADED |
| T1 | gas | refused | **as expected** | 1-byte flip in winning_models/gas_wm_gas_ens/checkpoints/x_wm_s3456.pt -> AssertionError: sha256 mismatch for winning_models/gas_wm_gas_ens/checkpoints/x_wm_s3456.pt: 472bcd84b947de60f75308c0b3305974e1b91c4588e01c2a7c7d8 |
| T2 | gas | refused | **as expected** | verify_checkpoints after the flip -> refused: checkpoint pin mismatch {'file': 'winning_models/gas_wm_gas_ens/checkpoints/x_wm_s3456.pt', 'sha256': '472bcd84b947de60f75308c0b3305974e1b91c4588e01c2a7c7d8fc2d658b24e', 'sha |
| T1c | fin | loads | **as expected** | control (untouched copy): LOADED |
| T1 | fin | refused | **as expected** | 1-byte flip in runs/fin/search/ckpt/frozen_wm_fin.pt -> RuntimeError: wm_fin: checkpoint runs/fin/search/ckpt/frozen_wm_fin.pt sha256 2664ab1ebfc1414132a0b3dcc9a47a814eeb879f095969818b3621f5bc45a7a5 != frozen 5b90558f4c2 |
| T2 | fin | refused | **as expected** | verify_checkpoints after the flip -> refused: loaded checkpoint copy differs from its pin {'file': 'runs/fin/search/ckpt/frozen_wm_fin.pt', 'sha256': '2664ab1ebfc1414132a0b3dcc9a47a814eeb879f095969818b3621f5bc45a7a5', 'l |
| T3 | power | refused | **as expected** | edit tools/power/wm_power.py -> refused: frozen code changed: tools/power/wm_power.py |
| T3 | gas | refused | **as expected** | edit tools/gas/wm_gas.py -> refused: frozen code changed: tools/gas/wm_gas.py |
| T3 | fin | refused | **as expected** | edit tools/fin/wm_fin.py -> refused: frozen code changed: tools/fin/wm_fin.py |
| T4 | fin | refused | **as expected** | pinned file whose bytes do not match -> exit 1: runs/validation/tamper/manifest_with_wrong_pin.json: {'diff-cached': 1}; failures: 1 |
| T4c | fin | accepted | **as expected** | control, correct pin -> exit 0 |
| T5 | all | refused | **as expected** | optimizer step under no_training() -> common.TrainingAttempted: training is disabled in deployment (tools/deploy/common.no_training) |
| T5 | all | refused | **as expected** | backward() under no_training() -> common.TrainingAttempted: training is disabled in deployment (tools/deploy/common.no_training) |
| D1 | oil | bit-identical | **as expected** | two runs, 1 thread: sha 6aa685ac05483fa3 vs 6aa685ac05483fa3 |
| D2 | oil | report | **as expected** | 1 vs 4 threads: bit-identical |
| D1 | power | bit-identical | **as expected** | two runs, 1 thread: sha d72f0ed7a2341f3e vs d72f0ed7a2341f3e |
| D2 | power | report | **as expected** | 1 vs 4 threads: bit-identical |
| D1 | gas | bit-identical | **as expected** | two runs, 1 thread: sha df7be700ca29937f vs df7be700ca29937f |
| D2 | gas | report | **as expected** | 1 vs 4 threads: bit-identical |
| D1 | fin | bit-identical | **as expected** | two runs, 1 thread: sha bd2268ce92173419 vs bd2268ce92173419 |
| D2 | fin | report | **as expected** | 1 vs 4 threads: bit-identical |

**27/27 as expected.**
