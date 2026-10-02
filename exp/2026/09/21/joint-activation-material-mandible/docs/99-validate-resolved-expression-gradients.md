# Scale-resolved expression gradient validation

The raw finite-difference validation failed its unchanged 5% agreement and two-scale plateau gates. All nine primal solves passed the established force and contact gates. This separates a primal acceptance result from a failed gradient check; it does not establish gradient agreement for the finite-tolerance solver.

## Command and provenance

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME='Validate scale-resolved expression gradients' CHERRIES_TAGS='expression,validation,pncg,finite-difference,rigid-eyes' uv run python src/99-validate-resolved-expression-gradients.py > logs/99-validate-resolved-expression-gradients-terminal.log 2>&1
```

[Comet run](https://www.comet.com/liblaf/apple/55b634d5b3344833ba7736f0b312a755). `ProfileJoint` archived the source and Git state without committing. The run used the original `joint_expression_equilibrium` PNCG runtime, frozen-neutral-004, rigid-eyes-001, and the hash-bound eye-neutral-forward-002 source checkpoint. It warm-started the base from the separately hashed step-1000 checkpoint of strict run 98, after confirming its fixed coordinates matched the base jaw. All perturbations warm-started from the accepted base with `seed_pose=base_pose`.

## Results

The probe activated all 288,235 muscle tetrahedra with `2e-6 MPa * I`. The scalar objective was `1e6 * mean(u[observation_ids] dot [0.31, -0.27, 0.19])`. The activation direction was identity at every active tet. The normalized pose direction was `[0.2, -0.3, 0.1, 0.4, -0.2, 0.5]`.

Signed central activation probes extend the physical stress parameter beyond its PSD optimization constraint solely for numerical differentiation; these are not accepted activation estimates.

| Parameter | Central step | Implicit derivative | Finite difference | Relative error |
| --- | ---: | ---: | ---: | ---: |
| activation | 0.0001 | -7616.37699 | -7331.44693 | 3.74102% |
| activation | 3e-05 | -7616.37699 | -6801.58075 | 10.69795% |
| pose | 1e-05 | 97909.0526 | 91992.4604 | 6.04295% |
| pose | 3e-06 | 97909.0526 | 80993.3913 | 17.27691% |

Activation FD plateau error: 7.22731%. Pose FD plateau error: 11.95649%. The adjoint relative residual was 8.96596012857e-08, passing its `1e-7` tolerance.

| State | PNCG steps | Accepted force |
| --- | ---: | ---: |
| base | 1 | 1.26288329443e-10 |
| activation-0-plus | 1495 | 1.39376749505e-10 |
| activation-0-minus | 1330 | 1.51033120945e-10 |
| activation-1-plus | 787 | 1.41227345255e-10 |
| activation-1-minus | 684 | 1.50256518309e-10 |
| pose-0-plus | 1547 | 1.51514197435e-10 |
| pose-0-minus | 1809 | 1.50259624563e-10 |
| pose-1-plus | 600 | 1.4651635491e-10 |
| pose-1-minus | 695 | 1.41885956898e-10 |

Every accepted force is at or below `1.5192003475221146e-10`. All terminal contact receipts reported no intersections, numerically valid contact, and minimum active distance above the 10 nm numerical CCD buffer. The final assertion correctly failed, and the process exited with status 1 after Cherries/Comet shutdown.

## Evidence and interpretation

`data/expression-scale-gradient-validation-001/summary.json` retains `success: false`, exact source hashes, original and refined seed hashes, the adjoint receipt, every solve receipt, and both FD scales. Each accepted equilibrium has an NPZ checkpoint and hash. Full source snapshots are under `sources/`; terminal and Cherries logs are retained under `logs/`.

The smaller perturbations produced larger disagreement. This is consistent with remaining primal equilibration error contaminating finite differences, but raw FD results alone do not prove that explanation. Follow-up script 100 evaluates signed adjoint residual corrections and independent force/parameter derivatives at the saved states. It preserves this failed raw validation rather than changing its tolerance or replacing its evidence.

## Cherries/Comet summary

```text
00:19:31.174 INF comet_ml.summary:generate_summary:154 ---------------------------------------------------------------------------------------
00:19:31.175 INF comet_ml.summary:generate_summary:155 Comet.ml Experiment Summary
00:19:31.175 INF comet_ml.summary:generate_summary:156 ---------------------------------------------------------------------------------------
00:19:31.176 INF comet_ml.summary:generate_summary:160   Data:
00:19:31.176 INF comet_ml.summary:generate_summary:170     display_summary_level : 1
00:19:31.177 INF comet_ml.summary:generate_summary:170     name                  : Validate scale-resolved expression gradients
00:19:31.177 INF comet_ml.summary:generate_summary:170     url                   : https://www.comet.com/liblaf/apple/55b634d5b3344833ba7736f0b312a755
00:19:31.177 INF comet_ml.summary:generate_summary:160   Others:
00:19:31.178 INF comet_ml.summary:generate_summary:170     Name                : Validate scale-resolved expression gradients
00:19:31.178 INF comet_ml.summary:generate_summary:170     cherries/cmd        : .venv/bin/python src/99-validate-resolved-expression-gradients.py
00:19:31.179 INF comet_ml.summary:generate_summary:170     cherries/comet/url  : https://www.comet.com/liblaf/apple/55b634d5b3344833ba7736f0b312a755
00:19:31.179 INF comet_ml.summary:generate_summary:170     cherries/end_time   : 2026-09-21 21:10:46.099525+08:00
00:19:31.180 INF comet_ml.summary:generate_summary:170     cherries/entrypoint : exp/2026/09/21/joint-activation-material-mandible/src/99-validate-resolved-expression-gradients.py
00:19:31.180 INF comet_ml.summary:generate_summary:170     cherries/exception  : AssertionError: {'schema': 'scale-resolved-expression-gradient-validation-v1', 'success': False, 'implementation_sha256': {'exp/2026/09/21/joint-activation-material-mandible/src/68-run-simple-skin-forward.py': 'f276d06e430a800779f7e3745e2b61210346304187c5d6d4b15f55289232d083', 'exp/2026/09/21/joint-activation-material-mandible/src/99-validate-resolved-expression-gradients.py': '2cc493ddbb6ebde89f1271a0949ce96f2d5869d59ab80430e78515949b6a59d1', 'exp/2026/09/21/joint-activation-material-mandible/src/joint_common.py': 'bd503a5324568dd98f20cfdf16c382b483088f09b269606ee6ef4147e34925b1', 'exp/2026/09/21/joint-activation-material-mandible/src/joint_contact.py': '8fa1601045c02cc3495f44178646a1575fbe364a05066175f7356cec7d8ac8ef', 'exp/2026/09/21/joint-activation-material-mandible/src/joint_data.py': '915dbd0e19 [truncated]
00:19:31.181 INF comet_ml.summary:generate_summary:170     cherries/exp_dir    : exp/2026/09/21/joint-activation-material-mandible
00:19:31.182 INF comet_ml.summary:generate_summary:170     cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
00:19:31.182 INF comet_ml.summary:generate_summary:170     cherries/start_time : 2026-09-21 20:51:23.949135+08:00
00:19:31.183 INF comet_ml.summary:generate_summary:160   Parameters:
00:19:31.183 INF comet_ml.summary:generate_summary:170     output_dir : exp/2026/09/21/joint-activation-material-mandible/data/expression-scale-gradient-validation-001
00:19:31.183 INF comet_ml.summary:generate_summary:160   Uploads:
00:19:31.184 INF comet_ml.summary:generate_summary:170     environment details : 1
00:19:31.184 INF comet_ml.summary:generate_summary:170     filename            : 1
00:19:31.185 INF comet_ml.summary:generate_summary:170     installed packages  : 1
00:19:31.185 INF comet_ml.summary:generate_summary:170     source_code         : 2 (12.40 KB)
00:19:31.186 INF comet_ml.summary:generate_summary:172
```
