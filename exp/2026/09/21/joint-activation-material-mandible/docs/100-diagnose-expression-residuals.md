# Expression residual and implicit-derivative diagnosis

All three diagnostic checks passed at the unchanged 5% threshold: corrected finite-difference agreement, corrected two-scale plateau, and independent force/parameter derivative agreement. The failed raw finite-difference validation from run 99 remains unchanged. These results strongly support residual equilibration error as the source of its disagreement, rather than an error in the implicit derivative chain near the tested state.

## Command and provenance

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME='Diagnose expression primal residual and implicit derivatives' CHERRIES_TAGS='expression,validation,adjoint,residual,rigid-eyes' uv run python src/100-diagnose-expression-residuals.py > logs/100-diagnose-expression-residuals-terminal.log 2>&1
```

[Comet run](https://www.comet.com/liblaf/apple/e2484ac657ea421c9b97da05fc3f4582). `ProfileJoint` archived source and Git state without committing. The diagnostic verified every bound source hash from run 99 and every saved equilibrium hash before use. It used the original PNCG runtime's physical model and exact Hessian-product adjoint solver; it performed no additional primal equilibration and changed no production module. All nine adjoint solves passed their residual checks. The process completed with exit 0 after Cherries/Comet shutdown.

## Method and sign

For each saved configuration, the free equilibrium residual is `r = ∂E/∂u_free`. The script solves `H p = -∂J/∂u_free`. The first-order estimate of the equilibrated scalar is `J_corrected = J + pᵀr`, equivalently `J - λᵀr` with `H λ = ∂J/∂u_free`. This sign follows from the linearized equilibrium correction `Δu = -H⁻¹r`.

Each perturbed state receives its own exact-HVP adjoint. The central differences use the corrected scalars at both original physical perturbation scales. Separately, the mechanical check holds free coordinates fixed, perturbs activation or prescribed jaw data, and finite-differences the force and direct objective. It contracts the force derivative with the base negative adjoint. This tests the parameter chain without requiring two perturbed equilibria.

## Corrected equilibrium finite differences

| Parameter | Central step | Corrected FD | Relative error |
| --- | ---: | ---: | ---: |
| activation | 0.0001 | -7617.46807274 | 0.014323494% |
| activation | 3e-05 | -7616.51032143 | 0.0017506212% |
| pose | 1e-05 | 97916.5302304 | 0.0076367351% |
| pose | 3e-06 | 97909.2198045 | 0.00017077063% |

The activation plateau error is 0.012573093%; the jaw plateau error is 0.0074659772%. Both are well below 5%.

## Independent force/parameter finite differences

| Parameter | Step | Force/parameter FD estimate | Relative error |
| --- | ---: | ---: | ---: |
| activation | 1e-06 | -7616.37698846 | 4.3051875e-10 |
| activation | 1e-05 | -7616.37698847 | 4.3065285e-10 |
| pose | 1e-07 | 97909.0393583 | 1.3528937e-07 |
| pose | 1e-06 | 97920.5568076 | 0.00011748507 |

The pose parameter derivative at `1e-6` has a larger truncation error than at `1e-7`, while still passing the criterion. The mechanical activation derivative agrees to approximately `4.3e-10` relative error.

## Measured residual corrections

| State | Raw scalar objective | Signed residual correction |
| --- | ---: | ---: |
| base | 172.250862844 | 0.00577776362984 |
| activation-0-plus | 171.526634369 | -0.025389477706 |
| activation-0-minus | 172.992923756 | 0.0318147501934 |
| activation-1-plus | 172.048968302 | -0.0202546584176 |
| activation-1-minus | 172.457063147 | 0.0286411158442 |
| pose-0-plus | 173.17647733 | 0.0593924990912 |
| pose-0-minus | 171.336628123 | -0.0590888984798 |
| pose-1-plus | 172.493628406 | 0.0567394499124 |
| pose-1-minus | 172.007668058 | -0.0447555212439 |

## Interpretation and limits

Run 99's raw activation errors were 3.74% and 10.70%; its raw jaw errors were 6.04% and 17.28%. After signed residual correction, the largest corresponding error is 0.0144%. The corrected differences also agree across perturbation scales. Together with the independent mechanical chain check, this local evidence supports using the implicit equilibrium derivative with explicit inexact-primal error monitoring.

The correction is a first-order estimate, not a rigorous objective error bound. It does not convert the raw PNCG finite-tolerance solver into an exactly differentiated algorithm. The diagnostic uses a linear displacement objective near one base state; it does not establish correctness for every future expression, large deformation, contact transition, or outer objective. A guarded fit should preserve the failed raw validation evidence and evaluate actual raw and residual-corrected decreases, including a meaningful margin against the estimated equilibration error. This diagnostic does not silently replace production acceptance policy.

## Outputs

`data/expression-residual-diagnostic-001/summary.json` contains the three independent pass flags, all raw scalars and signed corrections, both FD comparisons and plateau checks, exact source hashes, and the hash of run 99's failed summary. Its `eligible_as_production_validation: false` means it is an explicitly scoped diagnostic, not a standalone replacement for raw validation. Source snapshots and execution metadata are retained under `sources/` and `provenance.json`; terminal and Cherries logs are under `logs/`.

## Cherries/Comet summary

```text
00:02:55.685 INF comet_ml.summary:generate_summary:154 ---------------------------------------------------------------------------------------
00:02:55.686 INF comet_ml.summary:generate_summary:155 Comet.ml Experiment Summary
00:02:55.687 INF comet_ml.summary:generate_summary:156 ---------------------------------------------------------------------------------------
00:02:55.688 INF comet_ml.summary:generate_summary:160   Data:
00:02:55.688 INF comet_ml.summary:generate_summary:170     display_summary_level : 1
00:02:55.689 INF comet_ml.summary:generate_summary:170     name                  : Diagnose expression primal residual and implicit derivatives
00:02:55.690 INF comet_ml.summary:generate_summary:170     url                   : https://www.comet.com/liblaf/apple/e2484ac657ea421c9b97da05fc3f4582
00:02:55.690 INF comet_ml.summary:generate_summary:160   Others:
00:02:55.691 INF comet_ml.summary:generate_summary:170     Name                : Diagnose expression primal residual and implicit derivatives
00:02:55.691 INF comet_ml.summary:generate_summary:170     cherries/cmd        : .venv/bin/python src/100-diagnose-expression-residuals.py
00:02:55.692 INF comet_ml.summary:generate_summary:170     cherries/comet/url  : https://www.comet.com/liblaf/apple/e2484ac657ea421c9b97da05fc3f4582
00:02:55.692 INF comet_ml.summary:generate_summary:170     cherries/end_time   : 2026-09-21 21:14:32.642267+08:00
00:02:55.693 INF comet_ml.summary:generate_summary:170     cherries/entrypoint : exp/2026/09/21/joint-activation-material-mandible/src/100-diagnose-expression-residuals.py
00:02:55.693 INF comet_ml.summary:generate_summary:170     cherries/exp_dir    : exp/2026/09/21/joint-activation-material-mandible
00:02:55.694 INF comet_ml.summary:generate_summary:170     cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
00:02:55.694 INF comet_ml.summary:generate_summary:170     cherries/start_time : 2026-09-21 21:11:46.604573+08:00
00:02:55.695 INF comet_ml.summary:generate_summary:160   Parameters:
00:02:55.695 INF comet_ml.summary:generate_summary:170     output_dir     : exp/2026/09/21/joint-activation-material-mandible/data/expression-residual-diagnostic-001
00:02:55.696 INF comet_ml.summary:generate_summary:170     validation_dir : exp/2026/09/21/joint-activation-material-mandible/data/expression-scale-gradient-validation-001
00:02:55.696 INF comet_ml.summary:generate_summary:160   Uploads:
00:02:55.696 INF comet_ml.summary:generate_summary:170     environment details : 1
00:02:55.697 INF comet_ml.summary:generate_summary:170     filename            : 1
00:02:55.697 INF comet_ml.summary:generate_summary:170     installed packages  : 1
00:02:55.698 INF comet_ml.summary:generate_summary:170     source_code         : 2 (12.06 KB)
00:02:55.698 INF comet_ml.summary:generate_summary:172
```
