# Target-normal matching in the 2D activation grid

This experiment compares positional L2, L2 plus reference-space gradient matching,
and L2 plus target-normal matching in the historical four-activation-by-two-target
study. The four rows are unrestricted symmetric (3 DoF), PSD contraction-only
(3 DoF), learned-axis rank-one contraction (2 DoF), and fixed-x contraction
(1 DoF), independently parameterized on 400 muscle triangles. Target heights are
0.05 and 0.20; these are model length units, not millimeters.

## Controlled comparison

The [frozen protocol](00-protocol.md) retains the corrected physical-J energy,
100 by 10 mesh, materials and fixed sides/bottom. Every branch starts from its
cell's exact saved L2 update 200, resets Adam moments, and runs at most 600 more
updates with learning rate 0.003 times 0.995^step. This is a warm continuation,
not a new neutral-start comparison. The historical large-target unrestricted L2
trajectory failed after update 262; its unsafe late state was not used.

There are five branches per cell: L2 alone; L2 plus gradient matching at beta
0.05 and 0.25; and L2 plus target-normal matching at the same two weights. All
activation smoothness weights are zero. For either added term S,

```text
objective = L2 + beta * (L2_neutral / S_neutral) * S.
```

The normal term is half the weighted squared difference of corresponding unit
tangents (equivalently oriented unit normals) of the deformed and target top
segments. Weights are fixed reference segment lengths divided by reference span.
It matches the target's directions, rather than penalizing differences between
neighboring normals. Its derivative includes edge-length normalization. The
reference normalization does not guarantee a locally weak term at the warm start
or equal initial updates between objectives.

The primary comparison uses each cell's latest saved update common to all five
branches, including short failed trajectories. An added-loss candidate qualifies
only if its RMS position error is within 5% of same-step continued L2, its target
projection retains at least 95% of that control, and its accepted trajectory is
inversion-free through that step. Among qualifying candidates of one loss family,
choose the beta with lowest normal loss. These are predeclared exploratory
criteria, not universal tolerances. Endpoint metrics and failures are reported
separately from the matched-budget comparison.

## Commands and provenance

Working directory:
`exp/2026/09/21/normal-matching-profile`.
All numerical commands set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
env DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CHERRIES_NAME='2D normal matching: preflight validation' CHERRIES_TAGS='2d,normal-matching,validation' .venv/bin/python src/05-verify.py
CHERRIES_NAME='2D normal refinement smoke' CHERRIES_TAGS='2d,normal-matching,smoke' DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/10-run.py --output 09-smoke --heights 0.20 --modes unconstrained --max-updates 5
CHERRIES_NAME='2D target normal versus gradient refinement 4x2' CHERRIES_TAGS='2d,normal-matching,gradient-matching,4x2,warm-continuation' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/10-run.py
```

The main run's Comet record is
[598a9a14eb81407c8eeaeaaf7e4627c1](https://www.comet.com/liblaf/apple/598a9a14eb81407c8eeaeaaf7e4627c1).
The run archives resolved numerical source copies/hashes, gate results, starting
control/displacement states, baseline input hashes, optimizer settings and
neutral normalizers under `data/10-comparison`. Histories contain every tenth
update plus the last accepted state; scalar traces contain every accepted state.
A failed forward solve preserves its attempted control vector and last accepted
checkpoint. No fallback solver or automatic restart is used.

Git HEAD is `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the workspace contains prior
uncommitted research and physics changes. Numerical imported-source hashes match
the archived historical control study, so the experiment does not rely on HEAD
alone for reproducibility. There are no random initializations.

## Verification

Direct normal-loss derivatives and invariances, mapped displacement derivatives,
and full implicit derivatives for all four activation models passed. Maximum
relative errors were 3.42e-10 for the direct curve derivative, 7.68e-11 for the
mapped derivative, and 2.98e-7 for the full implicit derivatives (threshold 3e-5).
All eight historical step-200 states replayed with zero top-edge backtracking,
minimum physical J at least 0.1089178, and force residual at most 9.54e-11.
See `data/05-verification/checks.json` for thresholds, exact cases and hashes.

The independent endpoint verifier also passed: all 40 final accepted states and
all matched-step states satisfy the force tolerance and parameter constraints;
position, gradient and normal metrics agree with the stored traces. Actual
100 by 10 historical warm-start mixed-objective derivatives were checked in all
eight cells with two central finite-difference steps (1e-4 and 3e-4). Maximum
relative derivative error was 5.50e-6 and maximum disagreement between the two
finite differences was 7.65e-8. Source snapshots and live imported sources match.

The forward-displacement Hessian was checked for continued L2 and each selected
normal candidate at the matched step. Both unrestricted large-target states have
negative minimum eigenvalues (-0.00295919 for L2, -0.00283174 for normal beta=.05).
Those states are unstable equilibria, despite positive physical J and small force
residuals. The other tested Hessians have positive minimum eigenvalues. This is
a local discrete forward-stability check, not inverse convergence or anatomical
validation. The unrestricted large-target comparison is therefore unsuitable as
physical evidence for a better loss. Its L2 top chain also backtracks at one edge.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CHERRIES_NAME='2D normal matching: independent endpoint verification' CHERRIES_TAGS='2d,normal-matching,verification,selection' .venv/bin/python src/20-verify.py --input-dir 10-comparison --output 20-verification
```

Verifier Comet record:
[3aad0094fc10422c865005accb20dc90](https://www.comet.com/liblaf/apple/3aad0094fc10422c865005accb20dc90).

## Results

**L2 plus normal matching is useful for the large-target contraction-only and
learned-axis fits, but it is not a general cure for the fit/smoothness tradeoff.**
All small-target added-loss candidates lose too much target-directed motion under
the predeclared 95% retention threshold. None of the tested gradient mixtures
qualifies in any cell. Four normal candidates pass the fit/motion gates; one of
those is the unstable unrestricted case, and the fixed-x candidate increases the
second-derivative residual despite improving normal agreement.

Thirty-six branches completed 600 updates. Four branches stopped on a forward
line-search failure, all in the unrestricted large-target cell. Every saved
accepted state is inversion-free; failed proposals are not accepted equilibria.
The comparison uses update 600 in seven cells and update 40 in the unrestricted
large-target cell. The 40 branches took 613.65 seconds of numerical run time.

### Predeclared normal selection at the common step

Percent changes below are relative to continued L2 at the same step. Target
projection is dot(predicted displacement, target displacement) / ||target||^2;
it measures motion in the target's full displacement pattern, not peak height.
D2 is the RMS second reference-coordinate derivative of the vector displacement
error. It is a bumpiness diagnostic and **not geometric curvature**.

| Model | Target h | Common step | Selected beta | Position RMS change | Normal-angle RMS change | Target-projection change | D2 error change | Qualification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Unrestricted symmetric | 0.05 | 600 | None | — | — | — | — | Both weights lose over 5% target projection |
| PSD contraction-only | 0.05 | 600 | None | — | — | — | — | Both weights lose over 5% target projection |
| Learned axis | 0.05 | 600 | None | — | — | — | — | Both weights lose over 5% target projection |
| Fixed x | 0.05 | 600 | None | — | — | — | — | Both weights lose over 5% target projection |
| Unrestricted symmetric | 0.20 | 40 | .05 | +0.13% | -4.24% | -2.62% | -5.50% | Unstable equilibrium; short horizon |
| PSD contraction-only | 0.20 | 600 | .25 | +0.53% | -13.24% | -1.28% | -15.48% | Fit/motion gates passed |
| Learned axis | 0.20 | 600 | .25 | +0.26% | -6.60% | +3.52% | -17.90% | Fit/motion gates passed |
| Fixed x | 0.20 | 600 | .05 | +0.08% | -2.41% | -3.35% | +22.74% | D2 residual worsens |

The learned-axis h=.20 case is the clearest relevant example. At beta=.25,
position RMS changes from 0.139454 to 0.139822 (+0.26%), normal-angle RMS from
25.122 degrees to 23.464 degrees (-6.60%), and target projection from 0.110964 to
0.114869 (+3.52%). The second-derivative residual falls 17.90%. The weaker beta=.05
is also useful: it gives a smaller 5.38% normal-angle reduction, but a larger
31.30% D2 reduction, only +0.09% position RMS, and +2.93% target projection.
The predeclared selector chooses beta=.25 because it ranks qualifying candidates
by normal loss; that does not make it best on every shape diagnostic.

For small targets, normal beta=.05 retains only 63.8–71.4% of continued L2's target
projection. Its lower angle and D2 errors therefore come with substantial loss of
motion even though position RMS rises only 0.75–1.95%. Position RMS alone would
have accepted these results under a 5% budget. This demonstrates why the additional
motion criterion matters for an already underfit baseline.

These 2D fits remain far below the target's overall bulge, including continued
L2. In the h=.20 learned-axis case the target projection is only about 11%, so the
improvement is local shape agreement rather than successful global target
reconstruction. This differs from a 3D example whose L2 fit is already close.
No result here establishes that the target is unattainable; finite optimization,
activation capacity, material stiffness and boundary conditions remain separate
possibilities.

### All candidates at matched budgets

All values below come from independently checked saved histories. Lengths use
the demo's model units; all normal angles are degrees. Projection is reported as
a fraction of the complete target displacement pattern.

| Case | Variant | Step | Position RMS | Normal-angle RMS | Slope RMS | D2 residual | Target projection | Min J | Fit/motion eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h050-unconstrained | l2 | 600 | 0.033157 | 13.853 | 0.3211 | 27.985 | 0.14413 | 0.47853 | Control |
| h050-unconstrained | gradient-005 | 600 | 0.033985 | 8.507 | 0.1615 | 10.184 | 0.09383 | 0.45172 | No |
| h050-unconstrained | gradient-025 | 600 | 0.034766 | 7.861 | 0.1489 | 9.820 | 0.06611 | 0.31717 | No |
| h050-unconstrained | normal-005 | 600 | 0.033803 | 8.320 | 0.1640 | 9.942 | 0.10177 | 0.44144 | No |
| h050-unconstrained | normal-025 | 600 | 0.034473 | 8.129 | 0.1624 | 12.688 | 0.07437 | 0.38888 | No |
| h050-contraction_only | l2 | 600 | 0.034453 | 14.907 | 0.3008 | 32.743 | 0.09365 | 0.57165 | Control |
| h050-contraction_only | gradient-005 | 600 | 0.035159 | 7.731 | 0.1448 | 9.095 | 0.05544 | 0.60453 | No |
| h050-contraction_only | gradient-025 | 600 | 0.035700 | 7.216 | 0.1310 | 7.806 | 0.03198 | 0.80849 | No |
| h050-contraction_only | normal-005 | 600 | 0.034983 | 7.788 | 0.1516 | 9.024 | 0.06435 | 0.60801 | No |
| h050-contraction_only | normal-025 | 600 | 0.035546 | 7.133 | 0.1353 | 7.736 | 0.03862 | 0.58181 | No |
| h050-learned_direction | l2 | 600 | 0.034547 | 13.668 | 0.2974 | 27.410 | 0.09712 | 0.75725 | Control |
| h050-learned_direction | gradient-005 | 600 | 0.035293 | 7.607 | 0.1423 | 7.178 | 0.05196 | 0.7777 | No |
| h050-learned_direction | gradient-025 | 600 | 0.035827 | 7.108 | 0.1302 | 6.633 | 0.02849 | 0.76434 | No |
| h050-learned_direction | normal-005 | 600 | 0.035108 | 7.789 | 0.1508 | 7.816 | 0.06200 | 0.75732 | No |
| h050-learned_direction | normal-025 | 600 | 0.035697 | 7.067 | 0.1315 | 6.619 | 0.03337 | 0.71253 | No |
| h050-x_contraction | l2 | 600 | 0.035523 | 9.381 | 0.1993 | 10.535 | 0.06004 | 0.78707 | Control |
| h050-x_contraction | gradient-005 | 600 | 0.035885 | 6.970 | 0.1286 | 3.471 | 0.03454 | 0.87961 | No |
| h050-x_contraction | gradient-025 | 600 | 0.036177 | 6.720 | 0.1203 | 3.068 | 0.01745 | 0.8976 | No |
| h050-x_contraction | normal-005 | 600 | 0.035787 | 7.139 | 0.1347 | 4.016 | 0.04284 | 0.82134 | No |
| h050-x_contraction | normal-025 | 600 | 0.036018 | 6.768 | 0.1232 | 2.987 | 0.02713 | 0.89109 | No |
| h200-unconstrained | l2 | 40 | 0.133661 | 35.160 | 1.0271 | 67.346 | 0.24770 | 0.07009 | Control |
| h200-unconstrained | gradient-005 | 40 | 0.134863 | 32.910 | 0.9054 | 54.102 | 0.22915 | 0.0053777 | No |
| h200-unconstrained | gradient-025 | 40 | 0.135160 | 32.432 | 0.8944 | 53.747 | 0.21983 | 0.012174 | No |
| h200-unconstrained | normal-005 | 40 | 0.133835 | 33.670 | 0.9948 | 63.645 | 0.24121 | 0.080165 | Yes |
| h200-unconstrained | normal-025 | 40 | 0.134822 | 32.412 | 0.9420 | 59.135 | 0.22495 | 0.099825 | No |
| h200-contraction_only | l2 | 600 | 0.139555 | 28.037 | 0.7269 | 34.172 | 0.11300 | 0.18956 | Control |
| h200-contraction_only | gradient-005 | 600 | 0.140438 | 25.890 | 0.5861 | 19.990 | 0.09150 | 0.29179 | No |
| h200-contraction_only | gradient-025 | 600 | 0.142987 | 24.948 | 0.5223 | 16.230 | 0.04257 | 0.45658 | No |
| h200-contraction_only | normal-005 | 600 | 0.139705 | 25.440 | 0.7073 | 29.486 | 0.12176 | 0.26588 | Yes |
| h200-contraction_only | normal-025 | 600 | 0.140291 | 24.326 | 0.6988 | 28.882 | 0.11155 | 0.30726 | Yes |
| h200-learned_direction | l2 | 600 | 0.139454 | 25.122 | 0.6660 | 22.799 | 0.11096 | 0.71796 | Control |
| h200-learned_direction | gradient-005 | 600 | 0.140237 | 24.129 | 0.5515 | 6.131 | 0.08826 | 0.81641 | No |
| h200-learned_direction | gradient-025 | 600 | 0.143347 | 23.833 | 0.4813 | 3.435 | 0.03078 | 0.84415 | No |
| h200-learned_direction | normal-005 | 600 | 0.139579 | 23.770 | 0.6536 | 15.662 | 0.11421 | 0.774 | Yes |
| h200-learned_direction | normal-025 | 600 | 0.139822 | 23.464 | 0.6693 | 18.718 | 0.11487 | 0.76674 | Yes |
| h200-x_contraction | l2 | 600 | 0.139822 | 24.342 | 0.6411 | 11.780 | 0.12442 | 0.82045 | Control |
| h200-x_contraction | gradient-005 | 600 | 0.140489 | 23.972 | 0.5650 | 7.354 | 0.09235 | 0.88663 | No |
| h200-x_contraction | gradient-025 | 600 | 0.142325 | 23.988 | 0.5121 | 6.408 | 0.04973 | 0.91703 | No |
| h200-x_contraction | normal-005 | 600 | 0.139932 | 23.754 | 0.6536 | 14.458 | 0.12025 | 0.85894 | Yes |
| h200-x_contraction | normal-025 | 600 | 0.140056 | 23.560 | 0.6546 | 16.938 | 0.11357 | 0.82602 | No |

### Endpoints and numerical failures

These endpoints have unequal budgets and are not the primary comparison.
The unrestricted h=.20 normal beta=.05 branch reached update 600, while four
other objectives in that cell stopped earlier. Its completion does not override
the instability detected at the shared comparison state.

| Case | Variant | Last accepted update | Position RMS | Normal-angle RMS | Min J | Stop |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| h050-unconstrained | l2 | 600 | 0.033157 | 13.853 | 0.47853 | 600-update budget |
| h050-unconstrained | gradient-005 | 600 | 0.033985 | 8.507 | 0.45172 | 600-update budget |
| h050-unconstrained | gradient-025 | 600 | 0.034766 | 7.861 | 0.31717 | 600-update budget |
| h050-unconstrained | normal-005 | 600 | 0.033803 | 8.320 | 0.44144 | 600-update budget |
| h050-unconstrained | normal-025 | 600 | 0.034473 | 8.129 | 0.38888 | 600-update budget |
| h050-contraction_only | l2 | 600 | 0.034453 | 14.907 | 0.57165 | 600-update budget |
| h050-contraction_only | gradient-005 | 600 | 0.035159 | 7.731 | 0.60453 | 600-update budget |
| h050-contraction_only | gradient-025 | 600 | 0.035700 | 7.216 | 0.80849 | 600-update budget |
| h050-contraction_only | normal-005 | 600 | 0.034983 | 7.788 | 0.60801 | 600-update budget |
| h050-contraction_only | normal-025 | 600 | 0.035546 | 7.133 | 0.58181 | 600-update budget |
| h050-learned_direction | l2 | 600 | 0.034547 | 13.668 | 0.75725 | 600-update budget |
| h050-learned_direction | gradient-005 | 600 | 0.035293 | 7.607 | 0.7777 | 600-update budget |
| h050-learned_direction | gradient-025 | 600 | 0.035827 | 7.108 | 0.76434 | 600-update budget |
| h050-learned_direction | normal-005 | 600 | 0.035108 | 7.789 | 0.75732 | 600-update budget |
| h050-learned_direction | normal-025 | 600 | 0.035697 | 7.067 | 0.71253 | 600-update budget |
| h050-x_contraction | l2 | 600 | 0.035523 | 9.381 | 0.78707 | 600-update budget |
| h050-x_contraction | gradient-005 | 600 | 0.035885 | 6.970 | 0.87961 | 600-update budget |
| h050-x_contraction | gradient-025 | 600 | 0.036177 | 6.720 | 0.8976 | 600-update budget |
| h050-x_contraction | normal-005 | 600 | 0.035787 | 7.139 | 0.82134 | 600-update budget |
| h050-x_contraction | normal-025 | 600 | 0.036018 | 6.768 | 0.89109 | 600-update budget |
| h200-unconstrained | l2 | 253 | 0.131743 | 36.613 | 1.9857e-05 | Failed proposal 254: Forward line search failed, residual=2.540e-06 |
| h200-unconstrained | gradient-005 | 58 | 0.135091 | 32.271 | 0.00061096 | Failed proposal 59: Forward line search failed, residual=2.055e-06 |
| h200-unconstrained | gradient-025 | 48 | 0.135381 | 31.943 | 0.001896 | Failed proposal 49: Forward line search failed, residual=2.287e-06 |
| h200-unconstrained | normal-005 | 600 | 0.132549 | 32.572 | 0.059338 | 600-update budget |
| h200-unconstrained | normal-025 | 146 | 0.136675 | 28.176 | 0.018292 | Failed proposal 147: Forward line search failed, residual=6.786e-08 |
| h200-contraction_only | l2 | 600 | 0.139555 | 28.037 | 0.18956 | 600-update budget |
| h200-contraction_only | gradient-005 | 600 | 0.140438 | 25.890 | 0.29179 | 600-update budget |
| h200-contraction_only | gradient-025 | 600 | 0.142987 | 24.948 | 0.45658 | 600-update budget |
| h200-contraction_only | normal-005 | 600 | 0.139705 | 25.440 | 0.26588 | 600-update budget |
| h200-contraction_only | normal-025 | 600 | 0.140291 | 24.326 | 0.30726 | 600-update budget |
| h200-learned_direction | l2 | 600 | 0.139454 | 25.122 | 0.71796 | 600-update budget |
| h200-learned_direction | gradient-005 | 600 | 0.140237 | 24.129 | 0.81641 | 600-update budget |
| h200-learned_direction | gradient-025 | 600 | 0.143347 | 23.833 | 0.84415 | 600-update budget |
| h200-learned_direction | normal-005 | 600 | 0.139579 | 23.770 | 0.774 | 600-update budget |
| h200-learned_direction | normal-025 | 600 | 0.139822 | 23.464 | 0.76674 | 600-update budget |
| h200-x_contraction | l2 | 600 | 0.139822 | 24.342 | 0.82045 | 600-update budget |
| h200-x_contraction | gradient-005 | 600 | 0.140489 | 23.972 | 0.88663 | 600-update budget |
| h200-x_contraction | gradient-025 | 600 | 0.142325 | 23.988 | 0.91703 | 600-update budget |
| h200-x_contraction | normal-005 | 600 | 0.139932 | 23.754 | 0.85894 | 600-update budget |
| h200-x_contraction | normal-025 | 600 | 0.140056 | 23.560 | 0.82602 | 600-update budget |

### Convergence and interpretation

The objective/projection-gradient plots show progress within each objective.
Objective values should not be ranked across loss families: each includes a
different added penalty. Every 600-update endpoint is a fixed-budget result;
Adam's decaying learning rate and a flat objective trace do not certify inverse
stationarity. The reported projected-gradient mapping depends on the activation
coordinates, so cross-model gradient magnitudes are not physically equivalent.

Normal matching constrains direction but not segment length. Keeping positional
L2 helps anchor the target, yet a lower normal-angle error does not guarantee
less high-frequency residual, full amplitude, or a stable equilibrium. The
fixed-x large-target counterexample makes this limitation visible.

The evidence supports retaining L2 and testing a modest normal term for the
learned-axis 3D case, with explicit position and motion budgets, target-relative
roughness measures, and inversion/stability checks. It does not support replacing
L2 with normal loss or declaring a universally better mesh loss.

## Artifacts and run completion

- `data/10-comparison/summary.json`: all 40 accepted endpoints and failure records.
- `data/10-comparison/<cell>/<variant>/trace.csv`: objective, component errors,
  projected gradient, learning rate, forward/adjoint residuals and geometry at
  every accepted update.
- `data/10-comparison/<cell>/<variant>/history.npz`: saved displacement/control
  histories; `checkpoint.npz` stores the last accepted full optimizer state.
- `data/20-verification/checks.json`: independent metrics, constraints, forces,
  source checks, actual warm-start derivative checks and selected Hessians.
- `data/20-verification/selection.json`: exact selection receipt, including all
  rejected candidates, endpoint metrics and backtracking warnings.
- `data/30-figures/shared-step-profile-comparison.png`: selected-candidate 4 by 2.
- `data/30-figures/all-candidate-shared-step-profiles.png`: every candidate at the
  common step, including rejected curves.
- `data/30-figures/internal-wireframes.png`: interior meshes and target boundary.
- `data/30-figures/objective-and-projected-gradient-histories.png`: convergence
  diagnostics and failed proposal markers.
- `logs/10-run.log`, `logs/20-verify.log`: complete run logs and Comet summaries.

The main run and verifier both exited 0 after their Cherries/Comet shutdown hooks.
A known imported historical mesh-helper default registers the unused path
`data/10-pork-2d`, producing a missing-asset warning at shutdown. The actual new
comparison outputs exist and were independently inspected. The warning was not
used to suppress a numerical failure. Sources pass Ruff checks.

Main Comet summary excerpt (the full block is in `logs/10-run.log`):

```text
Comet.ml Experiment Summary
name: 2D target normal versus gradient refinement 4x2
url: https://www.comet.com/liblaf/apple/598a9a14eb81407c8eeaeaaf7e4627c1
cherries/cmd: .venv/bin/python src/10-run.py
cherries/entrypoint: exp/2026/09/21/normal-matching-profile/src/10-run.py
cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
cherries/start_time: 2026-09-21 01:41:16.294866+08:00
cherries/end_time: 2026-09-21 01:51:30.477208+08:00
```

Final figure rendering and visual QA completed after preserving the first layout
as `data/30-figures-initial`. Use only `data/30-figures` for the final figures.
All four PNGs were visually inspected: each profile uses the same physical scale,
all tested candidates remain visible, axis labels say deformed x, and the
unstable unrestricted large-target cell is explicitly marked. Endpoint failures
and horizons are in `data/30-figures/selection-and-endpoints.json`.
The final rendering run exited 0 after Cherries/Comet shutdown:
[327c7daa24eb4448aff1ab90191465cd](https://www.comet.com/liblaf/apple/327c7daa24eb4448aff1ab90191465cd).

```bash
CHERRIES_NAME='Target-normal profile: continuation comparison figures' CHERRIES_TAGS='target-normal,gradient-loss,2d,continuation,figures' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/30-render.py
```

The rendering log's Local-plugin overwrite warning concerns the run snapshot's
repeated artifact registration, not the numerical outputs. Final rendering
preserved the earlier layout separately and did not modify any simulation data.
