# Fixed-material expression fitting

The user approved the eye-inclusive loaded neutral and requested fitting all
36 expressions. This stage optimizes six symmetric positive-semidefinite active
stress coordinates for each of 288,235 muscle tetrahedra, plus one mandible hinge
angle per expression. There are 62,258,760 activation coordinates and 36 pose
coordinates across the suite. All material fields remain fixed. Run 005 continues
the accepted one-angle states from run 004, fitting expressions sequentially
with the [projected-descent correction](103-projected-descent-fix.md).
The earlier runs and their checkpoints remain preserved.
See [the hinge definition and validation](101-mandible-hinge.md).

## Mechanical and target contract

The constitutive reference is unchanged. Each target is the original expression
displacement added to the converged eye-inclusive neutral. The original passive
Stable Neo-Hookean materials, Poisson ratio 0.49, heterogeneous skin stiffness,
and prescribed skin stress remain active. Bulk baseline stress stays zero.
The full source cranium, mandible, and both fixed source eyes participate in
soft-tissue collision. The mandible moves rigidly; skull and eyes remain fixed.
The current contact model does not add soft-soft or rigid-rigid contact.

Each expression's area-weighted squared fitting error is divided by its original
expression displacement's area-weighted squared norm. It is not divided by
the displacement from the original constitutive reference. Thus a prediction
equal to the new neutral has normalized data loss one for every target.

Activation uses the existing Frobenius-orthonormal symmetric basis and
0.012328767123287673 MPa reference stress, with principal values constrained to
0–10 reference units. The 5 mm graph regularizer and effective-volume magnitude
penalty use weights recomputed on the adopted neutral. The magnitude coefficient
is 0.001. The jaw rotates about the fixed line from registered mandible landmark
1 to 9, with translation identically zero. Positive rotation opens the jaw.
Its exploratory bound is 0–40 degrees from the adopted neutral, which is not
independently verified dental occlusion. The scalar is normalized by 10 degrees;
its prior is `(0.01/6) * normalized_angle**2`, preserving the former six-coordinate
prior restricted to this axis. The bound and provisional hinge axis are modeling
assumptions, not an anatomical or collision certificate.

## Smoothness and optimizer

Before fitting, a discarded, small, spatially nonuniform PSD activation probe
is equilibrated. Independent target adjoints measure all 36 data-gradient RMS
values. Because smoothness is quadratic, its gradient is rescaled analytically
from the small feasible probe to the declared neighbor-RMS budget of 0.05.
The fixed coefficient is three times the largest ratio of data-gradient RMS
to this reference smoothness-gradient RMS. The largest ratio is repeated
at ten times tighter adjoint tolerance and must agree within 5%. The full
probe, forward receipt, individual ratios, and selected weight are saved.
The accepted activation field must also remain within the predeclared
0.05 normalized neighbor-RMS budget. The weight is not reduced when fit worsens.

Expressions are independent with fixed materials. One GPU works on one expression
until inverse convergence or an explicit optimizer failure, then advances to
the next expression. A per-expression iteration or overall wall budget stops the
run without declaring convergence or advancing to another target. Each trial
uses PNCG to the exact accepted-force threshold
1.5192003475221146e-10 MPa·m², followed by an implicit adjoint with relative
residual at most 1e-7. A conservative CCD boundary/inner line search and explicit
terminal intersection/gap checks protect soft-rigid contact. The PNCG settings
retain the neutral run's damping, restart interval, and step cap. The production
runner uses the original strict Armijo implementation. A zero-activation
invariant check is saved once for warm starts; targets retain the approved
neutral origin.
Inverted tetrahedra are reported rather than
automatically disqualifying a state, following the user's instruction.

Outer trials require Armijo objective decrease and the neighbor-RMS budget.
They also require residual-corrected Armijo decrease, and raw improvement must
exceed the sum of the two absolute first-order objective-error estimates
`abs(p_free · residual_free)`. These are estimates, not rigorous error bounds.
Rejected trials do not replace accepted checkpoints or optimizer moments.
When a projected Adam proposal is not a descent direction, a volume-metric
projected-gradient direction replaces it. Common positive scaling caps the
largest coordinate change at the configured learning rate. Its projection uses
one scalar metric per tetrahedron, making spectral projection compatible with
the descent direction. Adam moments reset only if that proposal is accepted.
The chosen direction is logged. Exhausted line
searches retain the last accepted state and identify the affected expression.
Each accepted state records separate loss terms, fit RMS, activation amplitude,
roughness, mandible pose, exact force, adjoint residual, and determinant tails.

Inverse convergence requires a small projected-gradient mapping in a fixed
effective-volume activation metric, a small jaw projected gradient, and a stable
five-objective window for five consecutive accepted updates. The estimated
primal objective error must also be at most `1e-6 * max(1, abs(objective))`.
An exactly zero projected direction is a separate constrained-stationarity
certificate: it can terminate immediately only with that same primal-accuracy
check, without inventing accepted updates or a stable history. If primal accuracy
is unresolved, it terminates explicitly without a convergence claim.
Budget exhaustion
or a small visual change alone is not convergence. This stage must converge
before treating it as the completed fixed-material preparation for the later
joint material fit.

## Outputs

- `data/expression-inputs-002`: immutable all-expression target bundle.
- `data/expression-fitting-005/protocol.json`: active run's numerical and source bindings.
- `calibration.json`, `calibration.jsonl`, `calibration-probe.pt`: smoothness evidence.
- `status.json`: live suite status; `summary.json`: latest terminal status.
- `expressions/<name>/initial.pt`, `latest.pt`: owned accepted states.
- `expressions/<name>/trace.jsonl`, `trials.jsonl`: accepted and rejected histories.
- Additional accepted snapshots every 25 updates and at convergence.

The historical one-angle round-robin run started at **2026-09-21 21:59:38 Asia/Shanghai**
under runtime-only user unit `apple-expression-fit.service`. Its Comet run is
[5ad3ef62e2d546f0980a6ba251a742e0](https://www.comet.com/liblaf/apple/5ad3ef62e2d546f0980a6ba251a742e0).
It was stopped at the user's scheduling request after preserving one accepted
update each for MouthOpen and BrowDownLeft. The sequential continuation keeps
those states, cached gradients and optimizer moments after checking compatibility.
This remains fixed-material fitting, not the final joint-material optimization.

From this experiment directory, the process command is:

```bash
CHERRIES_NAME='Sequential expression fitting with one mandible hinge angle' \
CHERRIES_TAGS='expression-fit,all36,sequential,fixed-material,rigid-eyes,pncg,mandible-hinge' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/93-fit-expressions.py \
  --continue-from data/expression-fitting-004
```

The transient service has a 49-hour lifetime; the runner has a 48-hour wall
budget and preserves accepted checkpoints. Console output is
`logs/93-expression-fitting-005-terminal.log`. The separate runtime-only
`apple-expression-fit-review.service` refreshes status each minute and renders
changed accepted checkpoints at most every 15 minutes, with immediate first
accepted and terminal renders. Repeated page builds skip asset archival and
unchanged image copies.

Live results over tailnet (private preview omitted).

Startup verified that the approved neutral already meets the production
force criterion: zero PNCG steps, **zero geometry change**, force
`1.499802320143145e-10`, minimum active contact gap `17.066746 µm`.

The first run (`expression-fitting-001`, Comet
[b06c6c2e572c442fbae0b5d815290022](https://www.comet.com/liblaf/apple/b06c6c2e572c442fbae0b5d815290022))
completed all 36 calibration adjoints. EyesLeft set the largest ratio; repeating
it at ten times tighter adjoint accuracy changed the ratio by only `2.67e-9`
relative. Inspection of the first proposal exposed an arbitrary-amplitude
dependence: its smoothness term alone was 2.276, versus total starting loss 1.
The run was deliberately interrupted before accepting an outer update.

The probe's neighbor RMS is `2.63528904e-5`; the declared reference/budget is
0.05. Quadratic homogeneity changes the weight from 9612.957 to **5.06658405**
while keeping the factor-three gradient dominance at that reference roughness.
The complete calibration is reused with hashes in run 002. A CPU check of the
actual graph confirmed the normalized coefficient is unchanged when the probe
amplitude is multiplied by ten. The first proposal's smoothness term becomes
0.00119958. This is a modeling-prior choice, not an experimentally measured
activation distribution; the hard neighbor budget remains unchanged.

## Historical six-coordinate run 002: first accepted expression updates

MouthOpen accepted its first update at **21:43:48 Asia/Shanghai**. This is an
actual outer fitting update, not the calibration probe. Its area-weighted fit
RMS decreased from **7.630657 to 7.623010 mm**; normalized data loss decreased
from 1 to 0.997996735, and total objective reached 0.998071715. The smoothness
contribution is `7.49738e-5`, neighbor RMS `0.000347909` (budget 0.05), and
volume-weighted tensor-stress RMS **12.047 Pa**.

The full and half jaw proposals failed boundary CCD; the quarter proposal
passed. Its forward solve took 3007 PNCG iterations / 362.1 seconds and reached
force `1.45185e-10`, below `1.51920e-10`. The adjoint took 8.31 seconds, with
relative residual `7.72e-8`. Terminal checks found **zero inverted tetrahedra**,
no soft-rigid intersections, and a **17.0802 µm** minimum active gap. The raw
decrease exceeds the sum of estimated primal errors by 0.00192358, and the
residual-corrected objective also decreases. The run then advanced to
BrowDownLeft. Neither expression convergence nor a joint-material result is
claimed by this first update.

BrowDownLeft subsequently accepted its first update: objective 0.999332989,
fit RMS 2.036931 mm, zero inverted cells, and successful forward/contact gates.
The runner advanced to NeckCompression. The rendered dashboard labels initial
states separately from accepted optimizer steps. Target, prediction, residual,
stress-field, and trend figures were inspected; full-face framing and independent
color legends were corrected. The live page and all ten figures in
`expression-fitting-visuals-9558ddf55464-c430b9a9` returned HTTP 200. Both runtime
services remained active after this check.

## Numerical validation during startup

The ordinary-force-tolerance directional check failed: active stress had 91.1%
relative disagreement and the jaw direction 30.4%. These are failed validation
results, not fitting results. The exact force Hessian was then independently
checked and matched centered finite-difference forces to 6.52e-8 relative error
on the real eye-inclusive mesh. A controlled double-precision quadratic test
confirmed that an experimental work-integral line search obtains the same
accepted step and energy decrease even after adding a 1e16 constant to the
energy. Its stricter-force runs were slow and were not admitted as successful
validation or selected for production.

The next gradient check used two larger perturbation scales
near the actual optimizer proposal: active stress 100 and 30 Pa; jaw coordinates
1e-5 and 3e-6 in the declared rotation/translation direction. Both scales must
agree with the implicit derivative within 5%, and the two finite differences
must also agree. Negative stress perturbations are only the smooth mathematical
extension used for central differences; accepted activation fields remain PSD.
All nine PNCG solves passed force/contact gates, but the raw finite-tolerance
gradient checks still failed. At 100/30 Pa activation, errors were 3.741%/10.698%;
at the two jaw steps, errors were 6.043%/17.277%. They remain recorded as failed
in `data/expression-scale-gradient-validation-001`.

Script 100 then separated the implicit equilibrium derivative from the numerical
error of the finite-tolerance PNCG result. With `H p = -J_u`, it evaluated the
first-order corrected objective `J + p · residual`. Corrected directional
errors were **0.01432% / 0.00175%** for activation and
**0.00764% / 0.00017%** for jaw pose. Their two-scale disagreements were
0.01257% and 0.00747%. Independent finite differences of mechanical
force versus the parameters matched the implicit chain to at most 0.01175%.
All three diagnostic gates passed without changing the 5% criterion or the
physical model. See [the diagnostic report](100-diagnose-expression-residuals.md).

The runner therefore explicitly admits the **implicit equilibrium derivative**
using the Hessian, force/parameter derivative, and corrected two-scale checks.
It does not relabel the raw finite-tolerance check as successful. The raw and
corrected decrease guards above address the measured residual sensitivity
during startup; the evidence is local and does not prove all future contact
transitions. All validation files and implementation sources are hash-bound.

A CPU transaction test of the final runner accepted a resolved quadratic step
and rejected a raw-decreasing but numerically unresolved step without changing
the last accepted checkpoint bytes. Its receipt is
`tmp/93-transaction-check/receipt.json`.
