# MouthOpen residual-aware pose proposal

Diagnostic164 established a specific failure: cell18514 remained positive in
the coupled seed but inverted during nonlinear correction. Lowering relative
damping from 1e-4 to 1e-5 did not resolve it. This experiment isolates one
numerical change at the original 1e-4 shift: account for the predicted force
residual correction when proposing the actual jaw increment at alpha 0.125.

The source is the exact independently audited run011 checkpoint, with both
Adam counters 267 and all saved moments. The strain direction and gradients
come from the hash-bound original-shift case in completed diagnostic164. The
pose QP is solved afresh and compared with its saved CPU certificate. Every
currently positive retained tetrahedron is checked against both the seed and
affine corrected determinant predictions.

For actual normalized pose increment `v`, the constraint is
`A v >= 1e-6 - J_old - min(residual_delta_J, 0) - alpha * q_delta_J`.
The residual term is not multiplied by alpha or divided by probe epsilon.
The resulting actual pose increment is applied once; the ordinary production
coupled predictor and CCD path remain unchanged. Armijo uses the actual joint
increment dot product. No diagnostic displacement, control or optimizer state
is adopted.

The script saves the raw predictor before CCD, the admitted seed before the
inversion gate, and the corrected or failed raw displacement before fresh
metrics. It preserves collision, exact saved skin pre-strain, IsFixed-only
constraints, exclusion only of all-four-fixed tetrahedra, physical force
acceptance 1e-8, internal force target 1e-9, maximum 100 retained inversions,
and maximum 1e-4 inverted rest-volume fraction. The single diagnostic has a
600-second wall budget after rebuilding physics. Numerical failure is
recorded explicitly and never described as convergence.

## Diagnostic reproducibility check and additive rerun

Output001 completed with exit 1 before the physical corrector because the
script incorrectly required two independent shifted-CG predictor solutions
to match within 5e-14 m. They differed by at most 6.27107974923713e-10 m.
Both had identical controls, predictions, RHS norm, shift and contact pattern;
native and sparse shifted relative residuals were below 1e-7. The independent
solves stopped after 1310 and 1332 operator applications. This comparison was
not a valid accuracy contract for iterative solutions.

The corrected script records maximum and L2 disagreement and requires exact
fixed values and each solve's native and sparse residual contract. The actual
production CCD-admitted seed still passes all physical checks independently.
Output001 and its original source snapshot remain preserved; no candidate was
adopted. This changes a diagnostic check, not the force or geometry policy.

Output002 runs the same proposal with this correction:

```bash
TMPDIR="$PWD/tmp/affine-corrector-002-runtime" \
CHERRIES_NAME='MouthOpen affine residual corrector validation' \
CHERRIES_TAGS='mouthopen,diagnostic,affine-residual,corrector' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/165-test-mouthopen-affine-corrector.py \
--output-dir data/mouthopen-affine-corrector-002 \
> tmp/165-mouthopen-affine-corrector-002.log 2>&1
```

Working directory is this experiment group. Exact job identity is in output002
`job.json`: PID 835820, process start ticks 4000509, tool session 33666.
CPU Ruff and compilation passed before launch. Source controls, moments and
counters267 remain the audited run011 state.

## Baseline affine result and lower-shift test

Output002 and Cherries completed with exit 0. The actual production seed had
100 inversions, but the corrected state had 102, newly inverting cells18514
and598977. Force 9.543556880263738e-10 passed the internal target and RMS
improved to 1.9801122458296738 mm, but the inversion gate rejected the result.
It was not adopted.

For cell18514: predicted seed J 0.0007447984086, actual seed J
0.0007431815576, corrected J -0.0003620016271. For cell598977: predicted
seed J 4.1293881e-6, actual seed J 7.5825178e-6, corrected J -3.1755295e-6.
The affine corrected prediction placed both at 1e-6; it underestimated the
nonlinear correction for this changed pose. Corrector motion reached
2.149760573346545e-5 m. This does not justify treating the affine model as a
geometry certificate or increasing the physical inversion allowance.

Output003 tests the already certified lower-shift case from completed164,
using that case's own source-Adam direction, gradients, determinant tangents
and residual response. Relative shift is 1e-5 and alpha remains0.125. Its
residual response has 8.25% native unshifted linear residual, compared with
40.64% at1e-4. This is an explicitly approximate numerical model. The command
above changes output/TMPDIR/log suffix002 to003 and adds
`--relative-shift 1e-5`; name is `MouthOpen lower-shift affine residual validation`,
tags `mouthopen,diagnostic,affine-residual,lower-shift`. Exact PID842356,
start ticks4024312 and session43529 are recorded in output003/job.json.

## Lower-shift corrected candidate passed

Output003 and Cherries shutdown completed with exit 0. The proposal performed
100 PNCG iterations and 5 Newton steps, then passed the unchanged physical,
internal force, collision and Armijo gates:

- RMS: 1.9802781442119066 mm, from 1.9818275442754778 mm.
- Raw force: 8.494559769427815e-10 (0.0008494559769427815 N).
- Retained inversions: 100; no newly inverted retained cells.
- Inverted rest-volume fraction: 1.2304029393884475e-5.
- Loss: 0.06744482973012987, from 0.06755041076135844.

Cell18514 stayed positive: predicted seed J 0.0015065863067, actual seed J
0.0015065577614 and corrected J 0.0010041285635. Cell598977 also stayed
positive, corrected J 7.22216675668245e-6. This is evidence that the new
proposal can pass actual nonlinear correction; it does not establish inverse
stationarity or guarantee later proposals.

[Comet output003](https://www.comet.com/liblaf/apple/d7c23e7bf55f4ad98b8a67ebc494f3a4).
Source hashes were rechecked and four live tailnet assets matched local bytes.
The candidate remains diagnostic-only. Run012 will recompute proposals from
the exact independently audited run011 checkpoint and all moments/counters267.
