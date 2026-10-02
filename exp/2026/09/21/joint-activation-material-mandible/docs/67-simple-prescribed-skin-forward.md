# Simple forward with prescribed heterogeneous skin

This diagnostic fixes the mandible and uses complete registered cranium and
mandible surfaces for frictionless soft-bone contact. Fat, aponeurosis, and
muscle have exactly zero additive baseline stress. Expression activation is
absent. Skin has prescribed spatially varying stiffness and tensile baseline
stress. The forward solver is strict PNCG; there is no inverse solve, adjoint,
Newton stage, or solver fallback.

## Geometry and reference

The admitted candidate002 soft-tissue positions become the new FEM reference:
`X_new = X_original + candidate002_displacement`, with initial displacement zero.
This removes the repair-induced bulk strain that caused the previously verified
negative-curvature witness. It is a deliberate change of mechanical reference,
not a claim that the original neutral anatomy was stress-free. Source bones,
fixed nodes, outer skin, connectivity, and tissue fractions remain unchanged.
The derived fixture is only admitted for this standalone forward; its retained
historical activation graph is unused and is not an admission for joint inverse
optimization.

The preparation run measured zero soft-bone intersections, positive reference
tet volumes, 6,185 active IPC representatives, and a minimum active feature gap
of 0.12172 micrometres. All 54,110 source bone triangles are retained. The two
fixed source bones have their previously documented mutual intersections;
bone-bone contact and a movable jaw are outside this test.

Contact uses a physical IPC barrier with distance 0.1 mm and stiffness 0.01 MPa.
These are numerical contact settings, not literature tissue material values.

## Skin field

See [the primary-source table and transfer rationale](66-skin-forward-literature.md).
Six Flynn (2013) regional inverse fits supply derived small-strain moduli and
isotropic means of directional prestress. Manual anatomical anchors, bilateral
transfer, and smooth convex interpolation construct a subject-specific field.
The membrane thickness is 1 mm and Poisson ratio is 0.46. Prestress preserves the
paper's fitted three-dimensional stress, rather than its 1.5 mm shell resultant.
Uncovered sites (including nose, lips, eyelids and scalp) use extrapolation.

The law is the project polynomial Stable Neo-Hookean bulk and exact plane-stress
membrane with additive active stress. Regional Ogden fits are therefore
constitutive priors; they are not a directly measured SNH parameter map.

## Solver and evidence

`68-run-simple-skin-forward.py` loads hash-bound geometry and skin arrays, asserts
zero bulk stress, and runs PNCG with relative force tolerance `1e-5`, absolute
force tolerance `1e-11` in internal MPa m² units, at most 5,000 steps, and strict
Armijo line search and a 50 micrometre infinity-norm trial displacement cap. The independent terminal force check is required even when
the optimizer reports success. A 30-minute budget is checked after accepted
steps; it cannot interrupt an in-flight native call.

Each accepted step is written to JSONL. Exact force is recomputed every ten
steps, with checkpoints at step zero, step one, every fifty steps and termination.
Operation timings, a 15-second heartbeat and 60-second stack dumps distinguish
slow native calls from slow iteration. PNCG uses the existing bulk-clamped/contact
Gauss-Newton directional quadratic; it is not an exact-Hessian Newton solve.

Preparation source: `src/67-prepare-simple-forward.py`.
Preparation artifacts: `data/simple-skin-forward-inputs-001/`.
Preparation Comet: <https://www.comet.com/liblaf/apple/fa7b52928d0a47e0a9026c3d18344b0a>.

Terminal solver evidence is recorded below and in [the forward run report](68-simple-skin-forward.md).

## First attempt: rejected

Run `data/simple-skin-forward-001/` used a 50 micrometre trial displacement cap.
It made 387 accepted energy-decreasing steps in 53.29 seconds, then blocked in
`collision.max_step_size` / TightInclusion CCD. The saved checkpoints had already
become invalid: step 1 had no inversions, step 50 had 13, step 200 had 59, and step
350 had 80. Decreasing energy therefore did not establish physical validity.
The last persisted displacement is step 350; step 387 exists only as a trace row.
SIGINT and SIGTERM could not unwind the native call, and the owned process was
terminated. The external interruption receipt preserves that distinction.

The initial free-gradient norms were 3.024 N for tissue and 7.673 N for contact,
with combined norm 8.248 N. These vector norms are not additive scalars. This
shows the initial load included substantial contact repulsion as well as skin
tension; the run does not isolate skin tension as the cause of inversion.

A fresh retry uses an experiment-local geometric step bound. For each tetrahedron,
let `F(t) = F + t G`; restrict the trial fraction so
`||t F^-1 G||_F <= 0.8`. Since the spectral norm is bounded by the Frobenius norm,
`I + t F^-1 G` cannot become singular anywhere along that segment. Starting from
positive determinant, this preserves orientation. IPC CCD receives the already
restricted step. This changes step selection only; the energy and material
parameters remain unchanged. A crossing-tetrahedron check verified the guard,
and every saved checkpoint now reports determinant and contact diagnostics.

## Guarded retry: not converged

Run `data/simple-skin-forward-002/` stopped cleanly after 47 accepted PNCG steps
and 20.73 seconds. Strict Armijo exhausted 40 backtracks. The final free-force
norm is 6.790 N against a threshold of 0.00008247 N. There are zero inverted
tetrahedra, but the minimum physical determinant is 4.1609e-6; this is near
element collapse, not equilibrium. The skin RMS displacement is only 0.002206 mm.

The limiting cell is pure-fat tetrahedron 658378, about 0.01027 mm from the
source mandible. Its reference volume is 5.606e-15 m³. All four nodes were in
the local clearance repair footprint. A separate complete-source IPC audit
finds zero soft-bone intersections both at rest and at the terminal state.
The minimum active contact gap is 0.05116 micrometres.

This localizes the observed failure to a nearly flattened, bone-adjacent fat
element under the combined contact and skin load. It does not prove that no
valid equilibrium exists or isolate one physical cause. No additional material
constraint or artificial determinant floor was added to manufacture convergence.

- [Terminal forward receipt](../data/simple-skin-forward-002/summary.json)
- [Deformation, fields and solver figures](../data/simple-skin-forward-visuals-002/summary.json)
- [Numerical geometry audit](../data/simple-skin-forward-inversion-audit-001/summary.json)
- [Run Comet](https://www.comet.com/liblaf/apple/1bd7343193554c31a22f317ef3b40dac)
- Mobile review (private preview omitted)
