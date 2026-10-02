# MouthOpen rigid-pose inverse

`data/inverse-mouthopen-rigid-002` was the inverse run for unrestricted
symmetric active strain per muscle tetrahedron and a six-degree-of-freedom
chin pose. It exited with an exact-adjoint failure during the first Armijo
candidate's gradient re-evaluation. It recorded zero accepted optimizer
iterations.

## Contract

The runner starts from the contact-corrected endpoint of
[`pose-rigid-resume-002`](../data/pose-rigid-resume-002/), uses the repaired
reference in `reference-clearance-002`, and targets the MouthOpen blendshape
from `blendshapes-005`. Its parameterization includes six symmetric
active-strain components per active muscle tetrahedron and all six normalized
jaw coordinates. Physical pose
coordinates are rotation radians followed by translation metres.

Each forward solve uses the hybrid PNCG-then-sparse-Newton primal with fixed
IPC stiffness 0.3386 MPa, raw force tolerance `1e-8 MPa m²` (0.01 N), and
the contact gates recorded by the runtime. The inverse run allows 50 accepted
iterations, at most 12 Armijo backtracks per iteration, and an 834 s wall
budget. Its source and input SHA-256 records are in
[`protocol.json`](../data/inverse-mouthopen-rigid-002/protocol.json).

The sparse adjoint uses an exact unshifted free CSR Hessian. It compares a
deterministic CSR product with the native Hessian product, then gates both
CSR and native true residuals at `1e-7`. The saved receipt includes the
operator proof, residuals, and operator-application count.

## Saved initial state

At the current saved iteration 0, the loss is `0.1345653382689913` and the
area-weighted skin RMS mismatch is **2.7992920538354924 mm**. The initial
pose is:

```text
[ 0.17388586765445685, -0.01240702857117505,  0.012728364310625942,
  0.0005163017883831333, -0.005821238348295399, 0.0012228731422144054]
```

Its rotation magnitude is 10.014843453723488 degrees and translation
magnitude is 5.970661786293199 mm. The initial active-strain vector is zero.

The inherited forward force is 0.009983599564342893 N, below its 0.01 N
threshold, and its contact gate is recorded as true. The geometry is still
invalid: 539 tetrahedra are inverted and the minimum `det(F)` is
`-79.28334659163697`. Thus the saved record correctly reports
`forward_converged: true`, `contact_valid: true`, `valid_forward: false`, and
`inverse_converged: false`. Passing force and contact tests does not make
this initialization physically valid.

The initial adjoint completed in 36.91481453302549 s. Its native relative
residual is `9.867957435037821e-08`, its CSR/native product relative error is
`1.7191972781286563e-16`, and it used 21,799 CSR operator applications in one
successful CG attempt. The fixed-state Lagrangian pullback checks are saved
in [`gradient-check.json`](../data/inverse-mouthopen-rigid-002/gradient-check.json);
they hold free coordinates fixed and rebuild fixed coordinates and contact
state for each pose perturbation. They are not fully resolved neighbouring
equilibrium finite differences.

## Forward-verified trial and failed adjoint

Trial 0 at iteration 1 passed the forward Armijo test with loss
`0.10836822494546867`, area-weighted fit RMS 2.5120749191738163 mm, force
0.00996023710869922 N, and all contact gates. Its true relative rigid motion
was 0.8652883273133697 degrees and 0.8660254037760805 mm, both below their
one-degree and one-millimetre bounds. The initialization receipt reports 554
inverted tetrahedra and minimum `det(F) = -68.13730227683844`, so it is not a
valid physical forward result.

The exact unshifted adjoint then failed after three 100,000-iteration CuPy CG
attempts. Its final CSR and native true relative residuals were respectively
`1.2123875684841569e-4` and `1.2123875687955695e-4`, above `1e-7`, despite a
CSR/native operator proof error of `3.059118124456074e-16`. The failed
adjoint therefore did not produce an optimizer update.

[`inverse-mouthopen-rigid-trial-001`](../data/inverse-mouthopen-rigid-trial-001/)
is the separate forward-verified export. Its status is
`forward_verified_trial_adjoint_failed`, its summary states
`accepted_optimizer_iterations: 0`, and it pins the source trial,
initialization endpoint, failure receipt, and terminal-log hashes. It is the
only appropriate artifact for reviewing the trial output.

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='MouthOpen joint rigid pose and Raw6 with verified exact adjoints' \
CHERRIES_TAGS='new-neutral,mouthopen,inverse,rigid-pose,active-strain,deadline' \
.venv/bin/python -u \
  src/130-inverse-mouthopen-rigid.py \
  --output-dir data/inverse-mouthopen-rigid-002 \
  --initialization-checkpoint data/pose-rigid-resume-002/endpoint.pt \
  --wall-seconds 834
```

- [Failed exact-adjoint inverse run](https://www.comet.com/liblaf/apple/e728ebb761644dffadd3805b3b5336b5)
- [Forward-verified trial export](https://www.comet.com/liblaf/apple/8da302dce46841d88b65c11d4862c3e9)

## Published review

Published and verified at **2026-09-23 15:56:58 +08:00**, before the requested
16:00 deadline: MouthOpen joint trial (private preview omitted).
The page includes front/side views with bones and eyeballs, saved fit/force
curves, and a pinned `endpoint.npz` download. The endpoint is a forward-verified
trial with failed adjoint; publication does not claim inverse convergence or
valid tetrahedral geometry.

```bash
CHERRIES_NAME='MouthOpen forward-verified joint trial final review' \
CHERRIES_TAGS='new-neutral,mouthopen,inverse,rigid-pose,review,deadline' \
.venv/bin/python -u \
  src/131-review-rigid-inverse.py \
  --run-dir data/inverse-mouthopen-rigid-trial-001
```

The successful renderer exited zero after Cherries shutdown:
[review Comet run](https://www.comet.com/liblaf/apple/3798fb93a6d2407db993ee2a96a09a79).
Its terminal log is `tmp/rigid-inverse-review-002.log`. An earlier review attempt
stopped at the export schema assertion; the export protocol compatibility
field was corrected, preserving the distinct `export_schema` and trial status,
before this successful rendering. The final viewer receipt pins the corrected
protocol and unchanged endpoint tensors.

The front, side and curve images were visually inspected. HTTP 200 and
byte-for-byte SHA-256 checks passed for the page, all three images, endpoint,
receipt and snapshot summary. The publication evidence is
`data/review-repaired-reference-005/rigid-inverse/publication.json`.
The existing transient tailnet server remains the serving process.

Future-source changes also clear stale adjoint diagnostics and check deadlines
through expensive adjoint stages. The inverse runner now saves a terminal
failure summary for a failed accepted-candidate gradient evaluation before
raising. These changes were compiled and linted; they were not substituted
into the already completed inverse002 run. Its archived sources remain the
runtime source authority. No Git changes were staged or committed.
