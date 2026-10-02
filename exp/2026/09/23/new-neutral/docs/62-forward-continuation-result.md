# Continued neutral reaches the force tolerance

Run `forward-repaired-reference-005` converged to **0.009250543 N**, below the
unchanged **0.01 N** force threshold. It took 41 additional Newton steps and
42.722 s of instrumented forward time. The saved endpoint has **two inverted
tetrahedra**, so `solver_converged=true` and `valid_forward=false`. It is a
diagnostic result, not an admissible neutral configuration.

| Quantity | Parent 001 | Continuation 005 |
| --- | ---: | ---: |
| Free-force norm (N) | 0.159060385 | 0.009250543 |
| Force threshold (N) | 0.01 | 0.01 |
| Total energy (J) | 2.471214861 | 2.458682484 |
| Barrier stiffness (MPa) | 0.3386 | 0.3386 |
| Minimum det(F) | 0.029106899 | -0.220242147 |
| Inverted tetrahedra | 0 | 2 |
| Minimum active contact gap (µm) | 89.442772 | 89.329918 |
| Weighted skin motion RMS (mm) | 0.823158 | 1.347343 |

The independent CPU audit identifies inverted tetrahedra **634260 and 691607**
among 1,146,517 elements. Contact remains intersection-free according to the
solver's contact audit and satisfies the existing 10 nm admissibility bound.
The independent review finds zero triangle intersections against bones and eyes.
The clearance-repaired material reference still has minimum gap
100.093782531 µm, above `d_hat = 100 µm`; this reference requirement is distinct
from the allowed deformed contact gap.

## Continuation and numerical settings

The continuation reconstructs parent 001's exact displacement, active-strain
fields, repaired constitutive reference, fixed rigid geometry, and adaptive IPC
state. Restart force and energy agree with the saved parent. It preserves the
original force anchor 3.023991481 N, relative tolerance `1e-3`, absolute tolerance
`1e-8 MPa·m²`, stiffness cap 16.93 MPa, and adaptive epsilon `1e-6`. The absolute
tolerance controls this run. No PNCG steps are repeated; the full successful
lineage contains 332 PNCG and 141 Newton steps.

Newton shift reuse changes the linear search system only. Physical energy,
gradient, Hessian, line search, CCD and stopping tolerance remain unchanged.
The existing reuse option was made configurable near the force threshold:
`reuse_shift_force_ratio=0` permits reuse until convergence. The shared solver
default remains 3, preserving other callers. This is a numerical setting for
this continuation, not a demonstrated universal solver improvement or a claim
of identical endpoints across policies.

The exact sparse HVP check has relative error `3.059e-16`. Focused CPU checks
`check_shift_reuse.py` and `check_newton_step.py` passed. The independent saved
result audit verifies source archives, parent hashes, continuation seed,
activation fields, reference coordinates, tolerance anchor, stiffness and CPU
tetrahedron determinants. Archived source files are authoritative; later
plotting and documentation edits do not change the saved run.

Earlier attempts remain recorded: 002 stopped during setup because regenerated
VTK bytes differed despite equal mesh arrays; the check now compares all mesh
arrays and connectivity. 003 with reset shifts and 004 with the old near-gate
reuse reset were interrupted after slow progress. Their last checkpoints and
interruption receipts are retained. The successful 005 lineage resumes 001
directly, so their steps are not included in its energy curve or timing.

The material law permits finite energy at negative det(F), and the surface IPC
barrier does not constrain interior tetrahedron orientation. A checkpoint
[diagnostic](61-tet-634260-diagnostic.md) locates one collapsing element in
interior, inactive fat. Reaching the residual tolerance therefore does not
resolve the volume validity problem.

## Reproduction and evidence

From `exp/2026/09/23/new-neutral`, choose a fresh output directory:

```bash
CHERRIES_NAME='Continue neutral equilibrium with sustained Newton shift reuse' \
CHERRIES_TAGS='neutral,active-strain,reference-clearance,hybrid,newton,continuation,adaptive-ipc,shift-reuse' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/61-continue-forward.py \
  --output-dir data/forward-repaired-reference-repeat
```

Defaults select parent 001, the same repaired reference, 5,000 maximum Newton
steps, shift reuse with near-gate ratio zero, and checkpoints every 100 steps.
The successful continuation stopped at its force gate after 41 steps.

- [Forward summary](../data/forward-repaired-reference-005/summary.json)
- [Independent audit](../data/forward-repaired-reference-005/independent-audit.json)
- [Endpoint](../data/forward-repaired-reference-005/endpoint.npz)
- [Run log](../data/forward-repaired-reference-005/forward.log)
- [Comet forward run](https://www.comet.com/liblaf/apple/a54a73c42c994768b5c646964762cce7)
- [Comet independent audit](https://www.comet.com/liblaf/apple/a9265d1b9ad9406c983d53bdbd4b1a30)
- [Saved review](../data/review-repaired-reference-005/index.html)

The review includes rigid bones and eyeballs, the actual displacement and a
labelled 10× display, the cumulative energy curve, and the inverted elements.
The tailnet address is <PRIVATE_PREVIEW_URL>. All 41 served files match their
local SHA256 hashes; the tailnet DNS address and linked local assets also pass.
The server remains a transient user service under `/run/user/1000`.
[Hosting receipt](../data/serve-continuation-005.json) and
[HTTP verification](../data/http-verification-continuation-005.json) record the
actual served result. Full skin vertex RMS motion is 1.4104 mm, with 0.6432 mm
RMS additional motion relative to the saved parent endpoint.

Review evidence: [endpoint](https://www.comet.com/liblaf/apple/d8c9c3a018104cbfa42ec0b508694f2a),
[motion](https://www.comet.com/liblaf/apple/34790f92bf9c47fc91f0e5ca3c292b22),
[energy](https://www.comet.com/liblaf/apple/d6b48621ea5242cfae1308d0ea2ce868), and
[anatomy](https://www.comet.com/liblaf/apple/9ed40c81a43c4850bf11fdab30e85a11).
Their completed Cherries logs are preserved under the forward run's
`review-logs/` directory.
