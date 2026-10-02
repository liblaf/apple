# Approved rigid mandible continuation

This record covers the completed forward initialization in
`data/pose-rigid-resume-002`. It reaches the approved six-degree-of-freedom
chin pose through two bounded relative updates. It is an initialization for
the separate `130-inverse-mouthopen-rigid.py` run, not a physically valid
forward result: the final state contains 539 inverted tetrahedra and its
minimum `det(F)` is `-79.28334659163697`.

## Final state

The exact final chin pose, in the runner's rotation-radians followed by
translation-metres convention, is:

```text
[ 0.17388586765445685, -0.01240702857117505,  0.012728364310625942,
  0.0005163017883831333, -0.005821238348295399, 0.0012228731422144054]
```

The total relative motion from the source pose is 1.972357248155914 degrees
and 1.9902205954310664 mm. Each update was 0.986178624077958 degrees and
0.9951102977155332 mm. The IPC stiffness remained fixed at 0.3386 MPa.

The final contact-corrected step ended at raw force
`9.983599564342892e-09 MPa m²`, or **0.009983600 N**, below the `0.01 N`
gate. Its recorded terminal gates all passed: force convergence, numerical
contact validity, no intersections, and the active-gap buffer. These contact
and force gates do not repair the inverted elements, so `valid_forward` is
false.

The final area-weighted skin RMS mismatch to the MouthOpen target is
**2.799292053835492 mm**. This CPU calculation follows the inverse objective:
it uses `rendering.npz` skin IDs and triangles, areas on the blendshape new
neutral, and the final endpoint displacement. It verified the rendering and
blendshape SHA-256 records in the protocol. For scale, target motion has an
area-weighted RMS of 7.631002707049816 mm; the normalized loss is
0.13456533826899125. This is a target-fit measurement only, not a physical
validation given the 539 inversions.

## Completed stages

The source no-contact recovery used the matched 3000-CG-direction checkpoint.
It began directly in Newton (`PNCG: skipped_newton_resume`), took 263 Newton
steps, and completed in 696.7552277850045 s at 0.009888747 N. The following
contact correction completed in 54.475813928991556 s at 0.008365377 N; its
minimum active gap was 67.3717965476642 micrometres.

| Stage | Carry | No-contact relaxation | Push-out | Contact correction | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relative update 1 | 9.559761 s | 120.500745 s | 21.873111 s | 35.920149 s | 189.150411 s |
| Relative update 2, final | 9.558840 s | 23.991363 s | 20.314375 s | 298.407969 s | 354.349409 s |

The full continuation took 544.5206985450059 s. The first update ended with
452 inversions and minimum `det(F) = -65.43180889773144`; the final update
ended with the 539 inversions stated above.

## CG-3000 recovery evidence

The source checkpoint is
[`endpoint-cg3000.pt`](../data/no-contact-newton-probe-001/endpoint-cg3000.pt),
selected from the matched no-contact Newton-direction probe. The probe held
the checkpoint, collision-disabled objective, tolerance, maximum step, and
shift policy fixed while changing only the inner CG cap. CG 1000 exhausted a
direction and retried with a shift; CG 3000 completed 1601 CG iterations
without that retry. This selects the source recovery path; it does not claim
that the probe's one-step residual was lower with CG 3000.

Probe command:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Probe no-contact Newton CG budget' \
CHERRIES_TAGS='new-neutral,mouthopen,rigid-pose,newton,cg-probe' \
.venv/bin/python -u \
  src/124-probe-no-contact-newton.py \
  --input-checkpoint data/pose-rigid-resume-001/newton-probe-fixture.pt
```

The [matched probe](https://www.comet.com/liblaf/apple/2222611bdf0b41aab238ba733d728106)
and its receipt preserve the two directions and their timings.

## Archived attempts

These outputs remain evidence; none was relabelled as successful recovery.

1. [`pose-rigid-001`](../data/pose-rigid-001/) reached its first carry,
   no-contact solve, and push-out, then stopped on non-finite PNCG energy in
   contact correction. The pushed checkpoint was retained.
2. [`pose-rigid-diagnostic-001`](../data/pose-rigid-diagnostic-001/) passed
   two contact-corrected updates, then hit the declared 600 s no-contact wall
   limit at 0.025612 N. It saved its last accepted partial state, not an
   unaccepted Newton trial.
3. [`pose-rigid-resume-001`](../data/pose-rigid-resume-001/) is the
   interrupted source-recovery attempt. It was superseded by the explicit
   matched-CG probe and `pose-rigid-resume-002`; it is not a completed forward
   result.

## Reproduction and records

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='Continue rigid pose with adequately solved Newton directions' \
CHERRIES_TAGS='new-neutral,mouthopen,rigid-pose,resume,newton,cg3000' \
.venv/bin/python -u \
  src/120-forward-chin-rigid-pose.py \
  --source-checkpoint data/no-contact-newton-probe-001/endpoint-cg3000.pt \
  --source-phase no_contact --source-start-from-newton true \
  --no-contact-linear-max-steps 3000 \
  --output-dir data/pose-rigid-resume-002 --off-wall-seconds 3600
```

The final run was recorded from 15:12:23 to 15:34:43 on 2026-09-23
Asia/Shanghai. Its terminal log is
[`pose-rigid-resume-002-terminal.log`](../tmp/pose-rigid-resume-002-terminal.log).
The preceding logs are retained as
[`pose-rigid-001-terminal.log`](../tmp/pose-rigid-001-terminal.log),
[`pose-rigid-diagnostic-001-terminal.log`](../tmp/pose-rigid-diagnostic-001-terminal.log),
and
[`pose-rigid-resume-001-terminal.log`](../tmp/pose-rigid-resume-001-terminal.log).

- [Initial failed contact correction](https://www.comet.com/liblaf/apple/cef9042452a24eeb83422b7c94604459)
- [Diagnostic continuation and timeout](https://www.comet.com/liblaf/apple/9e600779703c4ec0a4f5d3fed4a1ea74)
- [Interrupted source recovery](https://www.comet.com/liblaf/apple/12c712af16a24fbdae68636c98d0c496)
- [Completed CG-3000 recovery](https://www.comet.com/liblaf/apple/1b5f751dedd247faa6656f1ca7788dc1)
