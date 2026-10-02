# Corrected MouthOpen chin-pose feasibility

## Purpose

Test whether the 6-DoF chin pose estimated from the newly transferred
`MouthOpen` target can be imposed on the corrected `IsFixed` FEM boundary.
This is a CPU kinematic gate. It does not run collision, carry, equilibrium,
or inverse physics.

## Command

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Audit corrected MouthOpen chin pose feasibility' \
CHERRIES_TAGS='mouthopen,isfixed,chin-pose,cpu,kinematic-gate' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
  src/141-audit-isfixed-chin-pose-feasibility.py \
  > tmp/141-isfixed-chin-pose-feasibility.log 2>&1
```

Comet: <https://www.comet.com/liblaf/apple/daf39004804e482ebda307a0a922ffac>.

## Results

The area-weighted 27-vertex chin fit estimates 10.023102657 degrees of
rotation and 5.96800277 mm translation. It leaves 0.182875 mm weighted RMS
chin residual and 0.397076 mm maximum residual.

The corrected boundary contains 27,036 `IsFixed` vertices, including 6,145
mandible vertices, and 2,249 tetrahedra whose four vertices are `IsFixed`.
Those cells cannot change during a free-DOF solve. At the full chin fit, 168
such cells invert and their minimum J is -79.351147.

At 1% of the fitted pose, rotation is 0.100231 degrees and translation is
0.059680 mm. Its immutable-cell minimum J is 0.219439, with zero inversions.
The strict positive-J limit is 1.281004% of the fit (0.128396 degrees,
0.076450 mm). Requiring J >= 0.1 reduces the limit to 1.152956% (0.115562
degrees, 0.068808 mm).

## Use in the inverse

Initialize at 1% of the chin fit. Before CCD or an expensive forward solve,
apply the candidate rigid transform only to `IsFixed ∩ Mandible` and reject it
unless the minimum J over the 2,249 all-fixed cells is at least 0.1. This gate
must apply to every optimized pose, regardless of the requested 1-degree or
1-mm carry increment: those increments describe a step size and do not make a
pose beyond the immutable-cell limit feasible.

Harmonic carry is still useful for free and mixed boundary cells, but it cannot
repair an inverted tetrahedron whose four vertices are prescribed.

## Artifacts

The complete input hashes, chin pose, samples, and limits are in
`data/isfixed-chin-pose-feasibility-001/receipt.json`. The audit script is
`src/141-audit-isfixed-chin-pose-feasibility.py`.
