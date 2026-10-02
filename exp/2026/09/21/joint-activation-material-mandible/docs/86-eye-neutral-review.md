# Eye-inclusive neutral endpoint review

## Purpose

Independently inspect the converged fixed-eye neutral forward endpoint, without
changing its collision geometry or constitutive reference. This review checks
the saved FEM displacement against the raw full cranium, mandible, and source
eye triangles, then produces true-scale comparisons and ParaView assets.

## Command

```bash
cd exp/2026/09/21/joint-activation-material-mandible
CHERRIES_NAME='Eye neutral endpoint independent review' \
CHERRIES_TAGS='eyes,collision,neutral,review,paraview' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/86-review-eye-neutral.py
```

Cherries/Comet: [2f7903afc89d4b878a6010e35e77429f](https://www.comet.com/liblaf/apple/2f7903afc89d4b878a6010e35e77429f).

## Results

The endpoint from `eye-neutral-forward-002` passed the independent audit.
IPC reports no intersections, and VTK all-contact tests report zero
pure-soft triangle pairs with each of the source cranium, mandible, and eyes.
The FEM fixed-boundary maximum was `1.735e-18 m` (roundoff allowance
`4.106e-15 m`); appended eye displacement was exactly zero.

For containment only, the review creates an exact-coordinate-weld proxy from
the immutable source collider: 1,298 raw vertices become 1,284 vertices while
all 2,560 mapped source faces remain. Its two components are watertight. All
199,059 nonrigid FEM nodes are outside both eyes. The smallest endpoint
clearances are 92.01 and 92.43 micrometres. The IPC collider remains the raw
1,298-vertex / 2,560-triangle source mesh.

The final state has no inverted tetrahedra, with `det(F)` in
`[0.3741998, 1.6000546]`. These facts verify the saved endpoint's collision
and volume diagnostics; they do not validate anatomy or establish global
mechanical stability.

Relative to the previous adopted neutral, the unweighted RMS displacement over
the 15,299 observed surface nodes is `0.033334 mm`; its maximum is
`0.232678 mm`. The maximum over all FEM nodes is `0.680581 mm`.

## Outputs

`data/eye-neutral-forward-review-006/` contains:

- `00-solver-and-volume-trends.png`, true-scale original/adopted/final
  comparisons, periorbital maps, and full rigid context images.
- `eye-neutral-deformed-volume.vtu`, `eye-neutral-deformed-skin.vtp`, and
  `fixed-rigid-obstacles.vtm` for ParaView.
- `eye-containment-proxy.vtp` plus its exact-welding map and the hash-bound
  `summary.json` receipt.

The visual bundle compares the original constitutive reference, the adopted
no-eye neutral, and the converged eye-inclusive endpoint at common camera
scale. It does not amplify displacements.
