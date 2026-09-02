# HFP1 full Orbicularis oris from the head-superior view

This is a post-hoc visualization of the saved corrected HFP1 endpoint at inverse step 40. It does not rerun either inverse or forward physics.

## Images

![Full Orbicularis oris in the HFP1 face context](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-context.png)

The red tetrahedral surface is the complete selected Orbicularis oris in the translucent deformed face. The camera is on the anatomical `+Y` axis and looks toward `-Y`; `+Z` points toward the top of the image. This is a true head-superior, parallel-projection view with no deformation exaggeration.

The mouth ring is nearly edge-on from this direction, so its full projection reads as a thick curved band rather than the closed loop seen from the front. No spatial crop was applied.

![Full-muscle determinant diagnostics](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-determinants.png)

The three close-up panels use the same superior camera and the full reference/deformed union bounds. Gray wireframe is the reference muscle; the colored surface is the deformed muscle. Blue and red are the un-clipped global minimum and maximum, with white fixed at zero. Magenta marks surface-visible portions of cells for which both `det(F) < 0` and `det(Ainv) < 0`.

## Exact selection and result

- Source endpoint: `20-hfp1.vtu`, SHA-256 `f93bf583819048b5d81a674c4f409450e3cd1200e0d3811b3dc98811480d53dd`.
- Muscle mapping: `MuscleId = 254` is `Orbicularis oris001_Head_muscles_0`.
- Predicate: `ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5`.
- Selected volume: 10,484 tetrahedra and 3,248 compact points; no spatial crop.
- Saved inverse endpoint: step 40, 41 evaluations, zero recorded forward failures, and zero recorded adjoint failures.
- Inverse status: **not converged**. The fixed 40-step budget ended with `step_limit_smooth_decrease`; the visualization must not be read as a converged inverse solution.

The determinant recomputation over all 10,484 selected tetrahedra gives:

| Diagnostic | Full range | Negative tets | Negative rest-volume fraction |
| --- | ---: | ---: | ---: |
| `det(F)` | -4.4542 to 10.6362 | 29 | 0.4865% |
| `det(Ainv)` | -2.7187 to 7.7470 | 58 | 1.0261% |
| `det(G)` | -0.2267 to 1.5474 | 33 | 0.6004% |

There are 27 double-inverted tetrahedra, representing 0.4561% of the selected reference volume. These are volume-wide counts. A top-down opaque surface render can hide interior cells, so the magenta pixels are a locator, not a complete count by sight.

## Reopen and reproduce

- [Context PNG](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-context.png)
- [Context ParaView state](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-context.pvsm)
- [Determinant PNG](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-determinants.png)
- [Determinant ParaView state](../data/40-hfp1-orbicularis-oris-topdown/40-hfp1-orbicularis-oris-topdown-determinants.pvsm)
- [Final receipt](../data/40-hfp1-orbicularis-oris-topdown/receipt.json)
- [Renderer receipt](../data/40-hfp1-orbicularis-oris-topdown/renderer-receipt.json)
- [Render contract](../data/40-hfp1-orbicularis-oris-topdown/contract.json)
- [Experiment log](../logs/40-hfp1-orbicularis-oris-topdown.log)

The native ParaView renderer is pinned to ParaView 6.1.1. The preparation script checks the original endpoint, source summary, context surface, muscle mapping, topology, scalar statistics, camera, image dimensions, and output identities before completing its receipt.

The local run and receipts completed. Comet recorded the named experiment and metrics, but its final environment/git-patch upload reported incomplete logging because the shared worktree already contained a very large unrelated staged artifact set. The local contract, renderer receipt, final receipt, and PNG/PVSM identities are the authoritative record for this visualization.

```bash
cd exp/2026/08/19/material-physics-group-meeting-assets
CHERRIES_NAME="HFP1 full Orbicularis oris superior view" \
CHERRIES_TAGS="human-face,HFP1,orbicularis-oris,top-down,paraview,step-40" \
uv run python src/40-hfp1-orbicularis-oris-topdown.py
```
