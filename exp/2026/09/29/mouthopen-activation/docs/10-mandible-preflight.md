# MouthOpen: mandible pose and mesh feasibility

The MouthOpen target and a fresh mandible pose estimate are prepared. Activation fitting has not started: the accurate chin-based pose forces 173 fully prescribed tetrahedra to invert. A pose constrained to keep these cells positive fits the chin substantially worse. The next experiment depends on whether to repair the jaw–cranium interface or accept that weaker pose as an exploratory initialization.

![Neutral, MouthOpen target, and conflicting prescribed cells](../data/20-preflight/mouthopen-preflight-preview.png)

[Full-resolution figure](../data/20-preflight/mouthopen-preflight.png)

## Target and pose estimate

This uses the same historical no-skin fixture as the smile activation experiments: `exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture`. Its `MouthOpen` point-data displacement agrees bit-for-bit with the transferred expression bundle on all 15,299 fitting-surface vertices. The bundle's original neutral coordinates also agree exactly with the fixture. No pose or equilibrium from the later modified neutral was inherited.

An area-weighted proper rigid fit aligns the previously selected 27-vertex chin patch from the original neutral to its MouthOpen target. The provisional pivot comes from the registered mandible landmarks. The rotation-vector/pivot convention is `(x - pivot) @ R.T + pivot + translation`.

| Estimate | Rotation magnitude | Translation norm | Chin RMS |
| --- | ---: | ---: | ---: |
| No jaw motion | 0° | 0 mm | 23.3920 mm |
| Full rigid chin fit | 9.5337° | 6.4797 mm | 0.2133 mm |
| Hinge-only chin fit | 11.6061° | 0 mm | 4.9548 mm |
| Rigid fit constrained by prescribed-cell orientation | 7.6253° | 0.3711 mm | 8.9477 mm |

The unconstrained pose is `[0.1653980972, -0.0116702661, 0.0139467389, 0.0003865692, -0.0064528198, 0.0004454297]` in radians/metres, about pivot `[1.406947255, 2.19754493, 0.00298870087]` metres. Translation is expressed about that pivot, not at the world origin.

The chin is soft tissue, so this is a geometric pose estimate, not an independently measured bone motion. The patch is nearly planar (Kabsch covariance condition number 653.8). Its small fitting residual alone does not establish jaw-pose accuracy. The target itself is a transferred blendshape, not a measurement of this subject's jaw.

## Prescribed-cell obstruction

`IsFixed` remains the sole original FEM clamp. `FixedMask` matches it on all three components. `GroupName` maps group 28 to `Mandible`; 6,145 of its 7,510 vertices are fixed and receive the rigid displacement. Other fixed vertices retain their neutral positions. There are 27,036 fixed vertices and 2,249 tetrahedra with all four vertices fixed.

At the full rigid estimate, 173 of those tetrahedra have nonpositive signed volume; minimum `det(F)` is **−85.0408**. All 173 mix Cranium and Mandible vertices, have `FatFraction = 1`, and are inactive. Their vertex-group compositions are 71 cells with three cranium/one mandible vertices, 54 with two/two, and 48 with one/three. They lie across the moving-jaw/fixed-cranium interface, as shown in red.

Because every vertex of these cells is prescribed, neither a forward solve nor activation optimization can change their determinants at this pose. This is independent of smoothness strength and the free-node solver's convergence. Along the tested path that scales the rotation vector and pivot translation together, the first inversion occurs near **1.2062%** of the full pose. That is a diagnostic of this particular path, not a bound on every possible jaw trajectory.

## Constrained-pose probe

A second CPU experiment minimizes the same weighted chin error while imposing `det(F) >= 0.1` on all 974 fully fixed tetrahedra that mix moving and stationary vertices. The 0.1 margin is an explicit numerical guard against near collapse, not a measured material limit. It does not change `IsFixed`, the mesh, or any material.

SLSQP converged from both neutral and a 0.1% pose seed to the same local solution: 7.6253° rotation, 0.3711 mm translation, 8.9477 mm chin RMS, and minimum prescribed-cell determinant 0.1. This avoids the prescribed-cell inversions but leaves a large chin mismatch. Two agreeing initializations do not prove a global optimum. No free-tissue equilibrium, contact, or anatomical-pose validation was performed.

The accurate geometric estimate is therefore unsuitable for direct prescription on this mesh. A useful next step is to inspect and repair the jaw–cranium tissue/interface topology with the intended anatomy preserved. The alternative is an explicitly exploratory activation fit initialized with the constrained pose, accepting that activation may compensate for about 9 mm of chin mismatch. Silently relaxing `IsFixed` would change the model and is not part of either completed probe.

## Outputs and verification

- [Pose estimate and source hashes](../data/10-mandible/pose.json)
- [Prescribed-cell audit, cell IDs, materials, and pose-fraction sweep](../data/10-mandible/audit.json)
- [Constrained-pose candidates and iteration histories](../data/15-feasible-pose/result.json)
- [Rendering manifest](../data/20-preflight/manifest.json)
- [Prepared CPU arrays](../data/10-mandible/prepared.npz)

Both an independent array audit and the saved preparation agree on the mandible selector, 173 forced inversions, and their mixed-bone/pure-fat composition. All three scripts passed Ruff and formatting checks. Each run exited successfully after Cherries shutdown. The generated preview was visually checked for readable labels, correct target/neutral distinction, the shared camera, and visibility of the red interface cells. No numerical solver sources, historical artifacts, fixed constraints, or Git refs were changed.

## Reproduction

Working directory: `exp/2026/09/29/mouthopen-activation`.

```bash
CHERRIES_NAME='MouthOpen mandible pose and prescribed-cell preflight' CHERRIES_TAGS='mouthopen,mandible,pose,cpu,preflight' .venv/bin/python src/10-estimate-mandible.py
CHERRIES_NAME='MouthOpen chin pose with prescribed-cell orientation constraints' CHERRIES_TAGS='mouthopen,mandible,pose,cpu,feasibility' CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python src/15-fit-feasible-pose.py
CHERRIES_NAME='MouthOpen jaw estimate and prescribed-cell diagnostic' CHERRIES_TAGS='mouthopen,mandible,pose,visualization,cpu' CUDA_VISIBLE_DEVICES='' LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=2 .venv/bin/python src/20-render-preflight.py
```

The output folders must be absent for a new run; use new output paths for reruns. Original stdout is preserved in `tmp/10-estimate-mandible.log`, `tmp/15-fit-feasible-pose.log`, and `tmp/20-render-preflight.log`. Source snapshots and hashes accompany the numerical preparation. All Cherries profiles disable automatic Git commits.

Cherries/Comet runs: [pose preparation](https://www.comet.com/liblaf/apple/adc698586268408897e455a3dbe3f2ec), [constrained-pose probe](https://www.comet.com/liblaf/apple/9b3d8b2f0c6b4108a32519c0489f09aa), and [figure](https://www.comet.com/liblaf/apple/ae2c36d450af46bf8a495c65055c68b8). All recorded repository HEAD `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with pre-existing uncommitted work; the hash is provenance, not a clean-tree claim.
