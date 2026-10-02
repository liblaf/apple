# Shared-face tensor active-stress small study

## Purpose

This CPU-only study uses two tetrahedra sharing one triangular face. Both cells contain active muscle mixed with fat, at muscle fractions 0.3 and 0.8. It checks a finite PSD tensor-stress cap against a uniform compatible target and a deliberately incompatible same-edge target. It does not model fibers, skin, a face mesh, or physiological calibration.

## Execution and evidence

The completed v4 receipt is [`data/12-small-model-study-v4/summary.json`](../data/12-small-model-study-v4/summary.json). It ran through Cherries with the noncommitting profile, using the direct project virtual environment and Comet run [fcd1ffedf2234eb0888dd6c7d6332a08](https://www.comet.com/liblaf/apple/fcd1ffedf2234eb0888dd6c7d6332a08). The durable user unit exited successfully after writing 12 states, meshes, `states.npz`, the comparison image, and provenance.

The archived receipt has one stale interpretive sentence calling `Qref` the face-pilot cap. The numerical files are unchanged; the correction and the authoritative frozen face setting are recorded in [`interpretation-correction.json`](../data/12-small-model-study-v4/interpretation-correction.json) and [`data/14-frozen-face-settings/summary.json`](../data/14-frozen-face-settings/summary.json). The v5 figure derivative preserves that receipt byte-for-byte and carries the same correction in [`data/13-small-study-comparison-v5/interpretation-correction.json`](../data/13-small-study-comparison-v5/interpretation-correction.json); its manifest names that correction as superseding the copied `cap_interpretation` prose.

The current source has the corrected wording in [`src/12-small-model-study.py`](../src/12-small-model-study.py). The archived source copied into the v4 output remains the exact executed source.

## Fixture and checks

The vertices are constrained by six DOFs—vertex 1 at the origin, vertex 2 on the x-axis, and vertex 3 in the xy-plane. The recorded infinitesimal rigid-mode constraint matrix has rank 6. A nonzero nodal load was solved both passively and with `Q=0`; both accepted forward receipts report a gradient infinity norm below `1e-8`, and their positions agree exactly.

For each cell, passive energy is `fraction × stable(muscle) + (1 − fraction) × stable(fat)`. Tensor stress is a separate symmetric PSD tensor for each cell and enters as `fraction × 0.5 Q:(FᵀF−I)`. The study uses `Qref = 3μ = 0.030201342281879193 MPa` and compares it with `10Qref = 0.30201342281879195 MPa`.

## Results

Target-shape RMS is in fixture-length units. “Balance” is the locally force-balanced, PSD-projected, capped candidate; it is not a global reachability claim. “Bounded search” rows exhausted their fixed Nelder–Mead budget where marked and are candidates, not optima. The `×10 E` active-strain row is an authority sensitivity, not a matched stress-budget comparison.

| Target | Method | Target-shape RMS | Cap saturation | Outer status |
| --- | --- | ---: | --- | --- |
| Uniform compatible `Fa` | Qref balance | 0.04465018 | both cells | analytic candidate |
| Uniform compatible `Fa` | 10Qref balance | 4.6587e-10 | neither cell | analytic candidate |
| Uniform compatible `Fa` | active strain | 0.00494956 | — | accepted forward solve |
| Uniform compatible `Fa` | active strain, 10× muscle E | 0.000573852 | — | accepted forward solve |
| Uniform compatible `Fa` | Qref bounded search | 0.02171344 | both cells | fixed-budget candidate |
| Uniform compatible `Fa` | 10Qref bounded search | 4.6587e-10 | neither cell | fixed-budget candidate |
| Incompatible shared edge | Qref balance | 0.04093833 | both cells | analytic candidate |
| Incompatible shared edge | 10Qref balance | 0.02041095 | neither cell | analytic candidate |
| Incompatible shared edge | active strain | 0.02838401 | — | accepted forward solve |
| Incompatible shared edge | active strain, 10× muscle E | 0.03373646 | — | accepted forward solve |
| Incompatible shared edge | Qref bounded search | 0.03270992 | both cells | fixed-budget candidate |
| Incompatible shared edge | 10Qref bounded search | 0.00561227 | neither cell | fixed-budget candidate |

The compatible analytic candidate directly distinguishes the caps: the `Qref` cap clips every reported eigenvalue and leaves RMS `0.04465018`, whereas the `10Qref` candidate needs eigenvalues from `0.05650031` to `0.11691297 MPa`, remains below `0.30201342 MPa`, and reaches RMS `4.6587e-10`. This is the limited finite-cap feasibility evidence supporting the frozen face setting `10Qref`; it is not a physiological calibration or a claim that the small fixture predicts face behavior.

The incompatible target requests the same rest-unit shared edge at lengths 0.75 and 0.90. One nodal configuration cannot realize both. Its minimax mismatch bound is therefore `(0.90 − 0.75)/2 = 0.075` rest-edge-length units. That bound concerns the requested target geometry, not a solver failure or an endpoint inversion criterion.

For the conflicting-command rows, the reported nodal target-shape RMS is measured against the feasible least-squares nodal compromise, whose shared edge is 0.825. It is a different quantity from mismatch to the two incompatible local requests. An RMS below 0.075 therefore does not violate the shared-edge bound. Increasing muscle stiffness cannot remove that incompatibility; in this fixture the 10× active-strain result actually moves farther from the chosen nodal compromise.

![Receipt-derived contraction and target-error comparison](../data/13-small-study-comparison-v5/contraction-and-target-error.png)

[PDF figure](../data/13-small-study-comparison-v5/contraction-and-target-error.pdf)

## Reproducibility

The final command recorded by Cherries was:

```bash
CHERRIES_NAME='tensor-active-stress-small-model-final' \
CHERRIES_TAGS='tensor-active,two-tet,no-skin,finite-stress' \
COMET_AUTO_LOG_GIT_METADATA=false COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python \
  src/12-small-model-study.py --output-dir data/12-small-model-study-v4
```

The post-hoc figure reads only the final v4 summary and performs no equilibrium or fitting calculation. Its input and output hashes are in [`data/13-small-study-comparison-v5/manifest.json`](../data/13-small-study-comparison-v5/manifest.json).
