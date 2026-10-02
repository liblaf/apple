# Layered attachment benchmark: fitting one compliance does not identify a force path

## Result

A uniform middle-layer modulus can reproduce one scalar displacement from a patch with localized attachments almost exactly, but it does not reproduce the same patch under a changed loading direction or location. At the calibration load, the fitted uniform model matched the localized-attachment displacement to 4.17e-6 relative error. Under the held-out y-directed load, it underpredicted compliance by 41.55%; moving the x-load over the attachment band produced a 7.72% error. Even at the fitted load, the full-surface displacement fields differed by 0.00391 mm RMS.

This is a controlled synthetic mechanics result. It shows that one scalar external displacement is insufficient to distinguish a distributed attachment path from an effective continuum stiffness. It does not validate a facial layer, attachment geometry, sliding law, fiber direction, or material parameter.

![Calibration, holdouts, and mesh-refinement comparison](../data/10-benchmark/comparison.png)

The surface maps below show why a matched scalar is weaker evidence than a matched field. The dashed line is the center of the modeled attachment band at x = 10 mm. All three maps use the same x-directed 2 mN load centered at x = 15 mm and the same color scale.

![Surface displacement under the calibration load](../data/10-benchmark/surface-response.png)

## Controlled protocol

The model is a regular 20 × 10 × 2.5 mm patch. Its volume has three layers separated at z = 1.5 and 2.0 mm. The bottom surface at z = 0 is fixed in all three displacement components. All other volume boundaries are free. A Gaussian traction distribution on the top surface is normalized to a known total force. The reported loaded-patch displacement is the force-weighted mean displacement component parallel to that force.

All quantities use a consistent mm–N system:

| Quantity | Unit |
| --- | ---: |
| Length and displacement | mm |
| Force | N |
| Young's modulus | N/mm² = MPa |
| Energy | N·mm |
| Link stiffness | N/mm |

The lower 1.5 mm and upper 0.5 mm volume layers use isotropic Stable Neo-Hookean elasticity with E = 0.003 MPa and ν = 0.4. The middle 0.5 mm layer uses either E = 0.003 MPa in the homogeneous control or E = 0.0003 MPa in the mobile-layer and attachment cases. The top skin is a Koiter metric membrane with plane-stress Lamé parameters derived from E = 0.05 MPa, ν = 0.46, and thickness h = 0.5 mm unless a named control changes them. There is no bending term.

The attachment case adds two opposing families of tension-only axial links across the middle layer. Each link spans 0.5 mm vertically and 2.5 mm laterally in the x–z plane. Link weights follow a Gaussian band in x; their stiffnesses are normalized so that the sum is 0.3 N/mm at every mesh refinement. The `localized_attachment` band is centered at x = 10 mm and the `shifted_attachment` band at x = 5 mm. Each link begins at its rest length. These are illustrative discrete force paths, not digitized retinacula, ligaments, SMAS fibers, or dermal insertions.

Four loads define calibration and prediction:

| Load | Direction | Center x | Total force | Role |
| --- | ---: | ---: | ---: | --- |
| `calibration_x_right` | x | 15 mm | 0.002 N | Fit one scalar compliance |
| `holdout_y_right` | y | 15 mm | 0.002 N | Direction holdout |
| `holdout_x_over_band` | x | 10 mm | 0.002 N | Location holdout |
| `holdout_x_larger` | x | 15 mm | 0.006 N | Force-magnitude holdout |

The calibration varies only the middle-layer Young's modulus of a model with no links. Brent's method searches 0.0003–0.03 MPa until its `calibration_x_right` displacement matches the localized-attachment case. It found E = 0.000898914 MPa. All subsequent `matched_uniform` predictions use that value without refitting.

The run executed 52 equilibrium solves: four initial cases, nine scalar-calibration trials, one saved matched case, fifteen case/load holdouts, three stiffness/thickness controls, twelve fixed-parameter mesh checks, and eight central-difference sensitivity solves. Thirty-five main states were saved; the seventeen calibration and sensitivity evaluations are retained in `all-solves.json`.

## Scalar calibration and held-out predictions

The first table gives the force-weighted displacement along each applied force on the r = 1 mesh. The `homogeneous` model has no compliant middle layer. The `mobile_layer` introduces that layer without links. The two attachment cases differ only in attachment-band location. The `matched_uniform` model has no links and uses the single calibrated middle-layer modulus.

| Case | Calibration x, 2 mN (mm) | Holdout y, 2 mN (mm) | x over band, 2 mN (mm) | Calibration location x, 6 mN (mm) |
| --- | ---: | ---: | ---: | ---: |
| Homogeneous | 0.036601 | 0.047258 | 0.031409 | 0.109650 |
| Mobile layer | 0.082729 | 0.116051 | 0.074994 | 0.247857 |
| Localized attachment | 0.049313 | 0.112427 | 0.046672 | 0.147721 |
| Shifted attachment | 0.060921 | 0.112971 | 0.045489 | 0.182166 |
| Matched uniform | 0.049313 | 0.065710 | 0.043068 | 0.147721 |

The calibrated scalar agrees while the deformation field does not:

| Load | Localized attachment (mm) | Matched uniform (mm) | Compliance error | Full-surface vector RMS error (mm) |
| --- | ---: | ---: | ---: | ---: |
| Calibration x, 2 mN | 0.049312768 | 0.049312974 | +0.000417% | 0.003905 |
| Held-out y, 2 mN | 0.112427327 | 0.065710393 | −41.553% | 0.036379 |
| x over attachment band, 2 mN | 0.046672131 | 0.043067891 | −7.722% | 0.006009 |
| Calibration location x, 6 mN | 0.147721448 | 0.147721110 | −0.000229% | 0.011609 |

The direction holdout is the strongest discriminator because the illustrative links run obliquely in the x–z plane: the two models transmit the y load very differently. Moving the load relative to the band also reveals a mismatch. The 6 mN scalar remains matched because this small-deformation response is nearly proportional over the tested range; force magnitude alone did not distinguish the mechanisms here. Its surface-field error increased to 0.01161 mm, so agreement in one integrated displacement still did not imply agreement in spatial response.

Changing only the attachment-band center also changed the calibration displacement from 0.049313 to 0.060921 mm. That is evidence that the synthetic response depends on force-path location. It is not evidence that either chosen band represents a real facial attachment.

## Local two-load identifiability check

Around the localized-attachment case, the script perturbed the middle-layer modulus `E_m` and attachment-stiffness multiplier `k_a` by ±2% in log space. For the calibration x compliance and held-out y compliance, it estimated

$$
S_{ij}=\frac{\partial\log d_i}{\partial\log\theta_j} =
\begin{bmatrix}
-0.189388 & -0.133637\\
-0.602291 & -0.008876
\end{bmatrix},
$$

where rows are (`d_x`, `d_y`) and columns are (`E_m`, `k_a`). Increasing either stiffness reduces displacement, hence the negative entries. The y response is strongly sensitive to the continuum modulus and nearly insensitive to the attachment multiplier at this operating point; the x response contains sensitivity to both.

The singular values are 0.633305 and 0.124439, giving a condition number of 5.089. One scalar observation has maximum rank one and cannot locally identify two independent parameters. These two synthetic load directions give a full-rank local matrix, so they can distinguish the two parameter directions near the selected point in this noise-free model. This does not establish global uniqueness, robustness to measurement noise, or identifiability after adding thickness, prestress, geometry, friction, or more spatial parameters.

## Thickness–modulus ambiguity and absolute scale

The regional-thickness control uses a triangle field with area-weighted mean 0.5 mm and sampled range 0.3551–0.6449 mm. The `eh_equivalent` case halves that field and doubles skin `E` at fixed ν. Its maximum displacement difference from the original field is 4.69e-15 mm, numerical roundoff:

| Case | Mean `h` (mm) | Skin `E` multiplier | Loaded displacement (mm) | Surface RMS (mm) |
| --- | ---: | ---: | ---: | ---: |
| Regional thickness | 0.500 | 1 | 0.050399674 | 0.038534751 |
| Same `E h` | 0.250 | 2 | 0.050399674 | 0.038534751 |

At fixed Poisson ratio, both membrane Lamé parameters scale with `E`, and the implemented membrane energy is multiplied by `h`. The membrane therefore depends on the product `E h`; this experiment cannot identify `E` and `h` separately from membrane deformation. The test changes the membrane weight only. It does not create geometric thickness, through-thickness stress, or bending stiffness.

As a separate scale-bearing control, doubling every internal stiffness coefficient while keeping the known force fixed reduced the localized model's displacement from 0.049313 to 0.024666 mm, approximately one half. This does not contradict common-energy-scale invariance in displacement-only, internally driven equilibrium: the imposed external force is not scaled and supplies an absolute scale.

## Mesh check with frozen r = 1 parameters

The mesh study increases volume resolution while preserving geometry, material coefficients, load normalization, the total 0.3 N/mm link stiffness, and the uniform modulus calibrated only at r = 1. No coefficient is re-fitted on r = 2 or r = 3.

| Refinement | Points | Tetrahedra | Attachment links | Mobile x (mm) | Attached x (mm) | Uniform x (mm) | Attached y (mm) | Uniform y (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 180 | 576 | 70 | 0.082729 | 0.049313 | 0.049313 | 0.112427 | 0.065710 |
| 2 | 1,071 | 4,608 | 234 | 0.083005 | 0.049890 | 0.049512 | 0.115383 | 0.067511 |
| 3 | 3,250 | 15,552 | 494 | 0.083461 | 0.050199 | 0.049859 | 0.117010 | 0.068635 |

From r = 1 to r = 3, the displayed displacements changed by 0.88%–4.45%, depending on case and load. The fitted x agreement becomes −0.68% at r = 3 because the r = 1 coefficient is frozen. The held-out y error remains large: −41.55%, −41.49%, and −41.34% on r = 1, 2, and 3. The qualitative mechanism mismatch is stable across these three meshes, while the exact displacement is not claimed to be fully grid-converged.

All 52 solves reported `primary_success`. Iteration counts ranged from 70 to 345. Residual force norms ranged from 4.28e-10 to 3.64e-9 N; relative force-balance error remained below 1.68e-6. Every tetrahedron retained positive orientation, with minimum deformation determinant 0.95967 across the run.

## What this benchmark does and does not support

The benchmark supports three limited conclusions:

1. A uniform continuum coefficient can absorb the effect of a different force path for one chosen scalar response.
2. Independent load directions, locations, and full-field deformation can expose that compensation.
3. The current membrane cannot separate thickness from modulus when only their product enters the energy.

The middle layer is a compliant, fully bonded volumetric continuum. It permits distributed shear deformation between its upper and lower surfaces, but it is not a contact or explicit sliding interface. There are no independent coincident surfaces, normal gap, nonpenetration barrier, adhesion, friction, or measured traction–separation law. Calling this result a validation of “facial sliding” would therefore overstate the model.

The oblique x–z link families, their 2.5 mm lateral span, Gaussian density, band centers, and total stiffness are selected synthetic hypotheses. They were not measured from dissection, histology, MRI, ultrasound, DTI, or a same-donor anatomy. The patch has no facial geometry, muscles, fascia, retaining ligaments, fat compartments, dermal insertion footprints, prestress, gravity, contact, activation, viscoelasticity, or expression target. No anatomical or physiological validation follows from the reported errors.

The next informative use is a measured regional patch test with known load and internal or through-depth motion, followed by a held-out direction or location. Adding more whole-face parameters before obtaining those independent observations would make the same compensation problem larger.

## Reproducibility and artifacts

The final normal run exited successfully and is recorded at [Comet experiment `23af00c2239741c6a0b64454f8dff528`](https://www.comet.com/liblaf/apple/23af00c2239741c6a0b64454f8dff528). Cherries records:

| Field | Value |
| --- | --- |
| Name | `Layered attachment mechanics and identifiability final` |
| Tags | `biomechanics`, `attachments`, `skin-thickness`, `known-load`, `validation` |
| Entrypoint | `exp/2026/09/12/layered-attachment-benchmark/src/10-run-benchmark.py` |
| Git SHA | `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` |
| Start | `2026-09-12 17:30:40.287549+08:00` |
| End | `2026-09-12 17:30:50.541341+08:00` |
| Configuration | `maximum_refinement=3`, `smoke=false`, output `10-benchmark` |
| Runtime | Python 3.14.6; Torch 2.12.0+cu130; Warp 1.14.0; NumPy 2.4.6 |
| GPU | NVIDIA GeForce RTX 4090 |

The Cherries interval excludes some process startup and shutdown work; the complete process took about 19 seconds. The log contains a warning that Comet was imported after Torch, but the requested metrics, parameters, source files, and URL were recorded. Automatic Comet environment details and Git-patch capture were disabled to avoid the previously observed shutdown delay. Git did not auto-commit.

The expanded derivative checks exposed invalid mixed CPU/CUDA Koiter launches. Material creation and kernel launches now explicitly use the geometry device. The 200-Tape reproducer and formerly failing test order pass after the fix; struct caching also removes redundant native layouts, without being claimed as the crash cause. The full `tests/` suite passes **26 tests** at seeds 103, 1 and 17; an independent final run verified seed 103. Ruff, type checking and diff checks pass. The [device-fix receipt](../data/verification/koiter-device-fix-receipt.md) and [final test log](../data/verification/final-pytest-seed-103.log) preserve the evidence. An unrestricted repository-root pytest collection additionally discovers the unchanged legacy `benches/test_aggregation.py`, which imports unavailable Equinox; that separate collection failure is recorded in [its log](../data/verification/root-collection-missing-equinox.log). The benchmark itself completed all 52 solves and exited successfully after the final source was frozen.

Runnable command from the experiment directory:

```bash
cd exp/2026/09/12/layered-attachment-benchmark
COMET_AUTO_LOG_ENV_DETAILS=false \
COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME="Layered attachment mechanics and identifiability final" \
CHERRIES_TAGS="biomechanics,attachments,skin-thickness,known-load,validation" \
.venv/bin/python src/10-run-benchmark.py
```

For a one-case local smoke without Comet:

```bash
cd exp/2026/09/12/layered-attachment-benchmark
DEBUG=1 \
CHERRIES_NAME="Layered patch smoke" \
.venv/bin/python src/10-run-benchmark.py --smoke true --output 00-smoke
```

The final evidence is under [`data/10-benchmark`](../data/10-benchmark/):

- [`summary.json`](../data/10-benchmark/summary.json): protocol, 35 saved main runs, scalar calibration, holdout errors, `E h` check, and sensitivity matrix.
- [`all-solves.json`](../data/10-benchmark/all-solves.json): all 52 equilibrium receipts, including calibration and sensitivity evaluations.
- [`protocol.json`](../data/10-benchmark/protocol.json): units, model statement, runtime, and SHA-256 provenance.
- [`comparison.png`](../data/10-benchmark/comparison.png) and [`surface-response.png`](../data/10-benchmark/surface-response.png): the figures embedded above.
- `*.vtu`: volume mesh, layer/material arrays, fixed mask, and solved displacement for each saved case.
- `*.vtp`: top-surface force, thickness, and displacement arrays; `--links.vtp` files contain the attachment line geometry and per-link stiffness.
- [`source/10-run-benchmark.py`](../data/10-benchmark/source/10-run-benchmark.py), [`source/patch_model.py`](../data/10-benchmark/source/patch_model.py), [`source/_koiter.py`](../data/10-benchmark/source/_koiter.py), and [`source/_fiber_spring.py`](../data/10-benchmark/source/_fiber_spring.py): exact source snapshots consumed by the run.
- [`logs/10-run-benchmark.log`](../logs/10-run-benchmark.log): solver progress and the final Comet summary.

The source hashes recorded in `protocol.json` are:

| Snapshot | SHA-256 |
| --- | --- |
| `10-run-benchmark.py` | `c51df15d6f54008ba668b3401c234ab17041098527317833ba6dec374bfb1867` |
| `patch_model.py` | `3e152f9af2dde51cb164a4d868591cc6f2c7830d61a9fac5eeed62f66a01c681` |
| `_koiter.py` | `8a7d0f54cf940d16ded033745881eaf06eed48b0b95e60ed701e71be763de4dd` |
| `_fiber_spring.py` | `b710039d7d509950e8f482ecf89420383939c507558dd30632624f62ccb8b73b` |

An earlier normal run under [`data/09-initial-run`](../data/09-initial-run/) completed the same 52 numerical solves and wrote its outputs, but it did not finish the recording lifecycle cleanly. A duplicate Local output registration raised `FileNotFoundError`, after which Comet remained in its environment/Git-patch flush. Those numerical values agree with the clean rerun to floating-point noise, but `data/10-benchmark`, the final log, and the Comet URL above are the authoritative run receipt. The clean rerun disabled the two slow automatic Comet captures and had no Local error.

The intermediate successful run before the device correction is preserved in `data/09b-before-device-fix`; the run immediately before the final type annotation is in `data/09c-before-type-annotation`. Each retains its source snapshot and copied log. Only `data/10-benchmark` matches all four current source hashes and is the final evidence.
