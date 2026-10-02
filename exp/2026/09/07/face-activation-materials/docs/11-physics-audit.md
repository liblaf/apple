# Face inverse-physics audit and comparison contract

## Decision

Use the prepared June face volume, add the later hard-fixed extraction-cut
constraint, and use the corrected all-`IsFace` plane-stress skin with zero
prestrain. Keep the measured `Smile` displacement as the target, but do not
derive any material, rest metric, activation domain, or fiber estimate from
that target.

The primary comparison is a matched 2 x 2:

| Material model | Activation model | Role |
| --- | --- | --- |
| volume mixture only | `Raw6` | historical high-capacity diagnostic |
| volume mixture + corrected skin | `Raw6` | isolates the skin contribution under the historical coordinates |
| volume mixture only | `Region5` | constrained fiber-free comparison without a membrane |
| volume mixture + corrected skin | `Region5` | recommended complete pilot |

Every case must use the same named expression-muscle mask, target, fixation,
initial zero displacement, initial zero activation, solver tolerances, step
budget, and saved-state schedule. The no-skin rows are ablations, not proposed
anatomy. If `Region5` underfits, the next experiment should be per-cell `G5-S`
with the same determinant-one map and a shared-face smoothness term. Do not
enlarge the material matrix before this comparison identifies whether the
error comes from activation capacity or the membrane.

## Pinned fixture

The source volume is
`exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu`
(SHA-256 `8131d694...`). It is the `InFaceConvex` subset of the Melon head mesh.
It has 228,660 points, 1,146,517 oriented tetrahedra, 64,042 boundary points,
and 128,172 boundary triangles. `AponeurosisFraction`, `FatFraction`, and
`MuscleFraction` sum to exactly one in every tetrahedron. The preparation
logic and field definitions are in
[`_human_face_mesh.py`](../../../../06/17/human-face-smile-prestrain-v2/src/_human_face_mesh.py).

The target remains the actual point-data vector `Smile`. The loss domain is
exactly `IsFace & isfinite(Smile)`, giving 15,302 points and zero overlap with
the final fixed set. NaN target entries are excluded by the mask rather than
treated as observations. The target RMS displacement over this domain is
5.31014 mm.

The June mesh originally fixed 27,036 inherited points but left the artificial
extraction cut free. The fixture reproduces the later conservative policy:
a full-boundary triangle is a cut triangle when it touches a point whose
mapped `GroupId == -1`, and every vertex incident to those triangles is fixed
to exact zero. This identifies 13,165 cut triangles and 6,980 incident
vertices; 380 were already fixed and 6,600 are added, for 33,636 fixed
vertices and 100,908 fixed displacement DoFs. The authoritative earlier
implementation is
[`20-inverse-plane-stress-screen.py`](../../../../08/18/human-face-smile-plane-stress-skin/src/20-inverse-plane-stress-screen.py#L409-L620).
The cut constraint is a conservative boundary-condition bracket, not anatomical
ground truth.

The corrected skin source is
`exp/2026/08/18/human-face-smile-plane-stress-skin/data/10-corrected-baseline/skin-isface-e0200-p000.vtp`
(source SHA-256 `4c7ddce8...`). It contains 15,299 points and 29,899 triangles;
every triangle has three `IsFace` vertices. `GlobalPointId` maps every skin
point exactly to a volume point and the coordinates match exactly. This is a
facial region-of-interest membrane, not a complete epidermis.

The generated fixture is:

- `data/10-fixture/volume.vtu`: volume, cut fields, named activation fields,
  and rest-geometry fiber estimates;
- `data/10-fixture/skin.vtp`: corrected homogeneous plane-stress skin;
- `data/10-fixture/summary.json`: exact inputs, counts, label exclusions,
  formulas, and per-region confidence;
- `data/10-fixture/reconstruction-receipt.json`: source-snapshot and semantic
  reconstruction evidence after an interrupted alias-only regeneration.

The current volume is semantically equal to a fresh reconstruction across its
geometry and every point-, cell-, and field-data array. Its current file SHA
is `6c0cab0d...`; the first VTK serialization had SHA `d601d278...` and could
not be recovered byte-for-byte. The skin file stayed byte-identical at
`79eed2a5...`. The receipt deliberately makes no exact original-volume byte
reconstruction claim.

## Material model

The volume uses the existing additive fraction-weighted potentials:

| Constituent | Model | E (MPa) | nu | Active |
| --- | --- | ---: | ---: | --- |
| fat | stable Neo-Hookean | 0.003 | 0.49 | no |
| muscle | stable Neo-Hookean active strain | 0.030 | 0.49 | yes |
| aponeurosis | stable Neo-Hookean | 0.100 | 0.35 | no |

The three-dimensional Lamé conversion is
`lambda = E*nu / ((1+nu)*(1-2*nu))` and `mu = E / (2*(1+nu))`.
The volume builder is in
[`_human_face_forward.py`](../../../../06/17/human-face-smile-prestrain-v2/src/_human_face_forward.py#L84-L142).

The skin uses Koiter membrane energy with `E = 0.2 MPa`, `nu = 0.49`, and
thickness 0.001 m. Its two-dimensional plane-stress constants are
`lambda = E*nu / (1-nu^2)` and `mu = E / (2*(1+nu))`. `ActivationInv` is
exactly zero and the energy uses the fixed original reference area. Reusing
the June skin builder would be wrong because that builder applied the 3-D Lamé
conversion to the membrane.

Do not use HFP1 as the clean material baseline. HFP1 sets a low skin-stiffness
region from `TargetArea/RestArea` and computes c020 prestrain from the same
target. Its 40-update result reached 1.44186 mm target RMS, but retained 10
folded skin triangles and 58 inverted tetrahedra and stopped at the step limit.
It remains useful evidence that a positive stiffness floor suppresses some
corrugation; it does not establish a target-independent constitutive model or
convergence.

## Activation domain and coordinates

The historical mask selected every cell with `MuscleFraction > 1e-6`: 288,235
cells and 1,729,410 independent `Raw6` values. That domain includes labels
such as temporal fascia, the digastric fibrous loop, and the common tendinous
ring, so it is not an acceptable contractile-tissue definition.

The primary fixture selects 120,020 cells (41.64% of the historical domain)
from 35 exact `MuscleId` regions covering occipitofrontalis, levator labii,
buccinator, zygomaticus, risorius, mentalis, orbicularis oculi/oris, depressor
anguli/labii/septi/supercilli, nasalis, procerus, corrugator, levator anguli
oris, and platysma. `summary.json` records every included ID and all 68
historically active labels that were excluded. Both activation models use
this exact mask.

`Raw6` stores the six symmetric components of `A_inv - I` independently for
each selected cell. The implementation forms `G = F A_inv`; see
[`_stable_neo_hookean_active.py`](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py#L20-L46)
and [`_misc.py`](../../../../../../src/liblaf/apple/warp/fem/func/_misc.py#L46-L53).
The raw coordinates are unbounded and do not guarantee a positive-definite
`A_inv`. With 720,120 controls even on the narrowed domain, they are a
capacity reference rather than an interpretable inverse.

`Region5` gives each named region five components of a symmetric trace-free
matrix `H`, bounds its Frobenius norm, and uses `A_inv = exp(H)`. It therefore
keeps `A_inv` symmetric positive definite and `det(A_inv) = 1`, with 175
controls total. Region sharing enforces a piecewise-constant activation field
within each named muscle; the magnitude penalty limits the remaining
non-identifiability. This is a fiber-free mathematical active-strain field,
not a claim about local muscle physiology.

The fixture also includes `ActivationFiber` for later `F` ablations. Linear
regions use the first PCA axis of rest-state tetrahedron centroids weighted by
`Volume * MuscleFraction`. Both orbicularis oculi sides and orbicularis oris
use a PCA plane and per-cell tangent to its fitted ellipse. The sign is
canonical but physically irrelevant to `f outer f`. Confidence is the
eigenvalue-separation heuristic recorded in `ActivationFiberConfidence`; it
ranges from 0.20555 to 0.97870. Depressor septi is the weakest PCA estimate,
and orbicularis oris has ring confidence 0.51132. These fields use rest
geometry only and are not anatomical calibration.

## Runner contract

[`20-run-face-inverse.py`](../src/20-run-face-inverse.py) consumes only
`data/10-fixture` and its VTK fields. The minimal comparison interface is:

```text
--fixture data/10-fixture
--output-dir data/<case>
--method Raw6|Region5
--skin-factor 0|1
--steps <matched budget>
--amax <log-strain bound>
--magnitude <Region5 penalty>
--forward-rtol 1e-5
--forward-atol 1e-12
--adjoint-rtol 1e-6
--gradient-audit true|false
```

Each saved VTU must be a solved equilibrium, carry `RestPosition`,
`Displacement`, `TargetDisplacement`, `DetF`, and the full activation matrix,
and retain the activation/fixation provenance fields. Save the exact config,
source hashes, trace, line-search trials, summary, final state, and fixed-step
snapshots. Compare fit RMS, target projection, activation spectrum,
`det(A_inv)`, minimum `det(F)`, inverted tetrahedra, skin folds, and surface
edge/Laplacian residuals. A lower fit error does not compensate for inversion
or a failed forward/adjoint solve.

No lower-resolution volume artifact was found that preserves this exact outer
surface, target map, material fractions, and named muscle IDs. A coarsened
mesh would be a separate preparation experiment with transfer-error checks;
it should not silently replace the pinned fixture.

## Numerical risks and gates

- `nu = 0.49` makes fat and muscle nearly incompressible and the nonlinear
  system poorly conditioned. Record actual forward and adjoint residuals and
  iterations; solver success at a loose tolerance is not a derivative audit.
- The displacement observations cover only 15,302 facial points. Hidden
  surface motion and internal activation remain underdetermined, even with
  the named domain.
- The current model has no contact. Inspect lip/teeth penetration and skin
  folds directly; pointwise target loss cannot detect all of them.
- The hard-fixed cut can bias the recovered field. Its exact-zero readback is
  required, while its anatomical interpretation must remain limited to a
  boundary-condition bracket.
- Run a central finite-difference directional-gradient check for at least one
  reduced case before interpreting optimizer differences. A fixed update
  budget is an equal-compute comparison, not proof of convergence.
- Stop and reject a case on non-finite states, failed forward or adjoint solve,
  nonpositive activation eigenvalues in `Raw6`, or any new inverted
  tetrahedra. Report folds and inversions even when optimization continues
  for diagnostic purposes.

## Fixture preparation evidence

The fixture was produced from the experiment directory with:

```bash
CHERRIES_NAME='Prepare face activation materials fixture' \
CHERRIES_TAGS='face,inverse-physics,materials,activation,cpu-fixture' \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
uv run python src/10-prepare-face.py
```

The normal Cherries run completed and recorded 120,020 active cells, 35
activation regions, and 33,636 fixed vertices. Comet experiment:
<https://www.comet.com/liblaf/apple/22d34ce9a99947e79e90d4f1d9d9307e>.
The subsequent CPU readback verified topology, all required fields, the exact
skin-to-volume map, unit active fibers to `2.22e-16`, and confidence range
`[0.20555, 0.97870]`. No GPU solve was run as part of fixture preparation.
