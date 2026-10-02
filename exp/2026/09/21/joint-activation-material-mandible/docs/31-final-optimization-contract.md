# Adopted-neutral joint-optimization contract

This contract supersedes the Shared20/Spatial80 baseline-prestress plan. The
selected neutral is the converged prescribed-skin, full-source-skull
equilibrium. It eliminates fitting a baseline-prestress field in the inverse;
it does not change the constitutive reference configuration.

## Frozen state and coordinates

Let `X` be the FEM reference and `u0` the saved terminal FEM displacement. The
adopted neutral geometry is `x0 = X + u0`.

The inverse continues to evaluate bulk Stable Neo-Hookean and membrane energies
with the original `X`, original tetrahedral `Dm`, and original membrane tangent
frames. It solves absolute displacement `u`; the reported expression increment
is `v = u - u0`. Replacing mesh points with `x0` would set the constitutive
reference to the deformed geometry and is prohibited.

For a transferred source expression displacement `d_e` at an observation node,
the target world coordinate is `y_e = x0 + d_e`, or equivalently the target
displacement relative to `X` is `u0 + d_e`. Observation IDs, masks, cohort
order, splits, and transferred deltas are frozen and hash-bound. Observation
areas/weights and the activation graph are rebuilt on `x0` and likewise
hash-bound. The data loss compares predicted and target world coordinates, or
the exactly equivalent displacements relative to `X`.

## Constitutive and contact contract

The selected forward's heterogeneous per-triangle skin `E`, `nu=0.49`,
thickness, and baseline stress resultant are frozen. The per-face baseline
resultant remains in its original frozen tangent frame. Bulk baseline stresses
are zero and frozen. Source cranium and mandible meshes remain complete
registered frictionless IPC obstacles; no source triangle is removed. At zero
jaw pose, the full state is `[u0; cranium=0; mandible=0]`.

At zero activation, zero jaw pose, and unit skin multiplier, that state must
pass direct force, CCD, contact, fixed-boundary, determinant, and intersection
validation against the selected forward. This establishes a frozen equilibrium
state; it is not a stress-free rebase.

The skin stiffness multiplier scales the frozen per-face passive `mu` and
`lambda` together. It does not scale or refit the frozen baseline resultant.
Therefore `v=0` is an equilibrium only at multiplier one. At a changed
multiplier, an expression solve may relax from `u0`; reports must state this
instead of calling every material iterate a neutral equilibrium.

## Current optimization variables

For four expressions and 288,235 active muscle tetrahedra:

| Family | Shape | Scalars |
| --- | ---: | ---: |
| Muscle activation stress | `4 × 288235 × 6` | 6,917,640 |
| Mandible pose | `4 × 6` | 24 |
| Global log skin-stiffness multiplier | `1` | 1 |
| **Total** | | **6,917,665** |

Activation remains a symmetric PSD additive stress increment, projected with
the existing spectral cap. Its strong graph smoothness and magnitude penalty
apply separately for every expression. Jaw has its existing bounded rigid
six-DoF parameterization and prior. The global stiffness scalar receives a
scalar prior; it has no continuous spatial field and hence no spatial
smoothness term. If regional skin stiffness or another continuous material
field is introduced later, its physical-space smoothness must be declared and
strongly regularized before release.

No bulk baseline-stress coordinates or skin-baseline-stress coordinate are
trainable. Do not instantiate the legacy `SharedFieldParameters` or
`SpatialSharedFieldParameters` as an optimizer block: both would silently
reopen removed baseline-stress variables.

## Required path to Tuesday's trend

1. Produce and validate the frozen-neutral bundle from the selected forward.
2. Integrate the validated absolute-coordinate/material adapter into the joint
   runner and pass expression and jaw derivative checks with full-source contact.
3. Converge a fixed-skin-stiffness activation/jaw control with rich visual
   review.
4. Release the one skin multiplier with activation and jaw together, saving
   checkpoints and trend plots. At least one accepted, valid update is required;
   inverse convergence is not required by Tuesday.

The frozen bundle alone does not make the inverse runner ready. Until joint-
runner integration, derivative/control checks, and a simultaneous trajectory exist,
`final_joint_is_deliverable` remains false.

Evidence for the selected bundle is in [82-adopted-neutral.md](82-adopted-neutral.md).
