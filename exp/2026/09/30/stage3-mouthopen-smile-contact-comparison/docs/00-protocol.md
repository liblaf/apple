# Stage 3 MouthOpen-to-Smile matched transition protocol

## Objective

Produce two animations of the same Stage 3 fixed-axis contraction transition: one with the declared rigid contact model enabled and one with collision disabled. The comparison changes only collision handling.

## Controls and provenance

`data/10-stage3/endpoints.npz` supplies the two exact `rankone_fixed` active-strain endpoint tensors. `S_mouthopen` comes from the MouthOpen four-stage chain. `S_smile` comes from the Smile four-stage `l2-normal` chain, remapped from the full mesh to the repaired pruned mesh by saved original cell identity. The endpoint package also retains the Stage 3 scalar strengths and fixed axes, and verifies their reconstruction of each tensor.

The historical Smile Stage 3 checkpoint records `solver_valid=false`. It is accepted only as a fixed activation control for new forward solves; it does not certify the Smile geometry. The historical MouthOpen Stage 3 checkpoint records `solver_valid=true`.

The common initial geometry is `data/10-stage3/seed.npz`, copied from the independently audited Stage 4 contact run's frame 000. It stores displacement and jaw pose only. No Stage 4 activation tensor, scalar strength, or axis is used by either transition. Both variants first solve a fresh Stage 3 MouthOpen equilibrium from this common geometry.

## Matched solves

For every transition coordinate beta, each variant performs a fresh forward equilibrium solve with

`S(beta) = (1 - beta) S_MouthOpen + beta S_Smile`

and the prescribed jaw pose `(1 - beta)` times the saved MouthOpen pose about the shared pivot. The full symmetric tensors are interpolated directly. Although each endpoint is rank one, an intermediate tensor can be rank two; it is not projected back to a rank-one form.

Both variants use the same repaired mesh, fixed-point and mandible boundary contract, prescribed jaw pose, active-strain material model, material parameters, active-cell ordering, harmonic carry predictor, 121-frame cosine timing, solver budgets, and acceptance gates. The collision-on variant adds the existing frictionless IPC contact model. The collision-off variant has the same geometry available for diagnostic intersection reporting but applies neither contact energy nor CCD rejection.

## Acceptance and scope

An accepted state requires an absolute free-force residual no greater than `1e-8 MPa m2`, equivalent to `0.01 N`. Both variants allow the declared small inversion cap of at most `0.001` of volume cells; passing that cap does not establish orientation validity. The collision-on run also requires its CCD and declared contact audit to pass.

The contact scope is the pure-soft FEM boundary against the complete source cranium, mandible, and eyes. Bonded mixed attachment faces are excluded. Tissue self-contact and rigid-rigid contact are not enabled.

## Reporting boundary

No numerical comparison result is claimed until both fresh transition solves finish and their saved frames pass the corresponding audit. The run summaries, audits, and renders will retain the source receipts, force diagnostics, contact receipts, inversion diagnostics, and frame-level provenance.
