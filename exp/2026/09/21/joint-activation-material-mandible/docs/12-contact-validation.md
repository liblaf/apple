# Soft tissue–bone contact: implementation and numerical checks

Frictionless contact now contributes energy, forces and Hessian products to equilibrium, and its exact Hessian enters the implicit material and jaw derivatives. It acts between the pure soft FEM boundary and the pure cranium/mandible boundary. Mixed-label junction faces are treated as bonded and remain separate from sliding contact. This defines the computational interface; anatomical attachment accuracy remains unvalidated.

The frozen [configuration](../data/contact/config.json) uses the area-weighted physical IPC clamped-log barrier, activation distance **0.1 mm**, and stiffness **0.01 MPa**. These are numerical parameters, not measured interface properties. The energy unit follows the physical-barrier normalization in [IPC Toolkit v1.6.0](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/potentials/barrier_potential.cpp): MPa times volume, consistent with the tissue energies. Soft–soft and bone–bone contact are not enabled by this configuration.

Each primal rebuilds its contact state. CCD checks both prescribed mandible boundary changes and equilibrium steps. Every backward owns a fresh contact state at its saved equilibrium, so evaluating another expression cannot replace the first expression's contact Hessian.

The final jaw runner additionally uses a separate [rigid-bone CCD guard](29-rigid-bone-ccd.md)
for the pure FEM mandible against the cranium. This is a feasibility check on
the solver's straight boundary-vertex motion; it adds no bone-bone energy and
does not change the soft-tissue/bone force or derivative model above.

## Verified evidence

The [synthetic receipt](../data/contact-validation/summary.json) reports success for an active point–triangle interface coupled to a four-tetrahedron FEM model:

| Check | Observed result |
| --- | ---: |
| Contact energy directional derivative | Maximum relative error 6.07e-7 |
| Exact contact Hessian-vector product | Maximum relative error 6.33e-7 |
| Implicit material + all six jaw coordinates | Maximum vector relative error 4.85e-5 |
| Largest individual-coordinate relative error | 2.96e-4 |
| Direct crossing CCD | Step restricted to 0.3984375 |
| Prescribed-boundary crossing | Rejected at CCD fraction 0.328125 |
| Interleaved expression gradients | Zero difference from independent evaluations |

Finite-difference equilibrium states in the implicit check use an independent explicit 3×3 Newton solve. This is numerical validation, not identification of contact stiffness or proof of anatomical accuracy.

The full reference mesh [preflight](../data/contact-initial-validation-002/summary.json) has no initial soft–bone intersections in the selected surfaces. Its minimum active gap is **0.0225836 mm**. The collision mesh contains 63,531 vertices, 41,903 cranium triangles, 13,763 mandible triangles and 65,580 soft triangles; 6,926 mixed junction triangles are omitted from sliding contact.

LBVH's improved-max collision representation can select 169–171 equivalent representatives at the reference state. Energy, minimum gap and nodal forces agree to numerical precision. Counts are not contact area or a convergence metric. See [contact force and gap figures](44-contact-visuals.md) and the [current contact-enabled neutral state](45-neutral-state-visuals.md).

An independent [FEM contact-surface audit](18-fem-contact-surface-audit.md) found zero missing pure boundary faces and zero missing or extra cross-patch candidates at the reference and saved neutral states. Cranium and jaw collider vertices belong to their correct fixed/rigid supports. No nonadjacent mixed-to-pure-bone intersections were found. This supports the contact implementation for the chosen discretization; the 6,926 mixed faces excluded by policy have no independently verified attachment labels.

## Full-face validation status

The first contact-enabled full-face derivative run completed six bulk-stress checks, then made slow contact-limited progress during a skin-baseline perturbation. Its captured solver state had a minimum CCD fraction 2.38e-8. It was explicitly interrupted, with exit 130; the partial evidence is preserved in `data/face-gradient-validation-contact-cold-seed-interrupted/` and is not a complete validation.

The replacement run **passed all 16 checks**, using 33 forward solves. It uses the same frozen converged base equilibrium independently for every plus/minus perturbation and retains the 2% relative-error criterion, forward tolerances 1e-6/1e-12 and adjoint tolerance 1e-7. The largest relative error was **0.8973%**, for the smaller jaw-rotation perturbation. All 32 perturbed equilibria have valid contact receipts. This validates the declared FEM collider model; it does not establish complete source-bone coverage. The [complete receipt](../data/face-gradient-validation-contact/summary.json) and tailnet review (private preview omitted) preserve the results.

The opt-in Newton-CG solver separately **passed the same 16 full-face checks**, with 33 forward solves and maximum relative error **0.3005%** for skin stiffness. Its inner linear relative tolerance is 1e-3; the nonlinear and adjoint tolerances remain unchanged. Every perturbed equilibrium has a valid contact receipt, and terminal free-force norms range from 1.54e-15 to 8.35e-13. The [Newton receipt](../data/face-gradient-validation-contact-newton/summary.json) and [Comet run](https://www.comet.com/liblaf/apple/e994b7915b14403682645a9d976d3cb1) completed normally. Newton-CG fails visibly on invalid steps; it does not silently switch solvers. Calibration, control preparation and the final runner require matching solver receipts and checkpoint lineage.

The source-geometry relevance audit identifies small midface/oral regions near registered bone but farther from the declared FEM colliders. The [complete source-bone audit](17-source-bone-contact-audit.md) found 1,263 cranium and 492 mandible intersection pairs in the reference state that are not confined to bonded-junction geometry. Complete source bones therefore cannot be inserted directly as IPC obstacles. The primary experiment retains the declared FEM bone surfaces, and the review exposes the source-to-FEM difference as a modeling limitation. Contact-model coverage and numerical derivative correctness are separate questions.

## Reproduction and run records

Run from this experiment directory:

```bash
CHERRIES_NAME='Contact numerical validation' CHERRIES_TAGS='joint-inverse,contact,validation' uv run --frozen python src/12-validate-contact.py
CHERRIES_NAME='Full-face contact derivatives from a frozen converged base' CHERRIES_TAGS='joint-inverse,contact,full-face,validation' DEBUG=1 uv run --frozen python src/09-validate-face-gradients.py --output-dir data/face-gradient-validation-contact-reproduction --contact-spec data/contact/config.json --initial-checkpoint data/neutral-convergence-010-contact/terminal.pt
```

The original synthetic run is recorded in [Comet](https://www.comet.com/liblaf/apple/ff06e64ab0d04e77a560c9061e98d74a). Its scientific receipt completed before the metadata shutdown was interrupted; Git remained unchanged. The runner now uses the noncommitting experiment profile. Full-face validation uses local Cherries records. Future runs disable Comet's broad Git scans while retaining exact local source snapshots and input hashes.
