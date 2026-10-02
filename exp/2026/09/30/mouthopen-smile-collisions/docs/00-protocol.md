# Contact-enabled MouthOpen to Smile

## Current scope: existing solved activation

The user clarified that contact should be added while keeping the already solved activation. Stage 23 changes only the forward deformation solve. It reads the exact saved endpoint tensors and follows their existing blend; it does not optimize activation. Mesh, materials, prescribed jaw motion, and attachment policy stay as recorded in the parent experiment.

The fixed-only boundary audit found intersections beginning near 5.1603% of the prescribed MouthOpen pose. Forces on free tissue cannot resolve intersections entirely between prescribed vertices. Stage 23 therefore retains every boundary face but exempts fixed-fixed primitive pairs, with free-free and free-prescribed contact enabled. It also reports unfiltered intersections independently. This scoped solve cannot establish complete-boundary collision freedom. No separate rigid obstacle surfaces are added.

The stage-19 audit found intersections involving free tissue in both saved expression deformations, so simply starting a prevention barrier at those displacements is infeasible. The default initialization constructs a feasible displacement from rest, ramps the existing MouthOpen tensor and jaw together, and then follows the exact saved MouthOpen-to-Smile tensor blend. This initialization homotopy does not refit activation. A direct saved-state initialization is retained only as an explicit option and must pass the same feasibility gate.

Stage 23 uses the repeatable standard IPC set and PSD contact search curvature established below. Accepted states require no intersections within the declared contact scope, finite geometry and barrier values, strict free-force convergence, and the existing inverted-cell limit. A separately validated experiment-local assembled Hessian may replace matrix-free products for speed; it must preserve the bulk material derivatives. Earlier trial policies and failures below are retained as history.

## Initial full-boundary trial

The user requested collisions for the MouthOpen-to-Smile animation. This experiment adds frictionless self-contact over the complete extracted FEM boundary, including the lips, through the existing IPC physical barrier and continuous collision detection. Separate skull, jaw, teeth, and eye obstacle surfaces are outside this first self-contact run; their registered geometry and mapping are being examined separately.

The previous animation has intersecting boundaries at every saved state. Those displacements are not admissible initial states for collision prevention. Reuse only its tensor endpoints, tetrahedral mesh, prescribed jaw pose, and harmonic initialization weights. Start from the neutral rest mesh with zero activation, equilibrate with contact, and then ramp the MouthOpen tensor and jaw together from zero to full magnitude. Only after reaching MouthOpen, solve 121 cosine-spaced states to Smile, with `S(beta)=(1-beta)S_MouthOpen+beta S_Smile` and jaw pose `(1-beta)*pose_MouthOpen`. No inverse refit is performed.

The bulk model is the same corrected active-strain Stable Neo-Hookean model, with the same materials, pruned mesh, and authoritative IsFixed boundary policy. The full tensor drives the solve. Contact uses stiffness 0.0012 MPa, half the mean boundary-edge length as its activation distance, zero barrier dmin, and 10 nm CCD minimum separation. No collision surface patches are excluded, including near inverted cells.

Every prescribed carry must pass continuous collision detection before equilibrium. Newton uses contact energy, gradient and exact Hessian products, with the existing GPU contact Hessian upload backend. Contact candidates are rebuilt at every configuration. Accepted states must have finite geometry, free-force norm at most 1e-10 in the existing code units, no detected complete-boundary intersections, and valid contact separation and energy. The previously accepted limit of 0.1% inverted cells is retained and reported separately; this is not a mechanical-validity claim.

Failed continuation proposals are discarded and their parameter interval is bisected. The minimum parameter increment is 1e-5, with at most 12 subdivisions. Each solve permits 200 Newton steps and 3,000 CG steps per linear solve, with the same shift-reuse and physical force gate as the previous transition. The total continuation budget is 1,200 seconds, checked within the solver. Save all accepted checkpoints and the failure receipt if the endpoint cannot be reached. Do not produce or label a completed collision animation unless every frame passes.

Outputs are additive under this new experiment. Imported numerical sources and inputs are hash recorded, and existing results are preserved. Cherries automatic Git commits are disabled.

## Contact search curvature trial

The first exact-Hessian run stopped in neutral initialization after 186 accepted Newton steps: the force remained about 1.10e-7 and all eight subsequent Armijo retries failed. It reached no expression states. Preserve that run under `data/20-contact-transition`.

`src/21-run-contact-projected-search.py` repeats from the same rest state using IPC's `PSDProjectionMethod.CLAMP` for contact stencil Hessians in the forward search direction only. The bulk Hessian stays exact, and the energy, physical gradient, barrier parameters, CCD, convergence gates and budgets are unchanged. It evaluates no implicit adjoint. This tests whether contact search curvature caused the failure; it does not treat the failed state as converged.

The projected-search trial was interrupted with SIGINT after the independent real-mesh directional check found inconsistent rebuilt energy/gradient behavior. Its process exited 130, no expression frames were accepted, and `data/21-contact-projected-search/interruption-request.json` records the exact process identity and reason. The original summary's initializing status is superseded by this interruption record.

## Reproducible IPC contact set

The independent stage-17 comparison found non-repeatable improved-max active sets at identical positions, including when reusing the same broad-phase candidates. Standard `IPC` active sets were repeatable and passed the real-mesh directional derivative check at both rest and the failed state. Stage 22 therefore uses the standard IPC collision set, retaining physical barrier scaling, area weights, every boundary face, the same stiffness and activation distance, and the same strict force/CCD gates. The contact stencil Hessian is PSD-projected only for search. This changes the contact discretization from improved-max cancellation to an area-weighted sum; it is not a claim of the improved-max formulation's refinement convergence. See the [stage-17 measurements](../data/17-contact-set-reproducibility/summary.json).
