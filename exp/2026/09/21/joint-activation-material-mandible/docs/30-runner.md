# Final joint runner

`src/30-joint-pilot.py` implements numerical preparation and the final trend
experiment. It does not claim anatomical validation. A run is admitted from an
exact converged full-target neutral checkpoint, matching contact and derivative
receipts, and immutable input, material, solver, and shared-field lineage.

The experiment has three stages:

1. `calibrate` measures a strong activation-smoothness weight with independent
   same-tolerance and tighter-relative-tolerance checks.
2. `control_converge` optimizes four dense activation fields and four six-DoF
   jaw poses while the shared material field remains fixed. Budget exhaustion
   is not convergence.
3. `joint_trend` releases the shared field and records up to 100 accepted
   updates or 12 hours. This is a finite-budget trend experiment. It requires a
   genuinely converged fixed-shared parent and at least one accepted update.

The full optimizer and reporting contract is in
[31-final-optimization-contract.md](31-final-optimization-contract.md). The
one-shot lineage-preserving sequence is described in
[50-prepared-joint-sequence.md](50-prepared-joint-sequence.md).

## Jaw and contact contract

Activation is a positive-semidefinite six-coordinate tensor in every active
tetrahedron and expression. Jaw pose uses six normalized coordinates per
expression. Rotation-vector coordinates have a 10 degree scale and translation
coordinates have a 5 mm scale; the normalized proposal box is `[-1, 1]^6`.
These are computational search bounds, not biological jaw limits. Every row
starts at zero unless restored from an admitted control checkpoint.

The prepared manifest also contains an older source-surface screen near a
one-degree opening. That box came from source mandible/cranium and template-skin
intersection counts before the corrected FEM contact model. It is retained as
`legacy_source_box_qa` only. `joint_final_geometry.py` derives the live receipt
from the broad computational bound, rigid support consistency, and exact
topology-aware FEM lip, oral, soft-bone, and mandible-cranium endpoint checks.
The runner asserts that aggregate while keeping `anatomical_validation: false`.

The IPC energy is frictionless pure-soft versus pure-cranium/pure-mandible
contact. Mixed-label transition faces remain bonded by the FEM discretization.
Soft-soft and bone-bone IPC energies are disabled. Before every expression
solve, a separate pure-FEM cranium/mandible guard checks the straight boundary
vertex path from the accepted seed to the proposed jaw pose. It adds no energy,
force, or derivative and does not check a rigid rotation arc or complete source
bones. Rejected paths enter the explicit outer backtracking transaction.

The jaw preflight records zero, all 12 signed broad-box axis endpoints, and the
legacy one-degree center as an explicitly nonzero test candidate. Endpoint
failures characterize the computational domain and do not silently shrink the
proposal box. See [28-jaw-preflight.md](28-jaw-preflight.md).

## Shared fields

The runner infers the shared basis strictly from the admitted neutral protocol:

- `constant20`: 18 constant bulk-stress coordinates plus skin baseline and
  log-stiffness;
- `spatial80`: 78 anchored bulk coordinates from the exact audited basis plus
  the same two skin coordinates.

Spatial80 requires exact CPU field, basis, audit, and full-face directional
receipts and the frozen `0.5 * 100 * bulk_spatial_roughness` term. Details are
in [25-spatial-face-gradient-validation.md](25-spatial-face-gradient-validation.md).
Shared initialization and prior-center hashes pass through calibration,
control, and final checkpoints. Spatial bulk prior departure uses the audited
per-tissue mass quadratic so exact constant embedding retains constant20
semantics. Skin baseline stays centered on the admitted full-target neutral
state; skin log-stiffness is centered on the study-map multiplier of one.

## Admission evidence

The runner requires:

- a full `80.6 N/m` proxy target (`skin_prestress_fraction == 1.0`), recorded as
  a model-dependent study proxy rather than a measured biological prestress;
- optimizer, neutral-budget, contact, and preparation convergence receipts;
- the selected inner solver and production forward/adjoint tolerances;
- synthetic contact, CCD, Hessian, material, and all-six-jaw derivatives;
- the 16-direction full-face contact derivative receipt;
- the validated rigid-bone linear CCD implementation and input mapping;
- Spatial80 CPU and full-face receipts when that basis is selected; and
- exact prepared-input, implementation, field, contact, and parent hashes.

When Newton-CG is selected by the neutral checkpoint, its inner relative
tolerance is `1e-3`, with at most 12 Newton steps and no fallback. Forward and
adjoint tolerances remain `1e-6`, `1e-12` absolute, and `1e-7`. The matching
Newton full-face contact receipt passed all 16 checks with maximum relative
error 0.3005%.

## Optimizer and failure behavior

Every accepted evaluation solves neutral plus all four expressions, completes
their adjoints, checks shape/contact/geometry budgets, and writes an immutable
checkpoint and trace row. A trial begins from a full owned-state snapshot.
Expected numerical failures trigger explicit outer backtracking; rejected
trials do not advance Adam moments, parameters, seeds, warm adjoints, or the
accepted trace. Exhausted backtracking is a proposal failure, not convergence.
Unexpected programmer errors fail the run.

Fixed-shared convergence requires, for every expression and jaw row, a 99%
projected-gradient reduction from the recorded initial mapping, a small actual
reversible projected-Adam proposal, and an objective plateau over the declared
window. Dense activation diagnostics include volume-weighted and maximum-tet
proposal norms plus mass-normalized gradient summaries. The joint stage reports
the same activation and jaw evidence with shared projected gradients, tensor
spectra, bound occupancy, prior departures, spatial roughness, and skin
reference departures.

`terminal.pt`, `summary.json`, `trace.json`, protocol/readiness receipts, and
float64 visualization snapshots preserve lineage. Rendering is mandatory after
completed control and joint stages; the sequence runner rebuilds the review site
from those actual artifacts. Existing output directories are never overwritten,
and failed or unconverged preparation never advances to final trends.
