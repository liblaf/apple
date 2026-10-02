# Matched Smile inverse-fitting comparison

The user requested an actual inverse-physics fit of the `Smile` target with the
original and accelerated solvers, active stress, bone and eyeball collision,
and a visualization of both fitted results.

Both arms use the frozen `expression-inputs-002` bundle and the same
user-requested objective: normalized skin-position error plus spatial stress
smoothness, with magnitude and jaw penalty weights both zero. The optimization variables are six symmetric
active-stress coordinates per active muscle tetrahedron and one mandible hinge
angle. Materials, reference geometry, targets, observation weights, stress
projection, regularizers, and outer optimization settings are identical.
The added stress is a material tensor `Q` with energy
`0.5 Q:(F^T F-I)` and first-Piola contribution `FQ`. The six coordinates use
a Frobenius-orthonormal symmetric basis, scaled by `0.012328767123287673 MPa`
and projected to eigenvalues between zero and ten in normalized coordinates.
Zero initialization means zero additional expression stress; the frozen neutral
baseline stress remains present.
The normalized smoothness weight is reused from the hash-bound
`expression-fitting-007/calibration.json`; it is not recalibrated separately
for either solver.

The latest user instructions request projected Adam at learning rate `0.3`,
zero magnitude and jaw penalty weights, no outer step rejection, and loose
forward convergence. Both arms restart from the same tight neutral with a
20-update budget. Each update applies one full projected Adam proposal.
There is no outer backtracking, hard neighbor-RMS gate, non-descent direction
replacement, objective pre-screen, Armijo gate, or residual-error acceptance
gate. PSD stress projection and the bounded hinge remain the parameterization.
Failed physical solves stop visibly without a smaller-step retry.

Earlier runs are preserved as history: the `0.003` primary stopped after five
original updates; its hybrid never started. The subsequent screened `0.3`
primary stopped at zero accepted updates and the separate V100 host hybrid at
seven. The completed `0.003` hybrid pilot remains separate evidence.

Contact remains enabled for the entire fit. The existing eye-inclusive model
contains complete source cranium and mandible surfaces and both registered
eyeballs. It permits soft-tissue versus rigid-obstacle contact. Cranium and
eyes stay fixed; the mandible follows its differentiable hinge pose. This
comparison does not introduce soft-soft or rigid-rigid collision. An explicit
assembly audit records retained triangle counts and obstacle motion, and
geometric audits check the common neutral and final states.

The old arm uses the original accepted-force PNCG runtime. The new arm uses
`hybrid_diag`, with transition threshold
`max(final_atol, initial_force * 1e-3, newton_switch_atol)`, followed by
safeguarded matrix-free Newton-CG with scalar diagonal preconditioning.
The inverse-fit comparison opts into `newton_switch_atol=1e-7`: a proposal
already below that force starts Newton immediately. Existing solver defaults
retain a zero absolute floor. This empirical admission rule follows the exact
first-proposal probe; it is not a global convergence guarantee. The implicit
adjoint remains physical and unshifted.
Both comparison arms now target a forward free-force norm of `1e-8`, about
66 times the previous fixed `1.5192003475221146e-10`. The adjoint relative
tolerance stays `1e-7`. The first-order primal-error estimate is recorded as
a diagnostic, not an outer update gate. The tightly refined common seed is
retained byte-for-byte; older strict/screened timings are not pooled with the
new run.

A common neutral seed is prepared once, checked, and frozen. Both fits start
from exactly the same displacement, zero added active stress, and zero hinge
angle. Common preparation is reported separately from fit time. Each fit
records complete solve wall time, forward and adjoint receipts, rejected
proposals, accepted iteration history, objective, target RMS error, projected
stationarity, remaining primal-error estimate, and deformation validity.
An invalid physical evaluation terminates visibly in either arm without
shrinking the outer step or recording a successful update.

A one-update pilot checks the complete inverse path before the matched run.
The user selected a matched budget of 20 accepted inverse updates and
visualization, with explicit failure and wall limits. Reaching a budget does not establish inverse
convergence. Any extension or changed setting is recorded in the final report.

The primary timing comparison uses sequential runs on the same Paratera
RTX 4090 with eight IPC threads. V100 host's available RTX 3090s may be used for
independent pilots and validation; timings from different hardware are not
combined into a speedup. Existing unrelated jobs and outputs are preserved.
Remote runs use Cherries local evidence (`DEBUG=1`), because account credentials
are not copied to compute servers. Sources, inputs, settings, hardware, and
output checkpoints are bound by hashes.

The final visualization uses common front and side cameras for the target,
old-solver fit, and new-solver fit, a common positional-error color scale,
bone/eye context views, and objective/RMS curves against accepted updates and
elapsed time. Results and speedups will be reported only from completed
artifacts, with physical and optimization convergence distinguished.

## Timing isolation

Following the user's timing concern, the auxiliary V100 host RTX3090 full-Adam
pilot was stopped during its first forward solve, before any completed update.
No auxiliary experiment will run alongside the primary pair. Original and
hybrid run sequentially on the same Paratera RTX4090. The 2026-09-22 04:28 CST
audit found only the original comparison process on that GPU and no other
substantial CPU workload visible in the allocated environment. The allocation
has a ten-core CPU quota; provider-host CPU exclusivity is not established.
All V100 host pilot times remain excluded from the primary speedup.
Evidence: `data/timing-isolation-001.json`.
