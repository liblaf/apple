# Strongly smoothed active-stress face comparison

Historical v1 protocol. The updated numerical pipeline is described in [01-inverse-v2-protocol.md](01-inverse-v2-protocol.md); existing results below retain their original interpretation.

Run eight fits as two independent sequential chains for the same Smile target: position-only and position-plus-normal. Within each chain, stage 1 is unrestricted signed symmetric stress; stage 2 is positive-semidefinite stress initialized by projecting the completed stage 1; stage 3 keeps the strongest contraction eigenmode from completed stage 2 and refits its scalar strength with its axis fixed; stage 4 copies stage 3's stress exactly and then learns the unit axis as well as its nonnegative strength. Optimizer moments reset between stages. A failed or incomplete parent blocks its descendants; a different loss column remains independent.

This is a sequential restriction/release experiment, not eight fits from independent neutral starts. The first stage in each column starts at zero stress and neutral displacement. Stage 2 discards negative modes; stage 3 discards the two weaker modes. Every transferred tensor is re-equilibrated under the new stage's constraints. Axes inferred from a fit are effective model directions, not independently measured anatomical fibers.

## Fixed mechanics and observation

Preserve the fixture at `exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture`: 228,660 vertices, 1,146,517 tetrahedra, and 288,235 independently controlled active cells. Preserve its fixation, tissue fractions, muscle graph, target and surface correspondence. Add no skin potential, contact or jaw coordinates. The outer surface remains the observation mesh for position and normal loss.

Use polynomial Stable Neo-Hookean bulk energy and a signed symmetric reference stress Q:

    W(F,Q) = mu/2 * (||F||^2 - 3) - mu*(J-1)
             + lambda_code/2*(J-1)^2 + 1/2*Q:(F^T F-I), J=det(F)
    P(F,Q) = P_SNH(F) + F Q

Each tissue term is integrated with its frozen tissue fraction. Only muscle receives activation. Passive E values are fat 0.0112, muscle 0.012, aponeurosis 1.693 MPa, with nu=0.49 for all three. Recompute mu=E/[2(1+nu)] and lambda_code=lambda_classical+mu for the polynomial SNH energy. These are research-informed model settings, not calibrated material measurements for this subject. No baseline prestress is added.

Qhat=Q/Qref is dimensionless, with Qref=muscle mu=0.004026845637583893 MPa. Symmetric tensor coordinates use the Frobenius-isometric Mandel convention `(xx,yy,zz,sqrt(2)xy,sqrt(2)yz,sqrt(2)xz)`. The unrestricted model permits signed eigenvalues; PSD/strength constraints apply only to the respective constrained models. No stress cap is silently added. Learned rank-one activation uses an explicit amplitude and normalized nonzero vector; zero amplitude has zero axis gradient, which is recorded. A zero input tensor gets a declared deterministic initialization axis.

## Objective and strong smoothness calibration

    J = P/l_ref^2 + beta*N + eta*R(Qhat)

P is frozen-reference-area-weighted position vector MSE divided by three, in mm². l_ref=13.236093032531715 mm retains the selected 2 mm vector RMS / 5 degree normal-deviation calibration. N is reference-area-weighted squared unit-normal chord error; beta=0 or 1.

    R = ell_s^2 / V_m * sum_edges c_ij ||Qhat_i-Qhat_j||_F^2

Use ell_s=5 mm, total effective muscle volume V_m, and c_ij=shared-face-area/centroid-distance times the harmonic mean of muscle fractions. Graph edges join only same-muscle neighbors in the reference geometry. The full tensor is smoothed, avoiding axis-sign ambiguity. R is dimensionless and the same physical penalty applies to every parameterization.

A short matched pilot chooses one eta before the eight main fits. Both loss columns get eight-update unregularized type-1 controls. Candidate weights are 3, 30, 300 and 3000 times a recorded gradient-norm ratio measured at a fixed neutral-gradient proposal. Choose the first common candidate giving at least 75% lower squared-neighbor variation than both matched unregularized controls. Report retained data-fit progress as well; it is not hidden by the total objective. Pilot success does not guarantee that reduction for every endpoint. Never choose different eta values per row or loss.

## Optimizer and checks

Use 200 accepted projected-Adam updates per main stage, learning rate 0.05, epsilon 1e-8, betas (0.9,0.999), fresh moments per stage. Each proposal is projected into its parameterization and checked with up to 16 step halvings, Armijo coefficient 1e-4. If the projected Adam direction is not a descent direction, record the event, reset its moments and use the projected steepest-gradient direction with maximum pre-projection control displacement 0.05; all acceptance checks remain in force. Learned axes are normalized without discrete sign flips during optimization. Every accepted state must have a successful strict PNCG equilibrium, positive det(F) throughout the volume, and a successful adjoint with its actual residual within tolerance. Failed physical proposals are recorded and backtracked; programming failures remain visible. No result is labeled converged solely because its finite update budget ended.

Preserve per-stage initial and last tensors, displacements, optimizer states, objective/metric histories, numerical source hashes, fixture hashes, solver receipts and rejected proposals. Report position RMS, normal-angle RMS, full stress roughness and magnitude, physical J, and histories. Independently verify spectral transitions, tensor constraints, CPU-recomputed losses/roughness/J, and solver receipts at the end.

## Run and report lifecycle

Use named, tagged production Cherries runs for CPU parameterization checks, full-face derivative validation, calibration, each stage's figures and final verification. Keep existing research sources and other fitting jobs untouched. The task-specific report is served only on the tailnet address with a transient user service and a 72-hour runtime limit; no persistent service is installed. The page shows setup, pilots, running, blocked and completed states truthfully. Full raw fields are local artifacts; selected figures and metrics are served in the report.
