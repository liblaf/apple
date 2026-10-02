# Tensor active stress: fixed face comparison protocol

The experiment tests whether a fiber-free contractile stress field can fit the saved Smile target with the historical passive material, and whether smoothness or a soft preference for one principal tension improves its surface behavior. Skin energy is zero. Every arm retains six coordinates on each of the 288,235 active tetrahedra, for 1,729,410 scalar controls.

## Material and controls

Muscle energy density is

\[
W(F,Q)=W_{\mathrm{stable}}(F)+\tfrac12 Q:(F^T F-I),\qquad
P(F,Q)=P_{\mathrm{stable}}(F)+FQ.
\]

The whole muscle potential, including the active term, is multiplied by the saved muscle volume fraction. Fat and aponeurosis retain their historical passive models. The muscle material has no fiber input. At fixed finite Q, the active term is bounded below by minus half its trace, and its material tangent is positive semidefinite. Q = 0 restores the original passive energy, forces, and tangent.

The activation mask equals the set of positive-muscle-fraction cells in the frozen fixture; no cells are excluded for lack of a fiber direction. Keeping passive parameters fixed does not mean activation has no incremental stiffness: the finite active tangent is δP_active = δF Q.

Write Q = Qref Z, where Qref = 3 μ = 0.030201342281879193 MPa. Z uses six orthonormal symmetric coordinates: xx, yy, zz, sqrt(2) xy, sqrt(2) yz, sqrt(2) xz. After each Adam update, eigenvalues of Z are projected into [0, 10]. The cap is 0.30201342281879195 MPa on the reference second Piola active stress. It is a finite exploratory bound, not a calibrated physiological maximum or a uniform bound on Cauchy stress under arbitrary compression.

Eigenproblems are processed in batches of 1,024 cells to bound CUDA workspace. All cells are retained. Projection is outside automatic differentiation; the model receives Q directly, so initialization at zero has a nonzero data gradient.

## Matched tensor arms

1. PSD: data fitting alone.
2. PSD + smoothness: data fitting plus λs S.
3. PSD + smoothness + rank preference: data fitting plus λs S + λr B.

The fitting objective is the mean squared Cartesian displacement residual on the historical IsFace vertices, multiplied by 10^6 to express it in mm². Area-weighted vector fit RMS, motion RMS, and target projection are additional diagnostics, not a replacement objective.

For active volume V = sum(Ve φe), graph conductances wij = shared-face area / centroid distance times the harmonic mean of muscle fraction, and characteristic length ℓ = 5 mm,

\[
S=\frac{\ell^2}{V}\sum_{ij} w_{ij}\lVert Z_i-Z_j\rVert_F^2,\qquad
M=\frac{1}{V}\sum_e V_e\phi_e\lVert Z_e\rVert_F^2,
\]

\[
B=\frac{1}{V}\sum_e V_e\phi_e\left[(\operatorname{tr}Z_e)^2-\lVert Z_e\rVert_F^2\right].
\]

The graph contains 501,409 unique shared faces and does not connect different muscle labels. B is nonnegative for PSD tensors and vanishes at rank at most one, including zero. It is a soft penalty on all six controls. The magnitude term M is recorded but has zero objective weight. S/M, B divided by the volume average of trace(Z)², stress amplitude, motion, and fitting error must be considered together: reducing stress can reduce the unnormalized penalties without improving their shape.

## Settings chosen before final runs

The discarded three-step fit-only pilot starts from zero controls, rest displacement, and zero Adam moments. At its actual step 3, λs is set to make the smoothness-gradient RMS 25% of the fitting-gradient RMS; λr makes the rank-gradient RMS 15%. This yields λs = 5.9051714684188505 and λr = 119.62893380703123. Simulated next-step updates use the pilot's same Adam moments, epsilon, and PSD projection to record the actual effect of the penalties. The receipt is `data/14-frozen-face-settings/summary.json`.

Final runs restart from zero controls, rest displacement, and zero moments. All three tensor arms use Adam learning rate 0.3, epsilon 0.01, and 64 updates. The actual step-64 state is the primary endpoint. The lowest total-objective state is stored separately and cannot replace a final endpoint in the primary comparison.

A fourth, newly initialized 64-update Raw6 run preserves the historical active-strain energy and its direct symmetric offset coordinate scale. It is an evidence-continuity reference. The different energy and coordinates mean it cannot isolate the effect of the PSD constraint or establish an optimizer-independent comparison between activation spaces.

## Solver and geometry policy

Each evaluated state must have a successful forward equilibrium solve and a successful finite adjoint before its gradient is accepted. Forward PNCG uses 5,000 maximum iterations, relative tolerance 5e-4, absolute tolerance 1e-10, and its historical line-search implementation. The adjoint uses the historical CG / MINRES solver selection, 10,000 maximum iterations, and relative tolerance 5e-4. There are no hidden retries or tolerance changes.

No determinant, inversion, or appearance condition rejects a state. Actual det(F) extrema and inversion counts are retained as diagnostics. A numerical failure ends that arm and preserves its failed trial and last valid state; it must be reported as a failure, not as a step-64 endpoint.

## Comparison outputs

The predeclared surface protocol in `40-measurement-protocol.md` measures rest-normal high-pass displacement and target residual at 2, 5, and 10 mm on the same frozen skin and mouth region. The primary scale is 5 mm. The exact saved surface geometries are rendered at common physical scale, without deformation exaggeration. A smaller high-pass value alone is not evidence of an improved face: fitting error, motion, the target's own detail, and the actual images remain part of the assessment.

This is a fixed-budget diagnostic, not an inverse-convergence claim or proof of the activation space's reachable limit.
