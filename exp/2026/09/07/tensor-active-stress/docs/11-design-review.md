# Design review: face-scale tensor active-stress comparison

## Scope and verdict

This review covers `src/20-face-inverse.py`, `src/tensor_controls.py`, and the graph, material, and solver interface in `src/face_physics.py`. The intended face runs use 288,235 active tetrahedra, 103 labels, 501,409 same-label shared-face edges, no skin energy, the historical passive constants, exact zero controls and rest displacement, fresh Adam state, and 64 optimizer updates.

The implementation has the right core structure: direct dimensionless tensor coordinates, post-Adam PSD projection under `torch.no_grad()`, fail-fast forward and adjoint checks, effective-volume regularizers, and the actual common final step as the primary endpoint. The lambda pilot and post-hoc bumpiness calculation need frozen receipts before endpoint comparison.

## Fixed numerical cap

The tensor arms define

$$
Q_{\rm ref}=3\mu
$$

and use the predeclared finite cap

$$
q_{\max}=10Q_{\rm ref}=30\mu.
$$

Thus \(Q_{\rm ref}=0.0302013423\) MPa and \(q_{\max}=0.3020134228\) MPa. This cap was fixed before the face endpoints and is a bounded numerical design choice, not a physiological calibration. The earlier two-tetrahedron study does not justify or select this cap: its old run is invalidated because its barriers and tissue-fraction treatment do not match the face comparison. Do not cite that old run as cap evidence.

Keep the same \(q_{\max}\) across all PSD arms. Do not tune it from the reported 64-step fit, motion, or bumpiness endpoints. `detF`, inversion count, and other geometry outputs remain diagnostics only; this design does not reject or select states using a geometry threshold.

## PSD parameterization and projection

The tensor arms use \(z_i\in\mathbb{R}^6\) in a Frobenius-orthonormal symmetric basis,

$$
Z_i=
z_{i,0}E_{11}+z_{i,1}E_{22}+z_{i,2}E_{33}
+z_{i,3}\frac{E_{12}+E_{21}}{\sqrt 2}
+z_{i,4}\frac{E_{23}+E_{32}}{\sqrt 2}
+z_{i,5}\frac{E_{13}+E_{31}}{\sqrt 2},
\qquad Q_i=Q_{\rm ref}Z_i.
$$

`tensor_controls.matrices` and `coordinates` implement this basis consistently: the Euclidean norm of the six coordinates equals the tensor Frobenius norm.

The PSD path is sound. `project` runs under `torch.no_grad()` after each Adam update, clips the three eigenvalues to \([0,q_{\max}/Q_{\rm ref}]\), reconstructs the matrix, and stores the projected coordinates. The forward and adjoint consume that stored matrix directly. This avoids differentiating through `eigh` at the exact zero matrix, where the PSD projection has no unique derivative and the triple eigenvalue can destabilize eigenvector gradients. It also avoids the dead first derivative of an \(LL^T\) map initialized at \(L=0\).

The run evaluates states 0 through 64 and performs exactly 64 calls to `optimizer.step()`. Projection occurs before the next state is evaluated. The first state is exact zero; each evaluated tensor state after an update is projected.

Keep these checks in the receipts:

- state 0 reproduces the passive tensor-material solution;
- the state-0 fit gradient is finite and nonzero where the target supplies a signal;
- every reported tensor state has eigenvalues inside the cap tolerance;
- proposed Adam-step RMS, projection RMS, and accepted update RMS are recorded;
- zero-stress cell fraction and upper-cap cell fraction are recorded;
- the parameter and Adam state are restarted together for every arm.

Projected Adam retains moments from the unconstrained gradient. That is an algorithm choice shared by all PSD arms. Cap occupancy and the difference between proposed and accepted update norms show when this choice materially affects the trajectory.

## Lambda pilot

The smoothness, magnitude, and rank components are homogeneous quadratic in \(Z\). Their gradients are exactly zero at the required zero seed. State 0 cannot determine a regularization coefficient.

Use a single fit-only PSD pilot with four updates and `record_component_gradients=true`. Select coefficients only from saved states 2, 3, and 4:

$$
\lambda_R =
c_R\,
\frac{\operatorname{median}_{s=2:4}\lVert g_{\rm fit}^{(s)}\rVert_{\rm RMS}}
{\operatorname{median}_{s=2:4}\lVert g_R^{(s)}\rVert_{\rm RMS}},
\qquad
\lambda_B =
c_B\,
\frac{\operatorname{median}_{s=2:4}\lVert g_{\rm fit}^{(s)}\rVert_{\rm RMS}}
{\operatorname{median}_{s=2:4}\lVert g_B^{(s)}\rVert_{\rm RMS}}.
$$

Predeclare \(c_R\) and \(c_B\). Use the same global packed-coordinate RMS calculation for every component. If a denominator is zero or nonfinite, fail the selection visibly. Do not substitute a default coefficient.

The selection receipt must store the three source gradient files or hashes, all component RMS values, target ratios, selected coefficients, pilot cap, source/config hashes, and the statement that no 64-step endpoint metric was inspected. Discard the pilot tensor, displacement, and Adam moments. Restart every reported arm from exact zero/rest with a new Adam instance, and freeze the selected coefficients across the applicable PSD arms.

The magnitude gradient can remain a recorded diagnostic with `magnitude_weight=0` unless a separately motivated magnitude-regularized arm was requested.

## Smoothness normalization

The graph implementation uses the finite-volume conductance

$$
w_{ij}=
\frac{A_{ij}}{d_{ij}}\,
\frac{2f_if_j}{f_i+f_j},
$$

where \(A_{ij}\) is shared-face area, \(d_{ij}\) is centroid distance, and \(f_i\) is MuscleFraction. It retains only pairs with the same activation label. Since \(w_{ij}\) has units of length,

$$
R(Z)=
\frac{\ell^2}{V_{\rm muscle}}
\sum_{(i,j)\in E}w_{ij}\lVert Z_i-Z_j\rVert_F^2,
\qquad \ell=0.005\ {\rm m},
$$

is dimensionless when fixture coordinates are metres. The implementation uses

$$
\widetilde V_i=V_i f_i,
\qquad V_{\rm muscle}=\sum_i\widetilde V_i,
$$

for magnitude and rank terms, so those statistics are also globally effective-volume weighted.

The expected 501,409-edge assertion prevents a silent graph change. The report should also preserve edge-weight units, total effective volume, connected-component count, and conductance distribution. Smoothness omits cross-label faces by design; cross-label roughness may be reported descriptively, but it is not part of \(R\).

## Rank penalty and statistics

For PSD eigenvalues \(\lambda_1,\lambda_2,\lambda_3\ge0\),

$$
B(Z)=
\frac{1}{V_{\rm muscle}}
\sum_i \widetilde V_i
\left[(\operatorname{tr}Z_i)^2-\lVert Z_i\rVert_F^2\right] =
\frac{1}{V_{\rm muscle}}
\sum_i 2\widetilde V_i
(\lambda_1\lambda_2+\lambda_1\lambda_3+\lambda_2\lambda_3)
\ge0.
$$

The implementation computes this formula correctly in the orthonormal basis. It is zero for both rank-zero and rank-one tensors, so reducing amplitude can lower \(B\) without producing a rank-one active field.

The endpoint therefore needs the existing quantities together:

- `Q_rms_mpa` and `Q_trace_mean_mpa` for amplitude;
- `rank_penalty` for the objective component;
- `rank_mixing_fraction`, the effective-volume-weighted \(B/\operatorname{tr}(Z)^2\) ratio;
- `principal_tension_fraction`;
- `zero_stress_cell_fraction` and `upper_cap_cell_fraction`.

For a nonzero PSD field, the mixing fraction lies in \([0,2/3]\); zero indicates rank one on the active support and \(2/3\) indicates isotropic rank three. It is already `None` at the exact zero field. When interpreting a very low-amplitude endpoint, report the amplitude and zero-stress fraction beside the ratio. A small rank penalty or mixing ratio alone is not evidence of useful rank-one actuation.

The cap and zero occupancy fields are cell-count fractions, while the rank ratio and principal fraction are effective-volume weighted. Their names should retain that distinction. Adding volume-weighted cap/zero fractions would help, but is not required for launch if the existing definitions are reported exactly.

## Comparison limits and common-step logic

The Raw6 arm is the historical active-strain offset model. It uses

$$
A^{-1}=I+\operatorname{sym}(q_{\rm raw})
$$

with the historical off-diagonal coordinate scale and learning rate 0.3. It is a continuity reference, not an unrestricted-\(Q\) ablation. Raw6 and PSD active stress differ in constitutive law, coordinate scale, constraint, and physical units. Their endpoint difference cannot be attributed to the PSD constraint alone.

All PSD arms do share the normalized orthonormal \(Z\) basis, \(Q_{\rm ref}\), cap, learning rate, Adam epsilon, initialization, update count, forward solver, and logging policy. Those comparisons can isolate the added smoothness and rank objective terms once the pilot coefficients are frozen.

The loop and failure logic are suitable for a fixed-budget comparison:

- state 64 is the primary endpoint for every arm;
- best total-objective state is secondary and does not replace the common final endpoint;
- unsuccessful forward equilibrium, unsuccessful adjoint solve, or nonfinite control gradient stops the run and writes a failure receipt;
- `detF_min`, `detF_max`, and inversion count are recorded without gating the run;
- checkpoints contain the control, displacement, optimizer state, config, active IDs, and the physical \(Q\) or \(A^{-1}\) field.

The saved VTU endpoints permit post-hoc surface analysis. Before comparing endpoints, freeze one bumpiness formula and apply it to every arm and the target. The runner currently records fit and motion but does not calculate bumpiness. The reporting step must record the formula, normalization, mesh/connectivity source, code hash, and all arm values in one receipt. Do not choose among bumpiness formulas after viewing arm rankings.

## Prelaunch checklist

- Preserve the predeclared \(q_{\max}=10Q_{\rm ref}=30\mu\) numerical cap across all PSD arms; do not cite the invalidated small-model run as its justification.
- Verify the fixture assertions: 288,235 active tetrahedra, 103 labels, and 501,409 graph edges.
- Verify zero-\(Q\) passive equivalence and finite first fit gradient.
- Verify orthonormal pack/unpack and no-grad projection invariants.
- Complete the four-update fit-only pilot and freeze \(\lambda_R,\lambda_B\) from states 2–4.
- Reset tensor, displacement, and Adam state for every reported arm.
- Freeze the bumpiness metric before endpoint comparison.
- Keep state 64 as the primary endpoint and geometry as diagnostic output.
- Save exact commands, configs, source hashes, component-gradient receipts, solver receipts, traces, checkpoints, and endpoint meshes.

A failed invariant should stop visibly. Do not silently change the cap, coefficient, optimizer, control seed, solver tolerance, or endpoint rule during the comparison.
