# Minimal spatial shared-baseline contingency

## Trigger and scope

**September 21 status:** the 25% constant-basis stage reached constrained
stationarity but failed the neutral surface budget (`0.275980 > 0.25 mm`).
Its contact, force, adjoint, and inversion checks passed. The accepted
`spatial-baseline-audit-005` receipt establishes a full-rank, bounded 78-column
extension with exact constant embedding; its force-residual improvement is
modest and does not predict the nonlinear displacement budget. A bounded
nonlinear 25% probe is therefore the next test, conditional on CPU and full-face
spatial derivative validation. It preserves all numerical and shape limits.
No 50%/100% continuation or final joint run has been started from the failed state.

Keep the approved 20-coefficient model for the current 80.6 N/m continuation.
The constant-basis audit leaves 98.93% of the raw exact-rest skin force and
99.48% of the volume-weighted force uncancelled, but that linear result does not
show that nonlinear equilibrium or the 0.25 mm neutral-motion budget is
infeasible. Use this contingency only if the completed continuation fails the
unchanged numerical gates after solver convergence has been separated from
basis capacity.

The smallest useful spatial extension has **80 shared coefficients**:

- four bulk anchors each for fat and aponeurosis;
- five bulk anchors for muscle;
- six Frobenius-orthonormal symmetric coordinates per anchor;
- the existing uniform skin-resultant and global log-stiffness coordinates.

Thus the bulk block grows from 18 to 78 coordinates while the two skin
coordinates remain unchanged at indices 78 and 79. The flat order is fat
`[4,6]`, aponeurosis `[4,6]`, muscle `[5,6]`, skin baseline, then skin log
stiffness. Dense expression activation remains exactly `[4, 288235, 6]`; jaw
pose and the target cohort are unchanged.

## Convex field contract

For tissue $t$, let $C_{tk}\in\mathbb{R}^6$ be anchor coordinates and let

\[
\phi_{tck}\ge 0,\qquad \sum_k\phi_{tck}=1
\]

on every cell with tissue fraction above `1e-6`. Reconstruct the dimensionless
baseline tensor and physical stress as

\[
S_{tc}=\sum_k\phi_{tck}\,\operatorname{sym}(C_{tk}),\qquad
Q_{tc}=\mu_t S_{tc}.
\]

Project every anchor tensor to the existing dimensionless spectral interval
`[-0.9, 10]`. Since this interval is the Loewner-order convex set
`-0.9 I <= S <= 10 I`, every reconstructed cell automatically satisfies the
same bound. Setting all anchors of one tissue equal exactly embeds the current
constant model. Muscle activation remains a separate positive-semidefinite
per-expression field and is added only at `active_cell_ids`; the baseline and
activation bounds retain their current separate meanings.

Use a normalized inverse-distance basis in the reference geometry,

\[
w_{tck}=(\|x_c-a_{tk}\|^2+\rho^2)^{-1},\qquad
\phi_{tck}=w_{tck}/\sum_jw_{tcj},
\]

with `rho=5 mm`, matching the current smooth-length scale. Build weights within
each shared-face connected component. Components without an anchor use a
one-hot assignment to their nearest anchor; disconnected components have no
graph edge, so this does not create a hidden smoothness coupling. Store
nonnegative weights, cast and renormalize them in float64 before contraction,
and fail if any row is nonfinite, negative, or does not sum to one.

## Anchor placement supported by this fixture

The positive-fraction domains and shared-face connectivity are:

| Tissue | Positive cells | Graph edges | Components | Largest effective-volume share | Second share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fat | 1,042,175 | 1,998,674 | 71 | 99.9975% | 0.00169% |
| Aponeurosis | 125,174 | 201,354 | 809 | 99.9224% | 0.0101% |
| Muscle | 288,235 | 523,640 | 117 | 90.3163% | 8.3343% |

Each dominant support has substantial extent in all three principal directions:
fat standard deviations are 44.9/33.8/26.3 mm, aponeurosis
44.2/34.9/28.2 mm, and muscle 45.7/42.4/23.2 mm. Four non-coplanar anchors are
therefore the minimum that can express coarse variation in three dimensions.
Choose a deterministic greedy nondegenerate tetrahedron in the dominant
component after whitening cell centroids by the tissue-volume-weighted
covariance: start at the cell nearest the weighted centroid, then select the
cell farthest from that point, the cell farthest from the resulting line, and
the cell farthest from the resulting plane. Fail if the resulting tetrahedral
volume is numerically degenerate. This avoids assigning anatomical meaning to
world axes. Add the fifth muscle anchor at the
effective-volume centroid of its 8.33% secondary component. Tiny remaining
components inherit the nearest anchor.

The fraction fields support this geometric placement, but they do not provide
validated anatomical prestress landmarks. In particular, the aponeurosis was
heuristically constructed and muscle attachments remain uncertain. The 103
muscle labels are useful for reporting and stratified residual checks, but a
per-muscle baseline would be a much larger and less identifiable first
extension.

## Smoothness and priors

Build one full-tissue shared-face graph per bulk tissue using the existing
conductance definition from `10-prepare-inputs.py`:

\[
g_{ij}=A_{ij}/d_{ij}\;\operatorname{harmonic}(f_i,f_j).
\]

For basis matrix `Phi`, precompute the coarse matrices

\[
G_t={\ell^2\over V_t}\Phi_t^T L_t\Phi_t,\qquad
M_t=\Phi_t^T\operatorname{diag}(V_cf_{tc}/V_t)\Phi_t.
\]

Then for the `K_t x 6` coordinate matrix `C_t`, use

\[
R_{smooth,t}=\operatorname{tr}(C_t^T G_t C_t),\qquad
R_{magnitude,t}=\operatorname{tr}(C_t^T M_t C_t).
\]

This exactly evaluates the chosen field's normalized graph energy and
volume-weighted magnitude. Runtime cost is only 342 scalar quadratic terms
across the three tissues. The transient preprocessing graph has 2,723,668
edges; storing two int32 endpoints and one float64 conductance costs about
41.6 MiB, but only the 57-entry `G` and `M` blocks are needed at runtime.
Continue to report smoothness and magnitude separately. Freeze their objective
weights through a matched control rather than inferring them from the 98.93%
linear residual.

## Interface changes if the contingency is activated

Do not replace `SharedFieldParameters` in place. Add an experiment-local
spatial class in `src/joint_spatial_fields.py` with the same public skin properties and
one flat `coefficients` parameter, plus per-tissue anchor slices, basis buffers,
anchor-wise `project_`, and the coarse `G/M` regularizers.

`JointPhysics.materials` in `src/joint_physics.py` already ultimately supplies
one `active_stress` tensor per tetrahedron. Make its current constant
`expand(...).contiguous()` branch explicitly accept either `[3,3]` or
`[n_cells,3,3]` and fail on every other shape. No energy-kernel change is
needed: `StableNeoHookeanStress` in `src/joint_materials.py` already reads a
per-cell symmetric 3x3 field. Keep `StableNeoHookeanMembrane` unchanged because
the skin resultant remains uniform.

The contingency also requires explicit schema updates in:

- `src/21-neutral-converge.py`: free-coordinate indices, fixed skin index,
  projection, spectra, checkpoint shape, and continuation resume checks;
- `src/30-joint-pilot.py`: hard-coded shared count, neutral reference prior,
  Adam/checkpoint shape, stationarity diagnostics, and snapshot contract;
- `src/joint_diagnostics.py`: anchor and reconstructed-field summaries instead
  of the constant 20-vector assumptions;
- `src/09-validate-face-gradients.py` and the full-face derivative check:
  probe at least one anchor in every tissue and one spatial contrast mode;
- `src/31-render-final.py` and `src/46-render-optimization-state.py`: render
  reconstructed per-cell baseline principal stresses and record the new schema.
- `src/50-run-prepared-joint-sequence.py`: move the skin coordinate from 18 to
  78 and require the spatial checkpoint/basis hashes before orchestration.

Before the nonlinear probe, extend the exact-rest basis audit from 18 to
78 bulk columns and inspect both raw and dual-volume-weighted residuals.
Audit 005 performs that screen; the residual improvement is small. Force-space
least squares and nonlinear displacement fitting are distinct objectives, so
the screen supports a bounded test rather than a claim of shape feasibility.
The probe freezes `0.5 * 100 * mean_t R_smooth,t` as a separate objective term,
outside the unchanged `0.001 * prior_total`. Its weight may not be weakened
in response to an inconvenient result.

## Memory

The 80 float64 shared coefficients occupy 640 bytes. Support-only basis storage
for the 4/4/5 anchors contains 6,110,571 weights: 23.31 MiB as float32, plus
5.55 MiB of int32 cell IDs. Reconstructed float64 six-coordinate fields over
all positive tissue cells occupy 66.63 MiB if retained simultaneously for
autograd. A conservative incremental peak is therefore about 95.5 MiB; tissue
or cell chunking can reduce the transient part. The three full float64
per-tetrahedron 3x3 stress buffers are 236.18 MiB, but those buffers already
exist in the current constant implementation after `expand().contiguous()`.
The dense four-expression activation parameter remains 52.78 MiB and is not
increased by this contingency.

This design is a fallback hypothesis, not evidence that the current
continuation will fail or that the inferred spatial field would be anatomical
truth.
