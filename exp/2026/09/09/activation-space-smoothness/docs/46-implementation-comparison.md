# Implementation differences between experiments

The off/on comparisons change smoothness within a model. Comparisons across learned axis, corrected Raw6, and historical PSD also change the control space, initialization, optimizer scaling, and sometimes budget. Raw learning rates, penalty coefficients, and endpoint errors therefore do not provide a controlled ranking across models.

## Model and optimization implementation

Here B is the inverse active-strain matrix, C = B − I, and Z = BBᵀ − I is the common dimensionless effective field. Historical PSD instead represents additive stress Q = μZ directly through a normalized matrix M. S denotes the spatial variation penalty, evaluated on the specified tensor field.

| Implementation | Learned axis | Corrected Raw6 | Historical PSD |
| --- | --- | --- | --- |
| Per-cell controls | 3 values v; 864,705 scalars in total | 6 unscaled symmetric entries q; 1,729,410 scalars | 6 Frobenius-orthonormal symmetric coordinates q; 1,729,410 scalars |
| Map to effective field | C = vvᵀ; B = I + C; Z = (2 + ‖v‖²)vvᵀ | C = sym_unscaled(q); B = I + C; Z = 2C + C² | M = sym_orthonormal(q); Q = Q_ref M; Q_ref = 3μ, so Z = 3M |
| Admissible tensors | C and Z are PSD of rank at most 1; B is positive definite; no magnitude cap | C and B are unconstrained symmetric; Z ≽ −I; no magnitude cap; different signs of B can produce the same Z | Q is PSD of rank 0–3 after spectral projection; eigenvalues limited to 0–0.302013 MPa |
| Muscle implementation | Physical-volume active strain: shear term uses FB, volume terms use det(F) | Same physical-volume active-strain law as learned axis | Passive stable law plus additive ½ Q:(FᵀF − I) |
| Smooth arm | 0.0004450069704 × S(C) | 0.003214147722 × S(C) | 5.905171468 × S(M), equivalent to 0.6561301632 × S(Z) because Z = 3M |
| Initialization | Seed 20260909; ‖v‖² = 0.001; one randomly sampled axis per each of 103 labels, copied to its cells; each cell then optimized independently; zero displacement seed | Exact canonical step-200 controls and saved no-skin displacement; optimizer moments reset for each off/on re-fit | Off/on-64 start at q = 0 and rest displacement; off-1024 continues saved controls, displacement, moments, and counter |
| Adam settings in the original study | Fixed rate 7.857985795; ε = 0.01; β = (0.9, 0.999) | Fixed rate 0.3; ε = 0.01; β = (0.9, 0.999) | Off/on-64: rate 0.3; off-1024: 0.3 through global update 512, then 0.6; ε = 0.01; β = (0.9, 0.999) |
| Update constraints | No control projection, outer backtracking, physical-step cap, or inversion rejection | Same absence of outer constraints as learned axis | Spectral projection after Adam, outside the differentiation graph; upper cap never bound in the recorded runs, but the lower PSD constraint did |

For the same mapped Z, these muscle laws produce the same equilibrium force and deformation Hessian: setting Q = μ(BBᵀ − I) makes their energy difference independent of F. This correspondence does not make their inverse optimization equivalent. The parameterization, allowable Z, initialization, smoothing field, and Adam/projection history still differ. In particular, “learned axis” does not impose one fixed anatomical fiber direction per muscle.

## Run protocol and comparison role

| Run | Start and optimizer | Smoothness term | Executed result | Valid comparison role |
| --- | --- | --- | --- | --- |
| Axis-off | Shared seed-20260909 controls; zero displacement seed; fresh Adam at 7.858 | None | Accepted/evaluated states through 28; fit 6.04 mm; attempted 29 fails. Best fit 3.539 mm at 15 | Common prefix with Axis-on; selected actual-state match is 11/11 |
| Axis-on | Same controls and displacement seed; fresh Adam at 7.858 | 4.45007e−4 S(C) | Completed 128 updates; fit 0.681 mm | Update 128 is unpaired because Axis-off has no corresponding state |
| Raw6-off | Canonical step-200 controls plus saved no-skin displacement; fresh Adam at 0.3 | None | Completed 200 re-fit updates; fit 1.3978 mm | Equal-start, equal-budget off/on pair |
| Raw6-on | Exact same controls and displacement; fresh Adam at 0.3 | 0.00321415 S(C) | Completed 200 re-fit updates; fit 1.4047 mm | Selected actual-state match is endpoint 200/200 |
| PSD-off-64 | Zero controls, rest displacement, fresh Adam at 0.3 | None | Completed 64 updates; fit 3.82 mm | Historical equal-start, equal-budget pair with PSD-on-64 |
| PSD-on-64 | Same zero/rest/fresh start at 0.3 | 5.90517 S(M) | Completed 64 updates; fit 4.03 mm | Historical controlled smoothness comparison |
| PSD-off-1024 | Continues saved controls, displacement, moments, and counter; rate 0.3 then 0.6 after 512 | None | Completed global update 1024; fit 1.54 mm | Longer-run context; no matched smooth continuation |

The learned-axis initial q/C/Z arrays are bitwise identical across off/on, and equilibrated initial displacement differs by 1.38e−14 mm RMS. The first-update q difference is 1.97e−6, exceeding the predeclared 1e−6 gate. The report retains that failed verification gate rather than treating the pair as numerically identical throughout.

## Shared physical and numerical settings

The fixture hashes agree across these comparisons: 288,235 active tetrahedra, 15,302 finite fitted IsFace vertices, the same fixed constraints, and the same 501,409-edge graph connecting neighboring cells within the same muscle. All use zero skin energy and no contact. The data term is uniform Cartesian coordinate MSE over the finite fitted face vertices, multiplied by 10⁶; the reported vector RMS is a separate diagnostic.

The common penalty is S(X) = (ℓ² / V_a) Σ_(i,j ∈ E) w_ij ‖X_i − X_j‖²_F, with ℓ = 0.005 m, V_a = Σ_k V_k f_k, and w_ij = (A_ij / d_ij) · 2f_i f_j / (f_i + f_j). Here f is muscle fraction, A is shared-face area, and d is cell-centroid distance. Edges connect only cells with the same muscle label. The norm is Frobenius; the PSD coordinate norm is identical because its six coordinates are orthonormal. This is a spatial difference penalty, not a magnitude penalty.

| Shared setting | Recorded value |
| --- | --- |
| Passive materials, E in MPa | Aponeurosis: stable, E = 0.1, ν = 0.35; fat: stable, E = 0.003, ν = 0.49; muscle: stable, E = 0.03, ν = 0.49 |
| Forward equilibrium | PNCG; maximum 5,000 iterations; relative tolerance 5e−4; absolute tolerance 1e−10; implementation-default line search, recorded maximum 10 |
| Implicit adjoint | CG followed by MinRes when needed; each maximum 10,000 iterations and relative tolerance 5e−4 |
| Failure and geometry policy | Failed or nonfinite forward/adjoint evaluations stop the run. Inverted tetrahedra and stress spectra are recorded diagnostics; inversions alone do not reject the state |

## Source pointers

- [Current controls, mappings, and initialization](../src/activation_controls.py), [current runner](../src/20-run-case.py), [common physics and solvers](../src/study_physics.py), and [physical-volume muscle law](../src/volume_preserving_active.py).
- [Historical PSD runner](../../../07/tensor-active-stress/src/20-face-inverse.py), [PSD coordinates and projection](../../../07/tensor-active-stress/src/tensor_controls.py), and [additive-stress law](../../../07/tensor-active-stress/src/tensor_active.py).
- [Archived Raw6 re-fit runner](../../../08/local-skin-prestrain/src/20-run-case.py), [retained-evidence verification](../data/50-verification-v2/summary.json), and [selected state metrics](../data/40-comparison/selected-states.csv).
- [PSD rate-switch receipt at global update 512](../../../07/tensor-active-stress/data/102-larger-rate32/resume.json), [final continuation receipt](../../../07/tensor-active-stress/data/102-fit1024/resume.json), and [PSD final summary and cap](../../../07/tensor-active-stress/data/102-fit1024/summary.json). The [protocol](12-learned-axis-plan.md) records that the upper cap did not bind; the audit also checked every recorded primary-lineage trace row, including the historical smooth arm. The final summary alone describes the final state.

This comparison is a read-only audit of recorded implementations and saved runs. No new inverse fit was performed.
