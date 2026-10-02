# Fiber-free tensor active stress

**The six-coordinate tensor actuator is implemented and validated, but it has not recovered the Smile target in this 64-update face comparison.** Smoothness and the soft rank preference strongly change the stress field. Their lower absolute surface high-pass values accompany less expression motion and larger fitting errors. The Raw6 reference fits much better, with 88 inverted tetrahedra. None of these endpoints establishes a converged inverse solution.

[Open the interactive 3D comparison](viewer.html) · [Download PNG / PDF figures](records/figures.zip) · [Download scripts and evidence](records/reproducibility.zip)

## The saved face results

| Model, actual update 64 | Fit RMS (mm) | Motion RMS (mm) | Target projection | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| Raw6 reference | 1.018 | 4.754 | 91.5% | 88 |
| PSD | 3.820 | 1.560 | 26.6% | 1 |
| PSD + smoothness | 4.005 | 1.341 | 22.6% | 0 |
| PSD + smoothness + rank | 4.108 | 1.280 | 20.7% | 1 |

The common rest state has 5.096 mm fitting RMS and zero motion. A target projection of 100% would reproduce the requested displacement along its own direction; it is not an overall fit score.

Fit and motion below are area-weighted vector RMS values on the same face vertices. The optimizer uses uniform Cartesian mean-square fitting error; that objective and the area-weighted diagnostic are both retained in the evidence. Target projection measures how much of the requested displacement is produced along the target direction. All primary meshes are the actual state after 64 updates, with no best-iteration substitution.

![The four saved endpoints and exact Smile target at common scale](../data/50-face-comparison/face-comparison-front.png)

![Mouth detail with one common camera and true deformation scale](../data/50-face-comparison/face-comparison-mouth.png)

The static figures use one skin color, identical lighting, and a shared camera within each panel. The 3D viewer provides the saved rest state, steps 0 / 16 / 32 / 48 / 64, endpoint, target skin, and material display. It shows the full exterior of the saved tetrahedral states. The target has only measured skin displacement; no interior target tissue is invented.

## Does stress smoothing reduce surface detail?

Smoothness reduces full-face displacement high-pass RMS from 0.0993 to 0.0849 mm, but its normalized ratio changes only from 0.1788 to 0.1750. The corresponding mouth ratio changes from 0.2036 to 0.1972. Most of that absolute reduction accompanies the smaller deformation amplitude, despite a large reduction in normalized stress-field variation.

Adding the rank penalty reduces the displacement ratios further, to 0.1471 on the full face and 0.1536 around the mouth. It also worsens the mouth target-residual high-pass RMS from 0.4553 to 0.4815 mm. The exact target has 0.6208 mm of mouth displacement high-pass content, compared with only 0.1917 mm in the rank-penalized result. Removing that detail is not successful expression recovery.

Raw6 has more high-pass displacement than the target and fits it much more closely overall. Its residual still contains substantial high-pass content, and its inverted tetrahedra prevent treating the low fitting error as a physically acceptable solution.

Each cell gives **high-pass RMS in mm / normalized ratio** at 5 mm. The ratio uses the total normal RMS of that same field and region.

| State | Face displacement | Mouth displacement | Face target residual | Mouth target residual |
| --- | ---: | ---: | ---: | ---: |
| Raw6 | 0.292 / 0.203 | 0.662 / 0.255 | 0.182 / 0.299 | 0.325 / 0.341 |
| PSD | 0.099 / 0.179 | 0.244 / 0.204 | 0.183 / 0.151 | 0.428 / 0.237 |
| PSD + smoothness | 0.085 / 0.175 | 0.209 / 0.197 | 0.193 / 0.153 | 0.455 / 0.240 |
| PSD + smoothness + rank | 0.078 / 0.147 | 0.192 / 0.154 | 0.203 / 0.160 | 0.481 / 0.271 |
| Exact target | 0.254 / 0.158 | 0.621 / 0.226 | 0.000 / undefined | 0.000 / undefined |

The measurement uses the frozen exterior skin of 15,299 vertices and 29,899 triangles, all with finite original Smile targets. It separates two scalar fields: displacement along the frozen rest normal, and target residual along that same normal. A cotangent surface operator removes detail below a named spatial scale. The primary scale is 5 mm; 2 and 10 mm are fixed sensitivity views. The mouth region is an intrinsic 10 mm neighborhood of the lip vertices. Each ratio divides high-pass RMS by the total RMS of the same field in the same region. A zero denominator is reported as undefined.

The target's own high-pass displacement is included because an expression contains real fine-scale detail. Neither the absolute value nor its normalized ratio identifies an artifact on its own. Tangential surface motion is outside this measure.

![Normal high-pass surface displacement and target residual at 5 mm](../data/41-face-comparison/surface-highpass.png)

[PDF figure](../data/41-face-comparison/surface-highpass.pdf)

[Endpoint comparison CSV](../data/41-face-comparison/endpoint-comparison.csv) · [Surface comparison CSV](../data/41-face-comparison/surface-comparison.csv) · [All scales and surface-map provenance](../data/40-surface-measurements/summary.json)

## What the controls mean

Each of the 288,235 active tetrahedra has six independent symmetric coordinates: 1,729,410 controls in total. No fiber direction is read or supplied. For deformation gradient F, active tensor Q, and saved muscle fraction φ, the muscle contributions to energy density and first Piola stress are

```text
W_muscle(F, Q) = φ [W_stable(F) + ½ Q : (FᵀF − I)]
P_muscle(F, Q) = φ [P_stable(F) + FQ]
0 ⪯ Q ⪯ Qmax I
```

The entire muscle potential is multiplied by the saved muscle fraction. Fat and aponeurosis keep their historical passive energies. Setting Q to zero exactly restores the passive material. If Q = T ffᵀ, the new term reduces to the previous rank-one tension law. A full PSD tensor can express tension along more than one principal direction; those inferred directions are control directions, not independently validated anatomical fibers.

The nominal Young's-modulus inputs for muscle, fat, and aponeurosis remain 0.03, 0.003, and 0.1 MPa, with Poisson-ratio inputs 0.49, 0.49, and 0.35. The historical conversion to the implemented stable model is preserved. Skin energy is zero, and contact is disabled. Keeping passive parameters fixed does not remove activation's own finite tangent: at fixed Q, δP_active = φ δF Q.

The cap is Qmax = 0.3020134228 MPa = 10 Qref, with Qref = 3 μ = 0.0302013423 MPa. It bounds the reference active stress tensor. It is an exploratory finite bound, not a physiological calibration or a uniform Cauchy-stress bound under arbitrary compression.

Six orthonormal coordinates represent Z = Q / Qref. Adam updates those coordinates directly, then eigenvalue projection enforces 0 ⪯ Z ⪯ 10 I outside differentiation. The zero initial state therefore retains a nonzero fitting gradient. Eigensolvers process 1,024 tensors at a time to bound CUDA workspace while retaining every tetrahedron.

## Controlled comparison and optimization

The three tensor arms share the same energy, stress cap, initial rest state, zero controls, zero Adam moments, learning rate 0.3, epsilon 0.01, and 64 updates. Their only objective differences are the two penalties:

```text
PSD:                   data fitting
PSD + smoothness:      data fitting + λs S
PSD + smoothness+rank: data fitting + λs S + λr B

S = ℓ² / V × Σedges wij ‖Zi − Zj‖²
B = 1 / V × Σcells Ve φe [(trace Ze)² − ‖Ze‖²]
M = 1 / V × Σcells Ve φe ‖Ze‖²       (diagnostic only)
```

Here φ is muscle fraction, V is total active muscle volume, and ℓ = 5 mm. Graph conductance is shared-face area divided by centroid distance, multiplied by the harmonic mean of muscle fractions. Its 501,409 edges stay within muscle labels. The rank penalty is soft: all six coordinates remain available. It also vanishes at zero stress, so stress magnitude, motion, and fitting error accompany the rank statistics.

A discarded three-update pilot fixed λs = 5.9051714684 and λr = 119.6289338070 before the final runs. At the pilot's step 3, the weighted smoothness and rank gradients were set to 25% and 15% of the fitting-gradient RMS. The selection receipt also simulates their effect on the next projected Adam update using the same stored moments. All final arms restart from zero; the pilot is not a hidden prefix. The magnitude penalty has zero weight.

Raw6 is a newly initialized historical active-strain reference at the same update count. It uses a different constitutive law and direct symmetric coordinate scale. Equal Adam settings therefore do not establish equal physical update authority. Its comparison with the tensor arms cannot isolate the PSD constraint or establish either model's reachable limit.

![Fitting, motion, target projection, stress variation, stress magnitude, and principal-tension mixing through the fixed run](../data/41-face-comparison/optimization-traces.png)

[PDF figure](../data/41-face-comparison/optimization-traces.pdf)

At update 64, smoothness reduces S/M from 17.395 to 1.258, a 92.8% reduction relative to PSD alone. Q RMS falls 20.9%, motion falls 14.0%, and fitting RMS rises 4.8%. The change in S/M rules out uniform stress rescaling as the sole explanation: the spatial distribution of stress has changed as well.

Adding the rank penalty reduces the normalized principal-tension mixing measure from 0.2346 to 0.00661, while the largest principal tension's share rises from 86.0% to 99.5%. Q RMS falls a further 5.5% and motion a further 4.6%. This is a strong anisotropy change, rather than just uniform amplitude reduction. It does not demonstrate recovery of anatomical fibers.

The final maximum Q eigenvalues are 0.05133, 0.02936, and 0.02700 MPa for PSD, smoothness, and smoothness + rank. The upper cap is inactive throughout all three traces. The lower PSD boundary is active, and the optimization is still making fitting progress at its fixed endpoint. Raising the upper cap alone therefore does not address the current limitation; these runs do not separate finite optimization progress from restrictions of the admissible actuation space.

## Small-model checks

Two shared-face tetrahedra have independent six-coordinate Q fields and mixed muscle fractions 0.3 and 0.8, with fat filling the remainder. Six fixed coordinates remove all rigid modes. The compatible target is a common deformation gradient diag(0.82, 1.04, 1). A local force-balance candidate is projected into the PSD stress box and then tested in an actual equilibrium solve.

| Compatible-target method | Cartesian RMS position error, rest edge = 1 |
| --- | ---: |
| Qref-capped local balance | 0.0446502 |
| 10 Qref-capped local balance | 4.66 × 10⁻¹⁰ |
| Original active strain | 0.00494956 |
| Active strain with 10× muscle Young's modulus | 0.000573852 |

The finite 10 Qref tensor candidate reaches this compatible fixture to solver tolerance without changing passive stiffness. Its largest eigenvalue is 0.116913 MPa, below the cap. The lower-cap balance candidate saturates all six eigenvalues. These are specific feasible candidates; they are not global reachability results. The 10× modulus row is a stiffness sensitivity and does not use a matched stress budget.

For the incompatible case, the two cells request lengths 0.75 and 0.90 for the same physical shared edge. Any one realized length misses at least one request by at least 0.075. The plotted position errors for that case are measured against a feasible nodal compromise with edge length 0.825. They are a different quantity from the incompatible-request bound and cannot contradict it.

![Two-tetrahedron feasibility and shared-edge incompatibility](../data/13-small-study-comparison-v5/contraction-and-target-error.png)

[PDF figure](../data/13-small-study-comparison-v5/contraction-and-target-error.pdf)

The loaded Q = 0 solve exactly matches the loaded passive solve. All 12 reported small-model states pass the declared equilibrium-gradient infinity-norm threshold of 10⁻⁸. The bounded-search rows in the supporting figure are fixed-budget candidates, not optimizer-convergence claims.

[Small-model methods and evidence](12-small-model-study.md) · [Corrected cap interpretation](../data/12-small-model-study-v4/interpretation-correction.json)

## Validation and reproducibility

The actual Warp implementation passes energy, force, Hessian product, Hessian diagonal, Hessian quadratic, material-adjoint, rotation-covariance, and rank-one reduction checks. Q = 0 agrees exactly with the unmodified passive implementation on the audited state. The full 288,235-cell CUDA projection audit retains all controls and uses 584.6 MiB peak allocated memory in its isolated process. A full face solve requires additional mesh and solver memory.

All four arms have 65 accepted forward-and-adjoint states, including rest and actual update 64. Numerical success coexists with 88 inverted tetrahedra in Raw6, one in PSD, none in PSD + smoothness, and one in PSD + smoothness + rank. Their minimum det(F) values are −2.777, −0.01146, 0.09076, and −0.1402. Raw6 also has 120 cells with a non-positive eigenvalue in its inverse activation tensor. The bounded PSD stress restriction did not ensure non-inverted elements in this fixture.

Numerical validity uses successful forward and finite adjoint solves. Forward PNCG allows 5,000 iterations with relative tolerance 5 × 10⁻⁴ and absolute tolerance 10⁻¹⁰. The adjoint retains the historical CG / MINRES selection, 10,000 iterations, and relative tolerance 5 × 10⁻⁴. No tolerance relaxation, hidden retry, determinant rejection, or appearance filter is used. Tetrahedral inversions are recorded as diagnostics. A valid solve does not by itself establish physical acceptability or inverse convergence.

Each run stores the actual final state, separate best-objective state, explicit history, all controls, optimizer state, solver receipts, and source/input hashes. Post-processing binds the endpoint NPZ, VTU, trace, summary, and rendered geometry. The figure archive contains independent PNG and PDF assets. The evidence archive contains scripts, runtime dependencies, protocols, summaries, traces, and solver records. Full volume meshes, control arrays, and optimizer checkpoints remain in the experiment directory.

Runs use the project interpreter with the explicit noncommitting Cherries profile. Comet records run metadata and metrics; local archives hold the numerical assets. The four face processes shared one RTX 4090, so wall times are not single-process performance measurements. [Runtime and dependency versions](../data/05-runtime/summary.json) · [Archived surface/rendering/publishing helpers and package stubs](../data/06-helper-sources/summary.json)

| Evidence | Record |
| --- | --- |
| Constitutive and adjoint validation | [Methods and checks](10-tensor-active-validation.md) |
| Full-field projection and CUDA memory | [Methods and checks](10b-tensor-controls-validation.md) |
| Frozen face settings | [Selection receipt](../data/14-frozen-face-settings/summary.json) |
| Complete face protocol | [Model, objective, initialization, solver](20-face-comparison-protocol.md) |
| Frozen surface protocol | [Fields, operator, scales, regions, normalization](40-measurement-protocol.md) |
| Verified comparison | [Summary and bound source hashes](../data/41-face-comparison/summary.json) |
| Raw6 face run | [Summary](../data/20-raw6-reference/summary.json) · [Comet](https://www.comet.com/liblaf/apple/cda1d218bdca41f1bc72773bdf7901c4) |
| PSD face run | [Summary](../data/21-psd/summary.json) · [Comet](https://www.comet.com/liblaf/apple/731f27b415634d03ba87ad4fc54249eb) |
| PSD + smoothness face run | [Summary](../data/22-psd-smooth/summary.json) · [Comet](https://www.comet.com/liblaf/apple/9c8c5c45c94d4bc1ae36be2f3c41f82b) |
| PSD + smoothness + rank face run | [Summary](../data/23-psd-smooth-rank/summary.json) · [Comet](https://www.comet.com/liblaf/apple/3e192ae9703a4e8286dd62a208b85306) |

The implementation is in [tensor_active.py](../src/tensor_active.py), [tensor_controls.py](../src/tensor_controls.py), and the [face inverse runner](../src/20-face-inverse.py). The face batch is a fixed-budget diagnostic. Further optimization or an explicit physical update-scale comparison is needed before attributing its residual to model capacity.
