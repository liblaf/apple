# Human-face inverse physics: activation space, materials, and surface artifacts

**The bounded activation spaces produce smoother mouth surfaces at much smaller expression amplitude. None of these runs establishes a plausible full inverse reconstruction.** Two problems are visible in the saved equilibria: the raw activation field accompanies irregular surface motion and large active volume changes, while severe internal compression persists even with simple, volume-preserving muscle contraction. The unequal motion and incomplete optimization prevent an equal-expression causal comparison of surface artifacts.

[Open the interactive face comparison](viewer.html) · [Inspect the compressed tetrahedron in 3D](tet-viewer.html) · [Download figure assets](records/figures.zip) · [Download scripts and evidence records](records/reproducibility.zip)

## Inspect the geometry first

The viewer opens at the saved endpoint. Start with **Raw6 diagnostic**, select the mouth camera, and switch between reference and endpoint. Then select **FiberModes** with the same camera. The former has conspicuous lip-area folds and dents; the latter is smoother but moves much less. Turn on the supplied-target overlay to see the remaining expression error. The target overlay contains only the observed skin, with no fabricated interior motion.

![Raw6: reference at left, accepted diagnostic step 8 at right.](../data/75-final-face-comparisons-caption-legend/raw6-step8/mouth-closeup.png)

Raw6 is an early, nonconverged checkpoint. Its lower target error is accompanied by visibly irregular geometry and unrestricted active volume changes. It is a failure example, not the optimum of its much larger control space. [Inspect the full-face image](../data/75-final-face-comparisons-caption-legend/raw6-step8/full-head.png).

![FiberModes: reference at left, saved 40-step endpoint at right.](../data/75-final-face-comparisons-caption-legend/fiber-modes-fat049/mouth-closeup.png)

FiberModes preserves a much smoother mouth contour, but this comparison does not establish artifact removal at equal expression amplitude. Its surface displacement RMS is only 0.63 mm, against 5.10 mm for the target. [Inspect the full-face image](../data/75-final-face-comparisons-caption-legend/fiber-modes-fat049/full-head.png).

The viewer also provides a full-face camera, a three-quarter view, material cutaways, estimated fiber glyphs, and saved iteration histories. Geometry is shown at **1× displacement**, using the actual saved mesh. Averaged lighting normals improve illumination; vertex positions are not smoothed. The fixed cutaway cohort is identical across states and changes only what is displayed. [FiberModes history](fiber-modes-fat049/evolution.mp4) contains independently solved optimization iterates, not physical time or an interpolated motion.

## What the controlled comparisons show

All three primary screens use the same face, target, fixed jaw, skin, passive muscle, and stable fat at Poisson ratio 0.49. They start at zero activation and rest. Their planned budget is 40 outer iterations; the actual termination differs. The later learned-axis test has a separate nonzero initialization and is not an initialization-matched ranking.

| Saved result | Free controls | Target RMS error | Surface motion RMS | Minimum volume ratio J | Termination |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw6 | 720,120 | 3.459 mm | 2.354 mm | 0.350 | Diagnostic stop after accepted step 8; no independent rest reset |
| FiberModes | 140 | 4.838 mm | 0.627 mm | 0.351 | 40-step budget; nonstationary |
| Region5Modes | 700 | 4.631 mm | 0.685 mm | 0.200 | Line search stalled at step 8 near the geometric floor |
| Learned regional axes | 99 | 4.866 mm | 0.571 mm | 0.200 | Line search stalled at step 13 near the geometric floor |

The rest state has target RMS error 5.096 mm. All RMS values use the same reference-surface area weights. The determinant floor, J ≥ 0.2, only rejects extreme numerical states: retaining 20% of a tetrahedron's original volume is not a physiological acceptance criterion. Positive J and a successful equilibrium solve alone do not make a face plausible.

Raw6 permits an independent symmetric inverse-active matrix in every selected muscle tetrahedron. Its active map must remain positive definite at an accepted endpoint, but its magnitude, volume change, and spatial variation are otherwise unrestricted. At step 8, 91.0% of the muscle-fraction-weighted active volume has det(A_inv) outside [0.9, 1.1], and 18.2% has at least one natural active principal stretch below 0.65. [The exact activation audit](../data/47-raw6-activation-audit/summary.json) verifies these tensors directly from the saved controls. Its initial finite-difference gradient comparison agrees to 0.42% at perturbation 0.001; the smaller 0.0003 perturbation disagrees by 5.30%, indicating sensitivity to equilibrium-solver precision. The run was stopped after repeated long failed forward trials and slow progress; its saved accepted equilibrium was exported without a new solve. This weakens any quantitative optimization ranking.

FiberModes has four broad nonnegative Gaussian controls in each of 35 expression muscles. A convex partition interpolates the scalar contraction, so every cell respects the same bound. Region5Modes uses the same spatial partition with five trace-free tensor components per control. It does not require fibers, but it permits shear and unequal transverse response and is therefore a diagnostic kinematic space, not identified muscle physiology.

The learned-axis test keeps one uniaxial contraction per muscle while fitting its axis in 32 non-ring regions. Three orbicularis regions retain circumferential tangents. Its 35 × 3 stored array has 99 free coordinates because two coordinates in each ring are fixed. It starts at a small, target-independent contraction a = 0.02 to avoid the zero derivative of its squared-vector parameterization. Learned axes remain latent fitting variables, not recovered anatomy. This run also reaches the compression floor with little target recovery; fitting axes did not resolve the difficulty on this trajectory. Its independent rest solve differs by 0.000059 mm surface RMS. The viewer's fiber checkbox displays prepared geometry estimates; fitted axes and their validity mask are separate fields in the saved VTU.

The endpoint audits find no inverted tetrahedra or new static edge-face intersections on either the complete exterior or observed skin. This does not rule out visual artifacts: Raw6 has 23,417 cells with J < 0.8, compared with 35 for FiberModes and 9 for Region5Modes. At a 2 mm intrinsic high-pass scale, mouth-normal motion RMS is 0.149 mm for Raw6, 0.042 mm for FiberModes, and 0.036 mm for Region5Modes. The latter two also have much less total motion and larger high-pass target residuals, so these values cannot establish smoothing at matched expression. Full-face and mouth diagnostics at 2, 5, and 10 mm are retained in the [endpoint audit](../data/48-primary-endpoint-audits/summary.json) and [Raw6 audit](../data/46-raw6-step8-audit/summary.json).

## Internal compression responds to fat material and nearby actuation

An earlier 35-scalar run with fat Poisson ratio 0.46 stopped at J = 0.200, despite a target error of 4.871 mm and only 0.482 mm surface motion. The worst tetrahedron, cell 662949, is mostly passive fat, has an identity active map, and includes one fixed mandible vertex. Its rest shape is not an extreme sliver. The nearby actuator is depressor labii; neither a new surface intersection nor a tiny defective rest tetrahedron explains this local collapse.

The following forward tests hold all other controls and materials fixed and solve again from rest. They do not refit the expression.

| Change from the 35-scalar baseline | Target RMS error | Global minimum J |
| --- | ---: | ---: |
| Replay baseline, stable fat ν = 0.46 | 4.871 mm | 0.200 |
| Set both depressor-labii controls to zero | 4.903 mm | 0.533 |
| Change only stable fat to ν = 0.49 | 4.872 mm | 0.527 |
| Change only fat to logarithmic Neo-Hookean, ν = 0.49 | 4.872 mm | 0.546 |

Thus local compression can be reduced substantially without obtaining a better smile fit. Removing a nearby actuator and changing only fat compressibility each alter the compression, demonstrating sensitivity to actuation and fat material response. The present experiment does not separately identify the effect of a sliding bone interface because that interface was not changed.

The learned-axis run's worst cell is also inactive fat: cell 691949, with two fixed vertices, no artificial-cut incidence, J = 0.200, and minimum principal stretch 0.123. Thus bounded, volume-preserving activation does not by itself bound the strain of neighboring passive tissue. The fixed attachment remains a candidate contributor that needs its own counterfactual.

![Exact saved vertices of the original worst tetrahedron, with a fixed vertex marked.](../data/44-material-tet/cell-662949-four-states.png)

This figure and the [tetrahedron viewer](tet-viewer.html) track **the same cell 662949** across material replays. Its J values are 0.200, 0.540, and 0.637; the latter two are not the global minima in the table, because the most compressed cell moves elsewhere. The four vertices, edges, scale, camera, and rest-defined coordinate frame are preserved. [Download the separate PDF figure](../data/44-material-tet/cell-662949-four-states.pdf).

Changing only passive muscle ν from 0.46 to 0.49 at the FiberModes endpoint slightly worsens minimum J, from 0.351 to 0.337. This is a fixed-control forward sensitivity, not a new inverse fit. Changing only fat ν to 0.499 raises minimum J to 0.600, with target error 4.842 mm and motion RMS 0.574 mm. The direct independent rest solve failed at 10,000 iterations, so that initial attempt alone was insufficient evidence. A separate run then increased the same controls from zero through ten equal increments; all ten nonzero strict solves succeeded without relaxing tolerances. Its endpoint agrees with the earlier warm-start equilibrium to 0.000050 mm target-surface RMS. This supports the fixed-control material response along an independently constructed continuation path; it does not turn the failed direct reset into a success or establish uniqueness. [Continuation and exact-control receipts](../data/29-fat0499-ramp/summary.json).

Two initial forward probes failed and are retained: the half-depressor control exhausted the strict line search at an energy difference near numerical precision, and logarithmic fat at ν = 0.46 did not reach the requested equilibrium tolerance in 10,000 iterations. Neither is displayed as a successful counterfactual. [Material replay evidence](17-material-replays.md) records the completed and failed cases.

## Activation and material choices

The scalar models use the inverse active map

```text
A_inv = exp(a) f fᵀ + exp(-a/2) (I - f fᵀ)
0 ≤ a ≤ -log(0.65)
```

The corresponding natural active stretch contracts along the fiber by at most 35%, expands equally in both transverse directions, and preserves volume. The 35% cap is an experimental kinematic bound, not a measured facial-muscle limit. The objective adds a volume-weighted magnitude penalty of 0.001 normalized by a_ref = -log(0.8). Broad spatial controls remove cell-scale freedom; these face runs do not also apply a nonzero adjacency penalty. This distinguishes the executed model from the preliminary proposal.

The direction field uses rest-geometry PCA for elongated regions and ellipse tangents for three orbicularis regions. It never reads the smile target. Regional muscle controls and anatomical fiber fields have precedent in [Sifakis et al.](https://pages.cs.wisc.edu/~sifakis/papers/activations_siggraph_2005.pdf), but their normalized force activation is different from the active-strain bound used here. PCA is particularly uncertain for fan-shaped or branching muscles. An anatomy-transfer approach such as [Cong et al.](https://diglib.eg.org/items/f0e4e8eb-dc11-48cd-b4ad-1587800e86ee) would supply stronger anatomical information if a suitable template becomes available.

The selected reference scales are fat E = 3 kPa, passive muscle E = 24 kPa, skin E = 24 kPa with thickness 1 mm, and aponeurosis E = 100 kPa. Muscle and skin ν = 0.46; aponeurosis ν = 0.35. The primary screen uses fat ν = 0.49 following the fixed-control sensitivity. The skin/fat contrast is informed by [Picard et al.'s facial Yeoh model](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873870/), whose small-strain shear scales differ by approximately eight. Our passive-muscle scale is an experimental choice, and these stable Neo-Hookean tissues are not that paper's calibrated nonlinear Yeoh material. [The literature memo](12-literature.md) separates face-specific evidence from unsuitable back-skin tensile moduli and from proposed tests that were not run.

The production stable energy has infinitesimal first Lamé coefficient λ_code − μ. The experiment therefore uses λ_code = λ_classical + μ so the reported E and ν have their intended small-strain meaning. The logarithmic law uses λ_classical directly. This correction is recorded in the source and material receipts; the early debugging pilot predates it and is excluded from the comparisons above.

## Evidence limits and the next decision

The model contains 228,660 vertices and 1,146,517 tetrahedra. Activation is restricted to 120,020 cells in 35 named expression-muscle regions. The supplied Smile field is an observation target with unknown physical interior and equilibrium status. The jaw is fixed, mixed tissue is coupled to fixed bone vertices without sliding, contact is absent from the mechanics, skin has no prestrain, and fiber directions and tissue stiffness are uncertain. Static endpoint intersection audits cannot replace contact forces or continuous collision detection.

The numerical screen does not establish that these control spaces cannot fit a smile: none of the inverse endpoints satisfies the activation-only projected-gradient tolerance. That diagnostic also omits the determinant constraint's multiplier. The raw run has no independent rest reset. The other saved inverse endpoints were independently solved again from rest; their receipts record the resulting differences. A fixed iteration budget and different early stopping rules do not provide equal-convergence or runtime comparisons.

The practical next model should retain bounded, low-dimensional, volume-preserving contraction. Before widening activation freedom, inspect the lower-face attachment model and jaw pose, then test reviewed muscle paths and a better constrained volumetric material formulation. A nearly incompressible penalty can reduce collapse but also makes equilibrium harder to solve. These are the next hypotheses to test, not changes already validated by this report. A future acceptance test must require both adequate expression recovery and acceptable internal strain; low surface roughness by itself is insufficient.

The earlier block study (private preview omitted) provides a controlled mechanism/debugging case. Its clean-target results are not substituted for anatomical face validation.

## Reproduction and records

Use the [reproduction guide](../README.md), [executed protocol](16-execution-decisions.md), and [machine-readable final comparisons](../data/49-final-comparisons/comparisons.json). The matching [CSV](../data/49-final-comparisons/comparisons.csv) includes termination and audit status. Exact saved fields, source snapshots, input hashes, solver receipts, and failed attempts remain in the local experiment directory. The downloadable reproduction archive contains scripts and compact evidence records; the original source mesh and full NPZ/VTU states remain separate, as stated in the guide.

The viewer assets, standalone figures, and report are served privately through the Tailnet. Static geometry, local JavaScript dependencies, and linked file availability were checked. The scientific images were visually inspected. Interactive browser inspection timed out, so WebGL interaction and mobile layout have not been visually verified in a browser.
