# Face execution decisions

This file records the implemented experiment, including departures from the initial literature proposal. It is not a list of completed results; completed cases have their own `summary.json`, solver receipts, source copies, and exact equilibrium fields.

## Fixed model and observations

All primary comparisons use the same prepared face: 228,660 vertices, 1,146,517 positive rest tetrahedra, 35 named expression-muscle regions, and 120,020 active cells. Chewing muscles, eye movers, neck muscles, fascia, and tendons remain passive. Existing rigid-bone constraints and the additional fixed artificial cut are preserved. The jaw is fixed; there is no fitted jaw pose. Contact is absent. The skin is the corrected `IsFace` surface with 29,899 triangles, a 1 mm plane-stress Koiter shell, and zero prestrain.

The target is the supplied `Smile` displacement, not a measured physiological response or a known equilibrium. The objective uses lumped reference-surface area weights on finite `IsFace` targets. Its displacement RMS is 5.095908 mm. The fixture summary also records 5.310139 mm, which is the different, unweighted point RMS. The target is displayed only on the observed surface. No interior target displacement is invented.

## Activation comparisons

`FiberRegion` has one nonnegative scalar per named muscle: 35 parameters. `FiberModes` uses four nonnegative Gaussian partition controls per muscle: 140 parameters. Their convex interpolation preserves the contraction bound. The fixed directions come from rest-geometry PCA for elongated regions and ellipse tangents for three orbicularis regions. These are estimates, and broad or branching muscles can have unreliable directions. Neither construction reads the smile target.

The code consumes the inverse active map

```text
A_inv = exp(a) f f^T + exp(-a/2) (I - f f^T),
0 <= a <= -log(0.65).
```

The natural active stretch along the fiber is therefore between 0.65 and 1, with equal transverse expansion and determinant one. This 35% shortening cap is an experimental kinematic bound, not a measured facial-muscle limit. A magnitude prior with weight 0.001 uses `a_ref = -log(0.8)`.

`Region5Modes` is the alternative that does not require fibers: four spatial controls per region, each a symmetric trace-free log tensor, for 700 parameters. The Frobenius bound is `sqrt(1.5) * -log(0.65)`. It guarantees a positive-definite, determinant-one active map with bounded principal stretches. It permits shear and unequal transverse response, so it is a diagnostic kinematic control space rather than identified muscle physiology. The Gaussian controls are broad spatial modes; they are not anatomical compartments and are not the Laplacian eigenmodes proposed in the literature memo.

`Raw6` is an independent symmetric inverse-active matrix offset in each selected active tetrahedron: 720,120 parameters, without magnitude or adjacency priors. The current comparison still rejects non-positive active matrices and equilibria below the common geometric floor. It is thus an admissible raw baseline, not a reproduction of every historical unconstrained run.

## Material semantics

The initial face material B uses fat Young's modulus 3 kPa, passive muscle 24 kPa, skin 24 kPa, and Poisson ratio 0.46 for these soft tissues. Aponeurosis remains 100 kPa with Poisson ratio 0.35. The skin/fat contrast is informed by the small-strain ratio in a published Yeoh face model; passive muscle remains an uncertain experimental choice. These are not transplanted Yeoh nonlinear coefficients or a calibration of this subject.

The production polynomial Stable Neo-Hookean energy is

```text
W = mu/2 (I2 - 3) - mu (J - 1) + lambda_code/2 (J - 1)^2.
```

Its infinitesimal first Lamé coefficient is `lambda_code - mu`. These experiments consequently use `lambda_code = lambda_classical + mu`, so the stated E and nu are the actual small-strain parameters. The early three-step debugging pilot predates this correction and is excluded from matched material comparisons.

The fat-law counterfactual uses the existing logarithmic Neo-Hookean law

```text
W = mu/2 (I2 - 3) - mu log(J) + lambda_classical/2 log(J)^2.
```

It has the same infinitesimal constants but resists severe compression much more strongly and requires positive J. This is a constitutive sensitivity, not proof of a calibrated adipose law. Fixed-control tests change only fat law or fat Poisson ratio, or only the two depressor-labii controls. They re-equilibrate the full face without refitting activation. Failed trials are retained as failures; failed inner line searches must not become accepted physics states.

## Numerical and visual interpretation

All displayed simulated geometries come from saved equilibria. Optimization history is a sequence of equilibrium iterates, not physical time. Geometry is not smoothed for rendering. A cutaway is a display selection through the original mesh, not a new mechanical boundary.

The projected optimizer uses the same area-weighted target objective, reference muscle-fraction volume normalization, and positive-J acceptance floor. A determinant floor of 0.2 is a numerical rejection boundary, not a claim that an 80% local volume loss is physiologically reasonable. Full-tissue J distributions, principal stretches, boundary intersections, lip separation, and the relationship to fixed vertices determine how an endpoint should be interpreted. Static intersection tests do not replace continuous collision detection.

The reported projected-gradient quantity covers only the activation box or ball. It does not include a determinant-constraint gradient or multiplier. A line search stopped by the determinant floor therefore does not establish stationarity of the full constrained problem. Mixed tissue is attached to pointwise fixed mandible vertices without sliding or a compliant bone interface, which conditions the interpretation of the localized compression.

A smoother but nearly immobile face does not count as a successful inverse reconstruction. The 35-scalar first run stopped at the determinant floor with a nonstationary projected gradient, 4.870963 mm residual and target projection amplitude 0.047647. Its smooth exterior therefore establishes neither adequate smile recovery nor acceptable internal strain. It motivated the fixed-control material probes and the broader constrained spaces.

Intrinsic surface high-pass diagnostics at 2, 5, and 10 mm are supporting evidence. The main evidence is the same-camera, full-face and mouth comparison, target overlay, actual iteration history, estimated fiber display, and internal compression cutaway. The supplied target has no known physically correct interior, so residual roughness cannot by itself distinguish an artifact from an unattainable target component.

## Matched inverse screen after the fixed-control probes

The successful fixed-control probes support stable fat at nu = 0.49 as the first comparison material: it increases minimum J from about 0.20 to 0.53 with essentially unchanged target fit. The logarithmic law at nu = 0.49 changes that minimum to about 0.55 but takes substantially more inner iterations. This is a practical selection for the present solver and fixture, not a claim that the polynomial law is more anatomically accurate.

The matched screen starts each of Raw6, FiberModes, and Region5Modes from zero activation and rest, with 40 planned outer steps, forward rtol 1e-5 and atol 1e-12, adjoint rtol 1e-7, and the same skin, muscle, target, and geometry. Each receives an initial end-to-end finite-difference audit. A final independent reset from rest was planned for each; Raw6 was stopped early and did not receive that check. All three use the strict inner line search with up to 30 halvings; exhausted or nonfinite searches are rejected before the optimizer can commit them. A fixed outer budget is an exploratory comparison, not a guarantee of equal optimization convergence across parameter spaces. Actual termination and verification are identified by the result-directory receipts.

## Final execution departures and additional checks

Raw6 was stopped after accepted step 8 following repeated long forward failures and limited additional progress. Its saved accepted equilibrium was exported by CPU without re-solving; it is a diagnostic checkpoint, not a converged endpoint. Its initial finite-difference comparison passes the 2% criterion at epsilon 0.001, while epsilon 0.0003 gives a precision-sensitive 5.30% discrepancy. FiberModes exhausted 40 steps; Region5Modes stalled at step 8. Neither met the projected-gradient tolerance. Their independent rest checks succeeded.

The additional learned-axis model fits a single uniaxial contraction in each muscle: 32 non-ring regions use `H = 1.5 v v^T - 0.5 ||v||^2 I`, with `||v||^2 <= amax`; three orbicularis regions retain their prepared circumferential axes and one scalar amplitude. This gives 99 free coordinates in 105 stored slots. It begins at target-independent `a = 0.02`, rather than the zero initialization of the main screen, because the derivative at v = 0 vanishes. It stopped at step 13 near the determinant floor, with an independent rest check. Fitted directions are latent axes, not anatomy estimates validated against measurements.

Run 27 holds the FiberModes controls fixed and changes only passive muscle ν to 0.49. Run 28 changes only fat ν to 0.499; its warm-start equilibrium succeeds, but its independent direct rest solve fails at 10,000 iterations. Run 29 checks the same q and A_inv through ten equal activation increments from analytical rest. All nonzero stages pass the original strict tolerances and reach an endpoint close to the warm-start result. This is a separate quasistatic continuation check, not inverse optimization, physical time, or proof of uniqueness. Failed attempts are retained.
