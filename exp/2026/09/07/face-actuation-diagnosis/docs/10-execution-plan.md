# Diagnose face actuation before judging activation priors

This experiment extends the earlier face study after identifying two limitations in its design: most constrained controls were shared across a region or four broad modes, and a hard det(F) >= 0.2 acceptance rule truncated several informative trajectories. Neither the small final displacement nor the active-map determinant established a failure of the face model.

## Quantities and stopping rules

The constitutive calculation consumes the inverse active map A_inv, with elastic deformation G = F A_inv. The final physical deformation is F, and its determinant gives the local final volume ratio. det(A_inv) is a control diagnostic, not the volume change of the simulated muscle tetrahedron. Both fields will be retained and labeled separately.

No endpoint or trial is rejected solely for negative det(F), severe strain, or a non-positive-definite raw activation matrix. The raw baseline is deliberately unconstrained in its symmetric activation coordinates. Constrained parameterizations retain their declared activation bounds; these are not constraints on the final deformation. Successful finite equilibrium and a valid derivative are still required for an inverse step. Actual solver failures are recorded, not relabeled as equilibria. A failed independent rest reset does not discard the previously accepted inverse endpoint.

## Controlled sequence

1. Manually activate selected expression muscles with geometry-estimated directions. Compare the same controls with the existing skin membrane and with skin energy absent. Use several contraction amplitudes and save actual equilibria, including distorted ones. Inspect whether muscle shortening reaches the surface and whether smile elevators, lateral lip pulling, and mouth-ring contraction produce the expected directions of motion.
2. Run a per-tetrahedron Raw6 inverse with skin energy absent. Keep the target, observation weights, fixed vertices, active expression-muscle domain, fat and muscle materials unchanged. Establish the fit and visible roughness of this baseline before selecting a smoothing strength.
3. Retain per-tetrahedron controls and add within-muscle shared-face smoothness. Raw6-S retains six controls per active tetrahedron; G5-S retains five trace-free log-tensor controls; FiberSmooth retains one scalar per active tetrahedron. The objective uses the existing volume/fraction-weighted finite-volume graph penalty. A field can vary within a named muscle. These cases do not collapse the controls to tens of values.
4. Resume a previously floor-limited constrained endpoint with the geometric rejection removed, to isolate how much underfit came from that stopping rule. Use the same activation space, bounds, material and skin as the saved run.

The initial materials and geometry are copied from the previous study. The no-skin condition omits the Koiter potential entirely, rather than changing its stiffness by a small factor. The skin mesh remains available only to define the identical observation weights. The jaw and fixed constraints are initially unchanged; if manual responses are poor, attachment and actuator geometry are hypotheses to investigate rather than conclusions to assume.

## Evidence and comparison

Primary evidence is the saved full-face and mouth geometry, displacement vectors, target overlay, and manual response direction. Target error, projection onto the target, actual muscle F/stretch, final det(F), control-field variation, and solver termination support that inspection. Equal iteration counts do not imply equal convergence, and smaller motion does not by itself demonstrate better artifact suppression.

Each run records its exact command/configuration, copied sources, input hashes, snapshots and solver receipts. Normal Cherries runs use the no-commit profile and disable automatic Git/environment uploads. Earlier results are preserved in their original experiment group. This document is the execution plan; final findings will be written only after inspecting the generated states.
