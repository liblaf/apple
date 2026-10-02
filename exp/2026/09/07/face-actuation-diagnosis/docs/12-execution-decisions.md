# Execution decisions

The first matched comparison uses the current face fixture, 120,020 active tetrahedra, and 720,120 independent symmetric activation coordinates. Both runs start from zero displacement and zero activation, omit skin energy, and use identical materials, observations, constraints, tolerances, optimizer and step budget. The smooth run adds only the within-muscle graph penalty, with no magnitude penalty.

The initial smoothness weight is 0.0001, with a length scale of 5 mm. This is a screening choice made after the unregularized trajectory showed a dimensionless graph energy of about 226 at step 6, against a normalized squared fitting error of 0.436. A weight of 0.01 would make that graph contribution dominate the fitting term. The selected weight is not an optimized value or a preregistered choice. A useful comparison requires inspecting both achieved expression and field variation.

The graph audit finds 210,187 shared-face edges, 132 connected components, and 67 isolated cells. Isolated cells account for 0.0142% of the muscle-fraction-weighted active volume. The penalty preserves all optimization coordinates. It encourages small differences between neighboring cell values; it does not impose exact equality or a globally constant value in a muscle. Different named muscles are not coupled by the penalty.

The saved June no-skin endpoint is retained as a historical reference, with no new equilibrium claim. Its target and tetrahedral topology match the present fixture, but it uses a larger active set, different fixation, materials, loss weights, optimizer and solver tolerances. It therefore establishes a prior achievable fit with that configuration; it is not the matched control for the current regularization comparison.

Manual cases follow an identical numerical continuation path from rest through 10%, 30%, and 50% prescribed natural contraction for each independent muscle pattern and skin setting. The path is quasistatic numerical continuation, not physical time. The physical fiber stretch is computed from F and reported separately from the active map. The 50% setting means a prescribed stress-free axial stretch of 0.5; it does not assert that a loaded muscle shortens by 50%.

The manual sensitivity checks preserve the selected cells, directions and active map. They vary selected-muscle stiffness, fat stiffness, or aponeurosis stiffness to diagnose load resistance. These are material sensitivities, not calibrated tissue parameters. Increasing the selected-muscle modulus also changes its passive stiffness; it is not a pure active-force multiplier.

Tetrahedron inversion, large strain and raw active-map indefiniteness are recorded rather than used as geometry-based rejection rules. Actual failed equilibrium or derivative solves remain numerical failures. Simultaneous GPU processes share one device, so recorded wall times are not comparative performance benchmarks.
