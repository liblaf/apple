# Eye-neutral expression targets

The expression input bundle anchors all 36 transferred source blendshape
displacements at the converged eye-inclusive neutral. For every expression,
the target is `x = X + u_eye_neutral + d_original`; the FEM reference,
prescribed material fields, and source eye collider remain unchanged.

The objective uses `d_original` as its displacement scale, never the loaded
neutral displacement. The bundle retains 288,235 active muscle tetrahedra for
six-component active stress per tetrahedron and expression. Its graph metric,
effective active volumes, and observation areas are recomputed at the loaded
neutral.

`data/expression-inputs-002` validates all 36 fields and a maximum
target-coordinate reconstruction error of `4.44e-16 m`. It exposes
`expression_displacement_m` for the fitting objective and total targets for
runtime metrics.
