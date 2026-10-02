# Constant neutral-basis force audit

## Question and scope

The approved shared material field has 20 coefficients: 18 constant symmetric
bulk-stress coordinates, one uniform isotropic skin resultant, and one global
skin-stiffness multiplier. This audit fixes the skin resultant at the complete
continuation target, 80.6 N/m, fixes the skin-stiffness multiplier at one, sets
activation and jaw pose to zero, and asks whether the 18 bulk coordinates can
cancel the resulting free-node force at the exact reference geometry,
\(F=I\).

The 80.6 N/m value is the frozen, Flynn-derived continuation proxy in
`joint_fields.py`. It is not a measurement on this subject and it is not a
registered full-face stress map.

This is a force-space audit, not an equilibrium solve. It does not decide
whether a deformed neutral equilibrium can satisfy the 0.25 mm motion budget.

## Method

For each normalized bulk coordinate \(x_j\), the script assembled the exact
free-force column at \(F=I\), including the current fixed supports and current
bone-contact model:

$$
  A_j = g(x_j=1,N=0)-g(x=0,N=0).
$$

It separately assembled the prescribed skin load

$$
  f_{\mathrm{skin}}=g(x=0,N=80.6I)-g(x=0,N=0).
$$

The two targets were \(-f_{\mathrm{skin}}\) and
\(-(g_0+f_{\mathrm{skin}})\), where \(g_0\) retains reference contact. The
primary raw least-squares norm is the Euclidean norm of all free nodal forces.
The second norm weights each coordinate by \(1/\sqrt{V_i}\), where \(V_i\) is
the nodal tetrahedral dual volume, and normalizes the median weight to one. It
therefore diagnoses force-density balance on this nonuniform mesh. The
weighting is intentionally unclipped; the dual-volume range is
\(3.17\times10^{-15}\) to \(1.23\times10^{-8}\,\mathrm{m^3}\), so its result
must be read alongside the raw norm.

The bulk coordinates already include the tissue-specific stress scales
\(\mu_t\). The fit also enforces the approved signed spectral interval
\([-0.9,10]\) for every \(S_{0,t}/\mu_t\). Zero-centered ridge weights
\(0,10^{-4},10^{-2},1\) are reported only as sensitivity after normalizing the
weighted force objective; they are not a calibration of the final inverse
prior.

## Result

The 18 columns are full rank and numerically well conditioned after column
scaling (condition number 2.29 in the raw norm and 1.75 in the weighted norm).
An independent mixed skin/bulk assembly reproduced linear superposition to
relative error \(3.46\times10^{-16}\).

| Target and fit | Raw residual | Weighted residual | Raw squared norm represented | Weighted squared norm represented |
| --- | ---: | ---: | ---: | ---: |
| Skin only, raw optimum | 98.882% | 99.493% | 2.224% | 1.011% |
| Skin only, weighted optimum | 98.933% | 99.478% | 2.123% | 1.041% |
| Skin + reference contact, raw optimum | 98.882% | 99.494% | 2.224% | 1.010% |
| Skin + reference contact, weighted bounded optimum | 98.933% | 99.478% | 2.123% | 1.041% |

The skin load has raw force norm 3.79774 N. The reference passive/contact
force norm is 0.00137059 N, so including contact changes the result only in the
reported last digits. The weighted bounded solution leaves 3.75721 N raw
residual force, 0.008421 N RMS per free node, and 0.214316 N maximum at one free
node.

The material bounds do not cause this residual. The unconstrained weighted
solution is already strictly feasible and the constrained optimizer returns it
in one iteration. Its tensor eigenvalues, in units of each tissue's \(\mu_t\),
are:

| Tissue | Eigenvalues of \(S_0/\mu_t\) | Coordinate L2 departure from zero |
| --- | --- | ---: |
| Fat | -0.498, -0.281, -0.259 | 0.627 |
| Aponeurosis | -0.00203, -0.00141, -0.000829 | 0.00261 |
| Muscle | -0.310, -0.233, -0.171 | 0.424 |

The closest lower-bound slack is 0.402, and no eigenvalue occupies either
bound. A ridge weight of 0.01 lowers the total coordinate norm from 0.757 to
0.721 while changing the weighted residual from 99.4782% to 99.4789%; a ridge
weight of one lowers the norm to 0.194 but raises the residual to 99.7315%.

At exact rest, the spatially varying skin-curvature load is therefore almost
entirely outside the span of the three tissuewise constant stress tensors. The
result identifies a representational limitation of the constant basis at
\(F=I\); it does not establish nonlinear infeasibility. Material stiffness,
contact, geometry, and the finite-deformation response may still produce a
valid equilibrium inside the 0.25 mm budget. If the full-target continuation
cannot meet that budget, a spatial bulk-stress basis is a specific hypothesis
to test rather than a conclusion of this audit.

## Contact and execution evidence

The current reference contact state is numerically valid, with barrier energy
\(3.8158252\times10^{-14}\), minimum active distance 22.5836 µm, and CCD
fractions equal to one. Its candidate enumeration contained 171 active pairs,
two more than the older contact receipt's 169; barrier energy and minimum
distance reproduce within the declared tolerances. The first attempted audit
failed on an overly strict pair-count equality and is retained under
`tmp/neutral-basis-audit-001`; the successful source-snapshotted run is
`data/neutral-basis-audit`.

The successful GPU assembly took 8.34 seconds while neutral segment 003 and an
unrelated process were active. It did not invoke the equilibrium optimizer.

Run command:

```bash
DEBUG=1 CHERRIES_NAME=neutral-basis-audit \
  CHERRIES_TAGS=readiness,force-space,contact \
  uv run python \
  exp/2026/09/21/joint-activation-material-mandible/src/16-audit-neutral-basis.py \
  --output-dir \
  exp/2026/09/21/joint-activation-material-mandible/tmp/neutral-basis-audit-003
```

The machine-readable receipt is
`data/neutral-basis-audit/summary.json`; the small NPZ contains only the 18 by
18 normal equations, right-hand sides, and column norms, not the large force
matrix.
