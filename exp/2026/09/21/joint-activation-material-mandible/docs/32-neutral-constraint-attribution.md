# Neutral constraint attribution at the frozen Spatial25 endpoint

This is a read-only attribution audit of update 200 from
`neutral-convergence-025-contact-spatial80-metric-bfgs-001`. It does not run a
forward solve, change a parameter, or establish feasibility at the 50% or 100%
skin-prestress targets. The later `spatial80-metric-bfgs-002` segment is still
in progress and is deliberately excluded.

## Frozen evidence

The audited run ended at its declared 200-accepted-step budget. It is a valid
contact-enabled 25% state, but it is not an optimizer-converged neutral
preparation:

| Quantity | Frozen update-200 value | Declared gate |
| --- | ---: | ---: |
| Objective | `0.6532544927` | diagnostic |
| Projected-gradient infinity norm | `0.02429376072` | `<= 0.001` |
| Five-value objective relative range | `0.0007181659532` | `<= 0.00001` |
| Surface-motion RMS | `0.1830178964 mm` | `<= 0.25 mm` |
| Minimum `det(F)` | `0.5738316122` | `>= 0.25` |
| Inverted tetrahedra | `0` | `0` |
| Active contact pairs | `141` | diagnostic |
| Minimum active-pair distance | `50.2917 micrometres` | contact receipt valid |

All 200 accepted updates used one full-fraction Armijo trial. The trace minimum
surface RMS was `0.1829585287 mm`. The objective fell from `1.3748983779` to
`0.6532544927`, while the projected-gradient infinity norm fell from
`1.2074971294` to `0.02429376072`. This is evidence that the endpoint remains
limited by unfinished outer optimization; it is not evidence of a failed
material basis, contact solve, or deformation gate.

The terminal objective terms were:

| Term | Value |
| --- | ---: |
| Surface loss | `0.5359288064` |
| Muscle loss | `0.0925833050` |
| Weighted spatial roughness | `0.02323087149` |
| Weighted magnitude/skin prior | `0.001511509892` |

The skin resultant was fixed at `20.15 N/m`, the declared 25% fraction of the
`80.6 N/m` modeling proxy. The skin stiffness multiplier was exactly its upper
bound, `3.0`. Its raw log-multiplier gradient was `-0.5487523965`; subtracting
the `0.001 * log(multiplier)^2` prior gradient gives an inferred mechanical
component of `-0.5509496210`. The sign favors a further stiffness increase,
which projection prevents. This is direct evidence that the stiffness cap is
active at this endpoint. It is not a biological upper bound and does not prove
what a relaxed cap would do to the neutral displacement.

The bulk anchor eigenvalues remained inside the dimensionless signed-stress
bounds `[-0.9, 10]`: fat `[-0.28965, 0.13921]`, aponeurosis
`[-0.27814, 0.11240]`, and muscle `[-0.57190, 0.51813]`. The terminal contact
receipt was numerically valid, and no tetrahedron was inverted. Neither the
bulk spectral bounds nor contact validity was active as a recorded terminal
failure.

## CPU attribution method

The calculation used only the frozen trace/checkpoint and the audited basis
arrays on CPU with NumPy `float64`. Let `c` be the 78 bulk anchor coordinates,
and let the stored global matrices `G` and `M` be the three-tissue mean block
forms. The frozen objective uses

```text
0.5 * beta * c^T G c,  beta = 100
```

so its roughness gradient is `beta * G c`. The magnitude prior is the sum of
the three tissue forms rather than their mean. With prior weight `0.001`, its
bulk gradient is therefore `2 * 0.001 * 3 * M c = 0.006 M c`. The mechanical
bulk gradient reported below is the saved raw objective gradient minus those
two exact analytic gradients. No finite differences or FEM evaluation were
performed.

| Tissue | Mechanical gradient L2 | Roughness gradient L2 | Prior gradient L2 | cosine(mechanical, roughness) |
| --- | ---: | ---: | ---: | ---: |
| Fat | `0.05191448` | `0.03434086` | `0.00015837` | `-0.79489` |
| Aponeurosis | `0.03379075` | `0.01020519` | `0.00009228` | `-0.08762` |
| Muscle | `0.04097388` | `0.03514748` | `0.00034698` | `-0.79302` |

The strong H1 term materially opposes the equilibrium-fit gradient in the fat
and muscle blocks. This identifies a regularization tradeoff; it does not show
that `beta=100` is wrong or that reducing it will satisfy the neutral-motion
gate.

The 78-coordinate `M` matrix is full rank with condition number `22.43884`.
The `G` matrix has rank 54 and nullity 24. That nullity is structural:

- fat `G` has rank 3 for four anchors, giving one constant null direction;
- aponeurosis `G` has rank 3 for four anchors, giving one constant null
  direction;
- muscle `G` has rank 3 for five anchors, giving two constant null directions.

The muscle anchors have component labels `[0, 0, 0, 0, 1]`; the two anchored
muscle components contain about 90.32% and 8.33% of muscle support volume.
They can each carry an independent constant tensor. The extra muscle constant
gives six additional tensor-coordinate null modes, so the total is
`(1 + 1 + 2) * 6 = 24`. Fat and aponeurosis have many tiny disconnected support
components, but their unanchored-component policy assigns each entire component
to a nearest anchor; those pieces do not add independent anchor coordinates.
The two muscle scalar null eigenvalues are exactly zero and approximately
`3.25e-19`, so this count is not an arbitrary rank-threshold interpretation.

The earlier force-space audit is consistent with a coarse load span, not a
numerically singular basis. Its dual-volume-weighted residuals were 99.478%
for constant18, 99.076% for unconstrained spatial78, 99.127% for bounded
spatial78 without smoothness, and 99.443% for bounded spatial78 with
`beta=100`. The corresponding spatial78 force matrix had scaled condition
number `4.5403`. These are exact-rest, reference-configuration force residuals;
they do not decide nonlinear equilibrium or the `0.25 mm` displacement gate.

## Interpretation and conditional diagnostic

At the frozen 25% endpoint, the evidence supports three possible limitations:

1. the skin stiffness cap is active and the raw gradient favors increasing
   stiffness;
2. the strong H1 term opposes the mechanical gradient in fat and muscle;
3. the 13-anchor field offers only a modest improvement in the reference-state
   skin-load force span, despite acceptable numerical conditioning.

The first two are active-gradient observations. The third is a lower-cost
linear screen. None justifies extrapolating the 25% deformation to the 50% or
100% targets, declaring the full target infeasible, relaxing a gate, or adding
basis functions now.

If a future 50% or 100% run first satisfies the declared projected-gradient and
objective-plateau criteria but still exceeds `0.25 mm`, the smallest diagnostic
is a frozen-terminal constraint-attribution receipt:

1. On CPU, report all bound slacks, raw and projected gradients by parameter
   family, the exact `G`/`M` gradient subtraction above, and the mechanical
   gradient resolved in generalized `G v = lambda M v` modes.
2. From the same frozen primal state, evaluate one legal inward stiffness probe
   from `3.0` to `2.9`, with all other coordinates fixed.
3. Separately evaluate one small projected Spatial80 step along the negative
   regularizer-subtracted mechanical gradient, with stiffness fixed at `3.0`.
4. Apply the unchanged forward, contact, deformation, and receipt gates to both
   probes. These are single perturbation evaluations, not optimization or
   automatic fallback.

If the inward stiffness probe worsens the displacement objective, that would
implicate the cap. If the spatial mechanical step improves displacement but is
rejected only by the regularized total objective, that would implicate the
frozen H1 tradeoff. Only if neither perturbation has meaningful leverage would
one test a single `M`-orthogonal residual-driven enrichment mode. Any such
enrichment would be a new versioned sensitivity model, not evidence that the
current basis is infeasible.

## Immutable inputs and source hashes

The audited artifacts are:

- [summary](../data/neutral-convergence-025-contact-spatial80-metric-bfgs-001/summary.json),
  SHA-256 `af06dcde90ea0982314f92bb7905a0bc08989ef710f627bf1b8ccd2d0401a6b7`;
- [trace](../data/neutral-convergence-025-contact-spatial80-metric-bfgs-001/trace.json),
  SHA-256 `8695ef4f68895ca4e9e6ca9b1a3260569f82ef1894ba4d519157a7dbc1e5d41d`;
- [terminal checkpoint](../data/neutral-convergence-025-contact-spatial80-metric-bfgs-001/terminal.pt),
  SHA-256 `fb98ed5d398625c2900adf3241ab3a01d74633730f3c05a06170bdddb16f8c65`;
- [basis arrays](../data/spatial-baseline-audit-005/basis-and-normal-equations.npz),
  SHA-256 `693e07f7ba4b71415ec41e5e0791edf4a6fb65d0e4452f2fbbfcd7f5c96f095d`;
- [force audit receipt](../data/spatial-baseline-audit-005/summary.json),
  SHA-256 `f419c658dedb8eedc2e16a2278690708d631bf75e8da693d550c3034b88eb1fb`.

The frozen run archived its actual sources. The files directly governing this
audit were `21-neutral-converge.py`
`3dbc230004c2bd5693bf890c7e588d0661d2dfdce1418181df8e9b9af89c12a7`,
`23-audit-spatial-baseline.py`
`1ad10065b65bae778ea78e36059350896faced6fb5087eeaa456d0a7c17c8ba9`,
`joint_spatial_fields.py`
`fcb62322f0ca29d3d945badef3582e70ce66f46d1e1b3087396986e7f9711543`,
`joint_physics.py`
`4624d912395582e733da920e09d6e9c70d77f8fd1dd4b0d8466b2b4833609cb0`,
`joint_equilibrium.py`
`f73c66a8d99e7ef0f902d3d4d0884c2e30a613cf135ff85af7d19c6cea9d4a6c`,
`joint_contact.py`
`1f0b243f393bb68027c5696961878e5db6a61693d9997d66b331ddf397347644`,
and `joint_newton.py`
`7b7e8c4fe14cf5dd3e2ca5d95485491c72f9e0be4dded61fcd4f868c18762664`.
The frozen objective fingerprint is
`55aa0d64be811b42247298f956ca79af73a3b1e453c050ec85b957fb8e62473e`.
