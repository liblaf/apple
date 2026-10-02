# Stage 3 Smile forward setup comparison

Reuse the exact saved `S` tensor from the `l2-normal-rankone_fixed` checkpoint at inverse update 200. It is native active strain, `B=I+S`, with fixed muscle directions. Do not refit, rescale, rotate, or smooth the activation. The source checkpoint records `solver_valid=false`; this comparison solves displacement afresh.

| Condition | Reference and starting state | Mechanics |
| --- | --- | --- |
| Original inverse setup | Exact full historical inverse mesh; zero displacement | Fat, aponeurosis, muscle; no skin or collision |
| New setup | Full repaired reference from `reference-clearance-002`; saved loaded neutral from `forward-isfixed-001` | Same bulk tissues plus the saved heterogeneous active-strain skin membrane and complete source bone/eye IPC |

Both models retain all 1,146,517 original tetrahedra, 228,660 original points, 288,235 active-cell rows, and the authoritative `IsFixed` boundary. The new condition appends prescribed obstacle coordinates. The jaw stays at zero rigid pose. Tetrahedron connectivity and activation row identities are asserted exactly. The corrected reference changes free-node coordinates; it is part of the new setup, so this is a comparison of two complete setups rather than an isolated skin ablation.

Skin pre-strain is the saved 2×2 tangential inverse activation `B=sqrt(I+T/(h*mu))` for `StableNeoHookeanActiveMembrane`, with physical-J plane-stress regularization. It is the prescribed-tension equivalent used by the corrected neutral, rather than a measured subject-specific strain. Thickness is 1 mm. IPC is frictionless, soft tissue versus the complete cranium, mandible, and eyes; soft-soft and rigid-rigid pairs are disabled. Fix barrier stiffness at the neutral endpoint's recorded 1.3544 MPa.

The comparison uses safeguarded Newton from each condition's neutral, exact bulk curvature, PSD-clamped IPC curvature for the search, Armijo backtracking, and CCD. The physical energy and force remain unmodified. Both branches use an absolute free-force norm of `1e-10 MPa m² = 0.0001 N`, matching the inverse protocol. An initial unshifted search is regularized only when needed. Numerical stopping limits are 5,000 Newton updates, 10,000 CG iterations per search, and 1,200 seconds per branch. Exhausting a limit is a failed convergence test.

Record every solver update:

- Total potential energy in joules (`MPa m³ × 1e6`). Each setup has its own constitutive energy offset. Cross-setup vertical energy differences are not energies evaluated under one common model.
- Norm of the force residual on free coordinates in newtons (`MPa m² × 1e6`). Fixed-node reaction forces are excluded.
- Inverted cells: `det(F) <= 0`, divided by all 1,146,517 original tetrahedra, including fully prescribed cells. No inversion rejection is applied during this diagnostic; a converged residual with inverted cells is reported as physically invalid.

Save curves, source/input SHA-256 receipts, numerical endpoint arrays, contact geometry checks, and an independent CPU inversion/boundary audit. Curves show solver updates at full activation, not an activation ramp or inverse-optimization history.

Initial PNCG attempts are retained as diagnostics. The contact attempt violated the configured minimum gap after a CCD-limited update because that PNCG phase has no energy line search. The matched Newton comparison is the reported experiment.
