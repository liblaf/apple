# Facial inverse physics: current model and algorithm

Snapshot: 22 September 2026, 05:17 China Standard Time. Matched comparison complete.

**The hybrid solver took 13.0 minutes versus 24.3 minutes for original PNCG: 1.86× faster over 20 inverse updates.** Both runs used the same RTX 4090 sequentially, active stress, full bone and eyeball collision, projected Adam with learning rate 0.3, no outer rejection, and a common forward force tolerance of `1e-8`.

| Measurement | Original PNCG | Hybrid PNCG → Newton-CG |
| --- | ---: | ---: |
| Full updates / rejected outer trials | 20 / 0 | 20 / 0 |
| Measured inverse-update wall time | 1,455.17 s | 781.43 s |
| Forward time | 1,249.13 s | 557.58 s |
| Adjoint time | 178.98 s | 200.36 s |
| Final skin-position RMS | 4.920781 mm | 4.920538 mm |
| Final objective | 13.00435 | 13.00521 |
| Weighted stress smoothness | 12.08688 | 12.08783 |
| Inverted tetrahedra | 0 | 0 |
| Inverse converged | No | No |

The initial skin-position RMS was 5.137331 mm; both reduced it by about **4.2%**. Their final observed-skin surfaces differ by **0.007434 mm weighted RMS**, with a **0.037728 mm maximum node difference**. Neither recovered the target smile. The objective rose from about 1.0 to 13.0 because the retained smoothness contribution dominates; the full Adam trajectory is nonmonotone and the 20-update endpoint is not the best-RMS iterate. Both ended at the requested iteration budget.

Timings are sums of measured inverse-update wall time, including forward and adjoint work; common neutral preparation, some setup/checkpoint overhead, visualization and transfer are excluded. Forward work improved 2.24× and still accounts for 71% of hybrid update time; adjoint work accounts for 26%. This is one matched comparison, with no repeat-run uncertainty estimate. Earlier RTX 3090 pilots are excluded.

## Anatomical model

- **Volume:** 228,660 nodes and 1,146,517 tetrahedra. Stable Neo-Hookean fat, aponeurosis and muscle fields, plus an exact plane-stress Stable Neo-Hookean skin membrane. Passive fields and neutral prestress stay fixed during fitting.
- **Controls:** six symmetric active-stress coordinates on 288,235 muscle tetrahedra: 1,729,410 stress coordinates, plus one jaw hinge angle. The target observes 15,299 skin nodes.
- **Active stress:** `W_active = ½ Q:(FᵀF − I)`, giving `P_active = FQ`. Q is a positive-semidefinite material stress; added-stress eigenvalues are capped at ten times the reference stress of 12.3288 kPa. Zero added stress retains frozen neutral prestress.
- **Rigid anatomy:** complete cranium (35,162 triangles), mandible (18,948) and both eyes (2,560). Cranium and eyes are fixed; the jaw has one registered hinge, 0–40°, without translation. Mapped support nodes impose attachments. Both fitted jaw angles remained 0°.
- **Collision:** frictionless IPC between soft tissue and all bones and eyes throughout fitting; barrier distance 0.1 mm, CCD minimum distance 10 nm. No soft-soft or rigid-rigid contact. Both endpoints pass intersection/contact checks and have zero inverted tetrahedra.

Material fields and anatomical correspondence remain modeling assumptions, not subject-specific validation. Lip self-contact is outside the model. The target jaw pose is unavailable, so its visualization uses a neutral-reference mandible; fitted columns use saved jaw poses.

## One inverse update

1. **Full projected Adam, LR 0.3.** Minimize normalized area-weighted skin-position error plus spatial stress smoothness. Magnitude and jaw penalty weights are zero. PSD stress bounds and the bounded jaw parameterization remain.
2. **No outer rejection or backtracking.** One projected proposal is evaluated once. No neighbor-budget screen, slope fallback, Armijo loop, residual-error gate or smaller-step retry. Physical solve failures terminate visibly.
3. **Forward equilibrium.** Original: accepted-force PNCG. Hybrid: coarse PNCG followed by safeguarded matrix-free Newton-CG with diagonal preconditioning. Both stop at a free-force norm of `1e-8`; IPC/CCD and inversion checks remain active.
4. **Implicit differentiation.** Solve the physical, unshifted adjoint at relative tolerance `1e-7` for stress and jaw gradients. Newton direction shifts affect only the forward search system. Residual correction is diagnostic, not an outer-step gate.

PNCG establishes a coarse deformation; Newton uses curvature to remove the slow error modes near equilibrium. The hybrid switches at `max(final_atol, 1e-3 × initial_force, 1e-7)`; the absolute floor is an empirical warm-start heuristic. Implemented improvements also include gradient reuse, exact directional curvature and eight IPC threads. Contact uses the CPU IPCTK backend. The completed timing comparison measures their combined implementation; it does not isolate each change's contribution.

## Numerical accuracy

Final force norms were `9.236e-9` (original) and `7.253e-9` (hybrid), below the common `1e-8` criterion. Minimum active contact gaps were 16.607 and 16.651 µm; minimum det(F) values were 0.34571 and 0.34696. The loose force criterion is not a displacement- or gradient-error certificate, and agreement between these two loose-tolerance runs does not establish agreement with a tightly converged solution.

PNCG's relative rule is `max(atol, rtol * initial_force_norm)`. On the historical neutral initial-force reference `1.5192003475221145e-5`, relative `5e-4` corresponds to `7.596001737610573e-9`, or 7.60 mN. The current `1e-8` (10 mN) is 1.32× looser on that reference. The historical fixed `1.5192e-10` came from relative `1e-5`. Each warm start changes the relative-to-absolute conversion.

## Evidence and views

The run is `smile-fit-adam03-unconditional-004`; figures are `smile-fit-adam03-visuals-001`. Checkpoint hashes, full-step Adam state and common input hashes were verified after copying the completed artifacts. The auxiliary V100 host pilot was stopped before this comparison finished; its timings are excluded. Resource audits cover the allocated environment, not exclusive ownership of the provider's physical CPU.

[Front geometry](../data/smile-fit-adam03-visuals-001/smile-clay-geometry-front.png) · [Side geometry](../data/smile-fit-adam03-visuals-001/smile-clay-geometry-side.png) · [Target error](../data/smile-fit-adam03-visuals-001/smile-fit-error-front.png) · [Convergence](../data/smile-fit-adam03-visuals-001/smile-convergence.png) · [Active stress](../data/smile-fit-adam03-visuals-001/smile-active-stress.png).

[Detailed results](22-smile-fit-results.md) · [Protocol](20-smile-fit-protocol.md) · [Completed summary](../data/smile-fit-adam03-unconditional-004/summary.json) · [Verification](../data/smile-fit-adam03-unconditional-004/local-copy-verification.json) · [Visual receipt](../data/smile-fit-adam03-visuals-001/summary.json).

## Follow-up: zero stress smoothness

The later [zero-smoothness diagnostic](28-smile-zero-smoothness.md) preserved 19 full Adam updates, reaching 3.5582 mm RMS, then stopped on one inverted tetrahedron in proposal 20. It supports the strong-regularizer diagnosis while exposing a physical failure at larger stress changes. The completed old/new solver comparison above remains the historical 20-update result.
