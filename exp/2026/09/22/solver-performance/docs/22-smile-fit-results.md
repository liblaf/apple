# Smile inverse fit: matched original PNCG versus hybrid PNCG→Newton-CG

The completed matched comparison reduced measured inverse-update time from **1455.17 s** with original PNCG to **781.43 s** with hybrid PNCG→Newton-CG: **1.86× faster**. Both arms used the same frozen neutral, inputs, objective, hardware, and 20 full projected-Adam updates. Their final observed-skin RMS errors are effectively the same: **4.92078 mm** (original) and **4.92054 mm** (hybrid).

These are inverse-update timings (`fit_step_seconds`), including each arm's forward and adjoint work. They exclude the one shared neutral preparation, and this report does not make an all-run wall-time claim.

## Completed matched experiment

The run is [smile-fit-adam03-unconditional-004](../data/smile-fit-adam03-unconditional-004/summary.json), rendered in [smile-fit-adam03-visuals-001](../data/smile-fit-adam03-visuals-001/summary.json). It fits `Smile` (expression index 12) in the fixed-material facial model. The optimization variables are six symmetric material active-stress coordinates on each active muscle tetrahedron and one bounded mandible hinge coordinate. Active stress remains the material tensor (Q), with

$$
W_\mathrm{active}=\tfrac12 Q:(F^\mathsf{T}F-I), \qquad P_\mathrm{active}=FQ.
$$

The arms ran sequentially on the same Paratera RTX 4090 with eight IPC/OMP threads. The only changed primal runtime is the solver: production accepted-force PNCG versus `hybrid_diag` (PNCG warm-up followed by safeguarded Newton-CG). The common tight neutral seed is [shared-neutral-init.pt](../data/smile-shared-003/shared-neutral-init.pt) and is excluded from paired timing.

| Shared setting | Value |
| --- | --- |
| Outer optimizer | Full projected Adam, learning rate 0.3 |
| Outer policy | One full step per update; no outer rejection or backtracking |
| Magnitude / jaw penalty | 0 / 0 |
| Smoothness weight | 5.066584049455902 |
| Forward force tolerance | `1e-8` |
| Adjoint relative tolerance | `1e-7` |
| Hybrid Newton admission floor | `1e-7`; final physical tolerance remains `1e-8` |
| Maximum updates | 20 |
| Collision | Soft tissue versus complete cranium, mandible, and fixed registered eyeballs |

The [local copy verification receipt](../data/smile-fit-adam03-unconditional-004/local-copy-verification.json) confirms identical frozen input/calibration provenance and shared neutral hash, saved Adam LR 0.3, exactly 20 accepted full updates per arm, and zero rejected outer trials.

## Measured outcome

| Measure | Original PNCG | Hybrid PNCG→Newton-CG |
| --- | ---: | ---: |
| Inverse-update time | 1455.17 s | 781.43 s |
| Forward time | 1249.13 s | 557.58 s |
| Adjoint time | 178.98 s | 200.36 s |
| Final observed-skin RMS | 4.9207805 mm | 4.9205375 mm |
| Final data term | 0.91747495 | 0.91738434 |
| Final objective | 13.0043536 | 13.0052120 |
| Final weighted smoothness | 12.0868787 | 12.0878277 |

The initial objective was 1.00000299 and initial RMS was 5.1373309 mm. At 20 full updates, the objective is about 13 because the smoothness contribution is about 12; this is not a monotonic loss-minimization result because the requested outer policy accepts every full Adam step. The terminal data terms and RMS values are close between solvers, while the objective differs slightly through the smoothness state.

The direct terminal comparison at observed skin nodes gives **0.0074337 mm weighted RMS** displacement difference and **0.037728 mm maximum** node difference. This measures the two saved physical endpoints directly; it is not a fit-to-target error.

Neither arm is inverse-converged: both ended at the fixed 20-update budget and their stationarity checks remain false. The run therefore demonstrates a matched bounded trajectory and a solver-time reduction, not recovery of a converged Smile inverse solution.

## Physical validity

Both final states passed the required collision and geometry gates.

| Gate | Original PNCG | Hybrid PNCG→Newton-CG |
| --- | ---: | ---: |
| Final force / threshold | `9.24e-9` / `1e-8` | `7.25e-9` / `1e-8` |
| Inverted tetrahedra | 0 | 0 |
| Minimum det(F) | 0.34571 | 0.34696 |
| Active contact pairs | 5,243 | 5,232 |
| Minimum active contact gap | 16.607 µm | 16.651 µm |

The [matched summary](../data/smile-fit-adam03-unconditional-004/summary.json), per-arm collision receipts, and checkpoint hashes are retained with the output. The visualization is post-processing only and performed no physics solve.

![Convergence and timing](../data/smile-fit-adam03-visuals-001/smile-convergence.png)

![Common-view fit error](../data/smile-fit-adam03-visuals-001/smile-fit-error-front.png)

![Active-stress magnitude](../data/smile-fit-adam03-visuals-001/smile-active-stress.png)

Additional common-context and geometry views are available as [front motion](../data/smile-fit-adam03-visuals-001/smile-context-motion-front.png), [side motion](../data/smile-fit-adam03-visuals-001/smile-context-motion-side.png), [front geometry](../data/smile-fit-adam03-visuals-001/smile-clay-geometry-front.png), [side geometry](../data/smile-fit-adam03-visuals-001/smile-clay-geometry-side.png), and [side error](../data/smile-fit-adam03-visuals-001/smile-fit-error-side.png).

## Earlier pilot context

Earlier V100 host and strict-tolerance runs are historical diagnostics, not inputs to the matched speedup. The completed older 0.003 hybrid pilot is retained in its [receipt](../data/smile-hybrid-comp07-production-003/comp07-copy-receipt.json); its hardware, initialization, optimizer policy, and trajectory differ. The [forward-tolerance study](23-forward-tolerance.md) and [protocol](20-smile-fit-protocol.md) describe the prior accuracy and switching investigations.
