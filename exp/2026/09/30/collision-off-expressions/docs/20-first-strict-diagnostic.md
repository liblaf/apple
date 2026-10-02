# First strict MouthOpen diagnostic

The stricter equilibria remain within the original physical policy, but the objective differences from the six tiny parameter probes are unresolved. This result does not establish inverse convergence or model infeasibility. The original serial queue recovered automatically and independently audited MouthOpen-002 before starting Smile-002.

All rows below use identical activation parameters and exact normalized jaw pose from audited MouthOpen-001. Collision and CCD remain absent; original IsFixed constraints, all 3,408 free lip vertices, exact skin fields, and retained-cell policy were checked again.

| State | Free force (solver units) | Objective | Inverted retained cells | Inverted rest volume fraction |
| --- | ---: | ---: | ---: | ---: |
| Saved endpoint | 9.9999999990e-9 | 0.287704274315 | 99 | 3.00383855e-5 |
| Equilibrated at 1e-9 | 9.9989001835e-10 | 0.288555337232 | 89 | 3.00379713e-5 |
| Refined at 1e-10 | 9.8600816146e-11 | 0.289101708729 | 88 | 3.00379560e-5 |

The maximum coordinate change from the saved endpoint to the refined state is 2.56876 mm. Refining 1e-9 to 1e-10 alone changes the objective by 0.000546371498 and coordinates by up to 0.841211 mm. Thus the original 0.01 N force contract is insufficient to resolve the small objective differences used by these probes.

At the same refined state, the shifted adjoint residuals pass their respective shifted systems. Their residuals against the unshifted physical Hessian are 0.35477 for relative shift 0.001 and 0.12156 for shift 0.0001. The actual unshifted solve succeeds with relative residual 9.19278e-8, below its 1e-7 tolerance. The normalized-pose gradient L2 norms are respectively 0.34122, 0.61126 and 0.72765. A small-step stall from the original damped method cannot be interpreted as stationarity.

Joint, activation-only and pose-only probes at alpha 1e-6 and 2.5e-7 satisfy the force and geometry gates, but their seeds already meet both prior force tolerances. Their apparent refined-level loss decreases are at most 7.28e-8, below the empirical 0.0054637 resolution screen. Neither a descent witness nor convergence follows from those differences.

The next diagnostic reuses only the hash-bound refined displacement as a warm start, after checking exact activation, normalized and physical pose, and active IDs against the original checkpoint. It tightens force tolerances to 1e-11 and 1e-12 and enlarges the two probe scales to 1e-3 and 2.5e-4. It requires a successful unshifted adjoint, compares force-level and scale sensitivity, and distinguishes a feasible lower-loss endpoint from evidence supporting a local gradient. Every physical acceptance gate remains unchanged. It never edits an optimizer checkpoint.

Evidence: `data/probe-001-analysis.json`, `data/mouthopen-probe-001/summary.json`, the 394-file verified diagnostic bundle, and the 7-file verified controller bundle. Refined warm-start SHA-256: `c5203464020d35d4bf222589bd3bdb6629319e99e6e40ef9d1f0cfcf2734318e`. The first diagnostic summary SHA-256 is `ff2cf2a60ea68a48aa27642e6fc785c71c8f3f255eefc4f9c29483a38a79f390`.
