# Unchanged-parameter force refinement

The 1e-13 baseline passed the original physical gates. The subsequent requested 1e-14 solve failed with `Newton regularization exhausted`; the diagnostic correctly ended `unresolved_baseline`. Neither inverse convergence nor model infeasibility follows from this numerical failure.

The run held activation, normalized and physical jaw pose, and active cell IDs exactly at audited MouthOpen-001 values. Its initial displacement was the hash-bound successful 1e-12 baseline from the second diagnostic. Collision and CCD remained absent, and all original fixed-node, lip, material and retained-cell constraints were checked.

| State | Direct free force (solver units) | Objective | Inverted retained cells |
| --- | ---: | ---: | ---: |
| Prior successful 1e-12 baseline | 9.91514860e-13 | 0.289187969323 | 88 |
| Successful 1e-13 baseline | 9.99363506e-14 | 0.289188853284 | 88 |
| Failed attempt at 1e-14 | 1.10582681e-13 | 0.289188879283 | 88 |

The two successful baselines differ by 8.83960988e-7 in objective and at most 1.14620 micrometres in a coordinate. Their inverted rest-volume fraction remains 3.00379560e-5. The failed attempt remained within the geometry allowance but did not meet its requested force tolerance, so it is excluded as a refined baseline.

The failure receipt contains seven accepted Newton steps, earlier CG-budget and Armijo rejections, and eight final regularization attempts rejected by Armijo. The final shifts ranged up to about 27.85. Last recorded energies were around 2.77451642e-6. This diagnoses the observed line-search failure; it does not prove a universal floating-point limit. The saved failure archive retains displacement and exact parameters, and its force was evaluated directly from the physical model.

The next numerical test can use the two successful 1e-12 and 1e-13 baselines and parameter steps large enough to exceed their measured objective and coordinate drift. It must label any reuse of these states as replay, re-evaluate their forces and geometry, and genuinely re-equilibrate each new parameter trial at both levels. The existing empirical resolution, response and two-scale gradient guards remain required. The failed 1e-14 state must not enter that comparison as an accepted endpoint.

The diagnostic ran as `src/72-probe-force-resolution.py` under the normal Cherries profile with commits disabled. [Comet record](https://www.comet.com/liblaf/collision-off-expressions/c27ae9928c5349f3a05f73e757d3a145). The process exited zero after saving the unresolved result; its controller restored the original queue, verified MouthOpen-003's endpoint audit, and allowed Smile-003 to start.

Evidence: `data/force-resolution-001-analysis.json`, `data/mouthopen-force-resolution-001/summary.json`, and the saved baseline/failure archives and source snapshots. Final summary SHA-256: `9ef781ff0d0782ef3cde0ed14c1d4c7816dec855370e63da3102600a7262fa05`. Executed source SHA-256: `1879ec3d68c85845093d63cc3f3dde5e4db8c9fb10cf57f7d0509fac1a4c693f`.
