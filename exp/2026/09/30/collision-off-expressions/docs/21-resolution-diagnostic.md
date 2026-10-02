# Second strict MouthOpen diagnostic

The second diagnostic completed with `no_witness_unresolved`. It improved the resolution of the equilibrium substantially, but it did not establish inverse convergence or a resolved feasible descent. The original queue resumed automatically, independently audited Smile-002, and started MouthOpen-003.

All baseline states use the exact activation and normalized jaw pose from audited MouthOpen-001. The warm start is the first diagnostic's hash-bound refined displacement. Collision and CCD remain absent; the original IsFixed, free-lip, skin-field and retained-cell policies are unchanged.

| Baseline force tolerance | Measured free force (solver units) | Objective | Inverted retained cells |
| --- | ---: | ---: | ---: |
| 1e-11 | 9.94090651e-12 | 0.289179683247 | 88 |
| 1e-12 | 9.91514860e-13 | 0.289187969323 | 88 |

Both satisfy the existing physical gates. Refining between these levels still changes the objective by 8.28607612e-6 and a coordinate by up to 7.70814 micrometres. Both inverted rest-volume fractions are 3.00379560e-5.

The actual unshifted adjoint succeeds with physical relative residual 9.53173985e-8. The relative 0.001 and 0.0001 shifted solves have physical residuals 0.35484 and 0.12159 respectively, despite passing their shifted equations.

| Direction and alpha | Objective change at 1e-11 | Objective change at 1e-12 | Outcome |
| --- | ---: | ---: | --- |
| Joint, 0.001 | -7.86155e-5 | -8.66729e-5 | Unresolved |
| Joint, 0.00025 | -1.30305e-5 | -2.13816e-5 | Unresolved |
| Pose only, 0.001 | -7.83878e-5 | -8.64052e-5 | Unresolved |
| Pose only, 0.00025 | -1.29847e-5 | -2.13179e-5 | Unresolved |

The larger joint and pose probes nearly pass the empirical objective screen, but their strict-level decreases remain smaller than that screen. At the refined level their slopes agree across the two scales within about 1.3%, and with the unshifted prediction within 0.8–2.1%. These are promising directional checks, but their roughly 2.1-micrometre responses remain below the baseline refinement displacement. They fail the existing response-resolution guard. Activation-only changes are smaller and unresolved as well. No resolution threshold was relaxed.

Independent review recommends another additive diagnostic containing only unchanged-parameter re-equilibration at 1e-13 and 1e-14. It will bind the second diagnostic's saved state and source hashes, retain all force and geometry gates, and record loss and displacement drift. Subsequent parameter-probe scales will be chosen from that measured drift. Neither this result nor failure of a later numerical solve would alone establish model infeasibility.

Execution used `src/71-probe-resolution.py` under the normal Cherries profile, through the one-shot serial reservation controller. [Comet run](https://www.comet.com/liblaf/collision-off-expressions/f776275b678e47949e26ca074b3a7d1a). Numerical process exit code was zero; the controller recovered the original queue and verified Smile-002's endpoint audit.

Evidence: `data/resolution-001-analysis.json`, the verified `data/mouthopen-resolution-001/` bundle and `data/mouthopen-resolution-reservation-001/` receipts. Final summary SHA-256: `47a2c465249b2c0b3fe6ce52f9d85a9b9293b640fd36f8dba8478a0857239c19`. Source SHA-256: `b18d47dc16abda85731353b0808bf0783908562f7a5c14a5299e9aa33b534953`.
