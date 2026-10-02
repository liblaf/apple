# Working baseline after the physical-volume correction

2026-09-08. Continue from session `01a07f7c-3f97-7e82-ae4d-3b712b039bff` using the corrected active-strain baseline. The volume-dependent energy must use physical `J = det(F)`, independently of activation. Earlier attribution of the visible improvement primarily to PSD activation is superseded by the corrected baseline experiment.

The adopted energy is

$$
W(F,A_{\mathrm{inv}})=\frac{\mu}{2}(\|F A_{\mathrm{inv}}\|_F^2-3)
-\mu(J-1)+\frac{\lambda_0}{2}(J-1)^2,\qquad J=\det F.
$$

The original activation-dependent norm term remains. Both determinant terms and their derivatives use physical deformation. No PSD constraint, activation clamp, or inverse smoothness penalty is required for this corrected baseline.

The completed rerun retains the original six independent symmetric controls per active tetrahedron, zero initialization, target, material constants, fixed boundary, no-skin setting, and 200-update Adam budget. It completed all 201 state evaluations successfully. Its failure policy is stricter than the historical run, which continued after six failed forward solves; the stricter policy was never triggered in the rerun.

| Saved result | Old baseline | Corrected baseline | PSD comparison |
| --- | ---: | ---: | ---: |
| Endpoint step | 194 | 200 | 1024 |
| Area-weighted fit RMS, mm | 0.654339 | 1.835697 | 1.610246 |
| Area-weighted motion RMS, mm | 4.974140 | 4.150593 | 4.125968 |
| Active-muscle volume-weighted RMS of J minus 1 | 0.431684 | 0.015434 | 0.022861 |
| Inverted pure-muscle cells | 33 | 0 | 0 |

The corrected baseline already removes much of the historical bumpiness and has almost the same total motion as the PSD endpoint. This makes the determinant argument a demonstrated major confound in the earlier comparison. The stiffness formulas remain mathematical properties, but do not establish the main cause of the observed appearance. PSD remains a comparison model; its additional benefit has not been isolated at matched fit and optimization history.

The historical comparison is retrospective: the original run lacks an archived runtime-library hash, and the reconstructed historical helper was previously checked against only its first few steps. Together with its recorded forward failures, this limits attribution of an exact fraction of the improvement to the determinant change.

The corrected result has one inverted mixed cell containing 99.9023% fat. A finite volume penalty is not an inversion barrier. Its completed budget is not an inverse-convergence claim.

Canonical evidence remains in the originating worktree:

- Completed report and flat-shaded comparisons (historical worktree file: `exp/2026/09/08/physical-volume-baseline/docs/20-baseline-report.md`).
- Corrected material implementation (historical worktree file: `exp/2026/09/08/physical-volume-baseline/src/volume_preserving_active.py`).
- Run summary and saved endpoint (historical worktree file: `exp/2026/09/08/physical-volume-baseline/data/20-baseline/summary.json`).
- Counterexamples to the earlier stiffness-based causal explanation (historical worktree file: `exp/2026/09/08/analytical-active-mechanics/docs/30-challenging-the-explanation.md`).

On resuming here, the material and protocol were read, the flat-shaded mouth comparison was inspected, and all 23 recorded comparison input/output hashes passed verification. The original validation receipt reports six cases passing energy, stress, Hessian, activation and mixed-derivative checks against independent autodifferentiation, with maximum absolute discrepancy 5.33e-15. This continuation note changes the working interpretation; it does not modify historical sources or rerun the experiment.

Checkpoint clarification for subsequent work: the every-ten-step NPZ files contain field snapshots. The exact gradient and Adam moments are available only in the final `optimizer-latest.pt`; the frozen protocol's claim that each saved step includes them is incorrect.
