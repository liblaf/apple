# MouthOpen with coupled jaw and tissue prediction

Run date: 2026-09-30 (Asia/Shanghai).

## Result

The 20 accepted updates reduced area-weighted skin RMS from 7.625210953 to 6.304117916 mm (17.325%). The final jaw rotation magnitude is 1.125018296 degrees, and translation magnitude is 1.521550072 mm.

The independent rebuild confirmed free residual 0.00887735628828 N <= 0.01 N, configured contact feasibility, and 6 inverted retained tetrahedra. Their rest-volume fraction is 2.97931618303e-07 (2.97931618e-05%), with minimum det(F) -1.197069337. This is an accepted inversion approximation under the user-authorized policy, not an inversion-free result.

The inverse stopped at its declared update budget (`finite_budget_exhausted`), and **inverse convergence was not established**. The rendered mouth opening remains substantially smaller than the target. The forward endpoint converged.

Full surface review over Tailnet (private preview omitted).

## Model and predictor

- Excluded 2,249 cells whose four original FEM vertices have IsFixed. Retained 1,144,268 mechanical tetrahedra and 288,172 active muscle cells, each with six unrestricted Raw6 components. Bulk cells and aligned constitutive arrays are filtered; original vertices, IsFixed constraints, skin and collision geometry stay indexed consistently. Endpoint active IDs identify original cells; local material IDs are separate.
- The actual neutral free-gradient relative change after exclusion was 6.68558e-15; the free Hessian-product relative change was 5.59209e-17.
- Preserved the neutral skin pre-strain, mu, thickness, material reference, complete source bones and eyes. The inverse uses the inherited terminal IPC stiffness 1.3544 MPa, frozen for a consistent objective.
- Replaced the 1-degree / 1-mm proposal cap with CCD-admitted steps and nonlinear correction. The proposed update can be larger; actual accepted increments remain limited by predicted contact. Both jaw and free tissue move in each CCD-screened seed. Rigid arc deviation uses the mandible's radius, not the fixed skull's radius.
- The final predictor solves `(H_ff + lambda I) du_f = -(delta_f_material + H_fc delta_u_c)` with old material/contact Hessian and `lambda = 0.001 * mean(abs(diag(H_ff)))`. Sparse and native shifted residuals must both meet 1e-7. It is a numerical seed; the collision-on hybrid PNCG/Newton corrector evaluates the physical energy and force.
- Declared inversion allowance: at most 100 retained cells and at most 0.0001 retained rest-volume fraction. There is no per-cell minimum-J floor. Each accepted endpoint still requires finite force, convergence, and configured contact checks.

## Continuation and predictor repair

Run 001 accepted 12 updates and then encountered `line_search_stalled`. The original predictor included an unscaled `-r_old` term. Although the old residual met the forward contract, its predicted correction approached 0.410 mm while the requested jaw increment approached zero. CCD therefore kept rejecting arbitrarily small parameter updates.

The final predictor leaves that within-tolerance residual to the nonlinear corrector and predicts only the response to parameter changes. It asserts the source state meets the force tolerance. An actual full-model zero-update check returned RHS 0.0, free displacement 0.0 m, CCD fraction 1.0, and bit-exact unchanged full coordinates; model state restoration passed.

Run 002 restarted from the saved Run 001 endpoint, with Adam moments reset, and accepted 8 further updates. Both phases retain separate immutable protocols, source snapshots, endpoints, trial logs, and Comet records. The review joins their curves at the verified matching endpoint.

## Verification and limits

Five focused tests passed (cell filtering / ID remapping / retained geometry / IsFixed semantics). Static checks passed for the changed scripts. The independent endpoint audit rebuilt the model, restored saved pose and Raw6, checked original IsFixed/free arrays and free lip nodes, and verified skin pre-strain arrays exactly.

The independent contact check found 1268 active pairs, minimum active gap 75.519847 micrometres, and no intersections in the configured mesh. d_hat is 100 micrometres; the CCD minimum separation is 0.01 micrometres. Contact covers selected soft boundary against complete cranium, mandible and eyes. Soft-soft/lip-lip and rigid-rigid contact remain disabled. The soft surface still excludes any face with a Cranium or Mandible GroupId vertex; it is not all free boundary faces.

Six inverted cell IDs and locations are in `data/inverse-mouthopen-coupled-002/independent-audit.json`. The display uses the full original tetrahedral boundary, including geometry excluded from bulk assembly. The rendered target is transferred skin only. The adjoint retains a declared relative shift 0.001 and provides approximate gradients; the gradient test validates the fixed-state pullback, not a fully re-equilibrated finite-difference gradient.

The endpoint’s unweighted skin RMS is 6.496998608 mm; this differs from the area-weighted optimizer RMS above.

## Commands and artifacts

Working directory: `exp/2026/09/23/new-neutral`. Interpreter: `.venv/bin/python`. Solver launches used `OMP_NUM_THREADS=4`, human-readable `CHERRIES_NAME`, and `CHERRIES_TAGS` including `mouthopen,isfixed,active-strain,skin-prestrain,collision,coupled-predictor,retained-tets`.

```bash
python -u src/145-fit-mouthopen-coupled.py
python -u src/145-fit-mouthopen-coupled.py --output-dir data/inverse-mouthopen-coupled-002 --initialization-checkpoint data/inverse-mouthopen-coupled-001/checkpoint.pt --maximum-iterations 8
python -u src/146-audit-mouthopen-coupled.py --run-dir data/inverse-mouthopen-coupled-002
PYVISTA_OFF_SCREEN=true python -u src/147-review-mouthopen-coupled.py --run-dir data/inverse-mouthopen-coupled-002 --parent-run-dir data/inverse-mouthopen-coupled-001
```

Run 001 used its archived pre-repair predictor; current source includes the zero-update correction. Source snapshots under each run record that difference. Cherries Git commits were disabled, and pre-existing workspace changes were preserved. All four processes exited successfully and completed their shutdown hooks.

- [145-fit-mouthopen-coupled-001](https://www.comet.com/liblaf/apple/5b4caea3de684aa19ebdde05b0743491)
- [145-fit-mouthopen-coupled-002](https://www.comet.com/liblaf/apple/3af074a84745407f93c1f988522fcd1f)
- [146-audit-mouthopen-coupled-002](https://www.comet.com/liblaf/apple/892a1c1faffb4831ab335c278a4e5b06)
- [147-review-mouthopen-coupled-001](https://www.comet.com/liblaf/apple/80d69705d9c24085a114951f912fc30b)

Primary outputs: `data/inverse-mouthopen-coupled-002/endpoint.npz`, `checkpoint.pt`, `independent-audit.json`; `data/review-mouthopen-coupled-001/` contains the full-boundary meshes, target/fit comparisons, force and fit curves, jaw/inversion curves, and input hashes.
