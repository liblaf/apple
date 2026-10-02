# Strict HVP PNCG expression validation

The strict validation run did not complete its base equilibrium and provides no gradient-validation acceptance. It was deliberately interrupted after retaining a step-1000 checkpoint because the force residual remained far above the proposed `1e-12` threshold. The scale-resolved follow-up is `99-validate-resolved-expression-gradients.py`.

## Command and provenance

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME='Validate strict HVP PNCG expression gradients' CHERRIES_TAGS='expression,validation,pncg,hvp,rigid-eyes' uv run python src/98-validate-precise-hvp.py > logs/98-validate-precise-hvp-terminal.log 2>&1
```

[Comet run](https://www.comet.com/liblaf/apple/31090c35436649f79b45eea7162c919e). Cherries used `ProfileJoint`; source snapshots and Git status are in `data/precise-hvp-expression-validation-001/provenance.json`. The source was not changed after launching. The process exited with status 130 after SIGINT and completed its shutdown.

## Observed results

The base probe used `2e-6 MPa * I` on all 288,235 active tetrahedra and jaw pose `[1e-7, -1e-7, 1e-7, 1e-7, 0, -1e-7]`. The implementation used exact directional Hessian products and resolved Gauss-integrated line work with PNCG.

- Step 500 force: `2.1826755637372852e-10`.
- Step 1000 force: `1.8532492189234578e-10`.
- Best force among the every-50-step recorded samples: `1.183860765017346e-10`.
- Last recorded step 1050 force: `2.7491035884324937e-10`.
- Interrupted base elapsed time: `360.5362549739948 s`.

The recorded line searches resolved extremely small energy changes without sampled backtracking, but this did not establish practical convergence to the strict force threshold. Force residuals oscillated. No adjoint or finite-difference comparison was performed by this run.

## Evidence and limitations

`data/precise-hvp-expression-validation-001/summary.json` retains `success: false` and the interruption receipt. `base/trace.jsonl` contains accepted-state progress; `base/accepted-latest.npz` contains the full step-1000 displacement. `base-receipt.json` records `KeyboardInterrupt()`; the terminal log retains the traceback and Cherries/Comet shutdown evidence. The checkpoint is a warm start, not an equilibrium accepted at `1e-12`.

The originally proposed `1e-6 MPa` and `1e-6` pose finite-difference steps would demand greater primal resolution than normal fit proposals. Follow-up validation tests two larger physical perturbation scales while retaining the 5% gradient-agreement gate and the established force/contact acceptance checks. The strict run itself does not authorize expression fitting.
