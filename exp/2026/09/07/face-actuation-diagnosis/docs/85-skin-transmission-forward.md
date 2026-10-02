# Skin-transmission forward ablation

This is a forward-only 2 × 2 comparison of the saved historical Raw6 field and
its already validated strong within-muscle diffusion field, with skin factors
0 and 0.12. It does not optimize an activation field, smooth output geometry,
or reject geometry based on inversion or activation spectra.

The two skin-factor-0 cells are immutable references to the successful
`data/80-forward-field-diffusion/{baseline,strong}` states. The runner writes
only the two new skin-factor-0.12 forward equilibria, each initialized from
the same saved displacement `u` in
`data/38-historical-adam-raw6-continuation/final.npz`. It checks the source
checkpoint, selected field `q`, fixture volume/skin/summary, and saved no-skin
state hashes before solving.

`src/85-historical-adam-skin-physics.py` is a literal copy of
`src/historical_adam_physics.py` with one deletion: the guard that rejects a
nonzero `skin_factor`. The runner asserts that exact one-deletion relationship
before a solve. Every other physics and solver guard remains in force:
classical Lamé values, no contact, volume potentials, fixed values, target,
PNCG maximum 5000, forward/adjoint relative tolerance `5e-4`, absolute
tolerance `1e-10`, and the existing line-search implementation.

For the two new cells, the only altered constitutive term is the existing
Koiter skin block on the fixture's rest `skin.vtp`: `skin_factor=0.12`, so
`E=0.2 × 0.12 = 0.024 MPa`, `nu=0.46`, thickness `0.001 m`, and zero
prestrain. The runner records material and solver contracts, source hashes,
seed hash, Piola-at-seed/equilibrium diagnostics, and deformation/inversion
statistics without using them as an acceptance gate.

The completed run used this durable-service command, with the absolute environment
interpreter and the non-committing Cherries profile embedded in the runner:

```bash
systemd-run --user --unit=face-skin-transmission-85 \
  --working-directory=exp/2026/09/07/face-actuation-diagnosis \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=COMET_AUTO_LOG_GIT_METADATA=false \
  --setenv=COMET_AUTO_LOG_GIT_PATCH=false \
  --setenv=COMET_AUTO_LOG_ENV_DETAILS=false \
  --setenv='CHERRIES_NAME=Skin transmission forward ablation' \
  --setenv='CHERRIES_TAGS=gpu,face,forward,field-diffusion,skin-transmission,no-commit' \
  .venv/bin/python src/85-forward-field-skin-transmission.py
```

## Completed execution

The approved service ran on 2026-09-07 and exited successfully. It used the
command documented above with the absolute interpreter, CUDA device 0, the
three Comet autolog switches disabled, and `ProfileCometNoCommit`. Comet
recorded the run at
[Skin transmission forward ablation](https://www.comet.com/liblaf/apple/8f15dd061c094b688b79fdbef2d845c7).

The terminal receipt is
`data/85-forward-field-skin-transmission/service-exit-receipt.json`: unit
`face-skin-transmission-85.service`, invocation
`a09b1539634545bda1cfe1158980d719`, inactive/dead after completion, result
`success`, and main exit status 0. Full captured inner-solver stdout is kept in
`data/85-forward-field-skin-transmission/solver-stdout.log`.

Both new skin-on forward solves reported valid equilibria. The baseline field
with skin factor 0.12 gave area fit RMS 1.403238604 mm and motion RMS
4.156301709 mm. The strong-diffusion field with skin factor 0.12 gave area fit
RMS 1.841614082 mm and motion RMS 3.597674262 mm. These are forward outcomes
for the specified saved fields; they do not add an inverse result or a geometry
admissibility condition.
