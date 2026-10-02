# Signed-stress material validation

The experiment-local bulk and membrane implementations passed all 32 declared
CPU checks. The machine-readable receipt is
[`data/material-validation/summary.json`](../data/material-validation/summary.json),
with archived source hashes in
[`data/material-validation/provenance.json`](../data/material-validation/provenance.json).

The bulk gate used a signed symmetric stress with eigenvalues
`[-0.00691, 0.00437, 0.01254] MPa`. The skin gate used a signed tangent
resultant with eigenvalues `[-22.31, 42.31] N/m`. Thus the checks exercise the
signed baseline contract rather than only positive activation.

Both materials matched independent Torch energy, force, Hessian-vector product,
Hessian diagonal, and quadratic-form references on non-axis-aligned cells.
Directional finite differences agreed to relative errors below `1.1e-9`.
Rigid-transform energy invariance and force/Hessian covariance held to absolute
errors below `4e-17`. Warp mixed derivatives with respect to signed stress,
`lmbda`, and `mu`, plus skin thickness, matched Torch to absolute errors below
`3e-18`. The eliminated skin normal stretch left `|P33|=1.11e-16 MPa`.

The Cherries smoke command was:

```bash
COMET_AUTO_LOG_GIT_METADATA=0 \
COMET_AUTO_LOG_GIT_PATCH=0 \
COMET_AUTO_LOG_ENV_DETAILS=0 \
COMET_AUTO_LOG_CODE=0 \
DEBUG=1 CHERRIES_NAME=05-material-validation \
uv run python src/05-validate-materials.py --run-cuda-smoke=false
```

CUDA was deliberately skipped in this receipt while the GPU was occupied by
the approved full-mesh pilots. The independent full-mesh assembly smoke covers
device integration; this receipt isolates the constitutive formulas and their
derivatives.
