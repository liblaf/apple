# Validation before the learned-axis comparison

The CPU activation gate passed algebra, feasible projection, axis-sign
invariance, eigenaxis initialization, and tensor-smoothness derivative checks.
See [checks.json](../data/05-activation/checks.json).

The full implicit face derivative gate checks strength and tangent-axis
perturbations at `s=0.05` in every active cell, using the target-gradient-derived
neutral axes. Central differences use perturbations 0.01 and 0.005.

| Perturbation | Epsilon | Analytic slope | Numerical slope | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Strength | 0.01 | 10.2440309240 | 10.2443348474 | 0.002967% |
| Strength | 0.005 | 10.2440309240 | 10.2442512383 | 0.002151% |
| Tangent axis | 0.01 | 0.0832707431 | 0.0832821784 | 0.013733% |
| Tangent axis | 0.005 | 0.0832707431 | 0.0832961631 | 0.030527% |

All checks satisfy the predeclared 2% error and step-size-agreement thresholds.
The gate uses PNCG `rtol=1e-6`, `atol=1e-12`, maximum 10,000 iterations; the
adjoint tolerance remains `5e-4`. This tighter equilibrium accuracy is used
only for finite differences. Training uses the inherited forward `rtol=5e-4`,
`atol=1e-10`, maximum 5,000 iterations in both branches.

The first validation attempt at training tolerance failed the smaller axis
perturbation (14.51% relative error, versus 1.24% at the larger perturbation).
Reducing the solver stopping error brought both axis checks below 0.031%; the
error threshold itself was not relaxed. The failed attempt is preserved under
`data/06-preparation/`, with its exception in the corresponding Cherries log.
Its gate file says `running` because that first driver did not persist failure
status. It is not accepted as a passed gate. All comparisons instead require
[the tight gate](../data/06-preparation-tight/gradient-validation.json), its
matching source hashes, and the calibrated initial axes from that run.

The eight-update data-only calibration probe reduced the scaled gradient loss
from 7.3811095363 to approximately 6.8460. It supplies the coefficient scale,
not the starting state of any comparison branch. Every pilot and final branch
restarts at `s=0`, displacement zero, and fresh optimizer state.

Commands run from this experiment directory, with
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`:

```bash
DEBUG=1 CHERRIES_NAME='Learned-axis face: tight equilibrium derivative validation' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,gradient-only,validation' \
.venv/bin/python src/10-run.py \
  --phase prepare --output 06-preparation-tight
```

The process completed with exit code 0, including Cherries shutdown. The local
snapshot is under the repository's
`.cherries/runs/2026/09/19/learned-axis-gradient-face/10-run/2026-09-19T155341-Learned-axis-face-tight-equilibrium-derivative-validation/`.
