# Mixed-loss validation evidence

The loss algebra/autograd gate and the first full implicit derivative gate
completed successfully before the loss pilots. This report records actual
saved checks; selected-weight validation and final geometry verification are
reported separately in the final results.

The CPU [algebra receipt](../data/05-loss/checks.json) verifies beta=0.25,1,4,
pure-L2 and pure-gradient endpoints, equal neutral values, and the analytic
linear combination of constituent derivatives. The receipt hashes both the
loss implementation and the checking script. DEBUG is supplied by the command.

For beta=1, the [full implicit derivative gate](../data/06-validation/gradient-validation.json)
perturbs all activation strengths and tangent axes at strength 0.05. It includes
the equilibrium solve and implicit adjoint, using central differences at two
step sizes:

| Direction | Difference step | Analytic derivative | Numeric derivative | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Strength | 0.01 | 9.444515431 | 9.444859612 | 0.003644% |
| Strength | 0.005 | 9.444515431 | 9.444799153 | 0.003004% |
| Tangent axis | 0.01 | 0.239437263 | 0.239289262 | 0.061812% |
| Tangent axis | 0.005 | 0.239437263 | 0.239250407 | 0.078040% |

The declared relative tolerance is 2%, including the step-size agreement check.
Finite differences use forward maximum 10,000 iterations, relative tolerance
1e-6 and absolute tolerance 1e-12. Optimization retains maximum 5,000 iterations,
relative tolerance 5e-4 and absolute tolerance 1e-10. Both retain the existing
10-trial line search. Tightening validation equilibrium tolerance avoids
confounding small axis perturbations with approximate-solve error.

Normalization is recomputed from exact neutral displacement before any fitting:
`L20=8.656092875221388`, `Lg0=0.07413172241432929`, `K=L20`. No changing or
learned denominator is used. The common initial axes and passed rank-one
activation gate are inherited by hash from the preceding learned-axis study.

The initial pilot diagnostics also record actual projected first Adam updates
in physical tensor space; equal neutral objective value is not interpreted as
equal optimizer motion.

The selected beta=4 received a separate full-chain check after its fresh
eight-update smoothness-calibration probe. The
[selected-objective receipt](../data/10-preparation/gradient-validation.json)
also passed. Strength relative errors were 0.000499% and 0.000320%; tangent-axis
errors were 0.231886% and 0.251358% at difference steps 0.01 and 0.005. The maximum
remained below the predeclared 2% tolerance. Training tolerances were restored
after validation. The [calibration receipt](../data/10-preparation/calibration.json)
records base coefficient 5.702275093589208, obtained from data-gradient norm
0.04701076421640372 and regularizer-gradient norm 0.00824421190574542.

Working directory:
`exp/2026/09/19/mixed-loss-learned-axis-face`.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DEBUG=1 CHERRIES_NAME='Mixed face loss: normalized objective algebra' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,verification' \
.venv/bin/python src/05-verify-loss.py

DEBUG=1 CHERRIES_NAME='Mixed face loss: full implicit derivative validation' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,validation' \
.venv/bin/python src/10-run.py \
  --phase validate --output 06-validation --beta 1

DEBUG=1 CHERRIES_NAME='Mixed face loss: selected blend validation and smoothness calibration' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,calibration,validation' \
.venv/bin/python src/10-run.py \
  --phase prepare --output 10-preparation --beta 4
```

Existing output directories cause the runner to fail rather than overwrite an
experiment. For independent reruns, use a new output directory and pass the
corresponding validation/preparation paths to later phases.
