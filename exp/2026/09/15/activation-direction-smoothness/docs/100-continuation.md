# Why the free-activation run stopped, and a continuation test

## Meaning of FAILED

Only the **free-activation, h=0.20, no-smoothness** panel in the main off/on comparison stopped early. The activation proposed for update 263 triggered `Forward line search failed, residual=3.191e-06`. This refers to the inner mechanical equilibrium solve, which attempts to reduce elastic energy while preserving positive physical triangle area. Its 40 displacement backtracking trials could not satisfy the acceptance conditions. It is not an inverse-convergence message or proof that another control update or equilibrium path cannot work.

The published figure displays the successfully solved state at update 262. Its force residual is 6.71e−14, and its smallest Hessian eigenvalue is positive. Its minimum physical J=det(F), however, is only 0.00176353.

## Actual continuation test

The saved checkpoint contains q262, u262, and Adam moments m262/v262. The original proposal q263 and its moments were reconstructed to within 1e−14 absolute tolerance, preserving counter=263 and learning rate 0.03×0.99^262. The continuation kept the same fixed corrected Stable Neo-Hookean material, L2 objective, muscle band, free-activation parameterization, and zero smoothness weight.

The new optimizer tests fractions 1, 1/2, 1/4, … of each Adam update from the same immutable valid state. Rejected fractions do not advance moments. Acceptance requires the original 1e−10 equilibrium tolerance, the original minimum-J diagnostic threshold of 1e−6, an outer Armijo decrease in raw L2, and a positive equilibrium Hessian. This is a separate backtracked-Adam diagnostic, not the original equal-hyperparameter trajectory.

The first restart succeeds at a quarter of the original activation update. Thereafter the acceptable fraction shrinks rapidly:

| Accepted step | Fraction of nominal Adam update | L2/h² | Minimum physical J |
| ---: | ---: | ---: | ---: |
| 263 | 0.25 | 0.442254755 | 0.00035875 |
| 264 | 0.0625 | 0.442249142 | 3.9105e-05 |
| 265 | 0.0078125 | 0.442248462 | 3.2642e-06 |
| 266 | 0.00048828 | 0.442248420 | 1.2583e-06 |
| 267 | 6.1035e-05 | 0.442248414 | 1.0337e-06 |
| 268 | 1.9073e-06 | 0.442248414 | 1.0337e-06 |
| 269 | 1.9073e-06 | 0.442248414 | 1.0337e-06 |
| 270 | 1.9073e-06 | 0.442248414 | 1.0337e-06 |

The fit improves by **0.007414%** relative to step 262. Steps 268–270 repeat the same reported displacement loss and min J; their control changes are too small for the forward solver to resolve a new displacement at its tolerance. The test was manually interrupted during the next trial, with **exit code 130**, to stop repeated expensive evaluations at this plateau. The valid step-270 checkpoint and logged accepted-step metrics were retained. The interrupted run did not flush its in-memory full history or rejected-trial array, so those records are not claimed. The runner now stops cleanly at the first accepted update with exactly unchanged displacement and loss; the executed pre-change source is preserved under `data/100-backtracked-continuation/source/`.

## Independent endpoint verification

| Quantity | Original step 262 | Continued step 270 |
| --- | ---: | ---: |
| L2/h² | 0.4422812053 | 0.4422484145 |
| Minimum physical J | 0.00176353238 | 1.03368685e-06 |
| Force residual ∞-norm | 6.71e-14 | 2.44e-11 |
| Smallest Hessian eigenvalue | 1.03821e-05 | 1.03704e-05 |
| Projected-gradient ∞-norm | 0.000996802 | 0.00104404 |

The limiting element is **passive fat triangle 1348**, reference centroid (0.746667, 0.0633333), immediately above the muscle band. Its B=I. It has almost collapsed in physical space; this minimum J is not a small activation determinant in that cell. Both saved endpoints pass equilibrium, positive-J, fixed-boundary, and positive-Hessian checks. The nonzero inverse gradient means neither is demonstrated converged.

**Conclusion:** restarting is possible, but smaller steps along this Adam path lead toward element collapse and negligible fitting progress. Merely increasing the iteration budget does not address that behavior. Meaningful further progress should be investigated with updates that steer away from degenerating elements, while retaining matched optimizer settings when comparing models. This test does not rule out a better feasible path. The original comparison outputs and slide exports remain the original recorded study.

## Evidence and reproduction

[Continuation protocol](../data/100-backtracked-continuation/protocol.json) · [Saved checkpoint](../data/100-backtracked-continuation/checkpoint.npz) · [Logged accepted progress](../data/100-backtracked-continuation/logged-accepted-progress.csv) · [Independent checks](../data/110-continuation-checks/checks.json)

Working directory: `exp/2026/09/15/activation-direction-smoothness`. Normal Cherries runs used `COMET_AUTO_LOG_GIT_PATCH=false`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

```bash
CHERRIES_NAME='Continue stopped free activation with backtracked Adam' \
CHERRIES_TAGS='2d,activation,continuation,backtracking,l2' \
uv run python src/100-continue.py

CHERRIES_NAME='Verify continued free activation checkpoint' \
CHERRIES_TAGS='2d,activation,continuation,verification' \
uv run python src/110-check-continuation.py
```

Use a fresh `--output` for reruns. The current continuation source includes the explicit stagnation stop described above; its executed historical snapshot is retained for the interrupted diagnostic. The source hashes of study.py, activation_models.py, physics2d.py, and the mesh implementation match the original numerical run. The original checkpoint SHA-256 is unchanged.

Continuation: [Comet](https://www.comet.com/liblaf/apple/592bc6b8c5f44d058493a275a4fe5d82), [summary](../data/100-backtracked-continuation/continuation-comet-summary.txt), [terminal log](../logs/continue-terminal.log), exit 130 after manual interrupt and shutdown. Verification: [Comet](https://www.comet.com/liblaf/apple/abe50852f32342b594d1942bd2a23b84), [summary](../data/100-backtracked-continuation/verification-comet-summary.txt), [terminal log](../logs/check-continuation-terminal.log), exit **0**. Ruff and formatting pass for the current runner and verifier.
