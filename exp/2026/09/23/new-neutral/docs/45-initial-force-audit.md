# Initial force and adaptive barrier stiffness

The initial free-force norm is dominated by contact. Reconstructing the saved
active-strain model and evaluating its gradients at the recorded seed gives:

| Component | Initial free-force L2 norm (N) | Terminal free-force L2 norm (N) |
| --- | ---: | ---: |
| Contact barrier | 574.7594 | 0.9534 |
| All materials combined | 42.5641 | 1.0770 |
| Fat | 1.7888 | 1.4016 |
| Aponeurosis | 42.3151 | 1.2004 |
| Muscle | 0.5612 | 0.3751 |
| Skin | 3.0240 | 1.2967 |
| Total residual | 576.4153 | 0.5269 |

These are Euclidean norms of all unconstrained Cartesian force components,
not net resultant forces. Component vectors sum; their norms do not. Fixed
DOFs are excluded. The initial largest free nodal contact force is 340.760 N.
The raw model gradient has units MPa m², converted to N by multiplying by 1e6.

The seed is intersection-free but its minimum gap is only 54.974 nm, compared
with barrier activation distance 100 micrometers and minimum admissible gap
10 nm. Its 9,078 active contact terms generate a large barrier gradient.
Thus geometric feasibility does not imply small initial contact forces.
The final gap is 65.056 micrometers with 2,707 active contact terms.

## Adaptive stiffness and stopping rule

Adaptive IPC stiffness is enabled. Initialization prescribes
`kappa = 0.1 * max(E) = 0.1693 MPa`, using the aponeurosis modulus of 1.693 MPa.
It does not choose initial kappa by balancing material and contact gradients.
After each accepted solver step, the controller passes previous/current
minimum squared gaps to `ipctk.update_barrier_stiffness`, with geometric scale
from the reference bounding box, epsilon scale 1e-6, and maximum kappa 16.93 MPa.
It accepts proposed increases and invalidates derivative caches. A Newton line
search retains one fixed stiffness throughout that line search.

The saved run doubled kappa after PNCG steps 61, 73, 105, 137, and 194:

```text
0.1693 -> 0.3386 -> 0.6772 -> 1.3544 -> 2.7088 -> 5.4176 MPa
```

The force threshold is computed once at the initial anchor stiffness:
`max(1e-8, 1e-3 * initial_force)` in raw units, equivalent to
`max(0.01 N, 0.576415 N)`. It is not recomputed as kappa changes.
The final residual passes this rule, but the reference norm is dominated by
near-contact initialization. The residual remains about 49% of the terminal
material-force norm. Consequently this relative test alone does not establish
a tightly balanced endpoint; geometry also remains invalid with 13 inverted
tetrahedra. No tolerance, initialization, or forward result was changed by this
audit.

## Evidence and reproduction

The [audit receipt](../data/initial-force-audit-001/force-components.json) records
per-component norms, top nodal forces, input hashes, stiffness events, and
gradient decomposition checks. Reconstructed initial and terminal total norms
match the saved run. Evaluation used the same active-strain material sources
and strict input binding. Differentiable/adjoint entry points were guarded.
No forward steps were taken.

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='New neutral initial force audit' \
CHERRIES_TAGS='neutral,active-strain,force,contact,audit' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/45-audit-initial-force.py
```

Use a fresh `--output-dir` to reproduce. The
[Comet run](https://www.comet.com/liblaf/apple/8f3073d4ea004b2e8e9c793451eddf98)
completed with initial contact 574.759365 N, initial total 576.415317 N,
terminal contact 0.953444 N, and terminal total 0.526903 N. Ruff passed.
An inherited asset logging hook reported a missing `data/simple-skin-forward`
path in this experiment group; the explicit source binding, saved audit output,
and norm reproduction succeeded.
