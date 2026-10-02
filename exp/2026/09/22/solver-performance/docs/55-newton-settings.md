# Newton-CG settings

Updated on 2026-09-22. This changes the experimental Newton-CG solver and its
Newton-only, hybrid, and adaptive call sites. No full forward or inverse
performance benchmark was rerun.

| Setting | New default |
| --- | --- |
| True relative CG residual | `1e-3` |
| CG iteration limit | `1000` per shifted system |
| Preconditioner | `abs(diag(H + lambda I))` |
| Initial shift | `0`, reset each Newton iteration |
| Shift retries | `s = mean(diag(H))`, then multiply by `10` |
| Shift attempt limit | `8` total, including the unshifted attempt |
| Coordinate displacement cap | `0.5 * mean unique rest FEM edge length` |
| Armijo coefficient | `1e-4` |
| Backtracking | Multiply step size by `0.5`; `8` trials per direction |
| Newton CCD safety | `0.9`, applied once when CCD limits the step |
| Newton iteration limit | `100` |
| Forward wall limit | `None` |

The eight default shifts are `0, s, 10s, 100s, 1000s, 10000s, 100000s,
1000000s`. The specified signed mean must be positive and finite; an invalid
scale fails visibly. A failed linear solve or exhausted Armijo search advances
to the next shift. Each failed line search restores the original state before
retrying. Eight Armijo trials includes the initial trial, so its last step size
is the initial safe step size divided by `128`.

The rest-edge mean counts each undirected tetrahedral or membrane edge once,
including shared edges only once across material potentials. Collision-only
obstacle geometry does not enter this scale. It is computed once when the
accelerated runtime is installed. The cap bounds the maximum absolute free
coordinate displacement, rather than the Euclidean displacement of a vertex.

## Forward algorithm

The physical objective contains bulk and membrane elasticity, prescribed
active stress and prestress, and IPC contact with the anatomical obstacles.
Dirichlet coordinates remain prescribed. Each Newton iteration computes the
physical energy, free-coordinate force gradient, and Hessian diagonal. PCG
approximately solves `(H + lambda I) p = -g`; negative curvature, a failed
relative-residual check, or the CG budget causes a shifted retry. A successful
direction must be descent.

The solver caps its coordinate displacement, applies CCD, and backtracks until
the physical energy meets Armijo decrease. The shift changes only the search
system, never the physical energy or the implicit-adjoint Hessian. The existing
accepted-state force convergence criterion still decides equilibrium.

Hybrid mode first uses PNCG until the force reaches
`max(final_atol, 1e-3 * initial_force, newton_switch_atol)`, then uses Newton-CG.
Adaptive mode instead uses PNCG and inserts a single Newton correction when
its progress detector triggers. Both use the new Newton settings. PNCG's own
Armijo coefficient, displacement cap, and CCD safety are preserved; the Newton
CCD safety override is scoped to its CCD call and restored afterwards.

The runtime's default Hessian operator uses exact Warp FEM and IPC products.
The separately tested GPU sparse assembly backend is not installed by this
parameter change. Existing experiment snapshots preserve the earlier solver
sources and results; new runtime receipts record the effective Newton settings.

## Validation

Focused CPU checks cover the PCG tolerance and limit, absolute Jacobi
preconditioning, shift sequence, line-search limits and rollback, coordinate
cap, scoped CCD safety, nullable wall budget, and mesh-edge definition.
Existing accelerated solver, single Newton step, hybrid, adaptive, and explicit
shift-reuse checks passed. Runtime files and current runner sources parse;
the changed solver and helper modules pass Ruff.

Run the focused policy checks from the repository root:

```sh
.venv/bin/python exp/2026/09/22/solver-performance/src/check_newton_policy.py
.venv/bin/python exp/2026/09/22/solver-performance/src/check_mesh_step_scale.py
```

These are direct CPU contract checks, not Cherries benchmark runs. There is no
new performance, convergence-speed, or inverse-fitting result from this change.
