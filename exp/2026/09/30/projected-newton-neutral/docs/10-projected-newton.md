# Projected Newton for the IsFixed neutral forward solve

The neutral forward solve drops from **97.6 s to 21.1 s** (4.6×) on the local
RTX 4090. The only change is to the Newton search matrix: each element's
Hessian and the IPC barrier Hessian are projected to positive semidefinite
(PSD) before the sparse solve. PNCG is skipped. The projected run converges in
20 Newton steps instead of 340 PNCG + 134 Newton steps. The endpoint is valid:
0 inverted tetrahedra, feasible contact, force 0.0081 N against the unchanged
0.01 N gate.

The original neutral (`new-neutral/data/forward-isfixed-001`) turned out to be
**under-converged by about 0.17 mm RMS**. The error is up to about 0.9 mm at the
lower lip, mouth socket and chin, even though it passed the 0.01 N force gate.
Projected Newton reaches the true equilibrium at the same gate (details below).

## Why the solve was slow

This profile comes from the original `forward-isfixed-001` `timing.json`
(97.6 s total).

| Phase | Seconds | Cause |
| --- | ---: | --- |
| PNCG (340 steps) | 34.3 | Force oscillated between 0.5 and 5 N and only fell from 3.0 to 1.4 N. IPC candidate builds took 15.5 s (run twice per step) and CCD took 4.1 s. |
| Newton (134 steps) | 63.3 | See below. |
| PCG (27,600 iterations) | 20.7 | Up to 650 iterations per solve. |
| Numeric FEM refresh | 15.9 | 0.12 s × 133. |
| One-time FEM sparsity build | 12.0 | numpy `unique`/`searchsorted` over 18.3M block keys. |

Newton itself converged only linearly. The exact Hessian is **indefinite**
throughout the solve, and 131 CG solves stopped on negative curvature. The
Stable Neo-Hookean `λ(J−α)∂²J` term is indefinite even near rest, and with
ν = 0.49 it is large. At the endpoint, the per-material element Hessians with
negative eigenvalues number 707k for fat, 213k for muscle, 89k for aponeurosis
and 76 for skin. The bulk potentials overlap on mixed-fraction tetrahedra, so
these counts are element Hessians, not unique tetrahedra. The recovery added a global diagonal shift.
The `reuse` policy then divided the shift by 10 at each step until it failed
again, so shifts cycled between 2.8e-3 and 2.8e-6. The steps were damped
Levenberg–Marquardt steps rather than Newton steps.

## Changes

These changes are experiment-local, in `src/projected_hessian.py`. No library
code was changed.

1. **Per-element PSD projection** of the Newton search matrix, the standard
   projected Newton used by IPC.
   - Each 12×12 tetrahedron and 9×9 membrane Hessian is symmetrized.
   - A batched Cholesky with a 1e-10 relative shift screens out elements that
     are already PSD.
   - For the remaining elements, the negative spectrum is removed:
     `H+ = H − V min(Λ,0) Vᵀ`. The eigenpairs come from a scaled float32
     `eigh`, which is 4.5× faster than float64. The residual indefiniteness is
     at most 7e-6 of the element scale.
   - The IPC barrier Hessian uses `ipctk.PSDProjectionMethod.CLAMP`.
   - Energy, force, line search, CCD, the maximum step size and the stopping
     rule are unchanged.
2. **Newton-only scope.** `ProjectedHybridHessian` is used only by the hybrid
   solver's `SparseNewtonProblem`. The implicit-adjoint path in
   `mouthopen_runtime.py` keeps building the exact `HybridHessian`. The
   projected IPC Hessian is placed in `state.collision.hess` only during the
   CSR refresh, and the exact cache is restored afterwards.
3. **GPU sparsity build.** `torch.unique` replaces numpy `unique`/`searchsorted`
   for the FEM block topology. The output is bit-identical and takes 0.23 s
   instead of 4.5 s. Scatter slots are uploaded once, and the cell batch is
   16,384.
4. **Skip PNCG** (`--skip-pncg true`). Projected Newton does not need the
   warm start.

## Results

All runs start from zero displacement on the clearance-repaired reference,
unless a seed is listed.

| Run | Solver | Forward s | PNCG / Newton | Terminal force (N) | Inverted | Valid |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `new-neutral/.../forward-isfixed-001` | original hybrid, exact Newton | 97.6 | 340 / 134 | 0.0096 | 0 | yes |
| `forward-exact-control-001` | same, through the refactored assembly | 81.5 | 249 / 129 | 0.0099 | 0 | yes |
| `forward-projected-hybrid-001` | PNCG then projected Newton | 46.9 | 100 / 14 | 0.0086 | **1** | no |
| `forward-projected-newton-001` | projected, float64 `eigh`, CPU topology | 46.7 | 0 / 20 | 0.0099 | 0 | yes |
| `forward-projected-newton-002` | + float32 `eigh`, GPU topology | 21.9 | 0 / 21 | 0.0047 | 0 | yes |
| **`forward-projected-newton-003`** | + Newton-only subclass (final) | **21.1** | 0 / 20 | 0.0081 | 0 | yes |
| `forward-projected-newton-kappa8-001` | final, κ fixed at 1.3544 MPa | 22.8 | 0 / 25 | 0.0060 | 0 | yes |

The control's sparse Hessian-vector product matches the matrix-free product to
3.07e-16. This confirms the refactored assembly is exact when projection is off.
Final-run breakdown: 8.6 s projected FEM refresh (about 0.43 s per refresh),
8.1 s PCG (about 670 Jacobi-CG iterations per step), 0.6 s FEM construction,
and under 2 s of IPC work in total.

## Which equilibrium is correct

The endpoint of `projected-newton-003` differs from the original by 171 µm RMS.
The difference is concentrated at the lower lip, mouth socket, chin and neck.
Fixing κ at the original's final 1.3544 MPa gives the same difference, so it
does not come from the adaptive barrier stiffness path.

Both endpoints were then continued at a 100× tighter force target (1e-10
MPa·m², i.e. 0.0001 N), with the same fixed κ:

| Seed | Newton steps | Seconds | Movement during polish (RMS / p99 µm) | Final energy (MPa·m³) |
| --- | ---: | ---: | ---: | ---: |
| original `forward-isfixed-001` | 170 | 255.5 | 168.8 / 615.5 | 2.4575559e-06 |
| `projected-newton-kappa8-001` | 8 | 9.9 | 0.94 / 3.3 | 2.4575550e-06 |

The two polished endpoints agree to 4.35 µm RMS. Only 5 mouth-socket contact
vertices differ by more than 100 µm. The original was therefore far from its
equilibrium in the soft lower-face tissue, where the 0.01 N absolute force gate
allows large displacement error in 0.011 MPa fat. The projected result was
already within 1 µm RMS of equilibrium.

This affects downstream work built on the old neutral, including the
blendshape transfer and the MouthOpen and Smile fits. Their "neutral" lower lip
is about 0.2–0.5 mm away from the mechanical equilibrium.

## Limitations and next steps

- The runs are single runs, not repeated timings. Timing uses synchronized
  hierarchical scopes, so it adds sync overhead.
- The PSD projection changes only the search direction. The implicit adjoint
  must stay exact, and it does, because only the Newton `SparseNewtonProblem`
  uses the projected matrix. It has not been exercised in an inverse fit yet.
- The largest remaining costs:
  - **PCG:** Jacobi-preconditioned CG needs about 670 iterations per step. A
    3×3 vertex-block Jacobi preconditioner is cheap to add. Alternatively, a
    lagged cuDSS Cholesky factor could be used as the preconditioner: the
    projected matrix is SPD and its free CSR pattern is fixed. The one-time
    symbolic analysis costs about 11 s, so this pays off in inverse loops.
  - **Projection:** an analytic Stable Neo-Hookean eigensystem (Smith et al.
    2018) would replace the batched `eigh`.
- The loose absolute force gate should be revisited, because it certified an
  endpoint that was 0.17 mm RMS off.

## Reproduction

From `exp/2026/09/30/projected-newton-neutral`:

```bash
CHERRIES_NAME='Neutral projected Newton' OMP_NUM_THREADS=4 \
  .venv/bin/python -u src/10-forward-projected.py \
  --output-dir data/forward-projected-newton-repeat --skip-pncg true --projection clamp
```

`src/check_projection.py` checks the projection against exact `eigh` and the
GPU topology against the numpy build. `tmp/run-all.sh`, `tmp/run-d.sh` and
`tmp/run-tight.sh` record the remaining commands. `--projection none` with the
default hybrid path reproduces the exact control. `--seed-endpoint` and
`--target-force-override` drive the polish runs.
