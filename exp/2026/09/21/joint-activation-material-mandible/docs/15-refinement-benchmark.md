# Full-face Newton-CG refinement benchmark

## Scope

`src/15-benchmark-newton-refinement.py` tests a local exact-Hessian refinement from the frozen contact equilibrium written by `09-validate-face-gradients.py`. The benchmark uses the plus side of the fat shared-stress direction at `h=0.003`, with the same `q=0.01` activation field, zero jaw pose, contact configuration, and forward tolerances as validation 09.

The script runs a matched PNCG solve and Newton-CG independently from the same frozen displacement. Newton uses exact FEM plus IPC Hessian-vector products. The absolute Hessian diagonal is only a preconditioner. Every Newton step requires a successful linear solve, a finite descent direction, CCD admission, and Armijo decrease on a fresh owned contact state. There is no MINRES or PNCG fallback. The final nonlinear criterion remains `max(1e-12, 1e-6 * initial warm-state force norm)`.

## Results

Both report-worthy runs passed on the full face. They ran while the remaining full-face derivative validation and an unrelated user workload shared the GPU, so the wall times are explicitly contended.

| Method | Linear tolerance | Wall time | Nonlinear steps | Terminal force norm | Matched PNCG wall time | Measured speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Newton-CG | `1e-8` | 20.85 s | 2 | `1.04e-13` | 42.06 s | 2.02x |
| Inexact Newton-CG | `1e-3` | 10.85 s | 2 | `1.04e-13` | 49.86 s | 4.60x |

All four Newton steps used `CCD=1`, accepted `alpha=1` on the first Armijo trial, and retained 143 active contact pairs with a terminal minimum active gap near `49.65 um`. The inexact run's two measured linear relative residuals were `9.97e-4` and `9.98e-4`; nonlinear force convergence, rather than the linear tolerance, determined success.

The inexact terminal displacement differed from its matched PNCG solution by at most `3.05e-8 m` in any component. Its scalar face-fit loss differed by `3.20e-9`. The strict run's corresponding differences were `3.13e-8 m` and `1.81e-8`. Both terminal states had zero inverted tetrahedra and minimum `det(F)` about `0.650`.

Machine receipts and both displacement fields are under `data/refinement-benchmark/` and `data/refinement-benchmark-inexact/`. The normal Cherries runs are [strict Newton-CG](https://www.comet.com/liblaf/apple/cd4b1a3675d549e88d78f477265715bf) and [inexact Newton-CG](https://www.comet.com/liblaf/apple/afa6d844593440d498030830500dca8c).

## Decision boundary

This is one warm perturbation, not proof that Newton-CG is robust for all shared fields, continuation stages, or expression states. The inexact result supports a bounded solver pilot with the same explicit failure gates. It does not justify silently replacing PNCG, relaxing CCD, or using an inexact linear residual as the nonlinear convergence test.
