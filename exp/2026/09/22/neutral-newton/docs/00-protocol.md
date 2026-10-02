# Single forward recomputation of the neutral face

Start from the existing constitutive reference, as explicitly selected by the user.
The raw reference intersects the fixed eyeballs, so construct a geometric contact
initializer before solving. This initializer may move free soft-tissue nodes but
does not change the material reference, rigid geometry, material fields or skin
target. No saved equilibrium displacement enters the initialization.

The model retains the complete source cranium, mandible and both fixed eyeballs
as frictionless soft-versus-rigid IPC obstacles. The mandible's one rotation
coordinate is held at zero for neutral. Muscle expression activation and bulk
additive baseline stress are zero. Fat, muscle and aponeurosis have Young moduli
11.2, 12 and 1693 kPa, respectively, with Poisson ratio 0.49. Skin has the existing
127.382–257.861 kPa field, Poisson ratio 0.49, thickness 1 mm, and prescribed
30.400–80.830 N/m baseline membrane tension.

The bulk energy is

`W = mu/2 * (tr(F^T F)-3) - mu*(J-1) + lambda_code/2*(J-1)^2 + Q:(F^T F-I)/2`.

The coefficient `lambda_code = lambda_classical + mu` reproduces the specified
small-strain E and nu for this polynomial. Skin uses the corresponding exact
plane-stress membrane. Q is zero in the bulk for this neutral solve; skin's
prescribed baseline tension remains present.

Run exactly one Newton-CG forward solve after geometric initialization:

| Setting | Value |
| --- | --- |
| Gradient gate | `norm(g) <= max(1e-8, 1e-3 * norm(g_initial))` |
| Newton iteration limit | 100 |
| Coordinate displacement cap | 0.5 times mean unique reference FEM edge length |
| Armijo coefficient | 1e-4 |
| Backtracking | factor 0.5; at most 8 trials, including initial trial |
| Shift sequence per Newton iteration | 0, s, 10s, ..., 1e6s; 8 attempts total |
| Shift scale | `s = mean(abs(diag(H)))` |
| PCG preconditioner | `abs(diag(H + shift I))` |
| True relative PCG residual | 1e-3 |
| PCG iteration limit | 1000 per shifted system |

Hessian-vector products and the assembled diagonal are exact derivatives of
the physical energy. Process-local bindings remove the bulk's historical
per-element diagonal clipping and replace the IPC Gauss–Newton diagonal with
the exact cached Hessian diagonal. Selected diagonal entries are checked
against coordinate Hessian-vector products before the forward.

Contact retains the physical IPC barrier, with 0.1 mm activation distance and
0.01 MPa stiffness. CCD uses 0.1 nm numerical tolerance, a 10 nm minimum gap,
and a 0.9 step fraction when collision limits a trial. No PNCG stage, load
continuation, inverse solve, or alternate forward solver is used. The requested
Adam learning rate 1.0 is recorded but performs no updates in a neutral forward.

Gradient units are MPa m²; multiplying by 1e6 gives newtons. Force convergence,
positive tetrahedron orientation, and collision validity are reported separately.
The new output is saved as its own experiment result. Historical neutral bundles
and expression targets are preserved.

The historical geometry/material artifacts retain their original hashes. A new
runtime binding documents solver-package import migrations, generated version
metadata and lockfile changes; it verifies the historical data and material
implementation rather than modifying old receipts. Exact current sources are
archived with the run. Cherries records the command and Comet run; Git commits
are disabled by the experiment profile.
