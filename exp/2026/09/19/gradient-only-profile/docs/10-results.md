# Gradient-only fitting: smoother profiles, much weaker deformation

All eight gradient-only runs completed 1,200 updates with no forward-solver failures. Compared with the saved positional-L2 runs, gradient-only fitting reduced reference-slope error and second-derivative error in every case, but produced almost flat profiles. Peak displacement reached only **0.15–1.96% of the requested height**. Positional RMS error increased by **3.04–9.58%** (the largest increase compares against an early-stopped baseline).

This experiment does **not** support gradient-only loss as a satisfactory replacement for positional fitting under the current demo and optimizer settings. It suppresses the bumps while largely losing the desired deformation. This is a finite-budget outcome, not proof of the best attainable gradient-only fit or of target unreachability.

## Objective and controlled setup

For the full ordered top chain, including fixed corners,

$$
L_g=\frac1S\sum_i\frac{\|e_{i+1}-e_i\|^2}{\Delta x_i},\qquad
 e_i=u_i-(0,4h x_i(1-x_i)),\quad S=1.
$$

Only this scalar enters the adjoint and Adam update. Positional L2 and activation-smoothness weights are both **exactly zero**. Fixed endpoints anchor translation. Both displacement components are differentiated with respect to fixed reference x; the loss is a correspondence-based vector gradient match, not a normal-only or intrinsic geometric curve distance. Gradients are compared against the identically sampled target.

The 100 × 10 mesh, central muscle band, material, fixed sides/bottom, corrected physical-volume energy using J=det(F), four activation models, heights 0.05/0.20, inactive initialization, feasibility projections, 1,200-update budget, and Adam settings are inherited unchanged. Adam starts at 0.03 and multiplies its learning rate by 0.99 per update. No positional term, regularization, material adjustment, inverse line search, restarts, or alternate seed was added. See [protocol](00-protocol.md).

The saved L2 comparison is `exp/2026/09/15/activation-direction-smoothness/data/tune-w0`. Hashes of its physics, mesh/target helper, activation models, and study adapter exactly match the sources reused here. Raw gradient and positional objectives have different units; their numeric values are not compared to one another. The same raw-objective Adam hyperparameters are a controlled first trial, not separately tuned optima for each loss.

## Results

All quantities use the demo's coordinate units (reference width = 1). Position RMS uses the historical free-top-node definition. Slope RMS integrates the full-chain reference-gradient residual. The second-derivative metric is a weighted finite difference of that slope residual; it is **not geometric curvature**. Peak is maximum vertical displacement divided by target height.

| Target height | Model | Position RMS: L2 → gradient | Slope RMS: L2 → gradient | Second-derivative RMS: L2 → gradient | Peak/target: L2 → gradient |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.05 | Free symmetric | 0.03321 → 0.03594 | 0.3161 → 0.1131 | 27.054 → 0.434 | 29.63% → 1.96% |
| 0.05 | PSD contraction | 0.03458 → 0.03646 | 0.2899 → 0.1146 | 30.839 → 0.424 | 22.23% → 1.03% |
| 0.05 | Learned direction | 0.03467 → 0.03633 | 0.2861 → 0.1142 | 27.909 → 0.439 | 16.73% → 0.88% |
| 0.05 | Fixed x | 0.03561 → 0.03669 | 0.1856 → 0.1154 | 8.328 → 0.432 | 8.11% → 0.15% |
| 0.20 | Free symmetric * | 0.13301 → 0.14576 | 1.0650 → 0.4585 | 70.513 → 1.696 | 45.24% → 0.67% |
| 0.20 | PSD contraction | 0.13970 → 0.14608 | 0.7170 → 0.4593 | 33.669 → 1.739 | 20.04% → 0.85% |
| 0.20 | Learned direction | 0.13984 → 0.14615 | 0.6322 → 0.4597 | 18.195 → 1.706 | 15.12% → 0.43% |
| 0.20 | Fixed x | 0.14011 → 0.14670 | 0.6280 → 0.4614 | 11.581 → 2.126 | 16.09% → 0.30% |

\* The height-0.20 free-symmetric **L2** run stopped after update 262 because its next forward solve failed; its saved last-valid state is shown. Every gradient run and the other seven L2 runs reached 1,200 updates. This is not an equal-budget comparison for that one pair.

Across all eight comparisons, slope RMS fell by **26.5–64.2%** and second-derivative residual RMS by **81.6–98.6%**. Those improvements must be read with the near-flat shapes and worse position fit. The gradient-only objective fell only **0.076–4.12% from the undeformed starting state**. A visually smooth endpoint alone is not evidence that target shape has been recovered.

![All target and fitted profiles](../data/30-figures/profiles-all-cases.png)

![Learned-direction full deformed meshes](../data/30-figures/learned-direction-deformed-meshes.png)

[Machine-readable common metrics](../data/30-figures/summary.csv), [slope-error profiles](../data/30-figures/slope-error-profiles.png), and [metric comparison](../data/30-figures/final-shape-metrics.png).

## Why this can happen

For the analytic parabola on x∈[0,1] and zero endpoint displacement, integration by parts gives

$$
\int_0^1\|u'-t'\|^2dx
=\int_0^1\|u'\|^2dx-16h\int_0^1u_y\,dx+\frac{16h^2}{3}.
$$

Thus the target-dependent term rewards mean upward displacement, while the quadratic term penalizes displacement gradients. A local bump accompanied by compensating dips may improve positional fit but buy little in this metric. With almost incompressible material and fixed sides/bottom, broad upward motion is mechanically costly. This is a plausible explanation for the observed weak response, not a demonstrated separation of material capacity from optimization effects.

For the sampled target actually used here, the exact uniform-grid constant is `(16*h**2/3)*(1-dx**2)` and the linear term is `-16*h*dx*sum(u_y at interior top nodes)`. At dx=0.01 the initial loss divided by h² is 5.3328, matching the saved traces. The endpoints remove the constant nullspace: a zero gradient residual would recover the sampled target exactly. Near-flat outcomes are therefore **not** caused by missing translation anchoring or by the objective being indifferent to target amplitude.

At finite tangential displacement, area change also depends on u_x': the signed top-boundary area increment is `integral u_y*(1+u_x') dx`. The mean-u_y argument alone is consequently a small-displacement interpretation, not an exact incompressibility proof.

## Validation

- Direct displacement derivative: worst relative centered-difference error **3.57e-9**.
- End-to-end solve/adjoint/control derivatives: worst relative error **7.20e-6** across all four parameterizations (gate 2e-5).
- The sampled target has exactly zero loss and displacement gradient; fixed top corners are included.
- All 8 gradient and 8 L2 saved endpoints replayed with exactly zero fixed-node displacement. Maximum equilibrium residual across them: **9.20e-11**.
- Gradient endpoints: minimum physical J across cases **0.7499–0.9792**, with no inversions.
- The smallest algebraic eigenvalue of each constrained forward Hessian was positive: **1.40e-5–4.49e-5** for gradient endpoints and **1.04e-5–5.02e-5** for L2 endpoints. Eigenpair residuals were checked. This supports local forward stability of these endpoints, not inverse convergence.
- Scoped Ruff checks (excluding the repository-wide copyright rule CPY001), formatting, and `git diff --check` passed.
- Finite-difference checks, independent endpoint replay, source reuse hashes, and eigenvalue details: [checks.json](../data/20-checks/checks.json).

The decaying learning rate becomes approximately 1.7e-7 by the budget endpoint. A flat optimization trace is not a stationarity certificate. These results use synthetic 2D geometry and do not establish an anatomical result or full-face behavior.

## Reproduction and provenance

Working directory:

```text
exp/2026/09/19/gradient-only-profile
```

Report-producing experiment command (existing project interpreter):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='2D gradient-only target matching versus saved L2' \
CHERRIES_TAGS='gradient-only,2d,physical-J,activation-models' \
.venv/bin/python src/10-run.py --output 10-gradient
```

Local verification and plotting:

```bash
DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Gradient-only profile endpoint verification' \
CHERRIES_TAGS='gradient-only,2d,verification' \
.venv/bin/python src/20-verify.py

DEBUG=1 CHERRIES_NAME='Gradient-only profile comparison figures' \
CHERRIES_TAGS='gradient-only,2d,figures' \
.venv/bin/python src/30-render.py
```

The output directory is created exclusively to prevent overwriting evidence; choose a new `--output` for another optimization or verification run. Numerical source snapshots and SHA-256 hashes, Python/NumPy/SciPy versions, settings, and thread environment are in [protocol.json](../data/10-gradient/protocol.json). Each case preserves checkpoint, sampled history, full scalar trace, and summary. Git HEAD was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the workspace had pre-existing uncommitted research changes. New work for this task is confined to this experiment group; no commit was made.

Cherries used the experiment profile with Local, Logging, Git(commit=False), and Comet enabled for the full run. [Comet experiment](https://www.comet.com/liblaf/apple/b8d262c54dde405aade42eba3139c11d). [Complete run log](../logs/10-run.log). Numerical computation finished in about 232 seconds including initial setup. The turn was subsequently interrupted during Comet environment-metadata shutdown; the process is no longer running and no final Comet Experiment Summary was emitted. Final remote metadata/upload completion is therefore unconfirmed. All eight local numerical cases, independent validation, and figures were already complete. The historical mesh-helper import registers an unused `10-pork-2d` asset, causing a missing-asset warning; the intended `10-gradient` output and all eight cases were written and independently verified.
