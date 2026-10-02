# Pure Adam with the corrected Stable Neo-Hookean model

**Comparison extended:** a third case now enforces contraction in arbitrary principal directions using `B-I` positive semidefinite. The [three-way comparison, implementation and verification](./100-contraction-only.md) reproduces the original x/free trajectories exactly and adds new loss/shape plots and animations. This report preserves the earlier two-way run and its provenance.

2026-09-14. **The current comparison uses Adam throughout.** The earlier diagnostic used projected L-BFGS/Armijo; substituting that optimizer was my mistake. Its rejection counts and monotone accepted-loss plots describe L-BFGS only. The old data are preserved and labeled in [the previous report](./40-loss-overshoot.md).

Adam retains finite loss increases. It has no outer line search or loss-based rejection in this run. Both x-contraction cases decreased monotonically. Free activation had small loss increases, and the larger-target case eventually failed its next forward solve. None of these runs establishes inverse convergence.

## Model and optimizer

The production library and 2D model both retain the corrected physical determinant:

$$
B=A^{-1},\quad J=\det F,\qquad
W=\tfrac12\mu(\|FB\|_F^2-2)-\mu(J-1)+\tfrac12\lambda(J-1)^2.
$$

This is the plane-strain reduction of the corrected 3D law. The determinant terms do not use `det(F @ B)`. The [run protocol](../data/70-adam-comparison/protocol.json) records the physics source hashes; the [independent audit](../data/90-adam-endpoint-checks/endpoint-checks.json) verified the snapshots and live 2D physics. The earlier library regression suite passed 29 tests; this rerun did not change the material implementation.

The same historical rectangle, 100×10 grid, 2,000 triangles, 400 muscle triangles, bottom/side fixation, materials and parabola are retained. Target displacement is `(0, 4*h*x*(1-x))`, for h=0.05 and 0.20. Coordinates have no verified physical length calibration here; RMS units are L. Full geometry and material details are in [the model specification](./10-results.md#2-固定的实验设置).

Each case starts from identity activation and zero displacement:

- **x contraction:** `B=diag(1+a,1)`, with `a>=0`, 400 controls. Projection occurs after each Adam update. This restricts active strain, not the actual deformation F.
- **Free activation:** unconstrained symmetric B, 1,200 controls. This is an algebraic comparison without positivity, invertibility or amplitude constraints on B.

The update reproduces the historical 2D Adam settings: initial learning rate 0.03, per-update decay 0.99, betas `(0.9,0.999)`, epsilon `1e-8`, and at most 1,200 updates. Adam differentiates **raw top-node vector MSE**. Division by h² is for plotting and comparative gradient diagnostics only; it is not applied to the Adam gradient.

$$
q_t=\Pi\!\left(q_{t-1}-0.03(0.99)^{t-1}
\frac{\widehat m_t}{\sqrt{\widehat v_t}+10^{-8}}\right).
$$

Here projection is nonnegative clipping for x contraction and identity for free activation. There is no gradient clipping, step cap, outer Armijo filter, or L-BFGS refinement. The independent two-update Adam formula check had maximum absolute discrepancy **0**.

The inner Newton equilibrium solver still uses a line search, residual infinity-norm tolerance `1e-10`, and positive physical determinants. This inner search is separate from Adam. A failed equilibrium is recorded and stops that case; it is not assigned an invented loss or followed by an optimizer switch. A solved state with minimum J≤1e-6 would also stop the case diagnostically.

## Actual loss curves

![Adam loss histories](../data/80-adam-figures/adam-loss.png)

All solved states are plotted, including loss increases. Crosses mark increases larger than `max(1e-12, 1e-10*abs(previous_normalized_loss))`; final dots are the last solved states. Initial normalized loss is 0.53872053 for every case.

| Target / controls | Last solved update | Final MSE / h² | Final RMS / L | MSE reduction | Loss increases | Largest one-step rise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h=0.05 / x contraction | 1200 | 0.50709744 | 0.03560539 | 5.87% | 0 | 0% |
| h=0.05 / free B | 1200 | 0.44119758 | 0.03321135 | 18.10% | 7 | 0.05959% |
| h=0.20 / x contraction | 1200 | 0.49075139 | 0.14010730 | 8.90% | 0 | 0% |
| h=0.20 / free B | 262 | 0.44228121 | 0.13300845 | 17.90% | 10 | 0.33214% |

The first three cases reached their update budget. The fourth saved 263 valid states including the initial state, then its update-263 proposal failed: `Forward line search failed, residual=3.191e-06`. That is **263 successful equilibrium/adjoint evaluations plus one failed forward attempt**. No loss is available for that failed proposal. The original summary's `function_evaluations` field counts successful evaluations only; use the separate failure record to obtain total attempted evaluations.

![Adam loss changes and learning rate](../data/85-adam-update-plots/adam-updates.png)

The upper row uses a symmetric log scale to reveal small positive loss changes. The lower row shows the learning rate used to reach each solved state. In the three 1,200-update cases, the last applied rate is **1.75275e-7**, with maximum control changes about 1.65e-7–1.82e-7. Their normalized projected gradient infinity norms remain 1.36e-4, 3.24e-4 and 8.22e-5. The large free case has 9.97e-4. The flattening curves therefore do not establish stationarity; the inherited decay is making the late updates very small.

The observed rises establish that Adam can overshoot in this setup. They do not by themselves prove inaccurate gradients or identify an excessive global learning rate. L-BFGS constructs curvature estimates from gradient differences, so inexact equilibria or changes between equilibrium branches can disrupt those estimates. Adam avoids this curvature-history mechanism, but it still depends on useful gradients and solvable forward states. This run does not isolate the cause of each loss rise.

## Shapes and endpoint checks

![Small target, last solved states](../data/80-adam-figures/adam-deformation-h050.png)

![Large target, last solved states](../data/80-adam-figures/adam-deformation-h200.png)

| Endpoint | Minimum physical J | Smallest algebraic Hessian eigenvalue | Minimum singular value of B |
| --- | ---: | ---: | ---: |
| h=0.05 / x contraction | 0.797898 | 3.976e-5 | 1 |
| h=0.05 / free B | 0.466046 | 1.476e-5 | 3.891e-6 |
| h=0.20 / x contraction | 0.857661 | 3.370e-5 | 1 |
| h=0.20 / free B | 0.00176353 | 1.038e-5 | 8.493e-6 |

All four saved endpoints have positive smallest Hessian eigenvalues. These are independent dense smallest-algebraic eigenpair checks, not the nearest-zero values in the runner summary. Endpoint residuals are at most 6.72e-14, fixed boundary error is exactly zero, and integrated J agrees with the boundary area within 4.2e-17. Eigenpair relative residuals are at most 1.01e-16. These checks support local forward stability of the saved states, not inverse convergence or global geometric validity.

Free B improves fit within the allotted runs, but becomes nearly singular and indefinite. The small and large free cases have det(B)≤0 in 39.5% and 60.75% of muscle cells; their minimum eigenvalues of B are -0.5668 and -1.2002. Such controls are not validated muscle activations. The large free shape also has four top-edge segments reversing in x, despite positive element determinants. Neither a lower loss nor positive local J certifies a useful face model.

The target boundary would require discrete area ratios 1.3333 and 2.3332 under the fixed side/bottom boundary, whereas the saved shapes remain near unit area. This is a substantial conflict with the near-incompressible material, not a proof that the finite-penalty model can never fit the target.

## Next meeting and next experiment

Use four result slides: (1) the physical-J energy and its verified derivatives; (2) the two active-strain control spaces; (3) these Adam loss/learning-rate plots; (4) equal-scale shapes with physical J, B singular values and explicit stop reasons. Present the free-control fit gain together with its invalid activation modes. Do not transfer the old L-BFGS endpoint instability claims to these Adam endpoints.

Keep Adam as the baseline. The next controlled optimization experiment should vary only its learning-rate policy, comparing the inherited 0.99 decay with a constant rate while retaining raw loss, epsilon, initial state and physics. Before drawing a conclusion about the attainable fit, include a forward-generated reachable target as a diagnostic alongside the original parabola. For the failed large-target free case, a separate local probe from the same saved state can compare smaller Adam steps and stricter forward tolerance; do not mix that probe with a new activation constraint or call it a completed repair. These are proposed follow-ups, not results of this run.

## Artifacts and reproducibility

- [Loss PDF](../data/80-adam-figures/adam-loss.pdf), [loss-change/LR PDF](../data/85-adam-update-plots/adam-updates.pdf), and [plot statistics](../data/80-adam-figures/adam-plot-metrics.json).
- [h=0.05 animation](../data/80-adam-figures/adam-evolution-h050.mp4) and [h=0.20 animation](../data/80-adam-figures/adam-evolution-h200.mp4). Frames use every fifth saved update and the final update, without interpolation. In the large-target animation, the stopped free case is explicitly held at its last state.
- [Run summary](../data/70-adam-comparison/summary.json), per-case `trace.csv`/`history.npz`, [source snapshots](../data/70-adam-comparison/source), and [endpoint audit](../data/90-adam-endpoint-checks/endpoint-checks.json).
- Comet: [Adam computation](https://www.comet.com/liblaf/apple/12cec0f8628b4cc787f8a4029091c8e6), [figures](https://www.comet.com/liblaf/apple/936fe2f2fa864cba8f6b7799953bd9bc), [loss changes](https://www.comet.com/liblaf/apple/4fd9a680be134bb9b6a02fc62df23efa), [endpoint audit](https://www.comet.com/liblaf/apple/5454d5eb69304c66aadc18fec4afb30b).

Run from `exp/2026/09/14/fiber-contraction-parabola` with the environment below. Each script records normally to Comet with automatic Git commits disabled. Outputs refuse to overwrite an existing directory; use a new `--output` when repeating a run.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false

CHERRIES_NAME='Parabola pure-Adam comparison' \
CHERRIES_TAGS='2d,active-strain,parabola,adam,fiber-contraction' \
uv run python src/70-run-adam.py

CHERRIES_NAME='Pure Adam parabola loss and deformation plots' \
CHERRIES_TAGS='2d,parabola,adam,loss,active-strain' \
uv run python src/80-render-adam.py

CHERRIES_NAME='Adam retained loss changes and learning rate' \
CHERRIES_TAGS='2d,parabola,adam,overshoot,learning-rate' \
uv run python src/85-plot-adam-updates.py

CHERRIES_NAME='Parabola Adam endpoint checks' \
CHERRIES_TAGS='2d,active-strain,parabola,adam,verification' \
uv run python src/90-verify-adam.py
```

The Adam numerical computation completed locally. Its roughly 128,000 remote metric messages triggered Comet throttling; the process was terminated during the upload backlog after all local numerical outputs were written. A clean process exit is not claimed for that run, and remote metrics may be incomplete. The renderer, update plots and endpoint audit exited normally. Their logs contain the Comet summary blocks; local CSV and NPZ files supply the reported trajectories.

The completed run's final checkpoints contain controls and displacement, but not the full Adam moments. Its failed proposal separately contains controls and moments. **The source snapshot is the authoritative version for these results.** The current runner has subsequent bookkeeping-only changes to retain full last-valid Adam state, count failed forward attempts explicitly and reduce remote metric frequency. These changes do not retroactively add moments to the completed run. No commit or push was made; existing unrelated work remains in the checkout.

The updated runner passed a four-case, two-update [checkpoint smoke](../data/70-adam-checkpoint-smoke/summary.json): each checkpoint exactly matches its saved control/displacement endpoint, retains counter 2 and nonzero moments, and passes the independent Adam formula check. That smoke exited 0 after Comet shutdown. Ruff passed for scripts 70, 80, 85 and 90; `git diff --check` passed. The figures were visually inspected and both H.264 animations were verified as 241 frames at 20 fps (12.05 seconds).
