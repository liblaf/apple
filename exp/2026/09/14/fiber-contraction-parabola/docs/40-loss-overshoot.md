# L-BFGS loss curves and trial-step overshoot (historical diagnostic)

**Optimizer correction:** the plots below describe projected L-BFGS with Armijo, not Adam. This optimizer substitution was my mistake. The requested comparison has now been rerun with pure Adam; use the [Adam results and loss curves](./70-adam-results.md) for the current demo. The old data remain available to explain the earlier rejection counts.

2026-09-14. The 2D demo uses the corrected Stable Neo-Hookean law: **J=det(F)**. This is explicit in [physics2d.py:62](../src/physics2d.py:62), and its current hash matches the original run's [protocol.json](../data/10-comparison-final/protocol.json). Only the norm term uses F @ A_inv. Both determinant terms and their derivatives use physical F.

**The free-activation cases propose uphill trial steps, which the line search rejects. The accepted loss never increases.** The nearly flat h=0.20 x-contraction run is slow progress, not oscillating loss.

## All trial evaluations

The original trace saved only accepted states and forward-solve failures. Finite rejected losses were missing, so accepted-loss plots alone could not answer the overshoot question. A diagnostic replay added trial logging without changing the energy, optimizer, solver, initial conditions, or stopping rules. Every saved displacement and control array in all four replayed trajectories matches the original exactly: maximum absolute difference **0.0**.

The plotted loss is the optimizer's actual normalized objective,

$$
\ell=\frac{1}{h^2}\frac{1}{N_{\rm top}}\sum_i\|u_i-u_{T,i}\|^2.
$$

This is squared error divided by h², not the RMS used in the shape-comparison report. An uphill trial means its loss exceeds the current accepted loss by more than `max(1e-12, 1e-10*abs(current_loss))`.

| Target / control | Evaluations including initial state | Uphill trials | Largest relative uphill jump | Forward failures | Accepted loss increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| h=0.05 / x contraction | 141 | 0 | 0% | 18 | 0 |
| h=0.05 / free B | 392 | 308 | 3.748% | 1 | 0 |
| h=0.20 / x contraction | 301 | 0 | 0% | 0 | 0 |
| h=0.20 / free B | 104 | 7 | 3.127% | 53 | 0 |

![All trial losses and the accepted incumbent](../data/60-trial-loss-plots/trial-loss.png)

Black is the accepted loss held between accepted steps; red marks uphill proposals. Blue crosses show failed forward solves at a fixed display height, **not measured loss values**. All panels use the same vertical scale. The evaluation index starts at zero for the initial state.

For h=0.05 free B, 309 finite trials were rejected: 308 increased loss, and the last returned exactly the current loss. Their largest absolute increase was 0.01730850 in normalized loss. For h=0.20 free B, all seven finite rejected trials increased loss; the largest increase was 0.01464408. The remaining rejected trials failed the forward solve and have no equilibrium loss to plot.

## Accepted progress

![Accepted losses by update and by evaluation count](../data/40-loss-curves/accepted-loss.png)

| Target / control | Initial loss | Final accepted loss | MSE reduction |
| --- | ---: | ---: | ---: |
| h=0.05 / x contraction | 0.53872053 | 0.43702200 | 18.878% |
| h=0.05 / free B | 0.53872053 | 0.45609302 | 15.338% |
| h=0.20 / x contraction | 0.53872053 | 0.53804110 | 0.126% |
| h=0.20 / free B | 0.53872053 | 0.45276935 | 15.955% |

The outer Armijo rule accepts only a sufficiently decreasing loss, which explains the monotone accepted curves. It does not guarantee inverse convergence or stable forward equilibria. The existing endpoint checks still apply: all four inverse gradients remain above threshold, and both free-B endpoints have negative Hessian directions. See the [main results report](10-results.md).

The h=0.20 x-contraction run accepts every proposed step, but its per-step maximum control update is only 1.328e-4–1.350e-4. Its plateau therefore provides no evidence that its step is too large. A blanket reduction of the step size is not supported by these curves. In the free-B cases, rejected uphill proposals are clear, but a loss curve alone does not separate an overly large step on one smooth equilibrium branch from changes between equilibrium branches.

The [projected-gradient plot](../data/40-loss-curves/projected-gradient.png) shows the difference between declining loss and reaching the requested stopping tolerance. The [trial-loss-change plot](../data/60-trial-loss-plots/trial-loss-change.png) uses a symmetric log scale to expose small positive trial-loss changes.

## Reproducibility and artifacts

Working directory: `exp/2026/09/14/fiber-contraction-parabola`.

All three commands use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false`, normal Comet recording, and no automatic Git commit:

```bash
CHERRIES_NAME='Parabola accepted-loss and overshoot audit' \
CHERRIES_TAGS='2d,parabola,loss,overshoot,diagnostics' \
uv run python src/40-plot-loss.py

CHERRIES_NAME='Parabola trial-loss replay' \
CHERRIES_TAGS='2d,active-strain,parabola,replay,trial-loss' \
uv run python src/50-replay-trial-loss.py

CHERRIES_NAME='Parabola all-trial loss overshoot plots' \
CHERRIES_TAGS='2d,parabola,loss,overshoot,replay' \
uv run python src/60-plot-trial-loss.py
```

Existing output directories are preserved; scripts refuse to overwrite their normal output directory. Use a distinct output path for another run. The replay verifies SHA-256 of the original inverse runner, 2D physics, and optimizer before execution, then checks the complete accepted displacement/control histories afterward. The original four runs were not modified.

- [Accepted-loss audit JSON](../data/40-loss-curves/loss-audit.json) and [PDF](../data/40-loss-curves/accepted-loss.pdf).
- [Every trial in CSV](../data/50-trial-loss-replay/trial-loss.csv) and [replay identity checks](../data/50-trial-loss-replay/summary.json).
- [Trial audit JSON](../data/60-trial-loss-plots/trial-audit.json) and [PDF](../data/60-trial-loss-plots/trial-loss.pdf).
- Normal Comet runs: [accepted-loss plot](https://www.comet.com/liblaf/apple/c389649deb774b629155d50c679bbb6f), [identical replay](https://www.comet.com/liblaf/apple/8706e2619cb546fca86bf0772bbc97f1), [all-trial plot](https://www.comet.com/liblaf/apple/62b3765b4f474bceb74d9f78ee9973de). Their `Comet.ml Experiment Summary` blocks are in [40-loss-launch.log](../logs/40-loss-launch.log), [50-replay-trial-loss.log](../logs/50-replay-trial-loss.log), and [60-trial-plot-launch.log](../logs/60-trial-plot-launch.log).

The replay and both plot processes completed normally; all three exited 0. Plotting-source Ruff checks passed, and the figures were visually inspected. A draft replay with incomplete logging of the initial incumbent was stopped before analysis and preserved in `data/50-trial-loss-replay-unseeded`; only the corrected replay with matching histories supplies the trial data above.
