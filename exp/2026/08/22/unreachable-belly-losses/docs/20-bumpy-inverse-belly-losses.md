# Bumpy inverse belly evolution under L1, L2, and L-inf losses

## Outcome

The second controlled experiment now shows the requested behavior: both the 2D profile and the 3D height surface start smooth, then develop visible scallops or dimples while the inverse solve tries to fit a **smooth but unreachable** target. The bumps are therefore produced by the inverse parameterization, not copied from a bumpy target.

The clearest result is L-inf: its optimized state has the largest high-pass bump amplitude in both dimensions. Across all losses, the optimized 2D states are 5.52--6.17 times bumpier than the smooth target, and the optimized 3D states are 2.33--2.80 times bumpier.

![2D L-inf best bumpy state](../data/20-bumpy-inverse-belly-losses/2d/linf/best.png)

![3D L-inf best bumpy state](../data/20-bumpy-inverse-belly-losses/3d/linf/best.png)

## Why the first experiment stayed smooth

The original experiment was smooth by construction. Its 2D state was restricted to three broad actuator responses, and its 3D state to a broad $3\times3$ response grid. The membrane and bending terms smoothed those responses again. The deliberately unreachable target component was projected outside this small response space, so it could remain in the residual but could never appear in the optimized geometry. Changing L1, L2, or L-inf only selected different coefficients inside the same smooth span.

Unreachability alone does not create bumps. The missing ingredients were localized inverse controls and the absence of a smoothing regularizer on those controls.

| design choice | first experiment: smooth control | second experiment: bumpy inverse |
| --- | --- | --- |
| 2D controls | 3 broad responses, width 0.26 | 9 local responses, width 0.065 |
| 3D controls | $3\times3$ broad responses, width 0.31 | $5\times5$ local responses, width 0.10 |
| target | reachable teacher response plus an orthogonal narrow residual | smooth continuous-load equilibrium response |
| initial controls | 0.05 | 0, so bump formation starts from a smooth state |
| equilibrium smoothing | $k_m=0.9,\ k_b=0.15$ | finite but weaker: $k_m=0.25,\ k_b=0.015$ |
| possible optimized geometry | only broad, smooth combinations | discrete scallops and dimple grid |

The first experiment remains useful as the negative control: [smooth 2D L2 state](../data/10-unreachable-belly-losses/2d/l2/best.png) and [smooth 3D L2 state](../data/10-unreachable-belly-losses/3d/l2/best.png).

## Controlled model

Both models retain the same fixed-rim linear surface-equilibrium form,

\[
K = k_0 I + k_m L + k_b L^2,
\qquad
K u_j = f_j,
\qquad
u(c)=Rc=\sum_j c_j u_j,
\]

with $k_0=1$, $k_m=0.25$, $k_b=0.015$, and bounded controls $0\le c_j\le1$. The 2D profile has 161 points and nine local Gaussian actuator loads. The 3D surface has a $41\times41$ grid and 25 local loads. Their response peaks are normalized to 0.05 and 0.04, respectively.

The target is generated independently from a smooth cosine-squared distributed load passed through the same equilibrium operator. Its maximum depth is 0.040 in 2D and 0.035 in 3D. The inverse model can use only the finite local actuator set, so approximating the continuous smooth sag leaves the observed scallops between actuator sites.

No roughness, total-variation, or curvature penalty is applied. Adding such regularization would suppress the behavior this experiment is intended to expose.

## Unreachable-target certificate

For response matrix $R$ and target $t$, the unconstrained least-squares projection is

\[
c_p=\arg\min_c\lVert Rc-t\rVert_2,
\qquad
q=t-Rc_p.
\]

The saved certificate verifies $R^Tq\approx0$, full column rank, and a nonzero projection residual. Therefore $t\notin\operatorname{range}(R)$. The bounded feasible set is a subset of that range, so the target is also unreachable under $0\le c\le1$.

| model | points | controls | rank | max $\lvert R^Tq\rvert$ | unconstrained projection RMS | RMS / target RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | 161 | 9 | 9 | 4.01e-17 | 0.004784423 | 19.47% |
| 3D | 1681 | 25 | 25 | 3.81e-17 | 0.006030608 | 44.71% |

The 3D unconstrained projection uses some controls above one, so its bounded L2 optimum is slightly higher than the unconstrained projection floor. This does not weaken the span certificate.

## Losses, optimizer, and bump measurement

For residual $r=Rc-t$, the three target objectives are:

- L1: $\operatorname{mean}(|r|)$, reported as MAE.

- L2: $\sqrt{\operatorname{mean}(r^2)}$, reported as RMS.

- L-inf: $\max(|r|)$, reported as maximum error.

All six cases start from zero controls and use 500 projected-Adam updates with cosine learning-rate decay from 0.06 to 0.0003. Independent numerical global convex references use bounded least squares for L2 and HiGHS epigraph linear programs for L1 and L-inf.

Surface bumpiness is measured on induced displacement, not on the already curved rest belly. The primary measure is the RMS of

\[
u_{\mathrm{hp}}=u-G_\sigma*u,
\]

where the Gaussian filter width is 0.55 actuator spacings. Grid-scaled Laplacian RMS and high-pass peak-to-valley are also stored at every step. These measures depend on filter and grid scale, so comparisons are valid within each model, not numerically between 2D and 3D.

## Results

| model | optimized loss | best step | MAE | RMS | max error | gap to convex reference | bump RMS | bump / target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | L1 | 496 | 0.003551390 | 0.004951467 | 0.01349558 | 0.0005% | 0.004500814 | 5.52x |
| 2D | L2 | 346 | 0.003597730 | 0.004784423 | 0.01179782 | 0% | 0.004729850 | 5.80x |
| 2D | L-inf | 495 | 0.004073153 | 0.005066532 | 0.01062847 | 0.1223% | 0.005029049 | 6.17x |
| 3D | L1 | 498 | 0.003952071 | 0.006343669 | 0.02303299 | 0.0002% | 0.005060247 | 2.33x |
| 3D | L2 | 333 | 0.003993611 | 0.006323450 | 0.02297833 | 0% | 0.005161666 | 2.38x |
| 3D | L-inf | 334 | 0.004490613 | 0.006837862 | 0.02287898 | 0.0011% | 0.006082295 | 2.80x |

Each loss wins its own evaluation metric in both dimensions: L1 gives the smallest MAE, L2 the smallest RMS, and L-inf the smallest maximum error. In this particular construction, L-inf also produces the bumpiest best state. It repeatedly changes the active worst point and redistributes error spatially; this is a result for this controlled model, not a general theorem that L-inf must always be bumpier.

The histories show bumpiness rising from exactly zero during the first updates, overshooting, and then settling while the target error converges. The early frames at steps 0, 1, 2, and 5 are retained so that formation is visible rather than skipped by the regular ten-step sampling.

- 2D bump traces: [L1](../data/20-bumpy-inverse-belly-losses/2d/l1/bumpiness-evolution.png), [L2](../data/20-bumpy-inverse-belly-losses/2d/l2/bumpiness-evolution.png), [L-inf](../data/20-bumpy-inverse-belly-losses/2d/linf/bumpiness-evolution.png)

- 3D bump traces: [L1](../data/20-bumpy-inverse-belly-losses/3d/l1/bumpiness-evolution.png), [L2](../data/20-bumpy-inverse-belly-losses/3d/l2/bumpiness-evolution.png), [L-inf](../data/20-bumpy-inverse-belly-losses/3d/linf/bumpiness-evolution.png)

- Common error traces: [2D full](../data/20-bumpy-inverse-belly-losses/2d/metric-comparison.png), [2D first 100 steps](../data/20-bumpy-inverse-belly-losses/2d/metric-comparison-early.png), [3D full](../data/20-bumpy-inverse-belly-losses/3d/metric-comparison.png), [3D first 100 steps](../data/20-bumpy-inverse-belly-losses/3d/metric-comparison-early.png)

## Reusable outputs

| model/loss | best state | controls | bump trace | evolution | complete trace |
| --- | --- | --- | --- | --- | --- |
| 2D L1 | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l1/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l1/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l1/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/2d/l1/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/2d/l1/trace.csv) |
| 2D L2 | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l2/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l2/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/l2/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/2d/l2/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/2d/l2/trace.csv) |
| 2D L-inf | [PNG](../data/20-bumpy-inverse-belly-losses/2d/linf/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/linf/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/2d/linf/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/2d/linf/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/2d/linf/trace.csv) |
| 3D L1 | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l1/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l1/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l1/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/3d/l1/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/3d/l1/trace.csv) |
| 3D L2 | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l2/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l2/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/l2/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/3d/l2/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/3d/l2/trace.csv) |
| 3D L-inf | [PNG](../data/20-bumpy-inverse-belly-losses/3d/linf/best.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/linf/control-evolution.png) | [PNG](../data/20-bumpy-inverse-belly-losses/3d/linf/bumpiness-evolution.png) | [MP4](../data/20-bumpy-inverse-belly-losses/3d/linf/evolution.mp4) | [CSV](../data/20-bumpy-inverse-belly-losses/3d/linf/trace.csv) |

Machine-readable aggregate: [summary.json](../data/20-bumpy-inverse-belly-losses/summary.json) and [results.md](../data/20-bumpy-inverse-belly-losses/results.md). Each case also contains `history.npz` with all 501 control and surface states, exact-reference controls, error metrics, bump metrics, gradients, learning rates, and worst-point identities. The reproducible runner is [20-run-bumpy-inverse-belly-losses.py](../src/20-run-bumpy-inverse-belly-losses.py).

## Validation and provenance

- The aggregate reports `status: ok`, `complete: true`, 42/42 passed case checks, and the expected metric winner for every model and loss.

- All six CSV files contain 501 finite evaluations at consecutive steps 0--500. Their arrays and summaries agree with the NPZ files.

- All 22 PNGs decode successfully. All six evolution videos are H.264/yuv420p at 8 fps, with 55 selected frames and duration 6.875 seconds.

- The final Cherries data snapshot is byte-identical to the live result tree and is stored at `.cherries/runs/2026/08/22/unreachable-belly-losses/20-run-bumpy-inverse-belly-losses/2026-08-22T020606-Bumpy-inverse-belly-evolution-under-L1-L2-and-L-inf/`.

- Observed final runtime was 45.1 seconds on CPU with deterministic float64 Torch operations.

- Git base revision at execution: `77721e8569660c0f670ec060b747c3e30dff26de`. The new experiment is uncommitted; its exact source is archived in the Cherries snapshot.

- Runtime: Python 3.14.6, Cherries 3.0.2, NumPy 2.4.6, SciPy 1.17.1, Torch 2.12.0+cu130, Matplotlib 3.11.0.

The final run used Cherries in local debug mode because this shared checkout contains unrelated work and a normal Cherries profile may capture or commit it. No Comet run or Git commit was created. From this experiment directory, the exact command was:

```bash
DEBUG=1 MPLBACKEND=Agg \
CHERRIES_NAME='Bumpy inverse belly evolution under L1 L2 and L-inf' \
CHERRIES_TAGS='belly,bumpy,inverse-physics,unreachable-target,2d,3d,l1,l2,linf,optimization-evolution,final' \
/home/liblaf/Projects/liblaf/apple/.venv/bin/python \
src/20-run-bumpy-inverse-belly-losses.py
```

The runner refuses to overwrite a nonempty output directory.

## Scope and next step

This establishes bumpy evolution in a deliberately discretized linear influence-function inverse problem. It is not nonlinear material instability, a volumetric FEM simulation, or an anatomical prediction. The graph Laplacian is unscaled and the actuator responses are peak-normalized, so the stiffness and control values are dimensionless and not mesh-independent tissue parameters.

The next physical extension should keep this smooth-target/local-control protocol but replace the surface surrogate with a small active nonlinear solid: a quasi-2D strip embedded in 3D FEM and a shallow 3D slab. In that model, “unreachable” would need to be supported by persistent residual under the prescribed coarse control basis and bounds rather than by this linear orthogonal-projection certificate.
