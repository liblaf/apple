# Unreachable belly fitting under L1, L2, and L-inf losses

## Outcome

A deterministic reduced-order experiment was built and run for both a 2D belly profile and a 3D belly height surface. Each model fits the same deliberately unreachable target three times, using L1, L2, or exact L-inf as the optimization objective. Every run stores all 601 evaluations, a best-state figure, an actuator trace, and a 31/32-frame evolution video.

The experiment is intentionally a controlled surface-height surrogate. It isolates the effect of the target loss without the cost and extra failure modes of a nonlinear volumetric FEM solve.

## Model

The rim is fixed and only interior surface heights respond. Smooth Gaussian actuator loads are passed through the same linear equilibrium operator,

\[
K = k_0 I + k_m L + k_b L^2,
\qquad
K u_j = f_j,
\qquad
u(c) = A c = \sum_j c_j u_j,
\]

with \(k_0=1\), \(k_m=0.9\), \(k_b=0.15\), normalized actuator response amplitude 0.055, and bounded controls \(0\le c_j\le1\). The 2D profile uses 101 points and three actuators. The 3D surface uses a \(31\times31\) grid and a \(3\times3\) actuator array.

The rest height is a smooth cosine-squared mound. The target is

\[
t = A c_{\mathrm{teacher}} + a q,
\qquad
q = \frac{(I-AA^+)s}{\|(I-AA^+)s\|_\infty},
\]

where \(s\) is a narrow central depression seed. Because \(A^Tq\approx0\) and \(a>0\), the added component is outside the actuator-response space. Therefore the target cannot be reached by any controls, even before applying the box bounds.

| model | points | controls | unreachable amplitude |   max \( |        A^Tq |      \) | unconstrained projection RMS floor | floor / target RMS |
| ----- | -----: | -------: | --------------------: | -------: | ----------: | ------: | ---------------------------------- | ------------------ |
| 2D    |    101 |        3 |                 0.035 | 5.41e-16 |  0.01360269 | 39.801% |                                    |                    |
| 3D    |    961 |        9 |                 0.030 | 1.14e-15 | 0.003748060 | 10.022% |                                    |                    |

## Losses and optimization

For the scalar surface-height residual \(r=Ac-t\), the three objectives are:

- L1: \(\operatorname{mean}(|r|)\), reported as MAE.
- L2: \(\sqrt{\operatorname{mean}(r^2)}\), reported as RMS. This differs from the unnormalized vector 2-norm only by a fixed positive scale.
- L-inf: \(\max(|r|)\), reported as maximum error.

All cases start from controls 0.05 and use 600 projected-Adam updates with cosine learning-rate decay from 0.08 to 0.0005. For an independent reference, bounded L2 is solved with `scipy.optimize.lsq_linear`; bounded L1 and L-inf are solved as linear programs with HiGHS.

## Results

| model | optimized loss | best step | objective | MAE | RMS | max | exact optimum | relative gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | L1 | 598 | 0.009499987 | 0.009499987 | 0.01544650 | 0.04986591 | 0.009499973 | 0.000152% |
| 2D | L2 | 348 | 0.01360269 | 0.01075877 | 0.01360269 | 0.03500000 | 0.01360269 | 0% |
| 2D | L-inf | 535 | 0.02602499 | 0.01331447 | 0.01490001 | 0.02602499 | 0.02602461 | 0.001486% |
| 3D | L1 | 596 | 0.001378069 | 0.001378069 | 0.005182467 | 0.04453515 | 0.001378055 | 0.000971% |
| 3D | L2 | 540 | 0.003748060 | 0.002092019 | 0.003748060 | 0.03000000 | 0.003748060 | 0% |
| 3D | L-inf | 544 | 0.01638364 | 0.006585283 | 0.007903019 | 0.01638364 | 0.01638071 | 0.017858% |

Each loss wins its corresponding evaluation metric in both dimensions:

- L1 minimizes the average absolute error, while accepting a larger localized error near the target's central feature.
- L2 balances squared error over the full surface. Here it recovers the feasible teacher controls to numerical precision; the remaining residual is exactly the orthogonal unreachable component.
- L-inf reduces the worst error by redistributing it across the surface, at the expense of MAE and RMS. It reaches one control bound in 2D and three in 3D.

Exact L-inf is nonsmooth: only an active worst point supplies the selected subgradient. The worst point changes 216 times in 2D and 337 times in 3D, explaining the visibly noisier trajectory and its small remaining gap to the LP optimum. The late recorded best steps reflect tiny numerical fluctuations; the early-step plots show that most material improvement occurs in roughly the first 100 updates.

Raw 2D and 3D error magnitudes should not be compared directly because their grids, actuator sets, target shapes, and unreachable amplitudes differ.

## Reusable outputs

The metric figures use identical scales within each dimension. The best-state figures and videos show target/current geometry plus a signed residual curve or map with a shared symmetric scale across losses.

- 2D metric traces: [full 600 steps](../data/10-unreachable-belly-losses/2d/metric-comparison.png), [first 100 steps](../data/10-unreachable-belly-losses/2d/metric-comparison-early.png)
- 3D metric traces: [full 600 steps](../data/10-unreachable-belly-losses/3d/metric-comparison.png), [first 100 steps](../data/10-unreachable-belly-losses/3d/metric-comparison-early.png)
- Machine-readable aggregate: [summary.json](../data/10-unreachable-belly-losses/summary.json), [results.md](../data/10-unreachable-belly-losses/results.md)

| model/loss | best state | controls | evolution | complete trace |
| --- | --- | --- | --- | --- |
| 2D L1 | [PNG](../data/10-unreachable-belly-losses/2d/l1/best.png) | [PNG](../data/10-unreachable-belly-losses/2d/l1/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/2d/l1/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/2d/l1/trace.csv) |
| 2D L2 | [PNG](../data/10-unreachable-belly-losses/2d/l2/best.png) | [PNG](../data/10-unreachable-belly-losses/2d/l2/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/2d/l2/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/2d/l2/trace.csv) |
| 2D L-inf | [PNG](../data/10-unreachable-belly-losses/2d/linf/best.png) | [PNG](../data/10-unreachable-belly-losses/2d/linf/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/2d/linf/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/2d/linf/trace.csv) |
| 3D L1 | [PNG](../data/10-unreachable-belly-losses/3d/l1/best.png) | [PNG](../data/10-unreachable-belly-losses/3d/l1/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/3d/l1/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/3d/l1/trace.csv) |
| 3D L2 | [PNG](../data/10-unreachable-belly-losses/3d/l2/best.png) | [PNG](../data/10-unreachable-belly-losses/3d/l2/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/3d/l2/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/3d/l2/trace.csv) |
| 3D L-inf | [PNG](../data/10-unreachable-belly-losses/3d/linf/best.png) | [PNG](../data/10-unreachable-belly-losses/3d/linf/control-evolution.png) | [MP4](../data/10-unreachable-belly-losses/3d/linf/evolution.mp4) | [CSV](../data/10-unreachable-belly-losses/3d/linf/trace.csv) |

Each case also contains `history.npz`, its exact reference controls, every surface state, gradients, learning rates, and worst-point identities. Per-model `model.npz` files store the response matrix and target construction.

## Validation and provenance

- The aggregate reports `status: ok`, `complete: true`, and 30/30 passed case checks.
- All six CSV files contain finite, consecutive steps 0--600 and agree with the JSON/NPZ records.
- All 16 best/control/comparison PNGs decode successfully.
- All six MP4s decode as H.264/yuv420p at 8 fps. Five contain 32 selected frames; 3D L2 contains 31 because its best step 540 was already a regular 20-step snapshot.
- The final Cherries snapshot is byte-identical to the live data tree and is stored at `.cherries/runs/2026/08/22/unreachable-belly-losses/10-run-unreachable-belly-losses/2026-08-22T012826-Unreachable-belly-optimization-under-L1-L2-and-L-inf/`.
- Final runtime was 28.313 seconds on CPU with deterministic float64 Torch operations.
- Git revision at execution: `77721e8569660c0f670ec060b747c3e30dff26de`.
- Runtime: Python 3.14.6, Cherries 3.0.2, NumPy 2.4.6, SciPy 1.17.1, Torch 2.12.0+cu130, Matplotlib 3.11.0.

The final run used Cherries in local debug mode because the shared checkout has unrelated work and normal Cherries execution may capture or commit it. Consequently there is no Comet URL and no Git commit was created. `uv run --frozen` was not used because the editable build stalled in `hatch-vcs` while querying this unusually large checkout; the already-synchronized project virtual environment was used directly.

From this experiment directory, the exact final command was:

```bash
DEBUG=1 MPLBACKEND=Agg \
CHERRIES_NAME='Unreachable belly optimization under L1 L2 and L-inf' \
CHERRIES_TAGS='unreachable-target,belly,2d,3d,l1,l2,linf,optimization-evolution,final' \
/home/liblaf/Projects/liblaf/apple/.venv/bin/python \
src/10-run-unreachable-belly-losses.py
```

The runner refuses to overwrite a nonempty output directory.

## Scope and limitations

This is not a volumetric FEM model, a nonlinear material simulation, or a calibrated anatomical abdomen. It models one scalar surface-height degree of freedom per grid point, has no skin, contact, gravity, volume preservation, or tissue parameters, and imposes unreachability through a deliberately low-rank actuator-response space. The results establish the loss-function behavior for this controlled problem only; they do not establish anatomical target reachability or predict a human belly.

The next justified extension would be to reuse the same target/loss protocol on a small nonlinear solid-FEM block after deciding whether exact L-inf's nonsmooth, winner-switching behavior is acceptable or should be replaced by a smooth maximum surrogate.
