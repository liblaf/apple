# Sharp spines in the unrestricted active-stress fit

The frozen step-49 surface has real pointed protrusions beside the nose and around the lips. They remain visible with identical cameras and flat shading in the [comparison image](../data/44-live-shape-review-001/comparison.png). The earlier comparator is the corrected physical-volume, free active-strain result at update 200 from the five-mesh package. Surface targets agree to 1e-12 m after matching GlobalPointId. No geometry smoothing, registration, or new mechanics solve was applied.

The strongest identified mechanism is a change in admissible controls: arbitrary signed Q can make the local material's isochoric stiffness negative. The earlier real active-strain matrix could soften that stiffness toward zero, but could not make it negative by the activation term. This is a concrete constitutive problem with this unrestricted quadratic stress law, not evidence that active-stress formulations generally produce worse shapes.

## Saved observations

| Quantity | Earlier active strain, update 200 | Current signed stress, update 49 |
| --- | ---: | ---: |
| Position RMS using the current surface-area weights | 1.8357 mm | 4.5204 mm |
| RMS local neighbor residual of target-position error | 0.1089 mm | 0.2235 mm |
| Maximum local neighbor residual | 2.0374 mm | 11.2247 mm |
| Inverted tetrahedra | 1 | 692 |

The local neighbor residual is `(fit-target)_i - mean_neighbors(fit-target)`, using unique surface edges. It is a descriptive local roughness measure, not a curvature or convergence test. The old report's 1.763 mm figure uses an unweighted objective/domain; the table recomputes both surfaces with the same current weights.

Within the current run, position RMS fell from 5.0959 to 4.5204 mm, while normal-angle RMS rose from 9.07 to 13.12 degrees, minimum physical J fell to -2.2498, and the minimum triangle-area ratio reached 0.2931. The first recorded inverted cells appeared at step 42. The current objective includes neither normal loss nor active-stress smoothness. The earlier free active-strain comparator also had no skin and no smoothness, so their absence alone does not explain the difference.

## The missing lower bound

For the implemented corrected active-strain model, with B = A inverse,

$$
W_B(F)=\frac{\mu}{2}(\|FB\|_F^2-3)+g(\det F).
$$

The current stress model is

$$
W_Q(F)=\frac{\mu}{2}(\|F\|_F^2-3)+g(\det F)
 +\frac12 Q:(F^T F-I).
$$

Holding passive material parameters fixed, setting

$$
Q=\mu(BB^T-I)
$$

makes their forces and displacement tangents identical; their energy difference is independent of F. Consequently, every real B satisfies `Q + mu I >= 0` in the positive-semidefinite sense. If invertibility is required, the inequality is strict. Six unrestricted coordinates in B do not imply six unbounded signed coordinates in Q are mechanically equivalent.

At step 49, muscle mu is 4.02685 kPa, whereas the minimum Q eigenvalue is -23.0560 kPa. There are **111,912 / 288,235 active cells (38.83%)** below the strain-equivalent lower bound `lambda_min(Q) >= -mu`.

Accounting for the summed muscle, fat, and aponeurosis potentials, the smallest isochoric coefficient is

$$
c_i=f_{m,i}(\mu_m+\lambda_{\min}(Q_i))
     +f_{f,i}\mu_f+f_{a,i}\mu_a.
$$

It is negative in **77,621 active cells (26.93%)**, with minimum **-19.0292 kPa**. For a rank-one perturbation `H=a n^T`, where n is the corresponding eigenvector and a is perpendicular to `cof(F)n`, the determinant remains constant along that perturbation and the local second derivative is `c_i ||a||^2 < 0`. Thus the volumetric penalty cannot remove every such negative direction. This is a local loss of material ellipticity and a credible source of localized deformation. It is not a computation of the minimum eigenvalue of the assembled, boundary-constrained Hessian, nor a causal allocation of every observed spike.

Source evidence: [current energy](../src/stress_material.py), [mixture/material assembly](../src/stress_physics.py), and [earlier corrected energy](../../../14/dominant-activation-ablation/src/volume_preserving_active.py). The [CPU stability receipt](../data/44-live-shape-review-001/stability.json) records the checkpoint/fixture hashes and exact calculation.

This equivalence is specific to these polynomial energies and common passive parameters. It is not a general equivalence theorem for active strain and active stress; broader constitutive choices can differ even when uniaxial responses match ([Giantesio, Musesti and Riccobelli](https://arxiv.org/abs/1709.04977)).

## Solver and comparison limits

All 49 post-initial evaluations failed the forward convergence threshold while remaining finite and therefore accepted under the requested Adam policy. At step 49, the force residual is 2.0699e-5 against a 1e-10 threshold; the recomputed adjoint relative residual is 1.0443 against 1e-7. The inner forward energy keeps decreasing while the force residual grows, which is compatible with descending an unstable local mode. These are substantially inexact gradients; occasional noisy updates alone are not a complete explanation for this trajectory.

The comparison also changes passive moduli, the Lamé convention, objective normalization, solver, and iteration budget. The current muscle/fat/aponeurosis Young's moduli are 0.012/0.0112/1.693 MPa, versus 0.03/0.003/0.1 MPa in the earlier baseline. Therefore the result is not a controlled active-strain versus active-stress experiment.

A useful next diagnostic is a matched six-coordinate stress ablation with the strain-equivalent lower spectral bound, keeping materials, objective, initialization, and solver settings fixed. A small positive stiffness margin avoids the degenerate boundary. This still allows expansion; the later contraction-only condition Q >= 0 is stronger. Restoring smoothness can suppress spatial variation but does not guarantee the local bound. No bound, regularizer, inversion gate, or solver change was applied during this review; the authorized Adam fit remains running.

## Reproduction and verification

From `exp/2026/09/21/stress-activation-loss`:

```sh
LIBGL_ALWAYS_SOFTWARE=1 CUDA_VISIBLE_DEVICES='' \
CHERRIES_NAME='Smile live spine diagnosis' \
CHERRIES_TAGS='smile,inverse,diagnosis,render,cpu' \
.venv/bin/python src/44-review-live-shape.py
```

The [Cherries/Comet run](https://www.comet.com/liblaf/apple/662eb912344849e7bb0f9c75a9106867) completed successfully. Its [summary](../data/44-live-shape-review-001/summary.json) records frozen checkpoint step 49, SHA256 `228cb0f9f33775aa7d47d21c6173aeb252eb5e1ff6440ab672b4a10555e2ed11`, metrics, target correspondence, and source paths. [Surface arrays](../data/44-live-shape-review-001/surfaces.npz) preserve the rendered geometry. The six-panel image was visually inspected. The browser automation timed out; this review used saved geometry rendered with software OpenGL instead. Ruff passed. The fit retained PID 3160064, learning rate 0.05 and zero smoothness.

The renderer reads the live checkpoint once into bytes; rerunning it later reviews a newer checkpoint and requires a new output path. The stored surfaces retain this reviewed step. The stability receipt includes the source formula and hashes; reproducing its exact numbers requires the same checkpoint, since the live `last.npz` can advance.
