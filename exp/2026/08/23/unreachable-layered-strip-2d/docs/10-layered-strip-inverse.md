# Layered 2D unreachable-target inverse physics

## Outcome

This experiment reproduces the no-skin layered-block setup as a controlled 2D
plane-strain finite-element problem. A small muscle patch sits near the left end
of a stiff SMAS band, while every free point on the entire top asks for the same
upward displacement of `+0.1`. The inverse solve cannot produce that target. It
instead forms a localized mound above the muscle and, with independent
per-triangle activation, develops finer scallops on top of that mound.

The important result is that these are two different effects:

- The **large localized mound** comes from asking a small, edge-localized muscle
  to reproduce a uniform full-width target that lies outside its response span.

- The **fine scallops** are strongly amplified by unregularized per-cell
  controls, the nearly incompressible constant-strain triangles, and the mesh.
  They are not evidence of a nonlinear material instability.

All geometry and field images below were rendered by ParaView 6.1.1. FFmpeg was
used only to encode ParaView's PNG frames into H.264.

![Material and target setup](../data/21-paraview-step-by-step/setup-materials.png)

![Final unregularized inverse state](../data/21-paraview-step-by-step/evolution-step-by-step-final.png)

![Control-ablation comparison](../data/21-paraview-step-by-step/final-comparison.png)

The complete iteration-by-iteration evolution is in
[evolution-step-by-step.mp4](../data/21-paraview-step-by-step/evolution-step-by-step.mp4).
It contains one distinct frame for every inverse state from step 0 through 500.
The corresponding ParaView state is
[evolution-step-by-step.pvsm](../data/21-paraview-step-by-step/evolution-step-by-step.pvsm).
The original 15-checkpoint
[sampled preview](../data/20-paraview/evolution.mp4) is retained as a shorter
overview. The other ParaView states are
[final-comparison.pvsm](../data/21-paraview-step-by-step/final-comparison.pvsm),
and [setup-materials.pvsm](../data/21-paraview-step-by-step/setup-materials.pvsm).

## Controlled model

The strip occupies `[0, 1] x [0, 0.1]` and contains 2,000 alternating-diagonal
constant-strain triangles and 1,111 nodes. The bottom, left, and right boundaries
are fixed in both coordinates. The target contains the 99 non-corner top nodes
and assigns `(u_x, u_y) = (0, +0.1)` to each one.

There is no skin. The material layout and constants match the current June
layered-block experiment:

| region | bounds | Young's modulus (MPa) | Poisson ratio |
| --- | --- | ---: | ---: |
| fat | outside the middle band | 0.003 | 0.49 |
| SMAS | `0.04 <= y <= 0.06`, excluding muscle | 0.100 | 0.35 |
| muscle | `0.05 <= x <= 0.22`, `0.04 <= y <= 0.06` | 0.030 | 0.49 |

For engineering strain `[epsilon_xx, epsilon_yy, gamma_xy]`, the plane-strain
elasticity matrix is

```text
      [ lambda + 2 mu   lambda            0 ]
D  =  [ lambda          lambda + 2 mu     0 ]
      [ 0               0                mu ]
```

with the ordinary 3D Lame conversion. A muscle element has a symmetric active
strain control `[a_xx, a_yy, a_xy]`. It enters the linearized energy as

```text
Pi_e = A_e / 2 (B_e u_e + a_e)^T D_e (B_e u_e + a_e).
```

After assembly and elimination of fixed degrees of freedom,

```text
K_ff u_f = -C_f theta,
u_top = R theta,
R = P_top (-K_ff^-1 C_f).
```

The baseline has three controls on each of 68 muscle triangles, or 204 controls
in total, bounded to `[-1.5, +1.5]`. This bound matches the activation magnitude
reached by the corresponding 3D run; it is not a claim that such a strain is in
the linear regime. Projected Adam takes 500 updates with cosine learning-rate
decay. A separate bounded convex least-squares solution supplies the optimizer
reference.

This implementation is self-contained and uses the current Cherries/VTK/
ParaView experiment workflow. It does not import or copy the legacy April 2D
demo.

## Unreachable-target certificate

For target vector `t`, the unrestricted projection computes

```text
theta_p = argmin ||R theta - t||_2,
q = t - R theta_p.
```

The baseline response matrix is `198 x 204`, but has effective rank 49 and
effective nullity 155 at a relative singular-value threshold of `1e-10`. Its
retained condition number is `4.48e9`. The projection residual is 72.11% of the
target RMS, and the normalized orthogonality check is
`||R^T q|| / (||R|| ||q||) = 7.67e-11`. The nonzero orthogonal residual certifies
that the uniform target is outside this fixed linear response span even before
the activation bounds are applied.

The bounded optimum is farther away: its target-error RMS is 94.24% of target
RMS. At the unregularized Adam state, the mean top motion near the muscle is
`0.01797`, while the far-field mean is only `0.00180`, a factor of 9.99. This is
why the solve produces a left-side mound rather than uniform lift.

## Inverse evolution and control ablations

| case | best step | target-error RMS / target | top range | high-pass RMS | second-difference RMS | neighbor control-jump RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-cell tensor, no regularization | 500 | 94.25% | 0.03225 | 0.000921 | 0.002178 | 0.6575 |
| per-cell tensor + neighbor smoothing | 347 | 94.42% | 0.02730 | 0.000699 | 0.000621 | 0.0339 |
| one tensor shared by the muscle | 254 | 94.45% | 0.02832 | 0.000726 | 0.000639 | 0 |

Neighbor smoothing reduces the control-jump RMS by 94.8%, the mesh-scale
second-difference roughness by 71.5%, and high-pass roughness by 24.1%. The
target error rises by only 0.18 percentage points. Sharing one tensor over the
whole muscle gives a similar smooth surface. The broad mound remains in both
cases because neither change expands the muscle's spatial response span.

The unregularized Adam objective is only 0.0099% above the independent bounded
convex optimum, and the convex solution is itself bumpy (high-pass RMS
`0.000958`). Therefore the observed structure is not explained by Adam failing
to converge. It is a property of the bounded, underdetermined inverse problem
and its discretized controls.

The step-by-step time series contains all 501 consecutive states, `0, 1, 2, ...,
500`. The upper ParaView panel shows the evolving deformation and vertical
displacement. The lower panel shows the evolving activation norm on the same
deformed mesh. The magenta line is the unreachable target top. The movie runs at
10 fps, so each video frame advances the inverse solve by exactly one step.

## Material and discretization ablations

The following comparisons use the independent bounded convex solution, so they
do not mix optimizer differences into the result.

| ablation | target-error RMS / target | high-pass RMS | second-difference RMS | change in second-difference RMS |
| --- | ---: | ---: | ---: | ---: |
| baseline reference | 94.24% | 0.000958 | 0.002165 | -- |
| SMAS stiffness reduced to fat | 93.28% | 0.001390 | 0.003391 | +56.6% |
| fat and muscle `nu = 0.45` | 94.37% | 0.000825 | 0.000997 | -54.0% |
| half the x resolution | 94.18% | 0.002351 | 0.007942 | +266.8% |

These results refine the causal interpretation:

- The stiff SMAS is not the source of the bumps in this model. Removing its
  stiffness contrast makes both roughness measures larger. Here the stiff band
  helps transmit and spread the local actuation.

- Reducing near-incompressibility substantially reduces the mesh-scale
  oscillation. Linear triangles at `nu = 0.49` are susceptible to volumetric
  locking, so part of the scalloping is a discretization artifact.

- Halving the x resolution changes the per-element control basis and makes the
  roughness much larger. A feature this mesh-sensitive should not be interpreted
  as a mesh-independent physical wavelength.

The most defensible statement is therefore: **local actuation plus an
unreachable full-width target creates the large nonuniform mound; unregularized
element-wise controls, ill-conditioning, near-incompressible CST elements, and
mesh scale amplify the fine bumpy structure.**

## Scope and limitation

This is a small-strain linear plane-strain mechanism surrogate, not native Apple
nonlinear 2D solid FEM and not an anatomical prediction. The current Apple solid
path is tetrahedral, so an exact native 2D active-solid element is not available.
Using a one-cell-thick tetrahedral extrusion would introduce a different
front/back constraint problem rather than provide exact plane strain.

More importantly, the requested target equals the entire strip thickness and
the optimized activation reaches the 1.5 bound. The baseline maximum absolute
principal displacement strain is 3.61, maximum volumetric displacement strain
is 3.09, and maximum elastic principal strain is 2.10. All are far outside the
small-strain regime. Absolute shape and stress values must therefore not be used
quantitatively. This experiment identifies inverse/discretization mechanisms; it
does not establish nonlinear buckling or physical tissue instability.

A physically stronger follow-up should retain the same target, control, and
ablation protocol but use a stabilized mixed or higher-order near-incompressible
2D formulation, or add the missing tied-plane constraints to a shallow nonlinear
3D slab. A `+0.01` target should also be run as the small-strain scaling control.

## Reusable outputs and validation

- Numerical aggregate:
  [summary.json](../data/11-layered-strip-inverse-step-by-step/summary.json) and
  [results.md](../data/11-layered-strip-inverse-step-by-step/results.md).

- Full-step baseline VTK series:
  [history.vtu.series](../data/11-layered-strip-inverse-step-by-step/baseline-per-cell/history.vtu.series),
  complete controls in
  [history.npz](../data/11-layered-strip-inverse-step-by-step/baseline-per-cell/history.npz),
  and per-step metrics in
  [trace.csv](../data/11-layered-strip-inverse-step-by-step/baseline-per-cell/trace.csv).

- Step-by-step ParaView provenance:
  [render-receipt.json](../data/21-paraview-step-by-step/render-receipt.json).
  It records the pinned inputs, all 501 time values, ParaView version, output
  hashes, frame count, and ffprobe metadata.

Validation passed for all stiffness, projection, finiteness, and ablation
checks. Each of the three traces has 501 consecutive finite evaluations. The
equilibrium relative residuals are below `1e-15`. The full-step numerical export
contains 501 VTK states with exact time values `0` through `500`. ParaView
rendered 501 evolution frames at `1600 x 900`; the video is H.264/yuv420p at
10 fps with 501 frames and a duration of 50.1 seconds. The temporary PNG sequence
was removed only after ffprobe validated the encoded video; the VTK states,
initial/middle/final PNGs, and reloadable ParaView state remain. Numerical runtime
was 6.39 seconds on CPU, followed by 25.6 seconds for ParaView rendering and video
encoding.

The final commands were run from this experiment directory:

```bash
DEBUG=1 \
CHERRIES_NAME='Layered 2D inverse full step history' \
CHERRIES_TAGS='2d,plane-strain,fem,inverse-physics,unreachable-target,bumpy,optimization-evolution,every-step,paraview,final' \
/home/liblaf/Projects/liblaf/apple/.venv/bin/python \
src/10-run-layered-strip-inverse.py \
  --output-dir data/11-layered-strip-inverse-step-by-step \
  --full-step-history true

/usr/bin/pvpython src/20-render-layered-strip-paraview.py \
  --contract data/11-layered-strip-inverse-step-by-step/paraview-contract.json \
  --output-dir data/21-paraview-step-by-step \
  --fps 10 \
  --discard-frames
```

Both runners refuse to overwrite a nonempty output directory. Use a new output
directory for a rerun. The numerical source is
[10-run-layered-strip-inverse.py](../src/10-run-layered-strip-inverse.py), and the
ParaView source is
[20-render-layered-strip-paraview.py](../src/20-render-layered-strip-paraview.py).
