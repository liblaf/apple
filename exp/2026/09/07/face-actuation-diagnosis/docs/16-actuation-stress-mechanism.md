# Active-strain prestress mechanism

This CPU-only diagnostic evaluates the constitutive law at a deliberately fixed
physical deformation `F = I`. It does not solve the face model. Its stresses
and energy densities are for the pure-muscle constituent in MPa, before each
tetrahedron's `MuscleFraction` multiplies the quadrature weight.

## Implemented law

`StableNeoHookeanActive` forms `G = F @ Ainv`, then evaluates

\[
\Psi(G) = \tfrac12\mu(\lVert G\rVert_F^2 - 3)

- \mu(J-1) + \tfrac12\lambda(J-1)^2,
\qquad J=\det G.
\]

The implementation is [the active-energy kernel](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py#L20-L32) and [the first-Piola kernel](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py#L36-L46). It gives

\[
P(F,A_{\rm inv}) =
\left[\mu G + \{-\mu+\lambda(J-1)\}\operatorname{cof}G\right]A_{\rm inv}^{T}.
\]

At `F = I`, write `B = Ainv` and `j = det(B)`. Since
`cof(B) B^T = j I`, this simplifies exactly to

\[
P(I,B) = \mu BB^T + j\{-\mu+\lambda(j-1)\}I.
\]

The matched FiberModes material has `E = 0.024 MPa` and `nu = 0.46`.
The stable-material convention uses `mu = 0.008219178082 MPa` and
`lambda_code = 0.102739726027 MPa`; see [material setup](../../face-activation-materials/src/face_physics.py#L113-L129) and [the muscle construction](../../face-activation-materials/src/face_physics.py#L245-L258).

## c50 Fiber activation

For a unit fiber aligned with the first coordinate, `a = ln(2)` and
`gamma = 0.5` give

\[
A_{\rm inv}=\operatorname{diag}(2,2^{-1/2},2^{-1/2}),
\quad \det A_{\rm inv}=1.
\]

At `F = I`, the principal first-Piola stresses are
`(+0.024657534, -0.004109589, -0.004109589) MPa`, with Frobenius norm
`0.025333208 MPa`. The natural active deformation is `Ainv^-1`, so the
fiber's stress-free state contracts to one half its original length. This
agrees with the implemented Fiber map in [activation_models.py](../../face-activation-materials/src/activation_models.py#L84-L98).

## Equal-offset Raw6 comparison

Raw6 directly applies `Ainv = I + H`; `H` is not a matrix logarithm and its
coordinates are unbounded. [The source](../../face-activation-materials/src/activation_models.py#L36-L98) defines this distinction. To make the comparison explicit, let

\[
r=\lVert A_{\rm inv}^{\rm Fiber}-I\rVert_F=1.082392200292,
\qquad s=\pm r/\sqrt3=\pm0.624919428208,
\]

and set the volumetric Raw6 offset to `H = s I`. Both Raw6 cases have the
same active-map offset norm as c50 Fiber. At fixed `F = I`:

| case | `detAinv` | principal `P` (MPa) | `||P||_F` (MPa) |
| --- | ---: | ---: | ---: |
| Fiber c50 | 1.000000 | `+.024658, -.004110, -.004110` | .025333 |
| Raw6 dilation | 4.290377 | `+1.436811, +1.436811, +1.436811` | 2.488629 |
| Raw6 compression | .052768 | `-.004413, -.004413, -.004413` | .007643 |

The asymmetry is a property of this energy at equal direct-map offset: Raw6
dilation incurs a large volumetric `J` term, whereas compression does not
produce an equal and opposite stress. It is not a Fiber sign reversal.

`detAinv` is the active-map parameter determinant. In all three rows the
physical geometry is fixed, so `detF = 1`. The energy instead sees
`detG = detF * detAinv`, yielding `1`, `4.290377`, and `.052768` above.
Conflating these quantities would incorrectly describe the Raw6 comparison
as a physical volume change at this fixed configuration.

## Reproduction

Run from this directory:

```bash
python src/16-actuation-stress-mechanism.py
```

The script writes [summary.json](../data/16-actuation-stress/summary.json),
including central finite-difference checks of every component of `dPsi/dF`.
