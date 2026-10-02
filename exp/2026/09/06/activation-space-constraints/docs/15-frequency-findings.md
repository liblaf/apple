# Forward frequency probe: activation roughness reaches the surface

## Purpose

This experiment isolates the forward mechanism behind the proposed activation
smoothness prior. It asks whether two fiber-contraction fields with the same
volume-weighted mean and RMS but different spatial frequencies produce different
surface roughness in a real tetrahedral nonlinear FEM solve. It does not perform
inverse optimization.

The fixture is the production `StableNeoHookeanActive` muscle--fat block with
`G = F A_inv`: a `1 x 0.1 x 1` domain, `48 x 10 x 48` structured cells,
138,240 tetrahedra, and 27,648 active muscle tetrahedra. The bottom is fixed;
the sides and top are free. The muscle occupies `0.04 <= y <= 0.06`, leaving a
0.04-thick fat layer above it. The reference fiber is `(1, 0, 0)` and activation
is the isochoric map

$$
A^{-1}=e^a ff^T + e^{-a/2}(I-ff^T).
$$

All cases have mean log contraction `a0 = 0.15`. The modulations are sampled
cosine products with wave numbers 1 and 4, normalized on the actual tetrahedra
to zero volume-weighted mean and unit RMS, at amplitudes `b = 0.015` and `0.030`.
No field is clipped.

## Results

The high-frequency activation produced less total top displacement but more
short-scale surface content. At the declared high-pass smoothing length
`ell = 0.06 L`, `k = 4` increased top high-pass RMS by 70.2% at `b = 0.015`
and 70.1% at `b = 0.030` relative to `k = 1`. Its top Laplacian RMS was 166%
higher at both amplitudes. At the same time, its total top response RMS was
72.5% lower. Thus the result is a smaller-amplitude, shorter-wavelength surface
response rather than a larger displacement response overall.

| Case | Top response RMS | Top HP RMS, `.03 L` | Top HP RMS, `.06 L` | Top HP RMS, `.12 L` | Top Laplacian RMS | Interface HP RMS, `.06 L` | Min `det(F)` | Reset/continuation over signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k1, b=.015` | 3.19368e-4 | 1.53884e-5 | 4.55831e-5 | 1.33862e-4 | 3.97905e-2 | 2.93457e-5 | .963759 | 9.85e-6 |
| `k4, b=.015` | 8.77738e-5 | 3.72528e-5 | 7.75890e-5 | 8.77048e-5 | 1.05956e-1 | 7.09383e-5 | .966620 | 2.94e-5 |
| `k1, b=.030` | 6.38543e-4 | 3.08561e-5 | 9.12504e-5 | 2.67742e-4 | 7.96490e-2 | 5.86878e-5 | .959637 | 4.52e-6 |
| `k4, b=.030` | 1.75598e-4 | 7.45255e-5 | 1.55209e-4 | 1.75443e-4 | 2.12002e-1 | 1.41916e-4 | .963991 | 1.70e-5 |

Doubling `b` doubled the top `.06 L` high-pass response by factors 2.0018
(`k = 1`) and 2.0004 (`k = 4`). This near-proportional response is useful for
the later inverse ablation: the observed frequency effect is not a large-strain
failure at these amplitudes.

The interpretation depends on the physical cutoff. `k = 4` exceeds `k = 1` at
`.03 L` and `.06 L`, but is 34.5% lower at `.12 L`, where the broad `k = 1`
mode itself contributes strongly to the high-pass residual. The inverse study
must therefore predeclare a physical smoothing length and retain the planned
cutoff sensitivity rather than selecting the most favorable one afterward.

All nine equilibrium solves succeeded. Across the five accepted states,
`min det(F) >= 0.959637`, `min det(A_inv)` was one to floating-point accuracy,
and `min det(G) >= 0.959637`. Resetting each modulated solve to zero displacement
instead of continuing from the uniform equilibrium changed the top response by
at most `2.94e-5` of its signal RMS. This probe is consequently not explained by
inversion or an observed branch switch.

## Outputs

- `data/10-frequency/summary.json`: complete metrics, forward receipts, mesh
  hashes, and analysis definition.
- `data/10-frequency/frequency-response.png`: matched top/interface high-pass
  comparison.
- `data/10-frequency/top-highpass-response-maps.png`: fixed-scale response maps.
- `data/10-frequency/<case>/fields.npz`: exact source and response fields.
- `data/10-frequency/<case>/continuation.vtu`: accepted deformed tetrahedral
  state with activation and determinant fields.

## Reproduction

Run from `exp/2026/09/06/activation-space-constraints`:

```bash
CHERRIES_NAME='Activation frequency transfer in tetrahedral muscle-fat block' \
CHERRIES_TAGS='activation-space,fiber-contraction,forward-probe,tetra-fem,bumpiness' \
uv run python src/10-forward-frequency-probe.py
```

The recorded run used repository revision
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, Comet experiment
`810f208f5e434783aa79c766b65364c4`, and a Cherries profile that enabled Comet
and local provenance while setting Git `commit=False`. Source SHA-256 values at
execution were `9d8d3001...e28d35` for the runner,
`996bf2ed...34f73` for `block_physics.py`, and `4c7ae312...2ddda1` for
`activation_models.py`.

The physics and local artifact write finished at 24.6 seconds. Comet shutdown
then spent five minutes on automatic environment and Git-patch collection,
reported that those details were incomplete, and ended with `Failed to log run
in comet.com`. The URL and scalar summary were emitted, but the remote record
must therefore be treated as incomplete. The local summary, meshes, fields,
figures, and Cherries log are the run evidence.

## Limits

This is one structured mesh, one constant fiber direction, one fat thickness,
and one passive material setting. It establishes that activation frequency can
change surface roughness under controlled source magnitude; it does not show
that an inverse regularizer improves target reconstruction, establish a
physiological face fiber field, or attribute the historical face artifact to
activation roughness alone. The top-normal direction is world `y`, and the
Gaussian high-pass uses reflected boundaries on this rectangular fixture. The
anatomical study needs a surface operator defined in reference geometry,
explicit region masks, contact/intersection checks, and fiber-field uncertainty.
