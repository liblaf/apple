# Material-physics meeting outline

## Asset rule

Every linked PNG is a standalone image for later manual layout in PowerPoint.
No contact sheet, montage, or multi-panel meeting image is included. All 3D
images were rendered natively in ParaView and have a matching `.pvsm` state.
The two cuboid summary plots are separate standalone Matplotlib images.

## 1. Main result: step-40 2x2 inverse ablation

### Question

Do target-guided heterogeneous skin stiffness and target-guided pre-strain
address different failure modes in the inverse solve?

### Controlled design

| factor | assignment |
| --- | --- |
| `H0` | Skin `E = 0.2 MPa` on every `IsFace` triangle. |
| `HF` | Skin `E = 0.003 MPa` where raw `R > 1`; `0.2 MPa` elsewhere. |
| `P0` | `rho = 1`; skin `ActivationInv = 0`. |
| `P1` | `rho = 0.98^2 clip(R, 0.5, 1)`; isotropic in-plane pre-strain. |

`HF` uses the fat modulus, not zero stiffness, on the target-expanding region.
That region contains `16,723 / 29,899` `IsFace` triangles and represents
`54.5531%` of `IsFace` rest area. No mean-modulus matching is applied. `P1`
stores pre-strain on every `IsFace` triangle, including triangles with little
or no target expansion; the `0.98^2` factor supplies a uniform 2% in-plane
length tightening.

The four cases are:

| case | skin modulus | pre-strain |
| --- | --- | --- |
| `H0P0` | homogeneous `0.2 MPa` | none |
| `H0P1` | homogeneous `0.2 MPa` | `c020` |
| `HFP0` | `0.003 / 0.2 MPa` target-guided field | none |
| `HFP1` | `0.003 / 0.2 MPa` target-guided field | `c020` |

Everything else is shared:

- Skin: 1 mm Koiter membrane, `nu = 0.49`, corrected plane-stress Lamé
  conversion, and fixed original `RestArea`.
- Fat: Stable Neo-Hookean, `E = 0.003 MPa`, `nu = 0.49`.
- Muscle: active Stable Neo-Hookean, `E = 0.03 MPa`, `nu = 0.49`.
- Aponeurosis: Stable Neo-Hookean, `E = 0.1 MPa`, `nu = 0.35`.
- Boundary: every vertex incident to an artificial cross-section is fixed to
  exact zero displacement.
- Inverse: fresh-zero activation, displacement, and optimizer; Adam
  `lr = 0.3`; 40 updates and 41 evaluations.

### Shared volume-material cross-section

This coronal cross-section is a categorical visualization of the dominant
constituent at each volume point. It is for explaining anatomy only. The
actual physics uses continuous fractions satisfying
`FatFraction + MuscleFraction + AponeurosisFraction = 1`:

- Fat: Stable Neo-Hookean, `E = 0.003 MPa`, `nu = 0.49`.
- Muscle: active Stable Neo-Hookean, `E = 0.03 MPa`, `nu = 0.49`.
- Aponeurosis: Stable Neo-Hookean, `E = 0.10 MPa`, `nu = 0.35`.

#### Coronal dominant constituent

![Coronal dominant-constituent volume cross-section](../data/25-volume-cross-section/25-volume-cross-section-dominant-material.png)

[ParaView state: coronal dominant constituent](../data/25-volume-cross-section/25-volume-cross-section-dominant-material.pvsm)

### Slide 1: show the two material factors

Use these four standalone ParaView images to explain the row and column factors:

#### H0 Young's modulus

![H0 Young's modulus](../data/20-ablation-assets/20-ablation-material-young-h0.png)

[ParaView state: H0 Young's modulus](../data/20-ablation-assets/20-ablation-material-young-h0.pvsm)

#### HF Young's modulus

![HF Young's modulus](../data/20-ablation-assets/20-ablation-material-young-hf.png)

[ParaView state: HF Young's modulus](../data/20-ablation-assets/20-ablation-material-young-hf.pvsm)

#### P0 pre-strain

![P0 pre-strain](../data/20-ablation-assets/20-ablation-material-prestrain-p0.png)

[ParaView state: P0 pre-strain](../data/20-ablation-assets/20-ablation-material-prestrain-p0.pvsm)

#### P1 pre-strain

![P1 pre-strain](../data/20-ablation-assets/20-ablation-material-prestrain-p1.png)

[ParaView state: P1 pre-strain](../data/20-ablation-assets/20-ablation-material-prestrain-p1.pvsm)

Suggested message: `HF` is an exaggerated, target-guided softening test, while
`P1` changes the stress-free area over the whole face.

### Slide 2: show the four step-40 geometries

#### H0P0 geometry

![H0P0 geometry](../data/20-ablation-assets/20-ablation-step40-h0p0-geometry.png)

[ParaView state: H0P0 geometry](../data/20-ablation-assets/20-ablation-step40-h0p0-geometry.pvsm)

#### H0P1 geometry

![H0P1 geometry](../data/20-ablation-assets/20-ablation-step40-h0p1-geometry.png)

[ParaView state: H0P1 geometry](../data/20-ablation-assets/20-ablation-step40-h0p1-geometry.pvsm)

#### HFP0 geometry

![HFP0 geometry](../data/20-ablation-assets/20-ablation-step40-hfp0-geometry.png)

[ParaView state: HFP0 geometry](../data/20-ablation-assets/20-ablation-step40-hfp0-geometry.pvsm)

#### HFP1 geometry

![HFP1 geometry](../data/20-ablation-assets/20-ablation-step40-hfp1-geometry.png)

[ParaView state: HFP1 geometry](../data/20-ablation-assets/20-ablation-step40-hfp1-geometry.pvsm)

Use the same camera across all four. Discuss fit and visible surface quality
separately; neither should be inferred from the other.

### Slide 3: show fitting-error distributions

These four ParaView images show the full point-to-point displacement-error
magnitude, `||u_solved - u_target||`. This is the per-point quantity aligned
with the inverse fitting objective; its squared mean over the 15,302
`SmileLossMask` points gives the reported target RMS. All four use the same
no-clipping linear range from `0` to `9.931346158 mm`:

#### H0P0 point error

![H0P0 point error](../data/20-ablation-assets/20-ablation-step40-h0p0-point-error.png)

[ParaView state: H0P0 point error](../data/20-ablation-assets/20-ablation-step40-h0p0-point-error.pvsm)

#### H0P1 point error

![H0P1 point error](../data/20-ablation-assets/20-ablation-step40-h0p1-point-error.png)

[ParaView state: H0P1 point error](../data/20-ablation-assets/20-ablation-step40-h0p1-point-error.pvsm)

#### HFP0 point error

![HFP0 point error](../data/20-ablation-assets/20-ablation-step40-hfp0-point-error.png)

[ParaView state: HFP0 point error](../data/20-ablation-assets/20-ablation-step40-hfp0-point-error.pvsm)

#### HFP1 point error

![HFP1 point error](../data/20-ablation-assets/20-ablation-step40-hfp1-point-error.png)

[ParaView state: HFP1 point error](../data/20-ablation-assets/20-ablation-step40-hfp1-point-error.pvsm)

### Optional slide 3b: inspect the normal residual for bumpiness

The signed normal residual is the displacement error projected onto the target
normal, `(u_solved - u_target) dot n_target`. It distinguishes inward from
outward surface error and is the scalar field whose graph Laplacian defines
`L`. It is therefore useful for diagnosing bumpiness, but it omits tangential
error and is not the full fitting-error magnitude. The four optional maps use
one shared `+/-3.8157 mm` visualization range:

#### H0P0 normal residual

![H0P0 normal residual](../data/20-ablation-assets/20-ablation-step40-h0p0-normal-residual.png)

[ParaView state: H0P0 normal residual](../data/20-ablation-assets/20-ablation-step40-h0p0-normal-residual.pvsm)

#### H0P1 normal residual

![H0P1 normal residual](../data/20-ablation-assets/20-ablation-step40-h0p1-normal-residual.png)

[ParaView state: H0P1 normal residual](../data/20-ablation-assets/20-ablation-step40-h0p1-normal-residual.pvsm)

#### HFP0 normal residual

![HFP0 normal residual](../data/20-ablation-assets/20-ablation-step40-hfp0-normal-residual.png)

[ParaView state: HFP0 normal residual](../data/20-ablation-assets/20-ablation-step40-hfp0-normal-residual.pvsm)

#### HFP1 normal residual

![HFP1 normal residual](../data/20-ablation-assets/20-ablation-step40-hfp1-normal-residual.png)

[ParaView state: HFP1 normal residual](../data/20-ablation-assets/20-ablation-step40-hfp1-normal-residual.pvsm)

### Slide 4: report the exact step-40 metrics

| case | fit (mm) | D (deg) | L (mm) | area RMS | act. RMS | folds | inverted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `H0P0` | 2.720948 | 13.327645 | 0.217061 | 0.140860 | 0.0609580 | 25 | 47 |
| `H0P1` | 2.849029 | 5.579129 | 0.181487 | 0.072789 | 0.0548136 | 11 | 31 |
| `HFP0` | 1.481081 | 15.486252 | 0.190230 | 0.153874 | 0.0675393 | 28 | 54 |
| `HFP1` | 1.441860 | 7.564420 | 0.139811 | 0.094292 | 0.0628452 | 10 | 58 |

Metric definitions:

- Target RMS: pointwise displacement RMS on 15,302 `SmileLossMask` points.
- `D`: target-relative dihedral RMS in the strict target-contraction region.
- `L`: graph-Laplacian RMS of target-normal displacement residual on all
  `IsFace` vertices.
- Area RMS: rest-area-weighted RMS of `deformed area / target area - 1`.
- Activation RMS: RMS of the six symmetric `ActivationInv` components on
  active-muscle tetrahedra.
- Folds and inverted tetrahedra are validity diagnostics, not fit metrics.

### Balanced interpretation

- At either pre-strain level, `HF` cuts target RMS by `45.6-49.4%` and lowers
  `L` by `12.4-23.0%`. It simultaneously raises `D` by `16.2-35.6%`, raises
  area RMS by `9.2-29.5%`, and increases inverted tetrahedra. Selective
  softening therefore improves fit, but is not uniformly better.
- At either modulus level, `P1` cuts `D` by `51.2-58.1%`, `L` by
  `16.4-26.5%`, area RMS by `38.7-48.3%`, and folds by 14-18. Its target RMS
  changes only slightly: `+4.7%` for `H0` and `-2.6%` for `HF`.
- Relative to `H0P0`, `HFP1` reduces target RMS by `47.0%`, `D` by `43.2%`,
  `L` by `35.6%`, area RMS by `33.1%`, and folds from 25 to 10. However,
  inverted tetrahedra rise from 47 to 58, so this is the best observed
  fit/roughness trade-off at step 40, not a converged or fully valid solution.
- This is a one-subject, target-derived mechanistic ablation. The material map
  uses the target geometry and is not a learned, measured, or transferable
  anatomical Young's-modulus field. The cases are intentionally not
  mean-modulus matched.

Do not include the earlier `E = 0` or `gain-4/gain-8` experiments in the main
2x2 result; they answer different questions.

## 2. Lamé conversion correction

### Slide 5: explain the correction

The old setup passed 3D Lamé parameters into a 2D membrane. For
`E = 0.2 MPa`, `nu = 0.49`, the old `lambda = 3.288590604 MPa`; the corrected
plane-stress value is `0.128964337 MPa`. `mu = 0.067114094 MPa` is unchanged.
Thus the old `lambda` was `25.5x` too large and the in-plane area coefficient
`lambda + mu` was `17.114x` too large.

### Slide 6: show the conversion-only forward replay

#### Old 3D-Lamé geometry

![Old 3D-Lamé geometry](../data/10-lame-assets/10-lame-old-3d-geometry.png)

[ParaView state: old 3D-Lamé geometry](../data/10-lame-assets/10-lame-old-3d-geometry.pvsm)

#### Corrected plane-stress geometry

![Corrected plane-stress geometry](../data/10-lame-assets/10-lame-corrected-plane-stress-geometry.png)

[ParaView state: corrected plane-stress geometry](../data/10-lame-assets/10-lame-corrected-plane-stress-geometry.pvsm)

#### Old 3D-Lamé area strain

![Old 3D-Lamé area strain](../data/10-lame-assets/10-lame-old-3d-area-strain.png)

[ParaView state: old 3D-Lamé area strain](../data/10-lame-assets/10-lame-old-3d-area-strain.pvsm)

#### Corrected plane-stress area strain

![Corrected plane-stress area strain](../data/10-lame-assets/10-lame-corrected-plane-stress-area-strain.png)

[ParaView state: corrected plane-stress area strain](../data/10-lame-assets/10-lame-corrected-plane-stress-area-strain.pvsm)

Under the same pinned activation, RMS area strain changes from `0.4146%` to
`2.7844%`, a `6.72x` increase. This shows that the corrected membrane can
expand instead of behaving nearly area-incompressibly.

Keep the claim narrow: this is a fixed-activation, conversion-only forward
replay, not an inverse comparison. It demonstrates removal of artificial area
locking, but does not by itself improve target fit or remove bumpiness.

## 3. Cuboid fat-thickness probe

### Slide 7: show the controlled setup and 3D response

The block has a `1 x 1` footprint, bottom fat `0.04`, SMAS `0.02`, and top-fat
thickness `0.04 / 0.08 / 0.12`. At bottom pressure `0.60`, show the three
standalone ParaView renders with their identical camera, warp, and color range:

- [Top fat 0.04](../data/30-cuboid-fat-thickness/paraview/top-fat-0p04/30-cuboid-top-fat-0p04-paraview.png)
  ([state](../data/30-cuboid-fat-thickness/paraview/top-fat-0p04/30-cuboid-top-fat-0p04-paraview.pvsm))
- [Top fat 0.08](../data/30-cuboid-fat-thickness/paraview/top-fat-0p08/30-cuboid-top-fat-0p08-paraview.png)
  ([state](../data/30-cuboid-fat-thickness/paraview/top-fat-0p08/30-cuboid-top-fat-0p08-paraview.pvsm))
- [Top fat 0.12](../data/30-cuboid-fat-thickness/paraview/top-fat-0p12/30-cuboid-top-fat-0p12-paraview.png)
  ([state](../data/30-cuboid-fat-thickness/paraview/top-fat-0p12/30-cuboid-top-fat-0p12-paraview.pvsm))

### Slide 8: show the two surface-variation summaries

- [Top-surface p95-p05](../data/30-cuboid-fat-thickness/metrics/30-cuboid-top-surface-p95-p05.png)
- [Top-surface Laplacian RMS](../data/30-cuboid-fat-thickness/metrics/30-cuboid-top-surface-laplacian-rms.png)

From top-fat thickness `0.04` to `0.12`, p95-p05 decreases from `0.04359485`
to `0.03126095` (`28.3%`) and Laplacian RMS decreases from `8.561539` to
`4.813979` (`43.8%`). Maximum displacement changes by only `4.7%`.

Meeting-safe conclusion: thicker fat reduces absolute surface variation in
this controlled toy. Do not generalize it to anatomical faces or
scale-invariant smoothing; cases were independently remeshed, and normalized
Laplacian was non-monotone.

## Closing message

1. Correct plane-stress Lamé conversion removes a constitutive source of
   artificial skin rigidity.
2. In the step-40 ablation, heterogeneous softening mainly improves fit, while
   pre-strain mainly improves surface regularity; their combination gives the
   best measured balance but still contains inverted tetrahedra.
3. The cuboid probe supports fat thickness as a plausible smoothing mechanism,
   but only in a controlled toy setting.

The next inverse work should prioritize eliminating element inversions and
checking whether the same factor separation survives a target-independent
material field, additional subjects, and longer or better-converged solves.
