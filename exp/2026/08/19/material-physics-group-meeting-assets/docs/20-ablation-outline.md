# Corrected 2x2 inverse ablation: meeting outline

## Question and controlled design

Use one fixed inverse setup and cross two factors:

| case | IsFace skin Young's modulus | IsFace skin prestrain |
| --- | --- | --- |
| H0P0 | 0.2 MPa everywhere | off: rho=1 and ActivationInv=0 |
| H0P1 | 0.2 MPa everywhere | c020 |
| HFP0 | 0.003 MPa where raw TargetArea/RestArea>1; 0.2 MPa otherwise | off |
| HFP1 | 0.003 MPa where raw TargetArea/RestArea>1; 0.2 MPa otherwise | c020 |

For c020, `rho = 0.98^2 * clip(raw TargetArea/RestArea, 0.5, 1)` and
`ActivationInv = [rho^-1/2-1, rho^-1/2-1, 0]`. Both skin factors apply only to
the 29,899 triangles whose three vertices are `IsFace`; the artificial
cross-section is excluded.

All other anatomy and numerics are held fixed: fat is Stable Neo-Hookean at
0.003 MPa and nu=0.49; muscle is Active Stable Neo-Hookean at 0.03 MPa and
nu=0.49; aponeurosis is Stable Neo-Hookean at 0.1 MPa and nu=0.35. Skin is a
1 mm Koiter membrane with nu=0.49, the corrected plane-stress Lame conversion,
and the original RestArea. All vertices incident on the artificial cross-section
are hard-fixed to zero displacement. Each inverse starts from zero and uses
Adam at lr=0.3 for 40 updates (41 evaluations).

## Standalone material-assignment visuals

Show these one at a time; do not make a contact sheet:

1. `20-ablation-material-young-h0.png`: homogeneous 0.2 MPa skin.
2. `20-ablation-material-young-hf.png`: the 0.003/0.2 MPa heterogeneous skin.
3. `20-ablation-material-prestrain-p0.png`: no prestrain.
4. `20-ablation-material-prestrain-p1.png`: c020 stress-free-area ratio.

The Young's-modulus maps use a linear 0--0.2 MPa scale. The prestrain maps use
the same rho scale. The heterogeneous low-E region contains 16,723 triangles
and 54.5531% of the IsFace rest area.

## Standalone step-40 fitting-error visuals (primary)

Show H0P0, H0P1, HFP0, and HFP1 separately in that order. Use the
`*-point-error.png` files. The scalar is the objective-aligned pointwise
Euclidean magnitude `||Displacement-TargetDisplacement||`, in millimeters. All
four use one linear nonnegative scale from 0 to 9.931346 mm, the pooled maximum,
with no clipping. This is the appropriate field for discussing target fitting.

The inverse target RMS uses 15,302 LossMask points. The rendered IsFace surface
contains 15,299 of them; three objective points are off that surface. The
receipt records their squared-error contribution and verifies that the surface
point errors plus those three points reconstruct the reported target RMS exactly
for every case.

## Standalone signed-normal-residual visuals (optional diagnostic)

The `*-normal-residual.png` files are not the fitting objective. They show the
signed projection of the vector mismatch onto the target surface normal, which
is the field used to derive the `L` bumpiness diagnostic. If shown, all four use
the identical signed range of +/-3.815714 mm.

## Standalone step-40 geometry visuals

Use the matching `*-geometry.png` files, again one case at a time. The camera,
projection, lighting, and image resolution are identical across cases.

## Consistently recomputed terminal metrics

| case | target RMS (mm) | D (deg) | L (mm) | area RMS | activation RMS | folds | inverted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0P0 | 2.720948 | 13.327645 | 0.217061 | 0.140860 | 0.0609580 | 25 | 47 |
| H0P1 | 2.849029 | 5.579129 | 0.181487 | 0.072789 | 0.0548136 | 11 | 31 |
| HFP0 | 1.481081 | 15.486252 | 0.190230 | 0.153874 | 0.0675393 | 28 | 54 |
| HFP1 | 1.441860 | 7.564420 | 0.139811 | 0.094292 | 0.0628452 | 10 | 58 |

`D` is the contraction-region dihedral RMS and `L` is the residual-normal
Laplacian RMS. All values were recomputed with one registered implementation;
legacy bumpiness values were not mixed in.

## Meeting claim

- Heterogeneous softening is the fitting mechanism: target RMS falls 45.57%
  without prestrain and 49.39% with prestrain.
- Prestrain is the smoothing mechanism: it lowers D by 58.14% in the
  homogeneous branch and 51.15% in the heterogeneous branch; it also lowers L,
  area error, and folds in both branches.
- HFP1 is the best combined result among these four: target RMS 1.442 mm,
  L=0.140 mm, and 10 folded skin triangles. Relative to H0P0, these improve by
  47.01%, 35.59%, and 15 triangles respectively.
- Do not claim universal stability: HFP1 has 58 inverted tetrahedra. Also,
  softening alone worsens D and area error, which is precisely why the
  prestrain factor remains useful.

Frame this as a fixed-budget, target-derived ablation showing complementary
effects. It is not yet evidence of physiological material identification or
generalization to a new target.

## Provenance

The exact source paths, SHA-256 hashes, field-array hashes, point-error/RMS
linkage, ParaView version, and output hashes are in
`data/20-ablation-assets-receipt.json`. The augmented ParaView inputs containing
`TargetPointErrorMM` are in `data/20-ablation-render-inputs/`. The full metric
table is also available as `data/20-ablation-step40-metrics.json` and
`data/20-ablation-step40-metrics.md`. Historical E=0 selective studies and the
erroneous 3D-Lame skin cohort are explicitly excluded from this 2x2.
