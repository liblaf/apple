# Fat-floor skin inverse findings

## Outcome

Replacing exact zero skin stiffness on target-expanding triangles with a small
positive floor, `E = 0.003 MPa`, reduces visible and measured surface
corrugation. It does not remove the artifact completely, and it trades a small
amount of terminal target fit for smoother skin.

The clearest result is the prestrained pair. Relative to the old `E = 0`
case, the fat-floor case reduces residual-Laplacian RMS by 19.45% and folded
skin triangles from 16 to 10, while target RMS increases by 1.96%. Without
prestrain, residual-Laplacian RMS falls by 9.95% and folds fall from 39 to 28,
while target RMS increases by 3.00%.

The terminal native ParaView comparison is shown below.

![Terminal geometry comparison](../data/26-paraview-fat-floor-terminal/plates/26-paraview-terminal-geometry.png)

![Terminal target-normal residual comparison](../data/26-paraview-fat-floor-terminal/plates/26-paraview-terminal-normal-residual.png)

## Cases

All four columns use the corrected all-vertex `IsFace` membrane, 1 mm
thickness, plane-stress Lame parameters, fixed original `RestArea`, the
hard-fixed cut boundary, unchanged volume materials, and the same 40-update
fresh-zero inverse protocol.

| Case | Expanding triangles (`TargetArea / RestArea > 1`) | Prestrain |
| --- | ---: | --- |
| H1P0 | `E = 0` | none |
| HFP0 | `E = 0.003 MPa` | none |
| H1P1 | `E = 0` | c020 |
| HFP1 | `E = 0.003 MPa` | c020 |

Non-expanding triangles retain `E = 0.2 MPa`. The c020 field is
`rho = 0.98^2 * clip(TargetArea / RestArea, 0.5, 1)` with in-plane
`ActivationInv = rho^(-1/2) - 1`. Because HFP1 has nonzero stiffness on every
triangle, c020 is mechanically active on the expanding region as well.

`E = 0.003 MPa` is numerically equal to the fat Young's modulus, but the skin
is a 2D plane-stress membrane and fat is a 3D nearly incompressible solid. The
case is therefore a nonzero-floor sensitivity study, not a claim that the two
materials are mechanically equivalent.

## Equal-budget terminal results

| Case | Target RMS (mm) | Lres (mm) | Folds | Inverted | Activation RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1P0 | 1.4379 | 0.2552 | 39 | 41 | 0.06727 |
| HFP0 | 1.4811 | 0.2299 | 28 | 54 | 0.06754 |
| H1P1 | 1.4142 | 0.2246 | 16 | 50 | 0.06285 |
| HFP1 | 1.4419 | 0.1809 | 10 | 58 | 0.06285 |

Paired fat-floor effects:

| Comparison | Target RMS | Residual-Laplacian RMS | Folds | Inverted tets |
| --- | ---: | ---: | ---: | ---: |
| H1P0 -> HFP0 | +3.00% | -9.95% | -11 | +13 |
| H1P1 -> HFP1 | +1.96% | -19.45% | -6 | +8 |

The geometry plates show the same direction as the residual-Laplacian metric:
the fat floor softens high-frequency corrugation around the mouth, lateral
cheek, and lower-eye region, with the larger improvement in the c020 pair.
HFP1 is the smoothest of the four by residual-Laplacian RMS and fold count, but
it is not the best target-fit case.

## Numerical validity and limitations

- Both new cases completed 40 optimizer updates and 41 evaluations. All 82
  forward and adjoint solves succeeded; best state was step 40 in both cases.
- Every one of the 41 saved frames retains exact-zero displacement on all
  33,636 fixed vertices, including all 6,980 cut vertices.
- Result VTUs equal history frame 40 exactly, and all material and provenance
  readbacks pass.
- The runs use a fixed update budget and are not convergence claims.
- Folds and inverted tetrahedra remain warning-level defects. The fat floor
  reduces folds but increases inverted-tet counts, so it is not an
  unconditional admissibility improvement.
- The stiffness mask and c020 field are derived from this same target; the
  result is a deterministic target-informed ablation on one expression, not a
  physiological calibration or a generalization result.
- The old `E = 0` artifacts remain unchanged for comparison.

## Reproducibility

The canonical formal aggregate is
`data/20-fat-floor-skin-prestrain-inverse-summary-final.json` (270,811 bytes,
SHA-256
`82d48d6629b7760c0bf6df8fded8fdaae21c5edf7ad525f5965ce51ae2d2f0b2`).
The inverse producer SHA-256 is
`16afc483691850b04afeb4dc859d7e3dff4bd96c3a049314bf43fab60d6f26d4`.

The native ParaView 6.1.1 receipt is
`data/26-paraview-fat-floor-terminal-receipt.json` (SHA-256
`8ceaac5b1ac4f88f3d965d03b7c2215ba1f12bb0260301cdacbe23b3710eb862`).
The two PNGs have matching reusable `.pvsm` state files in the same directory.
