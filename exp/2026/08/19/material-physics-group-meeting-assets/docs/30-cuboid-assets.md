# Cuboid fat-thickness meeting assets

## Meeting-safe result

At bottom pressure `0.60` model stress, increasing the controlled cuboid's top-fat thickness from `0.04` to `0.12` reduced absolute top-surface p95-p05 variation by `28.3%` and finite-difference Laplacian RMS by `43.8%`.

This supports reduced absolute surface variation in this toy block only. It does not establish scale-invariant smoothing or an anatomical-face result.

## Controlled setup

- Block footprint: `1 × 1` model length.
- Bottom fat: `0.04`; SMAS: `0.02`; top fat: `0.04 / 0.08 / 0.12`.
- Fat: `E = 1`; SMAS: `E = 100`; all `nu = 0.49` in model units.
- Fixed SMAS pre-strain: `(0.8, 1.0, 0.8, 0, 0, 0)`.
- All displacement components fixed on four vertical sides; positive-y pressure applied on the free bottom-interior surface.
- Each thickness was independently remeshed and solved by continuation to `0.60`.

## Pressure-0.60 metrics

| top fat | p95-p05 | Laplacian RMS | max displacement | min detF |
| ---: | ---: | ---: | ---: | ---: |
| 0.04 | 0.04359485 | 8.561539 | 0.06819046 | 0.501739 |
| 0.08 | 0.03838787 | 5.781417 | 0.06641315 | 0.532779 |
| 0.12 | 0.03126095 | 4.813979 | 0.06496486 | 0.460690 |

All three cases were `PRIMARY_SUCCESS`, finite, display-valid, with zero inverted tetrahedra and zero flipped top triangles.

## Asset contract

- The three 3D PNGs are separate native ParaView 6.1.1 renders.
- Each PNG has a separate `.pvsm` state.
- Camera, parallel scale, warp factor `1.0`, and vertical-displacement color range are identical across all three images.
- The white outline is the undeformed rest shape.
- The p95-p05 and Laplacian RMS summaries are two separate standalone Matplotlib images; there is no combined meeting chart.

## Standalone meeting asset inventory

- `paraview/top-fat-0p04/30-cuboid-top-fat-0p04-paraview.png` with its `.pvsm`.
- `paraview/top-fat-0p08/30-cuboid-top-fat-0p08-paraview.png` with its `.pvsm`.
- `paraview/top-fat-0p12/30-cuboid-top-fat-0p12-paraview.png` with its `.pvsm`.
- `metrics/30-cuboid-top-surface-p95-p05.png`.
- `metrics/30-cuboid-top-surface-laplacian-rms.png`.

## Provenance and limitations

- Authoritative archived snapshot: `/home/liblaf/mnt/DATA41/cherries/liblaf/apple/runs/2026/08/11/fat-layer-thickness-sandwich/30-run-large-deformation-fat-thickness/2026-08-12T141945-fat-large-deformation-thickness`.
- The archived run used `DEBUG=1`, so it is local-only and has no Comet run.
- This asset build reran no simulation, inverse, adjoint, or optimizer; it only copied identity-pinned results and rendered/plot them.
- Maximum displacement changed only about `4.7%`, but normalized Laplacian was non-monotone (`180.99 → 138.80 → 141.92`).
- The thicknesses were independently remeshed; this does not remove volumetric remeshing bias. Self-collision was disabled.
- At `0.60`, integrated force differed by about `0.36%` across cases.
- The `0.12` case had one positive tetrahedron below `detF = 0.5` (`min detF = 0.461`) but passed the documented display gates.

Machine-readable identities and commands are in `../data/30-cuboid-assets-receipt.json`.
