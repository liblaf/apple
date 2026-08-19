# Large-deformation Fat-layer Thickness Sweep

## Purpose

Increase the simulated bottom pressure through continuation so that the
effect of top-fat thickness is visually legible without multiplying the
rendered displacement. The three reported thicknesses are 0.04, 0.08,
and 0.12; the reported pressures are 0.30, 0.45, and 0.60. All quantities
are model units; no SI calibration is asserted.
All displacement components are fixed on the four vertical sides;
positive-y pressure acts on the free bottom-interior surface. The SMAS
layer uses the fixed active pre-strain listed in the controlled config.

## Command

Working directory: `/home/liblaf/github/liblaf/apple/exp/2026/08/11/fat-layer-thickness-sandwich`

```console
DEBUG=1 CHERRIES_NAME=fat-large-deformation-thickness CHERRIES_TAGS=fat,thickness,large-deformation,continuation /home/liblaf/github/liblaf/apple/.venv/bin/python3 src/30-run-large-deformation-fat-thickness.py --overwrite true
```

## Selected common pressure

The highest attempted report pressure passing the heuristic display gate for all three thicknesses was **0.60** model stress. This is not a measured maximum-safe load.

| top fat [model length] | solver | steps | free-grad RMS [model force] | max displacement [model length] | top p95-p05 [model length] | top Lap RMS [model length^-1] | min detF | q0.001 detF |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.04 | PRIMARY_SUCCESS | 831 | 6.196e-09 | 0.06819 | 0.04359 | 8.5615 | 0.502 | 0.592 |
| 0.08 | PRIMARY_SUCCESS | 706 | 4.825e-09 | 0.06641 | 0.03839 | 5.7814 | 0.533 | 0.601 |
| 0.12 | PRIMARY_SUCCESS | 827 | 4.731e-09 | 0.06496 | 0.03126 | 4.814 | 0.461 | 0.605 |

At bottom pressure 0.60, on the common material-coordinate grid,
the thinnest-to-thickest
p95-p05 reduction was 28.3%, and the finite-difference Laplacian RMS reduction was 43.8%.
Maximum displacement changes by only 4.7%. The normalized Laplacian is 180.99 → 138.80 → 141.92, so scale-normalized smoothing is not monotone from 0.08 to 0.12. The supported claim is only
reduced absolute surface variation in this controlled block, not a claim
about anatomical faces or scale-invariant smoothing.

## Outputs

- `data/30-large-deformation-summary.json`
- `data/30-large-deformation-summary.csv`
- `figs/30-large-deformation-isometric.png`: simulated outer-surface u_y plus rest outline
- `figs/30-large-deformation-section.png`: layered central section plus rest outline
- `figs/30-large-deformation-top-uy.png`: shared-scale top-surface u_y on a common x-z grid
- Per-pressure VTU states and interpolated top-grid NPZ files under `data/`

## Safety and limitations

The display gate requires solver success, finite values, no inverted
tetrahedra, min detF >= 0.20, q0.001 detF >= 0.40, and no flipped top triangles. The continuation gate also requires solver success, finite values, no inversion,
no top flip, and min detF >= 0.10.
These are heuristic geometry screens, not physical-validity guarantees.
The 0.12 case has one tetrahedron below detF 0.5 at every attempted load,
although it remains positive and passes the stated display thresholds.
Self-collision remains disabled because the installed IPC path crashes
on the empty collision set in this sandwich model. Positive local detF
does not certify absence of global self-intersection, so the render must
also be inspected before drawing a mechanics conclusion.

Each thickness is remeshed independently, but the reported surface
metrics are interpolated in undeformed material coordinates onto the same
101 x 101 x-z grid. This avoids directly comparing
different graph neighborhoods, but does not remove surface interpolation
or independent volumetric-remeshing bias.
At bottom pressure 0.60, integrated force differs by 0.36% across the independently remeshed cases.

This run used DEBUG=1, so Cherries logging is local-only and has no Comet run. Inspect the completed local log for the command and runtime metadata.
