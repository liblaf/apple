# Lamé-conversion meeting assets

## Meeting claim

The old skin setup converted the 3D Young's modulus and Poisson ratio directly
to 3D Lamé parameters before passing them to a two-dimensional membrane. For
`E=0.2 MPa` and `nu=0.49`, this assigned
`lambda=3.288590604 MPa`. The corrected plane-stress reduction assigns
`lambda=0.128964337 MPa`; `mu=0.067114094 MPa` is unchanged. The old `lambda`
was therefore `25.5x` too large, and the isotropic in-plane area coefficient
`lambda + mu` was `17.114x` the corrected value.

The rendered pair is a conversion-only fixed-activation forward replay. Both
cases use the historical full membrane, historical `IsFixed` boundary, pinned
step-40 `e100-p000` activation, exact-zero displacement seed, 1 mm skin,
homogeneous `E=0.2 MPa`, and no pre-strain. Only the skin Lamé conversion
changes. Volume materials continue to use the 3D conversion.

## Separate ParaView assets

Every image below is an independent 1800 x 1800 ParaView 6.1.1 render with its
own `.pvsm` state. No contact sheet or comparison grid was generated. All four
renders use the same front-facing parallel camera.

### Old 3D-Lamé geometry

![Old 3D-Lamé geometry](../data/10-lame-assets/10-lame-old-3d-geometry.png)

- [PNG](../data/10-lame-assets/10-lame-old-3d-geometry.png)
- [ParaView state](../data/10-lame-assets/10-lame-old-3d-geometry.pvsm)

### Corrected plane-stress geometry

![Corrected plane-stress geometry](../data/10-lame-assets/10-lame-corrected-plane-stress-geometry.png)

- [PNG](../data/10-lame-assets/10-lame-corrected-plane-stress-geometry.png)
- [ParaView state](../data/10-lame-assets/10-lame-corrected-plane-stress-geometry.pvsm)

### Old 3D-Lamé area strain

![Old 3D-Lamé area strain](../data/10-lame-assets/10-lame-old-3d-area-strain.png)

- [PNG](../data/10-lame-assets/10-lame-old-3d-area-strain.png)
- [ParaView state](../data/10-lame-assets/10-lame-old-3d-area-strain.pvsm)

### Corrected plane-stress area strain

![Corrected plane-stress area strain](../data/10-lame-assets/10-lame-corrected-plane-stress-area-strain.png)

- [PNG](../data/10-lame-assets/10-lame-corrected-plane-stress-area-strain.png)
- [ParaView state](../data/10-lame-assets/10-lame-corrected-plane-stress-area-strain.pvsm)

The area maps show `100 * (A_deformed / A_rest - 1)` on the canonical 29,899
`IsFace` triangles. Both use the same symmetric `+/-7.322%` range, the rounded
pooled absolute 99th percentile (`7.322454%`).

## Quantitative readback

| conversion | mean area ratio | mean absolute area strain | RMS area strain |
| --- | ---: | ---: | ---: |
| old 3D Lamé | 1.000382 | 0.2382% | 0.4146% |
| corrected plane stress | 1.015400 | 1.8397% | 2.7844% |

The corrected response has `6.72x` the RMS area strain under the same muscle
activation. This supports the narrow conclusion that the old conversion
artificially locked in-plane area deformation.

It does **not** show that plane stress alone improves the target fit or removes
bumpiness. In this fixed-activation probe, target-area-weighted error changes
from `3.671 mm` to `3.683 mm`, target-relative dihedral RMS from `7.138 deg` to
`8.862 deg`, and residual-normal Laplacian RMS from `0.2439 mm` to `0.2507 mm`.
The images should therefore be described as a constitutive correction and
compliance comparison, not as a better inverse result.

## Suggested slide wording

Title: **Thin skin requires a plane-stress reduction**

- Old: 3D `lambda=3.289 MPa`; corrected: plane-stress
  `lambda=0.129 MPa`; `mu` unchanged.
- The old in-plane area coefficient was `17.1x` too large.
- With activation held fixed, RMS area strain changes from `0.415%` to
  `2.784%`: the corrected membrane can expand instead of behaving nearly
  area-incompressibly.
- This correction is necessary, but it does not by itself solve the visible
  bumpiness.

## Reproduction and provenance

Run from this experiment group:

```bash
DEBUG=1 \
CHERRIES_NAME="Meeting Lamé conversion assets" \
CHERRIES_TAGS="human-face,skin,lame,plane-stress,paraview,meeting-assets,debug" \
uv run --frozen python src/10-lame-run.py
```

The wrapper uses PyVista only to prepare and strictly read back the two
canonical `IsFace` VTP inputs. `/usr/bin/pvbatch` 6.1.1 renders every output
pixel and saves every state. The run executes no forward, inverse, adjoint, or
backward solve.

- [Strict receipt](../data/10-lame-receipt.json)
- [ParaView contract](../data/10-lame-contract.json)
- [Prepared inputs](../data/10-lame-inputs)
- [Run log](../logs/10-lame-run.log)

Visual review passed for all four PNGs: the front camera is identical, each
face is fully in frame, backgrounds are white, the geometry images contain no
overlay, and the two area-strain images have the same legible scalar bar.
