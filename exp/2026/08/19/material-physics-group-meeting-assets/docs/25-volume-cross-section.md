# Whole-anatomy volume-material cross-sections

These are three separate native ParaView 6.1.1 meeting assets, not a combined
image. Each PNG has a matching reopenable `.pvsm` state.

## Midsagittal mid-plane

![Midsagittal whole-anatomy volume-material cross-section](../data/25-volume-cross-section/25-volume-cross-section-midsagittal-dominant-material.png)

- Plane origin: `(1.407023565, 2.211267240, 0.044367692) m`
- Plane normal: `(1, 0, 0)`
- Intersected cells: 19,053 (17,116 fat, 1,667 muscle, 270 aponeurosis)
- [ParaView state](../data/25-volume-cross-section/25-volume-cross-section-midsagittal-dominant-material.pvsm)

## Coronal mid-plane

![Coronal whole-anatomy volume-material cross-section](../data/25-volume-cross-section/25-volume-cross-section-coronal-dominant-material.png)

- Plane origin: `(1.407023565, 2.211267240, 0.044367692) m`
- Plane normal: `(0, 0, 1)`
- Intersected cells: 20,808 (15,856 fat, 4,652 muscle, 300 aponeurosis)
- [ParaView state](../data/25-volume-cross-section/25-volume-cross-section-coronal-dominant-material.pvsm)

## Axial mid-plane

![Axial whole-anatomy volume-material cross-section](../data/25-volume-cross-section/25-volume-cross-section-axial-dominant-material.png)

- Plane origin: `(1.407023565, 2.211267240, 0.044367692) m`
- Plane normal: `(0, 1, 0)`
- Intersected cells: 14,023 (10,848 fat, 2,434 muscle, 741 aponeurosis)
- [ParaView state](../data/25-volume-cross-section/25-volume-cross-section-axial-dominant-material.pvsm)

## Material interpretation

| Category | Constitutive model | Young's modulus | Poisson ratio |
| --- | --- | ---: | ---: |
| Fat | Stable Neo-Hookean | 0.003 MPa | 0.49 |
| Muscle | active Stable Neo-Hookean | 0.030 MPa | 0.49 |
| Aponeurosis | Stable Neo-Hookean | 0.10 MPa | 0.35 |

Color is the dominant member of `FatFraction`, `MuscleFraction`, and
`AponeurosisFraction` in each intersected tetrahedron. This categorical field
is **visualization only**. The solver sums continuous fraction-weighted
constitutive energies, and the three active fractions sum bit-exactly to one on
all 1,146,517 tetrahedra.

All three planes pass through the same center of the pinned prepared-volume
bounds. The source is
`exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu`,
SHA-256
`8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563`.
Python/PyVista validates the source and prepares each small cross-section VTP;
ParaView performs all 3D rendering.

Run from `exp/2026/08/19/material-physics-group-meeting-assets`:

```bash
DEBUG=1 \
CHERRIES_NAME="Meeting three whole-anatomy volume cross-sections" \
CHERRIES_TAGS="meeting-assets,paraview,whole-anatomy,materials,cross-sections" \
uv run python src/25-volume-cross-section.py
```

The strict [contract](../data/25-volume-cross-section-contract.json),
[renderer receipt](../data/25-volume-cross-section-renderer-receipt.json), and
[final receipt](../data/25-volume-cross-section-receipt.json) pin all planes,
inputs, renderer settings, and output hashes.
