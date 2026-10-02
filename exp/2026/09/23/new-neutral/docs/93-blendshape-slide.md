# Transferred blendshapes on the new neutral

The slide contains the saved new neutral 005 and all 36 transferred expressions,
rendered with white surfaces in ParaView 6.1.1. The 9-column, 4-row expression
grid reads left to right, then top to bottom, in descending deformation order.
The neutral is a separate reference panel with the same orthographic camera,
viewport, scale, and lighting. Expression weight is 1.0.

The final slide includes expression names and omits displacement measurements
and diagnostic annotations. The meshes have zero specular reflectance, neutral
light colors, and ParaView light-kit intensity scaled to 0.85. Ambient lighting
is 0.5 and diffuse lighting is 0.3, reducing bright directional patches while
retaining facial contours.

## Outputs

- [8K PNG, 7680 × 4320](../data/93-blendshape-slide-no-highlights/transferred-blendshapes-new-neutral-8k.png)
- [1920 × 1080 preview](../data/93-blendshape-slide-no-highlights/transferred-blendshapes-new-neutral-preview.png)
- [Deformation ordering and original values](../data/93-blendshape-slide-no-highlights/deformation-ranking.json)
- [Rendering receipt](../data/93-blendshape-slide-no-highlights/render-receipt.json)

The PNG is exactly 16:9 and has 576 dpi metadata for a 13⅓ × 7½ inch slide.
The saved face renders use twice their final panel resolution before downsampling.
The output directory also contains the exact VTP surfaces, ParaView input settings,
and copies of both rendering scripts.

## Verification

The source bundle SHA256 matches its transfer manifest. Every rendered surface
has exactly the corresponding saved neutral or target coordinates and original
triangle connectivity. All 37 rendered faces have distinct file hashes. All 36
expression names occur once, the descending order is verified, and the final
PNG dimensions and artifact hashes were checked. Visual inspection confirmed
readable labels, unclipped meshes, and absence of the removed annotations.
Ruff check and formatting checks passed for both scripts. The revised lighting
keeps all geometry and ordering identical to the earlier export. On the neutral
mesh, the 99th-percentile grayscale intensity decreased from 234 to 198 out of
255, while the median remained similar (175 versus 180).

Ordering uses the RMS of each vertex's 3D displacement magnitude from the new
neutral, matching the bundle's existing statistic. This produces MouthOpenMax,
Scream, MouthOpen, and Compressed as the first four expressions. The values are
retained in the ordering JSON without appearing on the slide. This is a render
of the saved geometric targets; no forward solve is performed. The source
manifest's status and scope are preserved in the rendering receipt.

## Reproduction

From `exp/2026/09/23/new-neutral`, choose a fresh output path for a repeated run:

```bash
CHERRIES_NAME='ParaView blendshapes without lighting highlights - 8K slide' \
CHERRIES_TAGS='neutral,blendshapes,render,slide,white-mesh,paraview,no-highlights' \
.venv/bin/python -u \
  src/93-render-blendshape-slide.py --output data/93-blendshape-slide-no-highlights-rerun
```

The Cherries entrypoint invokes `/usr/bin/pvpython` for actual face rendering,
then uses Pillow to compose the slide. PyVista only exports geometry and normals.
The [completed Comet run](https://www.comet.com/liblaf/apple/037fe18544db49d9ba08914161996614)
records 36 expressions, 37 panels, width 7680, and height 4320. The terminal log is
[`logs/93-paraview-slide-no-highlights-terminal.log`](../logs/93-paraview-slide-no-highlights-terminal.log),
and the local Cherries run snapshot is under
`.cherries/runs/2026/09/23/new-neutral/93-render-blendshape-slide/` at repository root.
The working tree contained unrelated changes; no commit was created.
