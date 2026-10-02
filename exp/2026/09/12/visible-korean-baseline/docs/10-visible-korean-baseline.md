# Visible Korean same-donor surface baseline

## Purpose

This experiment imports one internally consistent head specimen as a reusable
surface reference for the facial-model exploration. It creates a traceable
selection of skin, mimetic muscles, selected cranial and facial bones, and the
two palpebral ligaments from the Visible Korean male head surface-model archive.
It does not turn those surfaces into a biomechanical model.

The data record is [Visible Korean male head dataset](https://zenodo.org/records/20151689),
DOI [10.5281/zenodo.20151689](https://doi.org/10.5281/zenodo.20151689), distributed
under CC BY-NC 4.0. The experiment preserves the downloaded `4. Surface
models.zip` input unchanged; its SHA-256 is
`93f9c523c077d05df2e0b88306cd30dd34d85860d45e8eb2793f945a10e04eea`.

The live Zenodo [Authors/Creators display](https://zenodo.org/records/20151689)
was checked on 2026-09-12. It identifies Chung Yoh Kim (Dongguk University
School of Medicine, ORCID 0000-0001-8074-076X) as **Distributor**, and Jin Seo
Park (Ajou university School of Medicine, ORCID 0000-0001-7956-4148) as **Data
manager**. Credit these displayed roles and cite both this Zenodo record and its
associated Data Descriptor. The manifest preserves the URL, check date, and
display-location receipt. This is a metadata-only correction; it does not
alter any geometry output or its hash receipt.

## Command and run evidence

Run from this experiment group:

```bash
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Visible Korean same-donor surface baseline' \
CHERRIES_TAGS='facial-anatomy,visible-korean,surface-qa' \
uv run python src/10-import-visible-korean.py
```

The normal `ProfileCometNoCommit` profile was used: it enabled Cherries local
and Comet evidence, while registering the Git plugin with `commit=False`.
The clean final run created [experiment c6f505075aa74081ae6e433cc3e896d8](https://www.comet.com/liblaf/apple/c6f505075aa74081ae6e433cc3e896d8)
and exited after its normal metric and output upload.

The final run sets `COMET_AUTO_LOG_ENV_DETAILS=false` and
`COMET_AUTO_LOG_GIT_PATCH=false`. Comet remains enabled for the experiment
metrics and assets, while the two expensive automatic environment snapshots are
disabled. The importer writes all derived files before Cherries registers its
single declared output directory. Independent post-run checks read every
exported semantic surface, confirmed all manifest hashes, and confirmed all 25
VTM blocks. The earlier run log that stalled during automatic environment
capture remains at [`logs/10-import-visible-korean-first-run-env-flush-stalled.log`](../logs/10-import-visible-korean-first-run-env-flush-stalled.log)
for diagnosis; it is not the final evidence run.

## Source selection and coordinate interpretation

The importer requires every selected filename to occur exactly once in the
archive. It selected 25 structures:

| Group | Structures | Kept triangles |
| --- | ---: | ---: |
| Skin | 1 | 25,914 |
| Mimetic muscles | 18 | 388,386 |
| Bones | 4 | 576,504 |
| Palpebral ligaments | 2 | 7,304 |
| Total | 25 | 998,108 |

The source STL format does not encode units, axes, or an origin. The source
image documentation and the observed anatomical scale support interpreting the
native coordinates as millimetres. This is an interpretation for exchange and
inspection, not a documented anatomical coordinate system. The preview therefore
uses the neutral labels “Native View A” and “Native View B”; it does not assign
left/right, anterior/posterior, or superior/inferior axes.

![Same-donor selected anatomy preview](../data/10-baseline/visible-korean-baseline-preview.png)

## Explicit cleanup rules

The source meshes were read with `trimesh` processing disabled. Each mesh then
received exactly two deterministic edits:

1. Coordinate-identical vertices were merged only to identify triangles with
   the same unordered vertex triple. One first-occurring, originally wound face
   was retained for each duplicate group.
2. After that deduplication, face-connected components with fewer than 100
   faces were removed. Every retained component is exported separately.

No hole filling, nonmanifold repair, normal reorientation, smoothing,
decimation, registration, interface matching, or tetrahedralization occurred.
The receipts in the manifest show 999,477 source triangles, 376 exact duplicate
faces removed, 287 small components removed, and 998,108 triangles retained.
All retained semantic surfaces are watertight and winding-consistent except the
combined `cranium` surface, which retains 12 nonmanifold edges. That condition
is reported, not repaired.

The likely source spelling `Depressor aguli oris muscle.stl` is preserved as it
appears in the archive; it is not silently renamed to an anatomical claim.

## Outputs

All durable outputs are below [`data/10-baseline`](../data/10-baseline/):

- [`visible-korean-manifest.json`](../data/10-baseline/visible-korean-manifest.json)
  contains source hashes, the exact selection, cleanup recipe, per-object
  topology receipts, and output hashes.
- [`visible-korean-selected-anatomy.vtm`](../data/10-baseline/visible-korean-selected-anatomy.vtm)
  is a 25-block VTK package whose nested blocks separate every retained major
  connected component.
- [`surfaces/`](../data/10-baseline/surfaces/) contains one merged VTP surface
  per semantic structure; [`components/`](../data/10-baseline/components/)
  contains the separate retained components.
- [`visible-korean-baseline-preview.png`](../data/10-baseline/visible-korean-baseline-preview.png)
  is the rendered common-native-frame inspection asset.
- [`logs/10-import-visible-korean.log`](../logs/10-import-visible-korean.log)
  records the processing metrics and source member receipts.

The 133,865,842-byte original archive is intentionally kept at
[`data/source/Surface-models.zip`](../data/source/Surface-models.zip), separate
from every derived output.

## What this supports and what it does not

This baseline offers same-donor outer skin, named mimetic-muscle volumes,
skeletal context, and palpebral-ligament geometry for registration, region
definition, visual inspection, and designing later segmentation or attachment
studies. The semantic selection makes it more useful than mixing unrelated
surface atlases, but it is still a reference surface package.

It does not encode SMAS, galea, other aponeuroses, retaining ligaments or dermal
attachments, measured muscle fiber directions, origins or insertions as explicit
metadata, material parameters, a volume mesh, contact definitions, or solver
state. Those absences mean it is not FEM-ready and cannot justify simulated
muscle forces or sliding laws by itself.

## Reproducibility

The only executable is [`src/10-import-visible-korean.py`](../src/10-import-visible-korean.py).
It verifies the archive MD5, SHA-256, and ZIP integrity before reading meshes;
fails if a selected member is missing or ambiguous; and hashes every exported
surface and preview. The post-run verification re-opened the 25 VTP surfaces and
the VTM package, checked their cell counts against the manifest, and rechecked
all listed hashes. `uv run ruff check src` passed.
