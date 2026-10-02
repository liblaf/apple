# Target surface in the mesh collection

Added [00-target.vtp](../data/five-deformed-meshes/00-target.vtp) to the same folder as the five result surfaces and tetrahedral meshes. The target has **15,299 vertices and 29,899 surface cells**, preserving the original surface topology, GlobalPointId correspondence and shared coordinate frame in meters.

Target positions equal `X + Smile` at each surface GlobalPointId. They match every collected result's `TargetPosition` array exactly. The mesh includes `RestPosition`, the target `Displacement`, `TargetPosition` and a zero `PointToPointErrorMm` field. No registration or new physics solve was applied.

The available target is incomplete in the volume: **33,200 of 228,660 volume vertices** have missing Smile displacements, affecting **197,691 of 1,146,517 tetrahedra**. All skin targets are finite. A complete target tetrahedral mesh cannot be exported from these coordinates without imputing values. No incomplete or imputed target VTU was added.

## Provenance and verification

- [Exporter](../src/101-export-target-surface.py).
- [Manifest with source/output hashes](../data/five-deformed-meshes/target-manifest.json).
- [Process log](../data/five-deformed-meshes/target-export.log).
- [Comet run](https://www.comet.com/liblaf/apple/29c31c7f5ac54635a958b8348a4d74e8).

The normal named Cherries run completed with exit code 0. Its summary records the geometry counts, missing-target counts, input/output configuration and source/Git metadata. The output was reopened and checked for finite positions, exact target positions, unchanged triangles and point IDs. The complete terminal log is retained with the collection. Local files are the export evidence.

Working directory: `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Export exact target surface for five-state comparison' \
CHERRIES_TAGS='face,target,mesh,export,physical-volume' \
.venv/bin/python src/101-export-target-surface.py
```

The exporter refuses to overwrite existing outputs. Use distinct `--output` and `--manifest` paths for a new export.
