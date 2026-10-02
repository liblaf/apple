# Five corresponding deformed tetrahedral meshes

The [mesh collection](../data/five-deformed-meshes/) now contains five `.vtp` skin surfaces and five `.vtu` tetrahedral meshes with matching filename stems. Each volume mesh has **228,660 vertices and 1,146,517 tetrahedra**. Coordinates use meters in the same reference frame as the surfaces.

| State | Volume mesh | Inverted cells | Size (MiB) |
| --- | --- | ---: | ---: |
| Free activation | [01-free-activation.vtu](../data/five-deformed-meshes/01-free-activation.vtu) | 1 | 121.30 |
| Dominant only | [02-dominant-only.vtu](../data/five-deformed-meshes/02-dominant-only.vtu) | 0 | 121.23 |
| Fixed-axis refit | [03-fixed-axis-refit.vtu](../data/five-deformed-meshes/03-fixed-axis-refit.vtu) | 0 | 121.83 |
| Released-axis continuation | [04-released-axis-continuation.vtu](../data/five-deformed-meshes/04-released-axis-continuation.vtu) | 16 | 121.34 |
| Learned-axis from scratch | [05-learned-axis-from-scratch.vtu](../data/five-deformed-meshes/05-learned-axis-from-scratch.vtu) | 94 | 121.09 |

The export uses `X+u` from each original saved result and retains the common fixture's tetrahedral connectivity. There was no new physics solve, remeshing, smoothing or registration. For every state, the skin points match the volume points indexed by the surface's `GlobalPointId` exactly. All five output files were reopened and checked for exact geometry, connectivity and public point/cell/field array round trips.

The volume files retain the fixture's material, region, boundary and expression attributes. `RestPosition` and `Displacement` are in meters. `DetF` is the physical deformation determinant and `IsInverted` marks `DetF <= 0`. `ActivationInverseMatrix` contains each state's actual B = A^-1, flattened row-major, with identity in inactive cells. Fixed-axis B is reconstructed from the saved strengths and initialized axes; released and scratch B are reconstructed from the saved three-component controls. The remaining states use their saved B directly. The fixture's old activation arrays have the `Fixture` prefix, and its `Volume` is named `ReferenceVolume`, to preserve their original/reference semantics.

The five checkpoints are unchanged from the [aligned comparison](96-aligned-five-way.md), including the original smoothness-on learned-axis scratch run at update 128. The saved geometry retains its recorded inversions. This packaging does not establish mesh validity or convergence.

## Provenance and execution

- [Exporter source](../src/100-export-tetmeshes.py).
- [Volume manifest, source records and hashes](../data/five-deformed-meshes/tetmesh-manifest.json).
- [Collection README](../data/five-deformed-meshes/README.md).
- [Complete process log](../data/five-deformed-meshes/tetmesh-export.log).
- [Comet run](https://www.comet.com/liblaf/apple/66bce036de1c466ebd5cc43e76504a3b).

The named Cherries run completed with exit code 0; the export completed at approximately 32 seconds. Its Comet summary records 15 point/tet/inversion metrics, configuration, source and Git metadata. Current experiment additions remain untracked and no Git commit was created. The known nonfatal Cherries Local internal log-copy error occurred during shutdown, so the full terminal log was copied manually. Local VTU files and their readback checks are the export evidence.

An initial attempt stopped after the first VTU write because PyVista adds private serialization metadata to the in-memory field data during `save()`. The comparison was corrected to capture the public attributes before saving and compare those to the reloaded mesh; the complete run then passed. The first attempt remains in the private `tmp/100-export-initial-attempt/` directory.

Working directory: `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Export five matching deformed tetrahedral meshes' \
CHERRIES_TAGS='face,activation,tetmesh,export,physical-volume' \
.venv/bin/python src/100-export-tetmeshes.py
```

Existing VTU files are protected by assertions. For a new export, use an alternate output directory containing the five source VTP files and pass it via `--output-dir`; preserve the completed collection.
