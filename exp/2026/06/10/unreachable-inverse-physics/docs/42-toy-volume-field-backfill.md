# Toy Volume Field Backfill

## Purpose

The completion audit found that the primary `20-toy-*` final and target VTUs did not yet contain per-tetra volume-change arrays, even though the summary tables had the aggregate volume metrics and the newer `30` sweep VTUs already contained those arrays.

This post-processing run backfilled the six primary toy inverse result VTUs and their target VTUs with per-tetra target and inverse volume diagnostics.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-inverse-physics
```

Command:

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="toy volume field artifact backfill" \
CHERRIES_TAGS="unreachable-inverse,toy,volume-fields,artifact-backfill" \
uv run python src/42-toy-volume-field-backfill.py
```

Comet run: <https://www.comet.com/liblaf/apple/40b3035e2a22494cb2416286c3e2211e>

Cherries summary fields from the run log:

- `cherries/entrypoint`: `exp/2026/06/10/unreachable-inverse-physics/src/42-toy-volume-field-backfill.py`
- `cherries/exp_dir`: `exp/2026/06/10/unreachable-inverse-physics`
- `cherries/start_time`: `2026-06-10 15:30:50.554760+08:00`
- `cherries/end_time`: `2026-06-10 15:30:51.473040+08:00`
- metric: `patched_cases = 6`

The Cherries process exited with code `0`. During shutdown, the Git plugin logged a sandbox-related failure while trying to run `git diff` through the Git LFS clean filter:

```text
Error cleaning Git LFS object: open .git/lfs/tmp/...: read-only file system
```

The local VTU rewrites, summary files, and Comet summary were still written. The Git plugin limitation is specific to the managed sandbox making `.git/lfs/tmp` read-only.

## Outputs

- `data/42-toy-volume-field-backfill-summary.json`
- `data/42-toy-volume-field-backfill-cases.csv`
- `data/42-toy-volume-field-backfill-table.md`
- `logs/42-toy-volume-field-backfill.log`

The run overwrote these final result VTUs and target VTUs in place:

- `data/20-toy-stretch-coarse.vtu`
- `data/20-toy-stretch-medium.vtu`
- `data/20-toy-stretch-fine.vtu`
- `data/20-toy-squash-coarse.vtu`
- `data/20-toy-squash-medium.vtu`
- `data/20-toy-squash-fine.vtu`
- matching `*-target.vtu` files

The final result VTUs now include:

- `VolumeInitial`
- `VolumeTarget`
- `VolumeInverse`
- `VolumeTargetRelChange`
- `VolumeInverseRelChange`
- `SignedVolumeInitial`
- `SignedVolumeTarget`
- `SignedVolumeInverse`
- `SignedVolumeTargetRelChange`
- `SignedVolumeInverseRelChange`

## Results

| case | tets | target signed dV min | target signed dV max | inverse signed dV min | inverse signed dV max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20-toy-squash-coarse` | 2304 | -100.0000% | 0.0000% | -137.5320% | 265.7950% |
| `20-toy-squash-medium` | 11760 | -200.0000% | 0.0000% | -271.6530% | 488.1490% |
| `20-toy-squash-fine` | 24000 | -200.0000% | 0.0000% | -384.7200% | 1113.6000% |
| `20-toy-stretch-coarse` | 2304 | 0.0000% | 100.0000% | -298.4190% | 1282.6900% |
| `20-toy-stretch-medium` | 11760 | 0.0000% | 200.0000% | -457.3500% | 1666.5000% |
| `20-toy-stretch-fine` | 24000 | 0.0000% | 200.0000% | -437.7550% | 2742.0000% |

## Analysis

This backfill does not change the inverse solution itself. It only adds the missing per-tetra diagnostic arrays to the saved final VTUs so the artifacts match the experiment requirement: volume change can now be inspected for each tetrahedron for both the target displacement and the inverse solution.

The extreme per-tetra signed-volume ranges are local cell diagnostics, not aggregate whole-mesh volume changes. The target ranges are expected for the toy targets because the prescribed top displacement is applied kinematically to only part of the boundary and is not a solved physical deformation. The inverse ranges are the corresponding best recovered physical states from the existing `20` run.

## Reproducibility

The script imports the shared volume helper from `src/20-toy-unreachable-inverse.py` and uses the existing final/target VTUs as inputs. `uv run python -m py_compile` and `uv run ruff check` passed for the script before running it.
