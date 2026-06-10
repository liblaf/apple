# Toy TetWild Area / Volume and Inverse Diagnostics

## Purpose

This run adds the TetWild path requested for the toy unreachable-inverse setup. It uses the same box geometry, fixed boundary, material split, stiffnesses, Poisson ratio, and stretch/squash target displacement as the structured-mesh `20` run, but generates tetrahedral meshes with TetWild at:

- `lr = 0.05`
- `lr = 0.02`
- `lr = 0.01`

Because TetWild does not conform to the SMAS and muscle boxes in this run, each tetra stores sampled volume fractions:

- `MuscleFraction`
- `AponeurosisFraction = max(0, SmasFraction - MuscleFraction)`
- `FatFraction = 1 - AponeurosisFraction - MuscleFraction`

The same deterministic sample points are used for SMAS and muscle classification inside each tetra, so the muscle fraction is clamped to stay within the sampled SMAS support.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-inverse-physics
```

Smoke command:

```bash
DEBUG=1 \
CHERRIES_NAME="toy tetwild smoke" \
CHERRIES_TAGS="unreachable-inverse,tetwild,smoke" \
uv run python src/50-toy-tetwild-experiments.py \
  --lrs 0.05 \
  --run-forward false \
  --run-inverse false
```

Report command:

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="toy tetwild area volume and inverse diagnostics" \
CHERRIES_TAGS="unreachable-inverse,tetwild,area-volume,toy,nu049" \
uv run python src/50-toy-tetwild-experiments.py
```

Comet run: <https://www.comet.com/liblaf/apple/1fad5eb1d49c4d588d5d8dd44c4853d2>

Cherries summary fields from the run log:

- `cherries/entrypoint`: `exp/2026/06/10/unreachable-inverse-physics/src/50-toy-tetwild-experiments.py`
- `cherries/exp_dir`: `exp/2026/06/10/unreachable-inverse-physics`
- `cherries/git/sha`: `b12ff858beeade588637b7766cc732ff27b1897f`
- `cherries/start_time`: `2026-06-10 15:51:36.847614+08:00`
- `cherries/end_time`: `2026-06-10 15:53:15.043919+08:00`

## Outputs

- `data/50-toy-tetwild-experiments-summary.json`
- `data/50-toy-tetwild-experiments-cases.csv`
- `data/50-toy-tetwild-experiments-table.md`
- `data/50-toy-tetwild-lr005-input.vtu`
- `data/50-toy-tetwild-lr002-input.vtu`
- `data/50-toy-tetwild-lr001-input.vtu`
- stretch/squash target VTUs for all three `lr` values
- `data/50-toy-tetwild-forward-lr005.vtu`
- `data/50-toy-tetwild-stretch-lr005.vtu`
- `data/50-toy-tetwild-squash-lr005.vtu`
- `data/50-toy-tetwild-stretch-lr005.vtu.series`
- `data/50-toy-tetwild-squash-lr005.vtu.series`
- `data/50-toy-tetwild-area-surfaces/*.vtp`
- `logs/50-toy-tetwild-experiments.log`

The two inverse `.vtu.series` outputs each contain `9` frames, from step `0` through step `80` with `series_stride = 10`.

## Results

| kind | case | lr | tets | active tets | signed dV | target/top dA | error / target | top y std | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| target | `50-toy-tetwild-stretch-lr005` | 0.05 | 2886 | 167 | 17.6941% | 0.0000% | | | kinematic target |
| target | `50-toy-tetwild-squash-lr005` | 0.05 | 2886 | 167 | -17.6941% | 0.0000% | | | kinematic target |
| target | `50-toy-tetwild-stretch-lr002` | 0.02 | 42343 | 2049 | 19.0257% | 0.0000% | | | kinematic target |
| target | `50-toy-tetwild-squash-lr002` | 0.02 | 42343 | 2049 | -19.0257% | 0.0000% | | | kinematic target |
| target | `50-toy-tetwild-stretch-lr001` | 0.01 | 369071 | 9363 | 19.5456% | 0.0000% | | | kinematic target |
| target | `50-toy-tetwild-squash-lr001` | 0.01 | 369071 | 9363 | -19.5456% | 0.0000% | | | kinematic target |
| forward | `50-toy-tetwild-forward-lr005` | 0.05 | 2886 | 167 | -4.6240% | 1.4474% | | 0.011763 | `primary_success` |
| inverse | `50-toy-tetwild-stretch-lr005` | 0.05 | 2886 | 167 | 3.5797% | 0.5303% | 87.6574% | 0.006956 | `not_converged_best_in_last_window` |
| inverse | `50-toy-tetwild-squash-lr005` | 0.05 | 2886 | 167 | -3.8146% | 0.2534% | 86.4629% | 0.007159 | `not_converged_best_in_last_window` |

## Analysis

The TetWild target diagnostics agree with the structured-mesh result: the prescribed top motion is mostly a volume-change demand. As the TetWild mesh is refined from `lr=0.05` to `lr=0.01`, the kinematic target signed-volume change rises from about `17.7%` to `19.5%`. The top surface itself has zero target area change because the prescribed motion is a uniform y-translation of the selected top points.

The `lr=0.05` forward contraction is a valid equilibrium under the requested activation `(-0.5, 0, 0, 0, 0, 0)`. It reaches `primary_success`, has no inverted tetrahedra, and changes signed volume by only `-4.6240%`. Its top response is nonuniform (`top y std = 0.011763`), which is consistent with the small active muscle patch creating spatially uneven motion rather than a uniform lift or squash.

The `lr=0.05` inverse runs reproduce the unreachable behavior on a TetWild mesh. Stretch asks for `+17.6941%` signed volume but recovers only `+3.5797%`. Squash asks for `-17.6941%` signed volume but recovers only `-3.8146%`. Both inverse runs keep large residuals, about `86%` to `88%` of the target displacement RMS, and their best states are still in the final convergence window.

This supports the same conclusion as the structured toy experiments: high Poisson ratio plus fixed sides/bottom and limited active support makes the uniform top motion largely unreachable. The solver responds by finding a smaller-volume-change branch with bumpy top-surface displacement, not by matching the target exactly.

## Limitations

Only the `lr=0.05` TetWild mesh was forward/inverse solved in this report-worthy run. The `lr=0.02` and `lr=0.01` TetWild meshes were generated and diagnosed for target area/volume fields, but not solved. The `lr=0.01` mesh has `369071` tetrahedra, so a full differentiable inverse solve at that resolution is substantially heavier than the coarse run.

The fraction fields are sampled with `16` deterministic points per tetra. This is enough for a diagnostic experiment, but boundary tetra fractions should be treated as approximate. Increase `--fraction-samples-per-tet` for a slower, less noisy fraction estimate.

The Cherries local snapshot plugin logged repeated `FileNotFoundError` messages while checking the copied run log path. The main process still exited with code `0`, wrote the summary/table/VTU artifacts, uploaded the Comet summary, and created the Cherries/Git commit listed above.

## Reproducibility

The script uses CUDA through Warp/Torch on `cuda:0` for the forward and inverse solves. `uv run python -m py_compile` and `uv run ruff check` passed for `src/50-toy-tetwild-experiments.py` before the report-worthy run.

The default run command can be narrowed or expanded with comma-separated CLI values:

```bash
uv run python src/50-toy-tetwild-experiments.py \
  --lrs 0.05,0.02,0.01 \
  --forward-lrs 0.05 \
  --inverse-lrs 0.05
```
