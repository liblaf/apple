# Fat `nu=0.499` activation continuation

## Purpose

This run tests whether the saved FiberModes activation from run 24 admits a
strict forward equilibrium when the fat Poisson ratio is changed from 0.49 to
0.499. It holds the saved control vector and activation tensor fixed at the
final endpoint and reaches that endpoint through ten equal activation
increments. Each nonzero stage starts from the preceding accepted equilibrium.

This is a quasistatic activation-continuation check. It is neither a new inverse
optimization nor a physical-time simulation.

## Why a continuation run was needed

Run 28 evaluated the same saved `q` and `Ainv` directly from the run-24 warm
displacement. That warm-seed solve converged, but the independent reset from
rest reached the 10,000-step forward limit. Run 28 therefore has a useful
candidate endpoint in
[`data/28-fiber-fat0499-replay/final.npz`](../data/28-fiber-fat0499-replay/final.npz),
but it has no completed `summary.json` and remains a failed rest-reset replay.

The continuation run starts at exact zero activation and exact zero
displacement, then solves at activation fractions 0.1 through 1.0. It uses the
same material and solver settings as run 28: stable fat with actual
`nu=0.499`, skin and muscle `nu=0.46`, forward `rtol=1e-5`, forward
`atol=1e-12`, and a strict line search with at most 30 halvings.

## Commands and run records

The working directory was
`exp/2026/09/07/face-activation-materials` at Git commit
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.

```bash
CUDA_VISIBLE_DEVICES=0 \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Fat nu .499 activation ramp' \
CHERRIES_TAGS='gpu,fat-nu0499,fiber-modes,activation-continuation' \
uv run python src/29-fat0499-activation-ramp.py
```

The completed run is [Comet experiment
`6699f8cf622b49bc9bab3b7a3827106a`](https://www.comet.com/liblaf/apple/6699f8cf622b49bc9bab3b7a3827106a).
Its measured continuation body took 627.33 seconds. The preserved terminal log
is
[`logs/29-fat0499-activation-ramp-terminal.log`](../logs/29-fat0499-activation-ramp-terminal.log).

The endpoint CPU audit used:

```bash
CUDA_VISIBLE_DEVICES='' \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Fat nu .499 continuation CPU audit' \
CHERRIES_TAGS='cpu,audit,fat-nu0499,activation-continuation' \
uv run python src/40-audit-face-results.py \
  --result-dirs data/29-fat0499-ramp \
  --output-dir data/48-fat0499-ramp-audit \
  --render true
```

The audit is [Comet experiment
`05b3e84acd6b43d583b519044d336e27`](https://www.comet.com/liblaf/apple/05b3e84acd6b43d583b519044d336e27)
and took 7.95 seconds in its measured body.

## Stage-zero attempt

The first attempt called the nonlinear solver at exact `q=0`, identity
`Ainv`, and `u=0`. The unloaded state had energy
`2.1343692326624112e-38` and gradient norm
`1.5719733639500755e-21`; the strict line search could not obtain a meaningful
decrease at that numerical scale and halted before any nonzero stage. No
tolerance was relaxed and no fallback solver was added.

The corrected run records stage zero analytically as the exact unloaded rest
reference. Strict nonlinear solves are still mandatory for every nonzero
activation fraction. The failed output and terminal evidence remain at
[`tmp/29-fat0499-ramp-stage0-failure`](../tmp/29-fat0499-ramp-stage0-failure/)
and
[`logs/29-fat0499-activation-ramp-stage0-attempt-terminal.log`](../logs/29-fat0499-activation-ramp-stage0-attempt-terminal.log).
The failed Comet run is
[`b6cd64d9c0dc4c57a002f0cae920a97d`](https://www.comet.com/liblaf/apple/b6cd64d9c0dc4c57a002f0cae920a97d).

## Continuation result

All ten nonzero stages returned `primary_success`. Every accepted stage had a
positive deformation determinant, a positive-definite activation tensor, zero
inverted tetrahedra, and exactly zero displacement on fixed vertices.

| stage | alpha | strict solver steps | fit RMS (mm) | min det(F) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | exact rest | 5.095908 | 1.000000 |
| 1 | 0.1 | 2362 | 5.072619 | 0.976473 |
| 2 | 0.2 | 2368 | 5.048898 | 0.951052 |
| 3 | 0.3 | 2352 | 5.024728 | 0.923503 |
| 4 | 0.4 | 2358 | 5.000094 | 0.893559 |
| 5 | 0.5 | 2355 | 4.974983 | 0.860933 |
| 6 | 0.6 | 2352 | 4.949380 | 0.822487 |
| 7 | 0.7 | 2343 | 4.923269 | 0.777213 |
| 8 | 0.8 | 2341 | 4.896636 | 0.725482 |
| 9 | 0.9 | 2336 | 4.869462 | 0.666534 |
| 10 | 1.0 | 2306 | 4.841729 | 0.599713 |

At the endpoint, `det(F)` ranges from 0.599713 to 1.235473, with no cells
below 0.5 and eight cells below 0.8. The minimum activation eigenvalue is
0.806226, `det(G)` has the same positive minimum as `det(F)`, and the target
projection amplitude is 0.054988. These values show a valid strict equilibrium
for the prescribed activation under the continuation path.

The final `q` and `Ainv` arrays are bitwise equal to both run 24 and run 28.
The final displacement is not bitwise equal to the run-28 warm-seed candidate,
but the difference is very small: 0.0000503 mm target-surface weighted RMS,
0.0000514 mm all-point vector RMS, and 0.0005703 mm maximum point norm. This is
evidence that both seeds reached practically the same endpoint. It does not
prove global uniqueness of the nonlinear equilibrium.

## Independent endpoint audit

The CPU audit independently recomputed deformation gradients from saved VTU
coordinates. It confirmed exact equality between the final NPZ displacement
and VTU coordinates, exact equality of active-cell `Ainv`, identity tensors on
inactive cells, and zero displacement over all 33,636 fixed vertices. Both the
complete tetrahedral boundary and visible `IsFace` skin have zero static
edge-triangle intersections at rest and at the endpoint.

On the intrinsic 10 mm mouth region, the 2 mm-scale normal high-pass RMS is
0.038684 mm for motion and 0.265849 mm for residual. These high-pass metrics are
descriptive; the target itself can contain legitimate high-frequency motion.

The minimum-determinant cell is cell 644127, a pure-muscle, actively controlled
cell in `Depressor septi001_Head_muscles_0`. Its four vertices are 132120,
131064, 106458, and 12291; vertex 132120 is fixed, and none is incident to the
artificial cut. Its principal stretches are 0.406693, 1.194168, and 1.234843.

## Outputs and reproducibility

The continuation output is
[`data/29-fat0499-ramp`](../data/29-fat0499-ramp/summary.json). It contains all
11 NPZ/VTU stage pairs, one success receipt per stage, the incremental trace,
the final endpoint, exact input and source receipts, copies of the run-28
reference metadata, and a 45-file artifact manifest. All manifest hashes and
stage readbacks were checked after the run.

The independent audit is
[`data/48-fat0499-ramp-audit`](../data/48-fat0499-ramp-audit/summary.json). Its
eight-file manifest was also rehashed after completion. The source snapshots
inside both output directories preserve the code used to produce each result.

The model has contact disabled. The collision check is static at the saved
endpoint and does not test continuous collision between continuation stages.
The ten equal increments establish this continuation path only; they do not
replace a physical loading law, an inverse fit at `nu=0.499`, or a search over
other equilibrium branches.
