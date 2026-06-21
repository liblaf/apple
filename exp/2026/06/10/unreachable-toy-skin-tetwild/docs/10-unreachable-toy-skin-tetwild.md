# Unreachable Toy Skin TetWild

## Purpose

This experiment builds the unreachable inverse-physics toy problem
under `exp/2026/06/10/unreachable-toy-skin-tetwild/`. Mesh preparation and
inverse solves are split so that TetWild setup stays separate from the per-case
CUDA solve:

- `src/10-prepare-toy-skin-tetwild.py` prepares one TetWild mesh.
- `src/20-inverse-toy-skin-tetwild.py` runs one selected inverse case per CLI
  invocation, or rebuilds the comparison table with `--compare-existing true`.

Every tetrahedron carries aponeurosis, fat, and muscle fractions. Material
values are `E = 0.10 MPa, nu = 0.35` for aponeurosis, `E = 0.003 MPa,
nu = 0.49` for fat, and `E = 0.030 MPa, nu = 0.49` for active muscle. The skin
is modeled as the surface triangle mesh with `E = 0.20 MPa`, `nu = 0.49`,
thickness `0.005`, and optional 10% tensile prestrain.

TetWild is called through `melon.ext.tetwild(surface, edge_length_fac=0.01)`;
`edge_length_fac` is relative. The toy boxes are:

- all: `(0, 1, 0, 0.1, 0, 1)`
- smas: `(0, 1, 0.04, 0.06, 0, 1)`
- muscle: `(0, 0.5, 0.04, 0.06, 0.4, 0.6)`

The bottom surface and four sides are fixed. The squash target moves free
top-surface points by `-0.05` in `y`; the visual stretch probe moves them by
`+0.1` in `y`.

## Commands

All commands are run from:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-toy-skin-tetwild
```

Mesh preparation:

```bash
DEBUG=1 CHERRIES_NAME="Prepare toy TetWild skin mesh" CHERRIES_TAGS="unreachable,toy,skin,tetwild,prepare,lr001" uv run python src/10-prepare-toy-skin-tetwild.py
```

Inverse solves use PNCG forward with `rtol = 5e-4`, `atol = 1e-10`, and
`max_steps = 5000`. The adjoint linear solve uses CG with MinRes fallback; CG is
configured with `rtol = 5e-4` and `atol = 0`. Adam is used for inverse
optimization.

Each inverse case is launched as its own CLI invocation:

```bash
DEBUG=1 CHERRIES_NAME="Toy squash <loss> <prestrain> <activation>" CHERRIES_TAGS="<tags>" uv run python src/20-inverse-toy-skin-tetwild.py --mode squash --loss-variant <l2|laplacian> --skin-prestrain-enabled <true|false> --activation-mode <per-tet|per-tet-smooth|shared>
```

For shared activation cases, append:

```bash
--inverse-lr 0.003
```

The completed inverse matrix has 12 squash cases:

- `--loss-variant l2` and `--loss-variant laplacian`
- `--skin-prestrain-enabled false` and `true`
- `--activation-mode per-tet`, `per-tet-smooth`, and `shared`

Per-tet modes optimize 10,834 six-DoF `ActivationInv` blocks
(65,004 scalar dofs). Shared mode optimizes one six-DoF `ActivationInv` block.

The comparison-only pass is:

```bash
DEBUG=1 CHERRIES_NAME="Compare toy squash bumpiness" CHERRIES_TAGS="unreachable,toy,squash,compare,bumpiness,lr001,activation-modes" uv run python src/20-inverse-toy-skin-tetwild.py --compare-existing true
```

## Outputs

Mesh preparation wrote:

- `data/10-toy-tetwild-lr001-prepared.vtu`
- `data/10-toy-tetwild-lr001-prepared-summary.json`

The prepared mesh has 71,284 points, 376,971 tetrahedra, 10,834 active muscle
tets, and 5,927 target top-surface points. Fraction sums are exactly 1.0 in the
summary.

Each inverse run writes:

- one final result mesh, `data/20-toy-tetwild-squash-*.vtu`
- one target mesh, `data/20-toy-tetwild-squash-*-target.vtu`
- one per-case summary, `data/20-toy-tetwild-squash-*-summary.json`
- one temporal VTKHDF file, `data/20-toy-tetwild-squash-*-steps.vtkhdf`,
  containing every inverse evaluation

The comparison pass writes:

- `data/20-unreachable-toy-skin-tetwild-summary.json`
- `data/20-unreachable-toy-skin-tetwild-table.md`

Loss-curve plotting writes:

- `figs/30-loss-curves/loss-vs-step-all-cases.png`
- one per-case plot under `figs/30-loss-curves/*-loss-vs-step.png`

No CSV artifacts are produced by this experiment.

## Results

All 12 squash inverse cases completed with `inverse/converged = true` and
stopped with `loss_plateau_20_steps`. The comparison summary reports
`complete = true`, 12 expected cases, 12 observed expected cases, and no missing
or extra case summaries.

| residual Laplacian | skin prestrain | activation mode | frames | best step | best loss | error RMS | error/target | top y std | residual lap RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| false | false | per-tet | 77 | 56 | 0.000776830 | 0.0482751 | 0.965503 | 0.00245395 | 0.00376430 |
| false | false | per-tet-smooth | 79 | 58 | 0.000782370 | 0.0484361 | 0.968722 | 0.00209523 | 0.00376399 |
| false | false | shared | 94 | 73 | 0.000802283 | 0.0490597 | 0.981193 | 0.00138682 | 0.00376420 |
| false | true | per-tet | 21 | 0 | 0.000833527 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |
| false | true | per-tet-smooth | 21 | 0 | 0.000833527 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |
| false | true | shared | 21 | 0 | 0.000833527 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |
| true | false | per-tet | 94 | 73 | 0.000909839 | 0.0481491 | 0.962982 | 0.00254559 | 0.00376307 |
| true | false | per-tet-smooth | 79 | 58 | 0.000919498 | 0.0484356 | 0.968712 | 0.00208291 | 0.00376385 |
| true | false | shared | 91 | 70 | 0.000940465 | 0.0490836 | 0.981672 | 0.00135785 | 0.00376422 |
| true | true | per-tet | 21 | 0 | 0.000971321 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |
| true | true | per-tet-smooth | 21 | 0 | 0.000971321 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |
| true | true | shared | 21 | 0 | 0.000971321 | 0.0500058 | 1.00012 | 0.0000771191 | 0.00376274 |

Without skin prestrain, per-tet activation reaches the lowest target error but
has the bumpiest top surface by `top_y_std`. Adding activation smoothness keeps
nearly the same residual Laplacian RMS while reducing top-surface variation.
Shared activation gives the smoothest top surface among no-prestrain runs, but
with the largest target error.

With 10% skin prestrain, all activation modes plateau at the initial best step.
The top surface is already very smooth, but the target error remains essentially
the full unreachable squash displacement.

## Loss Curves

The loss curves were generated from each case's `*-steps.vtkhdf` file using
`src/30-plot-loss-curves.py`.

![Loss vs step for all squash cases](../figs/30-loss-curves/loss-vs-step-all-cases.png)

## Learning Rate Probe

A follow-up probe reran the L2, no-skin-prestrain, per-tet `ActivationInv`
case with unrestricted six-vector activations. The inverse loop now records the
true lowest-loss step as `best/step`; the plateau stop uses a private
min-delta reference for 20 consecutive non-improving steps without recording a
separate "last significant improvement" artifact.

Commands:

```bash
CHERRIES_NAME=toy-skin-tetwild-lr-sweep CHERRIES_TAGS=lr030 COMET_LOG_GIT_PATCH=false uv run python src/20-inverse-toy-skin-tetwild.py --mode squash --loss-variant l2 --skin-prestrain-enabled false --activation-mode per-tet --inverse-lr 0.3 --inverse-loss-min-delta 1e-8 --inverse-max-steps 80 --require-convergence false --output-summary data/41-lr-sweep/lr030/summary.json --output-table data/41-lr-sweep/lr030/table.md
CHERRIES_NAME=toy-skin-tetwild-lr-sweep CHERRIES_TAGS=lr060 COMET_LOG_GIT_PATCH=false uv run python src/20-inverse-toy-skin-tetwild.py --mode squash --loss-variant l2 --skin-prestrain-enabled false --activation-mode per-tet --inverse-lr 0.6 --inverse-loss-min-delta 1e-8 --inverse-max-steps 50 --require-convergence false --output-summary data/41-lr-sweep/lr060/summary.json --output-table data/41-lr-sweep/lr060/table.md
CHERRIES_NAME=toy-skin-tetwild-lr-sweep-plots CHERRIES_TAGS=plot uv run python src/40-plot-lr-sweep-loss-curves.py
```

Summary:

| lr | evaluations | stop | best step | best loss | final loss | final plateau steps | activation RMS | activation max abs |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 81 | step_limit | 79 | 0.000732421 | 0.000732922 | 1 | 1.32879 | 8.19763 |
| 0.6 | 51 | step_limit | 50 | 0.000730082 | 0.000730082 | 0 | 1.64953 | 10.9414 |

`lr = 0.6` is the better candidate from this probe: it reaches a lower loss in
50 evaluations than `lr = 0.3` reaches in 81. Both curves still hit the
exploratory step cap rather than the 20-step plateau stop, so the case has not
visibly converged under these caps.

![Log loss vs step for LR sweep](../figs/41-lr-sweep/l2-no-prestrain-per-tet-lr-sweep-log-loss.png)

## Thin-Skin Stretch Probe

The stretch visual case was rerun after changing the toy shell thickness from
`1.0` to `0.005`. The case is still L2-only, no skin prestrain, per-tet
`ActivationInv`, target displacement `+0.1` in `y`, and `lr = 0.05`.

Command:

```bash
DEBUG=1 CHERRIES_NAME=toy-skin-tetwild-stretch-thin-skin CHERRIES_TAGS=stretch,thin-skin,thickness005,lr005 uv run python src/20-inverse-toy-skin-tetwild.py --mode stretch --loss-variant l2 --skin-prestrain-enabled false --activation-mode per-tet --inverse-lr 0.05 --inverse-loss-min-delta 1e-8 --inverse-max-steps 120 --require-convergence false --output-summary data/43-stretch-thin-skin/lr005/summary.json --output-table data/43-stretch-thin-skin/lr005/table.md
DEBUG=1 CHERRIES_NAME=toy-skin-tetwild-stretch-thin-skin-plots CHERRIES_TAGS=stretch,thin-skin,plot,thickness005 uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/43-stretch-thin-skin --output-dir figs/43-stretch-thin-skin
```

Summary:

| skin thickness | lr | evaluations | stop | best step | best loss | error RMS | activation RMS | activation max abs | mean top disp y | max top disp y |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.05 | 81 | step_limit | 80 | 0.00279729 | 0.0916071 | 0.361160 | 1.41934 | 0.00887135 | 0.0345964 |
| 0.005 | 0.05 | 121 | step_limit | 120 | 0.00254153 | 0.0873188 | 0.424075 | 2.17600 | 0.0185202 | 0.109263 |

The thin skin lowers the loss by about 9.1% relative to the earlier
`thickness = 1.0`, `lr = 0.05`, step-80 run. Mean target-surface displacement
roughly doubles, and the maximum target-surface displacement reaches the
requested `+0.1` scale, so this is more visually useful. The case still stops
at the exploratory step cap while loss is decreasing, so it has not converged.

![Thin skin stretch log loss](../figs/43-stretch-thin-skin/l2-no-prestrain-per-tet-lr-sweep-log-loss.png)

## Validation

Validation commands run successfully:

```bash
PYTHONPYCACHEPREFIX=/tmp/apple-pycache-check uv run python -m py_compile src/_toy_skin_tetwild.py src/10-prepare-toy-skin-tetwild.py src/20-inverse-toy-skin-tetwild.py
uv run ruff check src/_toy_skin_tetwild.py src/10-prepare-toy-skin-tetwild.py src/20-inverse-toy-skin-tetwild.py
PYTHONPYCACHEPREFIX=/tmp/apple-pycache-plot DEBUG=1 CHERRIES_NAME="Plot toy squash loss curves" CHERRIES_TAGS="unreachable,toy,squash,loss-curves,lr001" uv run python src/30-plot-loss-curves.py
```

Additional checks confirmed:

- every squash VTKHDF temporal file has the same frame count as
  `history/frames` and `inverse/evaluations`;
- the loss-curve plotting run wrote 13 PNG assets;
- no stretch artifacts remain in the experiment directory;
- no CSV artifacts are present;
- only `unreachable-toy-skin-tetwild` remains under `exp/2026/06/10/`.
