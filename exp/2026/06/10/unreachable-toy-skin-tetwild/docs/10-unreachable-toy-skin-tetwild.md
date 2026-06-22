# Unreachable Toy Skin TetWild

## Purpose

This experiment tests inverse activation on the unreachable toy box geometry:

- all: `(0, 1, 0, 0.1, 0, 1)`
- smas: `(0, 1, 0.04, 0.06, 0, 1)`
- muscle: `(0, 0.5, 0.04, 0.06, 0.4, 0.6)`

Each tetrahedron carries aponeurosis, fat, and muscle fractions. The volume
materials are aponeurosis `E = 0.10 MPa, nu = 0.35`, fat `E = 0.003 MPa,
nu = 0.49`, and active muscle `E = 0.030 MPa, nu = 0.49`. Optional skin is a
surface-triangle Koiter shell with default `E = 0.20 MPa`, `nu = 0.49`,
thickness `0.005`, and optional length prestrain. The original prestrain
setting was 10%; the later material sweep below tests smaller prestrain values
and softer skin moduli.

The inverse variable is one unrestricted six-component `ActivationInv` per
active muscle tet. The bottom and four sides are fixed. Squash moves the free
top target points by `-0.5` in `y`; stretch moves them by `+0.1` in `y`.

## Commands

Commands were run from:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-toy-skin-tetwild
```

The runs used local Cherries debug mode because the VTKHDF/VTU artifacts are
large:

```bash
DEBUG=1 CHERRIES_NAME="Toy TetWild prepare lr001" CHERRIES_TAGS="unreachable,toy,tetwild,prepare,lr001" uv run python src/10-prepare-toy-skin-tetwild.py --tetwild-lr 0.01 --output-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --output-summary data/10-meshes/lr001/summary.json
DEBUG=1 CHERRIES_NAME="Toy TetWild prepare lr0005" CHERRIES_TAGS="unreachable,toy,tetwild,prepare,lr0005" uv run python src/10-prepare-toy-skin-tetwild.py --tetwild-lr 0.005 --output-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --output-summary data/10-meshes/lr0005/summary.json
```

The initial coarse `lr=0.01` inverse runs used PNCG forward solves with `rtol=5e-4`,
`atol=1e-10`, `max_steps=5000`, and CG plus MinRes fallback adjoint solves
with `rtol=5e-4`, `atol=0`. Squash used Adam `lr=0.03` and
`inverse_loss_min_delta=5e-5`; stretch used Adam `lr=0.05` and
`inverse_loss_min_delta=5e-6`. All used `inverse_max_steps=120` or higher and
stopped by `loss_plateau_20_steps`.

The three inverse presets for each mode were:

```bash
--case-preset baseline
--case-preset skin
--case-preset skin-prestrain
```

The dense `lr=0.005` squash runs used the same three presets with
`--input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu`, Adam `lr=0.03`,
and `inverse_loss_min_delta=5e-5`:

```bash
DEBUG=1 CHERRIES_NAME="Toy TetWild squash lr0005 baseline inverse" CHERRIES_TAGS="unreachable,toy,squash,lr0005,inverse,baseline" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode squash --case-preset baseline --inverse-lr 0.03 --inverse-loss-min-delta 5e-5 --inverse-max-steps 120 --output-summary data/20-squash-lr0005/baseline/summary.json --output-table data/20-squash-lr0005/baseline/table.md
DEBUG=1 CHERRIES_NAME="Toy TetWild squash lr0005 skin inverse" CHERRIES_TAGS="unreachable,toy,squash,lr0005,inverse,skin" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode squash --case-preset skin --inverse-lr 0.03 --inverse-loss-min-delta 5e-5 --inverse-max-steps 120 --output-summary data/20-squash-lr0005/skin/summary.json --output-table data/20-squash-lr0005/skin/table.md
DEBUG=1 CHERRIES_NAME="Toy TetWild squash lr0005 skin prestrain inverse" CHERRIES_TAGS="unreachable,toy,squash,lr0005,inverse,skin-prestrain" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode squash --case-preset skin-prestrain --inverse-lr 0.03 --inverse-loss-min-delta 5e-5 --inverse-max-steps 120 --output-summary data/20-squash-lr0005/skin-prestrain/summary.json --output-table data/20-squash-lr0005/skin-prestrain/table.md
```

The dense `lr=0.005` stretch runs used Adam `lr=0.05` and
`inverse_loss_min_delta=5e-6`:

```bash
DEBUG=1 CHERRIES_NAME="Toy TetWild stretch lr0005 baseline inverse" CHERRIES_TAGS="unreachable,toy,stretch,lr0005,inverse,baseline" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode stretch --case-preset baseline --inverse-lr 0.05 --inverse-loss-min-delta 5e-6 --inverse-max-steps 160 --output-summary data/20-stretch-lr0005/baseline/summary.json --output-table data/20-stretch-lr0005/baseline/table.md
DEBUG=1 CHERRIES_NAME="Toy TetWild stretch lr0005 skin inverse" CHERRIES_TAGS="unreachable,toy,stretch,lr0005,inverse,skin" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode stretch --case-preset skin --inverse-lr 0.05 --inverse-loss-min-delta 5e-6 --inverse-max-steps 160 --output-summary data/20-stretch-lr0005/skin/summary.json --output-table data/20-stretch-lr0005/skin/table.md
DEBUG=1 CHERRIES_NAME="Toy TetWild stretch lr0005 skin prestrain inverse" CHERRIES_TAGS="unreachable,toy,stretch,lr0005,inverse,skin-prestrain" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode stretch --case-preset skin-prestrain --inverse-lr 0.05 --inverse-loss-min-delta 5e-6 --inverse-max-steps 160 --output-summary data/20-stretch-lr0005/skin-prestrain/summary.json --output-table data/20-stretch-lr0005/skin-prestrain/table.md
```

The comparison and plot commands were:

```bash
DEBUG=1 CHERRIES_NAME="Compare toy squash lr001" CHERRIES_TAGS="unreachable,toy,squash,lr001,compare,bumpiness" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --compare-existing true --output-summary data/20-squash-lr001/summary.json --output-table data/20-squash-lr001/table.md
DEBUG=1 CHERRIES_NAME="Compare toy stretch lr001" CHERRIES_TAGS="unreachable,toy,stretch,lr001,compare,bumpiness" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode stretch --compare-existing true --output-summary data/20-stretch-lr001/summary.json --output-table data/20-stretch-lr001/table.md
DEBUG=1 CHERRIES_NAME="Compare toy squash lr0005" CHERRIES_TAGS="unreachable,toy,squash,lr0005,compare,bumpiness" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode squash --compare-existing true --output-summary data/20-squash-lr0005/summary.json --output-table data/20-squash-lr0005/table.md
DEBUG=1 CHERRIES_NAME="Compare toy stretch lr0005" CHERRIES_TAGS="unreachable,toy,stretch,lr0005,compare,bumpiness" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr0005/toy-tetwild-lr0005.vtu --mode stretch --compare-existing true --output-summary data/20-stretch-lr0005/summary.json --output-table data/20-stretch-lr0005/table.md
DEBUG=1 CHERRIES_NAME="Plot toy squash lr001 losses" CHERRIES_TAGS="unreachable,toy,squash,lr001,plot,loss" uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/20-squash-lr001 --output-dir figs/20-squash-lr001
DEBUG=1 CHERRIES_NAME="Plot toy stretch lr001 losses" CHERRIES_TAGS="unreachable,toy,stretch,lr001,plot,loss" uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/20-stretch-lr001 --output-dir figs/20-stretch-lr001
DEBUG=1 CHERRIES_NAME="Plot toy squash lr0005 losses" CHERRIES_TAGS="unreachable,toy,squash,lr0005,plot,loss" uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/20-squash-lr0005 --output-dir figs/20-squash-lr0005
DEBUG=1 CHERRIES_NAME="Plot toy stretch lr0005 losses" CHERRIES_TAGS="unreachable,toy,stretch,lr0005,plot,loss" uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/20-stretch-lr0005 --output-dir figs/20-stretch-lr0005
```

After inspecting the squash histories, the `5e-5` squash threshold was too
large: the loss was still decreasing when the runs stopped. The follow-up
coarse skin+prestrain squash run used Adam `lr=1.0` and
`inverse_loss_min_delta=1e-8` to force a real plateau check:

```bash
DEBUG=1 CHERRIES_NAME="Toy TetWild squash lr001 prestrain lr1 long" CHERRIES_TAGS="unreachable,toy,squash,lr001,inverse,skin-prestrain,lr-tune,long" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --inverse-lr 1.0 --inverse-max-steps 120 --inverse-loss-min-delta 1e-8 --initial-activation-inv "[0,0,0,0,0,0]" --require-convergence false --output-summary data/21-squash-tuning-lr001/prestrain-lr1-steps120/summary.json --output-table data/21-squash-tuning-lr001/prestrain-lr1-steps120/table.md
DEBUG=1 CHERRIES_NAME="Plot toy squash lr001 tuned prestrain loss" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-prestrain,tune,plot" uv run python src/40-plot-lr-sweep-loss-curves.py --input-dir data/21-squash-tuning-lr001 --output-dir figs/21-squash-tuning-lr001
```

The skin material sweep kept the coarse `lr=0.01` TetWild mesh, Adam `lr=0.5`,
`inverse_max_steps=40`, `inverse_loss_min_delta=1e-6`, skin thickness `0.005`,
and compared skin Young's modulus and prestrain:

```bash
DEBUG=1 CHERRIES_NAME="Toy squash skin sweep e020_p00" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-sweep,e020_p00" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --skin-e 0.20 --skin-thickness 0.005 --skin-prestrain 0.00 --inverse-lr 0.5 --inverse-max-steps 40 --inverse-loss-min-delta 1e-6 --require-convergence false --output-summary data/22-skin-param-sweep-lr001/e020_p00/summary.json --output-table data/22-skin-param-sweep-lr001/e020_p00/table.md
DEBUG=1 CHERRIES_NAME="Toy squash skin sweep e020_p02" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-sweep,e020_p02" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --skin-e 0.20 --skin-thickness 0.005 --skin-prestrain 0.02 --inverse-lr 0.5 --inverse-max-steps 40 --inverse-loss-min-delta 1e-6 --require-convergence false --output-summary data/22-skin-param-sweep-lr001/e020_p02/summary.json --output-table data/22-skin-param-sweep-lr001/e020_p02/table.md
DEBUG=1 CHERRIES_NAME="Toy squash skin sweep e020_p05" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-sweep,e020_p05" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --skin-e 0.20 --skin-thickness 0.005 --skin-prestrain 0.05 --inverse-lr 0.5 --inverse-max-steps 40 --inverse-loss-min-delta 1e-6 --require-convergence false --output-summary data/22-skin-param-sweep-lr001/e020_p05/summary.json --output-table data/22-skin-param-sweep-lr001/e020_p05/table.md
DEBUG=1 CHERRIES_NAME="Toy squash skin sweep e010_p02" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-sweep,e010_p02" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --skin-e 0.10 --skin-thickness 0.005 --skin-prestrain 0.02 --inverse-lr 0.5 --inverse-max-steps 40 --inverse-loss-min-delta 1e-6 --require-convergence false --output-summary data/22-skin-param-sweep-lr001/e010_p02/summary.json --output-table data/22-skin-param-sweep-lr001/e010_p02/table.md
DEBUG=1 CHERRIES_NAME="Toy squash skin sweep e005_p02" CHERRIES_TAGS="unreachable,toy,squash,lr001,skin-sweep,e005_p02" uv run python src/20-inverse-toy-skin-tetwild.py --input-mesh data/10-meshes/lr001/toy-tetwild-lr001.vtu --mode squash --case-preset skin-prestrain --skin-e 0.05 --skin-thickness 0.005 --skin-prestrain 0.02 --inverse-lr 0.5 --inverse-max-steps 40 --inverse-loss-min-delta 1e-6 --require-convergence false --output-summary data/22-skin-param-sweep-lr001/e005_p02/summary.json --output-table data/22-skin-param-sweep-lr001/e005_p02/table.md
```

## Outputs

- Prepared meshes: `data/10-meshes/lr001/`, `data/10-meshes/lr0005/`
- Squash summaries/table/results/history: `data/20-squash-lr001/`
- Dense squash summaries/table/results/history: `data/20-squash-lr0005/`
- Stretch summaries/table/results/history: `data/20-stretch-lr001/`
- Dense stretch summaries/table/results/history: `data/20-stretch-lr0005/`
- Tuned coarse squash skin+prestrain run: `data/21-squash-tuning-lr001/`
- Coarse squash skin material sweep: `data/22-skin-param-sweep-lr001/`
- Coarse squash `E=0.20 MPa`, `5%` prestrain confirmation:
  `data/23-skin-param-confirm-lr001/e020_p05_lr05/`
- Log-y loss plots: `figs/20-squash-lr001/`, `figs/20-squash-lr0005/`,
  `figs/20-stretch-lr001/`, `figs/20-stretch-lr0005/`,
  `figs/21-squash-tuning-lr001/`

Each inverse case writes one final `.vtu`, one target `.vtu`, one per-case
summary JSON, and one temporal `.vtkhdf` containing every inverse evaluation.

## Results

The coarse `lr=0.01` comparison summaries are complete: three expected cases,
zero missing, and zero extras for squash and stretch.

| mode | case | frames | best step | best loss | error RMS | error/target | top y mean | top y std | residual edge RMS | residual lap RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squash | no skin | 23 | 22 | 0.0826318 | 0.497891 | 0.995782 | -0.002165 | 0.004581 | 0.087941 | 0.037638 |
| squash | skin | 23 | 22 | 0.0826969 | 0.498087 | 0.996174 | -0.001926 | 0.003173 | 0.087981 | 0.037635 |
| squash | skin + prestrain | 21 | 20 | 0.0828307 | 0.498490 | 0.996980 | -0.001513 | 0.001748 | 0.088023 | 0.037633 |
| stretch | no skin | 51 | 50 | 0.00285015 | 0.092469 | 0.924687 | 0.012302 | 0.026477 | 0.017846 | 0.007714 |
| stretch | skin | 55 | 54 | 0.00258724 | 0.088101 | 0.881006 | 0.017013 | 0.028786 | 0.017643 | 0.007539 |
| stretch | skin + prestrain | 52 | 51 | 0.00270624 | 0.090104 | 0.901040 | 0.011009 | 0.014104 | 0.017537 | 0.007530 |

Squash is essentially unreachable: all cases leave about 99.6% of the target
RMS error. Adding skin reduces top-surface variation, and prestrain gives the
smoothest top displacement, but fit worsens slightly.

The follow-up squash skin+prestrain rerun confirms the original run stopped too
early. Increasing Adam to `lr=1.0` and lowering `inverse_loss_min_delta` to
`1e-8` ran 119 evaluations and stopped by `loss_plateau_20_steps` with the true
best at step 98. It moves the top farther toward the `-0.5` target, but the
extra motion is bumpier:

| run | frames | best step | best loss | error/target | top y mean | top y std | top y min | top y max | top y range | displacement edge RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original skin + prestrain | 21 | 20 | 0.0828307 | 0.996980 | -0.001513 | 0.001748 | -0.007374 | 0.000313 | 0.007688 | 0.000159 |
| tuned skin + prestrain | 119 | 98 | 0.0809527 | 0.985613 | -0.007364 | 0.006191 | -0.023532 | 0.001956 | 0.025488 | 0.000843 |

This tuned run is the better target-fit result, but it is no longer the
smoothest-looking one. The geometry still appears strongly unreachable: even
after the longer solve it leaves about 98.6% of the target RMS error.

Because the toy box height is only `0.1`, the skin thickness `0.005` is 5% of
the box height. That is thick for a geometry-scaled membrane if the goal is
free target tracking, but it is reasonable as an explicit smoothing shell for
this unreachable toy problem. The first material adjustment should therefore be
reducing prestrain, not immediately making the shell much softer or thinner.

The coarse squash skin material sweep confirms that conclusion. All five runs
used the same optimizer budget and stopped by `step_limit`, so the table is a
fixed-budget comparison rather than a convergence ranking. The best smooth-first
setting in this sweep is `E = 0.20 MPa`, thickness `0.005`, prestrain `5%`.
It has the lowest top-surface range, lowest displacement edge RMS, lowest
near-muscle top-y standard deviation, and nearly the same near-muscle target
fraction as the rougher alternatives.

| skin E MPa | prestrain | best loss | error/target | top y std | top y range | displacement edge RMS | near-muscle y mean | near-muscle target fraction | near-muscle y std | activation max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0% | 0.0819490 | 0.991659 | 0.005837 | 0.061551 | 0.001640 | -0.015291 | 0.030583 | 0.012746 | 17.93 |
| 0.20 | 2% | 0.0819519 | 0.991677 | 0.005888 | 0.042664 | 0.000878 | -0.016584 | 0.033167 | 0.010893 | 15.14 |
| 0.20 | 5% | 0.0819259 | 0.991519 | 0.005481 | 0.033999 | 0.000640 | -0.016250 | 0.032499 | 0.008214 | 14.45 |
| 0.10 | 2% | 0.0818589 | 0.991114 | 0.007435 | 0.063206 | 0.001379 | -0.019789 | 0.039578 | 0.015603 | 16.52 |
| 0.05 | 2% | 0.0818701 | 0.991181 | 0.007964 | 0.072564 | 0.001721 | -0.020982 | 0.041965 | 0.017285 | 15.62 |

Lowering skin `E` from `0.20` to `0.10` or `0.05 MPa` increases local
near-muscle motion, but only from about 3.2% of the requested displacement to
about 4.0-4.2%, while top range and edge roughness roughly double relative to
the `0.20 MPa, 5%` run. Pointwise near-muscle minima show the same tradeoff:
the softest `0.05 MPa, 2%` run reaches `-0.0686`, while the smoother
`0.20 MPa, 5%` run reaches `-0.0333`. For this smooth-first toy inverse setup,
that is not a good trade.

A longer confirmation run with `E = 0.20 MPa`, thickness `0.005`, prestrain
`5%`, Adam `lr=0.5`, `inverse_max_steps=120`, and
`inverse_loss_min_delta=1e-6` improved local target motion while staying
smoother than the softer-skin alternatives by top range and displacement edge
RMS. It reached the step cap rather than a plateau, with the true lowest-loss
step at 119:

| run | best step | best loss | error/target | top y std | top y range | displacement edge RMS | near-muscle y mean | near-muscle y min | near-muscle target fraction | activation max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E=0.20`, `5%`, 40 steps | 40 | 0.0819259 | 0.991519 | 0.005481 | 0.033999 | 0.000640 | -0.016250 | -0.033262 | 0.032499 | 14.45 |
| `E=0.20`, `5%`, 120-step cap | 119 | 0.0812873 | 0.987647 | 0.008981 | 0.055574 | 0.001228 | -0.027335 | -0.052327 | 0.054670 | 67.23 |

The confirmation run is less smooth than the 40-step snapshot, but it deforms
substantially more toward the target and remains smoother by top range and edge
RMS than the `0.10 MPa` and `0.05 MPa` material sweep points. If more local
travel is still needed, sweep thickness `0.002-0.003` before dropping skin `E`
below `0.10 MPa`.

Stretch is more responsive. Skin without prestrain gives the best target fit,
while skin with prestrain gives the lowest top-y variation and lowest residual
Laplacian RMS among the stretch cases.

The dense `lr=0.005` squash comparison is also complete: three expected cases,
zero missing, zero extras, and all three cases stopped with
`loss_plateau_20_steps`.

| lr | mode | case | frames | best step | best loss | error RMS | error/target | top y mean | top y std | residual edge RMS | residual lap RMS |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | squash | no skin | 21 | 20 | 0.0829338 | 0.498800 | 0.997600 | -0.001217 | 0.001709 | 0.066133 | 0.028350 |
| 0.005 | squash | skin | 21 | 20 | 0.0829373 | 0.498811 | 0.997621 | -0.001194 | 0.002005 | 0.066140 | 0.028351 |
| 0.005 | squash | skin + prestrain | 21 | 20 | 0.0830145 | 0.499043 | 0.998085 | -0.000959 | 0.001076 | 0.066139 | 0.028345 |

At the denser resolution, squash is still unreachable. Refining from
`lr=0.01` to `lr=0.005` lowered the residual roughness metrics because the
surface sampling is denser, but it did not materially improve target
reachability: all dense squash cases still leave about 99.8% of the requested
top displacement error. The prestrain variant remains the smoothest by top-y
standard deviation and residual Laplacian RMS, while the no-skin variant has
the slightly lower target loss.

The dense `lr=0.005` stretch comparison is complete as well: three expected
cases, zero missing, zero extras, and all three cases stopped with
`loss_plateau_20_steps`.

| lr | mode | case | frames | best step | best loss | error RMS | error/target | top y mean | top y std | residual edge RMS | residual lap RMS |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | stretch | no skin | 77 | 76 | 0.00280802 | 0.091783 | 0.917827 | 0.013314 | 0.026779 | 0.013206 | 0.005671 |
| 0.005 | stretch | skin | 77 | 76 | 0.00243605 | 0.085488 | 0.854877 | 0.021812 | 0.033750 | 0.013250 | 0.005684 |
| 0.005 | stretch | skin + prestrain | 80 | 79 | 0.00252674 | 0.087064 | 0.870644 | 0.014823 | 0.018002 | 0.013214 | 0.005683 |

At the denser stretch resolution, the skin/no-prestrain case gives the lowest
target loss and smallest RMS error, but it is the bumpiest by top-y standard
deviation. The skin+prestrain case is the smoothest in top-y standard
deviation while keeping better target fit than the no-skin case. The residual
edge and residual Laplacian RMS differences are small across the three dense
stretch cases.

## Dense Mesh Feasibility

TetWild `lr=0.005` completed, producing 500,664 points, 2,766,827 tets, 69,870
active muscle tets, and 419,220 activation dofs. The full three-case squash
matrix completed in 275.0 s, 393.3 s, and 258.7 s respectively. The full
three-case stretch matrix completed with 77, 77, and 80 inverse evaluations; its
per-case temporal histories occupy about 11 GB total.

## Validation

Validation checks confirmed:

- all six coarse inverse summaries have `inverse/converged=true`;
- all six coarse runs stopped with `loss_plateau_20_steps`;
- all three dense `lr=0.005` squash summaries have `inverse/converged=true`;
- all three dense `lr=0.005` stretch summaries have `inverse/converged=true`;
- `best/step` is the true lowest-loss step from each trace;
- each checked VTKHDF history frame count equals `history/frames` and
  `inverse/evaluations`;
- the comparison summaries report complete matrices for squash and stretch;
- the dense `lr=0.005` squash comparison summary reports a complete matrix;
- the dense `lr=0.005` stretch comparison summary reports a complete matrix;
- the tuned coarse squash skin+prestrain VTKHDF has 119 frames matching
  `history/frames` and `inverse/evaluations`;
- each skin material sweep VTKHDF has 41 saved steps matching the 41 inverse
  evaluations;
- the `E=0.20 MPa`, `5%` prestrain confirmation VTKHDF has 121 saved steps
  matching the 121 inverse evaluations;
- log-y loss plots were generated for the coarse squash/stretch matrices, the
  dense squash/stretch matrices, and the tuned squash skin+prestrain rerun.
