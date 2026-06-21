# Forward Muscle Contraction Along X, Stronger Activation

## Purpose

Run the toy geometry forward solve with a stronger shared muscle
activation-inverse vector `[-0.9, 0, 0, 0, 0, 0]` on all active muscle tets.
This uses the prepared TetWild mesh, skin energy enabled, no skin prestrain, and
the existing PNCG forward settings.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-toy-skin-tetwild
```

Run command:

```bash
DEBUG=1 CHERRIES_NAME=toy-forward-contract-x09 CHERRIES_TAGS=forward,toy,contract-x09,skin,activation-inv uv run python src/25-forward-contract-muscle.py --activation-inv '[-0.9, 0, 0, 0, 0, 0]' --output-mesh data/25-forward-contract-x09/contract-x09.vtu --output-summary data/25-forward-contract-x09/summary.json
```

## Outputs

- `data/25-forward-contract-x09/contract-x09.vtu`
- `data/25-forward-contract-x09/summary.json`
- `logs/25-forward-contract-muscle.log`

The VTU contains `Displacement`, `DeformedPoint`, `ContractedPoint`,
`ActivationInv`, and `RecoveredActivationInv` arrays.

## Results

Forward solve succeeded in 3466 PNCG steps with relative gradient
`4.917799300255048e-4`.

Key displacement metrics:

- Max displacement norm: `0.09498931631048657`
- X displacement range: `[-0.037272740198782794, 0.0475363633336205]`
- Active-cell mean X displacement: `0.0039489733290834165`
- Active-cell max displacement norm: `0.0929454372260778`
- Fixed-point max displacement norm: `0.0`

Compared with the previous `[-0.5, 0, 0, 0, 0, 0]` run, the max displacement
norm increased from `0.03626179741224583` to `0.09498931631048657`, and the
solve became harder, increasing from 811 to 3466 PNCG steps.
