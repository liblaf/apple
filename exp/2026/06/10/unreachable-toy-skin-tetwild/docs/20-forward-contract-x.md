# Forward Muscle Contraction Along X

## Purpose

Run a single forward solve on the prepared toy TetWild geometry with the muscle
activation-inverse vector set to `[-0.5, 0, 0, 0, 0, 0]` on active muscle tets.
The run keeps skin energy enabled, uses no skin prestrain, and uses the existing
PNCG forward settings from the toy experiment helper.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-toy-skin-tetwild
```

Run command:

```bash
DEBUG=1 CHERRIES_NAME=toy-forward-contract-x CHERRIES_TAGS=forward,toy,contract-x,skin,activation-inv uv run python src/25-forward-contract-muscle.py
```

## Outputs

- `data/25-forward-contract-x/contract-x.vtu`
- `data/25-forward-contract-x/summary.json`
- `logs/25-forward-contract-muscle.log`

The VTU contains `Displacement`, `DeformedPoint`, `ContractedPoint`,
`ActivationInv`, and `RecoveredActivationInv` arrays.

## Results

Forward solve succeeded in 811 PNCG steps with relative gradient
`4.95363573051845e-4`.

Key displacement metrics:

- Max displacement norm: `0.03626179741224583`
- X displacement range: `[-0.015659705170250865, 0.019692274535304397]`
- Active-cell mean X displacement: `0.002997197296237233`
- Active-cell max displacement norm: `0.03533063830102892`

Model/run details:

- Points: `71284`
- Tets: `376971`
- Active muscle tets: `10834`
- Skin energy: enabled
- Skin triangles: `28802`
- Skin thickness: `0.005`
- Skin prestrain: disabled
