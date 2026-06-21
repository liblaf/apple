# Forward Muscle Contraction Along X, Positive Inverse Activation

## Purpose

Run the toy geometry forward solve with shared muscle activation-inverse vector
`[2.0, 0, 0, 0, 0, 0]` on all active muscle tets. This uses the prepared
TetWild mesh, skin energy enabled, no skin prestrain, and the existing PNCG
forward settings.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-toy-skin-tetwild
```

Run command:

```bash
DEBUG=1 CHERRIES_NAME=toy-forward-contract-xp20 CHERRIES_TAGS=forward,toy,contract-xp20,skin,activation-inv uv run python src/25-forward-contract-muscle.py --activation-inv '[2.0, 0, 0, 0, 0, 0]' --output-mesh data/25-forward-contract-xp20/contract-xp20.vtu --output-summary data/25-forward-contract-xp20/summary.json
```

## Outputs

- `data/25-forward-contract-xp20/contract-xp20.vtu`
- `data/25-forward-contract-xp20/summary.json`
- `logs/25-forward-contract-muscle.log`

The VTU contains `Displacement`, `DeformedPoint`, `ContractedPoint`,
`ActivationInv`, and `RecoveredActivationInv` arrays.

## Results

Forward solve succeeded in 258 PNCG steps with relative gradient
`4.926661798557514e-4`.

Key displacement metrics:

- Max displacement norm: `0.07095613989106005`
- X displacement range: `[-0.07088697766206382, 0.0007006683243225368]`
- Active-cell mean X displacement: `-0.021491436950600726`
- Active-cell max displacement norm: `0.06937173148600075`
- Fixed-point max displacement norm: `0.0`

Compared with the earlier negative x activation-inverse cases, this positive
inverse activation produced mostly negative x displacement and converged faster:
258 PNCG steps here versus 811 for `[-0.5, 0, 0, 0, 0, 0]` and 3466 for
`[-0.9, 0, 0, 0, 0, 0]`.
