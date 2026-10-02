# Stage 3 transition preparation

## Purpose

Prepare the fixed-axis Stage 3 MouthOpen and Smile controls for fresh collision-on and collision-off forward transition solves. The repaired fixture remains an external input.

## Command

```bash
cd exp/2026/09/30/stage3-mouthopen-smile-contact-comparison
CHERRIES_NAME='Stage 3 endpoint preparation' \
CHERRIES_TAGS='stage3,mouthopen,smile,contact,provenance,cpu' \
DEBUG=1 uv run python src/10-prepare-stage3.py
```

The profile disables Git commits. This CPU-only preparation did not run a solver or an inverse fit.

## Results

`data/10-stage3/mesh.npz` is a byte copy of the repaired-reference mesh. `endpoints.npz` contains direct Stage 3 `rankone_fixed` active-strain tensors, scalar strengths, and fixed axes for 288,172 pruned active cells. The Smile source has 288,235 active cells; 63 fully fixed cells are removed through the verified `OriginalCellId` mapping.

The Smile checkpoint has `solver_valid=false`. It is preserved as a historical control source and every Stage 3 state must be freshly re-equilibrated. The MouthOpen checkpoint has `solver_valid=true`.

`seed.npz` stores the audited Stage 4 frame-000 displacement and jaw pose as geometry-only warm-start data. It supplies no activation tensor to the new controls.

## Outputs

- `mesh.npz`: `43a6c25895d81b6ad0412a49056b5749f16a71d70caeacb03d7f16abc254b93e`
- `endpoints.npz`: `279ffa22b2a72d81165abcf0d9a01cbe7508119ec9bda01d74e1e10b71c07348`
- `seed.npz`: `6c2dd18958996518610870b84604a65de212bacf40cc9bb053e13d6282fd7d1f`

`data/10-stage3/summary.json` records every input receipt and per-array digest.
