# Raw6 accepted step-8 diagnostic endpoint

## Status

This is an exact export of the last accepted equilibrium from the interrupted
Raw6 run. Its status is `diagnostic_stop_not_converged`; it is not a completed
optimization. The saved KKT value is 12.3252877152, far above the configured
0.001 tolerance. No fresh-rest branch check was performed.

The run was stopped after two 10,000-step inner failures at outer step 7 and
subsequent difficult adjoint behavior. The endpoint exported here is the later
accepted outer-step-7 trial 3, persisted as checkpoint and trace step 8.

## Export command

From `exp/2026/09/07/face-activation-materials`:

```bash
CUDA_VISIBLE_DEVICES='' DEBUG=1 \
  CHERRIES_NAME='Raw6 accepted step 8 diagnostic export final local' \
  CHERRIES_TAGS='cpu,raw6,diagnostic-stop,accepted-endpoint,final-local' \
  uv run python src/23-export-raw6-diagnostic-step8.py
```

The CPU-only export completed in 5.3 seconds with zero physics solves.
`final.npz` is a byte-for-byte copy of
`data/23-raw6-fat049/diagnostic-stop-checkpoint.npz`; both have SHA-256
`5720b92af4167f5d5a288a251443c10a9980d04ef372262095e148505cc90e65`.
The reconstructed `final.vtu` passed an exact read-back check for topology,
coordinates, and every retained/generated array.

## Independent endpoint checks

NumPy and PyVista recomputed the target weights, fit, full-volume deformation
gradients, and active-cell activation diagnostics directly from the fixture and
checkpoint. This path does not import the solver or material implementation.

| Quantity | Step 8 |
| --- | ---: |
| Target RMS | 5.095908 mm |
| Fit RMS | 3.459011 mm |
| Motion RMS | 2.354488 mm |
| Target projection amplitude | 0.376366 |
| Projection residual / target RMS | 0.268002 |
| Minimum / maximum det(F) | 0.349748 / 3.341307 |
| Inverted tetrahedra | 0 |
| Minimum det(G) | 0.105126 |
| Activation eigenvalue range | 0.220374 to 1.801528 |
| Activation determinant range | 0.235356 to 2.555969 |
| Fixed readback maximum | 0 m |

All compared values match trace row 8 within absolute tolerance `1e-12` and
relative tolerance `5e-12`. The export also preserves the original config,
stop receipt, trace, trial receipts, material and optimizer source snapshots,
and their provenance hashes in `data/23-raw6-diagnostic-step8/`.

## Audit 40

```bash
CUDA_VISIBLE_DEVICES='' DEBUG=1 \
  CHERRIES_NAME='Raw6 diagnostic step 8 CPU audit' \
  CHERRIES_TAGS='cpu,raw6,diagnostic-stop,step8,audit' \
  uv run python src/40-audit-face-results.py \
  --result-dirs data/23-raw6-diagnostic-step8 \
  --output-dir data/46-raw6-step8-audit --render true
```

The audit completed in 9.6 seconds. It independently reproduced saved det(F)
with zero maximum absolute error. There are 129 cells below det(F)=0.5 and
23,417 below 0.8; 93.68% of the below-0.8 cells belong to the selected
expression set. The minimum is cell 658114, a Depressor septi muscle cell, at
det(F)=0.349748. The lowest 1,000 cells are 97.6% selected-expression cells.

IPC Toolkit 1.6 found no static endpoint intersections on either the complete
tet boundary or the visible `IsFace` skin. This is an endpoint-only check and
does not provide continuous collision detection over the optimization path.

At a 2 mm cotangent heat scale, full-face normal-motion high-pass RMS is
0.0769 mm and the 10 mm mouth-region value is 0.1488 mm. The corresponding
residual values are 0.0865 mm and 0.1791 mm. These are descriptive spatial
frequency summaries; the target itself contains real high-frequency content,
so they do not identify artifacts on their own.

All audit maps, metrics, rendered figures, source snapshots, and hashes are in
`data/46-raw6-step8-audit/`.
