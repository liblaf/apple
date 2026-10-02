# Selected neutral-checkpoint lineage visualization

`src/48-render-neutral-lineage.py` renders one branch whose ancestry is proven by
the `initial_checkpoint_sha256` stored in every neutral checkpoint. It resolves
the recorded parent path, checks the file hash, and slices each run's accepted
trace through the exact checkpoint update. Child update zero is omitted when
the prestress target is unchanged. A new target retains its initial evaluation
at the same cumulative accepted-update index and marks the target transition;
that evaluation is not counted as an optimizer update.
The same rule applies to an explicit constant-to-spatial basis transition.

This matters for the current branch: metric BFGS descends from the dedicated
Newton004 update-10 BFGS seed. The renderer therefore includes Newton004 updates
1--10 and excludes updates 11--60 from the abandoned continuation. It fails
instead of choosing a same-directory terminal checkpoint or blending branches
when a parent hash, trace endpoint, or shared coefficient does not match.

The compact figure uses cumulative accepted updates for the main x-axis and a
separate cumulative recorded wall-time panel. It marks the PNCG/SPG,
Newton-CG/SPG, Newton-CG/BFGS, and Newton-CG/metric-BFGS transitions. The panels
show objective, the `1e-3` projected-gradient threshold, skin and muscle neutral
RMS budgets, signed principal baseline stresses for all three bulk tissues, and
the prescribed skin resultant and bounded stiffness multiplier. The skin
prestress remains a model continuation proxy rather than a measurement on this
anatomy.

Run from the experiment directory without a GPU solve:

```bash
DEBUG=1 \
CHERRIES_NAME="Neutral selected-lineage visualization" \
CHERRIES_TAGS="joint-inverse,neutral,lineage,visualization" \
uv run --frozen python src/48-render-neutral-lineage.py \
  --tip-checkpoint data/neutral-convergence-010-contact-metric-bfgs-segment-003/checkpoint-0030.pt \
  --output-dir data/neutral-lineage-visuals
```

The selected tip is immutable `checkpoint-0030.pt`, rather than the running
segment's overwritten `terminal.pt`. The output directory is created with
`exist_ok=False`. `summary.json` records
every selected checkpoint, parent SHA256, source trace SHA256, local trace slice,
method regime, cumulative-update range, and the SHA256 of `selected-trace.json`.
That file freezes the exact rows used by the figure. The PNG is sized for mobile
review; the PDF keeps the same content for closer inspection.

## Rendered evidence

The local CPU render completed from immutable update 30 of the current metric
BFGS segment. The [PNG](../data/neutral-lineage-visuals/neutral-lineage.png),
[PDF](../data/neutral-lineage-visuals/neutral-lineage.pdf), and
[receipt](../data/neutral-lineage-visuals/summary.json) contain 67 accepted
updates across eight proven segments. The tip checkpoint SHA256 is
`17874553fb59b288e496833e8403ae496d28bb429508863954f46eb7a9f6dfd1`.
Newton004 contributes local updates 1--10; the receipt records 50 excluded rows.

At this frozen tip, objective is `0.2341174278`, projected-gradient infinity norm
is `0.0203546643`, skin-surface RMS is `0.114064 mm`, and muscle-centroid RMS is
`0.078545 mm`. The 40.75 minutes on the runtime panel is the sum of recorded
elapsed time for the selected slices, not the calendar duration between runs.
These values describe progress and remain short of the `1e-3` stationarity
threshold; they are not a convergence claim.

The later [converged 10% figure](../data/neutral-converged-010-lineage-visuals/neutral-lineage.png)
ends at local update 82, with 119 accepted updates on the selected branch.
Its objective is 0.2336667975 and projected-gradient infinity norm is 7.664e-5.
The stage passed its repeated stationarity, shape, and contact gates. This
establishes the 10% continuation stage only; full neutral preparation still
requires the 100% target. The [receipt](../data/neutral-converged-010-lineage-visuals/summary.json)
preserves exact hashes and selected trace slices.

The [stationary 25% figure](../data/neutral-stationary-025-lineage-visuals-v2/neutral-lineage.png)
ends at local update 87 and contains 206 accepted updates across nine segments.
The optimizer converged (projected-gradient maximum `1.067e-5`), but surface
RMS `0.275980 mm` exceeds the unchanged `0.25 mm` budget. Muscle RMS is
`0.196555 mm`; contact is valid, with minimum active gap `0.0487423 mm` and no
inverted tetrahedra. No 50% or 100% stage follows this failed preparation gate.

The renderer now uses float64 coordinate summaries from `joint_field_visuals.py`.
For spatial checkpoints, lines show unweighted anchor principal-stress means
and shaded regions show anchor ranges, explicitly not volume averages. Constant
checkpoints retain their single-tensor interpretation. Constant-to-spatial
embedding was checked against the old float64 tensor computation to `1e-12`;
the changed figure legend avoids overlapping early method-transition labels.
