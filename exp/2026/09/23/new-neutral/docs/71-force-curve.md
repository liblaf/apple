# Force curve for the continued neutral

The force review (private preview omitted) shows the full saved force
history and an enlarged Newton-phase view, both on logarithmic force axes.
The final residual is **0.009250543 N**, below the unchanged **0.01 N** threshold.
The figure keeps the geometry warning visible: two inverted tetrahedra prevent
accepting this force-converged endpoint as a valid neutral.

The 474 plotted states combine parent `forward-repaired-reference-001`
(initial + 332 PNCG + 100 Newton steps) with the 41 Newton steps in
`forward-repaired-reference-005`. The duplicate continuation initial state is
removed after verifying its force, energy and barrier stiffness against the
parent endpoint. No solve was rerun and no samples were smoothed.

The plot marks the barrier stiffness doubling at accepted step 193, the
PNCG-to-Newton transition at 332, and the continuation/shift-policy change at
432. The stiffness event coincides with a force increase from 5.778516 to
29.219880 N. The shared restart force is 0.159060385 N. Recorded residuals
use their recorded barrier stiffness; the curve breaks across a stiffness
change. Raw forces in MPa·m² are multiplied by 1e6 to obtain newtons.

Artifacts are stored under
[`data/review-repaired-reference-005/force/`](../data/review-repaired-reference-005/force/):
PNG, SVG, PDF, raw samples in CSV, a provenance receipt and an HTML page.
The main preview links the force detail and displays the full curve.

Run from `exp/2026/09/23/new-neutral`:

```bash
CHERRIES_NAME='Neutral force history with Newton continuation' \
CHERRIES_TAGS='neutral,active-strain,force,convergence,continuation,review' \
.venv/bin/python -u src/71-plot-force.py
```

For an intentional rerender of these generated plots, pass `--overwrite true`.
The final render moved the stiffness annotation clear of the legend and used
the name `Neutral force curve with verified annotation layout`.
[Comet run](https://www.comet.com/liblaf/apple/79b97af808054e10b4e1056b0f085708)
records 474 samples, continuation step 432, final force 0.009250543 N and
threshold 0.01 N. The completed terminal log is preserved in the forward run's
`review-logs/71-plot-force.log`.

Ruff and compilation checks passed. Visual inspection verified the logarithmic
axes, phase colors, continuation marker, stiffness annotation and tolerance.
All seven added/updated HTTP assets matched their local SHA256 hashes,
including the main page, as recorded in
[HTTP verification](../data/http-verification-force-005.json).
