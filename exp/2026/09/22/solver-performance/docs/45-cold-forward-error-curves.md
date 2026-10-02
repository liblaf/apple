# Cold-start error curves

The figure plots measured relative free-force residual against solver progress, with separate Newton and PNCG iteration scales. It is not a skin-position or energy-error plot. Both arms use initial force `1.2678782634149829e-05`. The actual hybrid switch is `0.0078871926` relative; final convergence is `0.00078871926` relative; the nominal `1e-3` reference is also shown.

![Cold-start force residual](../data/cold-forward-error-003/cold-forward-error.png)

Newton has 100 pre-step force records plus its terminal force. PNCG has only four intermediate logged samples at counters 500, 1000, 1500 and 2000, plus the common initial force and last optimizer gradient at failure counter 2289. Dashed PNCG lines are visual guides, not interpolation claims. Missing oscillations cannot be inferred. Its terminal sample is not an independent recomputation at a saved endpoint.

The exact synchronized totals are 204.124 s and 600.117 s. The logs do not contain exact per-state elapsed times; Newton logs a pre-update force after accepting its update. An exact elapsed-time curve therefore cannot be reconstructed and is not plotted. No GPU solve was rerun.

Source receipts: `data/cold-forward-comparison-001/summary.json` and `stdout.log`; hashes and every plotted coordinate are in `data/cold-forward-error-003/curve-data.json`. PNG, editable SVG and PDF are in the same directory. The temporary tailnet report includes the figure.

Reproduce from this experiment directory with a new output name:

```sh
DEBUG=1 CHERRIES_NAME=cold-forward-error-curves CHERRIES_TAGS=smile,performance,plot \
  uv run python src/45-plot-cold-error.py --output data/NEW-PLOT
```

Cherries records local logs and artifacts; remote Comet is disabled for this local analysis. Plot generation does not modify raw numerical receipts or physics.
