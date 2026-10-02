# Saved Newton energy curve

![Mechanical energy](../data/cold-forward-energy-003/cold-forward-energy.png)

The 100 pre-update energies at Newton iterations 0 through 99 decrease strictly across all 99 recorded transitions, from `-1.3004777189657777e-07` to `-9.0439253025954834e-07` solver energy units. Total recorded decrease is `7.7434475836297054e-07`. The right panel shows positive differences between successive recorded energies on a logarithmic axis.

This is the inner physical objective, not inverse skin-position loss or a force-error estimate. The absolute energy includes the model's chosen offset. The terminal post-update energy after step 100 was not saved. PNCG energies were not recorded, so no PNCG curve is inferred. The user selected plotting saved Newton energy; no GPU solve was rerun. Despite decreasing energy, the final force was `7.2515651046718926e-08`, above the `1e-8` convergence target.

Data and source hashes are in `data/cold-forward-energy-003/energy-data.json`; PNG, SVG and PDF are alongside it. Reproduce from the experiment directory with a new output directory:

```sh
DEBUG=1 CHERRIES_NAME=saved-newton-energy CHERRIES_TAGS=smile,performance,energy,plot \
  uv run python src/46-plot-newton-energy.py --output data/NEW-PLOT
```

Cherries local logging is enabled; remote Comet is disabled for this CPU-only plotting task. Raw solver artifacts remain unchanged.
