# Released-axis smoothness continuation

This experiment tests whether strengthening the existing activation-tensor smoothness penalty improves the irregular direction field in the four-stage smile fit. Three new released-axis runs start from the same saved fixed-axis endpoint. Each uses fresh Adam moments and the same 200-attempt budget; accepted updates and skipped attempts are reported separately.

## Controlled comparison

| Multiplier | Smoothness coefficient |
| --- | ---: |
| 1× | 7.2e-7 |
| 3× | 2.16e-6 |
| 10× | 7.2e-6 |

The parent is the update-200 L2-plus-normal fixed-axis checkpoint from `exp/2026/09/21/stress-activation-loss/data/51-visualization-checkpoints-002/l2-normal/l2-normal-rankone_fixed/last.npz`. Its active-strain tensor, fixed reference axes, and displacement seed are copied into every branch. The historical conversion to released-axis controls preserves zero-amplitude axes, and the original gradient-based zero-amplitude chart initialization remains enabled.

Other settings remain the archived experiment's: normal coefficient approximately 1, Adam learning rate 0.05, epsilon 1e-8, betas (0.9, 0.999), same graph and 5 mm smoothness normalization, materials, boundary conditions, solver tolerances, and finite approximate continuation policy. The comparison changes only the smoothness coefficient. No optimizer moments are inherited.

## Reproducibility

`src/00-prepare.py` restores the original 462-file deployment into `data/00-frozen-source/apple` and verifies every deployment hash. All 111 numerical source snapshots match the source freeze captured with the original L2-plus-normal chain. Fixture hashes also match. The preparation receipt and shared protocol record the parent and mesh SHA-256 hashes.

All three new branches run in the same local runtime: RTX 4090, Python 3.14.6, Torch 2.12.0+cu130, Warp 1.14.0, and CuPy 14.1.1. The historical source run used an RTX 5090 with Python 3.12.3 and Torch 2.12.1+cu130. Selection therefore uses the new 1× baseline, rather than assuming numerical identity with the historical released-axis endpoint. Runtime contention means timings are operational observations, not GPU benchmarks.

The numerical runner imports the isolated historical package before the current editable package and asserts that loaded Apple modules come from that isolated root. Material and tolerance dictionaries must equal the historical protocol, and mesh, target, weights, and graph arrays are checked against the captured mesh. Changes in this experiment's orchestration and logging do not modify the frozen numerical sources.

Run from this experiment group, using the repository's existing interpreter without changing dependencies:

```bash
SMOOTHNESS_RUN_LABEL=1x CHERRIES_NAME='Released-axis smoothness sweep, baseline 1x' CHERRIES_TAGS='smile,active-strain,smoothness,continuation,matched,1x' .venv/bin/python src/10-run-sweep.py --multiplier 1 --steps 200 --output 10-sweep/multiplier-1
SMOOTHNESS_RUN_LABEL=3x CHERRIES_NAME='Released-axis smoothness sweep, 3x' CHERRIES_TAGS='smile,active-strain,smoothness,continuation,matched,3x' .venv/bin/python src/10-run-sweep.py --multiplier 3 --steps 200 --output 10-sweep/multiplier-3
SMOOTHNESS_RUN_LABEL=10x CHERRIES_NAME='Released-axis smoothness sweep, 10x' CHERRIES_TAGS='smile,active-strain,smoothness,continuation,matched,10x' .venv/bin/python src/10-run-sweep.py --multiplier 10 --steps 200 --output 10-sweep/multiplier-10
```

Existing output directories cause an error. Use a fresh output directory for any rerun. The initially launched 1× process used the historical logging profile; subsequent branches use separate log filenames. Exact per-process stdout is retained in `tmp/run-1x.log`, `tmp/run-3x.log`, and `tmp/run-10x.log`. All profiles disable automatic Git commits.

## Measurements and proposed decision rule

For rank-one activation `S_i = a_i n_i n_i^T`, split each graph-edge contribution exactly:

`||S_i-S_j||_F^2 = (a_i-a_j)^2 + 2 a_i a_j [1-(n_i^T n_j)^2]`.

Sum both terms with the original graph conductances and regularizer factor. The second term measures sign-invariant, amplitude-weighted directional roughness. Report amplitude roughness, directional roughness, total roughness, activation magnitude, positional RMS, normal-angle RMS, minimum physical determinant, inverted-cell count, and solver receipts. Compute position/normal/determinant metrics independently from the saved checkpoint.

The predeclared exploratory criterion is at least 50% lower directional roughness than the new 1× endpoint, with both positional and normal RMS no more than 5% above that baseline. Choose the smallest multiplier meeting all three criteria. If neither meets them, do not increase the coefficient further within this experiment.

The endpoint gradient ratio is `||d(eta R)/dS||_* / ||d L2/dS||_*`, using the full symmetric tensor and dual effective-volume norm. It excludes the normal component from the denominator. The historical diagnostic reevaluates the final controls with a fresh approximate forward and component-adjoint solve; it is not necessarily the identical displacement state used by the last Adam update.

All these runs inherit an already approximate parent containing inverted cells. Improvement in fit or roughness does not certify equilibrium, convergence, or anatomical validity. The figures show only one positive principal mode on the deformed geometry, so tensor metrics remain the quantitative comparison.
