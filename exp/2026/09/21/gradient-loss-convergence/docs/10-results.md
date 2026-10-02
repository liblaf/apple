# Gradient-loss inverse-physics convergence curves

The saved runs **did not converge within their recorded budgets**. Their actual optimized
objectives were still decreasing, and the recorded inverse-optimization gradient norms
remained substantial. This analysis reads existing histories only; no fitting was resumed.

![Full loss and stationarity histories](../data/10-curves/loss-and-stationarity.png)

Each top panel plots the actual objective divided by its own neutral value. For smoothness-on
runs the solid curve includes the activation regularizer and the dotted curve shows only the data term.
Curve heights across objectives are not comparable
fit scores. The lower panels measure optimization gradients, not the surface-gradient data loss.

| Run | Updates | Initial objective | Final objective | Last 10 updates: decrease | Last 25 updates: decrease | Final optimization gradient / initial | Final inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw6-l2 | 100 | 8.656093 | 1.827852 | 7.95% | 20.36% | 25.41% | 1 |
| raw6-gradient | 100 | 7.381110 | 2.555468 | 4.08% | 10.80% | 20.76% | 1 |
| axis-off | 200 | 7.381110 | 3.762272 | 1.28% | 3.23% | 37.29% | 0 |
| axis-on | 200 | 7.381110 | 5.157575 | 0.56% | 1.43% | 58.07% | 0 |
| mixed-off | 200 | 8.656093 | 4.444767 | 1.31% | 3.30% | 39.13% | 0 |
| mixed-on | 200 | 8.656093 | 6.064170 | 0.59% | 1.49% | 56.57% | 0 |

![Final 25 updates](../data/10-curves/final-25-updates.png)

## Interpretation

The original Raw6 run has six unrestricted symmetric activation components per active cell.
Its lower panel uses the recorded ordinary activation-gradient RMS, normalized by its initial
value. L2 is included as the original matched control. The gradient branch first inverts at
update 79 and L2 at update 81; both finish with one inverted tetrahedron. They are neither
converged nor valid inversion-free endpoints. Both use constant Adam learning rate 0.3.

The learned-axis runs use contraction-only activation and a physical rank-one projected-gradient
diagnostic. Their declared rule requires objective relative span below 0.1% over 25 updates and
projected-gradient RMS below 1% of its initial value at two consecutive scheduled checks.
All learned-axis branches fail both thresholds at the endpoint and remain inversion-free.
They halve the learning rate after 100 updates: 0.3 for updates 1-100 and 0.15 for 101-200.
The 0.075 recorded at update 200 is a next-update setting that was never used. A reduced
slope after update 100 therefore cannot by itself be interpreted as convergence.

The final-window plot uses exactly J[T-25] through J[T], with decrease
100*(J[T-25]-J[T])/J[T-25]. All six histories are strictly decreasing; therefore this equals
the max-minus-min relative span used by the learned-axis stopping rule. No smoothing or
interpolation is applied. Values of 1.0 in unscheduled span rows are placeholders and are not used.

Forward equilibrium solver success is separate from inverse convergence. A finite loss plateau
alone would also be insufficient without stationarity and feasible geometry. These plots do not
establish how much further fitting could improve, an intrinsic model-capacity limit, or mechanical stability.

## Reproduction and audit

Working directory: `exp/2026/09/21/gradient-loss-convergence`.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
CHERRIES_NAME='Gradient-loss inverse physics: verified convergence curves' \
CHERRIES_TAGS='gradient-loss,inverse-physics,convergence,plots,3d' \
.venv/bin/python src/10-plot.py
```

Inputs and copied CSV, summary, and protocol receipts are recorded with SHA-256 hashes in
[the analysis receipt](../data/10-curves/summary.json). The script checks consecutive updates,
finite trace values, agreement with endpoint summaries, and independent recomputation of
the final scheduled span and projected-gradient ratio. The numerical inputs are unchanged.

Editable vector figures: [overview SVG](../data/10-curves/loss-and-stationarity.svg),
[final-window SVG](../data/10-curves/final-25-updates.svg).

The two gradient-only experiment groups are the primary evidence. The mixed L2/gradient
column is included separately as context for the latest discussion; it is not gradient-only.

## Rendering run and visual check

The final plot process and Cherries shutdown completed with exit code 0. Both PNGs
were inspected at full figure size; endpoint labels and threshold annotations are
readable and do not overlap. An independent read-only audit reproduced the final-window
decreases, gradient ratios, inversion history, and learning-rate interpretation.

[Final Comet run](https://www.comet.com/liblaf/apple/f3a8b4ce014741768aa2f7863a94ffd7).
The initial rendering encountered a Local-plugin input-snapshot collision because
external CSV files shared a basename. The final script registers uniquely located,
hash-verified local input copies, and the final log contains no plugin failure.
The initial artifacts and log remain preserved separately. The final log's directory
overwrite notice reflects aggregation of already copied inputs into the output snapshot.

The plotting source passes Ruff with CPY001 and the plotting function's C901
complexity rule exempted. No simulation source or historical result was modified.
Repository base: `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; unrelated existing
working-tree changes were retained. Local PNG/SVG and CSV receipts are the artifact
evidence; scalar/source logging does not imply that the figure assets were uploaded.

Recorded Comet summary:

```text
Comet.ml Experiment Summary
---------------------------------------------------------------------------------------
  Data:
    display_summary_level : 1
    name                  : Gradient-loss inverse physics: verified convergence curves
    url                   : https://www.comet.com/liblaf/apple/f3a8b4ce014741768aa2f7863a94ffd7
  Metrics:
    axis-off/final_gradient_ratio      : 0.37290687482286056
    axis-off/last_25_drop_percent      : 3.226679905937324
    axis-on/final_gradient_ratio       : 0.5807369017102807
    axis-on/last_25_drop_percent       : 1.4288571719772047
    mixed-off/final_gradient_ratio     : 0.391341708216195
    mixed-off/last_25_drop_percent     : 3.2975479568653987
    mixed-on/final_gradient_ratio      : 0.5657360199921428
    mixed-on/last_25_drop_percent      : 1.492253621284112
    raw6-gradient/final_gradient_ratio : 0.20763803016527038
    raw6-gradient/last_25_drop_percent : 10.79673653444926
    raw6-l2/final_gradient_ratio       : 0.25412332944811983
    raw6-l2/last_25_drop_percent       : 20.36223072919281
  Others:
    Name                : Gradient-loss inverse physics: verified convergence curves
    cherries/cmd        : .venv/bin/python src/10-plot.py
    cherries/comet/url  : https://www.comet.com/liblaf/apple/f3a8b4ce014741768aa2f7863a94ffd7
    cherries/end_time   : 2026-09-21 01:16:42.489452+08:00
    cherries/entrypoint : exp/2026/09/21/gradient-loss-convergence/src/10-plot.py
    cherries/exp_dir    : exp/2026/09/21/gradient-loss-convergence
    cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
    cherries/start_time : 2026-09-21 01:16:40.851282+08:00
  Parameters:
    output_dir : exp/2026/09/21/gradient-loss-convergence/data/10-curves
  Uploads:
    filename     : 1
    git metadata : 1
    source_code  : 1 (18.95 KB)

```
