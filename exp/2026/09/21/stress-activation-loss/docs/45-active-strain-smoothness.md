# Active-strain smoothness weight calibration

Select **`--smooth-weight 7.2e-07`** for the current unrestricted active-strain formulation, with the existing L2 normalization and 5 mm graph smoothness length. At frozen step 170, the regularization gradient norm is **0.100130 times the L2 gradient norm**. This analysis selected the coefficient; the running unregularized baseline was not changed or restarted.

## Measurement and selection

Both component norms come from the same step of `41-l2-unrestricted-active-strain-lr0p05-nosmooth-001/l2-symmetric6`. That fit uses no skin, L2 only, unrestricted six-component dimensionless active strain S, Adam learning rate 0.05, and epsilon 1e-8. The activation acts on the strain-energy norm; both volume terms retain physical det(F).

The established gradient metric is `||g||_* = sqrt(sum_i ||sym(g_i)||_F^2 / m_i)`, where `m_i` is normalized effective active-cell volume. Mandel coordinates preserve the Frobenius tensor norm. The coefficient is global, not a per-coordinate division by small L2 gradients.

| Quantity at step 170 | Value |
| --- | ---: |
| L2 gradient norm | 0.00567160758831 |
| Unweighted smoothness gradient norm | 788.748708386 |
| Exact coefficient for ratio 0.1 | 7.19063946224e-07 |
| Selected coefficient, two significant digits | 7.2e-07 |
| Weighted smoothness gradient norm | 0.000567899070038 |
| Achieved gradient ratio | 0.100130177 |
| Ratio in ordinary Euclidean Mandel control norm | 0.098140822 |
| L2 loss | 0.00557169197135 |
| Weighted smoothness loss | 0.000432652608617 |
| Smoothness / L2 loss ratio | 0.077651925 |

Selection: `eta = 0.1 * ||grad L2||_* / ||grad R||_*`, rounded to two significant digits. At the same chosen eta, trace steps 161–170 yield gradient ratios 0.092753–0.100130, median 0.096546. The loss ratio is a separate quantity and was not used for calibration.

The saved strain field was independently differentiated on CPU using `R = factor * sum_edges w_ij * ||S_i-S_j||_F^2` and edge derivatives `2 * factor * w_ij * (S_i-S_j)`. Both the raw smoothness value and its dual gradient norm match the fit trace to relative tolerance 1e-12. The L2 norm is the finite approximate adjoint gradient used by the optimizer; no new forward or adjoint solve was needed. Ruff lint and formatting checks pass.

## Scope

The source forward/adjoint evaluation is not convergence-certified (`solver_valid=false`); its solver receipt and 13 inverted cells are recorded as diagnostics. This does not invalidate the requested finite-gradient continuation calibration or gate optimization. The measured 0.1 ratio is global and state-dependent. It does not constrain every cell's contribution, represent an Adam update ratio, or establish the final ratio of a future regularized trajectory. Record that final ratio after the regularized fit; do not assume it stays exactly 0.1.

## Reproduction and artifacts

Working directory: `exp/2026/09/21/stress-activation-loss`.

```bash
CHERRIES_NAME='Smile active strain frozen step 170 smoothness calibration' \
CHERRIES_TAGS='smile,inverse,active-strain,l2,smoothness,gradient-calibration,cpu' \
.venv/bin/python \
  src/45-calibrate-active-strain-smoothness.py \
  --checkpoint data/45-active-strain-smoothness-001/calibration-state.npz \
  --output 45-active-strain-smoothness-002
```

Use a new output name when repeating. The first calibration captured the live step-170 checkpoint; the second replayed that exact frozen state with the reusable checkpoint option. Both return the same weight and ratio. No random sampling or optimizer update occurs in this CPU analysis.

- [Calibration and matched solver receipt](../data/45-active-strain-smoothness-002/calibration.json)
- [Frozen state](../data/45-active-strain-smoothness-002/calibration-state.npz)
- [Trace snapshot](../data/45-active-strain-smoothness-002/source-trace.csv)
- [Analysis script](../src/45-calibrate-active-strain-smoothness.py)
- [Terminal log](../tmp/45-calibrate-active-strain-smoothness-002.console.log)
- [Comet run](https://www.comet.com/liblaf/apple/2f70bcdf58644a92b3d1883ca1ba2f24)

The calibration receipt includes SHA-256 hashes of the frozen checkpoint, mesh, original fit protocol, and analysis script. All four hashes were checked after generation. The fit protocol records its numerical source archive. The worktree was already dirty; no automatic Git commit was enabled.

## Comet.ml Experiment Summary

```text
---------------------------------------------------------------------------------------
Comet.ml Experiment Summary
---------------------------------------------------------------------------------------
  Data:
    display_summary_level : 1
    name                  : Smile active strain frozen step 170 smoothness calibration
    url                   : https://www.comet.com/liblaf/apple/2f70bcdf58644a92b3d1883ca1ba2f24
  Metrics:
    l2_gradient_dual_norm                  : 0.005671607588311089
    selected_smooth_weight                 : 7.2e-07
    smoothness_gradient_dual_norm          : 788.7487083856289
    smoothness_to_l2_gradient_ratio        : 0.10013017670828736
    weighted_smoothness_gradient_dual_norm : 0.0005678990700376528
  Others:
    Name                : Smile active strain frozen step 170 smoothness calibration
    cherries/cmd        : .venv/bin/python src/45-calibrate-active-strain-smoothness.py --checkpoint data/45-active-strain-smoothness-001/calibration-state.npz --output 45-active-strain-smoothness-002
    cherries/comet/url  : https://www.comet.com/liblaf/apple/2f70bcdf58644a92b3d1883ca1ba2f24
    cherries/end_time   : 2026-09-23 06:52:15.768937+08:00
    cherries/entrypoint : exp/2026/09/21/stress-activation-loss/src/45-calibrate-active-strain-smoothness.py
    cherries/exp_dir    : exp/2026/09/21/stress-activation-loss
    cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
    cherries/start_time : 2026-09-23 06:52:15.052430+08:00
  Parameters:
    recent_steps          : 10
    source                : 41-l2-unrestricted-active-strain-lr0p05-nosmooth-001
    target_gradient_ratio : 0.1
  Uploads:
    filename     : 1
    git metadata : 1
    source_code  : 2 (7.41 KB)

```
