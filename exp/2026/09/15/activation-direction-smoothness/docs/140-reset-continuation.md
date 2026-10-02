# Reset forward displacement after failure; continue inverse Adam

[Updated aligned figure](../data/150-reset-figures/aligned-shape-activation-h200-reset-16x9.png) · [Vector SVG](../data/150-reset-figures/aligned-shape-activation-h200-reset-16x9.svg) · [Vector PDF](../data/150-reset-figures/aligned-shape-activation-h200-reset-16x9.pdf) · [Gallery](../data/150-reset-figures/index.html)

## Result

Resetting the forward displacement to the rest shape after each failure restored force-equilibrium convergence. The free-activation, zero-smoothness, h = 0.20 continuation completed the original 1,200-update budget. Seven failures occurred at steps 263–269; seven solves at steps 264–270 therefore started from zero displacement. Step 270 converged, and every subsequent forward solve through 1200 converged using the preceding displacement as its initial state.

The final normalized L2 is **0.4363050214**, a **1.3512%** decrease from the original step-262 checkpoint. However, this endpoint is a **mechanically unstable equilibrium**: its force residual is tiny, but its displacement Hessian has a negative eigenvalue. The updated free/off panel is explicitly labeled **UNSTABLE**.

| Quantity | Original step 262 | No reset, step 1200 | Reset after failure, step 1200 |
| --- | ---: | ---: | ---: |
| Normalized L2 / h² | 0.4422812053 | 0.4422487139 | 0.4363050214 |
| Raw L2 | 0.0176912482 | 0.0176899486 | 0.0174522009 |
| Force residual infinity norm | 6.7104e-14 | 2.6110e-4 | 1.5591e-15 |
| Minimum physical J = det(F) | 0.00176353 | 1.00000058e-8 | 0.01675906 |
| Final forward convergence | Yes | No | Yes |
| Final smallest symmetric Hessian eigenvalue | — | +1.1862e-5 | **−0.0076889854** |
| Post-262 failed forward solves | — | 938 | 7 |

The no-reset L2 is evaluated at a retained off-equilibrium iterate. It is included to explain the effect of resetting, rather than as a matched equilibrium-fit score. The three columns also represent different inverse trajectories after the original failure; the original 32-run study remains preserved.

## Behavior and fixed settings

This run starts again from the original saved step-262 checkpoint so that it can be compared with the previous no-reset continuation under the same total update budget. It does not append updates to the already exhausted no-reset step-1200 trajectory.

For each inverse iteration:

1. Evaluate the current activation using the current forward displacement seed.
2. If the forward solve fails, retain its last accepted finite iterate for the loss and approximate adjoint, then continue the Adam activation update.
3. Initialize the **next** forward solve with `u = 0` after failure. Preserve the updated activation, Adam moments, and iteration counter.
4. After a successful forward solve, initialize the next solve with that converged displacement.

There is no retry loop at fixed activation and no reset of activation, optimizer momentum, or learning-rate decay. The same fixed corrected Stable Neo-Hookean energy, L2 data loss, muscle band, mesh, free symmetric activation parameterization, zero smoothness weight, 250-iteration forward limit, force tolerance 1e-10, and positive-J Armijo rule are retained. The original step-263 activation proposal and Adam moments match bit-for-bit.

The behavior is controlled by `reset_forward_on_failure` in [120-continue-inexact.py](../src/120-continue-inexact.py). Its default remains false for the previous experiment; [140-continue-reset.py](../src/140-continue-reset.py) enables it and writes a new output directory. Checkpoints store both the evaluated displacement and the next forward seed. The trace records the actual seed norm and whether the preceding failure caused a reset.

## Stability and interpretation

An independent recomputation gives the final Hessian's minimum eigenvalue as −0.007688985363813333, with eigenpair residual 1.49e-15. For the unit eigenvector v, both small displacement perturbations lower energy while retaining positive triangle areas:

| Perturbation | Energy change | Minimum J |
| --- | ---: | ---: |
| +0.001 v | −3.91184e-9 | 0.01619267 |
| −0.001 v | −3.69121e-9 | 0.01732595 |

This verifies an unstable direction at force equilibrium. The existing forward success criterion checks the force residual, and does not require a positive-definite Hessian. “Stable Neo-Hookean” names the constitutive model; it does not certify stability of each returned equilibrium.

The seven failed solves still use approximate off-equilibrium adjoints, as requested. The run demonstrates that resetting can escape the frozen forward trajectory, but does not establish a stable mechanical solution or inverse convergence. The final inverse gradient is nonzero, and `inverse_stationarity` remains false. The final 50 updates reduce normalized L2 by only 0.000122% under the original decaying learning rate.

## Reproduction and evidence

Working directory: `exp/2026/09/15/activation-direction-smoothness`.

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Continue Adam with forward reset after failure' \
CHERRIES_TAGS='2d,activation,inexact-forward,forward-reset,continuation,l2' \
uv run python src/140-continue-reset.py \
  > logs/reset-continuation-terminal.log 2>&1
```

The numerical run and Cherries shutdown completed with **exit 0**. Computation took about 44 seconds. The Comet run used Git SHA `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, with automatic commits disabled; source hashes and snapshots identify the executed uncommitted experiment code. Use a fresh `--output` directory for reproduction because existing output directories are protected from overwriting.

- [Protocol and source hashes](../data/140-reset-continuation/protocol.json)
- [Final summary](../data/140-reset-continuation/summary.json)
- [Trace: every step 262–1200](../data/140-reset-continuation/h200-unconstrained-w0/trace.csv)
- [Checkpoint, including next forward seed](../data/140-reset-continuation/h200-unconstrained-w0/checkpoint.npz)
- [History and recorded input seeds](../data/140-reset-continuation/h200-unconstrained-w0/history.npz)
- [Independent post-run validation](../data/140-reset-continuation/independent-checks.json)
- [Terminal log](../logs/reset-continuation-terminal.log)
- [Comet run](https://www.comet.com/liblaf/apple/b13ebddb50734fdfa16202b4819bfd72)

The audit verifies 939 trace rows, 95 history snapshots, 932 converged records including the initial checkpoint, seven failed records, and the seven actual zero-seed resets. Current frozen-physics source hashes, copied source hashes, checkpoint reconstruction, and final history/checkpoint equality pass. Ruff and formatting checks pass for the modified runner and new entrypoint.

The historical `data/10-pork-2d` asset hook again emitted a missing-path warning at shutdown. The local artifacts above are authoritative; this report does not claim that all numerical artifacts were uploaded to Comet.

## Visualization receipt

The updated comparison replaces only the free/off endpoint and shows its unstable-equilibrium status. The other seven shape and activation geometries match their original saved glyph arrays exactly. All panels retain equal physical scales and the full-domain view; activation axes are transported to the deformed shape with `F n / ||F n||`.

The main PNG is **15,360 × 8,640**, exactly 16:9. SVG and one-page PDF exports contain no embedded raster images; the PDF page is 1,152 × 648 points. [Diagnostic histories](../data/150-reset-figures/reset-vs-no-reset-history.png) compare the no-reset and reset L2, force residual, and minimum physical J across every recorded update. The endpoint markers distinguish an unconverged retained state from an unstable force equilibrium.

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CHERRIES_NAME='2D rest-reset continuation figures' \
CHERRIES_TAGS='2d,activation,continuation,reset,figures' \
uv run python src/150-render-reset.py \
  > logs/150-render-reset-terminal.log 2>&1
```

Rendering and Cherries shutdown completed with **exit 0**. The full comparison preview and diagnostic plots were visually inspected. [Alignment and input checks](../data/150-reset-figures/alignment-checks.json) · [Delivery checks and file hashes](../data/150-reset-figures/delivery-checks.json) · [Rendering log](../logs/150-render-reset-terminal.log) · [Rendering Comet run](https://www.comet.com/liblaf/apple/654d1dba73674f9a971e358122d3770d).
