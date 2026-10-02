# Continue through forward failure: inexact h = 0.20 trajectory

Follow-up: [resetting forward displacement after failure](140-reset-continuation.md) restored forward convergence while preserving inverse Adam state. Its final equilibrium is mechanically unstable.

[Protocol](../data/120-inexact-continuation/protocol.json) · [Final summary](../data/120-inexact-continuation/h200-unconstrained-w0/summary.json) · [Trace](../data/120-inexact-continuation/h200-unconstrained-w0/trace.csv) · [Checkpoint](../data/120-inexact-continuation/h200-unconstrained-w0/checkpoint.npz) · [History](../data/120-inexact-continuation/h200-unconstrained-w0/history.npz)

[Aligned figure, 15,360 × 8,640 PNG](../data/130-inexact-figures/aligned-shape-activation-h200-inexact-16x9.png) · [SVG](../data/130-inexact-figures/aligned-shape-activation-h200-inexact-16x9.svg) · [PDF](../data/130-inexact-figures/aligned-shape-activation-h200-inexact-16x9.pdf) · [Gallery](../data/130-inexact-figures/index.html)

## Purpose and method

The free-activation, zero-smoothness, h = 0.20 run stopped at the original Adam proposal for step 263 because the inner Newton solve could not find a positive-J Armijo step. At the user's request, this diagnostic continues the original Adam schedule through update 1200 from the exact saved step-262 checkpoint and its exact reconstructed step-263 proposal.

The copied forward routine has the same energy, derivatives, damping sequence, and positive physical-J Armijo rule as the original solver. On a recoverable forward failure it returns the last accepted finite Newton iterate; it never returns a rejected trial. The original corrected Stable Neo-Hookean physics, L2 loss, 100 × 10 mesh, muscle band, unconstrained symmetric activation, zero smoothness weight, Adam moments, and learning-rate schedule are preserved. There is no outer backtracking, minimum-J diagnostic stop, or plateau stop.

The recorded Cherries entrypoint command was:

```bash
.venv/bin/python3 src/120-continue-inexact.py
```

Run from `exp/2026/09/15/activation-direction-smoothness`; the shell invocation was:

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Continue Adam through failed forward solves to 1200' \
CHERRIES_TAGS='2d,activation,inexact-forward,continuation,l2' \
uv run python src/120-continue-inexact.py \
  > logs/inexact-continuation-terminal.log 2>&1
```

The numerical process completed with **exit 0**.

## Result

The continuation recorded steps 262 through 1200: one converged forward solve at 262 and **938 line-search-failed forward solves** at 263–1200. The initial state is the original equilibrium solution; every later reported state is nonconverged.

| Quantity | Step 262, converged | Step 1200, nonconverged |
| --- | ---: | ---: |
| L2 / h² | 0.4422812053271873 | 0.44224871388806347 |
| Raw L2 | 0.017691248213087495 | 0.01768994855552254 |
| Force residual ∞-norm | 6.710384489418868e-14 | 2.6109605178155197e-4 |
| Minimum physical J = det(F) | 0.0017635323766390099 | 1.0000005785655142e-08 |
| Smallest symmetric Hessian eigenvalue | — | 1.1862212892e-05 |

The first failed forward solve, at 263, reaches the last admissible iterate at the positive-J floor: J_min = 1.0000005785655142e-08, with residual 3.190749798780099e-06. From saved step 270 through 1200 the displacement array is bitwise unchanged, while controls continue to change: `||q1200 − q270||∞ = 0.3416055283`. Therefore the apparent L2 change occurs only from 262 to 263; it is not continued shape or fit improvement through the remaining 937 updates. The residual instead increases to 2.6109605178155197e-4 as controls change against the fixed retained displacement.

Direct replay of the exact saved step-263 proposal from the step-262 displacement returns a displacement bitwise identical to the final step-1200 checkpoint (maximum absolute difference zero). All 938 post-262 L2 values are exactly equal. The relative initial-to-final L2 decrease is only about 0.00735%.

## Interpretation and limits

At a nonconverged returned state, the adjoint calculation is an **approximate off-equilibrium linearization**, not the exact gradient of the equilibrium-constrained inverse objective. The final adjoint relative residual is 2.2547291391193388e-13, so that linear solve is numerically accurate for its retained Hessian; it does not make the forward state an equilibrium or the reported L2 an equilibrium-fit value.

This answers the narrow request to keep taking the original Adam updates despite forward failure. It does not establish optimization convergence, a feasible equilibrium trajectory, or an improved final fit. The saved summary correctly records `inverse_stationarity: false`, `final_forward_converged: false`, and the final failure reason. The original comparison checkpoint and outputs remain unchanged.

## Independent audit evidence

The original checkpoint SHA-256 matches the protocol. The frozen hashes for `study.py`, `activation_models.py`, `physics2d.py`, the historical mesh source, and the original runner all match the copied source snapshot. A nontrivial successful solve reproduces the original solver bit-for-bit. Replaying the final checkpoint returns `line_search_failed` immediately with the stored displacement, residual, and J; B reconstructs bit-for-bit from the stored controls. The history contains 95 unique expected snapshots, `[262, 270, ..., 1200]`, and its final controls and displacement are bitwise identical to the final checkpoint.

[Comet run](https://www.comet.com/liblaf/apple/71f0b033dab34ffd88ec6ef05a80377d) · [Terminal log](../logs/inexact-continuation-terminal.log) · [Cherries log](../logs/120-continue-inexact.log)

The terminal log contains a warning that the historical `data/10-pork-2d` hook was absent during asset logging. The local artifacts linked above are the authoritative run record; this receipt does not claim that all artifacts were uploaded remotely.

## Updated visualization

The aligned h = 0.20 comparison replaces only the free/off panel with this step-1200 retained state and labels it **UNCONVERGED**. A two-line footnote explains that activation continued while shape remained fixed after step 263. The other seven shape and activation geometries match the prior saved glyph arrays exactly. Their forward states reached equilibrium; completing their inverse update budget does not establish inverse convergence.

The whole-domain comparison retains identical physical scales, activation axes transported by `F n / ||F n||`, 0.45-point thin equal-length glyphs, and a common signed activation color scale. The full 16:9 PNG is **15,360 × 8,640**. The one-page PDF is 1,152 × 648 points and contains no embedded raster images; the SVG contains 36,206 vector paths and no image elements. A full preview and 600 dpi PDF crop were visually inspected. See [alignment checks](../data/130-inexact-figures/alignment-checks.json), [file and vector checks](../data/130-inexact-figures/delivery-checks.json), and the source snapshots alongside the figures.

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Visualize inexact continuation at 1200 updates' \
CHERRIES_TAGS='2d,activation,inexact-forward,continuation,figures,16k' \
uv run python src/130-render-inexact.py \
  > logs/inexact-figures-terminal.log 2>&1
```

Rendering and Cherries shutdown completed with **exit 0**. [Rendering Comet run](https://www.comet.com/liblaf/apple/ca2a81e05cab49fb90af13cc6faf9d3e) · [Terminal log](../logs/inexact-figures-terminal.log) · [Comet summary](../data/130-inexact-figures/comet-summary.txt). Both scripts use a normal evidence profile with automatic Git commits disabled. Reproduction should choose a fresh `--output` directory because existing outputs are protected from overwriting.
