# Sequential four-stage MouthOpen activation fit

The user requested the same four-stage sequence used for Smile. The completed stage-56 MouthOpen fit was a separate PSD6-from-zero trial and is preserved. This new chain starts from zero activation at the verified full prescribed-jaw equilibrium from stage 49. It does not initialize from stage 56.

## Frozen protocol

The stages run sequentially, with 200 attempted Adam updates per stage plus an initial evaluation:

1. `symmetric6`: unrestricted symmetric dimensionless activation S, six controls per active cell; start S=0 and B=I.
2. `psd6`: initialize by PSD-projecting the completed stage-1 tensor, then fit six contraction-only controls.
3. `rankone_fixed`: retain the largest nonnegative eigenvalue and its principal axis from stage 2; refit one nonnegative amplitude with that axis fixed.
4. `rankone_learned`: copy stage-3 amplitude and axes, including axes at zero amplitude; optimize amplitude and a unit axis (three independent degrees of freedom).

Each stage receives the preceding accepted displacement as its forward seed and starts fresh Adam moments. Fixed-to-learned transfer is checked for equal physical tensors to floating-point precision. The established gradient-based chart update may replace the direction of an exactly-zero amplitude after a successful evaluation, without changing its zero tensor. Positive-amplitude directions are not replaced by that initialization rule.

All stages use L2 position plus normal loss, normal weight 1, smoothness weight `7.2e-6` (10x the earlier baseline), initial Adam learning rate 0.05, epsilon `1e-8`, and betas `(0.9, 0.999)`. The same full-tensor graph smoothness is used in every parameterization, with the existing 5 mm length and position normalization `L_REF_MM=13.236093032531715`. The coefficient is reused, not recalibrated for each stage.

The model, pruned mesh and chin-derived jaw pose match the preceding MouthOpen trial: 1,144,268 retained tetrahedra, 288,172 active cells, 501,313 within-region graph edges, no skin membrane and no contact forces. Exactly 2,249 fully fixed tetrahedra were removed in the saved derived fixture. The authoritative surviving `IsFixed` mask governs all three prescribed displacement components; all lip nodes remain free. The estimated full jaw pose is held fixed throughout fitting.

## Numerical acceptance and stop rule

The numerical policy matches stage 56: Newton search shifts are reused within each solve; physical free-force norm must be at most `1e-10`, and the unshifted adjoint must meet relative tolerance `1e-7`. Newton and CG budgets remain 100 and 3,000 iterations. There is no approximate-solve context. Every accepted objective, displacement, control and gradient must be finite.

Limited inversions remain allowed, as requested. At most 0.1% of retained cells may have nonpositive physical `J=det(F)` (1,144 cells); their number, minimum J and boundary self-intersections are saved. This threshold is an exploratory stop rule, not a mechanical-validity certificate. Signed activation in stage 1 is not silently PSD-clipped. Physical J and eigenvalues of B are recorded separately.

A rejected proposed update restores the preceding activation and Adam moments, halves the learning rate, and consumes one attempt. A failed initial stage evaluation blocks descendants rather than fabricating a gradient or transferring an incomplete stage. Completing 200 attempts is a budget result, not optimizer convergence. Endpoint component-gradient diagnostics are saved separately and report an explicit failure if unavailable.

## Evidence and execution

The new script is `src/70-run-four-stage-mouthopen.py`; outputs are under `data/70-mouthopen-four-stage/`. Each stage stores its initialization tensor and displacement seed, initial empty Adam state, parent checkpoint hash, update trace, successful solver receipts, rejected proposals, checkpoints, final optimizer state, component gradients and gradient balance. The chain stores its frozen numerical sources, process identity, mesh and runtime metadata. Earlier numerical sources and results remain unchanged.

From this experiment directory:

```bash
CHERRIES_NAME='MouthOpen complete four-stage activation chain' CHERRIES_TAGS='mouthopen,four-stage,active-strain,smoothness-10x,strict-solves' .venv/bin/python -u src/70-run-four-stage-mouthopen.py
```

After the chain finishes, independent CPU analysis will verify every transfer, optimizer reset, update budget, checkpoint hash, physical boundary, fit metric and solver gate. It will recompute full-S smoothness/L2 gradient ratios in the dual normalized effective-volume norm, excluding the normal gradient from the denominator. The figure will compare all four stage endpoints with a common camera and activation scale. No automatic Git commits or pushes are made.
