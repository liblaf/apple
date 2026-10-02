# Corrected inverse pipeline

The current numerical sources implement the fixes from the [inverse review](../../../../../../docs/research/2026-09-23-smile-inverse-solver-review.md). The original [v1 protocol](00-protocol.md) and its saved results remain historical evidence; v2 uses fresh output directories and requires new source-matched validation.

## Mechanics and solver

This is the existing no-skin fixture: fixed passive materials and boundary conditions, no membrane or membrane prestress, fixed jaw, and contact disabled. The exterior triangle mesh remains the observation surface. This is **not** the separate skin-on, contact-on, jaw-fitting model. Changing the contact/fixture choice requires its own explicit model binding and validation.

The forward adapter now uses the optimized neutral driver's safeguarded Newton-CG policy. It uses exact physical Hessian products and an unprojected bulk diagonal, a step cap of half the mean unique reference edge length, Newton shifts starting from the current mean absolute Hessian diagonal, and an independently measured force residual. There is no PNCG fallback. Default force tolerance is absolute `1e-10`, relative `0`; the adjoint relative tolerance is `1e-7`. These are starting settings to validate on the face, not a certified displacement-error bound. Newton shifts never enter the physical implicit adjoint. The PCG iteration cap is 10,000. The full-face probe required 1,368 unshifted iterations at the validation tolerance, exceeding the earlier 1,000 cap. Following the subsequent user instruction, this stress study starts every Newton-CG step at the mean absolute diagonal and skips the zero-shift attempt; further retries multiply the positive shift by 10. The physical implicit adjoint remains unshifted.

The implicit wrapper saves independent displacement and prescribed boundary values per call, rebuilds the corresponding contact state when applicable, and restores the current caller's state after backward. Validation requires solver success and small true residuals. During Adam fitting, finite unconverged forward states and adjoint solutions remain usable as approximate gradients; receipts still mark them unconverged. Nonfinite states or gradients remain unusable. A zero adjoint right-hand side returns an explicit successful zero-residual receipt without running a linear solver.

## Objective and stages

The two independent loss columns retain the exact existing balance: 2 mm positional **vector RMS** has the same objective contribution as a uniform 5 degree normal error. Position uses `P / 13.236093032531715^2`, with P the reference-area-weighted coordinate MSE in mm². Normals use the reference-area-weighted squared unit-normal chord error, with weight 0 or 1.

Smoothness penalizes the full dimensionless stress tensor on the fixed within-muscle graph. `20-calibrate-smoothness.py` now runs one declared L2-only unregularized pilot, takes both component gradients at its same saved nonzero-stress endpoint, and sets one shared coefficient so the weighted smoothness gradient is 0.1 times the **L2-only** gradient in the dual normalized effective-volume norm. The coefficient stays fixed across all eight fits. Endpoint component ratios are saved to `gradient-balance.json`; they are diagnostics, not convergence conditions. At an unconstrained interior stationary L2-plus-regularizer solution with nonzero component gradients, the ratio must be 1.

The stage sequence remains unrestricted symmetric 6-DoF → PSD 6-DoF → fixed-axis nonnegative scalar → learned-axis rank-one stress. PSD has no upper stress cap. Every transition starts a fresh Adam optimizer and re-equilibrates the projected tensor using the parent's displacement only as a seed. Stage 4 retains stage-3 axes, including at zero amplitude. If a zero-amplitude axis misses an activating direction, the symmetric total tensor gradient supplies a negative-eigenvalue direction; this axis change leaves the zero stress exactly unchanged. Only that cell's affected Adam moments reset.

Projected Adam permits objective increases, non-descent directions and finite unconverged forward or adjoint solves. It retains its moments and uses a fixed learning rate for these noisy updates. There is no outer Armijo line search, tighter-tolerance replay or gradient-descent fallback. The forward Newton solver keeps its own numerical safeguards.

Finite inverted tetrahedra are allowed during Adam continuation. The minimum determinant and inverted-cell count are diagnostics only; they do not reject updates or reduce the learning rate. An unusable state or gradient containing nonfinite values skips that proposal, restores controls and Adam moments, halves the learning rate and continues at the next attempt. Failure at initialization is retried within the same bounded budget. Programming errors remain fatal. `steps` counts attempted updates; summaries separately record successful optimizer updates, skipped attempts, approximate evaluations and the last usable step. A budget ending on an unusable attempt still transfers the last usable stress to the next restriction stage.

`last.npz` and `optimizer-latest.pt` hold the latest usable state with an explicit `solver_valid` flag. `best-available.npz` holds the lowest observed objective, including approximate solves. `best-valid.npz` and its optimizer checkpoint hold the lowest objective among states whose forward and adjoint solves passed their convergence checks; they may be absent. Approximate objective values are not certified rankings. Endpoint gradient-balance diagnostics use strict solves; a numerical failure marks that diagnostic unavailable and does not interrupt the remaining stages. Calibration likewise uses strict component gradients to freeze its coefficient. Budget completion is not an inverse-convergence certificate.

## Fresh validation and execution

Run from this experiment directory with a descriptive `CHERRIES_NAME` and `CHERRIES_TAGS`. `DEBUG=1` keeps validation local. The final CPU validation already completed under `data/05-activation-validation-inverse-v2-003/`; its receipt covers all four maps and the zero-amplitude activating direction. No full-face solve was run during this fix.

Before calibration or fitting, run the new full-face derivative validation and freeze its exact numerical source hashes:

```sh
CHERRIES_NAME='Smile inverse v2 derivative validation' CHERRIES_TAGS='smile,inverse,validation,no-skin' \
  uv run python src/10-validate-physics.py
CHERRIES_NAME='Smile inverse v2 validation binding' CHERRIES_TAGS='smile,inverse,validation,no-skin' \
  uv run python src/12-freeze-validation.py
```

Then, when running the experiment:

```sh
CHERRIES_NAME='Smile inverse v2 weak smoothness calibration' CHERRIES_TAGS='smile,inverse,calibration,no-skin' \
  uv run python src/20-calibrate-smoothness.py
CHERRIES_NAME='Smile inverse v2 two four-stage chains' CHERRIES_TAGS='smile,inverse,no-skin' \
  uv run python src/40-run-chains.py
```

Default outputs have the `inverse-v2` suffix and must not already exist. Provide fresh `--output`, `--validation`, `--calibration` or `--activation` paths when rerunning. Changed numerical sources invalidate old gates; there are no source-change exemptions. The old metadata-recovery script `11-finalize-physics-validation.py` and historical campaign supervisor `80-execute-campaign.py` are not part of this pipeline. The current chain driver does not update the old site or launch rendering by default.

## Single unrestricted L2 fit

`41-fit-l2-unrestricted.py` runs only `l2-symmetric6`. Its default strict-gate mode retains the full source-matched validation and external calibration requirements above. Its explicit `--mechanics-reference` mode instead records a prior passed, converged mechanics derivative check and every source difference, then runs a fresh 8-attempt unregularized L2 pilot with the currently selected Newton policy. The pilot component gradients are allowed to be approximate; the coefficient is frozen directly from the accepted pilot checkpoint's recorded physical gradient norms, without a calibration re-equilibration. Its convergence receipts and achieved 0.1 gradient balance are recorded. The 200-attempt fit starts again at zero stress with fresh Adam state. This reference mode does not label the changed runtime as source-matched full-face validation.

`42-plot-single-fit.py` plots the saved error and objective history and renders target/fit/error plates into a fresh output directory without updating the historical site.
