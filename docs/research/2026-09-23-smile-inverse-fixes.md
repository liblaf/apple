# Smile inverse fixes — 2026-09-23

Implemented the fixes from the [review](2026-09-23-smile-inverse-solver-review.md) without launching a Smile fitting campaign. Existing unrelated working-tree changes and historical experiment data were retained.

## Changes

- The reusable implicit wrapper saves solved displacement and prescribed values per autograd call; it restores mutable state after backward. Interleaved forward calls now produce the correct gradient.
- Forward and adjoint success, finiteness and independently evaluated residuals are recorded. Validation stays strict; Adam fitting explicitly permits finite unconverged solves and marks their checkpoints accordingly. Zero adjoint right-hand sides skip the linear solve safely.
- PNCG's stopping criterion now evaluates the force at the accepted state while preserving its original relative-tolerance reference.
- The no-skin stress study imports `liblaf.apple.solvers` and uses the optimized safeguarded Newton-CG policy with exact physical curvature, an unshifted adjoint and fresh accepted-force verification.
- Learned-axis optimization retains parent axes and releases zero-amplitude cells along an activating tensor-gradient eigenvector without perturbing their warm-start stresses.
- Smoothness calibration uses L2-only and smoothness tensor gradients at one common nonzero pilot state, with a frozen 0.1 calibration ratio. Actual endpoint ratios are recorded separately.
- Following the requested Adam policy, finite noisy updates continue without an outer Armijo test or tighter-tolerance replay. Unusable numerical proposals restore controls and moments, halve the learning rate and skip to the next attempt. Ordinary programming errors still propagate. Latest usable and best converged checkpoints are tracked separately; failed endpoint diagnostics do not stop the chain.
- Validation and fitting use new output directories and exact source-hash gates. The [v2 protocol](../../exp/2026/09/21/stress-activation-loss/docs/01-inverse-v2-protocol.md) describes mechanics, limits and commands.

## Verification

`uv run pytest tests src -q --no-cov`: **86 passed** on Python 3.14, including CUDA tests. Coverage includes delayed backward, boundary/material restoration, failed solves, zero RHS, accepted-state force, axis activation, component-gradient accounting, noisy Adam continuation, failure recovery, transfer of approximate parent states through all eight stages, and a real four-tetrahedron active-stress Newton/implicit-gradient comparison against finite differences at two stresses.

Focused Ruff checks and formatting checks pass. Type checks pass for the modified core inverse/PNCG implementation and inverse regression tests. `git diff --check` passes.

The final local Cherries [CPU parameterization receipt](../../exp/2026/09/21/stress-activation-loss/data/05-activation-validation-inverse-v2-003/receipt.json) passes all four maps, their spectral transfers, and zero-amplitude axis release. Maximum absolute finite-difference error is `9.176370774355291e-11`. `DEBUG=1` disabled remote Comet; the profile disables automatic Git commits.

The unrestricted default command `uv run pytest -q --no-cov` stops during collection because the unrelated legacy benchmark `benches/test_aggregation.py` imports the unavailable `equinox` package. That dependency was not added or changed. The Python-version/dependency-resolution Nox matrix was not run.

## Scope

Full-face derivative/tolerance validation, numerical smoothness calibration and the eight fits remain unrun. Their historical validation hashes cannot authorize the changed code; the new runner requires fresh validation. Small-model checks establish the fixes, not full-face convergence or performance.

The repaired study preserves its existing no-skin, fixed-jaw, contact-off fixture. It does not silently replace the separate skin-on/contact-on fixture, and no historical neutral checkpoint was promoted. If contact-on fitting is desired, bind that model explicitly before validating or calibrating it.

## Subsequent requested fit

The [L2 unrestricted fit record](../../exp/2026/09/21/stress-activation-loss/docs/41-l2-unrestricted-lr0p5.md) documents the subsequent full-face derivative reference, calibration, and active Adam run. The user requested learning rate 0.5 and Newton's first shift equal to the mean absolute Hessian diagonal. Finite inverted tetrahedra are accepted; determinant extrema and inversion counts are diagnostics, with no inversion-triggered rollback or learning-rate change. Resume support preserves the saved controls and Adam moments. After these changes, `uv run pytest tests src -q --no-cov` passed **90 tests**. The fit's source changes and approximate-solve receipts are retained separately from its converged mechanics reference.
