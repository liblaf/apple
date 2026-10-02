# Validation before the eight staged fits

The CPU activation checks passed for all four parameterizations, including PSD projection, strongest-mode transfer, exact rank-one axis release, and continuous learned-axis normalization. Maximum CPU directional derivative error was 9.176e-11.

The full-face signed-stress derivative check compared both loss columns, two stress directions, and two finite-difference steps. All eight checks passed; the maximum relative error was 0.00255822 (0.256%). Validation used tighter equilibrium and adjoint tolerances than fitting so finite-difference subtraction could resolve the derivative. This checks the loss/physics derivative near the specified probe state; it does not certify convergence of the subsequent fits.

Two earlier failed attempts remain in `data/10-validation-eigensolver-allocation` and `data/10-validation-production-tolerance`. The first exposed excessive GPU workspace for a large batched eigensolver; diagnostics now use CPU eigenvalues and constrained projection uses bounded GPU chunks. The second exposed insufficient equilibrium precision for finite-difference validation.

The successful numerical check then failed during source archival on a generated relative module path. `data/11-validation-finalization/recovery.json` preserves the eight numerical check rows and records metadata-only finalization. A complete source archive was captured after those numerical checks, not before them. `data/12-validation` binds that evidence, final CPU activation validation, and the final campaign sources. The new fitting runs verify this frozen source set before starting.

The optimizer loop also passed a separate small CPU analytic exercise through all four parameterizations and their transfers, with three accepted monotone updates each. This checks runner behavior only, not facial mechanics.

Run links:

- [CPU activation validation](https://www.comet.com/liblaf/apple/36ff3c806ece4b31a6403cb6987a4fa4)
- [Full-face numerical checks and archival failure](https://www.comet.com/liblaf/apple/0b38941c1ba94a1c9ba2234a23b25965)
- [Metadata finalization](https://www.comet.com/liblaf/apple/0233ef9ab1d04fa18966b8b85df07668)
- [Frozen campaign source validation](https://www.comet.com/liblaf/apple/f474d28c78774744b45f0058dd6c8a7f)
- [Smoothness calibration](https://www.comet.com/liblaf/apple/036dc43785bf4290b632c4cd2eb96c1b)
