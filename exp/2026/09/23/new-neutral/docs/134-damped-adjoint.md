# Damped MouthOpen adjoint probe

[`damped-adjoint-probe-001`](../data/damped-adjoint-probe-001/) evaluates an
explicitly damped, approximate implicit pullback at the same saved
forward-verified MouthOpen trial. It does not rerun optimization and does not
change the physical forward state: each variant recovered the saved endpoint
with zero reported displacement change, force `9.960237108699218e-09 MPa m²`
(0.009960237 N), and passing contact gates.

The free adjoint system is

```text
(H_ff + lambda I) p = -L_f
lambda = relative_shift * mean(abs(diag(H_ff)))
```

The assembled physical Hessian remains unshifted. For each CG solution, the
runtime evaluates both the CSR shifted residual and the native physical
residual plus the same `lambda p` term. The larger shifted relative residual
must be at most `1e-7`. The original unshifted residual is reported as the
bias caused by damping; it is not an exact-adjoint convergence gate.

| Relative shift | lambda | CG operator applications | Adjoint seconds | Max shifted relative residual | Original unshifted relative residual | Fixed-state q / max pose FD error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 2.899058231e-8 | 91,427 | 74.113 | 9.950886836e-8 | 0.01704577790 | 2.1993e-6 / 1.1182e-5 |
| 1e-4 | 2.899058231e-7 | 23,380 | 19.455 | 9.842660803e-8 | 0.1130161888 | 5.9081e-8 / 9.6662e-6 |
| 1e-3 | 2.899058231e-6 | 8,554 | 24.268 | 9.859253313e-8 | 0.3687370797 | 6.7144e-7 / 5.8291e-6 |
| 1e-2 | 2.899058231e-5 | 1,869 | 3.611 | 9.532533778e-8 | 0.6632622141 | 1.8789e-7 / 1.0613e-6 |
| 1e-1 | 2.899058231e-4 | 235 | 2.240 | 8.862341648e-8 | 0.8420718940 | 5.5776e-5 / 7.3175e-7 |

The `1e-5` result is the smallest successful *tested* relative shift. It is
not an optimum claim: its original-unshifted residual is still 0.0170. The
first result in each probe includes cold sparse setup; later variants reuse
symbolic/topology caches. Elapsed times therefore do not establish a fair speed
comparison. The measured unshifted residual rises as the shift increases;
these are approximate gradient measurements, not evidence that the original
implicit derivative converged.

- [Probe001 Comet run](https://www.comet.com/liblaf/apple/cae550a56b23424e976a596da202d2b7)
- [Probe002 Comet run](https://www.comet.com/liblaf/apple/f028719754c84619a1f6f77213879000)

The saved gradient checks use central differences of the fixed-state
Lagrangian pullback: free coordinates remain fixed, fixed coordinates are
rebuilt from each perturbed normalized jaw coordinate, and a fresh contact
state is rebuilt. They cover one q direction and all six normalized pose basis
directions at epsilons `1e-4` and `1e-5`. They do not resolve neighbouring
forward equilibria. Receipts are
[`gradient-check-00.json`](../data/damped-adjoint-probe-001/gradient-check-00.json),
[`gradient-check-01.json`](../data/damped-adjoint-probe-001/gradient-check-01.json),
and [`gradient-check-02.json`](../data/damped-adjoint-probe-001/gradient-check-02.json).

The source trial itself remains physically invalid, with 554 inverted
tetrahedra and minimum `det(F) = -68.13730227683844`; damping does not change
that geometry finding. The inverse result remains unconverged.

The inverse runner defaults to `adjoint_relative_shift = 0.0`, preserving the
unshifted exact implicit path. Positive values select the separate approximate
backward and save both shifted and unshifted residuals in the receipt.

No inverse optimization was rerun. Damping is available in 130 through
`--adjoint-relative-shift 0.00001`; default zero remains the exact path.
The diagnostic table and lip-boundary audit are served at
the tailnet diagnostics page (private preview omitted).
All published source JSON and image bytes were verified over HTTP against their
local SHA-256 hashes. The current fit endpoint and geometry did not change.
