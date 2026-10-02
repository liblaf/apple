# Released-axis smoothness continuation

Neither 3x nor 10x passes all three predeclared criteria. Both reduce directional roughness by at least 50%, but exceed the allowed 5% increase in position RMS. No stronger coefficient is selected.

The limits are direction roughness at most 50% of new 1x, and both position and normal RMS at most 105% of new 1x. The position limit is 1.2488 mm.

This report compares released-axis rank-one endpoints initialized from the same fixed-axis parent state. Geometry metrics were recomputed from each saved displacement; no forward solve was run. Each branch used the full 200-update budget; that records budget completion, not convergence certification. The endpoints retain their numerical solver-validity and inversion diagnostics below.

![Shapes and principal activation at a shared camera and scale](../data/30-comparison/smoothness-comparison-preview.png)

[Full-resolution comparison (10240 x 5760)](../data/30-comparison/smoothness-comparison-16x9.png)

## Exact roughness split

For each graph edge, `S_i = a_i v_i v_i^T` gives `||S_i-S_j||_F^2 = (a_i-a_j)^2 + 2 a_i a_j [1-(v_i^T v_j)^2]`. The table sums the amplitude and unoriented-axis terms with the saved graph conductance and regularizer factor. Directional reduction is relative to the new 1x endpoint.

Amplitude statistics use normalized active-volume weights; the RMS amplitude equals the active-strain tensor Frobenius RMS for unit axes. `Direction at parent amplitude` recomputes the direction term using the historical parent amplitudes and current axes, as a diagnostic only; selection still uses the actual field's direction term.

## Selection metrics

| Multiplier | Direction / 1x | Fit (mm) | Fit / 1x | Normal (deg) | Normal / 1x | Meets all criteria |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1x | 1.000 | 1.1893 | 1.000 | 3.9403 | 1.000 | False |
| 3x | 0.483 | 1.3018 | 1.095 | 3.9265 | 0.997 | False |
| 10x | 0.209 | 1.5468 | 1.301 | 3.6545 | 0.927 | False |

## Amplitude and roughness

| Multiplier | Mean amplitude | RMS amplitude | p95 amplitude | Max amplitude | Amplitude roughness | Direction roughness | Direction at parent amplitude |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 3.309 | 5.625 | 12.09 | 54.88 | 363.595 | 1253.6 | 1452.83 |
| 3x | 3.119 | 5.278 | 11.3 | 54.81 | 225.591 | 605.17 | 851.352 |
| 10x | 2.945 | 4.951 | 10.62 | 54.89 | 107.194 | 261.959 | 445.555 |

## Numerical diagnostics

| Multiplier | Adam updates | Skipped proposals | Gradient ratio | Inverted cells | min det(F) | solver_valid |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1x | 200 | 0 | 0.168 | 141 | -4.4185 | False |
| 3x | 200 | 0 | 0.286 | 90 | -2.9572 | False |
| 10x | 200 | 0 | 0.4462 | 125 | -3.6961 | False |

The gradient ratio is `||d(eta R)/dS||_* / ||d L2/dS||_*` in the full symmetric activation tensor S, with `||g||_* = sqrt(sum_i ||sym(g_i)||_F^2 / m_i)` and normalized effective active-cell volumes `m_i`. The numerator includes the smoothness coefficient; the denominator excludes the normal loss.

The saved endpoint diagnostic performs a fresh approximate forward solve and a separate L2 adjoint at the final controls. Its displacement can differ from the final Adam checkpoint used for the geometry metrics above. These ratios therefore describe approximate gradient diagnostics, not certified equilibrium sensitivities.

| Multiplier | Weighted smoothness gradient norm | L2 gradient norm | Diagnostic forward force norm | L2 adjoint relative residual |
| ---: | ---: | ---: | ---: | ---: |
| 1x | 0.000919917 | 0.00547532 | 2.90976e-07 | 0.281402 |
| 3x | 0.00181846 | 0.0063572 | 2.557e-07 | 0.327744 |
| 10x | 0.00367987 | 0.0082464 | 2.24868e-07 | 0.385051 |

All diagnostic forward and L2 adjoint solves miss their requested tolerances: force norm 1e-10 and adjoint relative residual 1e-7. The raw receipts are in each branch's `stage/gradient-balance.json`; all final checkpoints have `solver_valid=false` and inverted cells. Completing 200 Adam updates does not establish mechanical validity.

## Historical parent

The fixed-axis parent has amplitude roughness 372.171 and direction roughness 727.294 on the same saved graph. Its fit and normal metrics are 0.9668 mm and 2.2962 degrees.

Initialization audit: maximum tensor error from the parent is 2.84e-14; maximum cross-run initial displacement difference is 5.58e-11 (tolerance 1e-10).

![Fit, normal, and graph smoothness trajectories relative to the 1x run](../data/20-analysis/relative-trajectories.png)

## Interpretation limits

`solver_valid`, minimum determinant, and inverted-cell count are reported as numerical diagnostics only. This analysis makes no physical-validity claim. The directional term is sign-invariant and amplitude-weighted; zero-amplitude cells contribute no directional roughness.

Parent checkpoint SHA-256: `c1b181e429ceddd1a27f41e1ed193e98ac47987e4fcf6e4f7d50641314faee18`. Shared mesh SHA-256: `ec8b0ee87fbcb498d08cb8c42138ed462aae5de1524305fa70aa7b0b1f0ca6de`.
