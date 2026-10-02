# Fixed-control material and Depressor labii replays

## Decision

Use stable Neo-Hookean fat with `nu=0.49` for the first matched inversions. At
the exact saved FiberRegion B controls, this change raises the minimum
`det(F)` from 0.200012 to 0.526828, removes all cells below 0.5, and changes the
surface fit by only +0.000752 mm RMS. It converges from rest in 1,359 inner
steps and 21.5 seconds.

Logarithmic Neo-Hookean fat at the same `nu=0.49` gives a slightly higher
minimum `det(F)` of 0.545572, but takes 3,873 steps and 61.1 seconds. Its fit
differs from the stable `nu=0.49` result by only 0.000096 mm. This small endpoint
difference does not justify its roughly threefold solve cost for the first
matched inverse comparison.

The Depressor labii counterfactual supports the load-path diagnosis. Zeroing
only its two saved controls raises the minimum `det(F)` to 0.533243, while
reducing target projection from 0.04765 to 0.04150 and worsening the fit by
0.03188 mm RMS. Depressor labii therefore contributes to both the intended
surface motion and the severe compression near the fixed lower-mouth socket.
The failed 50% case prevents a dose-response claim.

## Controlled states

Every replay starts from exactly zero displacement. The baseline, stable
`nu=0.49`, and Neo `nu=0.49` files reproduce the saved `q` and `A_inv` arrays
from `21-fiber-region-B-v2/final.npz` exactly. The zero-Depressor case changes
only control IDs 24 and 25, which map to muscle IDs 162 and 163; its `A_inv`
differs in 2,336 active tetrahedra.

The original final state and fresh-rest baseline also agree geometrically. The
largest all-vertex displacement difference is 0.0236 micrometres, their
minimum `det(F)` values differ by `7.33e-7`, and every determinant-threshold
count is identical. The low determinant is therefore reproduced from rest and
is not evidence of a warm-start-dependent equilibrium branch.

The volume material constants use their actual infinitesimal `E,nu`. Stable
Neo-Hookean uses `lambda_code=lambda_classical+mu`; logarithmic Neo-Hookean uses
the classical Lamé coefficient. Aponeurosis remains stable at `nu=0.35`, muscle
remains stable-active at `nu=0.46`, skin remains plane-stress at `nu=0.46`, and
contact remains disabled. Only the named case variable and the two Depressor
controls change.

## Endpoint comparison

| state | inner solve | fit RMS (mm) | motion RMS (mm) | target projection | min `det(F)` | cells below 0.5 / 0.8 / 0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| saved 21-v2 final | saved | 4.870963 | 0.482293 | 0.047647 | 0.200011 | 4 / 56 / 327 |
| fresh baseline, stable fat `nu=0.46` | 1,192 steps, 19.9 s | 4.870964 | 0.482292 | 0.047647 | 0.200012 | 4 / 56 / 327 |
| Depressor labii 0%, stable fat `nu=0.46` | 1,159 steps, 18.9 s | 4.902843 | 0.474221 | 0.041499 | 0.533243 | 0 / 43 / 249 |
| stable fat `nu=0.49` | 1,359 steps, 21.5 s | 4.871716 | 0.467510 | 0.047235 | 0.526828 | 0 / 26 / 121 |
| Neo fat `nu=0.49` | 3,873 steps, 61.1 s | 4.871812 | 0.466675 | 0.047202 | 0.545572 | 0 / 20 / 112 |

Raising stable-fat `nu` from 0.46 to 0.49 is a fixed-control, fixed-model test.
It removes the extreme lower tail: the volume-weighted 0.01% quantile rises
from 0.910122 to 0.942369. The minimum moves from cell 662949, a 89.45%
fat cell carrying identity activation beside the fixed lower-mouth socket, to
inactive 97.66% fat cell 691951 in the same neighborhood. The constitutive
change improves this local passive-tissue response without adding activation
capacity.

Changing only the fat energy from stable to logarithmic Neo at `nu=0.49`
raises the minimum by another 0.018744 and reduces the number of cells below
0.8 from 26 to 20. The limiting cell then moves to active Depressor septi
muscle cell 644127. This is a real but small fat-model effect at the endpoint;
the much larger improvement comes from the `nu=0.49` compressibility setting.

Zeroing both Depressor labii controls at stable-fat `nu=0.46` also moves the
minimum from cell 662949 to passive fat cell 691951. This identifies the saved
Depressor labii activation as a driver of the original fixed-socket
compression through surrounding tissue. It does not establish that the
muscle label, fiber direction, or mandible fixation is individually wrong.
Those factors remain coupled in this one fixture.

## Surface and intersection checks

All five endpoints have zero exact IPC edge-triangle intersections on the full
128,172-triangle tetrahedral boundary and the visible 29,899-triangle skin.
The rest surfaces are also clear, and no endpoint adds or resolves a hit. These
are static endpoint tests; they do not perform continuous collision detection
between solver states.

The material changes do not introduce a high-frequency surface penalty. The
area-weighted scalar normal-motion high-pass RMS values are:

| state | 2 mm face / mouth | 5 mm face / mouth | 10 mm face / mouth |
| --- | ---: | ---: | ---: |
| fresh baseline | 0.01262 / 0.03194 | 0.03378 / 0.08326 | 0.07601 / 0.18418 |
| Depressor labii 0% | 0.01453 / 0.03695 | 0.03634 / 0.09006 | 0.07825 / 0.18979 |
| stable fat `nu=0.49` | 0.01242 / 0.03138 | 0.03324 / 0.08162 | 0.07473 / 0.18029 |
| Neo fat `nu=0.49` | 0.01240 / 0.03132 | 0.03320 / 0.08152 | 0.07467 / 0.18015 |

Values are millimetres. Stable and Neo fat at `nu=0.49` are marginally quieter
than the baseline at all three scales. The zero-Depressor endpoint is slightly
less smooth, although the absolute values remain small. Residual high-pass
values are nearly unchanged because all four fixed-control fits still leave
most of the supplied target unresolved.

## Numerical failures

Two cases intentionally have no endpoint and must not be interpreted as
physical instability:

- The 50% Depressor labii case exhausted all 30 strict Armijo halvings after
  17.49 seconds. At the rejected final trial, `alpha=2.12965e-10`,
  `f0=7.719504328882560e-9`, and `f_alpha=7.719504328882616e-9`. The energy
  difference is `5.6e-23`, consistent with a numerical decrease-resolution
  floor. The strict replay stopped instead of committing that rejected trial.
- Neo fat at `nu=0.46` reached the 10,000-step forward budget after 176.86
  seconds. Its final gradient norm was `1.23354e-9`; the last Armijo search
  succeeded. It missed the requested `rtol=1e-6`, `atol=1e-13` convergence
  test by the step budget. This is slow numerical convergence, not an endpoint
  geometry result.

No automatic fallback, warm start, tolerance relaxation, or material change
was applied to either failure.

## Evidence

The replay summary and per-case solver/material receipts are in
[`data/22-replays-v2`](../data/22-replays-v2/summary.json). Four successful
cases contain `final.npz` and `final.vtu`; all arrays are finite and every
unchanged-control case records an exact saved-`A_inv` match. The replay
manifest covers 26 files and was rechecked against current SHA-256 values and
sizes.

The independent CPU audit is
[`data/42-material-replays-audit`](../data/42-material-replays-audit/summary.json).
Its 28-file manifest was also rechecked. Case directories contain the complete
volume fields, visible-surface high-pass fields, metrics, and reusable face and
mouth renders. The audit ran with:

```bash
DEBUG=1 uv run python src/40-audit-face-results.py \
  --result-dirs 'data/21-fiber-region-B-v2,data/22-replays-v2/baseline-B,data/22-replays-v2/depressor-labii-0pct,data/22-replays-v2/fat-stable-nu049,data/22-replays-v2/fat-neo-nu049' \
  --output-dir data/42-material-replays-audit
```

The run used CPU geometry, sparse diffusion, and IPC checks only and completed
in 39.86 seconds. Its endpoint checks do not establish convergence between
saved states or safety along a future inverse trajectory.
