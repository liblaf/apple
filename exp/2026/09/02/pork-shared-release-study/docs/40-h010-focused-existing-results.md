# Focused 2-D streaky-pork study: saved h = 0.10 results

## Scope and evidence boundary

This is the focused report requested for the `1.0 × 0.1` specimen, with the
muscle band at `y = 0.04 … 0.06`. It uses **only existing saved histories**;
no physics solve was launched for this report. The target is the upward
parabola `u_y = h · 4(x/L)(1-x/L)`, with the bottom and both sides fixed in
both in-plane components. The top and interior are free.

The 2-D mesh contains 2,000 triangles (400 muscle and 1,600 fat), with
100 × 10 rectangular cells. Fat has `E = 0.003 MPa`, muscle has
`E = 0.03 MPa`, Stable Neo-Hookean elasticity is used, and no skin energy is
present. The inverse loss is L2 over the free top-node x/y components.

Important dimensional correction: a symmetric 2-D activation tensor has
**3 DoF per triangle**, not 6. Thus the independent control has
`400 × 3 = 1,200` DoF; a shared muscle control has 3 DoF total. Six DoF is
the corresponding symmetric 3-D tetrahedral tensor parameterization.

## Requested comparison matrix

| Requested result | Exact saved history | Status |
| --- | --- | --- |
| h=.10, ν=.49, independent per-muscle-triangle activation | [`height-high`](../../../../08/31/unreachable-pork-factor-study/data/10-pork-2d/height-high/summary.json) | Available |
| h=.10, ν=.35, independent activation | — | Not saved; no substitution made |
| h=.10, ν=.49, one shared 3-DoF activation | — | Not saved; no substitution made |
| h=.10 shared → independent warm start | — | Not saved; no substitution made |
| h=.20, ν=.49, one shared 3-DoF activation | [`h020-shared`](../data/20-canonical-h020/h020-shared/summary.json) | Available, but nonstationary |
| h=.10 vs h=.20, both shared | h=.10 side is absent | Not a valid comparison yet |
| h=.10, ν=.49, independent activation with L1 or L∞ loss | — | Not saved; no substitution made |

The older `l100-*` histories are deliberately excluded: `l100` identifies
the **length** `L=1.0`, while their saved target height is `h=.05`, not `.10`.

## Available h=.10 baseline

The exact saved h=.10 baseline is `height-high`: ν=.49, 1,200 independent
activation DoF, L2 loss, Stable Neo-Hookean elasticity, and no activation
bounds, regularizer, determinant constraint, inversion repair, contact, or
skin energy.

| quantity | saved value |
| --- | ---: |
| history states | 1,237 |
| initial L2 objective | 5.387205e-3 |
| final L2 objective | 2.222503e-5 |
| final top target RMS | 4.714343e-3 |
| final top target MAE / max | 3.135727e-3 / 2.156623e-2 |
| final equilibrium residual RMS | 7.289074e-11 |
| final reduced-gradient ∞ / RMS | 3.048481e-5 / 1.476465e-6 |
| forward / inverse evaluation failures | 0 / 0 |
| final minimum det(F) | −45.3323 |
| final minimum det(G) / det(A⁻¹) | −4.59021e-2 / −0.542936 |

Every saved forward evaluation converged, so the trajectory is renderable.
It did **not** satisfy strict inverse stationarity: refinement ended at its
stalled-restart limit and the final reduced gradient exceeds the recorded
`2e-8` infinity-norm target. The negative determinants are allowed artifacts
of the requested unconstrained formulation, not a claim of a physical muscle
configuration.

## The only saved shared high-target reference

`h020-shared` is kept solely as the saved `h=.20` shared reference. It has
one group covering all 400 muscle triangles (3 total activation DoF), ν=.49,
and the same geometry/material/loss family. It has 1,219 states, zero saved
forward/inverse failures, final L2 objective `1.019509e-2`, and final top RMS
`1.009707e-1`. Its final reduced-gradient ∞ norm is `5.659745e-3`, far above
the `2e-8` threshold; this saved history is nonstationary and must not be
ranked as a converged shared solution.

Because no h=.10 shared history exists, neither its shared square nor a
shared h=.10 → h=.20 comparison can be inferred from these data. The shared
activation square that can be rendered is therefore explicitly labelled
**h=.20, nonstationary**; it is not a h=.10 surrogate.

## Separate controlled loss comparison: h=.05 only

No h=.10 L1/L2/L∞ trio is saved. There is, however, a **separate controlled
h=.05 trio** in the earlier 2-D study. All three have the same `1.0 × 0.1`
domain, 100 × 10 mesh, band at `y=.04 … .06`, ν=.49, Stable Neo-Hookean
elasticity, 400 muscle triangles with 3 independent DoF each (1,200 total),
and no skin energy or activation/determinant constraints. Only the loss
changes. It is therefore useful for loss behavior, but it is not evidence
for the h=.10 baseline or its missing Poisson/shared-control comparisons.

| loss | saved history | states | final native objective | final top RMS | final top max | final gradient ∞ | solver status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| L2 | [`baseline`](../../../../08/31/unreachable-pork-factor-study/data/10-pork-2d/baseline/summary.json) | 1,245 | 2.430807e-6 | 1.559105e-3 | 3.712281e-3 | 3.078422e-5 | stalled restart; nonstationary |
| L1 | [`loss-l1`](../../../../08/31/unreachable-pork-factor-study/data/10-pork-2d/loss-l1/summary.json) | 1,406 | 4.390134e-4 | 8.708227e-4 | 4.053700e-3 | 3.996691e-4 | stalled restart; nonstationary |
| L∞ | [`loss-linf`](../../../../08/31/unreachable-pork-factor-study/data/10-pork-2d/loss-linf/summary.json) | 1,491 | 3.542454e-3 | 2.762894e-3 | 3.542454e-3 | 4.465623e-3 | stalled restart; nonstationary |

All three saved trajectories report zero forward and inverse evaluation
failures, and final equilibrium residual RMS values below `1e-10`. Their
native objectives are different norms and are not mutually comparable; the
common output-space RMS and maximum errors are shown only as descriptive
endpoint diagnostics. All three allow inversion and reach negative
determinants during their saved histories, so they remain numerical artifacts
rather than physically admissible muscle solutions.

## Visual deliverables

The accompanying renderer is restricted to material colors only: fat and
muscle are distinct filled triangles with thin framework edges; no metric is
encoded by color. It produces one video frame per saved state at 30 FPS and
a loss curve for each available history.

Expected rendered artifacts (created from saved frames only):

- h=.10 independent: `../data/70-focused-h010-existing-results/h010-direct/`
  (`evolution.mp4`, `final-shape.png`, loss curve, and render receipt).
- h=.20 shared reference: `../data/70-focused-h010-existing-results/h020-shared/`
  (`evolution.mp4`, `final-shape.png`, loss curve, shared activation square,
  and render receipt).
- Separate h=.05 loss trio: `../data/70-focused-h010-existing-results/h005-loss-{l2,l1,linf}/`
  (one material-only 30-FPS video, final shape, loss curve, and receipt per
  saved history).

The matrix deliberately has no empty videos or invented curves for the four
missing h=.10 comparisons. Once a matching saved history is supplied, it can
be added without changing the report boundary.

## Separate future work

Results outside the focused matrix remain in the separate
[later-consideration list](30-experiments-for-later-consideration.md). They
are not used for any conclusion above.

## Provenance

- h=.10 direct source: `exp/2026/08/31/unreachable-pork-factor-study/data/10-pork-2d/height-high/`.
- h=.20 shared source: `data/20-canonical-h020/h020-shared/`.
- h=.05 L1/L2/L∞ sources:
  `exp/2026/08/31/unreachable-pork-factor-study/data/10-pork-2d/{baseline,loss-l1,loss-linf}/`.
- Configuration evidence for the h=.10 source:
  `exp/2026/08/31/unreachable-pork-factor-study/src/10-run-pork-2d.py`.
- No physics execution was performed while preparing this report.
