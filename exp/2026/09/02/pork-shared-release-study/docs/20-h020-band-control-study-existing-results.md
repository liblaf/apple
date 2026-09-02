# 2-D pork, h = .20: band-muscle control study

## Scope and status

This note uses **existing saved results only**.  No inverse or forward physics
was rerun for this report.  It is deliberately limited to the requested
long-pork, band-muscle comparison:

- direct independent activation at \(\nu=.49\);
- the requested \(\nu=.35\) comparison, which is not available among the
  exact matched saved results;
- shared activation at \(\nu=.49\); and
- shared-then-independent activation at \(\nu=.49\), initialized from the
  saved shared endpoint.

All three saved endpoints included here are **nonstationary** under their
recorded convergence criteria: their tail/stationarity gates are false and
refinement stopped at the stalled-restart limit.  Therefore their endpoint
numbers and images describe recorded optimization paths, not convergence-
certified optima.  The shared-to-independent result is additionally marked
**exploratory**: it is a warm-start probe, not a converged comparison.

## Common physical problem

The rest domain is \([0,1]\times[0,.1]\), discretized as \(100\times10\)
rectangles (2,000 triangles).  The active band is \(.04\le y\le.06\): 400
muscle triangles and 1,600 fat triangles.  The bottom and two sides are fixed
in \(x,y\); the top and interior vertices are free.  The target vertical
displacement is

\[
u_y(x)=.2\,4x(1-x),
\]

so the two fixed side endpoints remain compatible with the parabola.  Both
materials use Stable Neo-Hookean elasticity, with fat \(E=.003\) MPa and
muscle \(E=.03\) MPa; both use \(\nu=.49\).  The loss is L2 on the free top
nodes' \(x,y\) components.  There is no skin energy, no activation bound or
regularizer, and no determinant/inversion constraint.

This is 2-D: **each muscle triangle has three activation degrees of freedom**.
The six-DoF statement belongs to a 3-D muscle tetrahedron, not this model.

## Existing matched histories

| protocol | activation parametrization | final L2 objective | top target RMS | final inverted-cell fraction | qualification |
| --- | --- | ---: | ---: | ---: | --- |
| Direct independent | 400 groups × 3 = 1,200 DoF | 0.00017849 | 0.013360 | 3.80% | Nonstationary saved endpoint |
| Shared | 1 group × 3 = 3 DoF | 0.01019509 | 0.100971 | 0.10% | Nonstationary saved endpoint |
| Shared → independent | 400 groups × 3 = 1,200 DoF | 0.00016128 | 0.012700 | 3.90% | Nonstationary exploratory warm start |

Every recorded accepted evaluation in these histories had a successful forward
solve; the saved summaries report zero forward, adjoint, inverse, nonfinite,
and refinement-trial-forward failures.  That reliability should not be
mistaken for inverse convergence: the final gradient gates and the 1%-tail
gates remain unmet in all three cases.

The shared result applies one identical 3-component activation vector to all
400 muscle triangles.  Its activation-neighbour jump is consequently zero,
but its endpoint fit is much poorer than either independent-control history.
Releasing the shared endpoint reproduces that endpoint exactly at handoff,
then expands the controls from 3 to 1,200 and starts the forward solve from the
stored strict shared displacement.  It produces a lower recorded objective
than direct independent activation, but it also returns to extensive folding;
because it is exploratory and nonstationary, this is a useful branch
observation rather than a ranking or a claim of improvement.

## \(\nu=.35\) comparison

There is no exact \(h=.20\), \(L=1\), band-muscle, otherwise-matched
\(\nu=.35\) saved endpoint in this study.  It is intentionally shown as
**unavailable**, not substituted with a different target, geometry, material
layout, loss, or optimizer horizon.  A valid comparison must be created later
from the same specification and assessed with the same stationarity criteria.

## Rendered evidence

The focused renderer uses only fat/muscle colour, a thin charcoal triangle
framework, and a thin pink target outline.  It does not colour by determinant,
loss, activation, or another metric.  The shared-activation square uses the
single 3-DoF control to make the spatially constant activation explicit in
both rest and deformed geometry.

The loss plot is a separate, monochrome visualization of the existing
`trace.csv` rows.  Its left panel compares each case on its own saved-step
axis; its right panel concatenates the shared path with its release and marks
the handoff.  It is not a convergence certificate: all paths are labelled
nonstationary, and the release remains exploratory.

![Saved L2 loss histories: local and shared-to-release cumulative views](../data/62-focused-h020-loss-curves/loss-curves.png)

- [Final three-way shape comparison](../data/60-focused-h020-materials/final-comparison.png)
- [Shared-activation square: rest and deformed](../data/60-focused-h020-materials/shared-activation-square.png)
- [Direct independent evolution, 30 FPS](../data/60-focused-h020-materials/direct/evolution.mp4)
- [Shared evolution, 30 FPS](../data/60-focused-h020-materials/shared/evolution.mp4)
- [Shared-to-independent evolution, 30 FPS](../data/60-focused-h020-materials/shared-release/evolution.mp4)
- [Focused-render receipt](../data/60-focused-h020-materials/render-receipt.json)
- [Loss-curve receipt](../data/62-focused-h020-loss-curves/receipt.json)

## What this report supports

At \(h=.20\) on this long band-muscle model, enforcing one shared activation
substantially suppresses spatial control variation and leaves a much larger
shape error in the saved endpoint.  Releasing that shared initialization
recovers a close recorded fit, but does not avoid folding.  The present
evidence does **not** establish how \(\nu=.35\) changes either effect, nor
does it identify a converged optimum for any protocol.
