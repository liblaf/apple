# 2-D pork experiments for later consideration

This is a holding list, separate from the focused \(h=.20\) band-control
report. The items below may have existing artifacts, but their numerical
findings are intentionally not mixed into that report because they change
more than the requested control protocol or Poisson-ratio comparison.

| experiment family | purpose when revisited | keep separate because |
| --- | --- | --- |
| Higher upward target, \(h=.30\) | Test a more demanding unreachable parabola. | Target amplitude changes. |
| Downward target | Test sign asymmetry by pushing the parabola downward. | Target direction changes the nonlinear branch. |
| \(h=.05\) factorial | Compare long/short domains, band/full muscle layouts, per-cell/shared controls, and \(\nu=.35/.49\). | Multiple interacting factors change together. |
| Full-muscle variants | Establish what happens when fat is removed from the active region. | The mechanism is no longer a fat-plus-muscle band. |
| Short \(.1\times.1\) variants | Test a compact pork geometry with the same nominal muscle band. | Length, curvature scale, mesh, and boundary relation differ. |
| OFAT mesh study | Compare practical mesh resolutions, including a dense case. | Discretization and number of controls change. |
| OFAT elasticity study | Compare linear elasticity with Stable Neo-Hookean. | Constitutive model changes. |
| Zero-displacement warm-start branch | Contrast releasing tiled shared controls from zero displacement rather than the stored shared displacement. | It is a separate branch-selection probe and is labelled exploratory/nonstationary. |
| Other historical variants | Preserve prior alternative targets, layouts, material choices, or diagnostic renderings for traceability. | They are not matched to the present h=.20 band-control question. |

When any of these are promoted into a main comparison, define the exact
matched baseline first, reuse the same saved-step/rendering contract, and
report both forward-solve reliability and inverse stationarity rather than
only the endpoint shape error.
