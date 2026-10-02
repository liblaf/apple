# Manual actuation and material sensitivity findings

## Question and controls

This experiment asks whether the current face model can transmit prescribed
muscle contraction into visible motion, and why the earlier FiberModes inverse
produced much less motion than the unconstrained Raw6 baseline.

The manual sweep activates three fixed muscle sets with the fixture's
geometry-estimated fibers: bilateral smile elevators, bilateral risorius, and
orbicularis oris. A prescribed natural fiber contraction `c` maps to
`a = -log(1-c)`. The constitutive input is the inverse active map

`Ainv = exp(a) ffT + exp(-a/2) (I-ffT)`.

Every prescribed `Ainv` has determinant one. It is separate from the solved
physical deformation `F`; `det(F)` is the actual local volume ratio, and the
muscle energy evaluates the elastic map `G = F @ Ainv`.

Each skin/pattern branch used 0 -> 10% -> 30% -> 50% as quasistatic numerical
continuation. These states are not a dynamic time history. Every pattern began
from rest, and corresponding skin/no-skin branches used byte-identical control
arrays. The solver used at most 10,000 steps with `rtol=1e-5` and `atol=1e-12`.
It imposed no geometric rejection, so a converged finite state would be retained
even if it contained inverted tetrahedra.

## Completed manual sweep

All 18 manual states reached strict finite equilibrium. None contained an
inverted tetrahedron; the minimum `det(F)` over all cases was 0.55161. The
complete [trace](../data/10-manual-activation/trace.csv), [summary](../data/10-manual-activation/summary.json), and per-case VTU/NPZ/diagnostic files are under
[`data/10-manual-activation`](../data/10-manual-activation/).

The strongest 50% states are:

| Pattern | Skin factor | selected muscle volume (m3) | weighted mean fiber stretch in `F` | surface RMS (mm) | Smile projection | lip RMS (mm) | mean lip radial XY (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smile elevators | 0 | 2.266e-6 | 0.93697 | 0.4901 | +0.05290 | 1.449 | +0.2827 |
| Smile elevators | .12 | 2.266e-6 | 0.94148 | 0.4061 | +0.04550 | 1.255 | +0.1900 |
| Risorius | 0 | 3.233e-7 | 0.91575 | 0.1612 | +0.00854 | 0.508 | +0.0112 |
| Risorius | .12 | 3.233e-7 | 0.92126 | 0.1275 | +0.00721 | 0.424 | -0.0071 |
| Orbicularis oris | 0 | 4.620e-6 | 0.61951 | 1.0669 | -0.07308 | 4.260 | -2.7364 |
| Orbicularis oris | .12 | 4.620e-6 | 0.63139 | 0.9806 | -0.06410 | 4.085 | -2.5194 |

Positive radial motion is outward and negative is inward. Orbicularis oris
therefore gives a large ring-closing response, while the prescribed smile
elevators produce only 5.29% of the supplied Smile target amplitude without
skin and 4.55% with skin. This demonstrates that the active material can drive
large directional deformation in this discretization. It does not validate the
muscle attachments, fiber estimates, missing contact, or anatomical fidelity.

At 50%, adding the existing skin membrane reduces surface RMS by 17.1% for the
smile set, 20.9% for risorius, and 8.1% for orbicularis. Skin is a measurable
load, but removing it does not recover the missing smile amplitude.

## Force transmission and coverage

The 50% no-skin smile state shortens the selected fibers by only 6.30% on a
muscle-fraction-volume-weighted mean, despite a prescribed 50% stress-free
contraction. Its selected cells are 32.86% muscle, 47.96% fat, and 19.18%
aponeurosis by volume. The exclusive vertex-adjacent cell ring is 9.66% muscle,
79.73% fat, and 10.61% aponeurosis. The active fibers therefore work inside a
larger passive mixture and a fat-dominated immediate neighborhood.

The smile set has 492 fixed vertices and 1,958 fixed-incident selected cells.
Neither the vertices nor the cells touch `CutBoundaryAddedFixed`; the September
cut additions are remote from these manually activated muscles. Cut fixation
cannot explain this manual smile response, although it can still alter the
whole-face inverse problem.

At the 50% no-skin smile equilibrium, the per-potential energy and fixed-force
resultant norms are:

| Potential | energy (J) | fixed resultant norm (N) |
| --- | ---: | ---: |
| Muscle | 0.01513 | 1.026 |
| Aponeurosis | 0.000762 | 0.630 |
| Fat | 0.000556 | 0.459 |

These forces are gradients of individual mesh energy components. They cancel
at equilibrium and are not measurements of anatomical muscle force. Their
relative size shows that both aponeurosis and fat materially oppose the active
component; force norms alone do not assign a single cause.

The current activation screen contains 120,020 tetrahedra, versus 288,235 in
the historical June control field. The current fixture fixes 33,636 vertices,
versus 27,036 historically; 6,600 are September cut additions. The
[historical readback](../data/11-historical-no-skin/summary.json) also shows that
the June state was saved under different controls and boundary conditions, so
its larger motion is context rather than a matched material counterfactual.

## Exact-control material sensitivities

Three no-skin variants replayed the exact saved 50% smile control array
(`SHA-256 39d6ca24e4c884b294b49efbd5d0a389ffb718d5af1ab9c5516d50dc92a876a5`)
from the baseline equilibrium. They are independent material re-equilibrations,
not independent rest-branch solutions. Each variant changed one declared
material array pair and retained all solved geometry, including inversions.

| Variant | surface RMS (mm) | ratio to baseline | Smile projection | weighted mean fiber stretch | min `det(F)` | inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.4901 | 1.00 | 0.05290 | 0.93697 | 0.6088 | 0 |
| Selected smile-cell muscle `lambda,mu` x10 | 2.1021 | 4.29 | 0.24707 | 0.77369 | -0.2262 | 3 |
| All fat `lambda,mu` x0.1 | 0.7575 | 1.55 | 0.07567 | 0.91210 | 0.4424 | 0 |
| All aponeurosis `lambda,mu` x0.1 | 0.8146 | 1.66 | 0.08842 | 0.89625 | -0.1424 | 1 |

All three strict solves succeeded. The [sensitivity summary](../data/11-manual-sensitivities-v2/summary.json) and exact material-array receipts are under
[`data/11-manual-sensitivities-v2`](../data/11-manual-sensitivities-v2/).
The selected-muscle variant scales both passive muscle stiffness and the force
generated by the same eigenstrain; it is not a pure active-stress multiplier.
The variants are diagnostic sensitivities, not calibrated material proposals.

The fourfold response to selected-cell muscle scaling, and the smaller but
clear responses to reducing either passive constituent, support an actuation to
surrounding-stiffness limitation. The inversions in the stronger variants also
show that simply increasing this ratio is not a usable fix by itself.

## Surface roughness on frozen geometry

The CPU-only [surface audit](../data/41-surface-roughness/summary.json) applies
the same rest-skin cotangent operator to immutable endpoints. It reports normal
displacement and target-residual high-pass fields at 2, 5, and 10 mm heat-kernel
scales, with natural Neumann conditions on the 707 open boundary edges. The
[endpoint manifest](41-surface-roughness-manifest.json) hashes every source
state and rejects `latest` paths.

| Endpoint | surface motion (mm) | Smile projection | full-face normal-displacement high-pass RMS at 2 / 5 / 10 mm (mm) |
| --- | ---: | ---: | ---: |
| Historical saved no-skin | 4.9741 | .96815 | .13295 / .28661 / .47319 |
| Manual c50 baseline | .4901 | .05290 | .01722 / .03904 / .06563 |
| Selected muscle `lambda,mu` x10 | 2.1021 | .24707 | .07032 / .15762 / .26433 |
| Fat `lambda,mu` x0.1 | .7575 | .07567 | .02563 / .05887 / .10079 |
| Aponeurosis `lambda,mu` x0.1 | .8146 | .08842 | .02896 / .06657 / .11403 |

The muscle-x10 state has 4.29 times the baseline motion and about 4.0 times its
high-pass RMS at each scale. This audit does not show a disproportionate rise
in high-pass content relative to total motion. The high-pass values remain
descriptive: the supplied target may itself contain real fine-scale motion, so
the metric cannot classify all high-frequency displacement as artifact.

## Why Raw6 can move much more

The [constitutive calculation](16-actuation-stress-mechanism.md) fixes `F=I`
and evaluates the production active energy. For the pure muscle constituent,
the 50% isochoric Fiber map produces a Piola stress norm of 0.02533 MPa. An
equal-`||Ainv-I||` isotropic Raw6 dilation produces 2.48863 MPa because it
engages the stable material's volumetric `lambda` term; the corresponding Raw6
compression produces only 0.00764 MPa. These are constitutive stresses before
`MuscleFraction` weighting.

This asymmetric comparison explains one capacity advantage available to
unconstrained Raw6. It does not imply that every volumetric Raw6 direction is
strong, and it does not turn `det(Ainv)` into a physical volume change. At the
fixed evaluation state, `det(F)=1`; only `det(G)=det(F)det(Ainv)` changes.

## Reproduction and limitations

The normal runs were:

```bash
cd exp/2026/09/07/face-actuation-diagnosis
CUDA_VISIBLE_DEVICES=0 \
COMET_AUTO_LOG_CODE=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="Manual face activation skin comparison" \
CHERRIES_TAGS="face,manual-activation,skin-comparison,diagnosis" \
.venv/bin/python src/10-manual-activation.py
```

Comet: <https://www.comet.com/liblaf/apple/31dc20981c7b44a697ea3c726084c34b>

```bash
cd exp/2026/09/07/face-actuation-diagnosis
CUDA_VISIBLE_DEVICES=0 \
COMET_AUTO_LOG_CODE=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="Exact-baseline smile material sensitivity replays v2" \
CHERRIES_TAGS="face,manual-activation,material-sensitivity,exact-control-replay,diagnosis" \
.venv/bin/python \
src/11-manual-sensitivities.py \
--output-dir data/11-manual-sensitivities-v2
```

Comet: <https://www.comet.com/liblaf/apple/83df19a1bc484f0c86e9a6e5338c655a>

The first sensitivity launch stopped before any solve because its receipt
helper requested a nonexistent material-dictionary key. That failure is
preserved in [`data/11-manual-sensitivities`](../data/11-manual-sensitivities/launch-failure.json); the corrected run used the new `-v2` output directory.

The fixture omits contact and uses geometry-estimated unoriented fiber lines.
The target is an externally supplied surface displacement, not a measured
muscle activation. These results establish numerical response and matched
material sensitivities for this fixture; they do not identify anatomically
correct muscle strength or a final production parameter set.
