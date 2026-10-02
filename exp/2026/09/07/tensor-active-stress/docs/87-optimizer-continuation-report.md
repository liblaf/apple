# PSD continuation and matched regularization

## Main result

Continuing the full-field PSD inverse from global step 64 to 256 lowered
area-weighted fit RMS from **3.8204 to 2.8247 mm**, a **26.06% reduction**.
Motion RMS rose from 1.5596 to 2.7088 mm. The saved final state has no inverted
tetrahedra and `detF_min = 0.11393`; these geometric diagnostics do not establish
physiological validity. The fit-only run ended at its declared update budget,
with projected-gradient RMS and maximum still about 46–48% of their step-64
values. It has not established inverse convergence or target reachability.

The reduced-epsilon face probe stopped before a solved post-update state because
it failed the frozen coordinate-wise replay check. Thus the face result comes
from continuing the historical baseline settings, `epsilon = 0.01` and
`learning_rate = 0.3`. No accepted calibrated face trajectory exists to compare
against it. The [frozen protocol](72-optimizer-protocol.md) defines the selection
and stopping rules; all 288,235 active tetrahedra and 1,729,410 tensor coordinates
were retained.

## Small known-reachable recovery

The small study used the validated two-tetrahedron mixed-material fixture from
[source 12](12-small-model-study.md), with two active tetrahedra and the same
PSD tensor coordinates, spectral box, and constitutive form used for the face
experiment. Its target was generated from a strictly interior PSD control and
an accepted equilibrium, so it was suitable for testing optimizer recovery but
not for making anatomical or unique-control claims. The complete receipt is
[here](../data/70-small-recovery-v4/summary.json), with the standalone
[screen figure](../data/77-optimizer-diagnostics/screen-256-diagnostics.pdf)
([PNG](../data/77-optimizer-diagnostics/screen-256-diagnostics.png)) and
[recovery figure](../data/77-optimizer-diagnostics/recovery-full-diagnostics.pdf).

![Known-reachable recovery with one learning-rate recalibration](../data/77-optimizer-diagnostics/recovery-full-diagnostics.png)

Three 256-update screen arms were completed with their first projected physical
stress update calibrated before recovery outcomes were examined. The selected
candidate used Adam epsilon `1e-6` and initial learning rate
`0.0035403437409342554`. Epsilon reduction alone did **not** recover the known
target within the prescribed first 4,096 updates: the relative displacement
error at that point was `0.06443`, above the `0.001` acceptance threshold. A
documented amendment, fixed before the face probes, allowed one learning-rate
recalibration at that point while preserving
`q`, displacement, both Adam moments, the scalar counter, target, loss, and
`epsilon = 1e-6`. It changed the learning rate to `0.1205714831013498` to
restore the declared projected physical update size.

The retained run then met the recovery criterion at step 5,371: relative
displacement error `0.0009998673`, with 5,372 accepted forward states. This
provides a displacement-recovery witness for this small system after the
single permitted learning-rate recalibration. The fixed initial rate had not
passed the gate at update 4,096. It does not transfer that learning rate to the face problem. The
face calibration below independently chooses its own rate from the frozen face
gradient and saved Adam state.

## Face step-64 calibration

The [face calibration receipt](../data/73-face-step-calibration-v2/summary.json)
replayed the saved step-64 face state with the existing forward and adjoint
tolerances. The replay passed: the area-fit RMS difference was
`3.71e-10 mm`, the area-motion RMS difference was `5.72e-09 mm`, and the data
objective difference was `6.15e-09 mm²`. The forward solve reported
`primary_success`; the adjoint also reported success. A scalar replay check was
corrected before either face probe: the loss is the unweighted fit RMS squared
divided by three, so its tolerance is derived from the existing `1e-6 mm` RMS
tolerance. The [protocol amendment](72-optimizer-protocol.md) records the
original rejection and exact differences; no forward or adjoint tolerance was
changed.

Using the same true gradient and Adam moments, the calibration selected epsilon
`1e-6` and learning rate `0.0033136508893221615`. Its actual projected physical
stress-update RMS was `0.00010592238795880169 MPa`,
`2.000000069` times the baseline first update. The installed Adam replay was
close to the closed-form shadow update at this calibration stage: maximum
control difference `1.78e-15`, with zero relative physical-update error in the
receipt. These checks freeze a proposed next update; they do not show that a
subsequent nonlinear face equilibrium has succeeded.

## The two 16-update probe outcomes

The calibrated probe was launched with the frozen epsilon and calibrated rate.
It retained the valid step-64 record but failed its strict first-update replay
assertion before a step-65 forward state was accepted. Its
[failure receipt](../data/74-calibrated16/failure.json) records an actual
physical-update relative error of `6.43e-08`, within the physical-update
tolerance, but a control maximum error of `3.93e-06` against a `1e-10`
tolerance. The last forward receipt is the step-64 recomputation and reports
`primary_success`, one solver step, and gradient norm `9.72e-11`. No accepted
post-update equilibrium, fit, motion, deformation determinant, or inversion
metric exists for this arm. The strict replay failure therefore disqualifies the
calibrated branch under the protocol; it should not be described as a physical
solver failure, a failed expression, or an optimizer-convergence result.

The historical baseline branch used epsilon `0.01` and learning rate `0.3`.
It completed all 16 real updates and its [summary](../data/74-baseline16/summary.json),
[trace](../data/74-baseline16/trace.csv), and saved volume states are retained. Area-fit RMS fell
from `3.8204402 mm` at step 64 to `3.6721150 mm` at step 80. Area-motion RMS
rose from `1.5596440 mm` to `1.7284972 mm`. The saved endpoint has
`detF_min = -0.0335459`, `detF_max = 1.9797187`, and one inverted tetrahedron;
these are reported side by side rather than used to rewrite the optimizer
selection rule. The endpoint remains a fixed-budget result with no inverse
convergence claim.

The [probe decision receipt](../data/81-probe-decision/summary.json) retained
the baseline checkpoint because the calibrated branch failed its numerical
validity contract. It names
`data/74-baseline16/optimizer-latest.pt` as the selected checkpoint and
specifies a 64-update next block to global step 144. This is a continuation
choice under the protocol, not a statement that the baseline endpoint is
physically acceptable.

## Fit-only continuation

The baseline checkpoint continued without changing its learning rate, epsilon,
Adam moments, target, material model, stress box, or solver tolerances. Each
block starts from the previous accepted displacement and exact optimizer state.
The stopping checks permitted blocks ending at steps 144 and 208, followed by
the bounded final block to step 256.

| Global step | Fit RMS (mm) | Motion RMS (mm) | Minimum detF | Inverted tets |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3.820440 | 1.559644 | -0.011456 | 1 |
| 80 | 3.672115 | 1.728497 | -0.033546 | 1 |
| 144 | 3.254825 | 2.208338 | 0.003938 | 0 |
| 208 | 2.980846 | 2.526572 | 0.067026 | 0 |
| 256 | 2.824686 | 2.708769 | 0.113930 | 0 |

Step 64 is the original saved endpoint. Later rows are the actual final states
of each completed continuation block. Inversions were recorded throughout and
never used to reject a trial or select a branch. The [final fit-only summary](../data/74-fit256/summary.json)
and [budget-stop decision](../data/81-fit256-decision/summary.json) retain the
endpoint and its stopping evidence. This continued progress shows that the
64-update result was unfinished optimization; it does not prove that the PSD
model can reproduce the target exactly.

## Regularization at equal and matched states

Three 16-update branches started from the exact step-256 fit-only checkpoint,
including its Adam moments and counter. The control retained the fit-only loss;
the other branches added smoothness weights `1.4762928671047126` and
`5.9051714684188505`. All 51 recorded states had successful forward and adjoint
solves and no inverted tetrahedra.

At equal local step 16, tensor variation falls strongly, but fit and motion
also change. The 5 mm surface high-pass ratio is the full-face RMS of the
high-pass normal displacement divided by the RMS of that same normal field;
it is not raw roughness amplitude or a physiological score.

| Branch, global step 272 | Fit RMS (mm) | Motion RMS (mm) | Tensor variation | Normalized surface HP |
| --- | ---: | ---: | ---: | ---: |
| Fit-only control | 2.779112 | 2.761991 | 4.667934 | 0.183556 |
| Weak smoothness | 2.879224 | 2.628131 | 1.518685 | 0.185824 |
| Full smoothness | 3.050046 | 2.409866 | 0.718505 | 0.188382 |

“Tensor variation” is the unchanged geometric smoothness of `Z = Q/Qref`
before multiplying by its loss weight. It is normalized by the fixed stress
reference, not divided by each run's tensor amplitude. Both smoothness branches
have a slightly larger normalized surface high-pass ratio at their actual
endpoints, despite the much smaller tensor variation.

The [matching receipt](../data/82-regularization-match/summary.json) inventories
all 256 positive-step pairs for each smoothness strength. It permits fit
mismatch at most `max(0.02 mm, 0.5% of control fit)` and motion mismatch at most
`max(0.02 mm, 1% of control motion)`. It selects the smallest squared distance
scaled by those tolerances, with the fixed earlier-step tie-break. The common
local-step-zero state is excluded.

| Selected state | Global step | Fit RMS (mm) | Motion RMS (mm) | Tensor variation | Normalized surface HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched control | 257 | 2.821767 | 2.712161 | 4.401174 | 0.183857 |
| Weak smoothness | 258 | 2.822542 | 2.710683 | 4.200225 | 0.183815 |
| Full smoothness | 257 | 2.826873 | 2.705338 | 4.105456 | 0.183923 |

At these selected matches, weak smoothness reduces tensor variation by 4.57%
and surface HP by 0.023%; full smoothness reduces tensor variation by 6.72%
and increases surface HP by 0.036%. These small surface differences support
no substantial smoothing claim. Both rank-mixing fractions remain above
`0.05`, but neither match meets the required **10% reduction in both tensor
variation and normalized surface HP**. The rank follow-up was therefore
**skipped**, as specified before these outcomes. Other saved pairs remain in
the inventories; the selected-match result is not a claim about every possible
weight, optimizer, matching rule, or longer continuation.

## Figures and saved geometry

The [complete comparison](../data/80-continuation-comparison/summary.json),
[endpoint table](../data/80-continuation-comparison/continuation-endpoints.csv),
and [all continuation traces](../data/80-continuation-comparison/continuation-traces.csv)
retain the actual measurements. The plot's cumulative stress-update metric
resets at the start of each continuation block.

![Optimization and geometric diagnostics](../data/80-continuation-comparison/continuation-optimization.png)

The [3D viewer](viewer.html) contains nine saved cases and their actual recorded
history, with front, three-quarter, and mouth cameras. Its skin/material toggle
and system, light, and dark themes are available for inspection. History uses
saved states without interpolation. All static panels use original geometric
scale and a common camera within each comparison; the rightmost column is the
exact target skin.

![Fit-only continuation: saved steps 64, 80, and 256, followed by the target](../data/85-continuation-render-v2/primary-front.png)

![Mouth detail for the fit-only continuation and target](../data/85-continuation-render-v2/primary-mouth.png)

![Equal-step control, weak smoothness, full smoothness, and target](../data/85-continuation-render-v2/regularization-front.png)

![Mouth detail for equal-step regularization endpoints and target](../data/85-continuation-render-v2/regularization-mouth.png)

![Comparable-fit/motion control, weak smoothness, full smoothness, and target](../data/85-continuation-render-v2/matched-front.png)

![Mouth detail for the selected comparable-fit/motion states and target](../data/85-continuation-render-v2/matched-mouth.png)

Separate PDF assets are available for the
[optimization traces](../data/80-continuation-comparison/continuation-optimization.pdf),
[primary front](../data/85-continuation-render-v2/primary-front.pdf),
[primary mouth](../data/85-continuation-render-v2/primary-mouth.pdf),
[equal-step front](../data/85-continuation-render-v2/regularization-front.pdf),
[equal-step mouth](../data/85-continuation-render-v2/regularization-mouth.pdf),
[matched front](../data/85-continuation-render-v2/matched-front.pdf), and
[matched mouth](../data/85-continuation-render-v2/matched-mouth.pdf).
The [renderer receipt](../data/85-continuation-render-v2/summary.json) and
[render manifest](85-render-manifest-v2.json) identify the exact saved volumes and
arrays behind every panel and history state. The calibrated probe has no solved
post-update state, so no such geometry is shown.

## Interpretation

The original step-64 PSD fit was still improving, and continuing it materially
reduced target error under the same physical model and full control field. This
study supplies no accepted lower-epsilon face trajectory and no inverse
convergence or capacity certificate. The small generated-target recovery shows
an optimizer witness on that small fixture only.

Within the tested smoothness strengths and 16-update window, lower tensor
variation did not produce a meaningful reduction in normalized surface content
at the preregistered comparable-fit/motion matches. The equal-step endpoint
results also carry a clear fit/motion cost. These observations support keeping
the fit, motion, tensor regularity, and surface measurements together when
choosing a later experiment.

## Reproducibility record

The current records were run from
`.venv/bin/python` with Python `3.14.6`,
PyTorch `2.12.0+cu130`, and CUDA `13.0`, as recorded in the calibration and
baseline provenance. The exact terminal captures are
[small recovery](../logs/70-small-recovery-v4-terminal.txt),
[face calibration](../logs/73-face-step-calibration-v2-terminal.log),
[calibrated probe](../logs/74-calibrated16-terminal.log),
[baseline probe](../logs/74-baseline16-terminal.log), and
[selection](../logs/81-probe-decision-terminal.log). The captured commands are
also recorded there. They include `src/70-small-recovery.py --output-dir
data/70-small-recovery-v4`, the calibrated `src/73-calibrate-face-step.py`
command with the small-recovery receipt and `--reduced-eps 1e-6`, and the two
explicit source-74 probe commands.

The corresponding Comet captures are [small recovery](https://www.comet.com/liblaf/apple/a98ebc6c43a649d29639053be76ba07b),
[calibration](https://www.comet.com/liblaf/apple/ea8d87fb26434718999823a45b0361a1),
[calibrated probe](https://www.comet.com/liblaf/apple/7eaed0d68d7a44f2a24eb0db71b51dbd),
[baseline probe](https://www.comet.com/liblaf/apple/c60a8fbf6dd549f1b7bde850456a0c6f),
and [selection](https://www.comet.com/liblaf/apple/39aa416f7ce24f04b630833eac6d3f6e).
These links are execution captures; the local JSON, CSV, checkpoint, and mesh
records above remain the direct evidence used here.

The later runs are retained with the same execution settings:

| Stage | Local summary | Terminal capture | Comet capture |
| --- | --- | --- | --- |
| Fit to 144 | [summary](../data/74-fit144/summary.json) | [log](../logs/74-fit144-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/7ac46b80c4b1435587df19fc13ea2b5f) |
| Fit to 208 | [summary](../data/74-fit208/summary.json) | [log](../logs/74-fit208-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/045ae029150d433eb4c29ede27318e9c) |
| Fit to 256 | [summary](../data/74-fit256/summary.json) | [log](../logs/74-fit256-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/2d6d46884c0f4d58af18ef6f4eca65ce) |
| Fit-only control | [summary](../data/74-reg-control/summary.json) | [log](../logs/74-reg-control-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/086f2dd6e2ff49fea06d108ef9abd256) |
| Weak smoothness | [summary](../data/74-reg-weak/summary.json) | [log](../logs/74-reg-weak-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/e72b54ae1fc94aac8bd72429be414597) |
| Full smoothness | [summary](../data/74-reg-full/summary.json) | [log](../logs/74-reg-full-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/2c6c402fcffd428295960e8013b68360) |
| Endpoint comparison | [summary](../data/80-continuation-comparison/summary.json) | [log](../logs/80-continuation-comparison-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/b5e962bae3814c2b850bb0599689b77b) |
| Matched-state decision | [summary](../data/82-regularization-match/summary.json) | [log](../logs/82-regularization-match-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/6f592c6c502e4c9d89222ed00371fd79) |
| Saved-state rendering | [summary](../data/85-continuation-render-v2/summary.json) | [log](../logs/85-render-continuations-v2-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/d6a82563e3b348fda9f26293015e1798) |

All runs used the noncommitting Cherries profile, a readable `CHERRIES_NAME`,
and comma-separated `CHERRIES_TAGS`, with `COMET_AUTO_LOG_GIT_METADATA=false`,
`COMET_AUTO_LOG_GIT_PATCH=false`, and `COMET_AUTO_LOG_ENV_DETAILS=false`.
The [comparison manifest](80-comparison-manifest.json) and
[matching manifest](82-regularization-match-manifest.json) identify every input
checkpoint and completed run. The runtime and helper-source records retain
[package versions and core sources](../data/05-runtime/summary.json) and
[external measurement/rendering helpers](../data/06-helper-sources/summary.json).
The [publication manifest](87-publication.json) lists the packaged source and
text evidence.

Reproduction requires the recorded local input meshes and optimizer checkpoints;
use the captured commands with fresh output directories. The scripts-and-evidence
archive contains the source and text receipts. The 3D viewer contains the
exported saved surfaces, while the full volume states and Adam checkpoints
remain in the experiment's local data directories.
