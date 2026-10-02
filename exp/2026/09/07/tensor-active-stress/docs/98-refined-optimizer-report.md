# Optimizer repeatability and continuation to step 512

The full-field PSD inverse reduced area-weighted fit RMS from **2.8247 to
2.3133 mm** between steps 256 and 512, an **18.10% reduction**. The
baseline settings won the two 32-update probes after their first physical
update magnitudes were matched. Both probes completed with valid forward and
adjoint solves. The final selected state has no inverted tetrahedra and
`detF_min = 0.33965`.

The run reached its declared update cap while fit was still improving. The
projected-gradient RMS and maximum remained at **61.32% and 68.09%** of their
common step-256 values. This is a fixed-budget result, with no inverse
convergence claim. Under the frozen protocol, **smoothness and rank follow-ups
are deferred** until fit progress settles.

## Matched 32-update probes

Three independent no-update forward/adjoint evaluations at each of steps 64
and 256 passed the declared full-field update-noise gates. The largest noise
was 0.004996% of an arm's own first update, below the 1% limit. The optimizer
replay and calibration details appear below.

The baseline used Adam `epsilon = 0.01`, learning rate `0.3`; the candidate
used `epsilon = 1e-6`, learning rate `0.0010191837147885963`. Both began with
the same controls, displacement, gradient, moments, and counter. Their first
projected physical stress-update RMS was `2.31677e-5 MPa`.

At step 288, fit RMS was **2.736222 mm for the baseline** and **2.784702 mm
for the candidate**. Both improved on the common start, but the candidate
finished 0.048480 mm above the baseline instead of beating it by the required
0.01 mm. The [probe decision](../data/93-probe/summary.json) therefore retained
the baseline. This conclusion applies to these settings and 32-update window;
it does not establish the best epsilon or learning rate in general.

![Saved common state and both actual probe endpoints, with the target at right](../data/97-refined-render/probe-front.png)

![Mouth detail for the common state, both probe endpoints, and target](../data/97-refined-render/probe-mouth.png)

## Selected continuation and stopping evidence

The selected baseline continued with unchanged optimizer settings in blocks
ending at steps 352, 416, 480, and 512. Each block restored its exact parent
controls, Adam moments, and counter, and used the parent displacement as the
next forward seed. No material, target, stress constraint, loss weight, or
solver tolerance changed.

| State | Global step | Fit RMS (mm) | Motion RMS (mm) | Minimum detF | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common replay | 256 | 2.824686 | 2.708769 | 0.113915 | 0 |
| Baseline probe | 288 | 2.736222 | 2.812116 | 0.145200 | 0 |
| Candidate probe | 288 | 2.784702 | 2.742297 | 0.140709 | 0 |
| Selected continuation | 352 | 2.586507 | 2.987161 | 0.205556 | 0 |
| Selected continuation | 416 | 2.463421 | 3.131100 | 0.262784 | 0 |
| Selected continuation | 480 | 2.359580 | 3.252578 | 0.315199 | 0 |
| Selected continuation | 512 | 2.313335 | 3.306700 | 0.339646 | 0 |

The common numerical row is the accepted step-256 sample-0 replay reused by
both probes. Endpoint rows are the actual final states of the completed runs.
All 288 post-update states across the two probes and four selected continuation
blocks had successful forward and adjoint solves and no inverted tetrahedra.
The separate no-update step-64 anchor retains its historical inversion; it is
not part of this 256-to-512 trajectory. Geometry is reported throughout and
was not used to select the optimizer.

Every continuation block exceeded its declared minimum progress. The gradient
ratios below use the common probe start at step 256, not the start of each
block. Each block's fit improvement uses its own accepted initial replay, so
sub-rounding replay differences need not equal subtraction of adjacent table
endpoints bit for bit.

| Block | Fit reduction (mm) | Required progress (mm) | Gradient RMS / step 256 | Gradient max / step 256 |
| --- | ---: | ---: | ---: | ---: |
| 288 to 352 | 0.149715 | 0.013681 | 80.50% | 81.60% |
| 352 to 416 | 0.123086 | 0.012933 | 71.46% | 75.54% |
| 416 to 480 | 0.103841 | 0.012317 | 64.35% | 70.39% |
| 480 to 512 | 0.046245 | 0.011798 | 61.32% | 68.09% |

The last 32 updates reduced fit RMS by **0.046245 mm**, above the
**0.011798 mm** progress threshold. There were zero consecutive low-progress
blocks, and neither gradient ratio met the 5% stopping condition. The
[final decision](../data/93-block512/summary.json) stopped at the global
512-update cap and recorded regularization as ineligible. No new smoothness
or rank branch was run in this study. Earlier regularization results remain
separate historical experiments.

![Recorded optimization traces and physical stress updates](../data/95-refined-comparison/continuation-optimization.png)

The cumulative update panel resets at each block boundary. Its values are sums
of per-update full-field RMS magnitudes, not the net tensor change and not a
single cumulative curve from step 256. The
[endpoint table](../data/95-refined-comparison/continuation-endpoints.csv),
[all traces](../data/95-refined-comparison/continuation-traces.csv), and
[comparison receipt](../data/95-refined-comparison/summary.json) retain the
measurements, source hashes, and checkpoint lineage. Motion RMS rose from
2.708769 to 3.306700 mm, and final `detF_max = 2.57390`; absence of inverted
tetrahedra alone does not establish physiological validity.

![Saved step 256 and selected step 512, with the target at right](../data/97-refined-render/selected-front.png)

![Mouth detail for saved step 256, selected step 512, and target](../data/97-refined-render/selected-mouth.png)

## Saved geometry and reusable figures

The [3D viewer](viewer.html) contains the saved step-256 parent, both completed
probe endpoints, and the selected step-512 endpoint. It includes front,
three-quarter, and mouth cameras, skin/material views, and system, light,
and dark themes. History uses actual saved states without interpolation.

The common step-256 *geometry case* is the historical saved parent. Each
probe's history instead begins at its actual cached sample-0 equilibrium;
the selected history follows the baseline and all four continuation blocks.
These records are separately pinned in the
[render manifest](97-refined-render-manifest.json) and
[renderer receipt](../data/97-refined-render/summary.json). All panels show
unamplified geometry at scale 1, a common camera within the panel, and the
exact target skin in the rightmost column.

Separate PDF assets are available for the
[optimization traces](../data/95-refined-comparison/continuation-optimization.pdf),
[probe front](../data/97-refined-render/probe-front.pdf),
[probe mouth](../data/97-refined-render/probe-mouth.pdf),
[selected front](../data/97-refined-render/selected-front.pdf), and
[selected mouth](../data/97-refined-render/selected-mouth.pdf).

## Repeatability and the matched first update

The no-update study used three fresh processes at global step 64 and three at
step 256. Each process restored the exact saved controls, displacement seed,
Adam moments, and counter. All six forward and adjoint evaluations succeeded
and passed the original saved-state replay tolerances. The full source and
input inventory was archived before sampling; the [freeze manifest](../data/90-frozen-inputs/manifest.json)
records the hashes. All 288,235 active tetrahedra and 1,729,410 tensor controls
were retained. The constitutive model, target, spectral constraint, objective,
forward and adjoint solvers, and their tolerances were unchanged.

The comparison uses the full physical tensor update in every active cell.
For each arm, the reported noise is the maximum RMS difference among the three
pairs of projected updates generated from the independently computed gradients.
Every such update starts from the same saved controls and Adam state. Each arm
must have noise below 1% of its own sample-0 update and below 10% of the
baseline-versus-candidate update difference. Both anchors passed:

| Anchor | Arm | Noise / own update | Noise / arm difference |
| ---: | --- | ---: | ---: |
| 64 | Baseline | 0.000484% | 0.000470% |
| 64 | Candidate | 0.004996% | 0.004850% |
| 256 | Baseline | 0.000375% | 0.000358% |
| 256 | Candidate | 0.003039% | 0.002895% |
| Required limit | Each arm | 1% | 10% |

![Measured full-field update noise and the two limits](../data/96-repeatability-plots/repeatability-noise.png)

The [complete receipt](../data/90-repeatability/summary.json) contains every
pairwise value, gradient and displacement comparison, optimizer replay check,
and calibration trial. The [PDF figure](../data/96-repeatability-plots/repeatability-noise.pdf)
is a separate reusable asset. The three area-weighted fit values at step 256
were identical at the stored precision, giving a measured fit range of zero;
the gradients had distinct hashes and small nonzero full-field differences.

At step 256, the baseline used `epsilon = 0.01` and learning rate `0.3`.
The candidate used `epsilon = 1e-6` and learning rate
`0.0010191837147885963`, calibrated before either post-update outcome was
examined. Their first physical update RMS values were
`2.316765895877857e-5 MPa` and `2.3167658958513915e-5 MPa`, respectively.
Thus the comparison matched the first projected update magnitude to one times
the baseline. Later update magnitudes were measured, not forced to remain
matched.

Both probes reused the exact accepted step-256 sample-0 state, cached gradient,
objective, and solver receipts for local step zero. Each first real optimizer
update was checked against a shadow update using that same gradient. The first
new nonlinear forward state was step 257; subsequent gradients were recomputed
normally in each arm. The historical checkpoint seed is separately recorded.
This separates optimizer-map replay from differences between independently
computed GPU gradients.

The separate [historical fixed-gradient replay](../data/94-fixed-replay/summary.json)
used the exact previously recorded learning rate `0.0033136508893221615`.
The installed Adam update and closed-form update agreed to a maximum normalized
coordinate error of `2.11e-15`, below the unchanged `1e-10` limit. Source 94
corrected an implementation mistake in the initial CPU diagnostic, which had
recalibrated the already recorded historical rate. The [correction record](94-fixed-replay-correction.md)
retains the failed attempt, explains the change, and records an execution-order
deviation: the initial CPU diagnostic and first samples ran concurrently,
with the corrected replay completed before aggregation and either probe. No
fresh sample, threshold,
optimizer trajectory, or physics setting was changed by that correction.

## Decision rules and interpretation

The [frozen protocol](91-next-optimizer-protocol.md) defines the comparison.
Both probes run for 32 updates from step 256. The candidate is selected only
if its final area-weighted fit RMS improves on the common initial state and
beats the baseline by at least the maximum of 0.01 mm, 5% of the baseline's
fit improvement, and ten times the measured step-256 fit repeatability range.

The selected arm then continues in blocks through steps 352, 416, 480, and
at most 512. A block has low progress when its fit reduction is less than
`max(0.01 mm, 0.005 * starting fit RMS)`. Two consecutive low-progress blocks
stop continuation. A second stopping rule requires both the RMS and maximum
of the projected-gradient mapping, evaluated at `eta = 1`, to fall to at most
5% of their common step-256 values. These are declared operational stopping
rules; they do not prove convergence or target reachability.

Smoothness testing is conditional on fit progress settling under those rules.
Reaching the update cap while still making progress does not make that test
eligible. Rank regularization requires an eligible smoothness result and the
separate matching and rank checks in the protocol. Determinants, inversions,
motion, and surface roughness are reported diagnostics; they do not alter the
optimizer selection rule.

The repeatability study bounds variation at two saved states under the current
runtime. It does not prove bitwise GPU determinism or bound accumulated
trajectory differences at every later state. The equal first-step comparison
also does not establish the best epsilon or learning rate over all possible
settings. Fit error measures agreement with this prescribed target; it does
not establish anatomical or physiological validity or uniqueness of the
recovered tensor field.

## Reproducibility record

Runs used `.venv/bin/python`, Python
`3.14.6`, PyTorch `2.12.0+cu130`, and CUDA `13.0`, with the noncommitting
Cherries profile. Each has a readable `CHERRIES_NAME` and comma-separated
`CHERRIES_TAGS`; `COMET_AUTO_LOG_GIT_METADATA`, `COMET_AUTO_LOG_GIT_PATCH`,
and `COMET_AUTO_LOG_ENV_DETAILS` were set to `false`.

The [source and input freeze](../data/90-frozen-inputs/manifest.json),
[sampling manifest](90-repeatability-manifest.json),
[aggregation manifest](90-repeatability-aggregation-manifest.json),
[comparison manifest](95-refined-comparison-manifest.json), and
[publication manifest](99-refined-publication.json) identify the inputs and
retained evidence. Per-run provenance records archive the sources, commands,
runtime, and exact parent hashes. The existing
[runtime archive](../data/05-runtime/summary.json) and
[helper archive](../data/06-helper-sources/summary.json) retain the package and
external helper records. The frozen protocol and failed historical diagnostic
were preserved alongside the corrected replay, as explained above.

| Stage | Local summary | Terminal capture | Comet capture |
| --- | --- | --- | --- |
| Step 64 sample 0 | [summary](../data/90-step64-sample0/summary.json) | [log](../logs/90-step64-sample0-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/830b5e4368b2413cb34a6dc8c7311d39) |
| Step 64 sample 1 | [summary](../data/90-step64-sample1/summary.json) | [log](../logs/90-step64-sample1-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/ef0621594f674b48a81e427c8430a6d0) |
| Step 64 sample 2 | [summary](../data/90-step64-sample2/summary.json) | [log](../logs/90-step64-sample2-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/b95d324a6e14439c86d0194d4177d450) |
| Step 256 sample 0 | [summary](../data/90-step256-sample0/summary.json) | [log](../logs/90-step256-sample0-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/847d1fde08ea477db315ba6c1b6c6530) |
| Step 256 sample 1 | [summary](../data/90-step256-sample1/summary.json) | [log](../logs/90-step256-sample1-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/c6a21186fcce42b9b0e394a284ad1fa4) |
| Step 256 sample 2 | [summary](../data/90-step256-sample2/summary.json) | [log](../logs/90-step256-sample2-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/8b13ac138e314cf9819855c34ce60024) |
| Corrected historical replay | [summary](../data/94-fixed-replay/summary.json) | [log](../logs/94-fixed-replay-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/af24354c9d834f0ba82082d154fda93d) |
| Repeatability and calibration | [summary](../data/90-repeatability/summary.json) | [log](../logs/90-repeatability-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/eb6984369e5f405c9b352f094e418ed7) |
| Baseline 32-update probe | [summary](../data/92-baseline32/summary.json) | [log](../logs/92-baseline32-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/4969d38de6f54d28bb0538968068d098) |
| Candidate 32-update probe | [summary](../data/92-candidate32/summary.json) | [log](../logs/92-candidate32-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/e333a6678f474127acdb7ee64fa16482) |
| Probe selection | [summary](../data/93-probe/summary.json) | [log](../logs/93-probe-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/a20f59e538e040bb81fa0e3adf60ced8) |
| Fit to 352 | [summary](../data/92-fit352/summary.json) | [log](../logs/92-fit352-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/1396ebbdbee84ccfabad3940f220af7c) |
| Decision at 352 | [summary](../data/93-block352/summary.json) | [log](../logs/93-block352-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/0fccaa5669b54aa2a4071abe12fdc28e) |
| Fit to 416 | [summary](../data/92-fit416/summary.json) | [log](../logs/92-fit416-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/9f36b86c99f74f7d9eb7766192a721e0) |
| Decision at 416 | [summary](../data/93-block416/summary.json) | [log](../logs/93-block416-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/0e6f0bc803fd41f795ae049801009c4c) |
| Fit to 480 | [summary](../data/92-fit480/summary.json) | [log](../logs/92-fit480-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/b01577ee891c491ca24702ea2117f4c5) |
| Decision at 480 | [summary](../data/93-block480/summary.json) | [log](../logs/93-block480-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/ede929a51dbc484b8c78824c717e28ca) |
| Fit to 512 | [summary](../data/92-fit512/summary.json) | [log](../logs/92-fit512-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/5d6e8acf3b5444b396be7d1f4a6f2265) |
| Decision at 512 | [summary](../data/93-block512/summary.json) | [log](../logs/93-block512-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/ae20b252245c49e3b139d6ae7169f0cb) |
| Endpoint comparison | [summary](../data/95-refined-comparison/summary.json) | [log](../logs/95-refined-comparison-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/95932cffa85f44318b10a0c8c39e997e) |
| Repeatability plot | [summary](../data/96-repeatability-plots/summary.json) | [log](../logs/96-repeatability-plots-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/688bfac629a74721bde8242f2c1c09ef) |
| Saved-state rendering | [summary](../data/97-refined-render/summary.json) | [log](../logs/97-refined-render-terminal.log) | [Comet](https://www.comet.com/liblaf/apple/9204128e0b2643b8b5d17f5558338cad) |

These Comet links are execution captures; the local JSON, CSV, checkpoint,
and geometry records are the direct evidence used for this report. Reproduction
requires the recorded local fixture meshes and optimizer checkpoints. Use the
captured commands with fresh output directories. The scripts-and-evidence
archive contains sources and text receipts, the figure archive contains
reusable PNG/PDF assets, and the viewer contains exported saved surfaces.
Full volume states, gradients, and Adam checkpoints remain in the local
experiment data directories.

Static figures and PDF exports were inspected, and the staged site validates
its local links, asset hashes, saved-state provenance, module graph, and archive
inventories. A live browser session was unavailable for interaction testing.
