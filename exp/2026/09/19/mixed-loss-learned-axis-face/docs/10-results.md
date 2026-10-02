# Combined positional and gradient-space loss on the 3D face

The combined loss was tested from neutral on the 3D face with learned-axis,
contraction-only activation. Both smoothness variants completed 200 updates.
The selected normalized blend uses 20% positional loss and 80% surface-gradient
loss. It improves both target errors from neutral, but the short pure-loss pilots
do not establish that combining losses is better than gradient-only fitting.

| Endpoint | Position RMS, mm | Gradient RMS | Tensor roughness R | Motion RMS, mm | Minimum physical J | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 5.095908 | 0.272271 | 0.000000 | 0.000000 | 1.000000 | 0 |
| Mixed, smoothness off | 3.989568 | 0.190322 | 3.789781 | 1.584488 | 0.444394 | 0 |
| Mixed, smoothness on | 4.388416 | 0.217233 | 0.065255 | 1.083638 | 0.679208 | 0 |

Smoothness reduces squared activation-tensor roughness by 98.28%,
at the cost of 10.00% higher position RMS and
14.14% higher gradient RMS. Motion is
31.61% smaller. Both traces remained inversion-free, and both
are finite-budget results: **neither met the inverse-convergence criteria**.

![Full-face target, neutral, mixed without and with smoothness](../data/30-figures/full-face-overview.png)

[Main Comet run](https://www.comet.com/liblaf/apple/379e105783e54566b516df510d8eccc7),
completed 2026-09-19 18:23:20 to 19:12:35 Asia/Shanghai. Numerical execution and
Cherries shutdown exited with code zero. The independent saved-mesh audit passed.

## Loss definition and matched setup

Use learned-axis contraction-only activation `B=I+snnT`, `s>=0`, `||n||=1`:
three physical degrees of freedom per active cell. Strength and axis both receive
gradients, followed by clamp/normalization after Adam. Every branch starts from
exact neutral and the same archived neutral-gradient-derived axes. These are
initial effective directions, not anatomical fibers.

The existing surface losses are combined as

```text
D_beta = K * (L2/L20 + beta * Lg/Lg0) / (1 + beta)
L20 = K = 8.656092875221388
Lg0 = 0.07413172241432929
L = D_beta + lambda * R
```

The selected beta=4 assigns 20% to normalized positional error and 80% to
normalized surface-gradient error. Both terms match the target; neither penalizes
surface variation independently of the target. The same-muscle tensor penalty
R separately controls activation variation.

The normalizers are fixed at neutral, and K retains the original L2 scale.
Equal neutral values do not imply equal Adam updates: initial physical B-step
RMS was 0.002697 for pure L2, 0.004108 for pure gradient, and 0.003736 for beta=4.
The complete [initial-step receipt](../data/08-loss-pilot/initial-update-diagnostics.json)
records all magnitudes and pairwise cosines. In this new normalized family,
pure-gradient scale is 116.76638; the preceding 200-update gradient-only study
used 99.56749. That older run is historical context, not a strictly matched
control for the new family.

The common axes favor early gradient-driven activation: after the first projected
update, the active-volume fraction with positive strength is 63.806% for pure L2,
99.996% for pure gradient, and 99.955% for beta=4 (191,596, 288,149, and 288,056 cells
respectively, out of 288,235). These values are computed from the saved
`initial-update-*.npz` tensors and physical active-cell volumes. Because axis
derivatives vanish when strength is zero, this is a material initialization
limitation for the short pure-loss comparison. It prevents interpreting that
ranking as intrinsic loss quality. The primary smoothness off/on comparison
shares both the data loss and the axes, so it does not have this between-loss
initialization difference.

All physics and fitting support remain the same: corrected physical J=det(F),
historical materials, skull fixation, skin correspondence, and full implicit
forward/adjoint solve. Adam lr=0.3, eps=0.01, betas=(0.9,0.999), with the same
halving every 100 updates. See the [frozen protocol](00-protocol.md) and
[validation evidence](05-validation.md).

## Matched 20-update loss pilots

These are finite-budget, no-smoothness endpoints from neutral. Every branch had
zero physical inversions throughout the pilot and completed all 20 updates.

| Data objective | Position RMS, mm | Gradient RMS | Tensor roughness R | Max normalized loss |
| --- | ---: | ---: | ---: | ---: |
| l2 | 4.871427 | 0.257973 | 0.070503 | 0.913838 |
| gradient | 4.847865 | 0.247186 | 0.145583 | 0.905020 |
| beta-0.25 | 4.863597 | 0.255482 | 0.076527 | 0.910903 |
| beta-1 | 4.855089 | 0.252015 | 0.094511 | 0.907719 |
| beta-4 | 4.849897 | 0.248983 | 0.122278 | 0.905778 |

Among beta=0.25,1,4, beta=4 minimized the predeclared maximum of normalized
position and gradient losses. The [selection receipt](../data/09-loss-selection/selection.json)
records the exact rule and every candidate. It is a training diagnostic, not
held-out validation or an optimal-weight claim.

The selected blend had lower errors than the pure-L2 pilot, but pure gradient
still had the lowest position and gradient errors at 20 updates. The blend's
activation was less rough than pure gradient. This does not establish the ordering
of converged optima, especially given the different initial Adam step magnitudes.

[Pilot Comet run](https://www.comet.com/liblaf/apple/6ae5ac5a36754d209eab8d1a0e060ed7).
The numerical process and Cherries shutdown exited with code zero.

![Twenty-update loss pilot comparison](../data/18-pilot-figures/pilot-tradeoff.png)

The [pilot plotting run](https://www.comet.com/liblaf/apple/0bdfb9cb70f3415a8b83b547266235f2)
also exited cleanly, and the figure was visually inspected.

## Smoothness selection for the chosen blend

A fresh eight-update probe gave base coefficient 5.702275093589208. Every
smoothness pilot then restarted from neutral. The selected coefficient was the
smallest tested positive coefficient achieving at least 75% lower tensor roughness
and zero final inversions. All recorded pilot iterates were inversion-free.

| Coefficient | Tensor roughness R | Roughness reduction | Position RMS, mm | Gradient RMS | Combined data-loss increase |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000000 | 0.122279 | 0.00% | 4.849923 | 0.248984 | 0.00% |
| 0.570228 | 0.092048 | 24.72% | 4.856436 | 0.249562 | 0.42% |
| 5.702275 | 0.028755 | 76.48% | 4.891448 | 0.252783 | 2.79% |
| 57.022751 | 0.003194 | 97.39% | 4.975806 | 0.260867 | 8.81% |

The selected coefficient reduced roughness by 76.48%, at a 2.79% increase in
combined data loss. Its positional RMS increased 0.86%, and gradient RMS
increased 1.53%. These pilot percentages are not assumed to persist at 200 updates.
The [selection receipt](../data/15-selection/selection.json) records every
candidate. The [smoothness pilot Comet run](https://www.comet.com/liblaf/apple/f5d85479abc547a38ae23f0c55005639)
completed normally, including Cherries shutdown.

## Local shape, motion, and convergence

All high-pass quantities below use the inherited 5 mm normal-field filter and
fixed regions on the reference skin. Residual means displacement minus target;
displacement-only high-pass energy can become smaller simply by moving less.
These are numerical shape diagnostics, not anatomical wrinkle validation.

| Region | Target-relative high-pass RMS, off, mm | Target-relative high-pass RMS, on, mm | Change with smoothness |
| --- | ---: | ---: | ---: |
| Mouth corner | 0.709664 | 0.853830 | +20.31% |
| Lateral cheek | 0.070678 | 0.074009 | +4.71% |
| Lower cheek / jaw | 0.100139 | 0.106653 | +6.51% |
| Primary region union | 0.315467 | 0.376939 | +19.49% |

The union's displacement-only high-pass RMS falls from 0.155960 to 0.104639 mm
(32.91% lower), while its target-relative high-pass residual rises
19.49%. Thus the calmer
appearance comes with reduced target detail and motion. The low-frequency normal
target projection over that union decreases from
0.339117 to
0.261998. Smoothness also
raises minimum physical J from 0.444394 to 0.679208, but positive
J and successful force solves do not certify mechanical stability.

![Shared-camera mouth-corner detail](../data/30-figures/details/region1-mouth-corner-overview.png)

The same-camera [lateral cheek](../data/30-figures/details/region2-lateral-cheek-overview.png)
and [lower cheek / jaw](../data/30-figures/details/region3-lower-cheek-overview.png)
views are also available. Every panel uses actual, unexaggerated geometry and
shared rendering settings. The off case shows a stronger mouth-corner crease;
the on case moves less, especially in the cheek and jaw. Both still substantially
underfit the target mouth opening.

The stopping rule requires two consecutive scheduled checks with objective
relative span below 0.001 and physical projected-gradient RMS below 0.01 of its
initial value. Neither branch met either criterion at the final check:

| Step 200 diagnostic | Smoothness off | Smoothness on | Required |
| --- | ---: | ---: | ---: |
| Recent objective relative span | 0.032975 | 0.014923 | < 0.001 |
| Projected-gradient RMS / initial | 0.391342 | 0.565736 | < 0.01 |

Span checks occur at steps 100, 125, 150, 175, and 200; the value 1.0 on other
trace rows is a placeholder, not a measured plateau statistic. The final state
is also the best recorded objective state for each branch. Learning rates were
0.3 for the first 100 updates and 0.15 for the next 100; the rate 0.075 in row 200
is the next-step setting and was not applied in this budget.

![Residuals, motion, activation roughness, projected gradient, and inversions](../data/30-figures/optimization-progress.png)

The [final rendering run](https://www.comet.com/liblaf/apple/cbfcf2bf8b8f4ebfa049ac3f660e6a9d)
exited zero after Cherries shutdown. The overview, all three regional views, and
progress plot were visually inspected. Reusable surfaces are
[target](../data/30-figures/skins/target.vtp),
[neutral](../data/30-figures/skins/reference.vtp),
[mixed off](../data/30-figures/skins/mixed-off.vtp), and
[mixed on](../data/30-figures/skins/mixed-on.vtp).

## Independent verification

The [CPU endpoint audit](../data/25-verification/checks.json) passed and its
process exited zero. It independently reconstructs activation tensors, evaluates
both raw surface losses and the graph penalty in NumPy, and computes physical
J from signed deformed/rest tetrahedron volume ratios. Endpoint losses and
objectives agree within 1.8e-15; extreme J values agree within 3.8e-15.

It verifies nonnegative strengths, unit axes, rank-one contraction constraints,
shared exact-neutral displacement/activation and archived axes, identical
learning-rate schedules, 201 successful solver receipts per branch, both
selection receipts, and all 96 archived source records plus fixture hashes.
Endpoint inversions are independently recomputed; zero inversions at every
other recorded update are supported by the training traces. This audit is not
an independent re-solve of every checkpoint or a Hessian stability test.

The earlier CPU loss/autograd checks and full implicit derivative gates also
passed. For the selected beta=4 objective, the worst directional finite-difference
relative error was 0.25136%, below the predeclared 2% threshold. See the
[validation report](05-validation.md).

## Relation to the preceding gradient-only test

The preceding learned-axis experiment also ran 200 updates from these same
neutral axes. The endpoint comparison below is useful historical context, but
its gradient scale was 99.56749 versus 116.76638 for the pure-gradient endpoint
of the new normalized family, and its smoothness coefficient was 5.62742650
versus 5.70227509 here. Adam has a nonzero epsilon, so objective scaling changes
the effective updates. These are not a fully matched test of converged loss
optima.

| Data loss | Smoothness | Position RMS, mm | Gradient RMS | Tensor roughness R |
| --- | --- | ---: | ---: | ---: |
| Previous gradient only | off | 4.171781 | 0.194387 | 3.214770 |
| Current mixed | off | 3.989568 | 0.190322 | 3.789781 |
| Previous gradient only | on | 4.496773 | 0.220259 | 0.058137 |
| Current mixed | on | 4.388416 | 0.217233 | 0.065255 |

The blend reaches lower numerical target errors in this historical comparison,
but lower minimum physical J. The cleanest within-study pure-loss controls are
the 20-update pilots above, where pure gradient slightly outperformed the blend
in both fit metrics. Their common gradient-derived axes and differing initial
Adam step magnitudes limit interpretation. No 200-update pure-L2 or pure-gradient
control was run in this new normalized family, so a general advantage for the
combined objective remains unproven.

## Reproduction

Working directory:
`exp/2026/09/19/mixed-loss-learned-axis-face`.
See [validation commands](05-validation.md) for the required CPU/full-chain
checks and selected-objective calibration. The exact numerical commands are:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

CHERRIES_NAME='Mixed face loss: L2 gradient and blend pilots' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,loss-pilot' \
.venv/bin/python src/10-run.py \
  --phase loss-pilot --output 08-loss-pilot --beta 1

DEBUG=1 CHERRIES_NAME='Mixed face loss: choose position-gradient balance' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,selection' \
.venv/bin/python src/09-select-loss.py

CHERRIES_NAME='Mixed face loss: smoothness coefficient pilots' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,smoothness,pilot' \
.venv/bin/python src/10-run.py \
  --phase smooth-pilot --output 12-smooth-pilot --beta 4

DEBUG=1 CHERRIES_NAME='Mixed face loss: choose smoothness coefficient' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,smoothness,selection' \
.venv/bin/python src/15-select-pilot.py

CHERRIES_NAME='Mixed L2 and gradient face: smoothness off versus on' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,gradient,position,smoothness,matched-comparison' \
.venv/bin/python src/10-run.py \
  --phase compare --output 20-comparison --beta 4 --steps 200 \
  --coefficient 5.702275093589208

DEBUG=1 CHERRIES_NAME='Mixed face loss: independent endpoint verification' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,verification' \
.venv/bin/python src/25-verify-results.py

CHERRIES_NAME='Mixed face loss: final comparison figures' \
CHERRIES_TAGS='mixed-loss,learned-axis,3d,smoothness,figures' \
.venv/bin/python src/30-render.py
```

This uses Python 3.14.6, Torch 2.12.0+cu130, Warp 1.14.0, float64, four CPU
threads, and an RTX 4090. The repository base is
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; unrelated pre-existing changes
were present. This study's edits are confined to the new experiment group.
Source and fixture hashes are stored in each run's protocol. Local Cherries
snapshots and logs are retained under the repository's
`.cherries/runs/2026/09/19/mixed-loss-learned-axis-face/` directory.

The pilot Comet summaries show `steps=200` because parameters are logged before
the pilot-mode override. Both saved pilot protocols, the `pilot_steps=20`
parameter, and all 21 trace evaluations per branch record the actual 20-update
budget. The primary comparison explicitly uses 200 updates.

Selected verbatim lines from the primary Comet summary in
[the main log](../logs/10-run.log):

```text
Comet.ml Experiment Summary
name                  : Mixed L2 and gradient face: smoothness off versus on
url                   : https://www.comet.com/liblaf/apple/379e105783e54566b516df510d8eccc7
mixed-off/fit_rms_mm [201]               : (3.9895678257650546, 5.095908027590781)
mixed-off/surface_gradient_rms [201]     : (0.19032211104669466, 0.2722714131419773)
mixed-on/fit_rms_mm [201]                : (4.38841610839298, 5.095908027590781)
mixed-on/surface_gradient_rms [201]      : (0.21723321093587494, 0.2722714131419773)
cherries/start_time : 2026-09-19 18:23:20.907949+08:00
cherries/end_time   : 2026-09-19 19:12:35.950716+08:00
```

Comet contains scalar, source, and metadata records; the checkpoint and mesh
artifacts linked in this report are local outputs. The run log does not claim
remote upload of those local meshes/checkpoints.
