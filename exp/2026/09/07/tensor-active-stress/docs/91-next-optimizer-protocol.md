# Preregistered optimizer repeatability and continuation protocol

## Scope and historical boundary

This experiment tests whether a lower Adam epsilon improves the completed
fit-only PSD continuation after global step 256 once GPU repeatability is
measured and the first update is made from one shared cached gradient. It keeps
the constitutive model, objective, target, feasible set, projection, forward and
adjoint solvers, and all solver tolerances unchanged. It retains all 288,235
active tetrahedra and all 1,729,410 orthonormal tensor controls.

This is a new experiment. It does not amend or reinterpret the failed calibrated
probe recorded by the [source-72 protocol](72-optimizer-protocol.md) and
[source-87 report](87-optimizer-continuation-report.md). That probe correctly
failed its preregistered replay requirement after independently recomputed GPU
gradients differed enough to move the projected control by more than the old
threshold. Its failure receipt and the completed baseline continuation through
step 256 remain historical evidence. The new run measures that variation first
and avoids cross-gradient equality by using the same accepted cached gradient
for calibration and the first update of both new arms.

No face outcome may change the repeatability gates, epsilon, learning-rate
search interval, probe length, branch rule, continuation stops, smoothness
weights, matching rule, or rank gate below.

## Frozen states and unchanged model

The no-update study has two anchors:

| Anchor | Optimizer checkpoint | Checkpoint SHA-256 | State evidence |
| ---: | --- | --- | --- |
| 64 | `../data/21-psd/optimizer-latest.pt` | `adf99d031850d423979feb11a0d844ee343757ffaa749ab77fe4aa4fd0de490d` | `final.npz` SHA-256 `df5642c97d774bdaacc7bb5fc8c361e352301edab736ac9d92e0306c69903596`; `final.vtu` SHA-256 `7d0e996c11ac386496db02e88ba820d2126abbe451f0df7a67d88745d8d0e21c` |
| 256 | `../data/74-fit256/optimizer-latest.pt` | `3566e630fba5dae173fed2db7a0f70eacafb63bfe3bf0ef84e67025bce4d8ca9` | `final.npz` SHA-256 `0787349508780be6ac942ef54dac563449c18355d07377f19350e252f850f04c`; `final.vtu` SHA-256 `6c3b257f15d49b6568dd21817088e967aa1bc53c7efb5be33514c995a0f6d363` |

Before any GPU sample, freeze a manifest containing these file hashes, the
complete source and input inventory, and hashes of checkpoint `q`, converged
displacement seed `u`, Adam moments `m` and `v`, and scalar counter `t`. Require
`q.shape == (288235, 6)`, `u.shape == (228660, 3)`, one Adam state, moment
shapes equal to `q`, and counters 64 and 256 at their respective anchors. Every
sample at an anchor must start in a fresh process from exactly those frozen
`q/u/m/v/t` values. A process must not overwrite the anchor or pass its solved
state to another sample.

Both anchors retain baseline Adam `epsilon = 0.01` and learning rate `0.3`.
The candidate uses `epsilon = 1e-6`; its learning rate is calibrated separately
at each anchor as specified below. Only the step-256 candidate rate can transfer
to an optimization probe. The fixture, passive material fields,
`Qref = 0.030201342281879193 MPa`, tensor map, spectral constraint
`0 <= Q/Qref <= 10 I`, target, loss, boundary conditions, collision state,
active-cell ordering, precision, and physics configuration must match the
completed source-74 lineage. There is no reduced control set, changed physics,
new solver tolerance, geometry repair, or retry policy.

## Three independent no-update samples at each anchor

Run exactly three fresh-process forward/adjoint samples at global step 64 and
exactly three at global step 256. Each sample loads the same frozen anchor,
sets `q.grad = None`, solves from the same saved displacement seed, evaluates
the unchanged fit-only objective, and computes its true adjoint gradient. It
does not apply an optimizer update or mutate the saved moments or counter.

Every sample must report successful forward and adjoint solves and finite
state, loss, fit, motion, and gradient values. Reuse the source-72 initial replay
tolerances: unweighted fit RMS, area-weighted fit RMS, and area-weighted motion
RMS must each agree with the saved anchor within `1e-6 mm`; the data-objective
tolerance is

\[
\frac{2 F_{\mathrm{saved}}\,10^{-6}+10^{-12}}{3}\;\mathrm{mm}^2,
\]

where `F_saved` is the saved unweighted fit RMS, and the loss/RMS identity must
agree within `1e-12 mm^2`. The projected controls must remain feasible to
`1e-10` in normalized coordinates. These are replay checks under the existing
forward and adjoint tolerances, not new physics tolerances.

Record each sample's input hashes, process identity, solver receipts, solved
`u`, objective and fit metrics, gradient, and their hashes. Define the
area-weighted fit repeatability range at anchor `s` as

\[
R_s = \max_j F_s^{(j)}-\min_j F_s^{(j)}, \qquad j\in\{0,1,2\}.
\]

Sample 0 is the accepted cached state at each anchor if and only if it passes
all checks. Samples 1 and 2 measure repeatability; they are not substitutes for
sample 0 and are not gradients that either final probe must reproduce exactly.

## Matched one-times calibration and noise gate

At each anchor, use only accepted sample 0 to calibrate the candidate. For a
gradient `g`, clone the exact frozen `q/m/v/t`, apply one Adam step, then apply
the unchanged spectral projection. Define the full projected physical update

\[
\Delta Q_i = Q_{\mathrm{ref}}
  \left[q_i^{\mathrm{projected}}-q_i^{\mathrm{before}}\right]
\]

through the existing orthonormal tensor map, and define

\[
\operatorname{rms}_Q(A)=
\left(\frac{1}{N}\sum_{i=1}^{N}\lVert A_i\rVert_F^2\right)^{1/2}.
\]

First calculate the baseline sample-0 update with `epsilon = 0.01` and learning
rate `0.3`. With `epsilon = 1e-6`, perform one monotone bracketed bisection over
learning rate `[0, 0.3]` to match the baseline projected physical update RMS at
that anchor. Accept a positive rate only when the candidate/baseline RMS ratio
is within 1% of `1.0`. Freeze the rate before inspecting any solved post-update
outcome. If the crossing cannot be bracketed within `[0, 0.3]`, the map is
nonfinite or nonmonotone near the crossing, or the tolerance is not met, stop
without widening the interval, changing epsilon, or selecting a fallback rate.

The step-64 one-times calibration exists only for the no-update noise study.
Also replay the historical step-64 candidate with its original two-times
calibrated learning rate `0.0033136508893221615` and record that diagnosis
separately. The historical setting does not enter the new one-times gate and
does not transfer to the step-256 probes.

For each anchor and arm `a` in `{baseline, candidate}`, hold that arm's rate
fixed and form `Delta Q_a^(j)` from each of the three independently measured
gradients while always cloning the same frozen `q/m/v/t`. Let

\[
S_a=\operatorname{rms}_Q(\Delta Q_a^{(0)}),\qquad
N_a=\max_{j<k}\operatorname{rms}_Q
  (\Delta Q_a^{(j)}-\Delta Q_a^{(k)}),
\]

and, because the sample-0 step magnitudes are matched, define the
baseline-versus-candidate direction separation

\[
D=\operatorname{rms}_Q
  (\Delta Q_{\mathrm{baseline}}^{(0)}-
   \Delta Q_{\mathrm{candidate}}^{(0)}).
\]

The noise gate passes at an anchor only when all values are finite, both
`S_a > 0`, `D > 0`, and each arm satisfies both

\[
N_a \le 0.01 S_a
\quad\text{and}\quad
N_a \le 0.10 D.
\]

Zero updates, zero direction separation, or another zero/no-signal denominator
cannot pass. Both the step-64 and step-256 gates must pass before the new probes
may run. Report every pairwise full-field value rather than only the maximum.

Separately replay the optimizer map more than once from identical cloned
`q/m/v/t` using the same serialized cached gradient. For each arm, the maximum
absolute normalized-coordinate difference between same-gradient projected
updates must be less than `1e-10`. This check isolates optimizer replay from GPU
gradient repeatability. It does not require a gradient from sample 1 or 2 to
produce sample 0's update.

## Matched 32-update probes from step 256

Fork exactly two fit-only probes from the accepted step-256 sample-0 state:

- **baseline:** `epsilon = 0.01`, learning rate `0.3`;
- **candidate:** `epsilon = 1e-6`, using the frozen step-256 one-times rate.

Both local-step-zero records must reuse the exact accepted sample-0 `q`, solved
`u`, cached `g`, objective, fit/motion values, and solver result, with hashes
binding those records to the repeatability receipt. The original step-256
checkpoint displacement seed remains separately recorded in sample-0
provenance. Do not run a redundant forward or adjoint evaluation before the
first update. Both arms start with identical `q/u/m/v/t`; only epsilon and the
preregistered learning rate differ.

The calibration shadow update and each arm's first real optimizer update use
the same serialized accepted sample-0 gradient. Each first update must replay
its corresponding projected shadow update with maximum normalized-coordinate
error below `1e-10`. This is a same-gradient replay. No cross-gradient replay is
required. The first new nonlinear forward solve is the state at global step
257. From that solved state onward, each arm recomputes its own true adjoint
gradient and advances its own unchanged Adam state normally.

Run both arms for exactly 32 updates, through global step 288. Every recorded
post-update state must come from a real successful nonlinear solve and a
successful adjoint, with finite metrics. Record at least each global step's
area-weighted and unweighted fit RMS, area-weighted motion RMS, objective,
forward/adjoint receipts, projected-gradient mapping, proposed and accepted
full physical update, cumulative update path, stress diagnostics,
`detF_min/max`, and inverted-tetrahedron count. Determinant and inversion values
are diagnostics; they do not select or reject a branch.

Let `F_256` be the area-weighted fit RMS of accepted step-256 sample 0,
`F_B` and `F_C` the valid step-288 baseline and candidate values, and
`G_B = F_256 - F_B`. Select the candidate only if both probes complete all 32
updates with valid solves, `F_C < F_256`, and

\[
F_B-F_C \ge
\max\left(0.01\ \mathrm{mm},\;0.05 G_B,\;10R_{256}\right).
\]

If the candidate completes but misses this rule, continue the completed
baseline. If the candidate fails, preserve its failure receipt and choose the
baseline only if the baseline completed all 32 updates validly. A failed prefix
is not an endpoint and must not be rendered. If the baseline is not a valid
completed probe, the comparison cannot select a branch and the experiment
stops. Do not retry a failed solve, change tolerances, shorten the comparison,
substitute a best state, or use geometry diagnostics as hidden gates.

## Selected fit-only continuation and stopping

Continue the selected exact step-288 `q/u/m/v/t` with its frozen epsilon and
learning rate in these blocks:

| Block | Updates | End step |
| --- | ---: | ---: |
| 1 | 64 | 352 |
| 2 | 64 | 416 |
| 3 | 64 | 480 |
| 4 | 32 | 512 |

Save full state and optimizer evidence at least every 16 updates. At every block
endpoint compute the area-fit improvement

\[
I_k=F_{\mathrm{start},k}-F_{\mathrm{end},k}.
\]

The block has low progress when

\[
I_k < \max(0.01\ \mathrm{mm},\;0.005F_{\mathrm{start},k}).
\]

Stop for settled fit progress only after two successive completed low-progress
blocks. A block that is not low progress resets the consecutive count.

Also compute the fixed-step projected-gradient mapping in normalized tensor
coordinates at the common accepted step-256 anchor and every block boundary,

\[
G_1(q)=q-P_{[0,10I]}(q-\nabla f(q)),
\]

without changing Adam state. Stop when both its RMS and maximum absolute
coordinate are no more than 5% of their corresponding step-256 anchor values.
This is a relative diagnostic criterion, not a convergence or stationarity
claim. Raw gradient size is not a stopping rule.

Evaluate both stopping rules only at completed block boundaries, before
launching another block. Step 512 is the hard cap. A relative projected-gradient
stop or a two-block progress stop reached before or at that cap is labeled
**fit progress settles**. If the run reaches 512 only because of the cap while
neither rule passes, label it a still-progressing fixed-budget endpoint and
defer the smoothness and rank experiments explicitly. A solver/adjoint failure
also stops the run with a failure receipt and does not establish settled fit.

Continue to log motion, full physical stress/update fields, tensor magnitude
and cap fractions, projected-gradient values, solver diagnostics,
`detF_min/max`, and inverted tetrahedra at all saved states. No determinant,
inversion, motion, or surface-content threshold silently changes the branch or
stopping rules.

## Conditional smoothness and rank follow-ups

Run smoothness follow-ups only from a valid endpoint labeled **fit progress
settles**. From its exact `q/u/m/v/t`, create three new 16-update forks that
differ only in the existing smoothness weight:

- fit-only control: `0`;
- weak smoothness: `1.4762928671047126`;
- full smoothness: `5.9051714684188505`.

All three retain the selected epsilon/rate, all 1,729,410 controls, projection,
physics, target, and tolerances. Save actual `Q/u/VTU` and complete optimizer
checkpoints for every step. A failed prefix is retained as failure evidence but
is not rendered or used as a matched endpoint.

Apply the existing source-82 matching rule without revision. Local step zero is
excluded. Match each smoothness arm separately against positive local steps of
the new fit-only control. A pair is admissible only when area-weighted fit differs
by at most `max(0.02 mm, 0.5%)` and area-weighted motion differs by at most
`max(0.02 mm, 1%)`. Among admissible pairs, minimize the squared fit/motion
distance after scaling by those tolerances; break ties by the earlier smooth arm
step and then the earlier control step. Report equal-step results as well.

The conditional rank rule is also unchanged. A smooth arm qualifies only if,
at its selected admissible match, it lowers both normalized tensor variation and
the frozen full-face 5-mm normalized surface high-pass ratio by at least 10%
relative to its matched control and retains `rank_mixing_fraction > 0.05`. If
both strengths qualify, choose the weak arm; otherwise choose the sole
qualifying arm. Fork that exact selected smooth checkpoint into matched
16-update smooth and smooth-plus-rank arms, adding only the previously frozen
`rank_weight = 119.62893380703123`. If no arm qualifies, record the two failed
10% gates and skip rank.

## Execution and evidence order

Freeze this protocol, the repeatability manifest, and all source/input hashes
before starting a GPU sample. Run the historical fixed-gradient diagnosis
separately, then the six no-update samples as six fresh processes, then aggregate
the repeatability and calibration receipt. If either anchor gate fails, stop
before launching a probe.

After a passed aggregate receipt, run the two source-92 probes into separate new
empty directories and apply the source-93 probe decision. Run only the selected
continuation block named by that decision. After every completed block, write a
new source-93 decision before starting another block. Every decision must bind
the protocol, repeatability receipt, input summary, selected checkpoint, and
next exact update budget by path and SHA-256.

Only a source-93 decision that stops with `regularization_eligible = true` may
authorize the initial three smoothness forks. Any later rank fork additionally
requires the completed matched-state decision and its exact selected smooth
checkpoint. Keep every failed-run receipt and prefix, but exclude a failed
prefix from endpoint rendering and branch selection. No stage may overwrite an
earlier output directory.

This bounded experiment can show optimizer progress and algorithmic differences
under the accepted PSD model. It cannot prove target reachability, inverse
convergence, anatomical validity, or that one epsilon is universally better.
