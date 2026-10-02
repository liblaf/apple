# Preregistered step-512 learning-rate comparison and continuation protocol

## Scope and immutable anchor

This experiment tests a single optimizer change after the completed fit-only PSD
continuation: Adam learning rate `0.6` versus `0.3`, with `epsilon = 0.01` in
both arms. The rates and epsilon values are fixed before any new face outcome is
examined. There is no calibration, learning-rate search, or equal-update-magnitude
requirement.

The sole anchor is global step 512:

- checkpoint: `../data/92-fit512/optimizer-latest.pt`;
- checkpoint SHA-256:
  `3864ed0bef1c7f71ab7e4384653afbd0690910225eb67ee677fe23c4184a548a`;
- completed-run receipt: `../data/92-fit512/summary.json`;
- optimizer counter: `t = 512`.

The experiment retains the exact anchor controls `q`, converged displacement
seed `u`, Adam moments `m` and `v`, and counter `t`. It retains all 288,235
active tetrahedra and all 1,729,410 normalized tensor controls. The fixture,
active-cell ordering, precision, constitutive model, passive material fields,
tensor map, PSD spectral constraint, stress cap, target, boundary conditions,
collision state, fit-only loss, forward and adjoint solvers, and every solver
tolerance remain unchanged from the completed source-92 lineage. No reduced
control set, geometry repair, tolerance retry, or best-state substitution is
permitted.

This is a new experiment. It preserves the step-256 study and
`91-next-optimizer-protocol.md` as historical evidence rather than extending or
rewriting them. No result may change a rate, epsilon, sample count, probe length,
selection rule, continuation schedule, stopping rule, or conditional
regularization rule below.

## Pre-sample freeze

Before starting any no-update sample, the named source-100 preflight freeze
helper must write a complete immutable manifest that binds by path, byte count,
and SHA-256:

- this protocol and every pre-existing scientific source used to sample, solve,
  differentiate, update, constrain, or measure the new experiment;
- the step-512 checkpoint, its owner receipt, saved NPZ/VTU state, trace, and
  complete parent lineage;
- every fixed fixture, mesh, target, configuration, helper, package/runtime, and
  other input needed for those scientific computations, including the recorded
  runtime and helper-source inventories;
- the serialized `q`, `u`, `m`, `v`, `t`, active-cell IDs, and their shapes,
  dtypes, and hashes.

Require `q.shape == (288235, 6)`, `u.shape == (228660, 3)`, one Adam state,
moment shapes equal to `q`, and `t == 512`. Source 100 must validate this frozen
manifest before accepting any sample. The frozen scientific inventory must be
complete and valid before the first GPU process starts. Later generated
receipts, decision manifests, comparison/render inputs, report prose, and
publication inputs are pinned by path and SHA-256 when they are created; they
may not alter the frozen physics, thresholds, or this protocol. A sample or
later stage must not overwrite the anchor, the manifest, or an earlier output
directory.

## Three fresh no-update samples

Run exactly three independent fresh-process forward/adjoint samples from the
same frozen step-512 `q/u/m/v/t`. Each process sets `q.grad = None`, solves from
the same saved `u`, evaluates the unchanged fit-only objective, and computes its
true adjoint gradient. It must not apply an optimizer update or mutate the saved
moments or counter.

Every sample must report successful forward and adjoint solves and finite state,
loss, fit, motion, and gradient values. Reuse the saved-state replay tolerances
from the step-256 protocol: unweighted fit RMS, area-weighted fit RMS, and
area-weighted motion RMS must each agree with the saved step-512 values within
`1e-6 mm`. If `F_saved` is the saved unweighted fit RMS, the unweighted
data-objective `F^2 / 3` must agree within

\[
\frac{2F_{\mathrm{saved}}\,10^{-6}+10^{-12}}{3}\;\mathrm{mm}^2.
\]

The loss/RMS identity must agree within `1e-12 mm^2`, and normalized controls
must remain feasible to `1e-10`. These checks reuse existing numerical
tolerances; they do not introduce a new physics or solver tolerance.

Record each sample's frozen-input binding, process identity, complete forward and
adjoint receipts, solved `u`, objective and surface metrics, gradient, and hashes.
For the area-weighted fit values `F_512^(j)`, define

\[
R_{512}=\max_{j\in\{0,1,2\}}F_{512}^{(j)}
       -\min_{j\in\{0,1,2\}}F_{512}^{(j)}.
\]

Sample 0 is the accepted cached state only if it passes every check. Samples 1
and 2 measure repeatability and cannot replace sample 0.

## Fixed optimizer replay and full-tensor noise gate

Using each sample gradient, always clone the same frozen `q/m/v/t` before
forming either arm's one-step optimizer map:

- baseline: learning rate `0.3`, `epsilon = 0.01`;
- candidate: learning rate `0.6`, `epsilon = 0.01`.

Apply the installed Adam update and then the unchanged PSD projection. Through
the existing orthonormal tensor map, define the full physical update for every
active cell as

\[
\Delta Q_{a,i}^{(j)}=Q_{\mathrm{ref}}
\left(q_{a,i}^{(j),\mathrm{projected}}-q_i^{\mathrm{before}}\right),
\qquad
\operatorname{rms}_Q(A)=
\left(\frac{1}{N}\sum_{i=1}^{N}\lVert A_i\rVert_F^2\right)^{1/2}.
\]

For each arm `a`, define its sample-0 signal and maximum pairwise noise as

\[
S_a=\operatorname{rms}_Q(\Delta Q_a^{(0)}),\qquad
N_a=\max_{j<k}\operatorname{rms}_Q
\left(\Delta Q_a^{(j)}-\Delta Q_a^{(k)}\right),
\]

and define the full sample-0 baseline-candidate separation as

\[
D=\operatorname{rms}_Q
\left(\Delta Q_{\mathrm{baseline}}^{(0)}-
      \Delta Q_{\mathrm{candidate}}^{(0)}\right).
\]

The gate passes only if all values are finite, both `S_a > 0`, `D > 0`, and
each arm satisfies both

\[
N_a\le 0.01S_a
\qquad\text{and}\qquad
N_a\le 0.10D.
\]

Report all three pairwise values per arm. `D` is the RMS of the full physical
tensor difference; it is not a direction-only statistic. The two sample-0
update magnitudes need not be equal. Zero updates, zero arm separation, or any
nonfinite quantity fails the gate.

For each arm, also replay the same serialized sample-0 gradient from identical
cloned `q/m/v/t` in three ways: repeated installed-optimizer execution,
closed-form Adam evaluation, and the installed projection. Each comparison of
the resulting projected normalized controls must have maximum absolute
coordinate error below `1e-10`. Record the unprojected and projected hashes and
full physical update metrics. This same-gradient contract is separate from the
fresh-process gradient-noise gate.

## Two 32-update probes through step 544

Only a passed source-100 aggregate receipt may authorize the probes. Fork both
arms from the accepted sample-0 `q`, solved `u`, cached `g`, objective, surface
metrics, solver receipts, `m/v/t`, and their hashes. Local step zero in both
traces must reuse those cached results exactly. The original checkpoint
displacement seed remains separately recorded in sample-0 provenance.

Do not run a redundant initial forward or adjoint evaluation. Each arm's first
real optimizer update must reproduce its corresponding accepted sample-0 shadow
update within the `1e-10` normalized-coordinate limit. The first new nonlinear
forward state is global step 513. From that state onward, each arm recomputes its
own true adjoint gradient and advances its own Adam state normally.

Run both arms for exactly 32 updates, through global step 544. Every post-update
record must come from a successful nonlinear forward solve and successful
adjoint and must contain finite metrics. Record each global step's area-weighted
and unweighted fit RMS, area-weighted motion RMS, objective, solver receipts,
projected-gradient mapping, proposed and accepted full physical update,
cumulative update path, tensor and stress diagnostics, `detF_min/max`, and
inverted-tetrahedron count.

Let `F_512` be accepted sample 0's area-weighted fit RMS, and let `F_B` and `F_C`
be the valid step-544 baseline and candidate values. Select the candidate only
if both probes complete all 32 updates validly, `F_C < F_512`, and

\[
F_B-F_C\ge
\max\left(0.01\ \mathrm{mm},\;0.05(F_{512}-F_B),\;10R_{512}\right).
\]

If the candidate completes but misses this rule, select the valid completed
baseline. If the candidate fails, preserve its failure receipt and select the
baseline only if the baseline completed all 32 updates validly. If the baseline
is invalid or incomplete, no branch can be selected and the experiment stops.
A failed prefix is not an endpoint. Do not retry a failed solve, alter a
tolerance, shorten the probe, or substitute a best state.

## Selected continuation through at most step 1024

Resume the selected arm's exact step-544 `q/u/m/v/t` and keep its frozen learning
rate and `epsilon = 0.01` for every later step. Continue only through decisions
at these block boundaries:

| Block | Updates | End step |
| --- | ---: | ---: |
| 1 | 64 | 608 |
| 2 | 64 | 672 |
| 3 | 64 | 736 |
| 4 | 64 | 800 |
| 5 | 64 | 864 |
| 6 | 64 | 928 |
| 7 | 64 | 992 |
| 8 | 32 | 1024 |

Save full state and optimizer evidence at least every 16 updates. At each
completed boundary define

\[
I_k=F_{\mathrm{start},k}-F_{\mathrm{end},k}.
\]

A block has low progress when

\[
I_k<\max(0.01\ \mathrm{mm},\;0.005F_{\mathrm{start},k}).
\]

Stop for settled fit progress after two consecutive completed low-progress
blocks; any block that is not low progress resets the count.

The projected-gradient rule uses the original common step-256 cached baseline
initial record in `../data/92-baseline32/summary.json` (SHA-256
`8133d0f14f68d5eaea4867119858377e097701f6949568cd661a426a831f12c7`), whose
projected-gradient mapping RMS and maximum absolute coordinate are respectively
`1.0370615515932716e-05` and `0.00023910319521114332`. Do not replace or reset
this reference at step 512 or at a later block. At every selected-continuation
block boundary in the table above, beginning at step 608, evaluate the same
`eta = 1` projected-gradient map without changing Adam state. This rule settles
fit progress only when both the RMS and maximum are at most 5% of their
corresponding common step-256 values.

Evaluate both stopping rules at completed block boundaries before launching the
next block. Step 1024 is the hard cap, and no automatic extension beyond it is
allowed. A two-low-block stop or joint projected-gradient stop before or at the
cap is labeled **fit progress settled**. Reaching the cap while neither rule
passes is a still-progressing fixed-budget endpoint; smoothness and rank remain
deferred. A solver or adjoint failure stops with a failure receipt and does not
establish settled fit progress.

Geometry, motion, surface roughness, stress magnitude, cap fractions,
`detF_min/max`, and inverted-tetrahedron counts remain reported diagnostics.
They are not branch or stopping gates. Operational settling does not prove
inverse convergence, stationarity, target reachability, model capacity,
anatomical validity, physiological validity, uniqueness, or a universally
superior learning rate.

## Conditional smoothness and rank follow-ups

Only a valid source-103 decision with **fit progress settled** may authorize
smoothness work. From that exact endpoint `q/u/m/v/t`, fork three 16-update arms
that differ only in smoothness weight:

- fit-only control: `0`;
- weak smoothness: `1.4762928671047126`;
- full smoothness: `5.9051714684188505`.

All three retain the selected rate, `epsilon = 0.01`, all controls, fit and
magnitude/rank weights, projection, physics, target, and tolerances. Save actual
`Q/u/VTU` states and complete optimizer checkpoints at every step needed for
matching.

Apply the frozen source-82 matching rule without revision, using solver-valid
positive local steps only and excluding local step zero. Match each smoothness
arm separately to the fit-only control. A pair is admissible when absolute
area-fit difference is at most `max(0.02 mm, 0.005 * control fit)` and absolute
area-motion difference is at most `max(0.02 mm, 0.01 * control motion)`. Minimize
the squared fit/motion distance after division by those tolerances; break ties by
the earlier smooth-arm step and then earlier control step. Preserve equal-step
results in the inventory.

A smoothness arm qualifies for rank testing only if its selected admissible
match lowers both normalized tensor variation and the frozen full-face
normalized 5-mm surface high-pass ratio by at least 10% relative to its matched
control and retains `rank_mixing_fraction > 0.05`. Choose the weak arm if both
strengths qualify; otherwise choose the sole qualifying arm. If neither
qualifies, record the failed gates and skip rank.

Fork the exact selected smooth checkpoint into two matched 16-update arms: a
smooth control with `rank_weight = 0` and an otherwise identical
smooth-plus-rank arm with `rank_weight = 119.62893380703123`. No rank run is
authorized without the completed matching receipt and its exact selected
checkpoint. Failed prefixes remain evidence but cannot be rendered or selected.

## Stage order and evidence products

The evidence workflow is:

1. **Source-100 preflight helper:** freeze the pre-existing scientific sources,
   fixed inputs, runtime/helper records, and full parent lineage described above.
2. **Source 100:** validate the preflight freeze, run the three fresh no-update
   samples, perform fixed-map replay and noise aggregation, and provide the sole
   gate authorizing probes.
3. **Document 101:** this preregistered protocol, included immutably in the
   source-100 freeze before any sample runs.
4. **Source 102:** both 32-update probes, the selected fit-only continuation, and
   any decision-authorized smoothness or rank runner modes.
5. **Source 103:** probe selection and every fit-continuation boundary decision.
   Each decision binds its protocol, aggregate receipt, parents, selected
   checkpoint, and exact next update budget by path and SHA-256 before the next
   run begins.
6. **Existing source 82:** apply the unchanged smoothness matching and rank
   qualification logic to a new manifest, producing a new exact-hash receipt.
7. **Source 105:** read-only comparison of actual completed endpoints and traces.
8. **Stage 107:** invoke the immutable existing source-85 renderer directly to
   create new saved-state figures and a new viewer in new output directories,
   without altering old source-85 outputs.
9. **Stage 108:** assemble the final report with a new temporary builder, using
   only completed, hash-pinned receipts.
10. **Source 109:** validate and publish a new subsite, without replacing
   the earlier optimizer report or its files.

Run these stages in numerical and decision dependency order. Every run uses a
new empty output directory and records its exact command and environment. GPU
runs archive their exact sources in the usual per-run provenance and bind their
fixed scientific inputs to the pre-sample freeze. Generated post-processing,
rendering, report, and publication inputs are pinned as they become available.
No GPU work begins before the helper has frozen this protocol and the fixed
scientific inputs and source 100 has validated that freeze. This bounded
experiment can establish reproducible behavior under the frozen model and its
operational rules; it cannot turn those rules into a convergence or capacity
claim.
