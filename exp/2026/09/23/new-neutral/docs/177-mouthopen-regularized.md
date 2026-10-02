# MouthOpen with position, normal and activation smoothness terms

The user requested L2 + normal + smooth activation regularization. The L2-only
run016 was interrupted with SIGINT after 9 accepted updates, counters 313/313.
The numerical process and Cherries completed with exit 130. Its saved endpoint
passed fresh independent audit146: position RMS 1.8571288929951484 mm, force
0.0009328461440043743 N, 100 retained inversions, inverted rest-volume fraction
1.284627435785114e-5 and valid collision. Its checkpoint, endpoint and final
progress row agree. The immutable solver summary still says running; the
interruption/lineage receipt and actual process exit record its finished status.
The audited full original tet boundary, bones/eyes and fit/force curves are
published from `data/review-mouthopen-coupled-016-complete`, with 15 tailnet
assets byte-verified. This endpoint is not inverse-converged.

## Objective

The new objective keeps the existing neutral-area-weighted position L2
normalization exactly and adds two nonnegative terms:

- Oriented unit face-normal chord error, averaged using fixed neutral skin
  triangle areas. Target normals come from corresponding MouthOpen triangles.
- Same-muscle face-neighbor tensor smoothness of B = I + symmetric(q). The
  Frobenius norm counts off-diagonal entries twice. Conductance is shared-face
  area / centroid distance times the harmonic muscle-volume fraction. Its sum
  is normalized by total physical active volume and a fixed 5 mm length squared.

There are 288172 active cells and 501313 same-muscle edges. Muscle identity is
`MuscleId` in the original mesh, indexed by saved active-cell IDs; none of the
active IDs has a negative muscle label. The graph has 853 connected components
and 406 singleton cells. Thus this requested smoothness term leaves component
constants and singleton activations unpenalized; no magnitude term is added.
Graph weights/topology and source inputs are recorded by hash.

Weights preserve the earlier active-strain chain's relative objective scaling
while retaining this run's L2 units: normal weight 9.039348924363768 and
smoothness weight 6.5083312255418886e-6. A uniform 5-degree normal error has the
same contribution as 2 mm vector position RMS. The smooth coefficient is the
prior 7.2e-7 coefficient multiplied by the common L2 normalization conversion
9.039348924363734. This is a declared starting choice, not an optimized tradeoff.

CPU replay at the exact source016 state gives L2 0.059317162441550206, normal
chord error 0.00969273076645069, normal contribution 0.08761597542786365,
raw activation roughness 12.13672922755798 and regularizer contribution
7.89898538076625e-5. Total objective is 0.14701212772322153. Position RMS
remains 1.8571288929951482 mm. The objective change therefore creates a
same-state jump in total loss, not a deformation or inverse update.

## Gradients, continuation and acceptance

Surface terms enter the existing implicit adjoint. The weighted smoothness
term supplies its direct activation derivative exactly once. The startup
fixed-state Lagrangian check now also evaluates this direct control term.
A toy integration check passes at relative q error 6.21e-12 when included and
fails at error 1.05 when omitted. This validates inclusion of the direct term;
it is not reduced-objective stationarity evidence.

Twenty-four CPU helper checks passed: existing L2 equality, surface/control
finite differences, prior weight rescaling, physical-volume graph weights,
Frobenius off-diagonal factors, constant fields, cross-muscle isolation,
rigid geometry/tensor invariance and collapsed-triangle failure. The exact
source endpoint also passed a separate CPU objective replay. Receipts:
`tmp/mouthopen-fit-objective-cpu-tests.json`,
`tmp/177-direct-control-pullback-check.json`, and
`tmp/177-source-objective-replay.json`.

Run017 starts from exact audited016 controls, displacement and all original
Adam moments/counters 313. New gradients use the new objective; preserved
moments retain prior history. Initial alpha is 0.0625. The production startup
check must pass before any inverse update. Armijo and the joint QP use total
objective gradients; positional RMS is explicitly computed from L2 alone.
Per-term metrics are saved in `loss_components`, and renderer147/live149 mark
this objective transition at the same lineage iteration.

Collision, exact saved skin pre-strain, IsFixed-only boundaries, all-four-fixed
exclusion, force acceptance 1e-8 (0.01 N), internal target 1e-9 (0.001 N),
retained inversion count at most 100 and volume fraction at most 1e-4 remain.
Relative adjoint/predictor shift is 1e-5; numerical trust, determinant margins
and increment bounds remain proposal rules. No diagnostic state is adopted.

## Command

```bash
mkdir -p tmp/run017-runtime
TMPDIR="$PWD/tmp/run017-runtime" \
CHERRIES_NAME='MouthOpen L2 normal and smooth active strain' \
CHERRIES_TAGS='mouthopen,inverse,normal,smoothness,active-strain' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/177-continue-mouthopen-regularized.py \
> tmp/177-continue-mouthopen-coupled-017.log 2>&1
```

Independent integration review found that total gradients reach the joint QP
and Armijo, with the direct regularizer included exactly once. Ruff and
compilation passed. Run017 launched with PID 1071178, start ticks 4994288,
boot `fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 25300; see job.json.

Startup exactly preserves source016 position RMS, pose, activation and counters
313/313. Force agrees within roundoff. Mixed objective components agree with
the CPU replay; graph hashes match. The full fixed-state pullback passes with
best relative q error 1.8724e-7 and worst pose-coordinate error 2.8341e-7,
under the unchanged 1e-3 threshold. The direct regularizer is included.
This is derivative validation, not reduced-objective stationarity.

The actual process, physical settings and changed source snapshots were verified
in `regularized-startup-verification.json`. Four live tailnet assets byte-match
run017; the audited full-surface page remains run016. The state pointer and
exact job receipt are authoritative. No commits or pushes.

At 02:58 UTC the first mixed-objective update passed 100 PNCG and 14 Newton
iterations at alpha 0.015625. Larger alpha 0.0625 failed seed CCD and alpha
0.03125 failed the original retained inversion gate. Accepted total loss is
0.14666192923038096, position RMS 1.8546385002413355 mm, force
0.0008389925258968517 N, 100 inversions and unchanged inverted volume fraction.
Both position and normal errors decreased. Raw activation roughness rose
slightly to 12.13877331917467; the regularizer is active, but each component
need not decrease independently in a weighted objective. All terms are logged
separately. Counters are 314/314 and the run continues.

The independent first-cache CPU audit verifies source q/pose/u and all four
moment updates exactly. Step-314 Adam increments agree within 3.47e-18 for q
and 2.78e-17 for pose. Objective/graph hashes and initial components agree with
CPU replay to 2.22e-16. See `independent-source-audit.json`. These startup and
accepted-step receipts are not a new independent physical endpoint audit or
inverse convergence evidence.

## User deadline

The user subsequently requested computation finish before 14:00 Asia/Shanghai
on 2026-09-30, maximizing useful progress beforehand, then visualization.
Optimization is scheduled to stop at 13:55 local (05:55 UTC), leaving five
minutes for clean shutdown. The existing continuation heartbeat now enforces
this cutoff, and one-shot heartbeat `finalize-mouthopen-before-14-00` wakes
this chat at 13:55 to stop the exact active numerical process, wait for Cherries,
audit the last consistent accepted inverse endpoint and publish its full
surface and curves. The active state pointer records the deadline and phase.
Any additive continuation launched before then must have a wall budget ending
before the cutoff; no inverse restart is authorized after it. If stationarity
has not been verified, the published result will be labelled deadline-limited.

## Final two-host collision-on allocation

The final user instruction is to keep only the collision-on pair, one solution
per machine: MouthOpen continues on this machine, and Smile runs on
paratera-4090. The previous collision-off queue is superseded and its remote
owner has been instructed to stop it cleanly and preserve all prior results.
This chat coordinates both collision-on results for the same 14:00 deadline.
`data/expression-pair-session-state.json` records the definitive allocation.

Smile is prepared in additive expression-specific runner178/wrapper179 from
the audited corrected-IsFixed neutral, with fresh activation, pose and Adam
state. It uses the transferred Smile target and the same collision/prestrain,
boundary, retained-tet and combined-objective policy as MouthOpen. The remote
owner verifies the GPU handoff and deploys the dependency/input closure.
Independent audit180 and renderer181 preserve explicit expression labels.
No local MouthOpen interruption or duplicate remote numerical job is planned.

The deadline monitor and finalizer cover both collision-on endpoints and
require both audited previews before declaring delivery complete. The latest
valid states will be labelled deadline-limited if stationarity is unverified.
