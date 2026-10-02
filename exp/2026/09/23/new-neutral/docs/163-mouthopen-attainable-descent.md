# MouthOpen continuation with an attainable decrease target

Run010 stopped after three accepted updates because its zero-pose witness
could not meet the determinant prediction constraints at the next proposal.
The saved endpoint is independently audited, with RMS 1.9914515183673682 mm,
force 0.0009277460627128851 N, 100 inversions, inverted rest-volume fraction
1.2304029393884475e-5 and both Adam counters 257. This is not inverse
convergence. Run010, its failed proposal inputs and its optimizer history are
preserved.

Independent audit146 and full-surface review147 finished normally, exit 0.
The checkpoint, endpoint and summary arrays, hashes and counters agree. The
full original tet boundary, bones/eyes and branch fit/force curves are published;
all 15 tailnet assets were byte-verified and the front comparison and curves
visually inspected. The fixed-control refinement remains marked at the same
optimizer iteration. The fit still opens less than the target.
[Audit](https://www.comet.com/liblaf/apple/270324351d8b4857ac9e4b2775a68cd0),
[review](https://www.comet.com/liblaf/apple/31e8be402f1e43049641f02efe0e9698),
audited preview (private preview omitted).

## Linear feasibility diagnosis

The complete attempt 4 matrix was saved before failure. Only original cell
688235 requires nonzero pose motion: its old determinant is 2.837156513e-5
and the strain-only prediction is -2.348915146e-5, requiring pose lift
2.448915146e-5 to meet the unchanged 1e-6 prediction margin.

Minimizing pose slope over the geometry constraints gives a best joint linear
slope of -0.0003209279050438198, compared with the infeasible original target
-0.0004799426706666772. The LP has minimum primal slack -2.29e-18,
nonnegative dual multipliers, dual stationarity residual 9.50e-18 and zero
duality gap. A half-optimum target of -0.0001604639525219099 supports a
nearest-Adam QP with minimum slack 3.90e-18, KKT residual 3.27e-17 and
complementarity residual 3.70e-18. Its minimum predicted determinant is
1.00000000008e-6. These certify the linear proposal only.

## Continuation rule

The additive run011 will use the exact independently audited run010 checkpoint,
all its moments and counters 257. No diagnostic candidate is adopted. The
original-target-feasible path remains unchanged. When that target is infeasible,
an explicit new policy computes the geometry-constrained minimum pose slope.
If the certified best joint slope is negative, half of it defines an attainable
target, and its LP minimizer initializes the nearest-Adam QP. Infeasible geometry,
a nonnegative best slope, inconsistent LP results or invalid certificates stop
visibly for diagnosis. Older wrappers retain their declared zero-pose policy.

Each proposed step still undergoes CCD, nonlinear force equilibrium, Armijo
and the original physical gates. These are identical to the checks a separate
one-step GPU diagnostic would perform, so the evidence supports testing the new
direction directly in the additive continuation's line search. Nonlinear
acceptance remains unverified until that step completes.

Collision, exact saved skin pre-strain, authoritative IsFixed-only constraints,
all-four-fixed-tet exclusion, original force acceptance 1e-8, internal force
target 1e-9, count cap 100 and inverted rest-volume fraction 1e-4 are unchanged.
Relative adjoint/predictor shift stays 1e-4, analytic q/pose probe epsilon 5e-5,
q/pose learning rates 0.002/0.1, no additional pose increment caps and initial
alpha 0.25. Numerical proposals do not certify inverse stationarity. No commits
or pushes.

## Run011 launch and verification

`src/163-continue-mouthopen-attainable-descent.py` enables
`projection_witness_policy="attainable_optimum"`; the base default remains
`"zero_pose"`. CPU tests cover a required nonzero jaw direction, preserved
feasible targets, non-descending optimum, infeasible geometry, unresolved or
unbounded LP and an invalid dual certificate. Existing zero-pose/default cases
also pass. The exact attempt4 production replay matched the diagnosed direction
to 1e-14. Ruff, formatting, compilation, CPU config imports and invalid-policy
validation passed without initializing CUDA.

The hash-bound CPU evidence is
`data/inverse-mouthopen-coupled-010/attainable-descent-diagnostic.json`.
Audited010 source files were verified unchanged after these checks.

```bash
TMPDIR="$PWD/tmp/run011-runtime" \
CHERRIES_NAME='MouthOpen attainable descent continuation' \
CHERRIES_TAGS='mouthopen,inverse,collision,attainable-descent' OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/163-continue-mouthopen-attainable-descent.py \
> tmp/163-continue-mouthopen-coupled-011.log 2>&1
```

PID 784785, start ticks 3758184, boot ID
`fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 18336. Exact process and
project TMPDIR are recorded in run011 `job.json` and the state pointer.
`attainable-descent-resume-verification.json` verifies startup source hashes,
RMS/pose/activation statistics, counters 257, numerical settings and physical
acceptance policy. The live page points to run011 and its three assets were
byte-verified over tailnet. The published audited full-surface preview is run010.

The first two updates accepted alpha 0.25 and 0.5, using 80/4 and 140/5
PNCG/Newton steps. Update 2 reached RMS 1.988574035068851 mm, force
0.0009005623744047287 N, 100 inversions and counters 259. Thus the new rule
has passed actual nonlinear correction beyond the previous failure. These
are live progress values; no inverse stationarity claim is made.

At 23:33 UTC, update 3 had accepted alpha 0.5 with 100 PNCG and 4 Newton
steps: RMS 1.9857478463262161 mm, force 0.0007992172890361223 N, 100
inversions and counters 260. All first three LP/QP certificates and
original/chosen-target archives were independently checked on CPU. The five
changed sources match their running snapshots. The exact process was active,
and summary/progress assets still matched tailnet bytes. The full audited
preview remains run010; run011 continues beyond this startup check.

[Comet run011](https://www.comet.com/liblaf/apple/ff3bd30a5606478d98112a07c014b7d6)
and live progress (private preview omitted).

## Finished without inverse convergence

Run011 finished normally after 10 updates with
`line_search_below_declared_resolution`. Late updates 7–10 had zero nonlinear
correction and shrinking alpha; corrected trials reached force tolerance with
101 retained inversions. The saved endpoint at counters 267 passed independent
audit and is now the full-surface preview: RMS 1.9818275442754796 mm, force
0.0009999463536325265 N and 100 inversions. Read
`164-mouthopen-corrector-determinants.md` for the next frozen diagnostic of
residual correction versus damping error. The limits remain unchanged.
