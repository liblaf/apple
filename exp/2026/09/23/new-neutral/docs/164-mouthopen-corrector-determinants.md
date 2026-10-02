# MouthOpen determinant change during nonlinear correction

Run011 finished normally with `line_search_below_declared_resolution` after
10 accepted updates. Both Adam counters are 267. Updates 7–10 used alpha
6.1035e-5 down to 3.8147e-6 and no nonlinear correction; force approached the
internal 1e-9 cutoff. At attempt 11, corrected trials down to alpha
1.9073486328125e-6 reached force tolerance but had 101 retained inversions.
The next alpha fell below the declared 1e-6 line-search resolution. This is
a numerical/feasibility stall, not inverse convergence.

The numerical process and Cherries shutdown completed with exit 0. Endpoint,
checkpoint and summary arrays/counters/hashes agree. A fresh independent
audit passed: RMS 1.9818275442754796 mm, force 0.0009999463536325265 N,
100 retained inversions, inverted rest-volume fraction 1.2304029393884475e-5
and feasible collision geometry. The full original tet boundary, bones/eyes
and branch fit/force curves were rendered and published; all 15 tailnet assets
were byte-verified and the front comparison inspected.

Audited run011 preview (private preview omitted).

## Evidence and unresolved cause

The LP/QP proposals remained certified; this failure is in corrected geometry.
For late rejected trials, the extra inverted rest-volume fraction is
2.584866693446365e-7, matching original cell 18514 to 3.7e-22. Its current
determinant is 5.231588533e-6. It has three fixed vertices and one free vertex,
so it cannot be excluded as all-fixed. The free apex lies about 5.53 nm above
the fixed face. The cell's fat fraction is 0.9755859375; no vertex is a lip.

At the rejected tiny alpha, the tangent model still predicts J about 5.23158e-6.
The volume fingerprint strongly suggests that correction inverts cell 18514,
but the failed corrected displacement/per-cell J arrays were not saved. The
exact crossing therefore remains to be measured.

The tangent correctly omits the old equilibrium residual: it solves
`(H + shift I) du = -(delta material force + H delta fixed)`.
This gives zero update at zero control change. The residual is not divided
into the small q/pose probes. However, a subsequent nonlinear solve can move
the tissue to remove that residual independently of control step size. Near
a flat tetrahedron, that correction may invalidate a determinant forecast.
The final strain tangent also retains a 3.34% native unshifted residual at
relative shift 1e-4; pose probes have about 0.016–0.026%. Damping error and
residual correction both need to be distinguished before another continuation.

## Frozen diagnostic164

The additive diagnostic binds the exact audited run011 controls, displacement,
moments and counters 267. It compares relative shifts 1e-4 and 1e-5 while
keeping internal force target 1e-9 and the original physical policy unchanged.
Fresh gradients and source-Adam proposals are evaluated per case. A common
strain direction supplies a separate comparison of tangent operators.

Each case computes a separately labelled residual-only linear response
`w = -(H + shift I)^-1 r`, holding all controls fixed. It records the response
and cofactor determinant derivative as a prediction; it is neither an
equilibrium nor an adopted state. This term is kept separate from the q/pose
epsilon probes. The comparison is between `J_old + alpha DJ[du]` and
`J_old + DJ[w] + alpha DJ[du]`, with the residual term unscaled by alpha.

The diagnostic tests a changed alpha 0.125 proposal through production CCD,
seed inversion, nonlinear forward, Armijo and physical gates. A baseline
tiny-alpha replay at 1.9073486328125e-6 additionally captures the actual
corrected determinants that were missing from the failed inverse run.
Full old/predicted/seed/corrected determinants and rejected displacement arrays
are retained. Numerical work has a declared 600-second budget per case;
budget exhaustion is a failed diagnostic, not convergence.

No diagnostic candidate or optimizer state is adopted. Collision, exact saved
skin pre-strain, IsFixed-only constraints, all-four-fixed-tet exclusion,
original force acceptance 1e-8, count cap 100 and inverted rest-volume fraction
1e-4 are unchanged. No physical allowance is loosened, no force tolerance is
tightened, and no commits or pushes are made.

## Launch

The diagnostic passed CPU compilation, Ruff, configuration, input hash and exact
checkpoint/endpoint checks before launch. PID 821218, process start ticks
3935259 and boot ID fe94509a-c363-40c5-9ffa-8a0cc2ad928e are recorded in
`data/mouthopen-corrector-determinants-001/job.json`. Tool session is 24833.
The exact command uses `src/164-diagnose-mouthopen-corrector-determinants.py`,
`OMP_NUM_THREADS=4`, and project temporary directory
`tmp/corrector-determinants-001-runtime`; output is
`data/mouthopen-corrector-determinants-001`, log is
`tmp/164-mouthopen-corrector-determinants-001.log`.

[Diagnostic Comet run](https://www.comet.com/liblaf/apple/aac7b3c3f2b142d6a709cf12af2b8bc6).
The state pointer and heartbeat now identify this diagnostic. The live page
retains audited run011 metrics and links the diagnostic receipt. All four
live HTML/JSON/progress assets matched their local bytes at startup.

If residual correction dominates, a single projection followed by multiplying
its pose direction by alpha is insufficient. A future proposal would need to
account for the residual intercept at each actual trial alpha, preserving the
unscaled residual term and the seed determinant constraint. This is a
conditional numerical design, not an adopted method; diagnostic results are
required first.

## Baseline replay confirms the crossing

At relative shift 1e-4, the tiny replay completed nonlinear correction and
reached raw force 9.684877628322077e-10. Its only newly inverted original
cell was 18514: old J 5.2315885329365045e-6, tangent prediction
5.231580461821902e-6, actual seed J 5.231581079316885e-6, corrected J
-2.2588028872846887e-5. The corrected state had 101 inversions and was
rejected by the original count cap. Maximum correction motion over all
vertices was 5.932962500402633 micrometers.

The separate residual-only linear response predicted dJ -7.437984085594881e-4
at that cell, giving an affine intercept near -7.3856682e-4. It has the
observed sign but substantially overpredicts this partial nonlinear
correction; it must not be described as an accurate corrected determinant.
Its native unshifted linear residual is 40.64%, despite a verified shifted
residual below 1e-7. The residual response remains a diagnostic prediction.

The larger alpha 0.125 baseline proposal was rejected at seed inversion
admission. The lower-shift case is still running. These cases do not change
run011 or its optimizer state.

## Completed diagnostic

Both cases and Cherries shutdown finished with exit 0. Baseline took 79.64 s;
lower-shift case took 69.44 s. All source controls, optimizer moments and
bound file hashes remained unchanged; completion evidence is saved in
`data/mouthopen-corrector-determinants-001/completion-verification.json`.

At shift 1e-5, alpha 0.125 passed seed collision/inversion admission with
100 inversions, then corrected to 101. The only new inversion was again
cell18514. Its predicted J was 4.702639966322394e-6, seed J
3.422643671786417e-6, corrected J -0.0003766810696617541. Raw force was
8.600595191550659e-10. Loss decreased from 0.06755041076135844 to
0.06744410098281733 and passed Armijo, but the physical count gate rejected
it. Lower damping alone did not resolve the stall.

The next diagnostic question is whether alpha-specific constraints that
protect both the seed and the separately predicted residual correction can
supply a feasible descending corrected proposal. This will be checked first
on the saved matrices. No physical policy or source checkpoint was changed.

## CPU feasibility of residual-aware trials

The saved matrices were used to test both shifts and trial alpha values 0.125,
0.25 and 0.5. Each QP uses the actual normalized pose increment `v`:

`A v >= margin - J_old - min(residual_delta_J, 0) - alpha * q_delta_J`.

This simultaneously protects the seed and the affine corrected prediction.
It leaves the residual offset unscaled and scales the strain increment and
original objective target by alpha. The existing certified attainable-descent
policy remains in use. Every result passed a separate full check of all
1,144,168 currently positive retained tetrahedra; all six initial active sets
already sufficed. Worst QP KKT residual was 1.88e-16 and worst margin slack
-8.74e-16. Inputs and helpers are hash-bound in `affine-proposal-cpu.json`.

The baseline-shift alpha 0.125 proposal changes normalized pose by
`[-0.0028357067135114876, 0.00015111120757592162, -0.0006556632139308396,
0.00013901142485563296, -0.0013664329709353632, 0.00017390582658192842]`.
Its pose change is about 0.02914 degrees and 0.01384 mm. It satisfies the
original target with joint increment slope -6.019532024187333e-5 and QP KKT
residual 6.39e-18. At cell18514 the predicted seed J is 0.000744798409 and
affine J is 1e-6. Nonlinear validation is still required. New frozen diagnostic
165 will test this one proposal without changing shift or adopting state.
