# MouthOpen continuation with residual-aware trial proposals

Run011 stalled at 1.9818275442754796 mm RMS with 100 retained inversions.
Diagnostic164 confirmed that cell18514 remained positive in the tangent seed
but inverted in nonlinear correction. A separate fixed-control residual
response explained a missing offset in the determinant forecast. The original
shift1e-4 affine proposal still failed physical admission, while diagnostic165
output003 at shift1e-5 passed 100 PNCG iterations, 5 Newton steps, force,
collision, Armijo and both inversion gates. Its candidate was not adopted.

Run012 is prepared from the exact independently audited run011 checkpoint:
controls, displacement, all Adam moments and counters267. It recomputes a
fresh lower-shift gradient, preserving the prior moments as optimizer history.
The initial trial alpha is0.125; q/pose rates remain0.002/0.1. Both adjoint
and predictor use relative shift1e-5. Damping remains a numerical aid, not
physical stiffness.

The opt-in method builds the full retained-tet q and six pose determinant
tangents once per inverse update, using cofactor derivatives with epsilon5e-5.
A separate zero-fixed-displacement residual response supplies an unscaled
intercept. Each line-search trial solves for its actual pose increment with
both seed and affine determinant constraints. It checks every currently
positive retained cell and adds omitted violated constraints explicitly.
The residual response is never inserted into a control derivative probe or
production seed. The actual pose increment is applied once, and Armijo uses
the actual joint increment. Adam moments/counters change only after an
accepted corrected state and its gradient reevaluation.

This changes the numerical proposal only. Production CCD, nonlinear force
equilibrium, collision validity, objective acceptance and original inversion
gates still decide each step. The internal force target remains1e-9; original
physical acceptance remains1e-8. Exact saved skin pre-strain, IsFixed-only
constraints, all-four-fixed-tet exclusion, maximum100 retained inversions and
maximum1e-4 inverted rest-volume fraction remain unchanged. No extra pose caps
are introduced. Old wrappers retain their prior projection method by default.

CPU tests cover alpha scaling of actual increments, the unscaled residual
intercept, immutable coefficient arrays, full positive-cell constraint closure,
and visible failure. The saved diagnostic proposal was reproduced through the
new CPU helper. The GPU cache and production integration require live startup
verification before any progress claim.

## Reviewed launch

Independent source review found no launch blocker. Ruff, CPU compilation,
configuration/import checks, new166 behavior checks, and existing149 projection,
LP/QP witness and cofactor checks passed. CUDA remained uninitialized during
CPU validation. The GPU was idle before launching one run.

```bash
TMPDIR="$PWD/tmp/run012-runtime" \
CHERRIES_NAME='MouthOpen affine residual continuation 012' \
CHERRIES_TAGS='mouthopen,inverse,collision,affine-residual,continuation' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/167-continue-mouthopen-affine-residual.py \
> tmp/167-continue-mouthopen-coupled-012.log 2>&1
```

Working directory is this experiment group. Exact PID851682, process start
ticks4062685, boot ID fe94509a-c363-40c5-9ffa-8a0cc2ad928e and tool session58946
are in `data/inverse-mouthopen-coupled-012/job.json`. The pointer and heartbeat
identify012. [Comet run012](https://www.comet.com/liblaf/apple/a6fb47d2ed3144599c6f68a1d0284e3a).

Startup reproduced source011 RMS1.9818275442754778 mm, pose rotation8.6282772
degrees, translation4.3784533 mm, force0.0009999463536325241 N and100inversions.
Protocol checkpoint/endpoint hashes match the independently audited source,
refinement is null and both initial optimizer counters are267. The startup
scaled gradient is0.0162214 with the explicitly approximate lower-shift adjoint;
this is not an inverse stationarity result.

## First accepted correction

The first run012 update was accepted at alpha0.125 after60 PNCG iterations
and5 Newton steps. RMS was1.9802796612356741 mm, force0.0008540888196111235 N,
retained inversions100 and counters268/268. The cache residual response had
8.25% native unshifted linear residual; exact material/fixed-state restoration
was verified. The small difference from the frozen diagnostic is consistent
with iterative linear solves, while actual physical gates were rechecked.

`affine-resume-verification.json` records source checkpoint/endpoint hashes,
exact initial RMS/pose/activation statistics and counters, numerical/physical
configuration checks, changed-source matches to the running snapshot, the
actual corrected accepted step and four byte-verified live assets. The source
is still not an inverse-converged result. The audited full-surface preview
remains run011 while012 continues.

## Second update stalled at the seed gate

Run012 and Cherries finished with exit 0 and status
`line_search_below_declared_resolution`, after one accepted update. Both Adam
counters are268. Every tested alpha at attempted update2, from0.25 through
1.9073486328125e-6, failed seed inversion admission; no nonlinear corrector ran
for these trials. The QP predictions still met both1e-6 margins. At alpha0.25,
actual rotation was0.32681 degrees and translation0.04420 mm; even the tiniest
alpha retained0.03554 degrees and0.002205 mm, with54 micrometers maximum free
seed motion. The finite affine repair cannot be removed by backtracking alpha.

Fresh independent audit146 passed for the saved endpoint: RMS
1.9802796612356741 mm, force0.0008540888196111258 N,100 inversions, inverted
rest-volume fraction1.2304029393884475e-5, collision feasible. Checkpoint and
endpoint arrays/counters/hash match. Renderer147 completed;15 full-surface,
bones/eyes and curve assets matched tailnet bytes. Front comparison and
fit/force curves were inspected. This is not inverse convergence.

Audited run012 preview (private preview omitted).

The next diagnostic is a frozen seed-prediction test using the saved attempt2
cache. The volume fingerprints suggest cells656201 and688235 cross zero in
small-alpha seeds, but raw rejected seeds must first confirm the IDs. At fixed
alpha0.125, measured absolute seed-model defects can update both shared seed
and affine QP forecasts, without accumulating errors or changing physics.
At most four seed attempts precede at most one actual equilibrium correction.
All raw seeds and per-cell determinants will be saved. No diagnostic candidate
or optimizer state is adopted.
