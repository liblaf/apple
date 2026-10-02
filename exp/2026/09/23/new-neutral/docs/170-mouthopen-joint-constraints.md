# Joint strain and pose constraint diagnostic

Diagnostic 169 reached force equilibrium but produced 103 retained inversions
and worse fit. The current determinant projection changes only six pose
coordinates while holding the 1,729,032-dimensional strain direction fixed.
This diagnostic tests whether that restriction prevents useful steps.

The source is the independently audited run012 endpoint and exact optimizer
state at counters 268. One differentiable source solve must return precisely
the source displacement with zero correction. At relative adjoint shift 1e-5,
a separate implicit adjoint supplies each selected geometric determinant's
strain and normalized-pose gradient. Initial rows are positive cells below
J = 1e-4 plus previously limiting cells 18514 and 155249: five cells in total.
Directional products are compared against saved source tangents. Shifted and
native residuals are recorded; these remain approximate derivatives.

Let x be the actual joint control increment, x0 be 0.125 times the fresh Adam
proposal from saved moments, and D be the diagonal inverse metric
learning_rate / (sqrt(proposed bias-corrected second moment) + 1e-12).
The proposal minimizes 0.5 (x-x0)^T D^-1 (x-x0). Determinant rows protect
both the seed forecast and the separate residual-offset forecast with target
J = 1e-6. A descent row requires at least 0.1 of the unprojected predicted
decrease. For constraints Bx >= b, the small dual QP uses K = B D B^T,
c = b - B x0, and x = x0 + D B^T lambda. Feasibility, nonnegative multipliers,
stationarity and complementarity must pass in normalized and original units.

The correction norm in this metric must be at most twice the baseline norm.
The maximum absolute strain increment must be at most max(0.01, 2 max|x0q|).
These are explicit numerical proposal trust gates; failed proposals are
rejected rather than clipped. No extra pose caps are added.

At most three actual seeds and 16 determinant rows are allowed. New observed
violations can add rows; an already-constrained observed violation stops
visibly. The actual seed must pass CCD and both original inversion caps,
preserve all previously positive cells, and have positive seed-plus-residual
forecasts. Its deficit against the separate 1e-6 model target is reported.
The previous 1e-10 margin comparison is not a physical acceptance condition.

One collision-on nonlinear corrector then tests the candidate. Original
physical force 1e-8, internal target 1e-9, contact, Armijo, count 100 and inverted
rest-volume fraction 1e-4 remain mandatory. All raw states and signs are saved
before acceptance. No diagnostic state or optimizer history is adopted.
The numerical budget is 600 seconds after rebuilding physics.

## Launch and CPU verification

Both implementation and independent review checked determinant orientation,
normalized pose differentiation, projection signs and metric scaling. Ruff and
compilation passed. Thirteen implementation fixtures and seven independent
analytic checks covered scaled rows/RHS from 1e-12 to 1e12, correlated
constraints, known primal solutions, infeasibility, invalid PSD and zero deficit.
CPU imports did not initialize CUDA. The independent receipt is
`tmp/170-root-cpu-review.json`.

```bash
TMPDIR="$PWD/tmp/joint-constraints-001-runtime" \
CHERRIES_NAME='MouthOpen joint strain and pose constraints' \
CHERRIES_TAGS='mouthopen,diagnostic,joint-projection,determinant-adjoints' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/170-diagnose-mouthopen-joint-constraints.py \
> tmp/170-mouthopen-joint-constraints-001.log 2>&1
```

PID 900174, start ticks 4297669, tool session 87551. The exact command,
boot identity and project TMPDIR are saved in the job receipt and state pointer.

## Result

Numerical work and Cherries shutdown completed with exit 1 at the explicit
strain step trust gate. All five determinant gradients passed both residual
and cached tangent comparisons. The joint QP was certified: minimum slack
-4.32e-17, joint slope -0.00025174854 against target -0.00010524834. Its metric
correction norm 0.00871428 was below 0.01917173. However, maximum absolute
strain increment 0.349631 exceeded the declared 0.01 limit; 652 components
exceeded 0.01. No actual seed or nonlinear corrector was run. This establishes
a proposal scaling problem, not inverse convergence or physical infeasibility.

All source hashes and optimizer state remained intact. Full gradients, metric,
QP inputs and certificates are saved for a CPU replay. The next diagnostic
will put the same component bound inside the projection rather than reject a
solution computed without it. Original physical gates remain.

Comet: <https://www.comet.com/liblaf/apple/80d80b506cb54a5f80ee61b6fb44bb1f>
