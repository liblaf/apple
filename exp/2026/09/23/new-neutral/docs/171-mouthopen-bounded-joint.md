# Bounded joint strain and pose projection

Frozen diagnostic170 validated five reduced determinant gradients and obtained
a certified descending joint direction. Its unbounded projection required a
maximum strain increment 0.349631, exceeding the declared numerical limit 0.01.
No actual seed was produced. The next test puts this same limit inside the
projection and reuses the validated gradients instead of repeating adjoints.

All source012 controls, displacement, optimizer moments and counters 268 remain
frozen. Inputs bind diagnostic170's objective gradient, Adam inverse metric,
determinant gradients, row order, lower bounds, source hashes and receipts.
The metric and total trust gate remain exactly those of diagnostic170.

For inequalities Bx >= b and strain increment box -0.01 <= xq <= 0.01,
pose increments remain unbounded. The exact box minimizer at multipliers
lambda >= 0 is x(lambda) = clip(x0 + D B^T lambda, lower, upper). The six-variable
negative dual is lambda^T(Bx-b) - 0.5 (x-x0)^T D^-1 (x-x0), with gradient Bx-b.
Rows and RHS magnitude are normalized for solving, then certificates are checked
in normalized and original units. Certificates include primal slack, dual signs,
complementarity, box multiplier signs, stationarity, duality gap and exact
bounded reconstruction. This solves the bounded projection; it does not clip
an already accepted unbounded step. Unresolved numerical failures remain visible.

CPU fixtures and the exact saved matrix are tested before a GPU launch. One
actual seed then faces CCD, both original inversion caps, positive original
cells and positive seed-plus-residual forecasts. One nonlinear corrector faces
internal force 1e-9, original physical force 1e-8, contact, Armijo, retained
inversions <= 100 and inverted rest-volume fraction <= 1e-4. The stricter
1e-6 determinant model margin remains a reported prediction target.

Raw seed/corrected states and all failures are preserved. No diagnostic state
or optimizer history is adopted. The numerical budget after rebuilding physics
is 600 seconds. The independently audited full-surface preview remains run012.

## Independent CPU feasibility check

A direct replay of the exact saved170 matrices solved the six-variable bounded
dual in 13 L-BFGS-B iterations (about 1.5 seconds). Minimum original-unit slack
was -1.985e-11, joint slope -0.000225857686 against target -0.000105248343,
maximum absolute strain increment 0.01, with 810 bound components. Metric
correction norm 0.00876752 passed the 0.01917173 trust limit. Complementarity
was 4.92e-15 and interior stationarity residual 8.67e-19. Bound multipliers had
correct signs. The input-hash-bound receipt is `tmp/171-root-bounded-cpu.json`.
This confirms a bounded descending linear proposal exists; actual CCD and
nonlinear physical validity remain untested. No candidate was adopted.

## Reviewed implementation and GPU launch

The implementation's independent CPU replay matched the root candidate within
1.23e-11 in controls. Its polished certificate has minimum slack -2.90e-17,
duality gap -6.78e-21 and 810 active strain bounds. Fifteen CPU cases exercised
box KKT, correlated constraints, zero deficit, inconsistent constraints and
row/RHS scaling. An extreme rescaled case remained explicitly unresolved when
rounding exceeded the original-unit tolerance; it did not claim infeasibility
or relax that tolerance. Ruff and compilation passed. Full frozen provenance,
box dual signs/certificates and physical gates received independent review.

```bash
TMPDIR="$PWD/tmp/bounded-joint-001-runtime" \
CHERRIES_NAME='MouthOpen bounded joint projection test' \
CHERRIES_TAGS='mouthopen,diagnostic,joint-projection,bounded-dual' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/171-test-mouthopen-bounded-joint.py \
> tmp/171-mouthopen-bounded-joint-001.log 2>&1
```

Exact PID 912677, start ticks 4364242, tool session 17410. The state pointer
and job receipt contain command, boot identity and project TMPDIR.

## Seed result

Numerical work and Cherries shutdown completed with exit 1 at the additional
positive-cell/affine forecast screen. No nonlinear corrector ran. The seed
passed rigid-arc CCD with fraction 1, no intersections, 100 retained inversions
and inverted rest-volume fraction 1.2741237208770825e-5. Cells 155249 and
614243 newly inverted, while 580952 and 682855 healed. The additional screen
flagged 18514 (residual forecast), 155249 and 614243. This is a failed proposal
forecast, not failure of the original count-and-volume seed policy.

All bound inputs and source optimizer state remain unchanged. A separate
diagnostic will correct the same saved seed under original physical admission
gates, explicitly retaining the failed extra screen. The original policy has
no minimum-J or fixed inversion-identity requirement.

Comet: <https://www.comet.com/liblaf/apple/89cdf54efd4b4724a19c685c17b3dc06>
