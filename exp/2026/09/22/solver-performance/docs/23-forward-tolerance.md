# Forward accuracy during Smile inverse fitting

The forward solve need not use one very small fixed residual throughout an
inverse fit. Its numerical error should be small compared with the shape
changes, objective decrease, and gradient accuracy needed by that inverse
update. This is a criterion to validate, not evidence that an arbitrary larger
force threshold is adequate for the present model.

## Latest requested tolerance and historical relative reference

The user subsequently requested no outer rejection, zero magnitude/jaw penalties,
and looser forward convergence. The restarting matched full-Adam run uses
`1e-8` in both arms; the former production setting below is historical.
The installed PNCG criterion is `norm(force) <= max(atol, rtol * initial_norm)`.
The first row of `simple-skin-forward-004/trace.jsonl` records initial norm
`1.5192003475221145e-5`. Its `protocol.json` used `rtol=1e-5`, producing the
fixed threshold `1.5192003475221146e-10` later inherited by expression fitting.
For that same reference, `rtol=5e-4` instead gives `7.596001737610573e-9`
code units, or `7.596 mN`. The new `1e-8` is 1.316 times looser than this
historical-reference equivalent. A new warm-started solve has a different
initial norm, so restarting a relative criterion is not equivalent to this
fixed reference-force threshold.

## Current implementation

The live matched comparison explicitly passes
`--forward-atol 1.5192003475221146e-10` to both solvers. The harness class's
`1e-12` default was used only by an interrupted stricter pilot. The production
threshold is an absolute L2 norm of the free-DOF energy gradient with relative
tolerance zero. It is neither mesh-normalized nor a displacement-error bound.
The energy uses MPa and metres, so this force norm is approximately
`0.151920035 mN` in SI units. A small-looking code-unit number is not itself
evidence of excessive physical precision.

For residual `r = dE/du` and local Hessian `H`, the remaining equilibrium
correction is approximately `delta_u = -H^{-1} r`. Thus equal residual norms
can imply different shape errors, depending on the modes left unresolved.
Newton and PNCG may also overshoot a requested threshold by different amounts.

The existing fitter solves an implicit adjoint and estimates the remaining
objective change as `p dot r`, where `p = -H^{-T} dL/du`. Outer acceptance
requires the Armijo margin to exceed the sum of the old and trial absolute
estimates, and also checks the corrected objective. This is a first-order
estimate, not a certified bound on shape or gradient error. At convergence it
is checked together with projected stationarity and objective stability.

Relevant implementation:

- `../src/20-fit-smile.py`: explicit common tolerance and matched run receipts.
- `../../../21/joint-activation-material-mandible/src/joint_expression_equilibrium.py`:
  accepted-state force criterion and terminal geometry checks.
- `../../../21/joint-activation-material-mandible/src/93-fit-expressions.py`:
  `evaluate`, residual-corrected outer Armijo, and convergence checks.

## Sensitivity probe

A separate V100 host GPU 1 probe compared identical active-stress proposals
at requested residual tolerances `1e-8`, `1e-9`, the production threshold, and
`1e-12`. It preserves the common seed, parameters, adjoint tolerance, bone and
eye collisions, and inversion checks. It records achieved residual,
forward time, observed skin-shape error, objective error, and active-stress/jaw
gradient differences against a tightly solved reference. The primary matched
20-update run continues with its recorded common tolerance.

The probe completed on V100 host physical GPU 1 with eight IPCTK threads. It used
the hash-recorded shared neutral and exact first projected-Adam Smile proposal,
then rebuilt the collision model for each arm. Every endpoint passed the full
cranium/mandible/rigid-eye contact audit and zero-inversion gate. Times below
separate total evaluation wall time from forward and implicit-adjoint time;
they are same-GPU observations, not a cross-device speed claim.

| Requested forward atol | Achieved force | Newton steps | Total / forward / adjoint (s) | Skin RMS difference vs tight (mm) | Activation-gradient relative L2 | Jaw-gradient relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-12 reference | 9.19e-15 | 4 | 56.11 / 28.46 / 26.36 | 0 | 0 | 0 |
| 1e-8 | 1.30e-9 | 1 | 36.85 / 9.88 / 25.77 | 1.26e-5 | 5.16e-4 | 1.84e-4 |
| 1e-9 | 4.42e-11 | 2 | 42.47 / 15.33 / 25.92 | 2.01e-6 | 1.47e-4 | 2.96e-5 |
| production 1.5192e-10 | 4.41e-11 | 2 | 42.36 / 15.28 / 25.87 | 1.98e-6 | 1.47e-4 | 2.96e-5 |

The 1e-9 and production arms overshot their requested residuals and reached
nearly the same Newton endpoint. On this full proposal, 1e-8 saved about 18.6
seconds of forward work but left a 5.16e-4 relative activation-gradient
difference. Its maximum observed-skin displacement error was 1.55e-4 mm.
This is useful local evidence, not a tolerance equivalence certificate.

The existing residual correction was also informative. At the full proposal,
the correction estimate was 1.77e-7 at 1e-8, about 3.02e-4 of the absolute
initial-to-proposal objective change; at the production threshold it was
7.17e-9, about 1.22e-5. These are first-order diagnostics, so they cannot
replace measured shape and gradient comparisons.

A half-sized version of the same proposal gives the counterexample to a fixed
1e-8 policy. Its initial force was 7.89e-9, below 1e-8, so direct Newton took
zero steps and returned the warm seed. The forward portion took 0.186 s, while
the implicit adjoint still took 26.02 s. Against its own tight 1e-12 endpoint,
the warm seed differed by 0.00365 mm weighted skin RMS and 2.19e-3 in
activation-gradient relative L2. Its correction estimate was -6.50e-4, 1.38
times the absolute trial objective change. This proposal needs refinement
before an outer Armijo decision; a loose absolute force threshold alone is
inadequate near smaller or warm-started updates.

Commands and complete hash-bound receipts are in
[smile-tolerance-probe-001](../data/smile-tolerance-probe-001/summary.json)
and
[smile-tolerance-probe-half-002](../data/smile-tolerance-probe-half-002/summary.json).
The primary command used `CUDA_VISIBLE_DEVICES=1`, `DEBUG=1`, eight IPCTK
threads, and `src/23-smile-tolerance-probe.py`; the half-proposal repeat added
`--proposal-scale 0.5 --atols 1e-8`.

The direct-Newton ladder alone cannot establish PNCG equivalence: equal
requested residuals can still leave different endpoint errors across solvers.

### Original-PNCG follow-up

The baseline-specific follow-up completed on the same GPU and exact full
proposal, with a 15-minute external cap that was not reached. The PNCG arms
used the same tight Newton reference rather than assuming that equal requested
force implies equal endpoint accuracy.

| PNCG requested atol | Achieved force | Steps | Total / forward / adjoint (s) | Skin RMS difference vs tight (mm) | Activation-gradient relative L2 | Correction / abs trial change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-8 | 8.85e-9 | 15 | 30.17 / 4.91 / 23.84 | 0.00722 | 1.93e-3 | 0.689 |
| 1e-9 | 9.74e-10 | 623 | 222.45 / 194.54 / 26.49 | 0.000930 | 1.27e-4 | 0.172 |

PNCG at 1e-8 is fast but fails the accuracy test for this proposal: its maximum
observed-skin difference is 0.0204 mm and its jaw-gradient relative L2 error
is 1.75e-3. Tightening to 1e-9 improves endpoint agreement substantially, but
costs 623 accepted PNCG steps and 194.5 seconds of forward work. In comparison,
direct Newton at the stricter production threshold took 15.3 seconds of
forward work on the same GPU and left only 0.000002 mm RMS shape difference.
That is about 12.7 times less forward time for this fixed proposal, not an
end-to-end fitting speedup. The production-PNCG endpoint was not timed in
this probe, so this table cannot quantify its saving from tolerance relaxation.
We retain the production threshold for the ongoing matched comparison.
These measurements motivate testing an adaptive method with explicit
endpoint/error checks; they do not establish that every `1e-9` solve is
unacceptable or that the historical tolerance is optimal.

The complete receipt is
[smile-tolerance-pncg-003](../data/smile-tolerance-pncg-003/summary.json).

## Adaptive policy to evaluate

A useful adaptive policy would start with cheaper solves and refine when
forward error is significant compared with the current inverse step. In
particular, an unresolved objective decrease should trigger a more accurate
forward evaluation before simply reducing the outer step. Near stationarity,
accuracy must increase, and the final state should be rechecked tightly.
Collision feasibility and inversion checks are independent of this numerical
accuracy choice and remain enforced.

This principle is consistent with adaptive inexact-gradient methods, which
control gradient error relative to optimization progress; their convergence
theorems do not automatically certify this nonconvex contact implementation.
See [Macedo and Bueno, 2025](https://arxiv.org/abs/2510.17581).
