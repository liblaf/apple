# Resolve cancellation in the bounded-dual line search

Run014 stopped at trial 23 after 22 accepted updates. Its endpoint passed fresh
physical audit at RMS 1.9621245226690815 mm, force 0.000996781585126526 N,
100 inversions and inverted rest-volume fraction 1.284627435785114e-5.
Both Adam counters are 290. This is not inverse convergence.

## Frozen failure evidence

L-BFGS returned success, but twelve Newton polish attempts left projected
residual at 7.030285464892927e-9. The final original-unit slack of
-2.9065599984174045e-9 failed the unchanged -1e-10 certificate gate. The
proposal was correctly rejected before CCD or nonlinear correction.

CPU replay of the exact saved point shows that a full Newton step has dual
objective decrease -7.5967e-17, compared with an Armijo requirement of
-1.5193e-20. Subtracting two separately evaluated dual objective scalars rounds
that decrease to zero. The active Hessian has positive eigenvalues; the full
step reduces projected residual to 3.33e-16 and original minimum slack to
-9.63e-16, with unchanged active component bounds. Frozen evidence is saved in
`tmp/run014-attempt23-newton-evidence.json` and its replay archive.

The refinement uses the same exact clipped quadratic objective and Armijo
condition, evaluating its difference directly. For `y = C.T @ mu`,
`z = clip(y)`, actual multiplier increment `delta`, `t = C.T @ delta`,
`z1 = clip(y + t)` and `dz = z1 - z`, the difference is

```text
slack(mu) dot delta + 0.5 * ||dz||^2 + (y + t - z1) dot dz.
```

This avoids subtracting nearly equal full objectives. All original primal,
dual, complementarity, stationarity, gap and box thresholds remain. CPU replay
and clipping-transition checks must pass before launching the continuation.

## Continuation

Script175 starts from exact audited014 q, pose, displacement, Adam moments and
counters 290/290. No diagnostic increment is adopted. Initial alpha is 0.125;
all bounded joint projection settings, extended startup finite differences and
physical policies are retained. The physical force acceptance remains 1e-8
(0.01 N), internal target 1e-9 (0.001 N), with count at most 100 and rest-volume
fraction at most 1e-4. No inversion identity or minimum-J constraint is added.

Run from this experiment directory:

```bash
mkdir -p tmp/run015-runtime
TMPDIR="$PWD/tmp/run015-runtime" \
CHERRIES_NAME='MouthOpen continuation with certified dual refinement' \
CHERRIES_TAGS='mouthopen,inverse,continuation,bounded-joint,dual-refinement' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/175-continue-mouthopen-certified-dual.py \
> tmp/175-continue-mouthopen-coupled-015.log 2>&1
```

Output: `data/inverse-mouthopen-coupled-015`. The pointer and job receipt will
identify the process after launch. A successful numerical proposal is only a
proposal; CCD, nonlinear force equilibrium, Armijo and original physical gates
still determine acceptance. Finished inverse stationarity requires separate
physical and reduced-objective audits.

## CPU verification

Ten CPU tests passed: eight clipping-transition cases, cancellation near a
stationary dual point, and the successful diagnostic171 matrix replay. The
full failed run014 trial23 API replay now passes at alpha 0.125 with minimum
original slack -1.2333816151559318e-16, normalized complementarity about
3e-15, normalized gap about 1.7e-15 and box stationarity 2.17e-19.
The metric correction norm 0.0197038836 is below the unchanged 0.0347718208
trust limit. One full Newton polish step was accepted without backtracking.

Root independently reconstructed original matrix constraints, box multipliers,
stationarity, complementarity, gap, descent and trust from saved arrays. All
passed. AST comparison confirms every original `solve_bounded_dual` assertion
is unchanged. An independent agent reviewed the identity, integration and
source binding without finding a blocker. Ruff and compilation passed.

Evidence: `tmp/stable-dual-polish-cpu-tests.json`,
`tmp/run014-attempt23-stable-replay/`, and
`tmp/175-root-independent-dual-certificate.json`. Numerical differences between
the replay's L-BFGS point and the frozen failed point are retained in their
respective receipts; both are resolved by the same exact decrease calculation.

Run015 has launched; exact identity is in its `job.json` and the active state
pointer. PID 991614, start ticks 4719710, boot
`fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 63144.

Startup matched source014 RMS, pose, activation statistics and counters
290/290 exactly, with force agreeing within floating-point reduction error.
The extended fixed-state derivative check passed. Config, source/input hashes
and four live tailnet assets were verified. The first alpha 0.125 step passed
16 PNCG iterations and the original gates, reaching RMS 1.95818938764492 mm,
force 0.0009603607774849843 N and 100 inversions, counters 291/291. It moved
past the numerical stage that stopped run014. This remains live progress, not
an independent finished-endpoint audit or inverse convergence.

See `certified-dual-resume-verification.json` and `latest-monitor.json` in
the run directory. The full-surface preview is audited run014; live progress
now follows run015. The numerical process is still running.

Independent source audit confirms exact checkpoint q/pose/displacement and
reconstructed step-291 moments; the helper snapshot matches the reviewed fix.
The second alpha 0.125 update also passed actual correction, reducing RMS to
1.9542827469135335 mm and force to 0.0009359287190406355 N, with 100
inversions. Its larger alpha 0.25 trial failed the original inversion gate
and was rejected. Neither numerical certificates nor physical gates were
loosened to obtain these updates.

Comet: <https://www.comet.com/liblaf/apple/f8877265500b459c9db8ac048a5fd51f>

## Finished endpoint and next diagnosis

Run015 ended with code 1 after numerical work and Cherries shutdown, at
attempted update 15 and alpha 0.5. Fourteen accepted updates reached RMS
1.9085361133435759 mm and counters 304/304. The stable dual-decrease fix
worked; this new failure involved a nearly saturated strain box and very large
dual multipliers. Its certificate was correctly rejected.

Fresh physical audit passed for the saved endpoint: force
0.0009821411795372357 N, 100 inversions and inverted rest-volume fraction
1.284627435785114e-5. Endpoint/checkpoint arrays and lineage match. The full
surface, bones/eyes and branch curves were published, with 15 assets verified.
This is not inverse convergence.

Frozen CPU analysis proves the failed alpha cannot satisfy the unchanged
correction trust limit, while smaller alphas certify. Report176 describes
explicit certified rejection and backtracking. No failed or diagnostic state
is adopted, and the physical policy remains unchanged.
