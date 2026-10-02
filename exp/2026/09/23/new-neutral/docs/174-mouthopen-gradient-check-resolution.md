# Refine the startup derivative check resolution

Run013 stopped before any inverse update because the fixed-state translation-x
Lagrangian check missed its relative agreement threshold of 1e-3. As epsilon
decreased from 1e-4 to 1e-5 to 1e-6, relative error decreased from 0.474048 to
0.0407565 to 0.00195874. The analytic prediction was 0.000833583678032535;
the last central difference was 0.0008319509040011219. This pattern motivates
checking smaller stencils. It does not by itself prove the derivative correct.

`mouthopen_gradient_check.py` now accepts an explicit epsilon sequence, recorded
in its receipt. The default remains the original three values. Script174
adds 3e-7 and 1e-7 for run014. All original samples are retained, and the same
1e-3 relative-error threshold remains. If none passes, startup stops visibly.
This checks a fixed-state Lagrangian derivative; it cannot establish
re-equilibrated reduced-objective stationarity.

Run014 still initializes from the exact independently audited run012 checkpoint,
including q, pose, displacement, all Adam moments and counters 268/268. There is
no run013 checkpoint or accepted update to continue. All bounded joint proposal
and physical settings are those documented in report173. Diagnostic172 remains
method evidence and is not adopted.

## Command

```bash
mkdir -p tmp/run014-runtime
TMPDIR="$PWD/tmp/run014-runtime" \
CHERRIES_NAME='MouthOpen bounded joint with resolved derivative check' \
CHERRIES_TAGS='mouthopen,inverse,continuation,bounded-joint,derivative-check' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/174-continue-mouthopen-refined-gradient-check.py \
> tmp/174-continue-mouthopen-coupled-014.log 2>&1
```

Output: `data/inverse-mouthopen-coupled-014`. Verify startup and nonlinear
accepted updates from its actual receipts. Inverse convergence is not yet
established; physical audit and reduced-objective stationarity checks remain
required for a finished candidate.

## Launch and validation

Ruff, compilation and CPU Config checks passed. The helper's CPU fixture
verified unchanged default sample count, extended sample count, expected
derivatives, state restoration and invalid-stencil rejection. CUDA was not
initialized by those checks.

Run014 launched around 01:37 UTC on 2026-09-30. PID 941069, start ticks 4521424,
boot `fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 25616. Exact receipt:
`data/inverse-mouthopen-coupled-014/job.json`. The independently audited source
endpoint stays run012; current accepted progress is recorded below.

Startup matched the audited source's RMS, pose, activation statistics and
both counters exactly; force agreed within 2.4e-18 N. The finer translation-x
check passed with relative error
9.82032515469295e-8 at epsilon 3e-7 and 1.5369772526707482e-7 at 1e-7. All
pose coordinates and q pass the original 1e-3 threshold. The first joint inverse
update has begun. Source/input hashes, numerical settings and four live tailnet
assets are verified in `bounded-joint-resume-verification.json`.

An independent CPU comparison verified exact source q, normalized pose and
displacement and reconstructed the proposed step-269 Adam moments from original
source012 history. Proposed q increments agree within 1.74e-18 and pose exactly.
See `independent-source-audit.json`.

At 01:40 UTC the first three updates passed actual nonlinear correction:

| Local update | Alpha | PNCG / Newton | RMS (mm) | Force (N) | Inversions |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.125 | 120 / 5 | 1.9783450030 | 0.0008821392 | 100 |
| 2 | 0.0625 | 140 / 5 | 1.9720125240 | 0.0008715739 | 99 |
| 3 | 0.015625 | 60 / 7 | 1.9702758614 | 0.0008957501 | 100 |

The first proposal closely reproduces the successful diagnostic172; later
proposals use fresh joint determinant gradients. Larger subsequent trials were
rejected for exceeding the original inversion count; smaller corrected trials
passed. Both counters are now 271. The third pose rotation norm is
8.584291797 degrees and translation norm 4.424253620 mm, with inverted
rest-volume fraction 1.284627435785114e-5. This is progress, not an independent
finished-endpoint audit or a stationarity certificate.

The exact numerical process remains running. `latest-monitor.json` records
the process identity and accepted forward receipts. The heartbeat continues
monitoring; no finite three-step stopping limit was introduced.

Comet: <https://www.comet.com/liblaf/apple/f3e8e60ff515451b8829383bff1b6073>

### 01:52 UTC monitor

Run014 reached accepted update 20, counters 288/288, RMS
1.96512051269298 mm, force 0.0009183859330394746 N and 100 inversions.
The earlier small, zero-correction seeds did not become an established stall:
updates 17, 18, 19 and 20 performed respectively 100/3, 80/7, 5/0 and 6/0
PNCG/Newton steps. Alpha recovered to 0.015625. The exact process is still
running; loaded sources and all four live tailnet assets were verified.

A read-only CPU review found that cell 580952 entered the selected determinant
set at update 6, following which alpha repeatedly doubled. Rejected-seed
rest-volume changes strongly implicate omitted cells 580952 and 682855, but
these are volume-fingerprint inferences because rejected raw seeds were not
saved. Full evidence and its limitations are recorded in
`tmp/run014-seed-selection-diagnostic.json`. There is no present reason to
interrupt the recovering run or change physical acceptance. This remains
progress without a reduced-objective convergence certificate.

### Completed with an unresolved numerical projection

Run014 ultimately stopped at attempted update 23, before that trial reached
CCD or nonlinear equilibrium. Both the numerical process and Cherries exited
with code 1. The saved endpoint has 22 accepted updates, counters 290/290,
RMS 1.9621245226690815 mm and force 0.0009967815851265247 N.

The bounded dual optimizer reported success, but twelve Newton polish attempts
left the normalized projected gradient unchanged at 7.030285464892927e-9.
Minimum original-unit slack was -2.9065599984174045e-9, failing the unchanged
-1e-10 certificate threshold. The code correctly rejected that proposal.
This is a numerical certificate failure, not physical infeasibility or inverse
convergence. Complete failed coefficients and certificate are retained under
`joint-projections/00023/`.

Fresh independent audit146 passed for the saved endpoint: force
0.000996781585126526 N, 100 inversions, rest-volume fraction
1.284627435785114e-5 and collision feasible. The frozen checkpoint and all
optimizer history remain available. CPU investigation is testing a numerically
accurate evaluation of the small dual objective decrease while retaining all
certificate thresholds. See the state pointer for the subsequent run.
