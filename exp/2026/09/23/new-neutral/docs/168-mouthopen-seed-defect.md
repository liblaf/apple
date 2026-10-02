# Frozen MouthOpen seed prediction correction

Run012 accepted one corrected step, then every attempted second-update seed
exceeded the retained inversion count. All18 trial QPs met their linear
seed/affine margins. Even the smallest alpha retained a finite pose repair
because the force-residual intercept is unscaled. Backtracking could not
remove the finite-seed model error. This is a numerical stall, not inverse
stationarity. The saved012 endpoint passed an independent physical audit and
was published with full boundary, bones/eyes and fit/force curves.

This diagnostic freezes the exact audited012 endpoint and optimizer state
(counters268) and uses the saved attempt2 tangent cache, source strain direction,
pose proposal and gradient. It does not take an optimizer step. The fixed
trial alpha is0.125; relative shift is1e-5. First it reproduces the saved QP
and records the actual production seed, CCD and every retained determinant
before any geometry rejection. The inferred crossing cells656201/688235
must be checked against those saved arrays.

At seed attempt `k`, measure the absolute defect against the original cached
linear model:

`e_k = J_actual_seed(v_k) - (J_old + alpha * q_delta_J + A v_k)`.

The next QP uses the same Jacobian and the latest absolute defect in both seed
and affine forecasts:

`A v >= margin - J_old - alpha * q_delta_J - min(residual_delta_J, 0) - e_k`.

The defect is not accumulated, multiplied by alpha, or divided by probe
epsilon. The original source q/pose/u is restored for every production seed.
The nearest requested actual pose increment and the certified original or
attainable descent rule remain. Every currently positive retained cell is
checked. The seed correction has four total attempts: initial plus at most
three corrections. Each attempt saves full raw displacement, J, defect,
constraints, certificates and sign changes.

An actual corrector is attempted only after the seed passes production CCD,
the original inversion gates, and both observed seed and observed
seed-plus-residual model margins. The observed affine forecast remains a
numerical prediction. The final state must independently pass force, collision,
Armijo and the original count/volume gates. A600-second diagnostic budget and
four-seed limit are explicit failure bounds, not convergence criteria. No
candidate or optimizer state is adopted.

Collision, exact saved skin pre-strain, IsFixed-only boundary labels,
all-four-fixed-tet exclusion, original physical force1e-8, internal force1e-9,
maximum100 retained inversions and maximum1e-4 inverted rest-volume fraction
remain unchanged. No commits or pushes are made.

## Reviewed launch

Root source review and an independent algebra review passed. CPU Ruff,
compilation, configuration and source/audit/cache hashes passed, and the
initial QP reproduced the saved trial01 pose increment to1e-12. CUDA remained
uninitialized during CPU checks.

```bash
TMPDIR="$PWD/tmp/seed-defect-001-runtime" \
CHERRIES_NAME='MouthOpen frozen seed prediction defect test' \
CHERRIES_TAGS='mouthopen,diagnostic,seed-defect,affine-residual' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/168-diagnose-mouthopen-seed-defect.py \
> tmp/168-mouthopen-seed-defect-001.log 2>&1
```

Working directory is this experiment group. PID868400, process start ticks
4131579, tool session36351 and boot IDfe94509a-c363-40c5-9ffa-8a0cc2ad928e are
recorded in `data/mouthopen-seed-defect-001/job.json`. The state pointer and
heartbeat identify this diagnostic. The full audited preview remains012.

## Four-attempt result and extended diagnostic

Output001 finished with exit 1 at its declared four-attempt limit; no nonlinear
corrector ran. All source hashes/moments remained unchanged. The initial raw
seed confirmed new inversions at656201 and688235. The next corrected proposal
inverted155249 and598977. Attempts2 and3 had100 inversions with no new ones,
but their observed model margins still missed the target. Worst model slacks:

| Seed attempt | Inversions | Minimum observed margin slack |
| --- | ---: | ---: |
| 0 | 102 | -9.801155423229151e-5 |
| 1 | 102 | -4.891724605819967e-6 |
| 2 | 100 | -3.021248096414579e-7 |
| 3 | 100 | -2.3439762636516765e-8 |

The consecutive deficit ratios0.050,0.062,0.078 support additional iterations
of this correction. Output002 therefore repeats the exact frozen experiment
with `--maximum-seed-attempts 12`, preserving the same600-second wall budget,
observed margin tolerance and physical gates. No state is adopted. Its command
changes output/TMPDIR/log suffix001 to002 and adds
`--output-dir data/mouthopen-seed-defect-002 --maximum-seed-attempts 12`.
Name is `MouthOpen seed defect correction continuation test`, tags unchanged.
Exact PID872715, start ticks4145545 and tool session86255 are in job.json.

The1e-10 observed model-slack tolerance is stricter than demonstrated predictor
repeatability: two identical-control1e-5 solves in165003 differed by roughly
6e-10 to2e-8 J at critical cells, despite passing the same1e-7 linear residual
contract. Output002 is left unchanged. If its defect stops contracting,
identical-control seed repeats at the limiting state should establish the
local numerical variation before selecting a conservative QP target buffer or
explicitly tighter linear residuals. Physical force/contact/inversion gates
must remain unchanged.

## Extended result: observed model margin reaches numerical variation

Output002 and Cherries finished with exit1 at12 seed attempts. No nonlinear
corrector ran. Attempts2 through11 all had100 retained inversions and no newly
inverted cells; CCD passed. The deficit briefly reached3.8755611e-10, then
varied between roughly3.6e-9 and3.5e-8. The final seed's smallest previously
positive J was9.887455854828645e-7, and its observed affine minimum was
9.992240342606124e-7. Both are positive, but they did not satisfy the separate
model target1e-6 within1e-10. Source hashes remained unchanged; no state was
adopted. This diagnostic failure does not establish physical infeasibility.

The next frozen diagnostic will measure repeated predictor variation at these
same final controls, then correct the original saved attempt11 seed. It will
explicitly record that the stricter model-margin check failed, while requiring
fresh CCD, original seed inversion gates, preservation of all previously
positive cells and positive observed affine predictions. The final nonlinear
state still must pass the original force/contact/Armijo/count/volume gates.
This tests the actual physical result before adding another numerical buffer.
