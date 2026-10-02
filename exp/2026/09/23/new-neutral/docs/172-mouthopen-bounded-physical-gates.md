# Correct the bounded joint seed under original physical gates

Diagnostic171 produced a seed with CCD fraction 1, no intersections, 100 retained
inversions and inverted rest-volume fraction 1.2741237208770825e-5. Two cells
healed and two newly inverted. It stopped before a corrector because an additional
old-positive/affine forecast screen failed. The original physical policy allows
up to 100 inverted cells and volume fraction 1e-4, with no minimum-J floor or
fixed inversion identities. The seed therefore passed original seed admission.

This diagnostic binds and replays that exact saved171 seed and target controls.
It also binds validated170 gradients/metric/projection and audited012 source
controls, displacement, moments and counters268. Fresh checks verify target
controls, fixed values, every retained determinant, old-to-seed rigid-arc CCD,
inversion count and volume. The extra forecast screen remains explicitly failed
in the receipt; it is not enforced as a physical condition.

One collision-on nonlinear corrector then starts from the original saved seed.
The final state must satisfy internal force1e-9, original physical force1e-8,
contact, actual cached Armijo slope, count<=100 and volume fraction<=1e-4.
Raw corrected states, signs and failures are saved before acceptance. No new
predictor, QP or optimizer update is performed, and no candidate is adopted.
The numerical budget after rebuilding physics is600 seconds.

## Launch

CPU checks verified all bound source/control/seed hashes and exact target
reconstruction. Root reviewed the code changes; Ruff and compilation passed.

```bash
TMPDIR="$PWD/tmp/bounded-physical-002-runtime" \
CHERRIES_NAME='MouthOpen bounded seed original physical gates' \
CHERRIES_TAGS='mouthopen,diagnostic,bounded-joint,original-physical-gates' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/172-test-mouthopen-bounded-physical-gates.py \
> tmp/172-mouthopen-bounded-physical-002.log 2>&1
```

Exact PID 916272, start ticks 4398327, tool session 12155; pointer and job
receipt record the command, boot identity and project TMPDIR.

## Result

The saved seed PASSED the actual nonlinear test. Numerical process and Cherries
shutdown exited 0. It used 140 PNCG and 5 Newton steps.
RMS decreased from 1.9802796612356741 mm to 1.97834570280376 mm.
Raw force was 8.842308565484633e-10, or 0.0008842308565484633 N.
The result had 100 retained inversions and inverted rest-volume fraction
1.252789283985614e-05; contact and Armijo passed.
The source hashes, controls and optimizer moments remained intact.

This confirms the joint bounded proposal can pass the original physical policy
even when the additional inversion-identity forecast screen fails. The candidate
is diagnostic evidence only; it is not adopted as an inverse checkpoint or
claimed converged. A new additive inverse run will start at exact audited012
controls and optimizer counters268 and use the validated proposal method.

Comet: <https://www.comet.com/liblaf/apple/cd6a1814caaa4a3f91a1a0cfb671f420>
