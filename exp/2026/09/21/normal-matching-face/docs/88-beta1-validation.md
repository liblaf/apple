# Beta 1 preflight

The [protocol](80-beta1-protocol.md) changes only beta to 1 in the frozen
numerical runner. Full-face derivative validation completed with exit code
zero after Comet shutdown. The largest relative finite-difference errors were
0.0005127127188725014 (0.0512713%) for the combined position, normal and
activation-smoothing objective and 0.018916842004136302 (1.8916842%) for the
normal-only objective. Both passed the unchanged 2% gate.

The directly tested cases were normal-only and combined-with-smoothness.
The unsmoothed combined derivative is covered by subtracting the independently
validated direct q-only regularizer contribution, as in the
[previous validation](58-strong-normal-validation.md). R does not enter
equilibrium or the displacement derivative, so this does not alter the
implicit adjoint contribution. This is a composition argument, not a claim
of a separate full-face unsmoothed finite-difference run.

Both one-update smoke branches completed with zero inversions and process
exit code zero. They began with zero controls, displacement, Adam moments
and adjoint initial guess. At neutral, L2 and the normal contribution are
each 8.656092875221388, giving total objective 17.312185750442776. After one
update, position RMS is about 5.02104 mm and normal RMS about 8.493915 degrees
for both branches. The independent CPU audit must pass before the main fits.

The independent CPU smoke audit subsequently passed and exited zero before
the main fits launched. Both branches have two accepted solve receipts and
completed update 1 with no failure. Saved controls match the first Adam update
within 4.163336342344337e-17 off and 2.7755575615628914e-17 on. All 97 numerical
source records and three fixture files hash-match. The main preflight manifest
records nine protocol/audit/derivative inputs by absolute path and SHA256.

- [Derivative receipt](../data/81-beta1-validation/gradient-validation.json)
- [Smoke outputs](../data/83-beta1-smoke)
- [Independent CPU smoke audit](../data/84-beta1-verification/checks.json)
- [Derivative Comet run](https://www.comet.com/liblaf/apple/65f640c85c7a4ae9a247711a00d8d997)
- Preserved logs: `logs/81-beta1-validation.log`, `logs/83-beta1-smoke.log`.

Exact derivative and smoke commands are in the protocol. Run from
`exp/2026/09/21/normal-matching-face` using the repository Python interpreter.

The CPU smoke audit command was:

```bash
DEBUG=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CHERRIES_NAME=beta1-smoke-verification CHERRIES_TAGS=normal-matching-face,beta1,smoke,verification,cpu uv run python src/85-verify-beta.py --comparison-dir 83-beta1-smoke --output 84-beta1-verification
```
