# Adaptive PNCG cold-start diagnostic

Seven Newton corrections each reduced force immediately, but this cold adaptive trajectory ended at 2.401e-05, above the Newton-only endpoint 7.252e-08. The adaptive arm stopped at an Armijo exhaustion and neither compared arm converged. No matched-convergence speedup is reported because the adaptive run did not converge.

The exact switch protocol was: Two consecutive 20-accepted-PNCG-update windows each reduce segment-best force by less than 10%; one safeguarded Newton correction; fresh PNCG optimizer and window history; repeat.

First correction: PNCG step 60, force 2.692e-05 to 1.590e-05, measured 3.649s. Last correction: PNCG step 620, force 4.025e-05 to 1.603e-05, measured 3.453s. All 7 corrections reduced force immediately; their measured total was 16.459s (7.7% of full forward wall time).

The plots use accepted-state measurements from the adaptive controller. Their elapsed seconds start after boundary setup, so they are distinct from full forward wall time. Red diamonds mark accepted Newton corrections. The force panel contains two standalone historical endpoint markers read from their raw receipts; they are not time curves. Historical mechanical energy is intentionally absent because those runs do not provide exact accepted-state timestamps.

## Historical baseline evidence

| Method | Time (s) | PNCG | Newton | Last force | Stop reason |
| --- | --- | --- | --- | --- | --- |
| newton_diag | 204.12 | 0 | 100 | 7.252e-08 | Newton iteration budget exhausted |
| hybrid_diag | 600.12 | 2289 | 0 | 1.662e-05 | declared forward wall budget exhausted |
| adaptive_diag pilot 002 | 32.59 | 90 | 0 | 3.100e-05 | Armijo search exhausted |
| adaptive_diag | 213.60 | 662 | 7 | 2.401e-05 | Armijo search exhausted |

## Preserved pilot 002

Pilot 002 stopped after `90` accepted PNCG updates and `0` Newton corrections in `32.589` seconds. Its last accepted force was `3.099981e-05` and its best observed accepted force was `6.447804e-06`. It exhausted Armijo before completing the 100-update window, so the two-poor-window trigger could not fire. The result is preserved as a failed trial; it is not overwritten or used as a speedup baseline.

## Source and input equivalence

The adaptive protocol records physical-source changes as `[]` and binds the checkpoint, neutral checkpoint, input hashes, and baseline protocol in its evidence. Best force, final recorded energy, HVP counts, and full traces remain in [evidence.json](../data/adaptive-pncg-report-003/evidence.json).

## Recorded command

```bash
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=cold-smile-adaptive-pncg-window20 CHERRIES_TAGS=smile,performance,cold-forward,pncg,newton,adaptive ${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python -u src/47-test-adaptive-pncg.py --checkpoint data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt --neutral-checkpoint data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt --window-steps 20 --output-dir data/adaptive-pncg-cold-003
```

The working directory was `${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/exp/2026/09/22/solver-performance`. This was a local Cherries run; `DEBUG=1` disables Comet in its profile, and this report has no Comet upload. Reproduction must use a new output directory.

## Timing interpretation

The controller trace includes accepted-state diagnostics and explicit Newton post-update energy evaluation. JIT and fixed-model operator prewarm are excluded. It starts after boundary setup, whereas the historical endpoint markers use full forward wall time. The candidate has dense JSON diagnostics; timed historical arms were not rerun. The plot therefore supports trajectory diagnosis, not directly comparable wall-time speedups; it does not establish a matched-convergence speedup when an arm fails.
