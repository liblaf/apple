# No-skin unrestricted L2 fit — Adam learning rate 0.5

Status: **stopped at the user's request after accepted step 15**. The user requested a fresh restart at learning rate 0.05 with no smoothness, because the old trajectory was broken. All checkpoints and evidence from this 0.5 trajectory are retained. See the [fresh L2-only run](41-l2-unrestricted-lr0p05-nosmooth.md).

This run follows the user's requested initial Adam learning rate **0.5**. The original pilot and main fit started from zero active stress and zero displacement with fresh Adam moments. Adam uses betas `(0.9, 0.999)` and epsilon `1e-8`. Finite approximate solves, objective increases, and inverted tetrahedra continue; the inversion count and minimum determinant are diagnostics only. Nonfinite numerical proposals restore controls and moments, halve the learning rate and continue.

The model has no skin energy, fixed jaw and no contact. L2 position error is the only fitting loss. Active stress is unrestricted symmetric Mandel6; smoothness penalizes the full stress tensor on the within-muscle graph. The smoothness coefficient is recalibrated from the new pilot at the requested 0.1 component-gradient ratio, then frozen.

Every Newton-CG step begins with the mean absolute physical Hessian diagonal as its positive shift, skipping zero shift. Subsequent retries multiply the positive shift by 10. The physical implicit adjoint is unshifted. The forward iteration cap is 100 and the PCG cap is 10,000; finite nonconvergence is recorded and allowed during Adam.

The [passed mechanics derivative reference](../data/10-validation-inverse-v2-002/checks.json) predates the user-requested positive-first shift. The new run records all source differences and does not claim that the earlier strict validation certifies convergence of approximate current states. [Preflight details](41-l2-unrestricted-inverse-v2.md).

The preceding learning-rate-0.05 run was interrupted at the user's request after 21 completed attempts, preserving its last checkpoint (4.987366950187247 mm RMS). This fresh run is not a continuation of that checkpoint, and recalibration means the pair is not a controlled learning-rate-only comparison.

## Command

Working directory: `exp/2026/09/21/stress-activation-loss/`.

```sh
CHERRIES_NAME='Smile L2 unrestricted stress Adam lr 0.5 resumed' \
CHERRIES_TAGS='smile,inverse,no-skin,l2,unrestricted6,adam,mean-abs-shift,lr0p5,resume' \
uv run python src/41-fit-l2-unrestricted.py \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --resume-checkpoint 41-l2-unrestricted-inverse-v2-lr0p5-002/l2-symmetric6/optimizer-latest.pt \
  --learning-rate 0.5 \
  --steps 200 \
  --output 41-l2-unrestricted-inverse-v2-lr0p5-003
```

Cherries records locally and to [Comet](https://www.comet.com/liblaf/apple/6d342b0be7ba4ebb8df8f02deb8e48c1); automatic Git commits are disabled. Current sources, checkpoints and receipts are under [data/41-l2-unrestricted-inverse-v2-lr0p5-003](../data/41-l2-unrestricted-inverse-v2-lr0p5-003/). The live console is [41-fit-l2-unrestricted-inverse-v2-lr0p5-003.console.log](../tmp/41-fit-l2-unrestricted-inverse-v2-lr0p5-003.console.log). Before this launch, **90 unit/behavior tests passed**, including finite inversion acceptance, nonfinite rejection, Adam resume, the real CUDA implicit derivative and positive-first-shift checks. The interrupted parent process remains on [Comet](https://www.comet.com/liblaf/apple/8fa792ca997d45008558610726538e38).

## Calibration and restart

The learning-rate-0.5 pilot completed 8 attempts with 7 accepted updates and one unusable-proposal skip. The last accepted state has RMS error 5.007540412025651 mm and minimum determinant 0.11379352273443166. Its forward solve is finite but unconverged. The pilot learning rate ended at 0.25 after the skip; the main fit still starts fresh at the requested 0.5.

An unnecessary additional equilibrium solve during calibration inverted three elements (`min detF=-0.19751793026121475`), so that state was discarded and the main fit did not start in the first process. The completed pilot is retained, with its exact checkpoint and component-gradient norms. Calibration now uses those already recorded values at the same accepted state, requiring no further equilibrium solve. L2 gradient norm: 0.0625342245086076; smoothness gradient norm: 380.7758291128779; coefficient for ratio 0.1: **1.64228450777189e-5**.

The failed calibration process is recorded at [Comet](https://www.comet.com/liblaf/apple/3b48b4f2f4e14b49a45a95bee93c52dc) and [failure receipt](../data/41-l2-unrestricted-inverse-v2-lr0p5/failure.json). The replacement run reuses only the pilot calibration evidence; the fitting controls and Adam state start from zero.

## Results

The first replacement process reached accepted step 7, with RMS error **5.00784910590293 mm** and minimum determinant **0.015547384103617199**. The previous inversion gate rejected attempts 8 through 22 and repeatedly reduced the learning rate. That gate contradicted the user's intended finite continuation policy and has been removed. These rejected attempts and the interrupted process remain in the [parent output](../data/41-l2-unrestricted-inverse-v2-lr0p5-002/).

The corrected run resumes the saved step-7 stress, displacement seed and Adam moments, restores learning rate **0.5**, and retains the frozen smoothness coefficient. It re-evaluates the saved controls under the corrected acceptance policy before taking the next Adam update. The new lineage records both source changes and the parent checkpoint; no pilot is rerun. Results are pending.

The resumed initial evaluation accepted **35 inverted cells** (`min detF=-0.6518900417235577`) at learning rate **0.5** with finite objective and gradient. Its RMS was **4.9776601616251055 mm**. This change from the parent's saved RMS comes from further forward iterations at unchanged controls, before any resumed Adam update; it is not an optimizer improvement measurement.

At the latest inspected checkpoint, step **11**, the corrected run has made four additional Adam updates with **zero skips**, still at learning rate **0.5**. RMS is **4.939700624066449 mm**, minimum determinant is **-4.139111136849377**, and **1,389** cells are inverted. These finite states continue as requested. The saved state remains approximate (`solver_valid=false`). Live checkpoint/summary files supersede this progress snapshot.

Runtime has increased with Newton linear-system difficulty. This runner shares the safeguarded Newton-CG policy but currently uses matrix-free Hessian products, not the recent hybrid GPU free-CSR path. The latter requires additional no-contact and passive `StableNeoHookean` assembly support before it can be used by this fixture. No solver backend was changed during this run.

## Process recovery from step 13

The `003` process disappeared after saving step 13 and completing the next forward solve's 100 Newton iterations. Its log ends without a Python exception or recorded shutdown. No corresponding kernel OOM or segmentation-fault event was found; the tool process session was unavailable. The exit cause is unconfirmed. Its interruption receipt and last checkpoint are preserved.

The last saved RMS was **4.938638579677709 mm**, a **3.09%** decrease from 5.095908027590781 mm. Total objective was **0.049559758698291885**, compared with the zero-stress initial value **0.04940857010416178**: the growing smoothness contribution offsets the positional improvement. All six resumed Adam updates were accepted at 0.5, with no skips. The state remains approximate.

The replacement process uses unchanged numerical source hashes, the same frozen coefficient and saved Adam state. It runs through the remaining budget to global step 200 under the transient user service `apple-smile-fit-20260923.service`, with automatic unit collection on exit and no persistent service installation. The current [Comet record](https://www.comet.com/liblaf/apple/c49dd00c5dbd4317a18ca803c46d1143), [output directory](../data/41-l2-unrestricted-inverse-v2-lr0p5-004/), and [console](../tmp/41-fit-l2-unrestricted-inverse-v2-lr0p5-004.console.log) supersede the earlier progress snapshot. The [tailnet viewer](43-live-progress.md) follows the replacement process and preserves the previous loss-curve segments.

The resumed initial evaluation at unchanged step-13 controls completed in 263 seconds, with RMS **4.938441819345878 mm**, objective **0.049556061052908947**, and 1,774 inverted cells accepted as diagnostics. The updated dashboard was verified to report the new live process and all three ancestry segments. The additional forward iteration changed the displacement slightly; no new Adam update is counted for this initial re-evaluation.

The fit command, run from this experiment directory with `CHERRIES_NAME='Smile L2 unrestricted Adam lr 0.5 checkpoint13 resumed'` and tags `smile,inverse,no-skin,l2,unrestricted6,adam,mean-abs-shift,lr0p5,resume`, is:

```sh
.venv/bin/python src/41-fit-l2-unrestricted.py \
  --mechanics-reference 10-validation-inverse-v2-002 \
  --resume-checkpoint 41-l2-unrestricted-inverse-v2-lr0p5-003/l2-symmetric6/optimizer-latest.pt \
  --learning-rate 0.5 --steps 200 \
  --output 41-l2-unrestricted-inverse-v2-lr0p5-004
```
