# Smile fit with stress smoothness disabled

## Purpose and controlled comparison

Test whether the strong spatial-stress regularizer explains the small motion in the completed 20-update Smile fit. Run one new hybrid PNCG → Newton-CG arm from the same frozen neutral, changing only the smoothness coefficient from `5.066584049455902` to `0`.

Keep projected Adam at learning rate `0.3`, 20 full outer updates, no outer rejection/backtracking, zero magnitude and jaw penalties, active stress with the existing PSD bounds, the same jaw hinge bounds, complete bone/eyeball contact, forward absolute force tolerance `1e-8`, adjoint relative tolerance `1e-7`, Newton admission floor `1e-7`, and eight IPC/OMP threads. The forward wall cap remains 1,200 seconds per evaluation. A failed physical solve stops visibly; there is no smaller-step retry.

The reference is the completed hybrid arm in `data/smile-fit-adam03-unconditional-004`. Its final RMS is `4.920537534465182 mm`, from `5.1373308827337585 mm`, with skin motion `0.48848094165899786 mm` weighted RMS from the actual initial equilibrium. Its measured inverse-update time is `781.4311027955264 s`.

The new run uses the same Paratera RTX 4090, after all earlier comparison jobs finished. No auxiliary inverse run is launched. Compare target error, motion, raw stress roughness, physical validity and update history. The objectives differ, so lower total objective alone is not evidence of improved fitting.

## Command

From `${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/exp/2026/09/22/solver-performance` on `paratera-4090`:

```bash
env DEBUG=1 CUDA_VISIBLE_DEVICES=0 \
CHERRIES_NAME='Smile Adam 0.3 zero stress smoothness' \
CHERRIES_TAGS=smile,inverse,active-stress,contact,full-adam,zero-smoothness \
OMP_NUM_THREADS=8 \
${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python src/20-fit-smile.py \
  --mode fit --method hybrid \
  --shared-dir data/smile-shared-003 \
  --output-dir data/smile-fit-adam03-no-smoothness-005 \
  --origin-metadata data/smile-origin-metadata.json \
  --maximum-iterations 20 --learning-rate 0.3 \
  --smoothness-weight 0 --magnitude-weight 0 --jaw-weight 0 \
  --outer-step-policy full_adam --trial-prescreen false \
  --forward-atol 1e-8 --adjoint-rtol 1e-7 \
  --newton-switch-atol 1e-7 --forward-wall-seconds 1200 \
  > data/smile-fit-adam03-no-smoothness-005.stdout.log 2>&1
```

## Results

The diagnostic stopped on its twentieth proposal with `ForwardConvergenceError: candidate has inverted tetrahedra`. It preserved **19 valid full Adam updates**, all with fraction 1.0, without an outer rejection or smaller-step retry. It did not complete the requested 20-update budget and did not converge. The last valid state is the result visualized here.

| Measurement | Smoothness 5.066584 | Smoothness 0 |
| --- | ---: | ---: |
| Stress smoothness weight | 5.066584 | 0 |
| Saved endpoint update | 20 | 19 |
| Target RMS at saved endpoint | 4.920538 mm | 3.558208 mm |
| Target RMS at matched update 19 | 4.921526 mm | 3.558208 mm |
| Skin motion from initial equilibrium | 0.488481 mm | 2.379874 mm |
| Neighbor stress RMS (normalized) | 0.139697 | 1.873849 |
| Active stress RMS | 7.621 kPa | 64.378 kPa |
| Recorded valid-update time | 13.02 min (20) | 50.23 min (19) |
| Minimum det(F), shown state | 0.346963 | 0.066803 |
| Inversions, shown state | 0 | 0 |
| Terminal outcome | 20-update budget reached | Inversion on proposal 20 |
| Inverse converged | False | False |

At matched update 19, target RMS is **27.70% lower** with smoothing disabled. Both runs began at exactly `5.1373308827337585 mm`. The baseline has saved geometry only at updates 5, 10, 15 and 20, so the rendered geometry comparison explicitly uses baseline 20 versus zero 19. No update 19 baseline displacement was reconstructed.

## Interpretation and physical limits

The earlier regularized endpoint had a smoothness gradient about 429 times larger than the positional gradient in coordinate RMS. Removing that term produces more actual skin motion and lowers positional error, supporting the regularizer diagnosis for this finite budget. Those raw gradient norms do not decompose Adam's preconditioned update. This experiment does not establish an optimal smoothness weight or a converged Smile fit.

At the shown endpoints, normalized neighbor stress variation rises **13.41 times**, and active stress RMS rises from 7.62 to 64.38 kPa. Neighbor stress roughness is not a skin-wrinkle metric. Only 850 of 864,705 added-stress eigenvalues are at the upper cap (about 0.098%); the global cap is not broadly saturated. The jaw remains at its 0-degree lower bound.

The last valid zero-smoothing state has force `9.244499162224907e-9 < 1e-8`, zero inverted tetrahedra, minimum determinant `0.06680317116795102`, and valid contact with minimum active gap `1.6001785809501429e-6 m`. These metrics belong to update 19. They must not be mistaken for the failed candidate's metrics. The raw failure JSON retains the prior valid state and the failure reason. The traceback separately records candidate RMS 3.5181882723 mm, force 7.786479892e-9, numerically valid contact, and exactly one inverted tetrahedron. Candidate minimum det(F) was not preserved. These candidate values are log-extracted evidence in the verification receipt, not a saved valid fit. A force criterion and a bone/eye contact check do not themselves constrain every interior tetrahedron's orientation.

Recorded valid-update time is 3013.776 seconds (2583.237 forward, 406.633 adjoint), excluding failed proposal 20. Baseline time is 781.431 seconds for 20 updates. Runs used the same RTX 4090 sequentially. This is a trajectory-cost comparison, not a repeated solver benchmark. The first zero-smoothing update alone took about 508 seconds, despite a nominally identical first proposal; not all timing variation can be attributed to changing regularization. Setup, some checkpoint overhead, transfer and rendering are excluded.

## Verification, outputs and reproducibility

Local copies verify the saved Adam learning rate, all 19 full-step trial records, zero weighted smoothness/magnitude/jaw terms throughout the trace, exact initial-displacement equality, frozen-input equality, and unchanged scientific solver settings apart from the explicit smoothness override. The runtime solver and parent fitter were frozen. The invocation ran under Cherries with `DEBUG=1`; local logs were retained, remote Comet logging was disabled, and the process exited with the physical failure. No further numerical retry was launched.

- [Terminal diagnostic](../data/smile-fit-adam03-no-smoothness-005/terminal-diagnostic.json)
- [Copy and physical verification](../data/smile-fit-adam03-no-smoothness-005/local-copy-verification.json)
- [Raw failure receipt](../data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/failure.json)
- [Visual metrics and provenance](../data/smile-no-smoothness-visuals-001/summary.json)
- [Front geometry](../data/smile-no-smoothness-visuals-001/smile-smoothness-clay-front.png)
- [Side geometry](../data/smile-no-smoothness-visuals-001/smile-smoothness-clay-side.png)
- [Motion](../data/smile-no-smoothness-visuals-001/smile-smoothness-motion-front.png)
- [Convergence and roughness](../data/smile-no-smoothness-visuals-001/smile-smoothness-convergence-roughness.png)

The new saved-state SHA256 is `8233290e5be520d852d501b1112218ef0b4ca437c986aa84d9f94561f80bacb4`. The original shared-neutral artifact SHA256 is `582381c8f32782becf389325086e79086c7daeab50bcf84bd2f1938d57cb3b38`. Harness 20 SHA256: `fd461dea71df36539903263fbc02174f2b1417b9a80184a001430d5ce2b858e2`; parent fitter 93: `1418e6aed9aeb95a0e74e4b7b9baa5b666df365309d0fff3932af46c45101cf1`; accelerated solvers: `ce09ea6472b47834dd639f8c5fd3f6617b25dc05e089a1f5971d85467d4bf8c7`.
