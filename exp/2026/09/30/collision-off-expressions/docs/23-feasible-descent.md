# Finite feasible descent after equilibrium refinement

`73-probe-feasible-descent.py` completed with `feasible_descent_witness` and exit 0. The pose-only alpha 0.04 trial provides a fully re-equilibrated lower-loss endpoint that passes both force levels, the original geometry policy, and the empirical objective and displacement screens. This is finite-step evidence; it does not establish a local derivative, first-order stationarity, an optimum, or inverse convergence.

The source is frozen at SHA-256 `b953bf15f1f0d6b416df067e4c3f602d26ab8ed9ffa0add6267da94a1adfad1d`. The final summary SHA is `fb88388fc5783d9bb9a45344f033a38107beb47d0159f0b30bd82826d63faa8b`. Inputs and solver policy are unchanged from the protocol described in `10-collision-off-expressions.md`.

| Direction | Alpha | Loss change at 1e-12 | Loss change at 1e-13 | Result |
| --- | ---: | ---: | ---: | --- |
| Joint | 0.04 | -0.00346994081 | unavailable | Refinement unresolved |
| Joint | 0.01 | -0.000872208862 | unavailable | Refinement unresolved |
| Pose only | 0.04 | -0.00345899669 | -0.00345920923 | Paired finite descent; response screen passes |
| Pose only | 0.01 | -0.000869476107 | -0.000869650138 | Paired loss decrease; response screen unresolved |

Force values and tolerances in this report are solver units; multiply by 1e6 for newtons. The pose-only fine forces are 9.876420729487481e-14 and 9.919540265779821e-14. Both pose trials retain 88 inverted tetrahedra, rest-volume fraction 3.003795595738308e-5, and introduce no new retained inversions. Root verification confirmed exact unchanged activation and active IDs in the pose archives. The unshifted adjoint's directly evaluated relative residual is 9.77866555759653e-8.

For alpha 0.04, baseline and trial refinement loss drifts are 8.83960988462551e-7 and 6.714288515174971e-7. Their sum times ten gives the empirical screen 1.555389839980048e-5; the loss reduction is about 222 times larger. The final maximum coordinate response is 84.0514 micrometres, versus baseline and trial refinement changes of 1.1462 and 2.7190 micrometres. The correction/seed ratio is 0.7722. Alpha 0.01 clears the loss screen but its 21.0148-micrometre response does not exceed ten times the combined coordinate refinement. The finite-scale slope disagreement is about 0.56%; this is not a formal error bound or local derivative validation.

Both joint refinements failed Newton regularization/Armijo checks. Their saved forces, 1.3745761120739674e-13 and 1.0379571106450148e-13, exceed the prescribed 1e-13 threshold. Their failed states remain diagnostic evidence, not successful fine endpoints. This observed solver failure does not establish a mathematical precision floor or model infeasibility.

The verified probe bundle contains 392 files, SHA-256 `ead3ead2eac039b2a531c8ee939d34928fc0d13a1a6131a92a90838c0cf46363`; the completed reservation bundle has seven files, SHA-256 `d7201e1cfb752a8b0d36902b87dbefe76514e005bd7cd8eb5b9ce61e7a907847`. Machine-readable evidence is `data/feasible-descent-001-analysis.json`. [Normal-profile Comet run](https://www.comet.com/liblaf/collision-off-expressions/4319f9e3ad4b47bca15bb180880a1190); Git commits were disabled.

The controller resumed the original queue, which independently audited Smile-003 and continued to MouthOpen-004. Smile-003 has 75 total updates, RMS 4.882018171720653 mm, force 0.009999999999999433 N, 95 inversions and rest-volume fraction 9.780016615642138e-6. Its 476-file bundle is verified (`20119d7da541159ff8a6d22c16fd8171a63757a7bfd659ce765929c9552b7731`). The 76-state review (private preview omitted) includes the full original boundary, cranium, mandible and eyes; its front image was inspected. [Render Comet run](https://www.comet.com/liblaf/apple/43d6a8a33f874281b70377c3f65d215b). Its finite budget and unchanged fit do not establish inverse convergence.

The next planned experiment is a bounded optimization pilot from the original MouthOpen-001 parameters and optimizer checkpoint, using the certified same-parameter 1e-12 displacement as a startup seed. It preserves moments, counters and convergence history while explicitly recording the 2.66 mm equilibrium refinement and objective change at zero optimizer updates. The witness's changed pose will not be substituted into that checkpoint. The pilot needs independent source review, a fresh serial reservation, direct force/geometry and unshifted-adjoint checks, an endpoint audit, and lineage rendering that retains the refinement event. No pilot result exists yet.
