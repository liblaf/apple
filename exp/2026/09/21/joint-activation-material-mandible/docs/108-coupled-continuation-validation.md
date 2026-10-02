# Full-contact jaw and active-stress continuation

**The 1° jaw plus 12.3288 Pa isotropic muscle-stress proposal passed.** Adaptive coupled prediction and strict PNCG correction reached the exact requested parameters in four internal substeps while retaining the full skull and both fixed eyes. The final equilibrium and implicit adjoint passed their unchanged tolerances.

This resolves the earlier initializer's artificial restriction: it moved the jaw before letting tissue respond, causing CCD to reject a 1° proposal at a 0.039858 fraction. The new solver advances tissue and jaw together, checks their combined motion, and handles necessary substeps internally. Real geometric obstructions can still reject a proposal; this is not a general feasibility guarantee.

## Measured result

| Cumulative opening | Strict PNCG iterations | Exact force / threshold | Minimum active gap |
| --- | ---: | ---: | ---: |
| 0.123469° | 348 | 0.9717 | 17.024 µm |
| 0.422195° | 353 | 0.8203 | 16.923 µm |
| 0.688176° | 401 | 0.9225 | 16.833 µm |
| 1° | 451 | 0.9795 | 13.707 µm |

The four stages used 1,553 PNCG iterations in total. No final soft–rigid intersections were present. The collision buffer remained 10 nm, Poisson's ratio remained 0.49, and the original material and baseline-stress fields were unchanged.

The final force norm was 1.48809e-10 in model units, below 1.51920e-10. The final implicit adjoint relative residual was 9.09693e-8, below the 1.05e-7 acceptance gate. MouthOpen RMS was 6.97633 mm versus approximately 7.63066 mm at the contact-on neutral. This prescribed-parameter forward benchmark is **not an inverse-fit convergence result** and is not a new finite-difference gradient validation.

| Work | Time |
| --- | ---: |
| Continuation, including rejected predictors and internal PNCG | 268.84 s |
| Final strict PNCG | 55.93 s |
| Final adjoint | 30.38 s |
| Total above | **355.15 s / 5.92 min** |

Model setup and logging/upload shutdown are excluded. The GPU was shared with another experiment; these are observed timings, not a controlled speedup benchmark. Seven predictor attempts included three geometric rejections, all recomputed before acceptance.

## Reproduce and inspect

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```sh
CHERRIES_NAME='Full face coupled continuation with strict PNCG substeps' CHERRIES_TAGS='joint-inverse,contact,coupled-predictor,pncg' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python src/108-benchmark-coupled-continuation.py --output-dir data/coupled-continuation-benchmark-002
```

The run completed successfully through Cherries shutdown. [Comet experiment d9fc723cf5bd43a1bcb998b6a55ee8ce](https://www.comet.com/liblaf/apple/d9fc723cf5bd43a1bcb998b6a55ee8ce) records the run. `data/coupled-continuation-benchmark-002/summary.json` contains every predictor and corrector receipt; `predicted.pt` and `corrected.pt` separate the feasible initialization from the final equilibrium. `provenance.json` and `sources/` bind the implementation used. The terminal log is `logs/108-coupled-continuation-benchmark-002-terminal.log`.

CPU regressions cover tangent algebra, owned-state restoration, scaled residual correction, retry recomputation, exact target completion, strict internal correction and failure rollback. See `data/coupled-predictor-check-004`, `data/coupled-continuation-check-003`, and the three `data/coupled-predictor-{projected-descent-003,pose-first-002,pose-collision-003}` receipts. See [method design and single-predictor evidence](106-coupled-contact-predictor.md) for the equations and primary literature.

## Expression fitting

Fresh run `data/expression-fitting-008` uses this method with collision enabled during both stages. It restarts from the approved contact-valid neutral, fits each expression's one-DoF jaw first, then jointly fits per-tetrahedron active stress and jaw pose. All 36 expressions remain sequential. The collision-off result from run 007 is preserved and is not used as a contact-valid seed. Run 008's own status and accepted checkpoints determine fitting progress; the benchmark above does not substitute for them.

The first real MouthOpen outer update was accepted at the full requested **1°** (outer line-search fraction 1). Five internal coupled substeps handled contact. RMS decreased from 7.63066 to 6.97636 mm, about 8.6%, and the normalized objective decreased from approximately 1 to 0.835876. The accepted shape had zero inverted tetrahedra. This is one accepted pose-only update, not pose or joint convergence. The run continued to the next pose step. [Run 008 on Comet](https://www.comet.com/liblaf/apple/ba63e715544045cf9b582db27e093e05); live tailnet review (private preview omitted).
