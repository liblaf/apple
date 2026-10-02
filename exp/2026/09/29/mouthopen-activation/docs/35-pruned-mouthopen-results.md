# MouthOpen with fully fixed tetrahedra removed

Removing all 2,249 fully fixed tetrahedra cleared the original prescribed-cell obstruction, but this first forward continuation stopped at **4.2679% of the estimated MouthOpen jaw motion**. A retained fat tetrahedron with three fixed corners and one free corner approached collapse. The last accepted state passed the declared force, positive-volume and FEM boundary intersection checks. It is not a completed MouthOpen solution, and activation fitting was not started.

Figure correction: the historical image below has incorrect jaw-overlay scaling for the partial checkpoint. Its skin geometry and numerical results are unchanged. The [new renderer and full-pose comparison](49-contact-off-forward-results.md) apply the prepared target pose and saved endpoint pose directly. The original image is retained for provenance.

![Historical comparison with the jaw-overlay limitation noted above](../data/40-pruned-comparison-002/mouthopen-pruned-comparison-preview.png)

[Full-resolution comparison](../data/40-pruned-comparison-002/mouthopen-pruned-comparison.png)

## What changed

The [fixture builder](../src/30-build-pruned-fixture.py) removed every original tetrahedron whose four vertices satisfy the original `IsFixed` mask. It did not select cells based on their behavior under a particular expression. All free vertices, the 15,299 fitting-surface vertices, the target displacements and surviving rest coordinates were retained exactly. The original fixture and previous experiment artifacts are preserved.

| Quantity | Original | Pruned |
| --- | ---: | ---: |
| Tetrahedra | 1,146,517 | 1,144,268 |
| Vertices | 228,660 | 227,900 |
| Fixed vertices | 27,036 | 26,276 |
| Free vertices | 201,624 | 201,624 |
| Active tetrahedra | 288,235 | 288,172 |
| Same-region active adjacency edges | 501,409 | 501,313 |
| Fitting-surface vertices | 15,299 | 15,299 |

The removed volume is 213.99 mm³, or 0.02719% of the original tissue volume. There are 103 surviving activation control regions; their IDs remain contiguous. Active volume weights and adjacency are rebuilt on the new domain. The runtime confirms that all three displacement components are fixed exactly where `IsFixed` says they are; jaw motion acts only on its intersection with `Mandible`. All lip vertices remain free. Compaction leaves 5,989 prescribed jaw vertices.

## Forward trial and observed stopping point

The [protocol](30-pruned-mouthopen-protocol.md) retains the historical no-skin, contact-force-free bulk mechanics and sets activation to zero. It uses the fresh chin-based rigid estimate from the original neutral, not a saved modified-neutral pose or smile activation. Each pose increment starts from a harmonic carry and is relaxed with safeguarded Newton-CG. A relative-deformation bound preserves positive volume throughout each linear update; continuous collision detection checks the complete extracted FEM boundary.

The run made 26 pose attempts. Nine were accepted after force convergence. The other 17 were rejected by the conservative volume bound on the proposed initialization. There were no solver-rejected pose attempts and no accepted inverted states. After the last accepted pose, halving the rejected increment would have reduced it below the declared minimum fraction of `1e-5`; the run stopped under that rule rather than forcing the jaw to the target.

| Metric | Neutral | Last accepted state |
| --- | ---: | ---: |
| Requested-pose fraction | 0% | 4.26795% |
| Minimum determinant, all retained cells | 1.0 | 0.000223820 |
| Inverted tetrahedra | 0 | 0 |
| Free-force norm, code units | 0 | 1.77179e-11 |
| Force acceptance threshold | 1e-10 | 1e-10 |
| Position RMS against full MouthOpen target | 7.60096 mm | 7.22773 mm |
| Detected FEM boundary intersections | No | No |

The accepted pose has 0.406894° rotation and 0.276551 mm pivot translation. Against the full target, the 27-vertex chin RMS is 22.3977 mm and fitting-triangle normal RMS is 7.10596°, weighted by reference triangle area.

The continuing phase took 52.1 seconds, excluding model and harmonic-weight setup. The process exited successfully after Cherries shutdown; its scientific status is `blocked_at_minimum_pose_step`, with `valid_forward=false` because the full prescribed pose was not reached.

The smallest determinant corresponds to new cell 670,920 / original cell 672,256. Its fixed corners belong to two cranium vertices and one mandible vertex; its fourth vertex is free and unlabelled. It has `FatFraction=1` and is inactive. It shares two vertices, but no entire face, with the deleted cells. The next-smallest determinant is about 0.225, so the extreme flattening is localized. An independent CPU reconstruction confirms that this same cell sets the last rejected carry's bound: 0.6490524271, matching the solver receipt. See the [independent endpoint audit](../data/36-pruned-audit/audit.json) for exact cell and boundary diagnostics.

## Interpretation and limits

Deletion removes elements with no free displacement degrees of freedom, but neighboring elements still connect independently moving bone attachments. This result shows that deletion alone did not let this continuation reach the target. It does **not** prove that every possible continuation path or other solver must fail. In particular, the terminating criterion is a conservative bound on the harmonic initialization, not an observed inversion at the final accepted state or a proof that no positive-volume equilibrium exists beyond it.

The extracted boundary has 63,282 vertices, 126,648 triangles and 28 nonmanifold edges. The neutral and accepted configurations have no detected boundary self-intersections. These checks add no contact force, do not check separate bone or eye obstacle meshes, and do not establish inside/outside containment or anatomical validity. The chin-derived pose is a geometric estimate from a transferred blendshape, not measured mandible motion.

No activation field was optimized, so there is no final activation smoothness/L2 gradient ratio to report for this trial. The next diagnostic should focus on the retained boundary-transition cell and the pose/attachment geometry rather than treating this partial state as a fitted MouthOpen expression.

## Artifacts and verification

- [Pruned mesh receipt](../data/30-pruned-fixture/summary.json), [volume](../data/30-pruned-fixture/volume.vtu), [skin](../data/30-pruned-fixture/skin.vtp), and [original-to-pruned mappings](../data/30-pruned-fixture/mapping.npz).
- [Forward summary](../data/35-forward-pruned-002/summary.json), [final accepted checkpoint](../data/35-forward-pruned-002/final.npz), [harmonic weights](../data/35-forward-pruned-002/harmonic-weight.npz), and [source manifest](../data/35-forward-pruned-002/source-manifest.json).
- [Independent endpoint audit](../data/36-pruned-audit/audit.json) and [rendering manifest](../data/40-pruned-comparison-002/manifest.json).
- [Fixture log](../tmp/30-build-pruned-fixture.log) and [forward log](../tmp/35-forward-pruned-mouthopen-002.log).

The root verification recomputed all five forward input hashes, all 100 archived source file hashes (101 module records), and every accepted checkpoint hash. Every saved accepted state has positive minimum determinant, force at or below `1e-10`, and a passing boundary intersection result. Runtime boundary assertions and the fixture's read-back checks passed. The comparison preview was visually inspected for consistent camera, target/state distinction and explicit incomplete-pose labeling. Targeted Ruff checks passed.

The first launch, preserved under `data/35-forward-pruned`, stopped during source archival before any jaw increment because a generated external module named `_ops.py` was mistaken for a physical project file. Source discovery was restricted to existing project source files, and the numerical trial ran in a separate `35-forward-pruned-002` folder. This did not change the material, mesh, pose or solver settings. No Git commits or pushes were made.

## Reproduction and Cherries records

Working directory: `exp/2026/09/29/mouthopen-activation`. The commands below reproduce the recorded runs only when their output folders are absent; use fresh output paths for additional trials.

```bash
CHERRIES_NAME='MouthOpen pruned mesh zero activation jaw continuation 002' CHERRIES_TAGS='mouthopen,mandible,pruned,forward,zero-activation' PYTHONUNBUFFERED=1 .venv/bin/python src/35-forward-pruned-mouthopen.py --output 35-forward-pruned-002
```

The Cherries profile records metadata and local sources with automatic Git commits disabled. Completed Comet runs: [fixture preparation](https://www.comet.com/liblaf/apple/64b0e906697443a6af4fac9f333b5e13), [forward continuation](https://www.comet.com/liblaf/apple/20cc94c61ca94553aaf7f3f75e6895a9), [endpoint audit](https://www.comet.com/liblaf/apple/2d3ffe953c454543b315411073361976), and [final comparison](https://www.comet.com/liblaf/apple/3b3d83dc623940f6a5589e264a9fbd6f). The saved runtime records Python, Torch, CUDA and RTX 4090 device details. No broader mechanical or activation-fit success is implied by process completion.

## Recorded forward summary

The block below is the recorded Comet summary with logging prefixes removed. The Git SHA identifies the checkout; pre-existing uncommitted work was present.

```text
---------------------------------------------------------------------------------------
Comet.ml Experiment Summary
---------------------------------------------------------------------------------------
  Data:
    display_summary_level : 1
    name                  : MouthOpen pruned mesh zero activation jaw continuation 002
    url                   : https://www.comet.com/liblaf/apple/20cc94c61ca94553aaf7f3f75e6895a9
  Metrics:
    completed_pose_fraction : 0.042679455876350414
    fit_rms_mm              : 7.227726507307236
    force_norm              : 1.7717912673342407e-11
    inverted_cells          : 0.0
    minimum_J               : 0.00022382022857637499
    valid_forward           : 0.0
  Others:
    Name                : MouthOpen pruned mesh zero activation jaw continuation 002
    cherries/cmd        : .venv/bin/python src/35-forward-pruned-mouthopen.py --output 35-forward-pruned-002
    cherries/comet/url  : https://www.comet.com/liblaf/apple/20cc94c61ca94553aaf7f3f75e6895a9
    cherries/end_time   : 2026-09-29 02:09:09.534017+08:00
    cherries/entrypoint : exp/2026/09/29/mouthopen-activation/src/35-forward-pruned-mouthopen.py
    cherries/exp_dir    : exp/2026/09/29/mouthopen-activation
    cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
    cherries/start_time : 2026-09-29 02:08:05.812788+08:00
  Parameters:
    fixture           : exp/2026/09/29/mouthopen-activation/data/30-pruned-fixture
    force_atol        : 1e-10
    initial_pose_step : 0.025
    linear_max_steps  : 3000
    linear_rtol       : 0.001
    max_newton_steps  : 100
    maximum_pose_step : 0.1
    minimum_pose_step : 1e-05
    volume_safety     : 0.8
    wall_seconds      : 1200.0
  Uploads:
    filename     : 1
    git metadata : 1
    source_code  : 2 (17.71 KB)

```
