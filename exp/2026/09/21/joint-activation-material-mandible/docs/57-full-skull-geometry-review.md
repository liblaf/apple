# Complete skull: geometry and initialization

The required collider is the complete registered melon source cranium and
mandible: **35,162 + 18,948 triangles**, with original source coordinates and
triangle IDs retained. The partial-FEM optimization was stopped at segment 002,
accepted update 111. Its checkpoint remains unchanged and is legacy evidence.

## Measurements

All 34,245 pure-soft boundary vertices and 65,580 triangles were checked against
the unchanged source bones. Distances are signed soft-node distances; raw
triangle intersections additionally detect crossings between vertices.

| State | Cranium intersection pairs | Mandible intersection pairs | Minimum nodal gaps, cranium / mandible | Physical det(F) range | Inversions |
| --- | ---: | ---: | --- | --- | ---: |
| Original reference | 1,857 | 766 | -0.07881 / -0.05345 mm | 1 / 1 | 0 |
| Legacy neutral update 111 | 1,419 | 725 | -0.22611 / -0.22029 mm | 0.53855 / 1.75338 | 0 |
| Initialization candidate 001 | 0 | 0 | +0.01000 / +0.01000 mm | -0.38629 / 2.66036 | 13 |
| Repaired initialization 002 | 0 | 0 | +0.01000 / +0.01000 mm | 0.25010 / 1.99990 | 0 |

Candidate 001 is **rejected**. It projected the inner soft boundary outside the
complete source bones and extended the displacement harmonically into the
volume. It moved neither source bone coordinates nor original fixed FEM nodes,
and did not change the FEM reference, targets, or topology. Its maximum nodal
displacement is 0.680 mm. Outer observation-weighted surface RMS is zero because
the modified surface is internal; this does not imply that all soft boundary
nodes are unchanged.

Candidate 002 repairs 183 local nodes while retaining all original fixed and
outer observation coordinates. All 1,601 incident tetrahedra satisfy the declared
volume bounds, and the global checks above pass. The maximum additional repair
is 0.209 mm. Actual IPC evaluation has 6,186 active pairs, positive minimum feature
distance 0.122 micrometres, and zero-increment CCD fraction 1. This admits a
**soft-bone initialization only**; equilibrium and final-launch flags remain false.
The exact initialization is bound by the v2
[admission receipt](../data/full-skull-initialization-candidate-002/admission.json).

The complete cranium and mandible themselves have 128 raw triangle intersection
pairs in the supplied neutral pose. Soft-bone contact excludes bone-bone pairs;
the earlier partial-FEM rigid guard is not evidence of complete-source bone-bone
validity. No new bone-bone validity claim is made.

## Visual evidence and timing

The mobile review (private preview omitted) includes complete front/side
skull renders and the four-state intersection/clearance comparison. All three
PNG assets were visually inspected and their tailnet HTTP responses checked.
The replacement adapter's independent synthetic derivative/CCD checks are
documented in [53](53-full-source-skull-contact.md).

The separate [contact-only timing receipt](../data/full-skull-contact-microbenchmark-invalid-candidate-001/summary.json)
measures the complete collider on candidate 001. It is explicitly an invalid-volume
diagnostic, not a forward/adjoint or production result. One warmup and five CPU
repeats give median candidate rebuild 15.52 ms, energy 0.875 ms, force 4.642 ms,
first Hessian assembly/product 17.85 ms, cached product 1.201 ms, and CCD 11.19 ms.
The minimum active IPC feature distance is 0.122 micrometres, smaller than nodal
clearances. These costs cannot be summed into a full optimizer slowdown without
the actual iteration and contact-state counts.

## Reproduction

Run from `exp/2026/09/21/joint-activation-material-mandible`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CHERRIES_NAME='Complete skull geometry and initialization review' \
CHERRIES_TAGS=joint-inverse,full-skull,geometry,visualization \
uv run --frozen python src/57-review-full-skull-geometry.py \
  --output-dir data/full-skull-geometry-review-003 \
  --repair-summary data/full-skull-initialization-candidate-002/summary.json
```

The normal, noncommitting Cherries run exited successfully:
[Comet 19edd1d4](https://www.comet.com/liblaf/apple/19edd1d484fa46f0bd1eb2bf6ed27a99).
Outputs are the three PNGs, `states.json`, `summary.json`, and archived source
provenance. Review 001 produced the same numerical evidence but failed during
Tk cleanup; reviews 002 and 003 use the noninteractive Agg backend and completed cleanly.

| Artifact | SHA-256 |
| --- | --- |
| Full-source geometry archive | `a706952109c4ad67a72692202412a827bc8d57ef1ccbd393dfa70a4d4190a6aa` |
| Initialization audit 001 summary | `e0b4986cb92f0a27a65f28a86c21780ef995482d37ff5b9a05abe1aedbdb425c` |
| Candidate 001 summary | `bb7d21c74462b82026383efe46b1595cfd48f02369fa8a35f55056387d6b0e3d` |
| Geometry review 002 summary | `753930460d9217170b88964c848624640b06570bdaadf8058a91ea2395d531eb` |
| Repaired initialization 002 summary | `c367abc4b2a81bac26be480b3730ba63595e1e62810315570105f24e046f9db4` |
| Geometry review 003 summary | `58ec21ace025b6a8067824bf89aac94a962350af166abf9946c0c84d7a354945` |

Git HEAD was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; unrelated working-tree
changes were preserved. No converged full-skull equilibrium or final joint
optimization has been produced by these geometry diagnostics.
