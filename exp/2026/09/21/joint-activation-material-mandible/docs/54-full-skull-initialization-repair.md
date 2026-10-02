# Full-skull initialization repair

`54-repair-full-skull-initialization.py` repairs the volume defects in the
collision-free candidate from `52-prepare-full-skull-initialization.py`. It is
an initialization projection. It is not a remesh, an equilibrium solve, or a
final-run admission.

Candidate 001 contained 13 inverted tetrahedra and 57 tetrahedra outside the
declared `0.25 <= det(F) <= 2` range. Ten inverted tetrahedra had only fixed or
soft-boundary vertices, so holding the soft boundary pointwise made repair
impossible. The repair therefore releases the 69 affected inner soft-boundary
vertices and adjacent interior vertices while holding every original fixed node
and every outer observation node exact.

The local SLSQP problem has 183 movable nodes and constrains all 1,601
tetrahedra incident to them. It minimizes motion from candidate 001, weights
soft-boundary motion 100 times interior motion, enforces internal determinant
bounds `[0.2501, 1.9999]`, and retains local 10 micrometre bone-clearance
halfspaces. These weights and the numerical determinant margin are geometric
repair choices, not tissue measurements.

The frozen result is
`data/full-skull-initialization-candidate-002/summary.json`:

- candidate NPZ SHA-256:
  `82db42385ec03d8dd5b997b41dbf2ef419d286978994a4b8f8be452af0d75b77`;
- raw displacement SHA-256:
  `685429827d8a6a18fa550d8ac1fd1f4aa00e9852b7e67e3839ae1d4a0931252b`;
- global determinant range: `0.250099999950266` to
  `1.9999000000064693`, with zero inverted or out-of-range cells;
- zero complete-source soft-cranium and soft-mandible triangle intersections;
- minimum soft-node signed clearance approximately 10 micrometres to each
  source bone;
- zero observation-surface RMS motion and 0.013967 mm RMS across the complete
  pure-soft boundary;
- maximum repair from candidate 001: 0.208902 mm;
- valid IPC state with 6,186 active soft-bone pairs, minimum active distance
  0.121721 micrometres, and zero-increment CCD fraction 1.

`admission.json` uses schema `joint-full-skull-contact-admission-v2` and binds
both the NPZ file hash and the raw float64 displacement hash. The current
adapter validation is
`data/full-skull-contact-adapter-validation-003/summary.json`, SHA-256
`73c06d180f4ca4bd4678afd6fffde1fbcfd43064f32177d90fc601e81461b50e`.

The admission covers soft tissue against the complete unchanged source
cranium and mandible. Source bone-bone contact is deliberately excluded by the
collision filter. The registered source bones intersect each other, so this
receipt has `equilibrium_converged=false` and `final_launch_ready=false` and
cannot authorize a jaw-domain or final optimization.

Reproduction command:

```bash
CHERRIES_NAME=full-skull-initialization-repair \
CHERRIES_TAGS=joint,full-skull,initialization,repair \
uv run exp/2026/09/21/joint-activation-material-mandible/src/54-repair-full-skull-initialization.py \
  --output-dir exp/2026/09/21/joint-activation-material-mandible/data/full-skull-initialization-candidate-002
```

The recorded Comet run is `e0f78235e352437eabe7b3c0e5b4b6e3`.
