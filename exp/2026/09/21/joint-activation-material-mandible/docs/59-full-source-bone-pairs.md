# Complete source bone-bone intersections

The supplied source cranium and mandible genuinely intersect at the registered
neutral pose. This is separate from the soft-tissue initialization repair.

The 128 raw VTK triangle-pair intersections all mutually straddle the opposing
triangle planes at a `1e-12 m` tolerance. Of these, 120 still straddle beyond the
`8.84 micrometre` source-coordinate precision threshold. Their intersection
segments have median length 0.274 mm and maximum 1.610 mm. Seven cranium vertices
lie inside the mandible beyond that threshold (minimum signed distance -0.211 mm),
and 26 mandible vertices lie inside the cranium (minimum -0.677 mm).

An independently assembled IPC mesh containing **all** source bone triangles,
filtered to cranium-versus-mandible pairs, also reports intersections. Therefore
the complete-source neutral pose is not a valid start for the earlier
intersection-free rigid-bone CCD contract. The partial-FEM guard cannot certify
these source meshes. The soft-bone adapter explicitly excludes bone-bone pairs;
its initialization and isolated zero-jaw performance checks do not imply a valid
full jaw-motion domain.

No source coordinates, topology, exclusions, or jaw pose were changed by this
audit. Resolving the source bone-bone mismatch remains a separate preparation
requirement before claiming full jaw-domain validity.

Reproduce from the experiment directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CHERRIES_NAME='Complete source skull bone pair audit' \
CHERRIES_TAGS=joint-inverse,full-skull,bone-bone,geometry \
uv run --frozen python src/59-audit-full-skull-bone-pairs.py
```

The noncommitting Cherries run exited successfully:
[Comet 5f062ab0](https://www.comet.com/liblaf/apple/5f062ab0fb934a328ddcebcc87d58bab).
The [summary](../data/full-skull-bone-bone-audit-001/summary.json) binds the unchanged
geometry audit. `pairs.npz` stores exact source triangle IDs, intersection
segments, lengths, and both sets of signed triangle-plane distances.
