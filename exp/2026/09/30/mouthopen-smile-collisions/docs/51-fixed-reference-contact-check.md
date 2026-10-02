# Fixed-reference contact admission

The repaired pruned reference supports the proposed pure-soft FEM versus full
source-rigid contact map at rest. This is a CPU contact-surface admission check;
it does not establish equilibrium for the MouthOpen pose.

Command:

```bash
CHERRIES_NAME='Fixed reference contact CPU admission' \
CHERRIES_TAGS='mouthopen,contact,fixed-reference,cpu,admission' \
uv run python src/51-check-fixed-reference-contact.py \
  --output 51-fixed-reference-contact-check
```

The receipt is [summary.json](../data/51-fixed-reference-contact-check/summary.json).
Comet recorded the successful run as
[9c17e33a1c7c4a4eae53000edb69efbf](https://www.comet.com/liblaf/apple/9c17e33a1c7c4a4eae53000edb69efbf).

The contact mesh has 34,245 pure-soft FEM vertices and 65,580 triangles. It
appends 28,349 fixed source nodes: 17,575 cranium, 9,476 mandible, and 1,298
eyes. It retains all 56,670 source triangles. The 6,513 mixed FEM attachment
triangles and 54,555 pure FEM bone triangles are outside the sliding surface.

At rest, the standard IPC collision set is empty, barrier energy and contact
force are exactly zero, no scoped soft-rigid intersections were detected, and
the empty set certifies a clearance of at least `dhat = 100 µm` for the
admissible pairs. The contact uses fixed stiffness `1.3544 MPa`, frictionless
IPC, a 10 nm CCD minimum separation, and `TightInclusionCCD` tolerance 0.1 nm
with 100,000 iterations.

The original 26276 `IsFixed` points are retained, including 5989 mandibular
points. The adapter preserves every original free DOF and appends 85,047 fixed
rigid coordinates. Its exported `full_reference_points`, `full_mandible_mask`,
and `contact.indices` provide the renderer and audit mapping. The full prepared
MouthOpen jaw displacement is not collision-free as one chord: CCD permits
only `0.0054338686` of that direct rest-to-pose path. The solver must therefore
continue the prescribed boundary rather than apply the entire pose in one step.
