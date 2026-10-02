# Rigid FEM bone collision guard

The declared FEM cranium and mandible are intersection-free at reference, but
all 12 signed axis endpoints of the computational jaw proposal box encounter
bone collision along the solver's linear boundary-vertex motion. The zero
probe passes. A box of ±10 degrees and ±5 mm therefore describes candidate
parameters, not a domain in which every pose or transition is feasible.

`joint_rigid_bone_collision.py` constructs a separate native IPC cross-bone
filter from the exact pure FEM bone faces: 29,286 vertices, 41,903 cranial
triangles, and 13,763 mandibular triangles. It adds no energy, forces, or
derivative terms. Its role is to reject a proposed boundary transition unless
the complete straight vertex path and endpoint are collision-free. The initial
configuration must also be intersection-free; an invalid start fails visibly.
This complements the existing frictionless soft-tissue/bone IPC barrier and
its differentiated contact state.

## Verified evidence

The current receipt is
`data/rigid-bone-ccd-validation-003/summary.json`, SHA-256
`a901ae4751212dfde6580ac328df783d8096325f7dd7b2b83bb1e269fe54850a`.
The [Comet run](https://www.comet.com/liblaf/apple/c25a3ae60a9d4c31b569c2339181e2c2)
completed with exit code zero and noncommitting `ProfileJoint` recording.

```bash
CHERRIES_NAME='Rigid bone CCD and warm-seed contract validation' \
CHERRIES_TAGS=joint-inverse,contact,ccd,jaw,validation \
uv run --frozen python src/29-validate-rigid-bone-ccd.py \
  --output-dir data/rigid-bone-ccd-validation-003
```

The synthetic check distinguishes a through-crossing from two disjoint
endpoints: the endpoint intersection test is false, while the collision-free
fraction is `0.49994659423828125`, so CCD rejects the transition. Stationary and
separating motions pass, and an intersecting start is rejected. On the face,
the warm displacement adapter gives exactly the same receipt as the pose
adapter for a 0.01-to-0.02 degree x-rotation. A seed that moves the fixed cranium
is rejected. The helper SHA is
`6e401c9de174d04f0d6a58ae54a6277e1c5cbe66543919b1639adcfb73ecd9cd`.

| Proposed coordinate | Negative direction safe fraction | Positive direction safe fraction |
| --- | ---: | ---: |
| Rotation x, 10 degrees | 0.08190918 | 0.22851562 |
| Rotation y, 10 degrees | 0.01501465 | 0.02069092 |
| Rotation z, 10 degrees | 0.01013184 | 0.01538086 |
| Translation x, 5 mm | 0.02319336 | 0.02081299 |
| Translation y, 5 mm | 0.03448486 | 0.01513672 |
| Translation z, 5 mm | 0.02233887 | 0.03063965 |

These are conservative fractions of a **linear interpolation of vertices**.
Multiplying a rotation angle by a fraction does not establish a safe rigid
rotation arc. No entire-box or anatomical validation follows from the samples.
Complete registered source bones, joint cartilage, and articulation constraints
are outside this guard. Their unresolved source geometry remains documented in
[the source-bone audit](17-source-bone-contact-audit.md).

## Visualization

`49-render-jaw-domain.py` renders the 12 measured fractions with a logarithmic
axis and an explicit full-step threshold. The figure and its source hash are in
`data/jaw-domain-visuals-001`. This is numerical collision evidence, not an
equilibrium result. The full contact-enabled jaw preflight is separate in
`28-run-jaw-preflight.py` and must record its actual converged/rejected cases.

The earlier `001` and `002` receipts remain archived. `003` adds the seed-adapter
and invalid-start checks; it does not change the measured extreme-pose fractions.
Targeted Ruff and compilation checks pass.
