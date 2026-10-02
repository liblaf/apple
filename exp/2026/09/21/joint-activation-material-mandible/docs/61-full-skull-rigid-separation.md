# Complete-source rigid mandible separation

`61-search-full-skull-rigid-separation.py` asks whether a small rigid mandible
pose can separate the unchanged registered source cranium and mandible. It does
not change either source mesh, the FEM reference, the full-skull initialization,
or any physics code.

The registered pose has 128 exact triangle intersection pairs. The search tests
pure translations of the form `d = normalize([a, -1, c])`, with `a` and `c` on
a coarse grid followed by two local refinements. Each translation ray is sampled
through 2 mm and checked for re-entry before a 20-step bisection. This is a
bounded endpoint search over sampled translations, not a global six-DoF
minimum.

The smallest sampled pair-free endpoint has magnitude 0.8251047 mm and
translation direction
`[-0.0397288, -0.9932196, 0.1092542]`. Its minimum signed distance is only
`3.14e-11 m`, so it is a touching threshold rather than a useful starting
clearance.

Adding the declared 10 micrometre bidirectional signed-distance target produces
the frozen candidate:

```text
rotation vector [rad] = [0, 0, 0]
translation [m]       = [-3.33501996e-05, -8.33754989e-04, 9.17130488e-05]
```

Its exact diagnostics are:

- translation magnitude, maximum mandible motion, and RMS mandible motion:
  0.8394468 mm;
- raw cranium-mandible intersection pairs: 0;
- `ipctk.has_intersections`: false;
- cranium vertices against mandible minimum signed distance: 208.527 µm;
- mandible vertices against cranium minimum signed distance: 10.1001 µm.

A pure negative world-Y translation becomes pair-free at approximately
0.8385305 mm. The sampled landmark-hinge rotations from -2° through +2° never
clear the meshes; the fewest pairs are 78 at +1°.

The registered pose already intersects, so collision-free-start CCD cannot
certify a trajectory from it. The reported pose is endpoint geometry only. It
is not an anatomical registration correction, it does not reuse the zero-pose
soft-tissue initialization, and it requires a new soft-tissue initialization
and equilibrium evaluation before any mechanics use. It is not final-launch
evidence.

Frozen evidence:

- `data/full-skull-rigid-separation-001/summary.json`, SHA-256
  `95151dfe48aa528487c27eb28f00614667927d7193e149a3862abf50a436a1e8`;
- `data/full-skull-rigid-separation-001/poses.npz`, SHA-256
  `7e13449c96403f6fb5a48fb9fd9a76e99ca014fb0868ffcf4e2f0d929f40024b`;
- source SHA-256
  `4b0fba0f64954ef135ed20e6472ac160387050470717df85a27db8e371c32591`;
- Comet run `0032aca454924bf688c10a3a741ee212`;
- wall time 3 minutes 15 seconds, within the predeclared five-minute budget.

Reproduction:

```bash
CHERRIES_NAME=full-skull-rigid-separation \
CHERRIES_TAGS=joint,full-skull,rigid-geometry,separation \
uv run exp/2026/09/21/joint-activation-material-mandible/src/61-search-full-skull-rigid-separation.py \
  --output-dir exp/2026/09/21/joint-activation-material-mandible/data/full-skull-rigid-separation-001
```
