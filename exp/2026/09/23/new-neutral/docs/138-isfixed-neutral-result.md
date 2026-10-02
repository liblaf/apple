# Fresh neutral equilibrium with IsFixed constraints

Run `forward-isfixed-001`, completed on 2026-09-29, reaches the force tolerance
with positive tetrahedron orientation and valid contact. It starts from zero
displacement at the clearance-repaired constitutive reference; no previous
neutral endpoint or mandible pose is used.

| Quantity | Result |
| --- | ---: |
| Fixed original FEM vertices | 27,036 |
| Fixed lip vertices | 0 |
| Free Cartesian DOFs | 604,872 |
| Initial residual | 3.023991481 N |
| Initial contact force | 0 N |
| Final residual | 0.009646026 N |
| Effective force tolerance | 0.01 N |
| Inverted tetrahedra | 0 / 1,146,517 |
| Minimum / maximum det(F) | 0.483974 / 1.363880 |
| Minimum active contact distance | 77.129824 micrometres |
| Required deformed contact distance | 0.01 micrometres |
| Weighted skin motion RMS | 1.358478 mm |
| Maximum vertex displacement | 4.571323 mm |
| PNCG / Newton steps | 340 / 134 |
| Instrumented forward time | 97.6387 s |
| PNCG / Newton time | 34.3436 / 63.2931 s |

The actual solver fixed and free DOF arrays were asserted equal to `IsFixed`
plus the separately appended rigid bone and eye DOFs before solving. Group
membership does not add constraints. The jaw pose remains zero throughout.

The material formulation remains multiplicative active strain. Bulk neutral
activation is identity; skin activation retains the existing prescribed
prestretch. Relative tolerance is `1e-3`, absolute tolerance is `1e-8 MPa m²`,
and the stopping rule is their maximum after scaling the relative term by the
initial residual. The absolute term controls this run. The observed relative
residual is approximately 0.003190; convergence is through the absolute gate.

Adaptive IPC stiffness increases from 0.1693 to 1.3544 MPa in three PNCG
updates, then stays fixed in the Newton phase. The reference has zero active
IPC terms at the initial state. Its previously revalidated minimum clearance
is 100.093783 micrometres, above the 100 micrometre activation distance.
Deformed contact is allowed inside the activation distance, while retaining
the positive gap requirement above.

The fresh hybrid entry now forwards the existing shift-reuse configuration to
its Newton phase. Previous fresh entries hardcoded reset despite exposing the
configuration in their protocol. This run uses `reuse` with force ratio zero,
and the executed trace verifies that policy. The unshifted physical sparse
Hessian agrees with the native Hessian-vector product to relative error
`3.0772251547390394e-16`. The shift changes only the linear search system.

This run validates the corrected configuration. Differences in shift policy,
adaptive stiffness history and runtime conditions mean comparisons with the
old invalidated run do not isolate the speed effect of changing the fixed set.
The old MouthOpen results and posed-mandible validity findings are not replaced
by this neutral-only solve.

## Independent verification

The completed saved-state CPU audit recomputes all 1,146,517 determinant ratios
in float64: minimum `0.483974044268967`, maximum `1.363880068541585`, and no
inversions. Its force and contact gates agree with the forward receipt.
It verifies 58 archived local sources and 352 archived runtime sources, as well
as the active-strain mapping and the corrected boundary binding.

The stronger supplemental audit directly checks the saved `FixedMask` against
the source `IsFixed`, and checks the rebase fixed-ID hash with its dtype/shape/
bytes convention. The runtime boundary receipt hashes the same IDs using bytes
only. Both checks pass. Fixed vertex displacement is at most
`1.734723475976807e-18 m`, consistent with zero-pose roundoff.

The auditor was extended after the forward source freeze; this is recorded as
source drift for `35-audit-active-strain.py`. Executed source archives remain
unchanged. The first audit receipt is preserved; the supplemental audit below
contains the explicit saved-mask and rebase hash checks.

The saved-state renderer independently finds zero triangle intersections with
the cranium, mandible and eyes. Its front and side views show the reference and
solved surfaces at identical true scale, alongside a force curve and anatomy
views. The review is served at
the tailnet preview (private preview omitted) by transient user
service `apple-neutral-isfixed-preview.service`. The HTTP index was compared
byte-for-byte with the saved page; published images and audit receipts were
also fetched and verified. No persistent service file was installed.

## Reproduction and evidence

The process completed with exit code zero after Cherries shutdown, using a
profile without automatic Git staging or commits. Sources and input hashes are
archived with the run. Use a fresh output directory when repeating:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Fresh neutral equilibrium with corrected IsFixed constraints' \
CHERRIES_TAGS='neutral,isfixed,active-strain,hybrid,adaptive-ipc' OMP_NUM_THREADS=4 \
.venv/bin/python -u src/138-forward-isfixed.py \
  > tmp/forward-isfixed-001.log 2>&1
```

- [Forward summary](../data/forward-isfixed-001/summary.json)
- [Runtime boundary receipt](../data/forward-isfixed-001/fixed-boundary.json)
- [Executed trace](../data/forward-isfixed-001/trace.jsonl)
- [Timing](../data/forward-isfixed-001/timing.json)
- [Complete log](../tmp/forward-isfixed-001.log)
- [Completed Comet run](https://www.comet.com/liblaf/apple/ac1bf7f7c0014f169805e017998a0fd2)
- [Independent audit](../data/forward-isfixed-001/independent-audit-isfixed-verified.json)
- [Completed supplemental audit](https://www.comet.com/liblaf/apple/ccd865ffd8984b13b00857133368fd96)
- [Saved visual review](../data/review-isfixed-001/index.html)
- [Review receipt](../data/review-isfixed-001/receipt.json)
- [Completed review render](https://www.comet.com/liblaf/apple/31419e2856bb43c78b2dac526494ea2b)
- [Completed anatomy render](https://www.comet.com/liblaf/apple/0bd13a99cb3145e0a24bfd66ded8d358)
