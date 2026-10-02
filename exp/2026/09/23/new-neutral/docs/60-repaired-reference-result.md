# Neutral from a reference with full barrier clearance

The repaired constitutive reference satisfies the requested minimum distance:
**100.093782531 µm ≥ d_hat = 100 µm** for every enabled soft–rigid pair.
An independent rebuilt IPC model confirms no intersections, zero inverted
reference tetrahedra, and unchanged fixed coordinates. At the zero-pose start,
the actual solver model has zero active contact terms and exactly zero barrier
energy and force.

The active-strain forward rerun preserves positive tetrahedra and valid contact,
but **does not converge** within its existing 100-step Newton budget. Its final
free-force norm is 0.159060385 N, above the required 0.01 N. The served endpoint
is labelled as an unconverged diagnostic result.

## Reference repair and model rebuild

Exact IPC vertex/face and edge/edge stencil projections were alternated with
signed-volume repair while holding fixed nodes and all rigid geometry in place.
The selected historical candidate is `reference-clearance-002/alternation-06.npz`.
The projection target was 100.1 µm; the accepted measured clearance is reported
separately above. The full collision set was queried at a 200 µm screening radius
for the independent audit, rather than relying on vertex samples alone.

The maximum reference correction is 0.904129 mm; the 95th percentile over all
228,660 FEM vertices is 0.0237084 mm. Connectivity is unchanged. Across 1,146,517
tetrahedra, the minimum repaired/original volume ratio is 0.050000000004 and
the 0.1th percentile is 0.765770. Positivity is verified; this is not a claim of
uniformly high element quality.

`reference_rebase.py` rebuilds the constitutive bulk gradients and fraction-
weighted integration volumes, membrane rest metrics and areas, IPC rest
vertices, and boundary map from the repaired coordinates. Material moduli,
prescribed activation/stress inputs used for the active-strain conversion,
fixed nodes, and full cranium/mandible/eye geometry are preserved. The repaired
mesh is the actual material reference; the displacement seed is zero.

The zero-pose rigid transform introduces at most 1.735e-18 m of floating-point
boundary displacement, below the recorded 4.117e-15 m roundoff bound. Free
displacements are exactly zero. Contact is evaluated at that actual state.

## Forward result

| Quantity | Repaired-reference run |
| --- | ---: |
| Initial material force | 3.023991481 N |
| Initial contact force | 0 N |
| Initial total force | 3.023991481 N |
| Force tolerance | 0.01 N |
| Terminal force | 0.159060385 N |
| PNCG / Newton steps | 332 / 100 |
| Instrumented forward time | 120.705 s |
| Adaptive barrier stiffness | 0.1693 → 0.3386 MPa |
| Minimum endpoint det(F) | 0.029106899 |
| Inverted endpoint tetrahedra | 0 |
| Minimum endpoint contact gap | 89.442772 µm |
| Independent triangle intersections | 0 |
| Surface motion RMS with stored observation weights | 0.823158 mm |

The full skin vertex motion has median 0.541444 mm, 95th percentile 1.800781 mm,
maximum 3.022211 mm, and unweighted RMS 0.880524 mm. The preview includes actual
1× motion and a clearly labelled display-only 10× view.

The force rule is unchanged: `max(1e-8, 1e-3 * initial_free_force)` in MPa·m²,
with multiplication by 1e6 to convert to newtons. The absolute term now controls
the threshold. Previously the near-contact initialization gave 576.415 N and
a 0.576415 N relative threshold. The geometry repair removes that contact-heavy
initial residual, rather than weakening the tolerances.

Adaptive stiffness doubled once, at PNCG step 193. The optimized hybrid still
uses PNCG followed by exact assembled sparse Newton-CG, with active strain in
bulk and membrane and guarded inverse/adjoint entry points. The saved sparse
HVP comparison passed its 1e-10 relative-error gate.

`d_hat` is the barrier activation distance. The requested clearance is imposed
on the **reference**; during deformation contact may become active below
`d_hat`. The endpoint retains the existing 10 nm admissibility requirement.
The independent audit reports solver convergence false, geometry valid true,
contact valid true, and therefore `valid_forward=false`.

## Evidence and reproduction

- [Reference independent audit](../data/reference-clearance-002/independent-audit.json)
- [Rebuilt-reference receipt](../data/forward-repaired-reference-001/reference-rebase.json)
- [Forward summary](../data/forward-repaired-reference-001/summary.json)
- [Forward independent audit](../data/forward-repaired-reference-001/independent-audit.json)
- [Saved review](../data/review-repaired-reference-001/index.html)
- [Geometry reproduction and validation scope](reference-clearance-reproduction.md)

Run from `exp/2026/09/23/new-neutral`, using a fresh output directory to repeat:

```bash
CHERRIES_NAME='New neutral from clearance-repaired reference' \
CHERRIES_TAGS='neutral,active-strain,reference-clearance,hybrid,pncg,newton,adaptive-ipc' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/60-forward-repaired-reference.py

CHERRIES_NAME='Repaired-reference active-strain neutral review' \
CHERRIES_TAGS='neutral,active-strain,reference-clearance,review,unconverged' \
.venv/bin/python -u src/20-review-neutral.py \
  --run-dir data/forward-repaired-reference-001 \
  --output-dir data/review-repaired-reference-001
```

Run `35-audit-active-strain.py --run-dir data/forward-repaired-reference-001`
between the forward and rendering commands. The reference audit uses
`55-audit-reference.py --reference-dir data/reference-clearance-002`.
The completed reference audit was launched from the repository root; its exact
input/output hashes and independently rebuilt geometry are recorded in its
receipt. Later runs use the experiment working directory and named Cherries runs.

The forward Comet summary records:

```text
name: New neutral from clearance-repaired reference
forward/seconds: 120.70523947000038
forward/stiffness_mpa: 0.3386
forward/success: 0.0
forward/terminal_force: 1.590603854852654e-07
forward/valid: 0.0
cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
```

[Forward run](https://www.comet.com/liblaf/apple/570a42ca761a4e72908210d8c6c57617),
[forward audit](https://www.comet.com/liblaf/apple/75216603e4c8455fa5b13204dc3ae69b),
[reference packaging](https://www.comet.com/liblaf/apple/d3b85b6ae4da47f6b3c5742f54eee917),
and [review](https://www.comet.com/liblaf/apple/1c059ed74e214942b752bd557b8144c4).
The archived sources are authoritative for each run. The independent forward
audit verified all 17 local experiment source hashes and 352 runtime source
hashes; subsequent display-only changes to scripts 20/40 are reported as drift.
No commit was made. An initial Cherries default hook accidentally staged the
workspace; the prior unstaged index was restored without altering working files
or HEAD, as recorded in `data/staging-recovery.json`. Subsequent profiles disable
automatic commits.

The tailnet review is served at <PRIVATE_PREVIEW_URL> using a transient user
service. All 22 linked HTTP assets matched their local hashes, including the
reference clearance audit. [Hosting receipt](../data/serve-repaired.json) and
[HTTP verification](../data/http-verification-repaired.json) record the actual
service and checks. The [motion review run](https://www.comet.com/liblaf/apple/9f44fba6a0a7432484c4562534e1a9b3)
records the displacement statistics and images.
