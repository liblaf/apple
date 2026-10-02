# Contact preflight on the pruned tissue boundary

The neutral self-contact pilot stopped before accepting an equilibrium: its Newton search exhausted regularization and Armijo retries after 186 accepted iterations, with free-force norm about `1.10e-7` against the `1e-10` requirement. The CPU diagnostics below identify an inconsistent active-contact construction under the pilot's `IMPROVED_MAX_APPROX` policy. This is a measured numerical obstruction at the saved states; it does not establish that another policy will converge.

## Runs and inputs

From `exp/2026/09/30/mouthopen-smile-collisions`, with the repository `.venv` interpreter:

```bash
CHERRIES_NAME='Diagnose pruned tissue contact stencils and force' CHERRIES_TAGS='collision,ipc,preflight,cpu,diagnostic' .venv/bin/python -u src/15-contact-preflight.py > logs/15-contact-preflight-terminal.log 2>&1
CHERRIES_NAME='Check tissue contact directional derivatives' CHERRIES_TAGS='collision,ipc,finite-difference,cpu,diagnostic' .venv/bin/python -u src/16-contact-directional-check.py > logs/16-contact-directional-check-terminal.log 2>&1
CHERRIES_NAME='Isolate IPC contact set reproducibility' CHERRIES_TAGS='collision,ipc,active-set,cpu,diagnostic' .venv/bin/python -u src/17-contact-set-reproducibility.py > logs/17-contact-set-reproducibility-terminal.log 2>&1
```

The runs completed Cherries shutdown and recorded [15](https://www.comet.com/liblaf/apple/dd6debf97eeb4823bb9d21c756ba27c2), [16](https://www.comet.com/liblaf/apple/d0c9c3a2e1864b5db3b4b253273f06f4), and [17](https://www.comet.com/liblaf/apple/378efee9d94e4771b5f27a85218cc00f) in Comet. Local results are [`data/15-contact-preflight/summary.json`](../data/15-contact-preflight/summary.json), [`data/16-contact-directional-check/summary.json`](../data/16-contact-directional-check/summary.json), and [`data/17-contact-set-reproducibility/summary.json`](../data/17-contact-set-reproducibility/summary.json). Each records hashes of the pruned volume, frozen contact module, script, and failed solver displacement. All three scripts passed Ruff. The Cherries Git SHA was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the profile disabled automatic commits.

## Surface and contact forces

The complete extracted boundary has 126,648 triangles and 63,282 vertices. Of those triangles, 51,225 have three prescribed vertices. The extraction also has 28 nonmanifold edges (56 incident vertices). The contact builder retained every boundary triangle; the physical barrier used `dhat=0.783672 mm`, `stiffness=0.0012 MPa`, area weighting, and a 10 nm CCD buffer.

At rest, the pilot contact policy constructed 335,305 active stencils, including 172,257 with all vertices fixed. The nearest active pair was an all-fixed edge-vertex pair with a 15.611 µm gap, away from the lips and nonmanifold vertices. The free contact-gradient norm was `1.814e-7`. At the failed solver displacement, it constructed 333,730 active stencils, 172,218 all-fixed. The nearest pair had a 0.712 µm edge-edge gap, contained free vertices, and was again away from the lips and nonmanifold vertices. The free contact-gradient norm was `1.604e-7`. Lip-node contact-gradient norms were about `5–6e-8`; nonmanifold-vertex norms were about `2.2e-8`. These are contact-only gradients before adding the bulk FEM force, so they do not partition the pilot's full residual.

## Reproducibility and derivative check

The first central differences used the unit negative free contact-gradient direction. At the failed state, the analytic derivative was `-1.606e-7`; fresh-state differences were `-6.623e-7`, `-6.636e-6`, and `-1.104e-5` at perturbations `1e-7`, `1e-8`, and `1e-9 m`. The active stencil count changed between positive and negative evaluations.

The follow-up repeated fresh construction five times at *identical* coordinates, and also compared IPC's three installed normal-collision set types:

| Set type | Identical rest-state energy range | Identical rest-state stencil counts | Largest gradient difference from first repeat | Failed-state fresh FD at `1e-8 m` |
| --- | ---: | ---: | ---: | ---: |
| `IMPROVED_MAX_APPROX` (pilot) | `2.409066–2.409382e-9` | `335,308–335,341` | `2.30e-8` | Relative error `64.0` |
| `IPC` | `1.863148e-8` to floating-point roundoff | `322,351` in every repeat | `1.19e-22` | Relative error `9.45e-9` |
| `OGC` | `9.620073e-10` to floating-point roundoff | `25,599` in every repeat | `6.20e-24` | Relative error `4.27e-9` |

At the failed state, `IMPROVED_MAX_APPROX` also varied between 333,770 and 333,880 active stencils at identical coordinates; `IPC` and `OGC` remained bit-stable in stencil count and nearly exact in energy and gradient. Holding the improved *active stencils* fixed during a central difference reduced its relative derivative error to roughly `1e-5` at `1e-7 m` and `1e-7` or less at smaller perturbations. Merely holding its broad-phase candidate list fixed did not remove the variation. The evidence places the inconsistency in the improved active-set construction or merge, not in the barrier's fixed-stencil derivative. It does not identify the particular low-level branch or establish a geometry-wide theorem.

The immediately testable numerical change is a separate pilot with `CollisionSetType.IPC`, retaining every face, the same barrier and area weighting, and the same force, CCD, and intersection gates. That policy changes the contact potential and the active-contact count; its result must be labeled as a new model trial. The existing pilot and its failed state remain intact.
