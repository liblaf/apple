# Adopted neutral face

The user accepted the Run011 deformed face after inspecting it in ParaView.
It is now the selected neutral in `data/current-neutral.json`, bound to
`data/frozen-neutral-004/manifest.json`. Baseline stress is fixed, with **zero
baseline-stress optimization variables**. All six transferred expression
displacement fields are preserved and applied to the new neutral; four are
training expressions and two remain reserved.

The saved geometry is a loaded equilibrium, not a stress-free material mesh.
Keep the original constitutive reference `X` and checkpoint displacement `u0`:

```text
neutral geometry:       x0 = X + u0
expression geometry:    x  = X + u0 + v
transferred target:     y  = x0 + original_expression_displacement
solver target:          u_target = u0 + original_expression_displacement
```

The old bulk and membrane reference geometry, skin baseline resultant, full
source skull, contact weights, and constitutive frames remain unchanged.
Bulk additive baseline stress remains zero; passive stresses induced by `u0`
remain present. Setting the saved deformed meshes as new stress-free FEM input
would discard those stresses and is not the implemented operation.

## Verified results

The [new validation receipt](../data/frozen-neutral-004/validation.json) rebuilds
the full model, evaluates its force without refitting baseline stress, and checks:

| Quantity | Result |
| --- | ---: |
| Neutral free-force norm | 0.000141868289 N |
| Original force threshold | 0.000151920035 N |
| Inverted tetrahedra | 0 / 1,146,517 |
| Full-source soft–bone intersections | None |
| Minimum active IPC-pair distance | 17.0666 µm |
| Active contact pairs | 4,343 |
| Baseline stress coordinates exposed by new material interface | 0 |
| Original expression displacement arrays | Exactly preserved |

The initialized PNCG primary force criterion passes at step zero. No additional
neutral optimization is performed. The translated coordinates preserve energy,
gradient, HVP and the tested CCD path. Repeated GPU reductions differ by at most
`3.57e-22` in energy and `3.97e-23` in gradient components (code units); the check
uses energy `rtol=1e-12, atol=1e-22` and derivative
`rtol=1e-10, atol=1e-20`. Coordinate equality and original target-field equality
are exact.

The same-muscle activation graph retains all 501,409 edges and 288,235 active
tetrahedra. Its geometric conductances, effective muscle-volume weights, and
observation area weights are recomputed on the adopted neutral. Training and
reserved splits, tissue labels and connectivity are unchanged. Strong field
smoothness remains part of the inverse design.

The new material interface exposes skin stiffness and per-tetrahedron expression
active stress, while preserving bulk and skin baseline fields. Jaw pose remains
expression-specific. A global skin-stiffness multiplier is the current shared
parameter choice; it scales the heterogeneous literature-derived map.

Changing stiffness can move the unloaded equilibrium even though baseline stress
is fixed. A 1% increase in skin stiffness produces **0.0189830 N** residual at
the frozen geometry; the probe then restores the original materials and residual.
Subsequent stiffness updates therefore require fresh equilibrium and neutral-drift
assessment. This adoption does not establish stability or validate the anatomy.

## Artifacts and implementation

- [Selected neutral](../data/current-neutral.json), with manifest hash.
- [Neutral volume](../data/frozen-neutral-004/neutral-volume.vtu) and
  [neutral skin](../data/frozen-neutral-004/neutral-skin.vtp), already deformed at
  true scale. Do not apply Warp By Vector again or use them as stress-free inputs.
- `state.npz`: frozen displacement, neutral positions, transferred target positions
  and total-displacement targets, graph metrics, weights, partitions and IDs.
- `target-*.vtp`: six transferred target surfaces for ParaView.
- `joint_frozen_neutral.FrozenNeutral.load()`: reads the selected bundle and
  checks every source/artifact hash. `expression_materials()` exposes no baseline
  parameters; `expression_residual()` uses the transferred targets.
- `NeutralIncrementProblem`: accepts free increments and evaluates the unchanged
  physical model at total displacement `u0 + v`.

The historical `30-joint-pilot.py` / `50-run-prepared-joint-sequence.py` workflow
still implements its older baseline-fitting contract. It must be integrated
with this adapter before launch; it has not been run under the new contract.
The adopted neutral replaces the old baseline-fitting preparation. Full-skull
expression/jaw derivative checks and a converged activation/jaw control remain
separate preparations before final optimization trends.

## Reproduction

From `exp/2026/09/21/joint-activation-material-mandible`, using a fresh output
directory:

```bash
CHERRIES_NAME='Freeze validated equilibrium and transfer expression targets' \
CHERRIES_TAGS=joint-inverse,neutral,frozen-baseline,nu049,target-transfer \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/82-adopt-converged-neutral.py \
  --output-dir data/frozen-neutral-005
```

The [successful Cherries/Comet run](https://www.comet.com/liblaf/apple/bd44efccf6714f0f917f9036eeafa80e)
exited zero after its logging hooks completed. Its source archive and command
receipt are in the bundle; no Git commit or push was made. Ruff and Python
compilation checks passed for the new adapter and producer.

Two unselected producer attempts remain as diagnostic records: candidate001
incorrectly demanded bitwise equality of CUDA reductions; candidate002 called
`minimize`, whose loop always takes a first step, before checking that the
accepted neutral remained exactly unchanged. Candidate003 checks the initialized
force criterion and preserves the approved checkpoint. Neither rejected attempt
changed the selected neutral or invalidated the original converged forward solve.

The final candidate004 additionally binds the reconstruction runner, joint mechanics
modules, Apple sources, historical reference sources, and lockfile hashes.
Loading or rebuilding rejects source drift. It also verifies that the physics
object consumes the adopted observation weights, effective muscle volumes, graph
conductance and total-displacement targets. Candidate003 was superseded after
code review found that its lower-level physics constructor still received the
older weighting arrays; its neutral force check itself was valid.

The mobile review was rebuilt and its served status matched the local file.
The reading server remains runtime-only, managed by the transient
`apple-joint-review-neutral.service`; it removes its runtime directory on stop.
