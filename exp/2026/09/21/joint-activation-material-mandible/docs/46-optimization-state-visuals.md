# Accepted optimization spatial-state renderer

`src/46-render-optimization-state.py` is the bounded spatial renderer for real
accepted control or final-joint checkpoints. It consumes the versioned
`visualization-*.npz` snapshots from `30-joint-pilot.py` and the VTU/VTP exports
from `31-render-final.py`. It does not solve the FEM system or create a result
when the requested real checkpoints are absent.

No final large joint optimization had started when this renderer was added, so
there is intentionally no `data/optimization-state-visuals` review set yet.
Synthetic plumbing output is not retained as scientific evidence. The first
real render must be launched only after `31-render-final.py` has exported the
selected accepted checkpoints, for example:

```bash
DEBUG=1 \
CHERRIES_NAME="Accepted joint spatial states" \
CHERRIES_TAGS="joint-inverse,optimization,spatial,contact,visual-qa,cpu" \
uv run --frozen python src/46-render-optimization-state.py \
  --run-dir data/<real-control-or-final-run> \
  --spatial-dir data/<matching-31-render-output> \
  --output-dir data/<matching-optimization-state-visuals>
```

The full path was exercised once with a clearly labeled one-expression fixture
using the real current-neutral float64 displacement, zero synthetic activation,
and a transferred target. `31` wrote one VTU and one VTP; `46` validated the
schema, rendered six assets, and rebuilt the same 138-pair numerically valid IPC
state and force totals as the neutral contact audit. The entire fixture and all
images were then deleted from `/tmp`; none appears in the review data.

The default selected labels are `0000`, `best`, and `terminal`. Missing labels,
schema mismatches, a non-accepted snapshot status, inconsistent prepared IDs,
missing full-state displacement, or missing VTU/VTP fields fail immediately.
Each snapshot must declare
`schema=joint-optimization-visualization-snapshot-v2`, its stage and checkpoint
label, `status=accepted_numerically_valid_state`, activation reference and cap,
the fixed symmetric-coordinate order, and the complete solved expression
displacements in float64. The observation displacement must be an exact slice
of that full field, so contact is never rebuilt from a quantized display state.
Version 2 additionally requires exact JSON material, shared-field and spatial
basis metadata. A Spatial80 snapshot must bind the archived basis hashes and
the fixed strong smoothness weight. Version 1 remains supported only for the
original constant20 model; it cannot silently carry spatial coefficients.

Version 2 adds minimum/maximum principal shared-baseline stress on fixed
coronal and sagittal reference sections, plus effective skin stiffness. These
maps read the per-checkpoint `shared-baseline-<label>.vtu` and
`shared-skin-<label>.vtp` exports from `31`; they check geometry ordering and
skin stiffness against checkpoint coefficients. One signed stress scale spans
every selected checkpoint, and the skin scale uses the fixed model bounds.
The current skin field remains spatially uniform. An in-memory contract check
accepted the real Spatial80 metadata and rejected both a corrupted basis hash
and spatial coefficients under the legacy schema. This is validator evidence,
not a new control or final simulation.

The aggregate is supplemented by individual fat, aponeurosis and muscle
largest-principal stress sections, each with its own signed scale fixed across
the selected checkpoints. These show the constitutive tensor before tissue
fraction weighting; otherwise the much stiffer aponeurosis hides the other
fields in an aggregate map. The expanded v3 export fixture rendered all 11
shared-field PNGs successfully, with geometry/order/field checks and visual
inspection. It is isolated under `data/shared-render-contract-validation-v3/`
and explicitly marked `test_fixture_only=true`, `scientific_result=false`.

For every expression and selected checkpoint, the renderer produces:

- fixed-camera front and side overlays of the frozen skin, transferred target,
  and solved FEM prediction;
- a front skin residual heat map in millimetres, checked numerically against the
  VTP residual exported by `31`;
- fixed coronal and sagittal active-muscle sections with largest-principal
  activation in MPa and centered 1.5 mm line glyphs for its physically unsigned
  principal axis;
- one exact owned-IPC oral close-up per checkpoint for the configured
  jaw-sensitive expression, rebuilt on CPU from the full solved displacement.

Activation always uses the frozen model cap as its color maximum. Residual uses
one scale across all selected checkpoints and expressions; by default its cap
is the joint 99th percentile and every image and receipt states the uncapped
maximum and clipped fraction. Contact-force magnitude uses one run-wide scale.
The saved `summary.json` records cameras, scales, caps, units, checkpoint hashes,
per-expression residual distributions, jaw poses, contact diagnostics, and
per-bone nodal-force summaries. IPC force is the negative energy gradient in
model MPa m2 multiplied by `1e6` to report N; nodal force is not pressure.

The renderer describes accepted numerical states. It never upgrades a control
or intermediate state to a final trend, and its receipt fixes
`anatomical_validation=false` and `promotion_ready=false`. Convergence and the
Tuesday trend claim remain properties of the final large-joint run summary and
its admission gates.
