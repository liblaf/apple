# Jaw contact diagnostics

`49-render-jaw-domain.py` renders the rigid-bone proposal diagnostics and saved
contact-enabled preflight states. The current gallery is
`data/jaw-preflight25-diagnostic003-visuals-003/`, with eight figures:

- conservative CCD fractions for the twelve broad jaw proposals;
- all fifteen full-face probe outcomes;
- accepted zero and +0.01° world-x states in front and side views;
- +0.01° minus resolved-zero surface motion in micrometres, in both views.

The total-state panels share a 0–0.75726 mm scale. The incremental panels isolate
the jaw response: skin RMS is about 7.04 µm and the maximum is about 19.0 µm.
Geometry is rendered at actual scale. The incremental color scale differs from
the total-reference-displacement scale and is explicitly labeled.

The source is the successful **diagnostic-only** preflight
`data/jaw-preflight-spatial25-diagnostic-003/summary.json`, SHA
`bf9a490ec699b9889f03d464c4d67fd58ac6f68a938d6c9f3e3bdc3b79bc6c0e`.
It starts from a frozen intermediate 25% neutral state and cannot authorize the
final optimization. All twelve extreme proposals and the old one-degree
candidate remain visible as rejected diagnostics. No rejected state is rendered
as a converged geometry. See [the preflight report](28-jaw-preflight.md) for
solver, contact and repeatability values.

The renderer checks input and rigid-validation hashes, common seed identity,
accepted numerical receipts, and saved state hashes, metadata, poses, shape and
dtype. Its summary records the preflight's admission flags and each image hash.
The separate rigid-bone chart refers only to the declared FEM collider and its
linear vertex paths; neither it nor a small accepted contact motion validates
complete source-bone anatomy or the whole proposal box.

Run from the experiment directory:

```bash
DEBUG=1 CHERRIES_NAME='Jaw contact diagnostic003 validated motion review' \
CHERRIES_TAGS=joint-inverse,contact,jaw,visualization \
uv run --frozen python src/49-render-jaw-domain.py \
  --jaw-preflight data/jaw-preflight-spatial25-diagnostic-003/summary.json \
  --output-dir data/jaw-preflight25-diagnostic003-visuals-reproduction
```

Rendering completed locally under Cherries with exit 0. The outcome chart and
side-view incremental map were visually inspected. These figures are included
in the tailnet review (private preview omitted).
