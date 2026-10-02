# Bones and eyeballs in the neutral preview

The tailnet preview (private preview omitted) now renders the exact source
cranium, mandible, and eyeballs beside the reference and saved active-strain
endpoint. The anatomy detail (private preview omitted) adds front and
side views with skin at 18% opacity. The motion view includes the same rigid
meshes in all panels; only skin displacement is magnified in its labelled
10x display.

The cranium has 17,575 vertices / 35,162 triangles; the mandible has
9,476 / 18,948; and the eyeballs have 1,298 / 2,560. Sources are the hash-bound
`simple-skin-forward-inputs-001/geometry.npz` and `rigid-eyes-001/eyes.vtp`.
The saved protocol has zero jaw rotation, so no rigid transform is applied.
Rendering asserts that all rigid coordinates remain exactly unchanged and
the saved skin equals repaired reference plus endpoint displacement.

The original solver state and convergence status are preserved. The main
preview links to the added images, transparent views, downloadable rigid
meshes, and provenance; the energy link remains available. Visual checks
covered the opaque front, transparent side, and motion front images. Ruff
passed, and `data/http-verification-anatomy.json` records HTTP/local hash
checks. Hosting remains the existing transient user service.

From `exp/2026/09/23/new-neutral`, with a fresh review destination when
repeating:

```bash
CHERRIES_NAME='Neutral preview with bones and eyeballs' \
CHERRIES_TAGS='new-neutral,anatomy,rigid-eyes,review' \
.venv/bin/python -u src/80-add-anatomy.py
```

The [Cherries/Comet run](https://www.comet.com/liblaf/apple/4b4df1de878e4976ba3000e4ceddb301)
completed with triangle metrics 35,162 / 18,948 / 2,560, source HEAD
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, and commits disabled. The first
attempt stopped before creating output because the manifest included a
`bytes` metadata field; verification now checks the recorded SHA-256 rather
than comparing dictionaries with different metadata keys. Its log is retained
as `tmp/anatomy-setup-failure.log`.
