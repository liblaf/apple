# Full fitted soft-surface render

Rendered the saved rigid MouthOpen trial with the complete 65,580-triangle
source soft surface, plus complete cranium, mandible, and eyes. The target
panel remains the observed face patch, because it has no complete-surface
target.

Run:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Render rigid inverse full fitted surface' \
CHERRIES_TAGS='mouthopen,rigid-inverse,full-surface,review' \
uv run python src/133-render-full-fit-surface.py
```

The receipt and SHA-256-pinned assets are in
`data/review-repaired-reference-005/rigid-inverse/full-surface/receipt.json`.
It contains 122,250 triangles total: 65,580 soft, 35,162 cranium, 18,948
mandible, and 2,560 eyes. Comet:
<https://www.comet.com/liblaf/apple/bcc2847cdf3b4e4aa2c5dba7ed9efce5>.

Root verification: full front/side PNGs and both fitted VTP downloads return
HTTP 200 at the tailnet viewer and exactly match the recorded SHA-256 hashes.
The complete front view was visually inspected.
