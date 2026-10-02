# Close-up views of the corrected physical-volume baseline

These figures show the saved step-200 result from the corrected active-strain baseline. Tighter orthographic cameras, matte white material, flat per-face shading, and lateral light reveal the remaining mouth-corner ridges and the stepped bands across the chin. The opposite mouth corner uses a mirrored key light to keep its surface readable.

All views use the actual saved deformation. The 15,299 surface points and 29,899 triangles are unchanged; the crops change only the camera. The visible triangle facets are part of the discrete surface representation, so their shading alone is not a quantitative measure of bumpiness.

| Figure | Vertical view span | Scale bar |
| --- | ---: | ---: |
| [Mouth and chin overview](../data/10-closeups/mouth-overview.png) | 90 mm | 10 mm |
| [Image-left mouth corner](../data/10-closeups/mouth-corner-left.png) | 48 mm | 5 mm |
| [Image-right mouth corner](../data/10-closeups/mouth-corner-right.png) | 48 mm | 5 mm |
| [Lower lip and chin, oblique](../data/10-closeups/chin-oblique.png) | 54 mm | 5 mm |
| [Image-left mouth corner with mesh edges](../data/10-closeups/mouth-corner-left-edges.png) | 48 mm | 5 mm |

Every PNG is 1,400 × 1,542 pixels, including its caption and scale bar. Left and right refer to the frontal image orientation.

![Image-left mouth corner](../data/10-closeups/mouth-corner-left.png)

![Image-right mouth corner](../data/10-closeups/mouth-corner-right.png)

![Lower lip and chin](../data/10-closeups/chin-oblique.png)

The source is the completed baseline's final checkpoint (historical worktree file: `exp/2026/09/08/physical-volume-baseline/data/20-baseline/final.npz`), verified against its final VTU and the earlier comparison's recorded hashes. The surface contains exactly the exterior triangles whose three rest-space vertices carry `IsFace`. The exported [step-200 surface](../data/10-closeups/corrected-step200-skin.vtp) retains each point's volume index for subsequent inspection.

The [render receipt](../data/10-closeups/summary.json) records all source hashes, cameras, lights, materials, and image hashes. [Validation](../data/10-closeups/validation.json) passed all 12 recorded file hashes, exact equality between exported surface coordinates and checkpoint coordinates, and all five PNG dimension/decoding checks. VTK flat interpolation was asserted during rendering. The renderer uses PyVista 0.48.4 and VTK 9.6.2. These are postprocessing outputs; no mechanics solve or optimization was run.

Run from `exp/2026/09/08/physical-volume-closeups`:

```bash
CHERRIES_NAME='Physical volume baseline close-up views' \
CHERRIES_TAGS='physical-volume,closeup,bumpiness,flat-shading,postprocessing' \
.venv/bin/python src/10-render-closeups.py
```

The renderer refuses to overwrite a populated output directory; use `--output-dir data/<new-name>` for a repeat. The earlier 800-pixel previews remain in `data/09-preview`. The final run uses the Comet/local profile with automatic Git commits disabled. Its [Comet record](https://www.comet.com/liblaf/apple/cda6f9bafe9745559965749d6b218d52) and [terminal log](../logs/10-closeups-terminal.log) preserve partial execution evidence. After rendering, Cherries reported a missing snapshot log, Comet reported incomplete logging, and the process ended with exit 143. Successful remote upload is not claimed. The canonical meshes, figures, and receipt were independently verified as described above.
