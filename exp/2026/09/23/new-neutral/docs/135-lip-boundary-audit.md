# Rendered lip boundary audit

`135-audit-lip-boundary.py` completed successfully with the displayed
`inverse-mouthopen-rigid-trial-001/endpoint.npz` endpoint, whose SHA-256 is
`38ad5705d5997cebc4e3753279a7a3e1cae255d5eedaa4f2e057fdc5450e4a5a`.
The repaired volume SHA-256 is
`50fb2dab7df76a69a1c875cd1291fa004e54e3ea28c05b9a8b7eff3b5b9c4894`, and
the full-soft-surface geometry SHA-256 is
`51f0200f2343b19870b7ad6ccaec52dad4dcf068fa0121fa96cd5ea859a12a4a`.

The fitted full soft surface contains 34,245 vertices and 65,580 triangles.
Of 3,408 volume vertices carrying `IsLip`, 3,020 are vertices of that rendered
surface. None of the 351 `IsLip` vertices prescribed as cranium-fixed and none
of the 15 mandible-prescribed `IsLip` vertices are rendered soft-surface
vertices. Their nearest fitted soft-surface vertices are respectively 0.101 mm
to 2.476 mm away (median 0.881 mm), and 0.378 mm to 1.446 mm away (median
0.936 mm). The red and blue points in the overlay therefore show nearby volume
labels projected over the surface, not fixed vertices of the displayed skin
mesh.

The exact fixed-boundary path is
[`joint_full_skull_contact.py`](../../../21/joint-activation-material-mandible/src/joint_full_skull_contact.py#L447): it initializes every full-node
boundary displacement to zero and assigns `rigid_displacement` only at the
original and appended mandible IDs. Thus the cranium-fixed `IsLip` volume nodes
are zero prescribed, the 15 mandible `IsLip` volume nodes follow the rigid jaw,
and the remaining 3,042 `IsLip` nodes are free. This establishes the numerical
constraint connection, but does not establish that the 165 points selected by a
world-y median are a visible anatomical lower lip. `IsLip` is a volume label and
the lower-half split is only a geometric display heuristic.

Run from the experiment directory:

```bash
CHERRIES_NAME='Audit rendered lip boundary assignments' \
CHERRIES_TAGS='mouthopen,lip,boundary,rendered-surface' \
uv run python src/135-audit-lip-boundary.py
```

The output is [receipt.json](../data/lip-boundary-audit-002/receipt.json) and
[mouth-lip-fixed-overlay.png](../data/lip-boundary-audit-002/mouth-lip-fixed-overlay.png).
The completed Comet record is
[20f2adb88394400fae0e90cc2d3f2172](https://www.comet.com/liblaf/apple/20f2adb88394400fae0e90cc2d3f2172).

The final audit JSON and plot were served and byte-verified over HTTP at
the diagnostics page (private preview omitted).
