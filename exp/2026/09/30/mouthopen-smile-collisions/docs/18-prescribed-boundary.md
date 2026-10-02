# Prescribed boundary collision audit

The requested full MouthOpen jaw pose makes the **all-fixed FEM boundary self-intersect**. This is a geometric obstruction independent of tissue activation and free-node equilibrium: the audited subset contains only triangles whose three vertices obey the saved `IsFixed` boundary condition. The first detected intersection on the exact rigid-pose samples lies between `0.05160217285` and `0.05160293579` of the full pose. The full-pose endpoint intersects.

## Command and provenance

From `exp/2026/09/30/mouthopen-smile-collisions`:

```bash
CHERRIES_NAME='Audit prescribed pruned boundary jaw motion' CHERRIES_TAGS='collision,ipc,fixed-boundary,geometry,cpu' .venv/bin/python -u src/18-audit-prescribed-boundary.py > logs/18-audit-prescribed-boundary-terminal.log 2>&1
```

The CPU run completed Cherries shutdown, passed Ruff, and recorded [Comet 263f1362](https://www.comet.com/liblaf/apple/263f136204e64996980ca5060c4814d5). [`data/18-prescribed-boundary/summary.json`](../data/18-prescribed-boundary/summary.json) records the pruned mesh, original fitted pose, script hashes, face counts, all sampled endpoint results, and chord CCD receipts. Its Cherries Git SHA was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the profile disabled automatic commits.

## Geometry and result

The selected boundary contains 51,225 triangles and 26,218 vertices. Its triangle groups are 39,482 pure Cranium, 11,484 pure Mandible, and 259 mixed or other. The selection uses all three vertices' `IsFixed` flags. The prescribed jaw subset is `IsFixed ∩ Mandible`; 5,984 of its 5,989 vertices occur on this selected surface. All other selected vertices remain stationary. The source pivot and six-component pose are scaled by each sampled fraction, with rotation evaluated through an exact rotation vector at that fraction.

| Full pose fraction | Fixed-only boundary intersects? |
| ---: | :--- |
| 0, 0.025, 0.05 | No |
| 0.1, 0.25, 0.5, 1 | Yes |

Sixteen endpoint bisection checks between 0.05 and 0.1 located the first detected crossing to the fractional bracket `(0.05160217285, 0.05160293579]`. This locates an onset along the sampled rigid-pose path to about `7.63e-7` in pose fraction. It does not identify a unique triangle pair or prove that other earlier transient crossings are absent.

Straight vertex-chord CCD returned collision-free fraction 1.0 for `0→0.025`, `0.61328125` for `0.025→0.05`, and `0.01953125` for `0.05→0.1`, with a 10 nm minimum CCD distance. The middle chord is restricted even though both exact rigid-pose endpoints are clear. A chord is not the curved rotation path, so its fraction cannot be read as the exact first collision time on that path. CCD was not applied after an intersecting start.

This audit tests only self-intersections of the extracted *fixed-only FEM boundary*. It does not evaluate free tissue, separately registered skull or eye obstacles, containment, or active contact forces. Because the full-pose intersection occurs among prescribed faces, changing only a free-tissue solver or activation field cannot make the current full-pose FEM boundary collision-free. A collision-valid full-pose model requires a change to the prescribed geometry, pose, or contact scope, each of which would define a different experiment.
