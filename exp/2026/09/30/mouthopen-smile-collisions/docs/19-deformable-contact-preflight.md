# Saved expression states under deformable tissue contact

None of the three saved expression states is an intersection-free start for the requested contact scope. The MouthOpen rank-one learned endpoint and both endpoints of the old MouthOpen-to-Smile transition have detected intersections involving at least one free FEM boundary vertex. The undeformed rest mesh is clear under the same scope. A direct fixed-activation equilibrium solve from any of those saved displacements would start inside its contact barrier.

## Scope and command

All 126,648 extracted FEM boundary triangles remain in the collision mesh. Standard `IPC` normal-collision stencils use the same physical barrier and area weighting as the prior pilot. A vertex patch filter gives every free boundary vertex a distinct positive patch ID and gives every fixed boundary vertex patch zero. Thus free–free and free–fixed pairs are eligible, while fixed–fixed pairs are exempt. The fixed-only overlaps remain available in the separately reported unfiltered audit. The numerical activation fields and saved displacements were only read, not modified.

From `exp/2026/09/30/mouthopen-smile-collisions`:

```bash
CHERRIES_NAME='Check saved expressions with deformable tissue contact' CHERRIES_TAGS='collision,ipc,expression,preflight,cpu' .venv/bin/python -u src/19-check-deformable-contact.py > logs/19-check-deformable-contact-terminal.log 2>&1
```

The Cherries run completed, passed Ruff, and recorded [Comet ef274b9b](https://www.comet.com/liblaf/apple/ef274b9b02184ef6835a117db2095508). The [summary](../data/19-deformable-contact/summary.json) includes hashes of the exact pruned fixture, contact module, script, and each saved state. The profile disabled automatic commits.

## Filter verification

The installed `make_vertex_patches_filter` was checked with two known crossing triangles. Unfiltered contact detected the crossing. Assigning patch zero to all six vertices excluded the fixed–fixed crossing from `has_intersections`. Giving the second triangle distinct free patches admitted the crossing; direct filter calls confirmed both free–fixed and free–free eligibility. On the real boundary, the filter mapped 26,276 fixed and 37,006 free vertices, with one distinct patch for each free vertex.

## Results

| Saved displacement | Unfiltered intersects? | Scoped intersects? | Scoped active stencils | Scoped minimum active gap |
| --- | :---: | :---: | ---: | ---: |
| Undeformed rest | No | No | 157,381 | 18.971 µm |
| MouthOpen four-stage rank-one learned | Yes | **Yes** | 141,276 | 14.414 µm |
| Old transition frame 120, MouthOpen | Yes | **Yes** | 142,404 | 0.496 µm |
| Old transition frame 000, Smile | Yes | **Yes** | 167,689 | 0.0880 µm |

The minimum active gap is the nearest *active IPC stencil*, not the minimum distance over every primitive pair. A positive active gap does not contradict an intersection detected elsewhere; the endpoint intersection test is a separate complete scoped geometry check. The saved learned MouthOpen state and old transition frame 120 are distinct saved displacements, both individually audited.

These results rule out a contact-feasible direct initializer from the saved endpoint displacements. A contact solve preserving each exact saved activation field would need an intersection-free state obtained by a declared continuation or geometry repair, with fixed jaw motion and the scoped intersection gate verified at every accepted state. This preflight neither solves equilibrium nor proves that such a continuation will reach the saved endpoints.
