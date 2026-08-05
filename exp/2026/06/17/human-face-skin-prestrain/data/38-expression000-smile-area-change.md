# Expression000 vs Smile Area Change

- `Expression000` mesh: `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/50-tetmesh-3191k.vtu`
- `Smile` mesh: `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu`
- Same point count: `True`
- Same cell count: `True`
- Max rest point delta: `0`
- Same `IsFace`: `True`
- Compared surface triangles with all vertices in `IsFace`: `29899`

## IsFace Triangle Area Change

| target | disp RMS | disp max | total area | q1 | median | q99 | >10% | >25% | >2x | stretch >10% | shrink >10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Expression000` | 0.00288432 | 0.00961969 | 0.996684 | 0.677457 | 1.00279 | 1.44623 | 23.19% | 8.40% | 0.14% | 8.43% | 14.77% |
| `Smile` | 0.00531014 | 0.0153947 | 0.995904 | 0.632231 | 1.00336 | 1.72515 | 23.30% | 9.94% | 0.77% | 10.51% | 12.80% |

## Takeaway

`Expression000` has a much smaller displacement magnitude and less extreme triangle-area distortion than `Smile` on these generated meshes.
`Smile` has 1.00x as many `IsFace` triangles beyond 10% area change, and 1.18x as many beyond 25%.
