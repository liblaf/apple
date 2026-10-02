# Active-stress chain verification

Source: `exp/2026/09/21/stress-activation-loss/data/40-runs`

- Receipt-consistency audit: **passed**
- Every stage completed its configured budget: **False**
- Geometry determinants are independently checked only for saved state files (initial, every 50 updates, and final); unsaved accepted iterates have solver receipts and trace evidence, not stored displacements.

| Stage | Status | Final update | Position RMS (mm) | Normal RMS (deg) | R |
| --- | --- | ---: | ---: | ---: | ---: |
| l2-symmetric6 | stalled_line_search_not_convergence_certified | 43 | 4.31521 | 11.145 | 0.0715724 |
| l2-psd6 | blocked_by_parent_failure | None | nan | nan | nan |
| l2-rankone_fixed | blocked_by_parent_failure | None | nan | nan | nan |
| l2-rankone_learned | blocked_by_parent_failure | None | nan | nan | nan |
| normal-symmetric6 | stalled_line_search_not_convergence_certified | 44 | 4.4108 | 8.02486 | 0.142226 |
| normal-psd6 | blocked_by_parent_failure | None | nan | nan | nan |
| normal-rankone_fixed | blocked_by_parent_failure | None | nan | nan | nan |
| normal-rankone_learned | blocked_by_parent_failure | None | nan | nan | nan |

Rendered stage assets were found under [`data/50-figures`](../data/50-figures/).

Reproduction: `CHERRIES_NAME='Stress activation results verification' CHERRIES_TAGS='stress-activation,verification,cpu' uv run python src/60-verify-results.py --source exp/2026/09/21/stress-activation-loss/data/40-runs --output 60-verification`.
