# Inverse-physics mobile visual report

`site/index.html` is a self-contained, mobile-first research brief with local
assets only. It summarizes the completed 2D pork studies and the post-hoc 3D
human-face muscle-section audit.

The selected PNG and H.264 assets in `site/assets/` are hard links to the
validated experiment outputs, so the report does not duplicate the large raw
ParaView frame directories. The four Markdown files in `site/records/` are
hard-linked source reports.

The page is intended to be served privately through Tailscale Serve at
`/inverse-physics-report/`. It contains no external scripts, fonts, analytics,
or network dependencies.
