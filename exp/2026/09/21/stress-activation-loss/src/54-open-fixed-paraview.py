"""Load the fixed-axis tetmesh and bones in the ParaView GUI."""

import json
import os
from pathlib import Path

from paraview.simple import (
    GetActiveViewOrCreate,
    OpenDataFile,
    RenameSource,
    Render,
    SaveScreenshot,
    SaveState,
    SetActiveSource,
    Show,
)

ROOT = Path(__file__).resolve().parents[6]

GROUP = ROOT / "exp/2026/09/21/stress-activation-loss"
OUTPUT = GROUP / "data/53-paraview-l2-normal-fixed-001"
HEAD = Path(os.environ["APPLE_MELON_HEAD"])
camera = json.loads(
    (GROUP / "data/52-four-stage-figures-009/summary.json").read_text()
)["camera"]
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1500, 1100]
view.UseColorPaletteForBackground = 0
view.Background = [0.9569, 0.9490, 0.9294]

tetmesh = OpenDataFile(str(OUTPUT / "l2-normal-fixed-axis.vtu"))
RenameSource("L2 + normal - fixed axis - update 200 (tetmesh)", tetmesh)
display = Show(tetmesh, view)
display.ColorArrayName = ["POINTS", ""]
display.Representation = "Surface With Edges"
display.DiffuseColor = [0.553, 0.588, 0.608]
display.AmbientColor = display.DiffuseColor
display.EdgeColor = [0.23, 0.26, 0.28]
display.LineWidth = 0.5
for filename, name in (("13-cranium.ply", "Cranium"), ("13-mandible.ply", "Mandible")):
    bone = OpenDataFile(str(HEAD / filename))
    RenameSource(name, bone)
    display = Show(bone, view)
    display.ColorArrayName = ["POINTS", ""]
    display.Representation = "Surface"
    display.DiffuseColor = [0.902, 0.875, 0.812]
    display.AmbientColor = display.DiffuseColor
view.CameraParallelProjection = 1
view.CameraPosition = camera["position"]
view.CameraFocalPoint = camera["focal_point"]
view.CameraViewUp = camera["view_up"]
view.CameraParallelScale = camera["parallel_scale"]
SetActiveSource(tetmesh)
Render(view)
SaveState(str(OUTPUT / "fixed-axis-tetmesh-bones.pvsm"))
SaveScreenshot(str(OUTPUT / "paraview-opened.png"), view, ImageResolution=[1500, 1100])
(OUTPUT / "paraview-ready.json").write_text(
    json.dumps({"state_loaded": True, "sources": ["tetmesh", "cranium", "mandible"]})
    + "\n"
)
