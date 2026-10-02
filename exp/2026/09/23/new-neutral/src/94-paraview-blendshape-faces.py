"""Render the slide's exact surface meshes using ParaView's light kit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from paraview import servermanager
from paraview import simple as pvs


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    pvs._DisableFirstRenderCameraReset()  # noqa: SLF001
    view = pvs.CreateView("RenderView")
    view.ViewSize = spec["resolution"]
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = spec["background"]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = spec["camera"]["position"][0]
    view.CameraFocalPoint = spec["camera"]["position"][1]
    view.CameraViewUp = spec["camera"]["position"][2]
    view.CameraParallelScale = spec["camera"]["parallel_scale_m"]
    view.UseLight = 1
    view.LightScale = 0.85
    view.KeyLightWarmth = 0.5
    view.FillLightWarmth = 0.5
    view.BackLightWarmth = 0.5
    view.HeadLightWarmth = 0.5
    view.UseFXAA = 1
    for index, surface in enumerate(spec["surfaces"]):
        source = pvs.XMLPolyDataReader(FileName=[surface])
        display = pvs.Show(source, view)
        display.ColorArrayName = ["POINTS", ""]
        display.Representation = "Surface"
        display.DiffuseColor = [1, 1, 1]
        display.AmbientColor = [1, 1, 1]
        display.Interpolation = "Gouraud"
        display.Ambient = 0.5
        display.Diffuse = 0.3
        display.Specular = 0.0
        pvs.Render(view)
        pvs.SaveScreenshot(
            str(Path(spec["output"]) / f"{index:02d}.png"),
            view,
            ImageResolution=spec["resolution"],
        )
        pvs.Hide(source, view)
        pvs.Delete(display)
        pvs.Delete(source)
        print(f"ParaView rendered {index + 1} / {len(spec['surfaces'])}", flush=True)
    manager = servermanager.vtkSMProxyManager
    (Path(spec["output"]) / "paraview.json").write_text(
        json.dumps(
            {
                "version": f"{manager.GetVersionMajor()}.{manager.GetVersionMinor()}.{manager.GetVersionPatch()}",
                "lighting": "ParaView light kit, neutral warmth, LightScale=0.85",
                "material": "white Surface; Gouraud; Ambient=0.5, Diffuse=0.3, Specular=0",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
