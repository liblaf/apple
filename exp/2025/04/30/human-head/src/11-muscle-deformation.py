from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    solution: Path = cherries.data("solution-100/solution_000101.vtu")


def main(cfg: Config) -> None:
    solution: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.solution)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cherries.data("10-tetgen.vtu")
    )


if __name__ == "__main__":
    cherries.run(main)
