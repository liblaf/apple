import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import pyvista.examples
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    duration: float = 5.0
    fps: float = 30.0
    period: float = 1.0

    # material properties
    density: float = 1e3
    E: float = 7e2  # Young's modulus
    nu: float = 0.4  # Poisson's ratio

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1.0 / self.fps


def main(cfg: Config) -> None:
    geometry: apple.Geometry = gen_geometry(cfg)
    scene: apple.Scene = gen_scene(cfg, geometry)
    writer = melon.SeriesWriter("data/examples/dynamics/bunny.vtu.series")
    writer.append(geometry.mesh)

    for it in grapes.track(range(1, cfg.n_frames + 1), description="Frames"):
        time: float = it * cfg.time_step
        dirichlet_values: Float[np.ndarray, " dirichlet"] = gen_dirichlet_values(
            geometry, time=time, period=cfg.period
        )
        field: apple.Field = scene.fields["displacement"]
        field = field.with_dirichlet(
            dirichlet_index=field.dirichlet_index, dirichlet_values=dirichlet_values
        )
        scene.fields["displacement"] = field
        solution: apple.OptimizeResult = scene.solve()
        ic(solution)

        scene = scene.step(solution["x"])
        geometries: dict[str, apple.Geometry] = scene.make_geometries()
        writer.append(geometries["bunny"].mesh)


def gen_geometry(cfg: Config, lr: float = 0.05) -> apple.Geometry:
    surface: pv.PolyData = pyvista.examples.download_bunny(load=True)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    geometry = apple.Geometry(mesh=mesh, id="bunny")
    geometry.density = cfg.density
    return geometry


def gen_dirichlet(
    geometry: apple.Geometry,
) -> tuple[Integer[np.ndarray, " dirichlet"], Float[np.ndarray, " dirichlet"]]:
    mesh: pv.UnstructuredGrid = geometry.mesh
    dirichlet_mask: Bool[np.ndarray, " points"] = np.zeros((mesh.n_points,), dtype=bool)
    dirichlet_values: Float[np.ndarray, " points 3"] = np.zeros(
        (mesh.n_points, 3), dtype=float
    )
    y_min: float
    y_max: float
    _x_min, _x_max, y_min, y_max, _z_min, _z_max = mesh.bounds
    y_length: float = y_max - y_min
    dirichlet_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 1] < y_min + 0.05 * y_length
    )
    dirichlet_values[dirichlet_mask] = np.asarray([0.0, 0.0, 0.0])
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = dirichlet_values

    dirichlet_mask: Bool[np.ndarray, "points 3"] = einops.repeat(
        dirichlet_mask, " points -> (points 3)"
    )
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    (dirichlet_index,) = np.nonzero(dirichlet_mask)
    return dirichlet_index, dirichlet_values.ravel()[dirichlet_index]


def gen_dirichlet_values(
    geometry: apple.Geometry, time: float = 0.0, period: float = 1.0
) -> Float[np.ndarray, " dirichlet"]:
    mesh: pv.UnstructuredGrid = geometry.mesh
    dirichlet_mask: Bool[np.ndarray, " points"] = mesh.point_data["dirichlet-mask"]
    dirichlet_values: Float[np.ndarray, "points 3"] = mesh.point_data[
        "dirichlet-values"
    ]
    n_dirichlet: int = np.count_nonzero(dirichlet_mask)
    x_min: float
    x_max: float
    x_min, x_max, _y_min, _y_max, _z_min, _z_max = mesh.bounds
    x_length: float = x_max - x_min
    x_translate: float = 0.3 * x_length * np.sin(2 * np.pi * time / period)
    dirichlet_values: Float[np.ndarray, "dirichlet 3"] = np.broadcast_to(
        [x_translate, 0.0, 0.0], (n_dirichlet, 3)
    )
    mesh.point_data["dirichlet-values"][dirichlet_mask] = dirichlet_values
    return dirichlet_values.ravel()


def gen_scene(cfg: Config, geometry: apple.Geometry) -> apple.Scene:
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    dirichlet_values: Float[np.ndarray, " dirichlet"]
    dirichlet_index, dirichlet_values = gen_dirichlet(geometry)
    domain: apple.Domain = apple.Domain.from_geometry(geometry)
    field: apple.Field = (
        apple.Field.from_domain(domain=domain, id="displacement")
        .with_dirichlet(
            dirichlet_index=dirichlet_index, dirichlet_values=dirichlet_values
        )
        .with_free_values(0.0)
        .with_velocities(0.0)
        .with_forces(0.0)
        .step()
    )
    scene = apple.Scene(time_step=jnp.asarray(cfg.time_step))
    scene.add_field(field)

    lambda_: float
    mu: float
    lambda_, mu = apple.utils.lame_params(E=cfg.E, nu=cfg.nu)
    elasticity = apple.energy.elastic.PhaceStatic(
        field_id=field.id, mu=jnp.asarray(mu), lambda_=jnp.asarray(lambda_)
    )
    scene.add_energy(elasticity)
    inertia = apple.energy.Inertia(field_id=field.id, mass=field.domain.point_mass)
    scene.add_energy(inertia)
    return scene


if __name__ == "__main__":
    cherries.run(main, play=True)
