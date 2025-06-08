import jax
import jax.numpy as jnp
import pyvista as pv
import warp as wp

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    duration: float = 5.0
    fps: float = 30.0

    # material properties
    density: float = 1e3
    E: float = 5e4  # Young's modulus
    nu: float = 0.4  # Poisson's ratio
    threshold: float = 1e-3

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1.0 / self.fps


def main(cfg: Config) -> None:
    geometry: apple.Geometry = gen_geometry(cfg)
    collision: apple.CollisionRigidSoft = gen_collision()
    scene: apple.Scene = gen_scene(cfg, geometry)
    scene = scene.replace(
        optimizer=apple.PNCG(d_hat=cfg.threshold, maxiter=10**3, tol=1e-5)
    )

    writer = melon.SeriesWriter("data/examples/dynamics/collision.vtu.series")

    def callback(result: apple.OptimizeResult) -> None:
        fields: dict[str, apple.Field] = scene.make_fields(result["x"])
        displacements: jax.Array = collision.resolve(
            points=fields["displacement"].points + fields["displacement"].values
        )
        result["x"] += displacements.ravel()

    for it in grapes.track(range(cfg.n_frames), description="Frames"):
        result: apple.OptimizeResult = scene.solve(callback=callback)
        # fields: dict[str, apple.Field] = scene.make_fields(result["x"])
        # displacements: jax.Array = collision.resolve(
        #     points=fields["displacement"].points + fields["displacement"].values
        # )
        # ic(jnp.abs(displacements).max())
        # result["x"] += displacements.ravel()
        scene = scene.step(result["x"])
        geometries: dict[str, apple.Geometry] = scene.make_geometries()
        writer.append(geometries["bunny"].mesh)


def as_warp_mesh(mesh_pv: pv.PolyData) -> wp.Mesh:
    return wp.Mesh(
        wp.from_numpy(mesh_pv.points, dtype=wp.vec3),
        wp.from_numpy(mesh_pv.regular_faces.ravel(), dtype=wp.int32),
    )


def gen_collision() -> apple.CollisionRigidSoft:
    mesh_pv: pv.PolyData = melon.load_poly_data(pv.examples.download_bunny(load=False))
    # mesh_pv = melon.mesh_fix(mesh_pv, check=False)
    mesh_pv.flip_faces(inplace=True)
    mesh_pv.translate([0.0, -0.28, 0.0], inplace=True)
    # mesh_pv: pv.PolyData = pv.Box((-0.1, 0.1, -0.1, 0, -0.1, 0.1), quads=False)
    # mesh_pv.translate([0.0, -0.1, 0.0], inplace=True)
    melon.save("data/examples/dynamics/collision-rigid.vtp", mesh_pv)
    mesh_wp: wp.Mesh = as_warp_mesh(mesh_pv)
    return apple.CollisionRigidSoft(mesh_wp=mesh_wp)


def gen_geometry(cfg: Config, lr: float = 0.05) -> apple.Geometry:
    surface: pv.PolyData = pv.examples.download_bunny(load=True)
    ic(surface)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    geometry = apple.Geometry(mesh=mesh, id="bunny")
    geometry.density = cfg.density
    return geometry


def gen_scene(cfg: Config, geometry: apple.Geometry) -> apple.Scene:
    domain: apple.Domain = apple.Domain.from_geometry(geometry)
    field: apple.Field = (
        apple.Field.from_domain(domain=domain, id="displacement")
        .with_free_values(0.0)
        .with_velocities(0.0)
        .with_forces(0.0)
        .with_forces(domain.point_mass[:, None] * jnp.asarray([0.0, -9.81, 0.0]))
        .step()
    )
    scene = apple.Scene(time_step=jnp.asarray(cfg.time_step))
    scene.add_field(field)

    lambda_: float
    mu: float
    lambda_, mu = apple.utils.lame_params(E=cfg.E, nu=cfg.nu)
    elasticity = apple.PhaceStatic(
        field_id=field.id, mu=jnp.asarray(mu), lambda_=jnp.asarray(lambda_)
    )
    scene.add_energy(elasticity)
    inertia = apple.Inertia(field_id=field.id, mass=field.domain.point_mass)
    scene.add_energy(inertia)
    # gravity = apple.Gravity(field_id=field.id, mass=field.domain.point_mass)
    # scene.add_energy(gravity)
    return scene


if __name__ == "__main__":
    cherries.run(main, play=True)
