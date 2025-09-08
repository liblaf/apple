import warp as wp

from liblaf.apple.warp.sim.energy._energy import Energy


class Elastic(Energy):
    def energy_density(self, F: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def energy_density_jac(self, F: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def energy_density_hess_diag(self, F: wp.array, output: wp.array) -> None:
        raise NotImplementedError

    def energy_density_hess_quad(
        self, F: wp.array, p: wp.array, output: wp.array
    ) -> None:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(self, F: wp.array, output: wp.array) -> None:
        raise NotImplementedError
