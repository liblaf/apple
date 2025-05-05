import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    E: float = 1e5
    nu: float = 0.47


def main(cfg: Config) -> None:
    lmbda, mu = apple.constitution.E_nu_to_lame(cfg.E, cfg.nu)
    ic(lmbda)
    ic(mu)


if __name__ == "__main__":
    cherries.run(main)
