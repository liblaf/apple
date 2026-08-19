import torch

from liblaf.apple import common


def test_lame_converter_plane_stress() -> None:
    lambda_3d, mu_3d = common.lame_converter(0.2, 0.49)
    lambda_, mu = common.lame_converter_plane_stress(0.2, 0.49)

    torch.testing.assert_close(
        lambda_3d, torch.tensor(3.288590604), rtol=1.0e-6, atol=0.0
    )
    torch.testing.assert_close(mu_3d, torch.tensor(0.067114094), rtol=1.0e-6, atol=0.0)
    torch.testing.assert_close(
        lambda_, torch.tensor(0.128964337), rtol=1.0e-6, atol=0.0
    )
    torch.testing.assert_close(mu, torch.tensor(0.067114094), rtol=1.0e-6, atol=0.0)
    torch.testing.assert_close(
        lambda_, 2.0 * lambda_3d * mu_3d / (lambda_3d + 2.0 * mu_3d)
    )
    torch.testing.assert_close(lambda_ / (lambda_ + 2.0 * mu), torch.tensor(0.49))


def test_lame_converter_plane_stress_preserves_shape() -> None:
    E = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    nu = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)

    lambda_, mu = common.lame_converter_plane_stress(E, nu)

    torch.testing.assert_close(lambda_, E * nu / (1.0 - nu.square()))
    torch.testing.assert_close(mu, E / (2.0 * (1.0 + nu)))
    assert lambda_.shape == E.shape
    assert mu.shape == E.shape
