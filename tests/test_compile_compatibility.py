import unittest

try:
    import torch

    from perslay import (
        GridSampledGaussianPerslayPhi,
        IntegratedGaussianPerslayPhi,
    )
    from perslay.models import PerslayRegressor
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None or not hasattr(torch, "compile"), "torch.compile unavailable")
class TorchCompileCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2468)
        birth = torch.rand(2, 16)
        persistence = 0.05 + torch.rand(2, 16)
        self.diagrams = torch.stack((birth, birth + persistence), dim=-1)
        self.weights = torch.rand(2, 16)

    def test_explicit_gaussian_sums_capture_as_one_graph(self):
        for constructor in (
            GridSampledGaussianPerslayPhi,
            IntegratedGaussianPerslayPhi,
        ):
            with self.subTest(constructor=constructor.__name__):
                phi = constructor(
                    image_size=[6, 4],
                    image_bnds=[[0.0, 1.0], [0.0, 1.2]],
                    sigma_x=0.12,
                    sigma_y=0.18,
                    normalize=True,
                    learnable_sigma=True,
                    flatten_order="row-major",
                )
                compiled = torch.compile(
                    phi.weighted_sum, backend="eager", fullgraph=True
                )
                expected = phi.weighted_sum(self.diagrams, self.weights)
                actual = compiled(self.diagrams, self.weights)
                torch.testing.assert_close(actual, expected)

    def test_integrated_regressor_compiles_and_backpropagates(self):
        model = PerslayRegressor(
            image_size=[6, 4],
            image_bnds=[[0.0, 1.0], [0.0, 1.2]],
            sigma_x=0.12,
            sigma_y=0.18,
            hidden_dim=8,
            weight_type="learnable_power",
            feature_map_type="integrated",
            normalize=True,
            learnable_sigma=True,
            flatten_order="row-major",
        )
        mask = torch.ones(self.diagrams.shape[:2], dtype=torch.bool)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        output = compiled(self.diagrams, mask)
        output.square().mean().backward()
        self.assertEqual(tuple(output.shape), (2,))
        for parameter in model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
