import math
import unittest

try:
    import torch

    from perslay import (
        ConstantPerslayWeight,
        GridSampledGaussianPerslayPhi,
        IntegratedGaussianPerslayPhi,
        LinearRampPerslayWeight,
        Perslay,
    )
    from perslay.models import FlattenRho, PerslayRegressor
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GridSampledGaussianTests(unittest.TestCase):
    def test_point_at_only_pixel_centre_has_unit_unnormalized_height(self):
        phi = GridSampledGaussianPerslayPhi(
            image_size=[1, 1],
            image_bnds=[[-0.5, 0.5], [0.0, 1.0]],
            sigma_x=0.5,
            sigma_y=2.0,
            normalize=False,
            learnable_sigma=False,
        )
        diagrams = torch.tensor([[[0.0, 0.5]]])
        values, shape = phi(diagrams)
        self.assertEqual(shape, (1, 1, 1))
        torch.testing.assert_close(values.squeeze(), torch.tensor(1.0))
        self.assertFalse(phi.log_sigma.requires_grad)

    def test_normalized_height_has_density_factor(self):
        phi = GridSampledGaussianPerslayPhi(
            image_size=[1, 1],
            image_bnds=[[-0.5, 0.5], [0.0, 1.0]],
            sigma_x=0.5,
            sigma_y=2.0,
            normalize=True,
            learnable_sigma=False,
        )
        values, _ = phi(torch.tensor([[[0.0, 0.5]]]))
        expected = torch.tensor(1.0 / (2.0 * math.pi))
        torch.testing.assert_close(values.squeeze(), expected)

    def test_sum_and_mask_reproduce_constant_weight_sampled_surface(self):
        layer = Perslay(
            weight=ConstantPerslayWeight(),
            phi=GridSampledGaussianPerslayPhi(
                image_size=[1, 1],
                image_bnds=[[-0.5, 0.5], [0.0, 1.0]],
                sigma_x=1.0,
                sigma_y=1.0,
                normalize=False,
                learnable_sigma=False,
            ),
            perm_op=torch.sum,
            rho=FlattenRho(),
        )
        diagrams = torch.tensor([[[0.0, 0.5], [0.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        output = layer(diagrams, mask=mask)
        torch.testing.assert_close(output, torch.ones((1, 1)))

    def test_flatten_order_changes_only_coordinate_order(self):
        arguments = dict(
            image_size=[2, 3],
            image_bnds=[[0.0, 2.0], [0.0, 3.0]],
            sigma_x=0.7,
            sigma_y=0.9,
            normalize=False,
            learnable_sigma=False,
        )
        diagrams = torch.tensor([[[0.25, 1.5]]])
        row_values, row_shape = GridSampledGaussianPerslayPhi(
            **arguments, flatten_order="row-major"
        )(diagrams)
        legacy_values, legacy_shape = GridSampledGaussianPerslayPhi(
            **arguments, flatten_order="legacy-x-major"
        )(diagrams)

        self.assertEqual(row_shape, (3, 2, 1))
        self.assertEqual(legacy_shape, (2, 3, 1))
        torch.testing.assert_close(
            row_values.squeeze(-1).transpose(-2, -1),
            legacy_values.squeeze(-1),
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class IntegratedGaussianTests(unittest.TestCase):
    def test_one_pixel_equals_exact_normal_probability_mass(self):
        phi = IntegratedGaussianPerslayPhi(
            image_size=[1, 1],
            image_bnds=[[-1.0, 1.0], [-1.0, 1.0]],
            sigma_x=1.0,
            sigma_y=1.0,
            normalize=True,
            learnable_sigma=False,
            flatten_order="row-major",
        )
        values, _ = phi(torch.tensor([[[0.0, 0.0]]]))
        one_dimensional = math.erf(1.0 / math.sqrt(2.0))
        expected = torch.tensor(one_dimensional**2)
        torch.testing.assert_close(values.squeeze(), expected, rtol=1e-6, atol=1e-7)

    def test_integrated_map_is_differentiable_in_bandwidth(self):
        phi = IntegratedGaussianPerslayPhi(
            image_size=[3, 3],
            image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
            sigma_x=0.5,
            sigma_y=0.5,
            normalize=True,
            learnable_sigma=True,
        )
        values, _ = phi(torch.tensor([[[0.0, 1.0]]]))
        values.square().sum().backward()
        self.assertIsNotNone(phi.log_sigma.grad)
        self.assertTrue(torch.isfinite(phi.log_sigma.grad).all())


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WeightAndModelTests(unittest.TestCase):
    def test_linear_ramp_values(self):
        diagrams = torch.tensor([[[0.0, 0.0], [0.0, 0.5], [0.0, 2.0]]])
        weights = LinearRampPerslayWeight(threshold=1.0)(diagrams)
        torch.testing.assert_close(weights, torch.tensor([[0.0, 0.5, 1.0]]))

    def test_regressor_builds_grid_sampled_feature_map(self):
        model = PerslayRegressor(
            image_size=[2, 3],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.3,
            hidden_dim=4,
            weight_type="linear_ramp",
            weight_threshold=0.1,
            feature_map_type="grid_sampled",
            normalize=False,
            learnable_sigma=False,
            flatten_order="legacy-x-major",
        )
        diagrams = torch.tensor(
            [[[0.1, 0.4], [0.2, 0.8]], [[0.3, 0.7], [0.0, 0.0]]]
        )
        mask = torch.tensor([[True, True], [True, False]])
        output = model(diagrams, mask=mask)
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
