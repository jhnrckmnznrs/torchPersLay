import unittest

import torch

from perslay import (
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
)
from perslay.models import PerslayRegressor, make_perslay_phi


class LearnableGridIntegratedGaussianTests(unittest.TestCase):
    def test_uniform_initialization_matches_fixed_grid_exactly(self):
        arguments = dict(
            image_size=[4, 3],
            image_bnds=[[-1.0, 2.0], [0.0, 3.0]],
            sigma_x=0.35,
            sigma_y=0.55,
            normalize=True,
            learnable_sigma=False,
            flatten_order="row-major",
        )
        fixed = IntegratedGaussianPerslayPhi(**arguments).double()
        learned = LearnableGridIntegratedGaussianPerslayPhi(
            **arguments, min_cell_fraction=0.05
        ).double()
        diagrams = torch.tensor(
            [[[-0.2, 0.4], [0.7, 1.9], [1.4, 2.0]]], dtype=torch.float64
        )
        weights = torch.tensor([[0.5, 1.25, 0.8]], dtype=torch.float64)

        fixed_values = fixed.weighted_sum(diagrams, weights)
        learned_values = learned.weighted_sum(diagrams, weights)
        torch.testing.assert_close(learned_values, fixed_values, rtol=1e-12, atol=1e-12)

        x_edges, y_edges = learned.grid_edges()
        torch.testing.assert_close(x_edges, fixed.x_edges, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(y_edges, fixed.y_edges, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(
            learned.grid_regularization(), torch.zeros((), dtype=torch.float64)
        )

    def test_edges_remain_ordered_and_respect_minimum_width(self):
        phi = LearnableGridIntegratedGaussianPerslayPhi(
            image_size=[3, 2],
            image_bnds=[[2.0, 8.0], [-1.0, 3.0]],
            sigma_x=0.5,
            sigma_y=0.5,
            learnable_sigma=False,
            min_cell_fraction=0.12,
        )
        with torch.no_grad():
            phi.x_width_logits.copy_(torch.tensor([30.0, -20.0, -40.0]))
            phi.y_width_logits.copy_(torch.tensor([-25.0, 25.0]))

        x_widths, y_widths = phi.grid_width_fractions()
        x_edges, y_edges = phi.grid_edges()

        self.assertEqual(x_edges[0].item(), 2.0)
        self.assertEqual(x_edges[-1].item(), 8.0)
        self.assertEqual(y_edges[0].item(), -1.0)
        self.assertEqual(y_edges[-1].item(), 3.0)
        self.assertTrue(torch.all(x_edges[1:] > x_edges[:-1]))
        self.assertTrue(torch.all(y_edges[1:] > y_edges[:-1]))

        self.assertGreaterEqual(x_widths.min().item(), 0.12 / 3.0 - 1e-7)
        self.assertGreaterEqual(y_widths.min().item(), 0.12 / 2.0 - 1e-7)
        torch.testing.assert_close(x_widths.sum(), torch.tensor(1.0))
        torch.testing.assert_close(y_widths.sum(), torch.tensor(1.0))

    def test_grid_gradient_matches_centered_finite_difference(self):
        phi = LearnableGridIntegratedGaussianPerslayPhi(
            image_size=[3, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.16,
            sigma_y=0.21,
            normalize=True,
            learnable_sigma=False,
            min_cell_fraction=0.02,
        ).double()
        with torch.no_grad():
            phi.x_width_logits.copy_(torch.tensor([0.2, -0.15, 0.1], dtype=torch.float64))
            phi.y_width_logits.copy_(torch.tensor([-0.1, 0.25], dtype=torch.float64))

        diagrams = torch.tensor(
            [[[0.18, 0.52], [0.43, 0.91], [0.77, 0.93]]], dtype=torch.float64
        )
        weights = torch.tensor([[0.9, 0.6, 1.1]], dtype=torch.float64)

        values = phi.weighted_sum(diagrams, weights).reshape(-1)
        coefficients = torch.tensor(
            [0.7, -0.2, 0.4, -0.3, 0.6, -0.5], dtype=torch.float64
        )
        objective = torch.dot(values, coefficients)
        objective.backward()
        autograd_value = phi.x_width_logits.grad[0].item()

        epsilon = 1e-6
        original = phi.x_width_logits.detach().clone()

        def evaluate(offset):
            with torch.no_grad():
                phi.x_width_logits.copy_(original)
                phi.x_width_logits[0].add_(offset)
            return torch.dot(
                phi.weighted_sum(diagrams, weights).reshape(-1), coefficients
            ).item()

        finite_difference = (evaluate(epsilon) - evaluate(-epsilon)) / (2.0 * epsilon)
        with torch.no_grad():
            phi.x_width_logits.copy_(original)

        self.assertAlmostEqual(autograd_value, finite_difference, places=7)

    def test_toy_optimization_moves_boundary_to_informative_location(self):
        torch.manual_seed(0)
        phi = LearnableGridIntegratedGaussianPerslayPhi(
            image_size=[2, 1],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.08,
            sigma_y=0.15,
            normalize=True,
            learnable_sigma=False,
            min_cell_fraction=0.0,
        )
        left = torch.tensor([[[0.2, 0.7]]])
        right = torch.tensor([[[0.4, 0.9]]])
        weights = torch.ones((1, 1))
        optimizer = torch.optim.Adam([phi.x_width_logits], lr=0.08)

        initial_boundary = phi.grid_edges()[0][1].item()
        with torch.no_grad():
            initial_separation = torch.square(
                phi.weighted_sum(left, weights) - phi.weighted_sum(right, weights)
            ).sum().item()

        for _ in range(180):
            optimizer.zero_grad()
            left_features = phi.weighted_sum(left, weights)
            right_features = phi.weighted_sum(right, weights)
            loss = -torch.square(left_features - right_features).sum()
            loss.backward()
            optimizer.step()

        learned_boundary = phi.grid_edges()[0][1].item()
        with torch.no_grad():
            final_separation = torch.square(
                phi.weighted_sum(left, weights) - phi.weighted_sum(right, weights)
            ).sum().item()

        self.assertAlmostEqual(initial_boundary, 0.5, places=6)
        self.assertAlmostEqual(learned_boundary, 0.3, places=3)
        self.assertGreater(final_separation, 20.0 * initial_separation)

    def test_regularization_loss_is_zero_uniform_and_positive_after_deformation(self):
        phi = LearnableGridIntegratedGaussianPerslayPhi(
            image_size=[4, 3],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.2,
            learnable_sigma=False,
            grid_regularization_strength=0.25,
        )
        torch.testing.assert_close(phi.regularization_loss(), torch.tensor(0.0))
        with torch.no_grad():
            phi.x_width_logits[0] = 2.0
        self.assertGreater(phi.grid_regularization().item(), 0.0)
        torch.testing.assert_close(
            phi.regularization_loss(), 0.25 * phi.grid_regularization()
        )

    def test_factory_and_regressor_expose_adaptive_grid(self):
        phi = make_perslay_phi(
            feature_map_type="integrated_learnable_grid",
            image_size=[3, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.2,
            normalize=True,
            learnable_sigma=True,
            flatten_order="row-major",
            min_cell_fraction=0.1,
            grid_regularization_strength=0.03,
        )
        self.assertIsInstance(phi, LearnableGridIntegratedGaussianPerslayPhi)

        model = PerslayRegressor(
            image_size=[3, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.2,
            feature_map_type="integrated_learnable_grid",
            normalize=True,
            flatten_order="row-major",
            min_cell_fraction=0.1,
            grid_regularization_strength=0.03,
        )
        diagrams = torch.tensor([[[0.2, 0.5], [0.55, 0.9]]])
        output = model(diagrams)
        self.assertEqual(output.shape, (1,))
        self.assertEqual(model.regularization_loss().ndim, 0)

    def test_checkpoint_round_trip_preserves_learned_grid(self):
        arguments = dict(
            image_size=[3, 2],
            image_bnds=[[0.0, 2.0], [0.0, 1.5]],
            sigma_x=0.25,
            sigma_y=0.35,
            normalize=True,
            learnable_sigma=True,
            min_cell_fraction=0.05,
        )
        original = LearnableGridIntegratedGaussianPerslayPhi(**arguments)
        with torch.no_grad():
            original.x_width_logits.copy_(torch.tensor([0.8, -0.3, 0.1]))
            original.y_width_logits.copy_(torch.tensor([-0.4, 0.6]))
            original.log_sigma.copy_(torch.log(torch.tensor([0.31, 0.47])))

        restored = LearnableGridIntegratedGaussianPerslayPhi(**arguments)
        restored.load_state_dict(original.state_dict())
        original_edges = original.grid_edges()
        restored_edges = restored.grid_edges()
        torch.testing.assert_close(restored_edges[0], original_edges[0])
        torch.testing.assert_close(restored_edges[1], original_edges[1])

        diagrams = torch.tensor([[[0.2, 0.7], [1.1, 1.5]]])
        weights = torch.tensor([[0.7, 1.2]])
        torch.testing.assert_close(
            restored.weighted_sum(diagrams, weights),
            original.weighted_sum(diagrams, weights),
        )

    def test_chunked_and_unchunked_adaptive_aggregation_match(self):
        arguments = dict(
            image_size=[5, 4],
            image_bnds=[[-1.0, 2.0], [0.0, 2.5]],
            sigma_x=0.3,
            sigma_y=0.4,
            normalize=True,
            learnable_sigma=False,
            min_cell_fraction=0.03,
        )
        unchunked = LearnableGridIntegratedGaussianPerslayPhi(
            **arguments, chunk_size=None
        ).double()
        chunked = LearnableGridIntegratedGaussianPerslayPhi(
            **arguments, chunk_size=3
        ).double()
        with torch.no_grad():
            logits_x = torch.tensor([0.4, -0.2, 0.1, -0.4, 0.3], dtype=torch.float64)
            logits_y = torch.tensor([-0.3, 0.5, 0.2, -0.1], dtype=torch.float64)
            unchunked.x_width_logits.copy_(logits_x)
            unchunked.y_width_logits.copy_(logits_y)
            chunked.x_width_logits.copy_(logits_x)
            chunked.y_width_logits.copy_(logits_y)

        torch.manual_seed(12)
        birth = -0.5 + 2.0 * torch.rand((2, 11), dtype=torch.float64)
        persistence = 0.1 + 1.6 * torch.rand((2, 11), dtype=torch.float64)
        diagrams = torch.stack((birth, birth + persistence), dim=-1)
        weights = torch.rand((2, 11), dtype=torch.float64)
        torch.testing.assert_close(
            chunked.weighted_sum(diagrams, weights),
            unchunked.weighted_sum(diagrams, weights),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_adaptive_weighted_sum_supports_fullgraph_compile(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")
        phi = LearnableGridIntegratedGaussianPerslayPhi(
            image_size=[3, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.25,
            normalize=True,
            learnable_sigma=True,
            min_cell_fraction=0.05,
        )
        diagrams = torch.tensor([[[0.2, 0.5], [0.6, 0.9]]])
        weights = torch.tensor([[1.0, 0.7]])
        compiled = torch.compile(phi.weighted_sum, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(diagrams, weights), phi.weighted_sum(diagrams, weights)
        )

    def test_invalid_adaptive_grid_configuration_is_rejected(self):
        common = dict(
            image_size=[2, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.2,
        )
        for value in (-0.1, 1.0, float("inf")):
            with self.subTest(min_cell_fraction=value):
                with self.assertRaises(ValueError):
                    LearnableGridIntegratedGaussianPerslayPhi(
                        **common, min_cell_fraction=value
                    )
        with self.assertRaises(ValueError):
            LearnableGridIntegratedGaussianPerslayPhi(
                **common, grid_regularization_strength=-1.0
            )


if __name__ == "__main__":
    unittest.main()
