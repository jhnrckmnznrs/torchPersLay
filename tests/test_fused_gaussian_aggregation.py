import io
import math
import unittest

try:
    import torch

    from perslay import (
        ConstantPerslayWeight,
        GridSampledGaussianPerslayPhi,
        IntegratedGaussianPerslayPhi,
        LearnablePowerPerslayWeight,
        Perslay,
    )
    from perslay.models import FlattenRho
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


def materialized_weighted_sum(phi, diagrams, weights):
    values, _ = phi(diagrams)
    expanded = weights
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return torch.sum(values * expanded, dim=1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FusedGaussianOutputTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        birth = torch.rand(3, 17, dtype=torch.float64) * 1.6 - 0.8
        persistence = torch.rand(3, 17, dtype=torch.float64) * 2.0
        self.diagrams = torch.stack([birth, birth + persistence], dim=-1)
        self.weights = torch.rand(3, 17, dtype=torch.float64)

    def assert_fused_matches_materialized(self, phi):
        phi = phi.double()
        expected = materialized_weighted_sum(phi, self.diagrams, self.weights)
        actual = phi.weighted_sum(self.diagrams, self.weights)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_grid_sampled_fused_matches_materialized_row_major(self):
        self.assert_fused_matches_materialized(
            GridSampledGaussianPerslayPhi(
                image_size=[7, 5],
                image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
                sigma_x=0.31,
                sigma_y=0.47,
                normalize=True,
                learnable_sigma=True,
                flatten_order="row-major",
            )
        )

    def test_grid_sampled_fused_matches_materialized_legacy_order(self):
        self.assert_fused_matches_materialized(
            GridSampledGaussianPerslayPhi(
                image_size=[7, 5],
                image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
                sigma_x=0.31,
                sigma_y=0.47,
                normalize=False,
                learnable_sigma=False,
                flatten_order="legacy-x-major",
            )
        )

    def test_integrated_fused_matches_materialized_normalized(self):
        self.assert_fused_matches_materialized(
            IntegratedGaussianPerslayPhi(
                image_size=[7, 5],
                image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
                sigma_x=0.31,
                sigma_y=0.47,
                normalize=True,
                learnable_sigma=True,
                flatten_order="row-major",
            )
        )

    def test_integrated_fused_matches_materialized_unnormalized_legacy(self):
        self.assert_fused_matches_materialized(
            IntegratedGaussianPerslayPhi(
                image_size=[7, 5],
                image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
                sigma_x=0.31,
                sigma_y=0.47,
                normalize=False,
                learnable_sigma=False,
                flatten_order="legacy-x-major",
            )
        )

    def test_chunked_matches_unchunked_for_both_representations(self):
        constructors = (GridSampledGaussianPerslayPhi, IntegratedGaussianPerslayPhi)
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__):
                common = dict(
                    image_size=[9, 6],
                    image_bnds=[[-1.0, 1.0], [0.0, 2.0]],
                    sigma_x=0.29,
                    sigma_y=0.41,
                    normalize=True,
                    learnable_sigma=True,
                    flatten_order="row-major",
                )
                unchunked = constructor(**common).double()
                chunked = constructor(**common, chunk_size=4).double()
                chunked.load_state_dict(unchunked.state_dict())
                expected = unchunked.weighted_sum(self.diagrams, self.weights)
                actual = chunked.weighted_sum(self.diagrams, self.weights)
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_empty_diagram_returns_zero_surface(self):
        diagrams = torch.empty((2, 0, 2))
        weights = torch.empty((2, 0))
        for phi in (
            GridSampledGaussianPerslayPhi(
                image_size=[4, 3],
                image_bnds=[[0.0, 1.0], [0.0, 1.0]],
                sigma_x=0.2,
                learnable_sigma=False,
            ),
            IntegratedGaussianPerslayPhi(
                image_size=[4, 3],
                image_bnds=[[0.0, 1.0], [0.0, 1.0]],
                sigma_x=0.2,
                learnable_sigma=False,
            ),
        ):
            output = phi.weighted_sum(diagrams, weights)
            self.assertEqual(tuple(output.shape), (2, 4, 3, 1) if phi.flatten_order == "legacy-x-major" else (2, 3, 4, 1))
            torch.testing.assert_close(output, torch.zeros_like(output))

    def test_chunk_size_validation(self):
        common = dict(
            image_size=[2, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
        )
        with self.assertRaises(ValueError):
            GridSampledGaussianPerslayPhi(**common, chunk_size=0)
        with self.assertRaises(TypeError):
            IntegratedGaussianPerslayPhi(**common, chunk_size=1.5)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FusedGaussianGradientTests(unittest.TestCase):
    def _compare_fast_and_reference_gradients(self, phi_constructor):
        torch.manual_seed(4321)
        birth = torch.rand(2, 11, dtype=torch.float64) * 0.8
        persistence = 0.05 + torch.rand(2, 11, dtype=torch.float64) * 0.9
        base_diagrams = torch.stack([birth, birth + persistence], dim=-1)

        common = dict(
            image_size=[5, 4],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.22,
            sigma_y=0.31,
            normalize=True,
            learnable_sigma=True,
            flatten_order="row-major",
            chunk_size=3,
        )

        fast = Perslay(
            weight=LearnablePowerPerslayWeight(init_scale=0.7, init_power=1.2),
            phi=phi_constructor(**common),
            perm_op=torch.sum,
            rho=FlattenRho(),
        ).double()
        reference = Perslay(
            weight=LearnablePowerPerslayWeight(init_scale=0.7, init_power=1.2),
            phi=phi_constructor(**common),
            perm_op=lambda tensor, dim: torch.sum(tensor, dim=dim),
            rho=FlattenRho(),
        ).double()
        reference.load_state_dict(fast.state_dict())

        fast_diagrams = base_diagrams.clone().requires_grad_(True)
        ref_diagrams = base_diagrams.clone().requires_grad_(True)
        mask = torch.tensor(
            [[True] * 9 + [False, False], [True] * 7 + [False] * 4]
        )

        fast_output = fast(fast_diagrams, mask=mask)
        ref_output = reference(ref_diagrams, mask=mask)
        torch.testing.assert_close(fast_output, ref_output, rtol=5e-12, atol=5e-12)

        coefficient = torch.linspace(
            0.1, 1.0, fast_output.numel(), dtype=torch.float64
        ).reshape_as(fast_output)
        (fast_output * coefficient).sum().backward()
        (ref_output * coefficient).sum().backward()

        torch.testing.assert_close(
            fast_diagrams.grad, ref_diagrams.grad, rtol=2e-10, atol=2e-11
        )
        torch.testing.assert_close(
            fast.phi.log_sigma.grad,
            reference.phi.log_sigma.grad,
            rtol=2e-10,
            atol=2e-11,
        )
        torch.testing.assert_close(
            fast.weight.raw_scale.grad,
            reference.weight.raw_scale.grad,
            rtol=2e-10,
            atol=2e-11,
        )
        torch.testing.assert_close(
            fast.weight.raw_power.grad,
            reference.weight.raw_power.grad,
            rtol=2e-10,
            atol=2e-11,
        )

    def test_grid_sampled_fast_path_gradients_match_materialized_path(self):
        self._compare_fast_and_reference_gradients(GridSampledGaussianPerslayPhi)

    def test_integrated_fast_path_gradients_match_materialized_path(self):
        self._compare_fast_and_reference_gradients(IntegratedGaussianPerslayPhi)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NumericalAndCheckpointTests(unittest.TestCase):
    def test_integrated_positive_tail_mass_does_not_cancel_to_zero(self):
        edges = torch.tensor([8.0, 9.0], dtype=torch.float32)
        means = torch.tensor([[0.0]], dtype=torch.float32)
        sigma = torch.tensor(1.0, dtype=torch.float32)
        mass = IntegratedGaussianPerslayPhi._normal_interval_masses(
            edges, means, sigma
        ).squeeze()

        expected = 0.5 * (
            math.erfc(8.0 / math.sqrt(2.0))
            - math.erfc(9.0 / math.sqrt(2.0))
        )
        self.assertGreater(float(mass), 0.0)
        self.assertAlmostEqual(float(mass) / expected, 1.0, places=5)

    def test_integrated_interval_mass_matches_float64_reference(self):
        edges = torch.tensor(
            [-6.0, -2.0, -0.25, 0.0, 0.4, 2.5, 6.0],
            dtype=torch.float64,
        )
        means = torch.tensor([[-1.2, 0.0, 1.7]], dtype=torch.float64)
        sigma = torch.tensor(0.73, dtype=torch.float64)

        actual = IntegratedGaussianPerslayPhi._normal_interval_masses(
            edges, means, sigma
        )

        scale = sigma * math.sqrt(2.0)
        z = (edges - means.unsqueeze(-1)) / scale
        expected = 0.5 * (torch.erf(z[..., 1:]) - torch.erf(z[..., :-1]))
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-15)

    def test_integrated_tiny_central_mass_does_not_cancel_in_float32(self):
        edges = torch.tensor([-1e-8, 1e-8], dtype=torch.float32)
        means = torch.tensor([[0.0]], dtype=torch.float32)
        sigma = torch.tensor(1.0, dtype=torch.float32)
        mass = IntegratedGaussianPerslayPhi._normal_interval_masses(
            edges, means, sigma
        ).squeeze()
        expected = math.erf(1e-8 / math.sqrt(2.0))
        self.assertGreater(float(mass), 0.0)
        self.assertAlmostEqual(float(mass) / expected, 1.0, places=5)

    def test_integrated_interval_mass_has_correct_gradient_at_pixel_edge(self):
        edges = torch.tensor([0.0, 1.0], dtype=torch.float64)
        sigma = torch.tensor(1.0, dtype=torch.float64)

        mean = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
        actual = IntegratedGaussianPerslayPhi._normal_interval_masses(
            edges, mean, sigma
        ).sum()
        actual.backward()

        reference_mean = torch.tensor(
            [[0.0]], dtype=torch.float64, requires_grad=True
        )
        z = (edges - reference_mean.unsqueeze(-1)) / (sigma * math.sqrt(2.0))
        reference = 0.5 * (torch.erf(z[..., 1]) - torch.erf(z[..., 0]))
        reference.sum().backward()

        torch.testing.assert_close(actual.detach(), reference.detach().sum())
        torch.testing.assert_close(
            mean.grad, reference_mean.grad, rtol=2e-13, atol=2e-14
        )

    def test_sampled_checkpoint_round_trip(self):
        phi = GridSampledGaussianPerslayPhi(
            image_size=[6, 4],
            image_bnds=[[-2.0, 2.0], [0.0, 3.0]],
            sigma_x=0.4,
            sigma_y=0.7,
            normalize=True,
            learnable_sigma=True,
        )
        self.assertTrue(phi.grid_x.is_contiguous())
        self.assertTrue(phi.grid_y.is_contiguous())

        buffer = io.BytesIO()
        torch.save(phi.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer)

        restored = GridSampledGaussianPerslayPhi(
            image_size=[6, 4],
            image_bnds=[[-2.0, 2.0], [0.0, 3.0]],
            sigma_x=1.0,
            sigma_y=1.0,
            normalize=True,
            learnable_sigma=True,
        )
        restored.load_state_dict(state)

        diagrams = torch.tensor([[[0.2, 0.9], [-0.5, 1.2]]])
        torch.testing.assert_close(phi(diagrams)[0], restored(diagrams)[0])

    def test_perslay_automatically_uses_fused_sum_and_respects_mask(self):
        phi = IntegratedGaussianPerslayPhi(
            image_size=[3, 2],
            image_bnds=[[0.0, 1.0], [0.0, 1.0]],
            sigma_x=0.2,
            sigma_y=0.3,
            normalize=True,
            learnable_sigma=False,
        )
        layer = Perslay(
            weight=ConstantPerslayWeight(),
            phi=phi,
            perm_op=torch.sum,
            rho=FlattenRho(),
        )
        diagrams = torch.tensor([[[0.2, 0.6], [0.9, 1.0]]])
        mask = torch.tensor([[True, False]])
        output = layer(diagrams, mask=mask)
        expected = phi.weighted_sum(diagrams, torch.tensor([[1.0, 0.0]])).reshape(1, -1)
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
