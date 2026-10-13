"""CPU-only regression and gradient parity tests for the v0.4.1 release."""
import tempfile
import unittest
from pathlib import Path

import torch

from perslay import (
    GridSampledGaussianPerslayPhi,
    IntegratedGaussianPerslayPhi,
    LearnableGridIntegratedGaussianPerslayPhi,
    LearnablePowerPerslayWeight,
    MLPPerslayWeight,
    ConstantPerslayWeight,
    Perslay,
)
from perslay.data import (
    PersistenceDiagramDataset,
    MultiPersistenceDiagramDataset,
    collate_persistence_diagrams,
    compute_diagram_bounds,
)
from perslay.models import FlattenRho, PerslayRegressor, make_perslay_phi


BOUNDS = [[0.0, 1.0], [0.0, 1.0]]


def make_phi(cls, *, checkpointed=False):
    return cls(
        image_size=[7, 5], image_bnds=BOUNDS,
        sigma_x=0.18, sigma_y=0.23,
        normalize=True, learnable_sigma=True,
        flatten_order="row-major", chunk_size=7,
        checkpoint_chunks=checkpointed,
    ).double()


class InputAndReductionTests(unittest.TestCase):
    def setUp(self):
        self.diagrams = torch.tensor([
            [[0.2, 0.4], [0.3, 0.8], [float("nan"), float("inf")]],
            [[0.2, 0.6], [float("inf"), float("nan")], [float("nan"), float("nan")]],
            [[float("nan"), float("inf")], [float("nan"), float("inf")], [float("nan"), float("inf")]],
        ], dtype=torch.float64)
        self.mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)

    def layer(self, op, weight=None):
        return Perslay(
            weight=weight or LearnablePowerPerslayWeight(),
            phi=make_phi(IntegratedGaussianPerslayPhi),
            perm_op=op, rho=FlattenRho()
        ).double()

    def test_masked_nonfinite_coordinates_never_contaminate_embeddings(self):
        for weight in [LearnablePowerPerslayWeight(), MLPPerslayWeight(BOUNDS)]:
            with self.subTest(weight=weight.__class__.__name__):
                layer = self.layer(torch.sum, weight=weight)
                result = layer(self.diagrams, self.mask)
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertTrue(torch.count_nonzero(result[2]) == 0)
                sanitized = torch.nan_to_num(self.diagrams, nan=0.0, posinf=0.0)
                torch.testing.assert_close(result, layer(sanitized, self.mask))

    def test_masked_mean_matches_manual_masked_sum_and_counts(self):
        layer = self.layer(torch.mean, weight=ConstantPerslayWeight())
        summed = self.layer(torch.sum, weight=ConstantPerslayWeight())
        summed.load_state_dict(layer.state_dict())
        output = layer(self.diagrams, self.mask)
        expected = summed(self.diagrams, self.mask) / self.mask.sum(dim=1).clamp_min(1)[:, None]
        torch.testing.assert_close(output, expected)
        self.assertEqual(torch.count_nonzero(output[2]), 0)

    def test_max_min_topk_handle_padding_and_all_empty(self):
        for op in (torch.max, torch.min, "topk2", "top2"):
            with self.subTest(operation=str(op)):
                layer = self.layer(op, weight=ConstantPerslayWeight())
                result = layer(self.diagrams, self.mask)
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertTrue(torch.count_nonzero(result[2]) == 0)
                unpadded = self.diagrams[:1, :2]
                expected = layer(unpadded)
                if op in (torch.max, torch.min):
                    torch.testing.assert_close(result[:1], expected)
                else:
                    # Padding must never outrank negative valid contributions.
                    torch.testing.assert_close(result[:1], expected)

    def test_reduction_respects_empty_point_axis(self):
        empty = torch.empty(2, 0, 2, dtype=torch.float64)
        mask = torch.empty(2, 0, dtype=torch.bool)
        for op in (torch.sum, torch.mean, torch.max, torch.min, "topk3"):
            with self.subTest(op=str(op)):
                result = self.layer(op, weight=ConstantPerslayWeight())(empty, mask)
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertEqual(torch.count_nonzero(result), 0)

    def test_mask_shape_validation(self):
        with self.assertRaises(ValueError):
            self.layer(torch.sum)(self.diagrams, torch.ones(3, 3))
        with self.assertRaises(ValueError):
            self.layer(torch.sum)(self.diagrams, torch.ones(2, 3, dtype=torch.bool))

    def test_csv_filtering_consistent_for_single_and_multi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "single").mkdir()
            (root / "single" / "a.csv").write_text("birth,death\n0,0.7\n0.3,inf\n0.8,0.5\nnan,0.3\n")
            (root / "single" / "b.csv").write_text("birth,death\n")
            (root / "target.csv").write_text("filename,target\na.csv,1\nb.csv,2\n")
            dataset = PersistenceDiagramDataset(root / "single", root / "target.csv")
            multi = MultiPersistenceDiagramDataset({"h0": root / "single"}, root / "target.csv")
            torch.testing.assert_close(dataset[0][0], multi[0][0]["h0"])
            self.assertEqual(tuple(dataset[0][0].shape), (1, 2))
            self.assertEqual(tuple(dataset[1][0].shape), (0, 2))
            x, mask, _, _ = collate_persistence_diagrams([dataset[1], dataset[1]])
            self.assertEqual(tuple(x.shape), (2, 1, 2))
            self.assertFalse(mask.any())
            self.assertTrue(torch.isfinite(torch.tensor(compute_diagram_bounds(dataset))).all())
            with self.assertRaises(ValueError):
                compute_diagram_bounds(dataset, persistence_min=1.0)


class CheckpointGradientTests(unittest.TestCase):
    def test_checkpoint_matches_output_and_all_gradients(self):
        torch.manual_seed(43)
        for cls in (GridSampledGaussianPerslayPhi, IntegratedGaussianPerslayPhi,
                    LearnableGridIntegratedGaussianPerslayPhi):
            with self.subTest(cls=cls.__name__):
                baseline = make_phi(cls)
                checked = make_phi(cls, checkpointed=True)
                if isinstance(baseline, LearnableGridIntegratedGaussianPerslayPhi):
                    with torch.no_grad():
                        baseline.x_width_logits.copy_(torch.randn_like(baseline.x_width_logits) * 0.2)
                        baseline.y_width_logits.copy_(torch.randn_like(baseline.y_width_logits) * 0.2)
                checked.load_state_dict(baseline.state_dict())
                d = torch.rand(2, 33, 2, dtype=torch.float64)
                d[..., 1] = d[..., 0] + 0.05 + 0.5 * d[..., 1]
                w = torch.randn(2, 33, dtype=torch.float64)
                d0, d1 = (d.detach().clone().requires_grad_(True) for _ in range(2))
                w0, w1 = (w.detach().clone().requires_grad_(True) for _ in range(2))
                y0 = baseline.weighted_sum(d0, w0)
                y1 = checked.weighted_sum(d1, w1)
                torch.testing.assert_close(y1, y0, atol=2e-12, rtol=2e-12)
                coefficients = torch.randn_like(y0)
                (y0 * coefficients).sum().backward()
                (y1 * coefficients).sum().backward()
                torch.testing.assert_close(d0.grad, d1.grad, atol=2e-12, rtol=2e-10)
                torch.testing.assert_close(w0.grad, w1.grad, atol=2e-12, rtol=2e-10)
                for (name0, param0), (name1, param1) in zip(
                    baseline.named_parameters(), checked.named_parameters()
                ):
                    self.assertEqual(name0, name1)
                    torch.testing.assert_close(param0.grad, param1.grad, atol=2e-11, rtol=2e-9)

    def test_checkpoint_improves_saved_tensor_footprint(self):
        torch.manual_seed(11)
        d = torch.rand(2, 1024, 2, dtype=torch.float32)
        d[..., 1] = d[..., 0] + 0.1
        w = torch.rand(2, 1024, dtype=torch.float32)
        counts = []
        for is_checkpointed in [False, True]:
            model = make_phi(LearnableGridIntegratedGaussianPerslayPhi,
                             checkpointed=is_checkpointed).float()
            saved_bytes = 0

            def save_hook(t):
                nonlocal saved_bytes
                saved_bytes += t.numel() * t.element_size()
                return t

            with torch.autograd.graph.saved_tensors_hooks(save_hook, lambda t: t):
                output = model.weighted_sum(d, w)
                loss = output.square().mean()
                loss.backward()
            counts.append(saved_bytes)
        self.assertLess(counts[1], 0.75 * counts[0], msg=f"saved bytes: {counts}")

    def test_requires_explicit_chunk_size_and_factory_propagation(self):
        with self.assertRaises(ValueError):
            IntegratedGaussianPerslayPhi(image_size=[2, 2], image_bnds=BOUNDS,
                                         sigma_x=0.2, checkpoint_chunks=True)
        model = PerslayRegressor(image_size=[2, 2], image_bnds=BOUNDS,
                                sigma_x=0.2, sigma_y=0.2,
                                feature_map_type="integrated_learnable_grid",
                                chunk_size=4, checkpoint_chunks=True)
        self.assertTrue(model.perslay.phi.checkpoint_chunks)
        self.assertTrue(make_perslay_phi("integrated", [2, 2], BOUNDS, 0.2, 0.2,
                                         True, True, "row-major", chunk_size=4,
                                         checkpoint_chunks=True).checkpoint_chunks)


class EndToEndCheckpointTests(unittest.TestCase):
    def test_checkpointed_regressor_trains_with_masked_invalid_padding(self):
        torch.manual_seed(12)
        configurations = dict(
            image_size=[6, 5], image_bnds=BOUNDS,
            sigma_x=0.2, sigma_y=0.24, hidden_dim=8,
            feature_map_type="integrated_learnable_grid",
            weight_type="learnable_power", normalize=True,
            flatten_order="row-major", chunk_size=3,
        )
        baseline = PerslayRegressor(**configurations).double()
        checkpointed = PerslayRegressor(
            **configurations, checkpoint_chunks=True
        ).double()
        checkpointed.load_state_dict(baseline.state_dict())
        points = torch.tensor([
            [[0.1, 0.4], [0.2, 0.7], [0.3, 0.8], [float("inf"), float("nan")],
             [float("nan"), float("nan")]],
            [[0.2, 0.5], [0.3, 0.7], [float("nan"), float("inf")],
             [float("inf"), float("nan")], [float("inf"), float("nan")]],
        ], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        baseline_output = baseline(points, mask)
        checkpointed_output = checkpointed(points, mask)
        torch.testing.assert_close(checkpointed_output, baseline_output)
        baseline_output.square().sum().backward()
        checkpointed_output.square().sum().backward()
        for a, b in zip(baseline.parameters(), checkpointed.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-7)

    def test_checkpoint_with_one_chunk_and_no_grad(self):
        phi = make_phi(IntegratedGaussianPerslayPhi, checkpointed=True)
        phi.chunk_size = 100
        d = torch.tensor([[[0.3, 0.7]]], dtype=torch.float64)
        w = torch.ones(1, 1, dtype=torch.float64)
        result = phi.weighted_sum(d, w)
        result.sum().backward()
        self.assertTrue(torch.isfinite(phi.log_sigma.grad).all())
        with torch.no_grad():
            inferred = phi.weighted_sum(d, w)
        torch.testing.assert_close(result, inferred)

    def test_checkpoint_graph_capture_torch_compile_eager(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        phi = make_phi(LearnableGridIntegratedGaussianPerslayPhi,
                       checkpointed=True).float()
        compiled = torch.compile(phi.weighted_sum, fullgraph=True, backend="eager")
        d = torch.tensor([[[0.2, 0.4], [0.3, 0.8]]])
        w = torch.ones(1, 2)
        torch.testing.assert_close(compiled(d, w), phi.weighted_sum(d, w))


if __name__ == "__main__":
    unittest.main()
