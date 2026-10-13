import sys
import unittest
from pathlib import Path

import yaml


EXPERIMENTS_DIRECTORY = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIRECTORY))

import cross_validate_bone_modulus_perslay as experiment  # noqa: E402


class FeatureMapConfigurationTests(unittest.TestCase):
    def test_example_configs_differ_only_in_feature_map_and_output(self):
        config_directory = EXPERIMENTS_DIRECTORY / "configs"
        with (config_directory / "grid_sampled_regression.yaml").open(
            encoding="utf-8"
        ) as handle:
            sampled = yaml.safe_load(handle)
        with (config_directory / "integrated_regression.yaml").open(
            encoding="utf-8"
        ) as handle:
            integrated = yaml.safe_load(handle)

        self.assertEqual(sampled["data"], integrated["data"])
        self.assertEqual(sampled["training"], integrated["training"])
        sampled_model = dict(sampled["model"])
        integrated_model = dict(integrated["model"])
        sampled_type = sampled_model.pop("feature_map_type")
        integrated_type = integrated_model.pop("feature_map_type")
        self.assertEqual(sampled_type, "grid_sampled")
        self.assertEqual(integrated_type, "integrated")
        self.assertEqual(sampled_model, integrated_model)

    def test_representation_defaults_are_explicit(self):
        sampled = experiment.resolve_feature_map_settings(
            "grid_sampled",
            gaussian_normalization="representation-default",
            learnable_sigma=True,
            flatten_order="representation-default",
        )
        integrated = experiment.resolve_feature_map_settings(
            "integrated",
            gaussian_normalization="representation-default",
            learnable_sigma=True,
            flatten_order="representation-default",
        )

        self.assertFalse(sampled.normalize_gaussian)
        self.assertEqual(sampled.flatten_order, "legacy-x-major")
        self.assertTrue(integrated.normalize_gaussian)
        self.assertEqual(integrated.flatten_order, "row-major")

    def test_matched_normalization_and_layout_can_be_requested(self):
        for feature_map_type in ("grid_sampled", "integrated"):
            settings = experiment.resolve_feature_map_settings(
                feature_map_type,
                gaussian_normalization="normalized",
                learnable_sigma=True,
                flatten_order="row-major",
            )
            self.assertTrue(settings.normalize_gaussian)
            self.assertEqual(settings.flatten_order, "row-major")

    def test_fixed_legacy_bandwidth_is_rejected(self):
        with self.assertRaisesRegex(
            experiment.core.ExperimentError, "requires learnable bandwidths"
        ):
            experiment.resolve_feature_map_settings(
                "legacy_gaussian",
                gaussian_normalization="representation-default",
                learnable_sigma=False,
                flatten_order="representation-default",
            )

    def test_parser_accepts_multiple_feature_maps(self):
        arguments = experiment.parser().parse_args(
            [
                "--feature-map-types",
                "grid_sampled",
                "integrated",
                "--gaussian-normalization",
                "normalized",
                "--flatten-order",
                "row-major",
                "--no-learnable-sigma",
            ]
        )
        self.assertEqual(arguments.feature_map_types, ["grid_sampled", "integrated"])
        self.assertEqual(arguments.gaussian_normalization, "normalized")
        self.assertEqual(arguments.flatten_order, "row-major")
        self.assertFalse(arguments.learnable_sigma)


class RepresentationAwareAggregationTests(unittest.TestCase):
    @staticmethod
    def metric_row(feature_map_type, rmse):
        return {
            "repeat": 0,
            "outer_fold": 0,
            "model": "perslay_variant_residual",
            "dimension_set": "h2",
            "target_mode": "variant_residual",
            "feature_map_type": feature_map_type,
            "split": "test",
            "variant_scope": "all",
            "number_of_samples": 30,
            "mae_MPa": rmse * 0.8,
            "rmse_MPa": rmse,
            "r2": 0.5,
            "best_epoch": 10,
        }

    def test_metric_summary_does_not_pool_feature_maps(self):
        rows = [
            self.metric_row("grid_sampled", 1.2),
            self.metric_row("integrated", 1.4),
        ]
        summary = experiment.aggregate_metric_rows(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["feature_map_type"] for row in summary},
            {"grid_sampled", "integrated"},
        )

    def test_each_feature_map_is_compared_with_the_same_baseline(self):
        baseline = {
            **self.metric_row("none", 1.5),
            "model": "variant_bvtv",
            "dimension_set": "none",
            "target_mode": "baseline",
        }
        rows = [
            baseline,
            self.metric_row("grid_sampled", 1.2),
            self.metric_row("integrated", 1.4),
        ]
        improvements = experiment.improvement_rows(rows)
        selected = [
            row for row in improvements if row["baseline"] == "variant_bvtv"
        ]
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {row["feature_map_type"] for row in selected},
            {"grid_sampled", "integrated"},
        )


if __name__ == "__main__":
    unittest.main()
