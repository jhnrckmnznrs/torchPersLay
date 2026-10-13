import sys
import unittest
from pathlib import Path

import pandas as pd


EXPERIMENTS_DIRECTORY = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIRECTORY))

import compare_bone_modulus_feature_maps as comparison  # noqa: E402


class FeatureMapComparisonTests(unittest.TestCase):
    @staticmethod
    def prediction_rows():
        rows = []
        targets = {"structure_0000": 4.0, "structure_0001": 8.0}
        for repeat in (0, 1):
            for structure_index, (structure_id, target) in enumerate(targets.items()):
                for feature_map_type, error in (
                    ("grid_sampled", 0.5),
                    ("integrated", 1.0),
                ):
                    rows.append(
                        {
                            "feature_map_type": feature_map_type,
                            "repeat": repeat,
                            "outer_fold": structure_index,
                            "sample_id": f"baseline__{structure_id}",
                            "structure_id": structure_id,
                            "variant": "baseline",
                            "target_MPa": target,
                            "prediction_MPa": target + error,
                            "retained_density_bvtv": 0.2,
                        }
                    )
        return pd.DataFrame(rows)

    def test_repeated_predictions_are_matched_and_averaged(self):
        rows = self.prediction_rows()
        comparison.validate_matched_predictions(
            rows, ("grid_sampled", "integrated")
        )
        averaged = comparison.average_repeated_oof(rows)
        self.assertEqual(len(averaged), 4)
        self.assertTrue((averaged["number_of_oof_predictions"] == 2).all())

    def test_paired_bootstrap_preserves_structure_as_resampling_unit(self):
        averaged = comparison.average_repeated_oof(self.prediction_rows())
        paired = comparison.paired_table(
            averaged, "grid_sampled", "integrated"
        )
        result = comparison.bootstrap_metric_difference(
            paired,
            first="grid_sampled",
            second="integrated",
            scope="all",
            number_of_replicates=25,
            random_seed=7,
        )
        self.assertEqual(result["number_of_structures"], 2)
        self.assertLess(result["rmse_difference_first_minus_second_MPa"], 0.0)
        self.assertEqual(
            result["bootstrap_probability_first_has_lower_rmse"], 1.0
        )

    def test_mismatched_assignments_are_rejected(self):
        rows = self.prediction_rows().iloc[:-1].copy()
        with self.assertRaisesRegex(SystemExit, "identical repeated test assignments"):
            comparison.validate_matched_predictions(
                rows, ("grid_sampled", "integrated")
            )


if __name__ == "__main__":
    unittest.main()
