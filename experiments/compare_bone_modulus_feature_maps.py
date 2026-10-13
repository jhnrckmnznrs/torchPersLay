#!/usr/bin/env python3
"""Paired analysis of PersLay feature maps on identical grouped CV folds."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = prediction - target
    denominator = float(np.sum(np.square(target - target.mean())))
    return {
        "number_of_samples": int(target.size),
        "mae_MPa": float(np.mean(np.abs(residual))),
        "rmse_MPa": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": (
            float(1.0 - np.sum(np.square(residual)) / denominator)
            if denominator > 0.0
            else float("nan")
        ),
    }


def average_repeated_oof(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "feature_map_type",
        "sample_id",
        "structure_id",
        "variant",
    ]
    averaged = rows.groupby(group_columns, as_index=False).agg(
        target_MPa=("target_MPa", "first"),
        prediction_MPa=("prediction_MPa", "mean"),
        retained_density_bvtv=("retained_density_bvtv", "first"),
        number_of_oof_predictions=("prediction_MPa", "size"),
    )
    return averaged


def validate_matched_predictions(
    rows: pd.DataFrame,
    feature_map_types: tuple[str, ...],
) -> None:
    duplicate = rows.duplicated(
        ["feature_map_type", "repeat", "sample_id"], keep=False
    )
    if duplicate.any():
        raise SystemExit(
            "Duplicate predictions were found for a feature map, repeat, and "
            "sample. Use a clean grouped-CV output directory."
        )

    reference = rows[rows["feature_map_type"] == feature_map_types[0]][
        ["repeat", "outer_fold", "sample_id", "structure_id", "variant"]
    ].sort_values(["repeat", "sample_id"], ignore_index=True)
    for feature_map_type in feature_map_types[1:]:
        candidate = rows[rows["feature_map_type"] == feature_map_type][
            ["repeat", "outer_fold", "sample_id", "structure_id", "variant"]
        ].sort_values(["repeat", "sample_id"], ignore_index=True)
        if not reference.equals(candidate):
            raise SystemExit(
                "Feature maps do not contain identical repeated test assignments. "
                "They cannot be compared as paired experiments."
            )


def paired_table(
    averaged: pd.DataFrame,
    first: str,
    second: str,
) -> pd.DataFrame:
    identifying_columns = ["sample_id", "structure_id", "variant"]
    left = averaged[averaged["feature_map_type"] == first][
        identifying_columns + ["target_MPa", "prediction_MPa"]
    ].rename(columns={"prediction_MPa": f"prediction__{first}"})
    right = averaged[averaged["feature_map_type"] == second][
        identifying_columns + ["target_MPa", "prediction_MPa"]
    ].rename(
        columns={
            "target_MPa": "target_MPa_second",
            "prediction_MPa": f"prediction__{second}",
        }
    )
    paired = left.merge(right, on=identifying_columns, validate="one_to_one")
    if not np.allclose(
        paired["target_MPa"], paired["target_MPa_second"], rtol=0.0, atol=0.0
    ):
        raise SystemExit("Target values differ between feature-map outputs.")
    return paired.drop(columns="target_MPa_second")


def bootstrap_metric_difference(
    paired: pd.DataFrame,
    *,
    first: str,
    second: str,
    scope: str,
    number_of_replicates: int,
    random_seed: int,
) -> dict[str, float | int | str]:
    selected = paired if scope == "all" else paired[paired["variant"] == scope]
    structures = np.asarray(sorted(selected["structure_id"].unique()), dtype=object)
    if structures.size == 0:
        raise SystemExit(f"No structures are available for variant scope {scope!r}.")

    target = selected["target_MPa"].to_numpy(float)
    first_prediction = selected[f"prediction__{first}"].to_numpy(float)
    second_prediction = selected[f"prediction__{second}"].to_numpy(float)
    first_metrics = regression_metrics(target, first_prediction)
    second_metrics = regression_metrics(target, second_prediction)

    rng = np.random.default_rng(random_seed)
    rmse_differences = np.empty(number_of_replicates, dtype=float)
    mae_differences = np.empty(number_of_replicates, dtype=float)
    for index in range(number_of_replicates):
        sampled_structures = rng.choice(
            structures, size=structures.size, replace=True
        )
        bootstrap_sample = pd.concat(
            [
                selected[selected["structure_id"] == structure_id]
                for structure_id in sampled_structures
            ],
            ignore_index=True,
        )
        bootstrap_target = bootstrap_sample["target_MPa"].to_numpy(float)
        first_bootstrap_metrics = regression_metrics(
            bootstrap_target,
            bootstrap_sample[f"prediction__{first}"].to_numpy(float),
        )
        second_bootstrap_metrics = regression_metrics(
            bootstrap_target,
            bootstrap_sample[f"prediction__{second}"].to_numpy(float),
        )
        rmse_differences[index] = (
            first_bootstrap_metrics["rmse_MPa"]
            - second_bootstrap_metrics["rmse_MPa"]
        )
        mae_differences[index] = (
            first_bootstrap_metrics["mae_MPa"]
            - second_bootstrap_metrics["mae_MPa"]
        )

    return {
        "first_feature_map": first,
        "second_feature_map": second,
        "variant_scope": scope,
        "number_of_structures": int(structures.size),
        "number_of_samples": int(selected.shape[0]),
        "first_rmse_MPa": first_metrics["rmse_MPa"],
        "second_rmse_MPa": second_metrics["rmse_MPa"],
        "rmse_difference_first_minus_second_MPa": (
            first_metrics["rmse_MPa"] - second_metrics["rmse_MPa"]
        ),
        "rmse_difference_ci025_MPa": float(np.quantile(rmse_differences, 0.025)),
        "rmse_difference_ci975_MPa": float(np.quantile(rmse_differences, 0.975)),
        "bootstrap_probability_first_has_lower_rmse": float(
            np.mean(rmse_differences < 0.0)
        ),
        "first_mae_MPa": first_metrics["mae_MPa"],
        "second_mae_MPa": second_metrics["mae_MPa"],
        "mae_difference_first_minus_second_MPa": (
            first_metrics["mae_MPa"] - second_metrics["mae_MPa"]
        ),
        "mae_difference_ci025_MPa": float(np.quantile(mae_differences, 0.025)),
        "mae_difference_ci975_MPa": float(np.quantile(mae_differences, 0.975)),
    }


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Compare PersLay Gaussian feature maps using paired repeated OOF "
            "predictions and structure-level bootstrap intervals."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument("--cv-output", type=Path, required=True)
    argument_parser.add_argument("--output-dir", type=Path, default=None)
    argument_parser.add_argument("--model", default="perslay_variant_residual")
    argument_parser.add_argument("--dimension-set", default="h0_h2")
    argument_parser.add_argument(
        "--feature-map-types",
        nargs="+",
        default=["grid_sampled", "integrated"],
    )
    argument_parser.add_argument("--bootstrap", type=int, default=5000)
    argument_parser.add_argument("--seed", type=int, default=2026)
    return argument_parser


def main() -> int:
    arguments = parser().parse_args()
    if arguments.bootstrap < 1:
        raise SystemExit("--bootstrap must be positive.")
    feature_map_types = tuple(dict.fromkeys(arguments.feature_map_types))
    if len(feature_map_types) < 2:
        raise SystemExit("Select at least two feature maps for comparison.")

    prediction_path = arguments.cv_output / "predictions.csv"
    if not prediction_path.is_file():
        raise SystemExit(f"Missing {prediction_path}")
    output_directory = arguments.output_dir or (
        arguments.cv_output / "feature_map_comparison" / arguments.dimension_set
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(prediction_path)
    if "feature_map_type" not in predictions.columns:
        raise SystemExit(
            "predictions.csv has no feature_map_type column. Rerun cross-validation "
            "with script version 2 before comparing representations."
        )
    selected = predictions[
        (predictions["split"] == "test")
        & (predictions["model"] == arguments.model)
        & (predictions["dimension_set"] == arguments.dimension_set)
        & (predictions["feature_map_type"].isin(feature_map_types))
    ].copy()
    available = set(selected["feature_map_type"].astype(str).unique())
    missing = [name for name in feature_map_types if name not in available]
    if missing:
        raise SystemExit(f"Missing test predictions for feature maps: {missing}")

    validate_matched_predictions(selected, feature_map_types)
    averaged = average_repeated_oof(selected)
    averaged.to_csv(output_directory / "oof_average_by_feature_map.csv", index=False)

    metric_rows: list[dict[str, float | int | str]] = []
    scopes = ["all"] + sorted(averaged["variant"].astype(str).unique().tolist())
    for feature_map_type in feature_map_types:
        map_rows = averaged[averaged["feature_map_type"] == feature_map_type]
        for scope in scopes:
            scoped = map_rows if scope == "all" else map_rows[map_rows["variant"] == scope]
            metric_rows.append(
                {
                    "feature_map_type": feature_map_type,
                    "variant_scope": scope,
                    **regression_metrics(
                        scoped["target_MPa"].to_numpy(float),
                        scoped["prediction_MPa"].to_numpy(float),
                    ),
                }
            )
    pd.DataFrame(metric_rows).to_csv(
        output_directory / "feature_map_metrics.csv", index=False
    )

    comparison_rows: list[dict[str, float | int | str]] = []
    for pair_index, (first, second) in enumerate(
        itertools.combinations(feature_map_types, 2)
    ):
        paired = paired_table(averaged, first, second)
        paired.to_csv(
            output_directory / f"paired_predictions__{first}__vs__{second}.csv",
            index=False,
        )
        for scope_index, scope in enumerate(scopes):
            comparison_rows.append(
                bootstrap_metric_difference(
                    paired,
                    first=first,
                    second=second,
                    scope=scope,
                    number_of_replicates=arguments.bootstrap,
                    random_seed=arguments.seed + 100 * pair_index + scope_index,
                )
            )
    pd.DataFrame(comparison_rows).to_csv(
        output_directory / "paired_feature_map_comparisons.csv", index=False
    )

    configuration = {
        "cv_output": str(arguments.cv_output.resolve()),
        "model": arguments.model,
        "dimension_set": arguments.dimension_set,
        "feature_map_types": list(feature_map_types),
        "bootstrap_replicates": arguments.bootstrap,
        "random_seed": arguments.seed,
        "independent_resampling_unit": "structure_id",
    }
    (output_directory / "comparison_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote feature-map comparison to {output_directory}")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nPaired feature-map differences (first minus second):")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
