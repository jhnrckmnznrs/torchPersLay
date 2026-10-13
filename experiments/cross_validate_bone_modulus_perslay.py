#!/usr/bin/env python3
"""Repeated grouped cross-validation for bone-modulus PersLay models.

This script is designed for the baseline, uniformly thinned, and locally
weakened structures in ``data/``. All samples sharing the same
``structure_XXXX`` identifier are kept in the same split, so sibling variants
never leak across training, validation, and testing.

The script compares three non-neural baselines:

1. training-mean modulus;
2. one pooled BV/TV power law;
3. a stronger variant-aware BV/TV power law with separate intercepts and slopes
   for the selected structural variants.

It can train the following PersLay models:

``raw``
    Predict apparent Young's modulus directly from persistence diagrams.

``power_residual``
    Predict the log-modulus residual left by the pooled BV/TV power law.

``variant_residual``
    Predict the log-modulus residual left by the variant-aware BV/TV model.
    This is the recommended main model because it asks topology to explain only
    architecture-dependent variation beyond BV/TV and protocol/variant effects.

Exact duplicate compression from ``train_bone_modulus_perslay.py`` is reused.
No persistence points are subsampled.

Matched sampled-versus-integrated experiment
---------------------------------------------

python experiments/cross_validate_bone_modulus_perslay.py \\
    --data-root data \\
    --output-dir outputs/bone_modulus_feature_map_comparison \\
    --variants baseline uniform weakened \\
    --folds 5 \\
    --repeats 10 \\
    --models variant_residual \\
    --dimension-sets h0 h2 h0_h2 \\
    --feature-map-types grid_sampled integrated \\
    --gaussian-normalization normalized \\
    --flatten-order row-major \\
    --learnable-sigma \\
    --weight-type normalized_learnable_power \\
    --device auto

Fast smoke test
---------------

python experiments/cross_validate_bone_modulus_perslay.py \\
    --data-root data \\
    --output-dir outputs/bone_modulus_grouped_cv_smoke \\
    --folds 2 --repeats 1 --epochs 3 --patience 2 \\
    --models variant_residual --dimension-sets h0 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import train_bone_modulus_perslay as core
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_bone_modulus_perslay as core


SCRIPT_VERSION = 2
SPLITS = ("train", "validation", "test")
DIMENSION_SETS: dict[str, tuple[int, ...]] = {
    "h0": (0,),
    "h1": (1,),
    "h2": (2,),
    "h0_h1": (0, 1),
    "h0_h2": (0, 2),
    "h1_h2": (1, 2),
    "h0_h1_h2": (0, 1, 2),
}
MODEL_MODES = ("raw", "power_residual", "variant_residual")
BASELINE_MODELS = ("mean", "bvtv_power", "variant_bvtv")
FEATURE_MAP_TYPES = ("legacy_gaussian", "grid_sampled", "integrated")
GAUSSIAN_NORMALIZATION_MODES = (
    "representation-default",
    "normalized",
    "unnormalized",
)
FLATTEN_ORDER_MODES = (
    "representation-default",
    "legacy-x-major",
    "row-major",
)


@dataclass(frozen=True)
class LinearLogModel:
    """Linear model for log(E) using log(BV/TV) and optional variant effects."""

    model_name: str
    variants: tuple[str, ...]
    reference_variant: str
    coefficient_names: tuple[str, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class FoldDefinition:
    repeat: int
    fold: int
    training_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass
class TrainedRun:
    result: dict[str, Any]
    metrics_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    history_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class FeatureMapSettings:
    """Fully resolved settings for one Gaussian PersLay feature map."""

    feature_map_type: str
    normalize_gaussian: bool
    learnable_sigma: bool
    flatten_order: str


def resolve_feature_map_settings(
    feature_map_type: str,
    *,
    gaussian_normalization: str,
    learnable_sigma: bool,
    flatten_order: str,
) -> FeatureMapSettings:
    """Resolve representation-aware defaults into an explicit configuration."""
    if feature_map_type not in FEATURE_MAP_TYPES:
        raise core.ExperimentError(
            f"Unknown feature-map type {feature_map_type!r}."
        )
    if gaussian_normalization not in GAUSSIAN_NORMALIZATION_MODES:
        raise core.ExperimentError(
            f"Unknown Gaussian-normalization mode {gaussian_normalization!r}."
        )
    if flatten_order not in FLATTEN_ORDER_MODES:
        raise core.ExperimentError(f"Unknown flatten-order mode {flatten_order!r}.")
    if feature_map_type == "legacy_gaussian" and not learnable_sigma:
        raise core.ExperimentError(
            "legacy_gaussian requires learnable bandwidths. Use grid_sampled or "
            "integrated for a fixed-bandwidth experiment."
        )

    default_normalization = feature_map_type == "integrated"
    if gaussian_normalization == "representation-default":
        normalize_gaussian = default_normalization
    else:
        normalize_gaussian = gaussian_normalization == "normalized"

    default_flatten_order = (
        "row-major" if feature_map_type == "integrated" else "legacy-x-major"
    )
    resolved_flatten_order = (
        default_flatten_order
        if flatten_order == "representation-default"
        else flatten_order
    )
    return FeatureMapSettings(
        feature_map_type=feature_map_type,
        normalize_gaussian=normalize_gaussian,
        learnable_sigma=learnable_sigma,
        flatten_order=resolved_flatten_order,
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def metric_to_dict(metric: core.Metrics) -> dict[str, Any]:
    value = asdict(metric)
    if not math.isfinite(value["r2"]):
        value["r2"] = None
    return value


def stable_signature(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def make_outer_folds(
    structure_ids: Sequence[str],
    *,
    folds: int,
    repeats: int,
    inner_validation_fraction: float,
    random_seed: int,
) -> list[FoldDefinition]:
    if folds < 2:
        raise core.ExperimentError("--folds must be at least 2.")
    if folds > len(structure_ids):
        raise core.ExperimentError("--folds cannot exceed the number of structures.")
    if repeats < 1:
        raise core.ExperimentError("--repeats must be positive.")
    if not (0.0 < inner_validation_fraction < 0.5):
        raise core.ExperimentError(
            "--inner-validation-fraction must be between 0 and 0.5."
        )

    definitions: list[FoldDefinition] = []
    sorted_groups = np.asarray(sorted(structure_ids), dtype=object)

    for repeat in range(repeats):
        rng = np.random.default_rng(random_seed + repeat)
        shuffled = sorted_groups.copy()
        rng.shuffle(shuffled)
        outer_test_folds = np.array_split(shuffled, folds)

        for fold, test_array in enumerate(outer_test_folds):
            test_groups = tuple(sorted(str(value) for value in test_array.tolist()))
            test_set = set(test_groups)
            development = [
                str(value) for value in shuffled.tolist() if str(value) not in test_set
            ]

            inner_rng = np.random.default_rng(
                random_seed + 100_000 * (repeat + 1) + fold
            )
            inner_rng.shuffle(development)
            validation_count = max(
                1, int(round(inner_validation_fraction * len(development)))
            )
            if validation_count >= len(development):
                validation_count = len(development) - 1
            validation_groups = tuple(sorted(development[:validation_count]))
            training_groups = tuple(sorted(development[validation_count:]))

            if not training_groups or not validation_groups or not test_groups:
                raise core.ExperimentError(
                    f"Invalid grouped fold repeat={repeat}, fold={fold}."
                )

            definitions.append(
                FoldDefinition(
                    repeat=repeat,
                    fold=fold,
                    training_groups=training_groups,
                    validation_groups=validation_groups,
                    test_groups=test_groups,
                )
            )
    return definitions


def fold_assignments(
    records: Sequence[core.SampleRecord],
    definition: FoldDefinition,
) -> tuple[dict[int, str], dict[str, list[int]]]:
    train = set(definition.training_groups)
    validation = set(definition.validation_groups)
    test = set(definition.test_groups)
    if train & validation or train & test or validation & test:
        raise core.ExperimentError("A structure appears in more than one fold subset.")

    assignments: dict[int, str] = {}
    indices: dict[str, list[int]] = {split: [] for split in SPLITS}
    for record in records:
        if record.structure_id in train:
            split = "train"
        elif record.structure_id in validation:
            split = "validation"
        elif record.structure_id in test:
            split = "test"
        else:
            raise core.ExperimentError(
                f"Structure {record.structure_id} was omitted from a fold."
            )
        assignments[record.index] = split
        indices[split].append(record.index)

    splits_by_structure: dict[str, set[str]] = {}
    for record in records:
        splits_by_structure.setdefault(record.structure_id, set()).add(
            assignments[record.index]
        )
    leaked = {
        key: sorted(value) for key, value in splits_by_structure.items() if len(value) != 1
    }
    if leaked:
        raise core.ExperimentError(f"Grouped split leakage detected: {leaked}")
    return assignments, indices


def variant_design_matrix(
    log_bvtv: np.ndarray,
    sample_variants: Sequence[str],
    variant_order: Sequence[str],
    *,
    include_variant_effects: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = [np.ones_like(log_bvtv), log_bvtv]
    names = ["intercept", "log_bvtv"]
    if include_variant_effects:
        reference = variant_order[0]
        for variant in variant_order[1:]:
            indicator = np.asarray(
                [1.0 if value == variant else 0.0 for value in sample_variants],
                dtype=np.float64,
            )
            columns.extend((indicator, indicator * log_bvtv))
            names.extend(
                (
                    f"indicator__{variant}_vs_{reference}",
                    f"log_bvtv_x_indicator__{variant}_vs_{reference}",
                )
            )
    return np.column_stack(columns), tuple(names)


def fit_log_bvtv_model(
    records: Sequence[core.SampleRecord],
    training_indices: Sequence[int],
    *,
    variants: Sequence[str],
    include_variant_effects: bool,
) -> LinearLogModel:
    variant_order = tuple(variants)
    log_bvtv = np.log(
        np.asarray([records[index].bvtv for index in training_indices], dtype=np.float64)
    )
    log_target = np.log(
        np.asarray([records[index].target for index in training_indices], dtype=np.float64)
    )
    sample_variants = [records[index].variant for index in training_indices]
    design, names = variant_design_matrix(
        log_bvtv,
        sample_variants,
        variant_order,
        include_variant_effects=include_variant_effects,
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, log_target, rcond=None)
    if rank < design.shape[1]:
        raise core.ExperimentError(
            "The BV/TV baseline design matrix is rank deficient. Check that each "
            "training fold contains every selected variant."
        )
    return LinearLogModel(
        model_name=("variant_bvtv" if include_variant_effects else "bvtv_power"),
        variants=variant_order,
        reference_variant=variant_order[0],
        coefficient_names=names,
        coefficients=tuple(float(value) for value in coefficients),
    )


def predict_log_bvtv_model(
    model: LinearLogModel,
    records: Sequence[core.SampleRecord],
) -> np.ndarray:
    log_bvtv = np.log(np.asarray([record.bvtv for record in records], dtype=np.float64))
    sample_variants = [record.variant for record in records]
    design, names = variant_design_matrix(
        log_bvtv,
        sample_variants,
        model.variants,
        include_variant_effects=(model.model_name == "variant_bvtv"),
    )
    if names != model.coefficient_names:
        raise core.ExperimentError("BV/TV baseline design columns changed unexpectedly.")
    return design @ np.asarray(model.coefficients, dtype=np.float64)


def predictions_from_array(
    records: Sequence[core.SampleRecord],
    indices: Sequence[int],
    predictions: np.ndarray,
) -> list[tuple[int, float, float]]:
    return [
        (index, records[index].target, float(predictions[index])) for index in indices
    ]


def calculate_metrics_rows(
    rows_by_split: Mapping[str, Sequence[tuple[int, float, float]]],
    records: Sequence[core.SampleRecord],
    *,
    repeat: int,
    fold: int,
    model: str,
    dimension_set: str,
    target_mode: str,
    feature_map_settings: FeatureMapSettings | None,
    best_epoch: int | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in SPLITS:
        rows = list(rows_by_split[split])
        scopes = ["all"] + sorted({records[index].variant for index, _, _ in rows})
        for scope in scopes:
            selected = (
                rows
                if scope == "all"
                else [row for row in rows if records[row[0]].variant == scope]
            )
            metric = core.regression_metrics(selected)
            output.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "model": model,
                    "dimension_set": dimension_set,
                    "target_mode": target_mode,
                    "feature_map_type": (
                        "none"
                        if feature_map_settings is None
                        else feature_map_settings.feature_map_type
                    ),
                    "normalize_gaussian": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.normalize_gaussian
                    ),
                    "learnable_sigma": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.learnable_sigma
                    ),
                    "flatten_order": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.flatten_order
                    ),
                    "split": split,
                    "variant_scope": scope,
                    "number_of_samples": metric.number_of_samples,
                    "mae_MPa": metric.mae,
                    "rmse_MPa": metric.rmse,
                    "r2": metric.r2,
                    "best_epoch": "" if best_epoch is None else best_epoch,
                }
            )
    return output


def calculate_prediction_rows(
    rows_by_split: Mapping[str, Sequence[tuple[int, float, float]]],
    records: Sequence[core.SampleRecord],
    *,
    repeat: int,
    fold: int,
    model: str,
    dimension_set: str,
    target_mode: str,
    feature_map_settings: FeatureMapSettings | None,
    base_log_predictions: np.ndarray | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in SPLITS:
        for index, truth, prediction in rows_by_split[split]:
            record = records[index]
            base_log = (
                float("nan")
                if base_log_predictions is None
                else float(base_log_predictions[index])
            )
            actual_log_residual = (
                float("nan") if base_log_predictions is None else math.log(truth) - base_log
            )
            predicted_log_residual = (
                float("nan")
                if base_log_predictions is None
                else math.log(max(prediction, np.finfo(float).tiny)) - base_log
            )
            output.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "model": model,
                    "dimension_set": dimension_set,
                    "target_mode": target_mode,
                    "feature_map_type": (
                        "none"
                        if feature_map_settings is None
                        else feature_map_settings.feature_map_type
                    ),
                    "normalize_gaussian": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.normalize_gaussian
                    ),
                    "learnable_sigma": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.learnable_sigma
                    ),
                    "flatten_order": (
                        ""
                        if feature_map_settings is None
                        else feature_map_settings.flatten_order
                    ),
                    "split": split,
                    "sample_id": record.sample_id,
                    "structure_id": record.structure_id,
                    "variant": record.variant,
                    "source_filename": record.source_filename,
                    "retained_density_bvtv": record.bvtv,
                    "target_MPa": truth,
                    "prediction_MPa": prediction,
                    "error_MPa": prediction - truth,
                    "absolute_error_MPa": abs(prediction - truth),
                    "base_log_prediction": base_log,
                    "actual_log_residual": actual_log_residual,
                    "predicted_log_residual": predicted_log_residual,
                }
            )
    return output


def subset_cache(
    cached: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    branch_names: Sequence[str],
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    return {
        index: {name: values[name] for name in branch_names}
        for index, values in cached.items()
    }


def make_loaders(
    *,
    records: Sequence[core.SampleRecord],
    split_indices: Mapping[str, Sequence[int]],
    cached: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    dependencies: Any,
    batch_size: int,
    num_workers: int,
    device: Any,
) -> dict[str, Any]:
    DatasetClass = core.make_dataset_class(dependencies)
    datasets = {
        split: DatasetClass(records, split_indices[split], cached) for split in SPLITS
    }
    return {
        split: dependencies.DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=core.collate_compressed,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        for split in SPLITS
    }


def evaluate_model(
    model: Any,
    loader: Any,
    *,
    records: Sequence[core.SampleRecord],
    normalization: core.TargetNormalization,
    target_mode: str,
    all_base_log: Any | None,
    device: Any,
    torch: Any,
) -> list[tuple[int, float, float]]:
    model.eval()
    rows: list[tuple[int, float, float]] = []
    with torch.no_grad():
        for diagrams, multiplicities, masks, _, _, record_ids in loader:
            diagrams = core.move_mapping(diagrams, device)
            multiplicities = core.move_mapping(multiplicities, device)
            masks = core.move_mapping(masks, device)
            record_ids_device = record_ids.to(device, non_blocking=True)
            normalized = model(diagrams, multiplicities, masks)
            signal = normalized * normalization.std + normalization.mean
            if target_mode == "raw":
                predictions = signal
            else:
                if all_base_log is None:
                    raise core.ExperimentError("Residual model has no BV/TV baseline.")
                predictions = torch.exp(all_base_log[record_ids_device] + signal)
            for record_id, prediction in zip(
                record_ids.tolist(), predictions.detach().cpu().tolist()
            ):
                index = int(record_id)
                rows.append((index, records[index].target, float(prediction)))
    return rows


def train_one_perslay(
    *,
    records: Sequence[core.SampleRecord],
    cached_all: Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    assignments: Mapping[int, str],
    split_indices: Mapping[str, Sequence[int]],
    branch_names: Sequence[str],
    target_mode: str,
    base_log_predictions: np.ndarray | None,
    dependencies: Any,
    device: Any,
    arguments: argparse.Namespace,
    seed: int,
    repeat: int,
    fold: int,
    dimension_set: str,
    feature_map_settings: FeatureMapSettings,
) -> TrainedRun:
    torch = dependencies.torch
    core.seed_everything(seed, torch)

    cached = subset_cache(cached_all, branch_names)
    image_bounds = core.compute_training_bounds(
        cached,
        assignments,
        branch_names,
        padding_fraction=arguments.bounds_padding_fraction,
    )
    branch_mass_scales = core.compute_training_mass_scales(
        cached, assignments, branch_names
    )

    training_targets = np.asarray(
        [records[index].target for index in split_indices["train"]],
        dtype=np.float64,
    )
    if target_mode == "raw":
        training_signal = training_targets
    else:
        if base_log_predictions is None:
            raise core.ExperimentError("Residual target requested without baseline.")
        training_signal = np.log(training_targets) - base_log_predictions[
            np.asarray(split_indices["train"], dtype=int)
        ]

    signal_std = float(training_signal.std(ddof=0))
    if signal_std <= 0.0:
        raise core.ExperimentError("Training signal has zero variance.")
    normalization = core.TargetNormalization(
        mean=float(training_signal.mean()), std=signal_std
    )

    loaders = make_loaders(
        records=records,
        split_indices=split_indices,
        cached=cached,
        dependencies=dependencies,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
    )
    model, branch_settings = core.make_model(
        branch_names=branch_names,
        image_bounds=image_bounds,
        image_size=arguments.image_size,
        sigma_multiplier=arguments.sigma_multiplier,
        branch_mass_scales=branch_mass_scales,
        hidden_dimension=arguments.hidden_dimension,
        dropout=arguments.dropout,
        weight_type=arguments.weight_type,
        weight_hidden_dimension=arguments.weight_hidden_dimension,
        feature_map_type=feature_map_settings.feature_map_type,
        normalize_gaussian=feature_map_settings.normalize_gaussian,
        learnable_sigma=feature_map_settings.learnable_sigma,
        flatten_order=feature_map_settings.flatten_order,
        dependencies=dependencies,
    )
    model = model.to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    all_base_log = (
        None
        if base_log_predictions is None
        else torch.tensor(base_log_predictions, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=arguments.scheduler_factor,
        patience=arguments.scheduler_patience,
    )
    loss_function = torch.nn.MSELoss()

    history_rows: list[dict[str, Any]] = []
    best_validation_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for diagrams, multiplicities, masks, targets, _, record_ids in loaders["train"]:
            diagrams = core.move_mapping(diagrams, device)
            multiplicities = core.move_mapping(multiplicities, device)
            masks = core.move_mapping(masks, device)
            targets = targets.to(device, non_blocking=True)
            record_ids_device = record_ids.to(device, non_blocking=True)

            if target_mode == "raw":
                signals = targets
            else:
                if all_base_log is None:
                    raise core.ExperimentError("Residual model has no BV/TV baseline.")
                signals = torch.log(targets) - all_base_log[record_ids_device]
            normalized_targets = (signals - normalization.mean) / normalization.std

            optimizer.zero_grad(set_to_none=True)
            predictions = model(diagrams, multiplicities, masks)
            loss = loss_function(predictions, normalized_targets)
            if not torch.isfinite(loss):
                raise core.ExperimentError(
                    f"Non-finite loss in repeat={repeat}, fold={fold}, "
                    f"model={target_mode}, dimensions={dimension_set}, epoch={epoch}."
                )
            loss.backward()
            if arguments.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), arguments.gradient_clip
                )
            optimizer.step()

            count = int(targets.shape[0])
            total_loss += float(loss.item()) * count
            total_samples += count

        training_mse_normalized = total_loss / max(total_samples, 1)
        validation_rows = evaluate_model(
            model,
            loaders["validation"],
            records=records,
            normalization=normalization,
            target_mode=target_mode,
            all_base_log=all_base_log,
            device=device,
            torch=torch,
        )
        validation_metric = core.regression_metrics(validation_rows)
        scheduler.step(validation_metric.rmse)

        history_rows.append(
            {
                "repeat": repeat,
                "outer_fold": fold,
                "target_mode": target_mode,
                "dimension_set": dimension_set,
                "feature_map_type": feature_map_settings.feature_map_type,
                "normalize_gaussian": feature_map_settings.normalize_gaussian,
                "learnable_sigma": feature_map_settings.learnable_sigma,
                "flatten_order": feature_map_settings.flatten_order,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_mse_normalized": training_mse_normalized,
                "validation_mae_MPa": validation_metric.mae,
                "validation_rmse_MPa": validation_metric.rmse,
                "validation_r2": validation_metric.r2,
            }
        )

        improved = (
            validation_metric.rmse
            < best_validation_rmse - arguments.minimum_improvement
        )
        if improved:
            best_validation_rmse = validation_metric.rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if arguments.verbose_epochs and (
            epoch == 1 or epoch % 10 == 0 or improved
        ):
            print(
                f"    epoch {epoch:4d}: train normalized MSE="
                f"{training_mse_normalized:.6f}, validation RMSE="
                f"{validation_metric.rmse:.6f} MPa"
            )

        if epochs_without_improvement >= arguments.patience:
            break

    if best_state is None:
        raise core.ExperimentError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)

    final_rows: dict[str, list[tuple[int, float, float]]] = {}
    for split in SPLITS:
        final_rows[split] = evaluate_model(
            model,
            loaders[split],
            records=records,
            normalization=normalization,
            target_mode=target_mode,
            all_base_log=all_base_log,
            device=device,
            torch=torch,
        )

    model_name = f"perslay_{target_mode}"
    metrics_rows = calculate_metrics_rows(
        final_rows,
        records,
        repeat=repeat,
        fold=fold,
        model=model_name,
        dimension_set=dimension_set,
        target_mode=target_mode,
        feature_map_settings=feature_map_settings,
        best_epoch=best_epoch,
    )
    prediction_rows = calculate_prediction_rows(
        final_rows,
        records,
        repeat=repeat,
        fold=fold,
        model=model_name,
        dimension_set=dimension_set,
        target_mode=target_mode,
        feature_map_settings=feature_map_settings,
        base_log_predictions=base_log_predictions,
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "repeat": repeat,
        "outer_fold": fold,
        "model": model_name,
        "target_mode": target_mode,
        "dimension_set": dimension_set,
        "feature_map": asdict(feature_map_settings),
        "branch_names": list(branch_names),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_rmse_MPa": best_validation_rmse,
        "number_of_parameters": parameter_count,
        "target_normalization_training_only": asdict(normalization),
        "image_bounds_training_only": image_bounds,
        "branch_mass_scales_training_only": branch_mass_scales,
        "branch_settings": branch_settings,
        "learned_parameters": core.learned_parameters(model, torch),
        "metrics": {
            split: metric_to_dict(core.regression_metrics(final_rows[split]))
            for split in SPLITS
        },
    }
    if arguments.save_fold_models:
        result["checkpoint"] = {
            "model_state_dict": best_state,
            "normalization": asdict(normalization),
            "image_bounds": image_bounds,
            "branch_mass_scales": branch_mass_scales,
            "branch_settings": branch_settings,
        }

    # Return the state separately through the result only when requested; the
    # caller writes it with torch.save because JSON cannot encode tensors.
    result["_model_state_dict"] = best_state if arguments.save_fold_models else None

    del model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return TrainedRun(
        result=result,
        metrics_rows=metrics_rows,
        prediction_rows=prediction_rows,
        history_rows=history_rows,
    )


def baseline_results_for_fold(
    *,
    records: Sequence[core.SampleRecord],
    split_indices: Mapping[str, Sequence[int]],
    repeat: int,
    fold: int,
    variants: Sequence[str],
) -> tuple[
    dict[str, LinearLogModel],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    training_mean = float(
        np.mean([records[index].target for index in split_indices["train"]])
    )
    mean_predictions = np.full(len(records), training_mean, dtype=np.float64)

    power_model = fit_log_bvtv_model(
        records,
        split_indices["train"],
        variants=variants,
        include_variant_effects=False,
    )
    variant_model = fit_log_bvtv_model(
        records,
        split_indices["train"],
        variants=variants,
        include_variant_effects=True,
    )
    power_log = predict_log_bvtv_model(power_model, records)
    variant_log = predict_log_bvtv_model(variant_model, records)

    prediction_arrays = {
        "mean": mean_predictions,
        "bvtv_power": np.exp(power_log),
        "variant_bvtv": np.exp(variant_log),
    }
    models = {"bvtv_power": power_model, "variant_bvtv": variant_model}
    base_logs = {"bvtv_power": power_log, "variant_bvtv": variant_log}

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name, predictions in prediction_arrays.items():
        rows_by_split = {
            split: predictions_from_array(
                records, split_indices[split], predictions
            )
            for split in SPLITS
        }
        metrics_rows.extend(
            calculate_metrics_rows(
                rows_by_split,
                records,
                repeat=repeat,
                fold=fold,
                model=model_name,
                dimension_set="none",
                target_mode="baseline",
                feature_map_settings=None,
                best_epoch=None,
            )
        )
        prediction_rows.extend(
            calculate_prediction_rows(
                rows_by_split,
                records,
                repeat=repeat,
                fold=fold,
                model=model_name,
                dimension_set="none",
                target_mode="baseline",
                feature_map_settings=None,
                base_log_predictions=(
                    None
                    if model_name == "mean"
                    else base_logs[model_name]
                ),
            )
        )
    return models, base_logs, metrics_rows, prediction_rows


def numeric(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=np.float64)


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["split"] != "test":
            continue
        key = (
            str(row["model"]),
            str(row["dimension_set"]),
            str(row["target_mode"]),
            str(row["feature_map_type"]),
            str(row["variant_scope"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model, dimension_set, target_mode, feature_map_type, variant_scope = key
        result: dict[str, Any] = {
            "model": model,
            "dimension_set": dimension_set,
            "target_mode": target_mode,
            "feature_map_type": feature_map_type,
            "variant_scope": variant_scope,
            "number_of_outer_evaluations": len(group),
        }
        for field in ("mae_MPa", "rmse_MPa", "r2"):
            values = numeric([row[field] for row in group])
            result[f"{field}_mean"] = float(np.mean(values))
            result[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[f"{field}_median"] = float(np.median(values))
            result[f"{field}_q025"] = float(np.quantile(values, 0.025))
            result[f"{field}_q975"] = float(np.quantile(values, 0.975))
        output.append(result)
    return output


def improvement_rows(
    metrics_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[tuple[int, int, str, str, str, str], Mapping[str, Any]] = {}
    for row in metrics_rows:
        if row["split"] != "test":
            continue
        key = (
            int(row["repeat"]),
            int(row["outer_fold"]),
            str(row["model"]),
            str(row["dimension_set"]),
            str(row["feature_map_type"]),
            str(row["variant_scope"]),
        )
        lookup[key] = row

    output: list[dict[str, Any]] = []
    for key, row in sorted(lookup.items()):
        repeat, fold, model, dimension_set, feature_map_type, variant_scope = key
        if not model.startswith("perslay_"):
            continue
        for baseline in ("bvtv_power", "variant_bvtv"):
            baseline_key = (repeat, fold, baseline, "none", "none", variant_scope)
            baseline_row = lookup.get(baseline_key)
            if baseline_row is None:
                continue
            model_rmse = float(row["rmse_MPa"])
            baseline_rmse = float(baseline_row["rmse_MPa"])
            model_mae = float(row["mae_MPa"])
            baseline_mae = float(baseline_row["mae_MPa"])
            output.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "model": model,
                    "dimension_set": dimension_set,
                    "feature_map_type": feature_map_type,
                    "variant_scope": variant_scope,
                    "baseline": baseline,
                    "model_rmse_MPa": model_rmse,
                    "baseline_rmse_MPa": baseline_rmse,
                    "rmse_difference_MPa": model_rmse - baseline_rmse,
                    "rmse_reduction_fraction": (
                        1.0 - model_rmse / baseline_rmse
                        if baseline_rmse > 0.0
                        else float("nan")
                    ),
                    "model_mae_MPa": model_mae,
                    "baseline_mae_MPa": baseline_mae,
                    "mae_difference_MPa": model_mae - baseline_mae,
                    "model_wins_rmse": int(model_rmse < baseline_rmse),
                    "model_wins_mae": int(model_mae < baseline_mae),
                }
            )
    return output


def aggregate_improvements(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["dimension_set"]),
            str(row["feature_map_type"]),
            str(row["variant_scope"]),
            str(row["baseline"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model, dimension_set, feature_map_type, variant_scope, baseline = key
        rmse_differences = numeric([row["rmse_difference_MPa"] for row in group])
        rmse_reductions = numeric([row["rmse_reduction_fraction"] for row in group])
        mae_differences = numeric([row["mae_difference_MPa"] for row in group])
        output.append(
            {
                "model": model,
                "dimension_set": dimension_set,
                "feature_map_type": feature_map_type,
                "variant_scope": variant_scope,
                "baseline": baseline,
                "number_of_outer_evaluations": len(group),
                "mean_rmse_difference_MPa": float(np.mean(rmse_differences)),
                "median_rmse_difference_MPa": float(np.median(rmse_differences)),
                "rmse_difference_q025_MPa": float(np.quantile(rmse_differences, 0.025)),
                "rmse_difference_q975_MPa": float(np.quantile(rmse_differences, 0.975)),
                "mean_rmse_reduction_fraction": float(np.mean(rmse_reductions)),
                "median_rmse_reduction_fraction": float(np.median(rmse_reductions)),
                "mean_mae_difference_MPa": float(np.mean(mae_differences)),
                "rmse_win_count": int(sum(int(row["model_wins_rmse"]) for row in group)),
                "rmse_win_fraction": float(
                    np.mean([int(row["model_wins_rmse"]) for row in group])
                ),
                "mae_win_count": int(sum(int(row["model_wins_mae"]) for row in group)),
                "mae_win_fraction": float(
                    np.mean([int(row["model_wins_mae"]) for row in group])
                ),
            }
        )
    return output


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Repeated leakage-safe grouped cross-validation for raw and "
            "BV/TV-residual PersLay modulus prediction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument("--data-root", type=Path, default=Path("data"))
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bone_modulus_grouped_cv"),
    )
    argument_parser.add_argument(
        "--variants", nargs="+", default=["baseline", "uniform", "weakened"]
    )
    argument_parser.add_argument("--folds", type=int, default=5)
    argument_parser.add_argument("--repeats", type=int, default=10)
    argument_parser.add_argument(
        "--inner-validation-fraction", type=float, default=0.15
    )
    argument_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_MODES,
        default=["raw", "variant_residual"],
    )
    argument_parser.add_argument(
        "--dimension-sets",
        nargs="+",
        choices=tuple(DIMENSION_SETS),
        default=["h0", "h2", "h0_h2"],
    )
    argument_parser.add_argument(
        "--target-column", default="apparent_youngs_modulus_MPa"
    )
    argument_parser.add_argument(
        "--bvtv-column", default="retained_density_bvtv"
    )
    argument_parser.add_argument("--minimum-persistence", type=float, default=0.0)
    argument_parser.add_argument("--image-size", nargs=2, type=int, default=(10, 10))
    argument_parser.add_argument(
        "--bounds-padding-fraction", type=float, default=0.05
    )
    argument_parser.add_argument("--sigma-multiplier", type=float, default=1.0)
    argument_parser.add_argument(
        "--feature-map-types",
        nargs="+",
        choices=FEATURE_MAP_TYPES,
        default=["legacy_gaussian"],
        help=(
            "Gaussian PersLay discretizations to evaluate on identical grouped "
            "folds. Use grid_sampled integrated for the main comparison."
        ),
    )
    argument_parser.add_argument(
        "--gaussian-normalization",
        choices=GAUSSIAN_NORMALIZATION_MODES,
        default="representation-default",
        help=(
            "Kernel normalization applied to every selected feature map. The "
            "representation default is unnormalized for legacy/grid sampling "
            "and normalized for integrated pixels."
        ),
    )
    argument_parser.add_argument(
        "--learnable-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn anisotropic birth and persistence bandwidths.",
    )
    argument_parser.add_argument(
        "--flatten-order",
        choices=FLATTEN_ORDER_MODES,
        default="representation-default",
        help=(
            "Pixel-vector ordering. Representation defaults are legacy x-major "
            "for sampled maps and row-major for integrated pixels."
        ),
    )
    argument_parser.add_argument("--hidden-dimension", type=int, default=128)
    argument_parser.add_argument("--dropout", type=float, default=0.10)
    argument_parser.add_argument(
        "--weight-type",
        choices=(
            "constant",
            "learnable_power",
            "normalized_learnable_power",
            "mlp",
        ),
        default="normalized_learnable_power",
    )
    argument_parser.add_argument("--weight-hidden-dimension", type=int, default=16)
    argument_parser.add_argument("--batch-size", type=int, default=4)
    argument_parser.add_argument("--epochs", type=int, default=300)
    argument_parser.add_argument("--patience", type=int, default=40)
    argument_parser.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    argument_parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    argument_parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    argument_parser.add_argument("--gradient-clip", type=float, default=5.0)
    argument_parser.add_argument("--scheduler-patience", type=int, default=10)
    argument_parser.add_argument("--scheduler-factor", type=float, default=0.5)
    argument_parser.add_argument("--random-seed", type=int, default=2026)
    argument_parser.add_argument("--num-workers", type=int, default=0)
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--cache-dir", type=Path, default=None)
    argument_parser.add_argument("--rebuild-cache", action="store_true")
    argument_parser.add_argument("--save-fold-models", action="store_true")
    argument_parser.add_argument(
        "--retrain-existing",
        action="store_true",
        help="Retrain runs even when a completed per-run result already exists.",
    )
    argument_parser.add_argument("--verbose-epochs", action="store_true")
    argument_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete the entire output directory before starting.",
    )
    return argument_parser


def run(arguments: argparse.Namespace) -> None:
    if arguments.minimum_persistence < 0.0:
        raise core.ExperimentError("--minimum-persistence cannot be negative.")
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.patience < 1:
        raise core.ExperimentError(
            "--epochs, --batch-size, and --patience must be positive."
        )
    if min(arguments.image_size) < 2:
        raise core.ExperimentError("Each image-size coordinate must be at least 2.")
    if not math.isfinite(arguments.sigma_multiplier) or arguments.sigma_multiplier <= 0:
        raise core.ExperimentError("--sigma-multiplier must be finite and positive.")
    if (
        not math.isfinite(arguments.bounds_padding_fraction)
        or arguments.bounds_padding_fraction < 0
    ):
        raise core.ExperimentError(
            "--bounds-padding-fraction must be finite and nonnegative."
        )
    if not 0.0 <= arguments.dropout < 1.0:
        raise core.ExperimentError("--dropout must be in [0, 1).")
    if arguments.hidden_dimension < 1 or arguments.weight_hidden_dimension < 1:
        raise core.ExperimentError("Hidden dimensions must be positive.")
    if arguments.learning_rate <= 0.0 or arguments.weight_decay < 0.0:
        raise core.ExperimentError(
            "--learning-rate must be positive and --weight-decay nonnegative."
        )
    if not 0.0 < arguments.scheduler_factor < 1.0:
        raise core.ExperimentError("--scheduler-factor must be in (0, 1).")
    if arguments.scheduler_patience < 0 or arguments.num_workers < 0:
        raise core.ExperimentError(
            "--scheduler-patience and --num-workers must be nonnegative."
        )

    selected_feature_map_types = tuple(dict.fromkeys(arguments.feature_map_types))
    feature_map_settings = tuple(
        resolve_feature_map_settings(
            feature_map_type,
            gaussian_normalization=arguments.gaussian_normalization,
            learnable_sigma=arguments.learnable_sigma,
            flatten_order=arguments.flatten_order,
        )
        for feature_map_type in selected_feature_map_types
    )

    data_root = arguments.data_root.resolve()
    output_directory = arguments.output_dir.resolve()
    if arguments.overwrite_output and output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    selected_dimension_sets = tuple(dict.fromkeys(arguments.dimension_sets))
    selected_models = tuple(dict.fromkeys(arguments.models))
    variants = tuple(dict.fromkeys(arguments.variants))
    all_dimensions = tuple(
        sorted(
            {
                dimension
                for name in selected_dimension_sets
                for dimension in DIMENSION_SETS[name]
            }
        )
    )
    if not all_dimensions:
        raise core.ExperimentError("At least one persistence dimension is required.")

    records = core.read_records(
        data_root,
        variants=variants,
        dimensions=all_dimensions,
        target_column=arguments.target_column,
        bvtv_column=arguments.bvtv_column,
    )
    data_signature = stable_signature(
        {
            "variants": list(variants),
            "target_column": arguments.target_column,
            "bvtv_column": arguments.bvtv_column,
            "records": [
                {
                    "sample_id": record.sample_id,
                    "structure_id": record.structure_id,
                    "variant": record.variant,
                    "target": float(record.target),
                    "bvtv": float(record.bvtv),
                    "diagrams": {
                        name: core.file_signature(path)
                        for name, path in sorted(record.diagram_paths.items())
                    },
                }
                for record in records
            ],
        }
    )
    structure_ids = sorted({record.structure_id for record in records})
    folds = make_outer_folds(
        structure_ids,
        folds=arguments.folds,
        repeats=arguments.repeats,
        inner_validation_fraction=arguments.inner_validation_fraction,
        random_seed=arguments.random_seed,
    )

    cache_directory = (
        arguments.cache_dir.resolve()
        if arguments.cache_dir is not None
        else data_root / ".perslay_diagram_cache"
    )
    cached, compression_summary = core.build_or_load_cache(
        records,
        cache_directory=cache_directory,
        minimum_persistence=arguments.minimum_persistence,
        rebuild=arguments.rebuild_cache,
    )

    dependencies = core.import_dependencies()
    torch = dependencies.torch
    device = core.choose_device(arguments.device, torch)

    run_configuration = {
        "script_version": SCRIPT_VERSION,
        "data_root": str(data_root),
        "variants": list(variants),
        "number_of_structures": len(structure_ids),
        "number_of_samples": len(records),
        "folds": arguments.folds,
        "repeats": arguments.repeats,
        "inner_validation_fraction": arguments.inner_validation_fraction,
        "models": list(selected_models),
        "dimension_sets": list(selected_dimension_sets),
        "minimum_persistence": arguments.minimum_persistence,
        "image_size": list(arguments.image_size),
        "bounds_padding_fraction": arguments.bounds_padding_fraction,
        "sigma_multiplier": arguments.sigma_multiplier,
        "feature_maps": [asdict(settings) for settings in feature_map_settings],
        "gaussian_normalization_mode": arguments.gaussian_normalization,
        "learnable_sigma": arguments.learnable_sigma,
        "flatten_order_mode": arguments.flatten_order,
        "hidden_dimension": arguments.hidden_dimension,
        "dropout": arguments.dropout,
        "weight_type": arguments.weight_type,
        "weight_hidden_dimension": arguments.weight_hidden_dimension,
        "batch_size": arguments.batch_size,
        "epochs": arguments.epochs,
        "patience": arguments.patience,
        "minimum_improvement": arguments.minimum_improvement,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "gradient_clip": arguments.gradient_clip,
        "scheduler_patience": arguments.scheduler_patience,
        "scheduler_factor": arguments.scheduler_factor,
        "random_seed": arguments.random_seed,
        "device": str(device),
        "compression_summary": compression_summary,
        "scientific_design": {
            "grouping_variable": "structure_id",
            "all_sibling_variants_kept_together": True,
            "feature_maps_use_identical_grouped_folds": True,
            "feature_maps_use_identical_initialization_seeds": True,
            "outer_test_groups_never_used_for_training_or_early_stopping": True,
            "inner_validation_controls_early_stopping": True,
            "all_bvtv_models_fitted_on_inner_training_only": True,
            "diagram_bounds_fitted_on_inner_training_only": True,
            "diagram_mass_scaling_fitted_on_inner_training_only": True,
            "target_normalization_fitted_on_inner_training_only": True,
            "duplicate_compression_exact_under_sum_aggregation": True,
        },
    }
    save_json(output_directory / "config.json", run_configuration)

    print(f"Using device: {device}")
    print(
        f"Loaded {len(records)} samples from {len(structure_ids)} independent "
        f"structure groups."
    )
    number_of_trainings = (
        arguments.repeats
        * arguments.folds
        * len(selected_models)
        * len(selected_dimension_sets)
        * len(feature_map_settings)
    )
    print(
        f"Experiment grid: {arguments.repeats} repeats x {arguments.folds} folds x "
        f"{len(selected_models)} model modes x {len(selected_dimension_sets)} "
        f"dimension sets x {len(feature_map_settings)} feature maps = "
        f"{number_of_trainings} PersLay trainings."
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for definition in folds:
        assignments, split_indices = fold_assignments(records, definition)
        fold_directory = (
            output_directory
            / "folds"
            / f"repeat_{definition.repeat:02d}"
            / f"fold_{definition.fold:02d}"
        )
        fold_directory.mkdir(parents=True, exist_ok=True)

        print(
            f"Repeat {definition.repeat + 1}/{arguments.repeats}, "
            f"fold {definition.fold + 1}/{arguments.folds}: "
            f"train={len(definition.training_groups)} structures, "
            f"validation={len(definition.validation_groups)}, "
            f"test={len(definition.test_groups)}"
        )

        for record in records:
            assignment_rows.append(
                {
                    "repeat": definition.repeat,
                    "outer_fold": definition.fold,
                    "sample_id": record.sample_id,
                    "structure_id": record.structure_id,
                    "variant": record.variant,
                    "split": assignments[record.index],
                }
            )

        baseline_models, base_logs, baseline_metric_rows, baseline_prediction_rows = (
            baseline_results_for_fold(
                records=records,
                split_indices=split_indices,
                repeat=definition.repeat,
                fold=definition.fold,
                variants=variants,
            )
        )
        all_metric_rows.extend(baseline_metric_rows)
        all_prediction_rows.extend(baseline_prediction_rows)
        save_json(
            fold_directory / "baseline_models.json",
            {
                name: asdict(model) for name, model in baseline_models.items()
            },
        )

        for model_index, target_mode in enumerate(selected_models):
            if target_mode == "raw":
                base_log_predictions = None
            elif target_mode == "power_residual":
                base_log_predictions = base_logs["bvtv_power"]
            elif target_mode == "variant_residual":
                base_log_predictions = base_logs["variant_bvtv"]
            else:
                raise core.ExperimentError(f"Unknown target mode: {target_mode}")

            for dimension_index, dimension_set in enumerate(selected_dimension_sets):
                dimensions = DIMENSION_SETS[dimension_set]
                branch_names = tuple(core.dimension_name(value) for value in dimensions)
                seed = (
                    arguments.random_seed
                    + 1_000_000 * definition.repeat
                    + 10_000 * definition.fold
                    + 100 * model_index
                    + dimension_index
                )
                for settings in feature_map_settings:
                    signature_payload = {
                        "data_signature": data_signature,
                        "variants": list(variants),
                        "target_column": arguments.target_column,
                        "bvtv_column": arguments.bvtv_column,
                        "target_mode": target_mode,
                        "dimension_set": dimension_set,
                        "feature_map": asdict(settings),
                        "seed": seed,
                        "hyperparameters": {
                            key: run_configuration[key]
                            for key in (
                                "minimum_persistence",
                                "image_size",
                                "bounds_padding_fraction",
                                "sigma_multiplier",
                                "hidden_dimension",
                                "dropout",
                                "weight_type",
                                "weight_hidden_dimension",
                                "batch_size",
                                "epochs",
                                "patience",
                                "minimum_improvement",
                                "learning_rate",
                                "weight_decay",
                                "gradient_clip",
                                "scheduler_patience",
                                "scheduler_factor",
                            )
                        },
                    }
                    signature = stable_signature(signature_payload)
                    run_directory = fold_directory / (
                        f"perslay_{target_mode}__{dimension_set}__"
                        f"{settings.feature_map_type}__{signature}"
                    )
                    result_path = run_directory / "result.json"
                    metrics_path = run_directory / "metrics.csv"
                    predictions_path = run_directory / "predictions.csv"
                    history_path = run_directory / "training_history.csv"

                    if result_path.is_file() and not arguments.retrain_existing:
                        print(
                            f"  Loading completed {target_mode}/{dimension_set}/"
                            f"{settings.feature_map_type} ({signature})."
                        )
                        all_metric_rows.extend(read_csv_rows(metrics_path))
                        all_prediction_rows.extend(read_csv_rows(predictions_path))
                        all_history_rows.extend(read_csv_rows(history_path))
                        continue

                    print(
                        f"  Training {target_mode}/{dimension_set}/"
                        f"{settings.feature_map_type} ..."
                    )
                    run_directory.mkdir(parents=True, exist_ok=True)
                    trained = train_one_perslay(
                        records=records,
                        cached_all=cached,
                        assignments=assignments,
                        split_indices=split_indices,
                        branch_names=branch_names,
                        target_mode=target_mode,
                        base_log_predictions=base_log_predictions,
                        dependencies=dependencies,
                        device=device,
                        arguments=arguments,
                        seed=seed,
                        repeat=definition.repeat,
                        fold=definition.fold,
                        dimension_set=dimension_set,
                        feature_map_settings=settings,
                    )

                    model_state = trained.result.pop("_model_state_dict", None)
                    if arguments.save_fold_models and model_state is not None:
                        torch.save(
                            {
                                "model_state_dict": model_state,
                                "result_metadata": trained.result,
                            },
                            run_directory / "best_model.pt",
                        )
                        trained.result.pop("checkpoint", None)

                    representation_fields = (
                        "feature_map_type",
                        "normalize_gaussian",
                        "learnable_sigma",
                        "flatten_order",
                    )
                    write_csv(
                        metrics_path,
                        (
                            "repeat",
                            "outer_fold",
                            "model",
                            "dimension_set",
                            "target_mode",
                            *representation_fields,
                            "split",
                            "variant_scope",
                            "number_of_samples",
                            "mae_MPa",
                            "rmse_MPa",
                            "r2",
                            "best_epoch",
                        ),
                        trained.metrics_rows,
                    )
                    write_csv(
                        predictions_path,
                        (
                            "repeat",
                            "outer_fold",
                            "model",
                            "dimension_set",
                            "target_mode",
                            *representation_fields,
                            "split",
                            "sample_id",
                            "structure_id",
                            "variant",
                            "source_filename",
                            "retained_density_bvtv",
                            "target_MPa",
                            "prediction_MPa",
                            "error_MPa",
                            "absolute_error_MPa",
                            "base_log_prediction",
                            "actual_log_residual",
                            "predicted_log_residual",
                        ),
                        trained.prediction_rows,
                    )
                    write_csv(
                        history_path,
                        (
                            "repeat",
                            "outer_fold",
                            "target_mode",
                            "dimension_set",
                            *representation_fields,
                            "epoch",
                            "learning_rate",
                            "training_mse_normalized",
                            "validation_mae_MPa",
                            "validation_rmse_MPa",
                            "validation_r2",
                        ),
                        trained.history_rows,
                    )
                    save_json(result_path, trained.result)

                    all_metric_rows.extend(trained.metrics_rows)
                    all_prediction_rows.extend(trained.prediction_rows)
                    all_history_rows.extend(trained.history_rows)

                    test_overall = next(
                        row
                        for row in trained.metrics_rows
                        if row["split"] == "test" and row["variant_scope"] == "all"
                    )
                    print(
                        f"    best epoch={trained.result['best_epoch']}; test RMSE="
                        f"{float(test_overall['rmse_MPa']):.4f} MPa, R2="
                        f"{float(test_overall['r2']):.4f}"
                    )

        # Write cumulative files after every outer fold, so an interrupted run
        # still leaves usable partial results.
        write_csv(
            output_directory / "fold_metrics.csv",
            (
                "repeat",
                "outer_fold",
                "model",
                "dimension_set",
                "target_mode",
                "feature_map_type",
                "normalize_gaussian",
                "learnable_sigma",
                "flatten_order",
                "split",
                "variant_scope",
                "number_of_samples",
                "mae_MPa",
                "rmse_MPa",
                "r2",
                "best_epoch",
            ),
            all_metric_rows,
        )
        write_csv(
            output_directory / "fold_assignments.csv",
            ("repeat", "outer_fold", "sample_id", "structure_id", "variant", "split"),
            assignment_rows,
        )

    metric_summary = aggregate_metric_rows(all_metric_rows)
    improvements = improvement_rows(all_metric_rows)
    improvement_summary = aggregate_improvements(improvements)

    write_csv(
        output_directory / "predictions.csv",
        (
            "repeat",
            "outer_fold",
            "model",
            "dimension_set",
            "target_mode",
            "feature_map_type",
            "normalize_gaussian",
            "learnable_sigma",
            "flatten_order",
            "split",
            "sample_id",
            "structure_id",
            "variant",
            "source_filename",
            "retained_density_bvtv",
            "target_MPa",
            "prediction_MPa",
            "error_MPa",
            "absolute_error_MPa",
            "base_log_prediction",
            "actual_log_residual",
            "predicted_log_residual",
        ),
        all_prediction_rows,
    )
    write_csv(
        output_directory / "training_history.csv",
        (
            "repeat",
            "outer_fold",
            "target_mode",
            "dimension_set",
            "feature_map_type",
            "normalize_gaussian",
            "learnable_sigma",
            "flatten_order",
            "epoch",
            "learning_rate",
            "training_mse_normalized",
            "validation_mae_MPa",
            "validation_rmse_MPa",
            "validation_r2",
        ),
        all_history_rows,
    )
    write_csv(
        output_directory / "metric_summary.csv",
        tuple(metric_summary[0].keys()) if metric_summary else (),
        metric_summary,
    )
    write_csv(
        output_directory / "paired_improvements.csv",
        tuple(improvements[0].keys()) if improvements else (),
        improvements,
    )
    write_csv(
        output_directory / "improvement_summary.csv",
        tuple(improvement_summary[0].keys()) if improvement_summary else (),
        improvement_summary,
    )

    print("\nTest-set summary across outer folds:")
    overall_rows = [row for row in metric_summary if row["variant_scope"] == "all"]
    variant_scopes = sorted(
        {
            str(row["variant_scope"])
            for row in metric_summary
            if str(row["variant_scope"]) != "all"
        }
    )
    for row in overall_rows:
        print(
            f"  {row['model']:26s} {row['dimension_set']:8s} "
            f"{row['feature_map_type']:16s} "
            f"RMSE={float(row['rmse_MPa_mean']):.4f} +/- "
            f"{float(row['rmse_MPa_std']):.4f} MPa; "
            f"R2={float(row['r2_mean']):.4f}"
        )

    for scope in variant_scopes:
        print(f"\nTest-set summary for variant={scope}:")
        rows = [row for row in metric_summary if row["variant_scope"] == scope]
        for row in rows:
            print(
                f"  {row['model']:26s} {row['dimension_set']:8s} "
                f"{row['feature_map_type']:16s} "
                f"RMSE={float(row['rmse_MPa_mean']):.4f} +/- "
                f"{float(row['rmse_MPa_std']):.4f} MPa; "
                f"R2={float(row['r2_mean']):.4f}"
            )

    print("\nPaired improvement against variant-aware BV/TV:")
    selected_improvements = [
        row
        for row in improvement_summary
        if row["variant_scope"] == "all" and row["baseline"] == "variant_bvtv"
    ]
    for row in selected_improvements:
        print(
            f"  {row['model']:26s} {row['dimension_set']:8s} "
            f"{row['feature_map_type']:16s} "
            f"mean Delta RMSE={float(row['mean_rmse_difference_MPa']):+.4f} MPa; "
            f"wins={float(row['rmse_win_fraction']):.1%}"
        )

    print(f"\nResults written to: {output_directory}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        run(arguments)
    except core.ExperimentError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
